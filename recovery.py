import csv

from losses import completion_network_loss, noise_loss
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion, dist_inversion
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--device', type=str, default='4,5,6,7', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--improved_flag', action='store_true', default=False, help='use improved k+1 GAN')
    parser.add_argument('--dist_flag', action='store_true', default=False, help='use distributional recovery')
    
    #Phase1,2,3 related args
    # iter_list: for phase 1, comma separated list of iteration budgets to evaluate; for phase 2 and 3, only the first value will be used as fixed iteration budget
    # num_ids: number of identities to attack 
    # num_seeds: number of random seeds to run for each attack 
    # phase: which phase to run (phase1/phase2/phase3)
    parser.add_argument('--iter_list', type=str, default='300,600,1200', help='Comma separated iteration budgets')
    parser.add_argument('--num_ids', type=int, default=10, help='Number of identities')
    parser.add_argument('--num_seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--phase', type=str, default='phase1', help='phase1 for iteration experiment, phase2 for attack comparison, phase3 for defense evaluation')
    
    #Phase 3 related args added
    # defense: "none" | "noise" | "smooth" - to specify the type of defense to apply during attack evaluation
    # noise_sigma: the standard deviation of Gaussian noise to add if defense is "noise"
    # smooth_alpha: the smoothing factor for label smoothing if defense is "smooth"
    # model_trained_against: specify which target model the GAN was trained against (VGG16, IR152, FaceNet64)
    parser.add_argument('--defense', type=str, default='none',
                    help='none | noise | smooth')
    parser.add_argument('--noise_sigma', type=float, default=0.01)
    parser.add_argument('--smooth_alpha', type=float, default=0.1)
    parser.add_argument('--model_trained_against', type=str, default=None,
                        help='Model GAN was trained against')
    args = parser.parse_args()

    # parse iter_list into a list of integers
    iter_list = [int(x) for x in args.iter_list.split(',')]

    logger = get_logger()

    # changes to save results from phases
    results_dir = './results_' + args.phase
    os.makedirs(results_dir, exist_ok=True)

    logger.info(args)
    logger.info("=> creating model ...")   
    
    z_dim = 100
    ###########################################
    ###########     load model       ##########
    ###########################################
    G = Generator(z_dim)
    G = torch.nn.DataParallel(G).cuda()

    if args.improved_flag == True:
        D = MinibatchDiscriminator()

        # Phase3 - loading different GAN checkpoints based on which target model the GAN was trained against 
        if args.model_trained_against == "IR152":
            path_G = './improvedGAN/improved_celeba_G_IR152.tar'
            path_D = './improvedGAN/improved_celeba_D_IR152.tar'

        elif args.model_trained_against == "FaceNet64":
            path_G = './improvedGAN/improved_celeba_G_facenet.tar'
            path_D = './improvedGAN/improved_celeba_D_facenet.tar'

        else:
            # default = VGG16
            path_G = './improvedGAN/improved_celeba_G.tar'
            path_D = './improvedGAN/improved_celeba_D.tar'

    else:
        D = DGWGAN(3)
        path_G = './improvedGAN/celeba_G.tar'
        path_D = './improvedGAN/celeba_D.tar'
    
    D = torch.nn.DataParallel(D).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=False)

    if args.model.startswith("VGG16"):
        T = VGG16(1000)
        path_T = './target_model/target_ckp/VGG16_88.26.tar'
    elif args.model.startswith('IR152'):
        T = IR152(1000)
        path_T = './target_model/target_ckp/IR152_91.16.tar'
    elif args.model == "FaceNet64":
        T = FaceNet64(1000)
        path_T = './target_model/target_ckp/FaceNet64_88.50.tar'

    
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).cuda()
    path_E = './target_model/target_ckp/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)


    ############         attack     ###########
    
    # column headers for saving results csv based on phase
    columnn_headers_phase1 = [
        'Model',
        'Iterations',
        'Top1_Acc',
        'Top5_Acc',
        'Acc_Var',
        'Acc5_Var',
        'Runtime_sec'
    ]
    
    columnn_headers_phase2 = [
        'Model',
        'Attack_Type',
        'Iterations',
        'Top1_Acc',
        'Top5_Acc',
        'Acc_Var',
        'Acc5_Var',
        'Runtime_sec',
        'Delta_from_Baseline',
        'Improvement_Percentage'
    ]

    column_headers_phase3 = [
        'Model_Trained_Against',
        'Test_Model',
        'Attack_Type',
        'Defense',
        'Noise_Sigma',
        'Smooth_Alpha',
        'Iterations',
        'Top1_Acc',
        'Top5_Acc',
        'Acc_Var',
        'Acc5_Var',
        'Runtime_sec'
    ]

    csv_filename = f"{args.model}_{args.phase}.csv"
    if args.phase == "phase3":
        csv_filename = "results.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    logger.info(f"=> Running {args.phase}")

    if os.path.exists(csv_path):
        csv_mode = 'a'
    else:
        csv_mode = 'w'
    
    write_headers = not os.path.exists(csv_path)

    with open(csv_path, mode=csv_mode, newline='') as file:
        writer = csv.writer(file)

        if args.phase == "phase1":
            if write_headers:
                writer.writerow(columnn_headers_phase1)

            #Phase 1: Iterations vs execution time and accuracy
            for iter_budget in iter_list:

                logger.info(f"==> Running with iter_times={iter_budget}")
                iden = torch.from_numpy(np.arange(args.num_ids))

                if args.dist_flag:
                    acc, acc5, var, var5, runtime = dist_inversion(
                        G, D, T, E,
                        iden,
                        itr=0,
                        iter_times=iter_budget,
                        improved=args.improved_flag,
                        num_seeds=args.num_seeds
                    )
                else:
                    acc, acc5, var, var5, runtime = inversion(
                        G, D, T, E,
                        iden,
                        itr=0,
                        iter_times=iter_budget,
                        improved=args.improved_flag,
                        num_seeds=args.num_seeds
                    )

                print(f"Results for {args.model} with iter_times={iter_budget}: Acc={acc:.4f}, Acc5={acc5:.4f}, Var={var:.6f}, Var5={var5:.6f}, Runtime={runtime:.2f} sec")

                writer.writerow([
                    args.model,
                    iter_budget,
                    acc,
                    acc5,
                    var,
                    var5,
                    runtime
                ])
                file.flush()

        elif args.phase == "phase2":
            if write_headers:
                writer.writerow(columnn_headers_phase2)
            
            #Phase 2: Attack Comparison
            if args.improved_flag and not args.dist_flag:
                attack_name = "Improved_GAN"
                improved_flag = True
                dist_flag = False
            elif args.dist_flag:
                attack_name = "Distributional_Recovery"
                improved_flag = True
                dist_flag = True
            else:
                attack_name = "Baseline"
                improved_flag = False
                dist_flag = False

            baseline_acc = 0.0 
            
            #getting baseline acc from csv if exists
            if attack_name != "Baseline" and os.path.exists(csv_path):
                with open(csv_path, mode='r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['Attack_Type'] == 'Baseline':
                            baseline_acc = float(row['Top1_Acc'])
                            break

            logger.info(f"==> Running {attack_name} attack")
            iden = torch.from_numpy(np.arange(args.num_ids))
            iter_budget = iter_list[0] #only 1 fixed number of iterations for phase 2 

            if dist_flag:
                acc, acc5, var, var5, runtime = dist_inversion(
                    G, D, T, E,
                    iden,
                    itr=0,
                    iter_times=iter_budget,
                    improved=improved_flag,
                    num_seeds=args.num_seeds
                )
            else:
                acc, acc5, var, var5, runtime = inversion(
                    G, D, T, E,
                    iden,
                    itr=0,
                    iter_times=iter_budget,
                    improved=improved_flag,
                    num_seeds=args.num_seeds
                )

            #deltas
            if attack_name == "Baseline":
                baseline_acc = acc
                delta = 0.0
                improvement = 0.0
            else:
                delta = acc - baseline_acc
                improvement = (delta / baseline_acc) * 100 if baseline_acc > 0 else 0.0

            print(f"Results for {attack_name} attack: Acc={acc:.4f}, Acc5={acc5:.4f}, Var={var:.6f}, Var5={var5:.6f}, Runtime={runtime:.2f} sec, Delta={delta:.4f}, Improvement={improvement:.2f}%")   
                
            writer.writerow([
                args.model,
                attack_name,
                iter_budget,
                acc,
                acc5,
                var,
                var5,
                runtime,
                delta,
                improvement
            ])
            file.flush()

        else:
            #Phase 3: Transfer attack and Defense Evaluation
            if write_headers:
                writer.writerow(column_headers_phase3)

            if args.dist_flag:
                attack_name = "Distributional_Recovery"
                improved_flag = True
            elif args.improved_flag:
                attack_name = "Improved_GAN"
                improved_flag = True
            else:
                attack_name = "Baseline"
                improved_flag = False

            logger.info(f"==> Phase3: {attack_name}")
            iden = torch.from_numpy(np.arange(args.num_ids))
            iter_budget = iter_list[0]

            if args.dist_flag:
                acc, acc5, var, var5, runtime = dist_inversion(
                    G, D, T, E,
                    iden,
                    itr=0,
                    iter_times=iter_budget,
                    improved=improved_flag,
                    num_seeds=args.num_seeds,
                    defense=args.defense,
                    noise_sigma=args.noise_sigma,
                    smooth_alpha=args.smooth_alpha
                )
            else:
                acc, acc5, var, var5, runtime = inversion(
                    G, D, T, E,
                    iden,
                    itr=0,
                    iter_times=iter_budget,
                    improved=improved_flag,
                    num_seeds=args.num_seeds,
                    defense=args.defense,
                    noise_sigma=args.noise_sigma,
                    smooth_alpha=args.smooth_alpha
                )

            print(f"Results for {attack_name} attack with defense={args.defense}, noise_sigma={args.noise_sigma}, smooth_alpha={args.smooth_alpha}: Acc={acc:.4f}, Acc5={acc5:.4f}, Var={var:.6f}, Var5={var5:.6f}, Runtime={runtime:.2f} sec")

            writer.writerow([
                args.model_trained_against if args.model_trained_against else args.model,
                args.model,
                attack_name,
                args.defense,
                args.noise_sigma,
                args.smooth_alpha,
                iter_budget,
                acc,
                acc5,
                var,
                var5,
                runtime
            ])
            file.flush()

    logger.info(f"Results saved to {csv_path}")