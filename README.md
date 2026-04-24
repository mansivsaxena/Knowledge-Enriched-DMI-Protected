# Experimental Task

# Knowledge-Enriched-Distributional-Model-Inversion-Attacks

**Knowledge Enriched Distributional Model Inversion Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

## About this work
The paper primarily assumes a white-box set-up where the GAN is trained using the same architecture as the target model. There is no investigation into whether a GAN prior trained on a ”simple” architecture can effectively attack a ”complex” one, or the other way round. 
Our question - **Do GAN priors trained on one architecture generalize asymmetrically to other architectures, potentially outperforming white-box alignment?**
More specifically, we look at whether a ”general” face recognition model which was trained on a simpler architecture (VGG16) can effectively attack modern architectures (IR152, FaceNet64), and if this transfer can outperform the white-box approach.

## Experiments
* 1: Iteration Budget v/s Convergence - **How does the iteration budget affect attack accuracy, variance, and runtime when using the Improved GAN objective across different architectures?**
* 2: Different Types of Attacks Comparisions - **How do baseline, improved GAN, and distributional recovery attacks compare across different model architectures?**
* 3: Cross-Model Transfer Attacks (and Defenses) - **Does an inversion-specific GAN trained on one model transfer asymmetrically and possibly outperform white-box set-up?**

## Requirement
This code has been tested with Python 3.6, PyTorch 1.0 and cuda 10.0. 

## Note
* All code runs were done in Google Colab since I have MacOS and Cuda is not straightforward to set up.
* All steps provided below are to run the code in google colab

## Getting Started 
* Download this code repo locally
* Download pre-trained checkpoints for the 3 models (VGG16, IR152, FaceNet64) from https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN?usp=sharing and save to folder "target_model" in your code directory 
* Pretrained binary GAN and inversion-specific GAN can be downloaded at https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing. Download it and save to folder "improvedGAN" in your code folder. 

## Running
* After following the above steps, download the .ipynb notebooks "running-attacks.ipynb" and "statistical-analysis.ipynb" and open them on Google colab.
* Make sure the checkpoints are downloaded and saved in your code folder
* Then upload this code folder to your google drive
* Make sure you open the Colab notebooks with the same Google Drive account that you used to save the code folder
* Run the cells in google colab - first run the "running-attacks.ipynb" notebook, then the "statistical-analysis.ipynb" notebook

## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
