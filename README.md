# Knowledge-Enriched-Distributional-Model-Inversion-Attacks

**Knowledge Enriched Distributional Model Inversion Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

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
