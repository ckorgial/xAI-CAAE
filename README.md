# Face Aging by Explainable Conditional Adversarial Autoencoders

This repository was created with the aim of incorporating an Explanatory System in the Conditional Adversarial Autoencoder (CAAE). The main source of inspiration was the paper [Age Progression/Regressionby Conditional Adversarial Autoencoder](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Age_ProgressionRegression_by_CVPR_2017_paper.html) proposed by (Zhang, Song, et al.). Moreover, the paper [xAI-GAN: Enhancing Generative Adversarial Networks via Explainable AI Systems](https://arxiv.org/abs/2002.10438) proposed by (Nagisetyy, Graves, et al.) formed the base of adding xAI methods in CAAE. Endly, thanks to Mattan Serry and the [AgeProgression](https://github.com/mattans/AgeProgression) work, we devoloped our code in PyTorch.

# Our Paper

This code was used to implement the experiments of our Paper which published in the ***Image and Vision Computing Journal "Special Issue on Deep Learning Techniques Applied to Faces"***.

[Face Aging by Explainable Conditional Adversarial Autoencoders]()

# Dataset

The dataset we used for training was the CACD + UTKFace which consists of 21267 images in 7 age classes. FGNET with 1002 images used for the test purpose.

# Prerequisites

1. Python 3.7
2. Pytorch 1.2.0
3. Captum 0.4.0
4. Lime 0.2.0.1

# Implementation

The control and change of the Explainable Artificial Intelligence System is done through the net.teachSplit() in mainCAAEsplit.py script. If it is a need to train the Original CAAE, without the addition of the xAI System, explainable is set to **False**. The xAI System is activated by setting explainable to **True**. Different methods of xAI in CAAE can be added by setting the explanation_type with '**saliency**' or '**shap**' or '**lime**'.

# Virtual Environment

```shell
conda create --name <env> --file requirements.txt
```

# Training

```shell
python mainCAAEsplit.py --mode train --epochs 200 --input data/CACD_UTKFace --output checkpoints
```
# Testing 

```shell
python mainCAAEsplit.py --mode test --load checkpoints/epoch200 --input data/FGNET --output results --age 0 or 1 --gender 0 or 1
```

# Authors

***Christos Korgialas and Evangelia Pantraki***



