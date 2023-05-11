# Face Aging by Explainable Conditional Adversarial Autoencoders

This repository was created with the aim of incorporating an Explanatory System in the Conditional Adversarial Autoencoder (CAAE). The main source of inspiration was the paper [Age Progression/Regressionby Conditional Adversarial Autoencoder](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Age_ProgressionRegression_by_CVPR_2017_paper.html) proposed by (Zhang, Song, et al.). Moreover, the paper [xAI-GAN: Enhancing Generative Adversarial Networks via Explainable AI Systems](https://arxiv.org/abs/2002.10438) proposed by (Nagisetyy, Graves, et al.) formed the base of adding xAI methods in CAAE. Endly, thanks to Mattan Serry and the [AgeProgression](https://github.com/mattans/AgeProgression) work, we devoloped our code in PyTorch.

# Our Paper

This repo contains the codes of the proposed xAI-CAAE described in the [**Face Aging by Explainable Conditional Adversarial Autoencoders**](https://www.mdpi.com/2313-433X/9/5/96) paper. The paper was published in the ***Journal of Imaging - MDPI***.

# Dataset

The dataset we used for training was the CACD + UTKFace which consists of 21267 images in 7 age classes. FGNET with 1002 images used for the test purpose. The data folder can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1AvYtsiAiZaO611AMGBK8fSFCqrUlBOOf?usp=sharing).

# Prerequisites

1. Python 3.7
2. Pytorch 1.2.0
3. Captum 0.4.0

# Implementation

The control and change of the Explainable Artificial Intelligence system is achieved through the *net.teachSplit()* in mainCAAEsplit.py script. If it is a need to train the Original CAAE, without the addition of the xAI system, *explainable* is set to **False**. The xAI system is activated by setting *explainable* to **True**. Different methods of xAI in CAAE can be added by setting the *explanation_type* to '**saliency**' or '**shap**'.

# Virtual Environment

```shell
conda create -n xaicaae python=3.7 anaconda
```

```shell
conda activate xaicaae
```

```shell
pip install -r requirements.txt
```

# Training

```shell
python mainCAAEsplit.py --mode train --epochs 200 --input data/CACD_UTKFace --output checkpoints
```
# Testing 

```shell
python mainCAAEsplit.py --mode test --load checkpoints/epoch200 --input data/FGNET --output results/checkpoints/epoch200 --age 0 to 6 --gender 0 or 1
```

# Directory Tree

```
xAI-CAAE   
│   consts.py  
│   mainCAAEsplit.py  
│   modelSplit_v2.py   
│   README.md
|   requirements.txt
│   utils.py
|   utils_xai.py
└───data
    └───CACD_UTKFace
    └───FGNET
```

 # Citation
 
 ```
@Article{jimaging9050096,
AUTHOR = {Korgialas, Christos and Pantraki, Evangelia and Bolari, Angeliki and Sotiroudi, Martha and Kotropoulos, Constantine},
TITLE = {Face Aging by Explainable Conditional Adversarial Autoencoders},
JOURNAL = {Journal of Imaging},
VOLUME = {9},
YEAR = {2023},
NUMBER = {5},
ARTICLE-NUMBER = {96},
URL = {https://www.mdpi.com/2313-433X/9/5/96},
ISSN = {2313-433X},
DOI = {10.3390/jimaging9050096}
}
```

# Authors
Feel free to send us a message for any issue.

***Christos Korgialas (ckorgial@csd.auth.gr) and Evangelia Pantraki (epantrak@csd.auth.gr)***



