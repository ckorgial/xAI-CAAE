# Pytorch Implementation of the Explainable Conditional Adversarial Autoencoder

This repository was created with the aim of incorporating an Explanatory System in the Conditional Adversarial Autoencoder (CAAE). The main source of inspiration was the paper "Age Progression/Regressionby Conditional Adversarial Autoencoder" proposed by (Zhang, Song, et al.). Moreover, the paper "Enhancing Generative Adversarial Networks via Explainable AI Systems" proposed by (Nagisetyy, Graves, et al.) formed the base of adding xAI methods in CAAE. Endly, thanks to Mattan Serry and the AgeProgression work, we devoloped our code in PyTorch.

# Dataset

The dataset we used for training is the CACD + UTKFace which consists of 21267 images in 7 age classes. FGNET with 1002 used for the test purpose.

# Our Paper

# Prerequisites

1. Python 3.7
2. Pytorch 1.2.0
3. Captum 0.4.0
4. Lime 0.2.0.1
5. NumPy, scikit-learn, OpenCV, imageio and Matplotlib

# Implementation

The control and change of the Explainable Artificial Intelligence System is done through the net.teachSplit() in mainCAAEsplit.py script. If it is a need to train the Original CAAE, without the addition of the xAI System, explainable is set to False. The addition of the xAI System is activated by setting explainable to True. Different methods of xAI in CAAE can be added by setting the explanation_type with 'saliency' or 'shap' or 'lime'.

# Training

python mainCAAEsplit.py --mode train --epochs 200 --input data/CACD_UTKFace --output Checkpoints

# Testing 

python mainCAAEsplit.py --mode test --load Checkpoints/epoch200 --input data/FGNET --output Results --age 0 or 1 --gender 0 or 1

# Authors

Christos Korgialas and Evangelia Pantraki 


