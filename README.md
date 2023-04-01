# SignatureIdentificationAndVerification

First rank winner in the Computer Vision Course Competition for class 2022-2023 at Ain Shams University.

This computer vision projct aims to do 3 main things:

1. Identify which person a signature belongs to (5 people: personA, personB, personC, personD, personE).
2. Verify if the signature is real or forged.
3. In a document, detect where the signal is.

Data format: 5 folders for 5 people, each contains 2 folders for training and testing, that contain png images

# Project Lifecycle

1. Preprocessing
2. Idntification
3. Verification
4. Object Detection
5. Deployment on Microsof Azure

Project can be found at: https://github.com/NourKamaly/SignatureIdentificationAndVerification

# Tech Stack

Programming Languages: Python 3.9, JavaScript

Markup Languages: HTML

Style Sheet Language: CSS, Sass

Libraries used: cv2, os, NumPy, Keras, TenserFlow, matplotlib, tqdm, glob , sklearn, PyTorch

# Signature Identification
4 models were experimented with:
1. VGG 16
2. Inception v3 
3. ResNet 50
4. Vision Transformers (implemented but havn't been run yet)

# Signature Verification
We experimented with the Siamese Neural Network (sometimes called a twin neural network) is an artificial neural network that uses the same weights while working in tandem on two different input vectors to compute comparable output vectors.Often one of the output vectors is precomputed, thus forming a baseline against which the other output vector is compared. This is similar to comparing fingerprints but can be described more technically as a distance function for locality-sensitive hashing.
![Siamese](https://user-images.githubusercontent.com/76780379/229292250-9f08b3fa-d686-42d8-8fa3-c8059ea80b27.png)
We used this network to compare the input signature claiming to be a specific person.

# Object Detection

2 models were experimented with: 
1. YOLO v5 
2. YOLO v7

YOLO v5 ended up with better results

![ObjectDetection](https://user-images.githubusercontent.com/76780379/229291676-2f7a5036-9ff5-4c2d-8059-d2f017740572.png)

# Google Colab / Kaggle notebooks

Data augmentation: https://colab.research.google.com/drive/1-R-sVWvq27pp3pij6lWWNy9gJVuw-ApL?usp=sharing

VGG 16: https://colab.research.google.com/drive/15Zahd23WxDjUNjhsuraHOCGtYs0-F_ub?usp=sharing

Inception v3: https://colab.research.google.com/drive/1Ji063ZVjKWyXvas-88cZ7NoRhgoNxGGS?usp=sharing

ResNet 50: https://colab.research.google.com/drive/1t3ursrEiTEuiIUTMbLeZicn32SHTP__L?usp=sharing

YOLO v5: https://www.kaggle.com/code/saraosmanbaza/signature-detection-yolov5

YOLO v7: https://www.kaggle.com/code/saraosmanbaza/yolov7-signature-detection

Testing Script: https://colab.research.google.com/drive/12hdNSzO0Su-G3T3TBCmbt6xgvbggmzN1?usp=sharing

Google drive that has the dataset, augmented pictures,saved models: https://drive.google.com/drive/folders/139Jt4j3DrYHtpB-Wq1d0CnRmh63a8frf?usp=sharing

# Deployment

https://user-images.githubusercontent.com/76780379/229291557-92a3e179-3062-4969-9e2f-eb73f92e0605.mp4




