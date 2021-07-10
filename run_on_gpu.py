import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from scipy import spatial
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers, Model
from lib.load import load_real_images, load_synthetic_images
from lib.extractor import ImgExtractor
from lib.model import build_model

from PIL import Image, ImageEnhance, ImageOps


"""Define customer variable."""
epochs = 10
use_aug = True  # False
lowB = 0.34
upB = 1
# python run_on_gpu.py

"""Dont' change below"""
# S0: Load Images
real_images = load_real_images("Dataset/Real_Effusion")
synthetic_images = load_synthetic_images("Dataset/Synthetic_Effusion")

### 0.1 Prepare data augmentation (rotate as an example)

# Create same size of syn images through rotation

if use_aug == True:
    # 旋转90度
    augModel = tf.keras.Sequential(
        [layers.experimental.preprocessing.RandomRotation(0.5),]
    )

    def rotate_img(img_arr, augModel):
        augmented_img_arr = augModel(img_arr)
        return augmented_img_arr

    synImgAugArr = [
        rotate_img(synthetic_images[0][i], augModel)
        for i in range(len(synthetic_images[0]))
    ]

    print(len(synImgAugArr), synImgAugArr[0].shape)

    print(
        "before data aug, data size: ",
        len(synthetic_images[0]),
        len(synthetic_images[1]),
        len(synthetic_images[2]),
    )

    ### 1.3 Merge data augmentation into syn images variable, 相同格式并入, 0 -> arr, 1-> folder name 2-> file name
    synthetic_images[0] += synImgAugArr  # 都是list，所以可以相加
    synthetic_images[1] += synthetic_images[1]
    synthetic_images[2] += synthetic_images[2]
    print(
        "After data aug, data size: ",
        len(synthetic_images[0]),
        len(synthetic_images[1]),
        len(synthetic_images[2]),
    )

# S1: Extract Features
### 1.1: Prepare images arr into model input format
realImgMIArr = np.squeeze(np.array(real_images[0]))
realImgMIArr.shape

synImgMIArr = np.squeeze(np.array(synthetic_images[0]))
synImgMIArr.shape

### 1.2 Use pretrained VGG16 model to extract features
extractor = ImgExtractor(model="VGG16")

realExtFea = extractor.model.predict(realImgMIArr)
synExtFea = extractor.model.predict(synImgMIArr)
print(realExtFea.shape, synExtFea.shape)
### 1.3 Append Extracted feature to current list

# - real_images is a list has 3 element -> 1: raw arr 2: file names 3: extracted features
# - synthetic_images is a list has 4 element -> 1: raw arr 2: folder name 3: file name 4: extracted features

real_images.append(realExtFea)
synthetic_images.append(synExtFea)

# S2: Compare similarity Score

# - 两两对比: 利用index 循环
simi_record = []
for r_index in range(len(real_images[0])):
    for s_index in range(len(synthetic_images[0])):
        r_fea = real_images[2][r_index]
        s_fea = synthetic_images[3][s_index]

        # cal similarity + record real path + record syn path
        simi_score = cosine_similarity(r_fea.reshape(1, -1), s_fea.reshape(1, -1))[0][0]
        r_path = real_images[1][r_index]
        s_path = synthetic_images[1][s_index] + "/" + synthetic_images[2][s_index]

        r_rawArr = real_images[0][r_index]
        s_rawArr = synthetic_images[0][s_index]

        simi_record.append([r_path, s_path, simi_score, r_rawArr, s_rawArr])
simiDF = pd.DataFrame(
    simi_record, columns=["realPath", "synPath", "simiScore", "rRawArr", "sRawArr"]
)

# S3: Build MI datasets based on simi Score
def build_mi(simiDF, lowB, upB):
    return simiDF.query(f"{lowB} <= simiScore <= {upB}")


trainDF1 = build_mi(simiDF, lowB, upB)
# trainDF2 = build_mi(simiDF, 0.14, 0.34)
# trainDF3 = build_mi(simiDF, 0.04, 0.14)
# trainDF4 = build_mi(simiDF, 0, 0.04)

# S4: Train model to compare accuracy
clsModel = build_model()

# 1 class, posArr is X , posLabel is Y
posArr = np.array(trainDF1.iloc[:, 3].tolist())
negArr = np.array(trainDF1.iloc[:, 4].tolist())

posLabel = np.array([0, 1] * len(posArr)).reshape(-1, 2)
negLabel = np.array([1, 0] * len(negArr)).reshape(-1, 2)
posArr = np.squeeze(posArr)
negArr = np.squeeze(negArr)
print(posArr.shape, negArr.shape, posLabel.shape, negLabel.shape)

X = np.concatenate([posArr, negArr])
y = np.concatenate([posLabel, negLabel])
print(X.shape, y.shape)

### 4.2 Train model, 400 data as an example
# 1 class, posArr is X , posLabel is Y
def cal_accuray_for_trainDF(trainDF, epochs=10):
    posArr = np.array(trainDF.iloc[:, 3].tolist())
    negArr = np.array(trainDF.iloc[:, 4].tolist())
    # 0 class
    posLabel = np.array([0, 1] * len(posArr)).reshape(-1, 2)
    negLabel = np.array([1, 0] * len(negArr)).reshape(-1, 2)

    posArr = np.squeeze(posArr)
    negArr = np.squeeze(negArr)

    X = np.concatenate([posArr, negArr])
    y = np.concatenate([posLabel, negLabel])

    clsModel.fit(X, y, batch_size=128, verbose=1, epochs=epochs)
    loss, accuracy = clsModel.evaluate(X, y)
    return loss, accuracy


loss, accuracy = cal_accuray_for_trainDF(trainDF1, epochs=epochs)
print(loss, accuracy)
