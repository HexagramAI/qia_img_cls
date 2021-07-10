import numpy as np

import os
import pandas as pd

from tensorflow.keras.preprocessing import image
from scipy import spatial
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

from PIL import Image, ImageEnhance, ImageOps


def load_real_images(folder):
    """Load real images folder, return list, first element is array, second is file name.
    images[0] -> list of array, shape (224, 224, 3)
    images[1] -> file names
    """
    images = [[], []]  # img, filename
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in [".jpeg", ".jpg", ".png"]]):
            img = image.load_img(os.path.join(folder, filename), target_size=(224, 224))
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                images[0].append(img)
                images[1].append(filename)
    return images


def load_synthetic_images(folder):
    """Load real images folder, return list, first element is array, second is file name.
    images[0] -> list of array, shape (224, 224, 3)
    images[1] -> folder name 
    images[2] -> file name
    """
    images = [[], [], []]  # img, folder name, filename
    for name in os.listdir(folder):
        sub_dir_path = folder + "/" + name
        if os.path.isdir(sub_dir_path):
            for filename in os.listdir(sub_dir_path):
                if any([filename.endswith(x) for x in [".jpeg", ".jpg", ".png"]]):
                    img = image.load_img(
                        os.path.join(sub_dir_path, filename), target_size=(224, 224)
                    )
                    img = img_to_array(img)
                    img = img.reshape((1,) + img.shape)
                    if img is not None:
                        images[0].append(img)
                        images[1].append(name)
                        images[2].append(filename)
    return images


# def create_model():
#     # loading vgg16 model and using all the layers until the 2 to the last to use all the learned cnn layers
#     vgg = VGG16(include_top=True)
#     model = Model(vgg.input, vgg.layers[-2].output)
#     return model


def extract_features(imgs_arr):
    features = np.zeros((len(imgs_arr), 4096))
    imgs_arr = np.array(imgs_arr)
    for i in range(imgs_arr.shape[0]):
        features[i] = model.predict(imgs_arr[i])
    return features


if __name__ == "__main__":
    print("Loading Images\n")
    real_images = load_real_images("/Users/rita/Dataset/Real_Effusion")
    synthetic_images = load_synthetic_images("/Users/rita/Dataset/Synthetic_Effusion")

    print("Loading Model\n")
    model = create_model()

    print("Extracting Features\n")
    real_feature = extract_features(real_images[0])
    synthetic_feature = extract_features(synthetic_images[0])
