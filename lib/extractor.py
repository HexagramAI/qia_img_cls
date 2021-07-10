import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model


class ImgExtractor(object):
    def __init__(
        self, model="VGG16",
    ):
        self.models = [
            "VGG16",
            "VGG19",
            "ResNet50",
            "DenseNet121",
            "DeseNet169",
            "inception_v3",
            "inceptionResNetV2",
        ]
        if model == "VGG16":
            self.model = tf.keras.applications.VGG16()
        elif model == "VGG19":
            self.model = tf.keras.applications.VGG19()
        elif model == "Resnet50":
            self.model = tf.keras.applications.ResNet50()
        elif model == "DenseNet121":
            self.model = tf.keras.applications.DenseNet121()
        elif model == "DeseNet169":
            self.model = tf.keras.applications.DeseNet169()
        elif model == "inception_v3":
            self.model = tf.keras.applications.inception_v3()
        elif model == "inceptionResNetV2":
            self.model = tf.keras.applications.inceptionResNetV2()
        else:
            print("you must select one model from {}".format(self.models))

    def rotate_img(self, img_arr):
        data_augmentation = tf.keras.Sequential(
            [layers.experimental.preprocessing.RandomRotation(0.2)]
        )
        # layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"), # heart location matters

        augmented_img_arr = data_augmentation(img_arr)

        return augmented_img_arr

    def extract_features(self, img_path):
        """Get features from image. 针对之前我们写的代码. Only for single images."""
        raw_arr = self.read_img(img_path)
        # generate x, dim of x should be (None,224,224,3)
        img_arr = np.expand_dims(raw_arr, axis=0)
        features = self.model.predict(img_arr)
        return features

    def read_img(self, img_path):
        # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        # raw_arr = tf.keras.preprocessing.image.img_to_array(img)
        raw_img = Image.open(img_path)
        if np.array(raw_img).shape[-1] == 4:
            """Some image has 4 dimension, first 3 are RGB, 4 is toumingdu."""
            raw_img = raw_img.convert("RGB")
        resized_img = raw_img.resize((224, 224))
        raw_arr = np.array(resized_img)
        return raw_arr

    def show_img(self, img_path, s):
        img_arr = self.read_img(img_path)
        plt.imshow(img_arr, aspect="auto")
        plt.show()
