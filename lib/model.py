"""For classification model."""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model():
    # (224 * 224 * 3) -> (5 * 5 * 20) > 500 > 50 > 20 > 2
    model = tf.keras.applications.VGG16(include_top=True)
    num_classes = 2

    # define layer
    flat = layers.Flatten(name="qiaFlat")  # 5 * 5 * 20 -> 500
    dense1 = layers.Dense(50, activation="relu", name="qiaDense")  # 500 > 50
    dropout1 = layers.Dropout(0.5, name="qiaDropout")  # 50 -> 50% 的数随机变0
    batch1 = layers.BatchNormalization()  # no need to do
    output = layers.Dense(num_classes, activation="softmax")  # 50 -> 2 (0, 1)
    # number of classes in your dataset e.g. 20

    # define model input process
    x = flat(model.output)  # 5 * 5 * 20 -> 200
    x = dense1(x)  #
    x = dropout1(x)
    x = batch1(x)
    predictions = output(x)

    # create graph of your new model
    new_model = Model(model.input, predictions)

    # compile the model
    new_model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return new_model
