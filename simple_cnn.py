import tensorflow as tf
import keras
import cv2
import numpy as np
from get_data import get_cats_image, get_dogs_image

# CNN model
models = keras.Sequential([
    # input layers
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        input_shape=(64, 64, 3), # rgb
        activation="relu"
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # hidden layers
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(2, 2),
        activation="relu"
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(2, 2),
        activation="relu"
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # output layers
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="relu"
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # flatten layer (flat to vector 1d)
    keras.layers.Flatten(),
    # fully connected output layers
    keras.layers.Dense(
        units=512,
        activation="relu"
    ),
    keras.layers.Dense(
        units=2,
        activation="softmax"
    )
])

# get data and normalize
batches_image = []
labels = []
dogs_data = get_dogs_image()
cats_data = get_cats_image()

# create batch and labels
for image in dogs_data:
    batches_image.append(image)
    labels.append(0) # label for dogs is 0

for image in cats_data:
    batches_image.append(image)
    labels.append(1) # label for cats is 1

# convert batches and labels to tensor
batches_image = tf.stack(batches_image)
labels = tf.constant(labels)
labels = keras.utils.to_categorical(labels, num_classes=2)
# normalize batches value
batches_image = batches_image / 255

# compile model
models.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['acc']
)

# training model
models.fit(
    x=batches_image,
    y=labels,
    epochs=20, # loop for training
    batch_size=16
)

# that good
# ok i have two images
# i'll create predict function for test

def predict(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255
    img = tf.stack([img]) # add batch dim
    y_predict = models(img)
    print(tf.argmax(y_predict, axis=-1))
    return "here is cat" if bool(int(tf.argmax(y_predict, axis=-1))) else "here is dog"

print(predict("./test_dog.jpg"))
# it's true!
# let's try wwith dog image