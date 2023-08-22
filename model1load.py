import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# print(os.listdir())




# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


y_test = pd.DataFrame(y_test)
y_test = y_test.idxmax(1)

y_test = [classes[i] for i in y_test]



model = load_model(r"models/alexnet_model")


# print(model.summary())

print("PREDCITION : ",model.predict(x_test[0]))


