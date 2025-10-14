from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from common.constants import CATEGORIES

import numpy as np

from data_loader import load_data
from common.model import create_model

X, y = load_data(CATEGORIES)

X = np.expand_dims(X, axis=1)

y = to_categorical(y, num_classes=len(CATEGORIES))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(input_shape=(1, 2048), num_classes=len(CATEGORIES))

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)

model.save("resnet_lstm_model.h5")
