import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_data
from common.constants import *
from common.logger import log_data_quantities
from common.model import create_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X_train,X_test,y_train,y_test,test_videos= load_data()

y_train=to_categorical(y_train,num_classes=3)
y_test=to_categorical(y_test,num_classes=3)

model=create_model()

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save("cnn_model.h5")

print("Model cnn_model.h5 olarak kaydedildi.")
