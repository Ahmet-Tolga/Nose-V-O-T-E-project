import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from common.constants import *
from common.model import create_model
from data_loader import load_data
from common.logger import log_data_quantities

FOLDER_PATH = "../../../../img/ote"

X, y = load_data(FOLDER_PATH)

log_data_quantities(y)

y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(FRAME_COUNT=FRAME_COUNT,NUM_CLASSES=NUM_CLASSES)

print("CNN model is training!")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=(X_test,y_test))

print("Model training completed!")
