import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split

from common.constants import *
from common.model import create_model
from data_loader import load_videos
from common.logger import log_data_quantities
from tensorflow.keras.utils import to_categorical

FOLDER_PATH = "../../../../img/ote"

X, y = load_videos(FOLDER_PATH)

log_data_quantities(y)

y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(frame_count=FRAME_COUNT, img_size=IMG_SIZE,num_classes=NUM_CLASSES)

print("CNN model is training!")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=(X_test,y_test))

print("Model training completed!")
