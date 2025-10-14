import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from common.evaluation import adjusted_precision_by_prevalence
from tensorflow.keras.utils import to_categorical

from common.constants import *
from common.model import create_model
from data_loader import load_videos
from common.logger import log_data_quantities

FOLDER_PATH = "../../../../../img/ote"

X, y = load_videos(FOLDER_PATH)

log_data_quantities(y)

y=to_categorical(y,num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(frame_count=FRAME_COUNT, img_size=IMG_SIZE,num_classes=NUM_CLASSES)

print("CNN model is training!")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=(X_test,y_test))

print("Model training completed!")


#Classification report

y_pred = model.predict(X_test)

y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

print(y_test_labels)
print(y_pred_labels)

custom_prev = {0:235, 1:22, 2:3}

print(classification_report(y_test_labels, y_pred_labels))

ap_custom, labels, cm, supports = adjusted_precision_by_prevalence(
    y_test_labels,
    y_pred_labels,
    prevalences=custom_prev
)

print("\nAdjusted precisions (dataset prevalence):")
for k in labels:
    print(f" class {k}: {ap_custom[int(k)]:.4f}")

