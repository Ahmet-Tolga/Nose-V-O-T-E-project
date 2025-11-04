import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from common.evaluation import adjusted_precision_by_prevalence
from tensorflow.keras.utils import to_categorical

from common.constants import *
from common.model import create_model
from data_loader import load_data
from common.logger import log_data_quantities

FOLDER_PATH = "../../../../img/v"

X, y = load_data(FOLDER_PATH)

log_data_quantities(y)

y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(FRAME_COUNT=FRAME_COUNT,NUM_CLASSES=NUM_CLASSES)

print("Bidirectional LSTM model is training!")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=0.2)

print("Model training completed!")

#Classification report

y_pred = model.predict(X_test)

y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

print(f"Test Accuracy is {accuracy_score(y_pred_labels,y_test_labels)}")

print(y_test_labels)
print(y_pred_labels)

custom_prev = {0:73, 1:132, 2:120}

print(classification_report(y_test_labels, y_pred_labels))

ap_custom, labels, cm, supports = adjusted_precision_by_prevalence(
    y_test_labels,
    y_pred_labels,
    prevalences=custom_prev
)

print("\nAdjusted precisions (dataset prevalence):")
for k in labels:
    print(f" class {k}: {ap_custom[int(k)]:.4f}")
