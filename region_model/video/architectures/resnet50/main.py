import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from common.constants import FRAME_COUNT, NUM_CLASSES, FEATURE_DIM,IMG_SIZE,EPOCHS,BATCH_SIZE
from common.model import create_model
from data_loader import load_dataset
from common.logger import log_data_quantities

X, y = load_dataset("../../../../img", img_size=(IMG_SIZE,IMG_SIZE), num_frames=FRAME_COUNT)

log_data_quantities(y)

y = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

model = create_model(NUM_CLASSES=NUM_CLASSES, FRAME_COUNT=FRAME_COUNT, FEATURE_DIM=FEATURE_DIM)

print("Bidirectional LSTM model training started!")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)
print("Model training completed!")

model_path = "resnet_bilstm_model.h5"
model.save(model_path)
print(f"Model saved as {model_path}")

