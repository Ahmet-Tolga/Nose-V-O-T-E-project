import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


n_samples_train, height, width = X_train.shape
n_samples_test, _, _ = X_test.shape

# Reshape the data to 2D before scaling
X_train_reshaped = X_train.reshape(n_samples_train, height * width)
X_test_reshaped = X_test.reshape(n_samples_test, height * width)

# Initialize and apply the StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_reshaped)
X_test_scaled = sc.transform(X_test_reshaped) # Use transform for the test set

# Reshape the data back to 3D for the model
X_train_final = X_train_scaled.reshape(n_samples_train, height, width)
X_test_final = X_test_scaled.reshape(n_samples_test, height, width)

model = create_model(FRAME_COUNT=FRAME_COUNT,NUM_CLASSES=NUM_CLASSES)

print("CNN model is training!")
model.fit(X_train_final, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=(X_test_final,y_test))

print("Model training completed!")
