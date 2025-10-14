import sys
import os
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_dataset
from sklearn.model_selection import train_test_split

from common.logger import log_data_quantities
from common.model import build_cnn3d

X, y = load_dataset("../../../../img")

log_data_quantities(y,"Number of data")

y=to_categorical(y,num_classes=3)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model = build_cnn3d()
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test,y_test))
