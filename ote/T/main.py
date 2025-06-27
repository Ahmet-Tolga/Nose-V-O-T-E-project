import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from common.constants import *
from common.model import create_model
from common.logger import log_collapse_existence,log_collapse_type,log_final_test
from collapse_existence import load_data_for_collapse_existence,load_data_for_collapse_type,load_data_for_final_test

#Collapse existence model traning section

X, y = load_data_for_collapse_existence(FOLDER_PATH)

log_collapse_existence(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

collapse_existing_model = create_model(frame_count=FRAME_COUNT)

print("collapse existing model is training!")
collapse_existing_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=(X_test,y_test))

#Collapse type model training section

X, y = load_data_for_collapse_type(FOLDER_PATH)

log_collapse_type(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

collapse_type_model = create_model(frame_count=FRAME_COUNT)

print("collapse type model is training!")
collapse_type_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_data=(X_test,y_test))

#Final test

X_final, y_final = load_data_for_final_test(FOLDER_PATH)

log_final_test(y_final)

true_labels = []
predicted_labels = []

for i in range(len(X_final)):
    x = X_final[i:i+1]
    true_label = y_final[i] 

    pred_collapse_prob = collapse_existing_model.predict(x, verbose=0)[0][0]
    pred_collapse = 1 if pred_collapse_prob > 0.5 else 0

    if pred_collapse == 0:
        predicted_labels.append(0) 
        true_labels.append(true_label)
    else:
        pred_type = collapse_type_model.predict(x, verbose=0)
        predicted_type = np.argmax(pred_type[0]) 

        predicted_labels.append(predicted_type)
        true_labels.append(true_label)

accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Final test accuracy: {accuracy:.2f}")




