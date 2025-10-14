import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.constants import *
from common.model import create_model
from data_loader import load_videos
from common.logger import log_data_quantities
from common.evaluation import adjusted_precision_by_prevalence

FOLDER_PATH = "../../../../img/v"
X, y = load_videos(FOLDER_PATH)
log_data_quantities(y)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold_no = 1

test_accuracies=[]
train_accuracies=[]

for train_index, test_index in kf.split(X, y):
    print(f"\nFold {fold_no}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

    model = create_model(frame_count=FRAME_COUNT, img_size=IMG_SIZE, num_classes=NUM_CLASSES)

    print(f"Training Fold {fold_no} ...")
    model.fit(X_train, y_train_cat,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_test, y_test_cat),
              verbose=1)

    train_accuracy=model.evaluate(X_train,y_train_cat)
    test_accuracy=model.evaluate(X_test,y_test_cat)

    train_accuracies.append(train_accuracies)
    test_accuracies.append(test_accuracies)


    K.clear_session()

    fold_no += 1


print(train_accuracies)
print(test_accuracies)

print(f"Train accuracy={np.mean(train_accuracies)},Test accuracy={np.mean(test_accuracies)}")