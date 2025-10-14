import  numpy as np
import os

from common.feature_extracter import extract_features

def load_data(CATEGORIES, base_folder="../../img"):
    X = []
    y = []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(base_folder, category)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y
