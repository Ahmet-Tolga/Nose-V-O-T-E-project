from tensorflow.keras.applications import ResNet50
from common.constants import IMG_SIZE

def create_feature_extractor(IMG_SIZE=IMG_SIZE,channels=3):
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(IMG_SIZE,IMG_SIZE, channels))
    base_model.trainable = False
    return base_model
