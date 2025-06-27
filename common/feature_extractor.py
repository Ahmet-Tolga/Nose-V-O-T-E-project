from tensorflow.keras.applications import ResNet50

def create_feature_extractor(IMAGE_SIZE=224,channels=3):
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(IMAGE_SIZE,IMAGE_SIZE, channels))
    base_model.trainable = False
    return base_model
