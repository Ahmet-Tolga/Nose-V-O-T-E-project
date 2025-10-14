import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from common.constants import IMG_SIZE,FRAME_COUNT

def build_cnn3d(input_shape=(FRAME_COUNT,IMG_SIZE,IMG_SIZE,1)):
    model = models.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding="same", input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((1, 2, 2)),

        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
