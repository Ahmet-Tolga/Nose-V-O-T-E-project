import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from common.constants import IMG_SIZE,FRAME_COUNT,NUM_CLASSES


def create_model(frame_count=FRAME_COUNT,img_size=IMG_SIZE,num_classes=NUM_CLASSES):
    input_shape = (frame_count,img_size , img_size, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding="same", input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((1, 2, 2)),

        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model