import tensorflow as tf

def create_model(frame_count=25,img_size=120,num_classes=3):
    input_shape = (frame_count,img_size , img_size, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, kernel_size=(3,3,3), activation='relu',padding="same", input_shape=input_shape),
        tf.keras.layers.MaxPooling3D(pool_size=(1,2,2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv3D(64, kernel_size=(3,3,3), padding="same",activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1,2,2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv3D(64, kernel_size=(3,3,3),padding="same", activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1,2,2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model