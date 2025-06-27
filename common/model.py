import tensorflow as tf

def create_model(frame_count=25):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(frame_count, 2048)), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
        tf.keras.layers.Lambda(lambda x: x[:, frame_count // 2, :]),
        tf.keras.layers.Activation('sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model