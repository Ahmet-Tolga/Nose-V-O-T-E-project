import tensorflow as tf
from common.constants import FRAME_COUNT,NUM_CLASSES

def create_model(NUM_CLASSES=NUM_CLASSES,FRAME_COUNT=FRAME_COUNT):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAME_COUNT, 2048)), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_CLASSES)),
        tf.keras.layers.Lambda(lambda x: x[:, FRAME_COUNT // 2, :]),
        tf.keras.layers.Activation('softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model