import tensorflow as tf
from tensorflow.keras.layers import LSTM
from common.constants import IMG_SIZE,FRAME_COUNT,NUM_CLASSES

def create_model(frame_count=FRAME_COUNT,img_size=IMG_SIZE,num_classes=NUM_CLASSES):
    input_shape = (frame_count,img_size * img_size)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(LSTM(64, input_shape=input_shape, return_sequences=True)),
    tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(LSTM(32, return_sequences=False)),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0003,
        decay_steps=5000,
        decay_rate=0.9)
    
    train_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(train_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model