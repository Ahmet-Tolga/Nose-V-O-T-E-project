import tensorflow as tf

def create_model(frame_count=25,img_size=120,num_classes=3):
    input_shape = (frame_count,img_size * img_size)

    model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),

    tf.keras.layers.LSTM(64, return_sequences=False),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    
    train_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(train_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model