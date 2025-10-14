import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from common.constants import *
import numpy as np

def positional_encoding(seq_len, embed_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
    angle_rads = pos * angle_rates
    pe = np.zeros((seq_len, embed_dim))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(pe, dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, inputs])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x_ff = layers.Dense(ff_dim, activation='relu')(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x

def create_video_transformer(frame_count=FRAME_COUNT, img_size=IMG_SIZE, embed_dim=EMBEDDING_DIM, num_heads=4, ff_dim=256, num_classes=NUM_CLASSES, num_layers=2):
    frame_input = Input(shape=(img_size, img_size, 1))
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(frame_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embed_dim, activation='relu')(x)


    frame_embed_model = Model(frame_input, x)

    # Video input
    video_input = Input(shape=(frame_count, img_size, img_size, 1))
    frame_embeddings = layers.TimeDistributed(frame_embed_model)(video_input)

    # Positional encoding

    pos_encoding = tf.expand_dims(positional_encoding(frame_count, embed_dim), 0)
    frame_embeddings = layers.Lambda(lambda x: x + pos_encoding)(frame_embeddings)

    # Transformer encoders
    x = frame_embeddings
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size=embed_dim//num_heads, num_heads=num_heads, ff_dim=ff_dim)

    # classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(video_input, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
