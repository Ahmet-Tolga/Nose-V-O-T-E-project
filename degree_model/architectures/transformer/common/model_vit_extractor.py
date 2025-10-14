import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Lambda

def vit_frame_encoder(img_size=(150,150), patch_size=16, embed_dim=128, num_heads=4, num_layers=4):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    inputs = tf.keras.Input(shape=(*img_size, 1))

    patches = Lambda(
        lambda x: tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
    )(inputs)

    n_patches_h = img_size[0] // patch_size
    n_patches_w = img_size[1] // patch_size
    n_patches = n_patches_h * n_patches_w
    patches = layers.Reshape((n_patches, -1))(patches)

    x = layers.Dense(embed_dim)(patches)

    pos_indices = tf.range(n_patches)
    pos_emb = layers.Embedding(input_dim=n_patches, output_dim=embed_dim)(pos_indices)
    pos_emb = tf.expand_dims(pos_emb, 0)
    x = x + pos_emb

    for _ in range(num_layers):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = layers.Dense(embed_dim * 2, activation='relu')(x)
        ff = layers.Dense(embed_dim)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    frame_embedding = layers.GlobalAveragePooling1D()(x)
    frame_embedding = layers.LayerNormalization()(frame_embedding)
    frame_embedding = layers.Dense(embed_dim, activation='relu')(frame_embedding)

    return Model(inputs, frame_embedding, name="vit_frame_encoder")


def video_transformer_model(frame_count, img_size, embed_dim=128, num_heads=4, ff_dim=512, num_classes=3):
    vit_encoder = vit_frame_encoder(img_size=img_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=2)

    video_input = layers.Input(shape=(frame_count, img_size, img_size, 1))

    frame_embeddings = layers.TimeDistributed(vit_encoder)(video_input)

    pos_indices = tf.range(frame_count)
    pos_emb = layers.Embedding(input_dim=frame_count, output_dim=embed_dim)(pos_indices)
    pos_emb = tf.expand_dims(pos_emb, 0) 

    frame_embeddings = frame_embeddings + pos_emb 

    x = frame_embeddings
    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dense(embed_dim)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(video_input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
