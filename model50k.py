import tensorflow as tf

def build_model50k():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    
    # First convolutional block
    x = tf.keras.layers.Conv2D(7, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Second convolutional block
    x = tf.keras.layers.Conv2D(14, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Third convolutional block
    x = tf.keras.layers.Conv2D(28, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Four more pairs of convolutional layers
    for _ in range(4):
        x = tf.keras.layers.Conv2D(28, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    # Final convolutional block
    x = tf.keras.layers.Conv2D(28, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # MaxPooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    
    # Flatten layer
    x = tf.keras.layers.Flatten()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(54, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
