import tensorflow as tf

def build_model1():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
  
    
    # First convolutional block
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    
    # Second convolutional block
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Skip connection after the second convolutional block
    skip_connection1 = tf.keras.layers.Conv2D(128, (2, 2), strides=(2, 2), padding='same')(x)
    skip_connection1 = tf.keras.layers.BatchNormalization()(skip_connection1)
    
    # Third convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, skip_connection1])
    
    # Skip connection after the third convolutional block
    skip_connection2 = tf.keras.layers.Conv2D(128, (2, 2), strides=(1, 1), padding='same')(x)
    skip_connection2 = tf.keras.layers.BatchNormalization()(skip_connection2)
    
    
    # Fourth convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, skip_connection2])
    
    # Skip connection after the 5th convolutional block
    skip_connection3 = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)
    skip_connection3 = tf.keras.layers.BatchNormalization()(skip_connection3)
    
    
    # Fifth convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Sixth convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, skip_connection3])
    
    # Seventh convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # MaxPooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    
    # Flatten layer
    x = tf.keras.layers.Flatten()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
