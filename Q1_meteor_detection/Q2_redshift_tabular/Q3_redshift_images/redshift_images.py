# ----------------------------------------
# Q3: Redshift Prediction using Images
# ----------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images
train_data = datagen.flow_from_directory(
    "images/",
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "images/",
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(train_data, validation_data=val_data, epochs=5)

print("Model training completed")
