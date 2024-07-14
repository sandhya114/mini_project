from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Function to create the verification model
def create_verification_model(input_shape=(256, 256, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train the verification model
def train_verification_model():
    model = create_verification_model()

    # Data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CTraining',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CValidation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary'
    )

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=40
    )

    return model

# Train and save the verification model
if __name__ == '__main__':
    model = train_verification_model()
    model.save('vmodel2.h5')
    
