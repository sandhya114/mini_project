from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(input_shape=(256, 256, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    model = create_model()
    
    # Data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CTraining/Training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        classes=['Early_Blight', 'Healthy', 'Late_Blight']
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CValidation/Validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        classes=['Early_Blight', 'Healthy', 'Late_Blight']
    )
    
    model.fit(
        train_generator,
        epochs=60,
        validation_data=validation_generator
    )
    
    return model

# Train and save the model
if __name__ == '__main__':
    model = train_model()
    model.save('model.h5')
