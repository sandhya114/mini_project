import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to create the ResNet50-based disease classification model
def create_resnet50_model(input_shape=(256, 256, 3), num_classes=3):
    # Load ResNet50 model pretrained on ImageNet without the top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add new top layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Function to train the ResNet50-based disease classification model
def train_resnet50_model():
    # Create the ResNet50-based model
    model = create_resnet50_model()

    # Data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CTraining/Training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        classes=['Late_Blight', 'Healthy', 'Early_Blight']
    )

    validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CValidation/Validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        classes=['Late_Blight', 'Healthy', 'Early_Blight']
    )

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator
    )

    return model

# Train and save the ResNet50-based disease classification model
if __name__ == '__main__':
    model = train_resnet50_model()  
    model.save('resnet_classify.h5')
