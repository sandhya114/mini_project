from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os

# Function to create the verification model using ResNet50
def create_verification_model(input_shape=(256, 256, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Unfreeze some layers
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to train the verification model
def train_verification_model():
    model = create_verification_model()

    # Data generators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        brightness_range=[0.6, 1.4],
        contrast_range=[0.6, 1.4],
        saturation_range=[0.6, 1.4],
        hue_range=0.1
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CTraining',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary',
        shuffle=True  # Ensure data is shuffled
    )

    validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/galip/OneDrive/Desktop/dataset/Combined_dataset/CValidation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary',
        shuffle=False   # Typically, validation data is not shuffled
    )

    # Debugging: Print out class indices and a batch of data
    print("Class indices:", train_generator.class_indices)
    for data_batch, labels_batch in train_generator:
        print("Data batch shape:", data_batch.shape)
        print("Labels batch shape:", labels_batch.shape)
        break

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('resnet50_best.keras', save_best_only=True, monitor='val_loss')

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]
    )

    return model

# Train and save the verification model
if _name_ == '_main_':
    model = train_verification_model()
    model.save('resnet50.h5')