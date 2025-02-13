import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set the path to your dataset
dataset_path = r'C:\Users\HP\OneDrive\Desktop\RICEDATASET'

# Set the image dimensions (adjust as needed)
img_width, img_height = 150, 150

# Set the number of classes (change this based on your dataset)
num_classes = 9

# Set other hyperparameters
batch_size = 32
epochs = 1 # Increase the number of epochs for better training

# Data augmentation to increase the diversity of training examples
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of the data will be used for validation
)

# Load and augment the data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  
)

# Get the class indices and corresponding class labels
class_indices = train_generator.class_indices
class_labels = list(class_indices.keys())
class_names = [
    "Hispa",
    "bacterial_leaf_blight",
    "leaf_blast",
    "Brown_spot",
    "Healthy",
    "Shath_Blight",
    "leaf_scald",
    "narrow_brown_spot",
    "Tungro"
]

def build_mobilenet_model():
    base_model = MobileNet(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the pre-trained weights

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
best_model_mobilenet = build_mobilenet_model()

history_mobilenet = best_model_mobilenet.fit(
    train_generator,
    epochs=epochs,
    validation_data=train_generator,  # You need to use the validation data during training
    validation_steps=train_generator.samples // batch_size
)

# Make predictions
sample_image_path = r'C:\Users\HP\OneDrive\Desktop\EXAMPLE.jpg'
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(img_width, img_height))
sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image = np.expand_dims(sample_image, axis=0)
sample_image = sample_image / 255.0  # Rescale the image
predictions = best_model_mobilenet.predict(sample_image)
predicted_class_index = np.argmax(predictions)

# Get the predicted class label
predicted_class_label = class_names[predicted_class_index]
print(f'Predicted class label using MobileNet: {predicted_class_label}')

# Confusion Matrix
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify the subset to be used for validation
)

y_true = validation_generator.classes
y_pred = best_model_mobilenet.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (MobileNet)')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()

# Classification Report
print("Classification Report (MobileNet):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot accuracy during training for MobileNet
plt.plot(history_mobilenet.history['accuracy'], label='Training Accuracy (MobileNet)')
plt.plot(history_mobilenet.history['val_accuracy'], label='Validation Accuracy (MobileNet)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy during Training (MobileNet)')

plt.show()