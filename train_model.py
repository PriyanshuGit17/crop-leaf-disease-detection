import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================================
# Dataset Path
# ================================
dataset_path = 'leaf_dataset/train'

if not os.path.exists(dataset_path):
    raise FileNotFoundError("❌ Dataset folder not found at: leaf_dataset/train")


# ================================
# Image Generator
# ================================
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# ================================
# CNN Model
# ================================
model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# ================================
# Train the Model
# ================================
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)


# ================================
# Save Model + Class Indices
# ================================
MODEL_NAME = "leaf_disease_model.h5"

model.save(MODEL_NAME)

with open("class_indices.json", 'w') as f:
    json.dump(train_generator.class_indices, f)

print("\n====================================")
print("✅ Model trained and saved successfully!")
print(f"📁 Model saved as: {MODEL_NAME}")
print("📁 Class indices saved as: class_indices.json")
print("====================================\n")
