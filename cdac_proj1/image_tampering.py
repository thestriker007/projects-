import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import cv2
import io



def compute_ela(image_path, quality=90):
    try:
        
        original = Image.open(image_path).convert('RGB')
        
        # Save to memory instead of writing to disk to prevent I/O errors
        temp_io = io.BytesIO()
        original.save(temp_io, 'JPEG', quality=quality)
        temp_io.seek(0)
        
        temporary = Image.open(temp_io)
        
        ela_image = ImageChops.difference(original, temporary)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
            
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        return ela_image
    except Exception as e:
        print(f"Error computing ELA for {image_path}: {e}")
        return None


def prepare_dataset(dataset_path, image_size=(128, 128), sample_limit=None):
    X = []
    y = []
    
    categories = {'Au': 0, 'Tp': 1}
    
    for category, label in categories.items():
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
            
        print(f"Processing {category} images...")
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                ela_img = compute_ela(image_path)
                
                if ela_img:
                    # Resize and append to arrays
                    ela_img = ela_img.resize(image_size)
                    ela_array = np.array(ela_img) / 255.0 
                    
                    X.append(ela_array)
                    y.append(label)
                    count += 1
                
                if sample_limit and count >= sample_limit:
                    break
    
    return np.array(X), np.array(y)


DATASET_PATH = '/Users/princeroy/cdac_proj1/CASIA2'
X, y = prepare_dataset(DATASET_PATH, sample_limit=None)
print("Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")


def build_model(input_shape):
    model = Sequential()
    
    # Data Augmentation to prevent overfitting
    model.add(tf.keras.layers.RandomFlip("horizontal", input_shape=input_shape))
    model.add(tf.keras.layers.RandomRotation(0.05))
    
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Block 4
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Block 5 - Further abstracts features and reduces spatial dimensions
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Transition to Dense Layers
    model.add(Flatten()) # Shrinks from 16384 to 8192 parameters vs previous model
    
    # Fully Connected Layer
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid')) 
    
    return model

input_shape = (128, 128, 3)
model = build_model(input_shape)

model.compile(optimizer=Adam(learning_rate=0.0005), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

epochs = 50
batch_size = 32

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr]
)



plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")


def predict_image(image_path, model, image_size=(128, 128)):
    ela_img = compute_ela(image_path) 
    if ela_img:
        resized_ela = ela_img.resize(image_size)
        img_array = np.array(resized_ela) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = "Tampered / Forged" if prediction > 0.5 else "Authentic"
        
        plt.figure(figsize=(10, 5))
        
        # Show Original Image
        plt.subplot(1, 2, 1)
        original = Image.open(image_path).convert('RGB')
        plt.imshow(original)
        plt.title(f"{label} ({confidence*100:.2f}%)")
        plt.axis('off')
        
        # Show ELA
        plt.subplot(1, 2, 2)
        plt.imshow(ela_img)
        plt.title('ELA Enhancement')
        plt.axis('off')
        plt.show()
        
        return label, confidence
    else:
        print("Error processing image.")
        return None, None

