import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import time

def load_images_with_cv2(folder_path, img_size=(128, 128)):
    """
    Load images from folders using OpenCV where each folder represents a class label.

    Args:
        folder_path (str): Path to the dataset folder (e.g., './dataset').
        img_size (tuple): Target image size (width, height).

    Returns:
        tuple: (images, labels, class_dict)
            - images (np.ndarray): Array of preprocessed images.
            - labels (np.ndarray): Array of integer labels corresponding to images.
            - class_dict (dict): Dictionary mapping class names (folder names) to integer labels.
    """
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    if not class_names:
        raise ValueError(f"No subdirectories found in {folder_path}. Each subdirectory should represent a class.")

    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
    print("-" * 30)

    total_images_loaded = 0
    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        print(f"Processing class: {class_name}")
        images_in_class = 0
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            # Check if it's a file and has a common image extension
            if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # Load image using OpenCV
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}. Skipping.")
                        continue

                    # Preprocess the image
                    img_resized = cv2.resize(img, img_size)  # Resize
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img_normalized = img_rgb / 255.0  # Normalize pixel values to [0, 1]

                    images.append(img_normalized)
                    labels.append(class_dict[class_name])
                    images_in_class += 1
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                 # Optional: print a message for non-image files or sub-subdirectories
                 # print(f"Skipping non-image file or directory: {img_path}")
                 pass
        print(f" -> Loaded {images_in_class} images.")
        total_images_loaded += images_in_class
        print("-" * 30)

    if total_images_loaded == 0:
         raise ValueError(f"No images were loaded from {folder_path}. Check the directory structure and image files.")

    print(f"Total images loaded: {total_images_loaded}")
    return np.array(images), np.array(labels), class_dict

def build_classification_model(input_shape, num_classes):
    """
    Builds a simple CNN model for image classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.models.Sequential: The compiled Keras model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv2d_1'),
        MaxPooling2D((2, 2), name='maxpool_1'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D((2, 2), name='maxpool_2'),
        Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        MaxPooling2D((2, 2), name='maxpool_3'),
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense_1'),
        Dense(num_classes, activation='softmax', name='output') # Softmax for multi-class classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot labels
                  metrics=['accuracy'])
    model.summary() # Print model summary
    return model

def train_and_save_model(dataset_path="dataset", img_size=(128, 128), epochs=50, batch_size=32):
    """
    Loads data, trains an image classification model, and saves the model and class dictionary.

    Args:
        dataset_path (str): Path to the root dataset folder.
        img_size (tuple): Target image size (width, height).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    start_time = time.time()

    # --- 1. Load and Preprocess Data ---
    print("--- Loading and Preprocessing Data ---")
    try:
        X, y, class_dict = load_images_with_cv2(dataset_path, img_size=img_size)
        num_classes = len(class_dict)
        # Convert labels to one-hot encoding
        y_one_hot = to_categorical(y, num_classes=num_classes)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\nData shapes: X={X.shape}, y_one_hot={y_one_hot.shape}")

    # --- 2. Split Data ---
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y # Stratify helps maintain class distribution
    )
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set:     {X_test.shape[0]} images")

    # --- 3. Build Model ---
    print("\n--- Building Model ---")
    input_shape = (img_size[0], img_size[1], 3) # height, width, channels
    model = build_classification_model(input_shape, num_classes)

    # --- 4. Train Model ---
    print("\n--- Training Model ---")
    # Early stopping to prevent overfitting and save time
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        # callbacks=[early_stopping],
        verbose=1 # Show progress bar per epoch
    )

    # --- 5. Evaluate Model ---
    print("\n--- Evaluating Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # --- 6. Save Model and Class Dictionary ---
    print("\n--- Saving Model and Class Dictionary ---")
    model_save_path = "image_classification_model_cv2.h5"
    class_dict_save_path = "class_dict_cv2.pkl"

    try:
        model.save(model_save_path)
        print(f"Model saved successfully to {model_save_path}")
        with open(class_dict_save_path, "wb") as f:
            pickle.dump(class_dict, f)
        print(f"Class dictionary saved successfully to {class_dict_save_path}")
    except Exception as e:
        print(f"Error saving model or class dictionary: {e}")

    end_time = time.time()
    print(f"\n--- Training Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # --- Configuration ---
    DATASET_FOLDER = "./train"  # IMPORTANT: Change this to the path of your dataset folder
    IMAGE_SIZE = (128, 128)     # Target image size (width, height)
    EPOCHS = 50                 # Number of training epochs
    BATCH_SIZE = 32             # Batch size

    # Check if dataset folder exists
    if not os.path.isdir(DATASET_FOLDER):
        print(f"Error: Dataset folder '{DATASET_FOLDER}' not found.")
        print("Please create the folder and organize your images into class subfolders:")
        print(f"{DATASET_FOLDER}/")
        print(f"  ├── classA/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── image2.png")
        print(f"  └── classB/")
        print(f"      ├── image3.jpeg")
        print(f"      └── ...")
    else:
        train_and_save_model(
            dataset_path=DATASET_FOLDER,
            img_size=IMAGE_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )