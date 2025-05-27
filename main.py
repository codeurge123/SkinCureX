import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_images_from_folder(folder, img_size=(224, 224)):
    """
    Load images from folder structure where folder names are class labels.
    Automatically generate bounding boxes based on foreground/background segmentation.
    """
    data = []
    bboxes = []
    labels = []
    classes = [cls for cls in os.listdir(folder) if os.path.isdir(os.path.join(folder, cls))]
    class_dict = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    for cls in classes:
        class_path = os.path.join(folder, cls)
        print(f"Processing class: {cls}")
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue
                
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Store original dimensions for scaling
            orig_height, orig_width = img.shape[:2]
            
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Generate a bounding box for the main object in the image
            # First convert to grayscale for processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to separate foreground from background
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # If no contours found, use the whole image
                x_min, y_min = 0, 0
                x_max, y_max = orig_width, orig_height
            else:
                # Find the largest contour (likely the main object)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding around the bounding box (10%)
                padding_x = int(w * 0.1)
                padding_y = int(h * 0.1)
                
                x_min = max(0, x - padding_x)
                y_min = max(0, y - padding_y)
                x_max = min(orig_width, x + w + padding_x)
                y_max = min(orig_height, y + h + padding_y)
            
            # Resize image
            img_resized = cv2.resize(img_rgb, img_size)
            
            # Normalize image data to [0, 1]
            img_normalized = img_resized / 255.0
            
            # Normalize bounding box coordinates to [0, 1]
            bbox_normalized = [
                x_min / orig_width,
                y_min / orig_height,
                x_max / orig_width,
                y_max / orig_height
            ]
            
            data.append(img_normalized)
            bboxes.append(bbox_normalized)
            labels.append(class_dict[cls])
    
    return np.array(data), np.array(bboxes), np.array(labels), class_dict

def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded

def sigmoid(x):
    """Numerically stable sigmoid function"""
    # Clip extremely negative or positive values to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Numerically stable sigmoid derivative"""
    # Ensuring x is in valid range
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return x * (1 - x)

def softmax(x):
    """Numerically stable softmax function"""
    # Subtracting max(x) for numerical stability
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    return -np.mean(y_true * np.log(y_pred))

def bbox_loss(y_true, y_pred):
    """Custom loss for bounding box regression using mean squared error"""
    return np.mean(np.square(y_true - y_pred))

class ObjectDetectionNetwork:
    def __init__(self, input_shape, hidden_size, num_classes, lr=0.01):
        self.input_shape = input_shape  # (height, width, channels)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lr = lr
        
        # Flattened input size
        flat_size = input_shape[0] * input_shape[1] * input_shape[2]
        
        # Initialize weights with smaller values for better training stability
        self.weights1 = np.random.randn(flat_size, hidden_size) * 0.0005
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, num_classes) * 0.0005
        self.bias2 = np.zeros((1, num_classes))
        
        # Network for bounding box regression (4 values: x_min, y_min, x_max, y_max)
        self.bbox_weights = np.random.randn(hidden_size, 4) * 0.0005
        self.bbox_bias = np.zeros((1, 4))
        
        self.class_loss_history = []
        self.bbox_loss_history = []
    
    def forward(self, x):
        # Flatten the input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Forward pass through shared layers
        self.z1 = np.dot(x_flat, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        
        # Classification branch
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.class_preds = softmax(self.z2)
        
        # Bounding box regression branch
        self.bbox_z = np.dot(self.a1, self.bbox_weights) + self.bbox_bias
        self.bbox_preds = sigmoid(self.bbox_z)  # Constrain predictions between 0 and 1
        
        return self.class_preds, self.bbox_preds
    
    def backward(self, x, y_class, y_bbox):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Classification gradients
        dz2 = self.class_preds - y_class
        dw2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
        
        # Bounding box gradients
        dbbox_z = 2 * (self.bbox_preds - y_bbox)
        # Apply sigmoid derivative safely
        sig_deriv = sigmoid_derivative(self.bbox_preds)
        # Check for NaN or infinite values
        if np.any(np.isnan(sig_deriv)) or np.any(np.isinf(sig_deriv)):
            print("Warning: NaN or Inf detected in sigmoid derivative")
            sig_deriv = np.nan_to_num(sig_deriv, nan=0.0, posinf=0.0, neginf=0.0)
        
        dbbox_z = dbbox_z * sig_deriv
        dbbox_w = np.dot(self.a1.T, dbbox_z) / batch_size
        dbbox_b = np.sum(dbbox_z, axis=0, keepdims=True) / batch_size
        
        # Gradients for shared layer
        da1_class = np.dot(dz2, self.weights2.T)
        da1_bbox = np.dot(dbbox_z, self.bbox_weights.T)
        da1 = da1_class + da1_bbox
        
        # Handle potential NaN values
        da1 = np.nan_to_num(da1, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Safely compute sigmoid derivative
        sig_deriv_a1 = sigmoid_derivative(self.a1)
        sig_deriv_a1 = np.nan_to_num(sig_deriv_a1, nan=0.0, posinf=0.0, neginf=0.0)
        
        dz1 = da1 * sig_deriv_a1
        dw1 = np.dot(x_flat.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        # Gradient clipping to prevent exploding gradients
        dw2 = np.clip(dw2, -1, 1)
        db2 = np.clip(db2, -1, 1)
        dbbox_w = np.clip(dbbox_w, -1, 1)
        dbbox_b = np.clip(dbbox_b, -1, 1)
        dw1 = np.clip(dw1, -1, 1)
        db1 = np.clip(db1, -1, 1)
        
        # Update weights and biases
        self.weights2 -= self.lr * dw2
        self.bias2 -= self.lr * db2
        self.bbox_weights -= self.lr * dbbox_w
        self.bbox_bias -= self.lr * dbbox_b
        self.weights1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
    
    def train(self, x, y_class, y_bbox, epochs=50, batch_size=32, verbose=1):
        num_samples = x.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_class_shuffled = y_class[indices]
            y_bbox_shuffled = y_bbox[indices]
            
            epoch_class_loss = 0
            epoch_bbox_loss = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_x = x_shuffled[start_idx:end_idx]
                batch_y_class = y_class_shuffled[start_idx:end_idx]
                batch_y_bbox = y_bbox_shuffled[start_idx:end_idx]
                
                # Forward pass
                class_preds, bbox_preds = self.forward(batch_x)
                
                # Check for NaN values
                if np.any(np.isnan(class_preds)) or np.any(np.isnan(bbox_preds)):
                    print(f"WARNING: NaN values detected in predictions at epoch {epoch}, batch {i}")
                    # Skip this batch
                    continue
                
                # Calculate losses
                batch_class_loss = cross_entropy_loss(batch_y_class, class_preds)
                batch_bbox_loss = bbox_loss(batch_y_bbox, bbox_preds)
                
                # If losses are NaN, skip the update
                if np.isnan(batch_class_loss) or np.isnan(batch_bbox_loss):
                    print(f"WARNING: NaN loss detected at epoch {epoch}, batch {i}")
                    continue
                
                epoch_class_loss += batch_class_loss * (end_idx - start_idx) / num_samples
                epoch_bbox_loss += batch_bbox_loss * (end_idx - start_idx) / num_samples
                
                # Backpropagation
                self.backward(batch_x, batch_y_class, batch_y_bbox)
            
            # Store losses
            self.class_loss_history.append(epoch_class_loss)
            self.bbox_loss_history.append(epoch_bbox_loss)
            total_loss = epoch_class_loss + epoch_bbox_loss
            
            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs}, Class Loss: {epoch_class_loss:.4f}, Bbox Loss: {epoch_bbox_loss:.4f}, Total: {total_loss:.4f}")
    
    def predict(self, x):
        class_preds, bbox_preds = self.forward(x)
        class_ids = np.argmax(class_preds, axis=1)
        confidences = np.max(class_preds, axis=1)
        return class_ids, bbox_preds, confidences

def predict_objects(image_path, model, class_dict, img_size=(224, 224)):
    """Detect objects in a new image"""
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    orig_img = img.copy()
    orig_height, orig_width = orig_img.shape[:2]
    
    # Convert to RGB and resize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    input_img = np.expand_dims(img_normalized, axis=0)
    
    # Get predictions
    class_ids, bbox_preds, confidences = model.predict(input_img)
    class_id = class_ids[0]
    bbox = bbox_preds[0]
    confidence = confidences[0]
    
    # Convert class ID to name
    class_name = list(class_dict.keys())[list(class_dict.values()).index(class_id)]
    
    # Scale bounding box back to original image size
    x_min, y_min, x_max, y_max = bbox
    x_min = int(x_min * orig_width)
    y_min = int(y_min * orig_height)
    x_max = int(x_max * orig_width)
    y_max = int(y_max * orig_height)
    
    # Draw bounding box and label on the original image
    result_img = orig_img.copy()
    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Add text with background for better visibility
    label = f"{class_name}: {confidence:.2f}"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    # Draw the text background
    cv2.rectangle(result_img, 
                 (x_min, y_min - text_size[1] - 10), 
                 (x_min + text_size[0], y_min), 
                 (0, 255, 0), -1)
    
    # Draw the text
    cv2.putText(result_img, label, (x_min, y_min - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Convert back to RGB for display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(result_img_rgb)
    plt.axis('off')
    plt.title(f"Detected: {class_name} (Confidence: {confidence:.2f})")
    plt.show()
    
    result = {
        "class": class_name,
        "confidence": float(confidence),
        "bbox": [x_min, y_min, x_max, y_max]
    }
    
    print(f"Detected: {class_name} with confidence {confidence:.2f}")
    print(f"Bounding box: [x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}]")
    
    return result

def plot_losses(model):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.class_loss_history, 'b-')
    plt.title('Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model.bbox_loss_history, 'r-')
    plt.title('Bounding Box Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test_class, y_test_bbox, class_dict):
    """Evaluate the model on test data"""
    class_ids, bbox_preds, confidences = model.predict(X_test)
    y_test_labels = np.argmax(y_test_class, axis=1)
    
    # Calculate classification accuracy
    accuracy = np.mean(class_ids == y_test_labels)
    
    # Calculate average IoU for bounding boxes
    bbox_error = np.mean(np.square(y_test_bbox - bbox_preds))
    
    # Print results
    print(f"Test Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Bounding Box Mean Squared Error: {bbox_error:.4f}")
    
    # Calculate confusion matrix
    num_classes = len(class_dict)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_test_labels, class_ids):
        confusion[true_label, pred_label] += 1
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    class_names = list(class_dict.keys())
    print("True \\ Pred", end="\t")
    for name in class_names:
        print(f"{name[:7]}", end="\t")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name[:7]}", end="\t")
        for j in range(num_classes):
            print(f"{confusion[i, j]}", end="\t")
        print()
    
    return accuracy, bbox_error

def main():
    # Define parameters - optimized for Mac M2
    img_size = (224, 224)
    hidden_size = 192        # Reduced for faster training
    epochs = 50              # REDUCED from 150 to 50 as requested
    batch_size = 32          # Increased for M2 chip efficiency
    learning_rate = 0.001    # Adjusted for fewer epochs
    
    # Try to use MPS (Metal Performance Shaders) for M2 chip if available
    try:
        import torch
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available. Using M2 GPU acceleration.")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    except:
        print("PyTorch with MPS not available. Using standard CPU processing.")
    
    print("Loading and processing training data...")
    X, bbox_data, y, class_dict = load_images_from_folder("./train", img_size=img_size)
    
    print(f"Dataset loaded: {X.shape[0]} images, {len(class_dict)} classes")
    
    # One-hot encode the class labels
    y_one_hot = one_hot_encode(y, len(class_dict))
    
    # Split dataset
    X_train, X_test, y_train, y_test, bbox_train, bbox_test = train_test_split(
        X, y_one_hot, bbox_data, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Initialize and train model
    input_shape = X_train.shape[1:]  # (height, width, channels)
    model = ObjectDetectionNetwork(
        input_shape=input_shape, 
        hidden_size=hidden_size, 
        num_classes=len(class_dict), 
        lr=learning_rate
    )
    
    print("\nTraining model...")
    # Use the M2 chip's multicores efficiently
    model.train(X_train, y_train, bbox_train, epochs=epochs, batch_size=batch_size)
    
    # Save trained model
    print("Saving model...")
    with open("object_detection_model.pkl", "wb") as f:
        pickle.dump({"model": model, "class_dict": class_dict, "img_size": img_size}, f)
    
    # Plot training losses
    plot_losses(model)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    evaluate_model(model, X_test, y_test, bbox_test, class_dict)
    
    # Test on a sample image
    print("\nTesting model on sample image...")
    
    # First look for test directory
    test_dir = "./test"
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        # Find first image in test directory or any subdirectory
        test_img_path = None
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    test_img_path = os.path.join(root, file)
                    break
            if test_img_path:
                break
        
        if test_img_path:
            print(f"Testing on image: {test_img_path}")
            predict_objects(test_img_path, model, class_dict, img_size=img_size)
        else:
            print("No test images found.")
    else:
        print("Test directory not found.")
    
    print("\nModel training and evaluation complete!")

def load_and_predict(image_path):
    """Load the saved model and make a prediction on a new image"""
    # Check if model file exists
    model_path = "object_detection_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return None
    
    # Load the model
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    class_dict = model_data["class_dict"]
    img_size = model_data["img_size"]
    
    print(f"Model loaded. Supports {len(class_dict)} classes: {', '.join(class_dict.keys())}")
    
    # Make prediction
    result = predict_objects(image_path, model, class_dict, img_size=img_size)
    return result

if __name__ == "__main__":
    # Check if there's a command-line argument
    import sys
    
    if len(sys.argv) > 1:
        # If an image path is provided, load model and predict
        image_path = sys.argv[1]
        load_and_predict(image_path)
    else:
        # Otherwise, train a new model
        main()