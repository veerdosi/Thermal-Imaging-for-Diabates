import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from pathlib import Path

class ThermalImagePreprocessor:
    def __init__(self, base_resolution=(224, 224)):
        self.base_resolution = base_resolution
        self.scaler = StandardScaler()
    
    def load_image(self, image_path):
        """Load and preprocess a thermal image."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    def normalize_temperature(self, image):
        """Normalize temperature values."""
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized
    
    def segment_foot(self, image):
        """Simple foot segmentation using Otsu's method."""
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    def extract_rois(self, image, mask):
        """Extract regions of interest."""
        # Assuming foot is properly aligned
        h, w = image.shape
        rois = {
            'toes': image[0:h//3, :],
            'midfoot': image[h//3:2*h//3, :],
            'heel': image[2*h//3:, :]
        }
        return rois
    
    def compute_features(self, image, rois):
        """Compute thermal features from the image and ROIs."""
        features = {}
        
        # Global features
        features['mean_temp'] = np.mean(image)
        features['std_temp'] = np.std(image)
        features['skewness'] = skew(image.flatten())
        features['kurtosis'] = kurtosis(image.flatten())
        
        # ROI-specific features
        for roi_name, roi in rois.items():
            features[f'{roi_name}_mean'] = np.mean(roi)
            features[f'{roi_name}_std'] = np.std(roi)
            features[f'{roi_name}_max'] = np.max(roi)
            features[f'{roi_name}_min'] = np.min(roi)
        
        return features

class ThermalImageDataset:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.preprocessor = ThermalImagePreprocessor()
        
    def load_dataset(self):
        """Load and preprocess all images from the dataset."""
        data = []
        labels = []
        
        # Verify the expected folders exist
        for group in ['control', 'dm']:
            group_path = self.base_path / group
            if not group_path.exists():
                raise ValueError(f"Required folder '{group}' not found in {self.base_path}")
            print(f"Found {group} folder at {group_path}")
            
        # Process images from each group
        for group in ['control', 'dm']:
            group_path = self.base_path / group
            print(f"Processing images from {group} group...")
                
            for img_path in group_path.glob('*.jpg'):  # Adjust extension as needed
                try:
                    # Load and preprocess image
                    img = self.preprocessor.load_image(str(img_path))
                    normalized_img = self.preprocessor.normalize_temperature(img)
                    mask = self.preprocessor.segment_foot(normalized_img)
                    rois = self.preprocessor.extract_rois(normalized_img, mask)
                    features = self.preprocessor.compute_features(normalized_img, rois)
                    
                    # Store features and label
                    data.append(features)
                    labels.append(1 if group == 'dm' else 0)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return pd.DataFrame(data), np.array(labels)

class ThermalImageAnalyzer:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.cnn_model = None
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
    
    def build_cnn(self, input_shape):
        """Build CNN architecture."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        self.cnn_model = model
        return model
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train both RF and CNN models."""
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_val, y_val)
        print(f"Random Forest Validation Score: {rf_score}")
        
        # For CNN, we need to reshape data appropriately
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            X_val = X_val.values
        
        # Train CNN (assuming data is properly shaped)
        if self.cnn_model is None:
            self.build_cnn((224, 224, 1))  # Adjust shape as needed
        
        # Reshape data for CNN if needed
        # X_train_cnn = X_train.reshape(-1, 224, 224, 1)
        # X_val_cnn = X_val.reshape(-1, 224, 224, 1)
        
        # history = self.cnn_model.fit(
        #     X_train_cnn, y_train,
        #     validation_data=(X_val_cnn, y_val),
        #     epochs=10,
        #     batch_size=32
        # )
        
    def perform_clustering(self, X):
        """Perform unsupervised clustering."""
        kmeans_clusters = self.kmeans.fit_predict(X)
        dbscan_clusters = self.dbscan.fit_predict(X)
        return kmeans_clusters, dbscan_clusters
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        rf_pred = self.rf_model.predict(X_test)
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred))
        
        # CNN evaluation if needed
        # if self.cnn_model is not None:
        #     X_test_cnn = X_test.reshape(-1, 224, 224, 1)
        #     cnn_pred = (self.cnn_model.predict(X_test_cnn) > 0.5).astype(int)
        #     print("\nCNN Classification Report:")
        #     print(classification_report(y_test, cnn_pred))

def main():
    # Initialize dataset handlers with your actual paths
    train_dataset = ThermalImageDataset('./train')
    val_dataset = ThermalImageDataset('./val')
    
    print("\nProcessing Training Dataset:")
    print("-" * 50)
    
    # Load and preprocess data
    X_train, y_train = train_dataset.load_dataset()
    X_val, y_val = val_dataset.load_dataset()
    
    # Initialize analyzer
    analyzer = ThermalImageAnalyzer()
    
    # Train models
    analyzer.train_models(X_train, y_train, X_val, y_val)
    
    # Perform clustering
    kmeans_clusters, dbscan_clusters = analyzer.perform_clustering(X_train)
    
    # Evaluate on validation set
    analyzer.evaluate_model(X_val, y_val)
    
    # Optional: Save models
    # joblib.dump(analyzer.rf_model, 'rf_model.joblib')
    # analyzer.cnn_model.save('cnn_model.h5')

if __name__ == "__main__":
    main()
