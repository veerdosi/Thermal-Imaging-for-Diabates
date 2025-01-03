import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
import seaborn as sns
from datetime import datetime
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThermalImagePreprocessor:
    def __init__(self, base_resolution=(224, 224), temp_range=(20, 40)):
        """
        Initialize preprocessor with configuration
        Args:
            base_resolution: Target resolution for normalized images
            temp_range: Expected temperature range in Celsius
        """
        self.base_resolution = base_resolution
        self.temp_range = temp_range
        self.scaler = StandardScaler()
        
    def load_image(self, image_path):
        """Load and validate thermal image."""
        img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)  # For 16-bit thermal images
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Validate image properties
        if len(img.shape) != 2:
            raise ValueError(f"Expected grayscale image, got shape {img.shape}")
            
        return img
    
    def remove_artifacts(self, image):
        """Remove noise and artifacts."""
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(image, sigma=1)
        
        # Remove outlier pixels (very hot or cold spots)
        lower_bound = np.percentile(smoothed, 1)
        upper_bound = np.percentile(smoothed, 99)
        cleaned = np.clip(smoothed, lower_bound, upper_bound)
        
        return cleaned
    
    def normalize_temperature(self, image):
        """
        Convert raw pixel values to absolute temperatures and normalize
        """
        # Convert to temperature values (assuming linear relationship)
        temp_range = self.temp_range[1] - self.temp_range[0]
        temp_image = self.temp_range[0] + (image * temp_range / 65535.0)  # For 16-bit images
        
        # Normalize to 0-1 range
        normalized = (temp_image - self.temp_range[0]) / temp_range
        
        return normalized, temp_image
    
    def segment_foot(self, image):
        """
        Advanced foot segmentation using multiple techniques
        """
        # Otsu's thresholding
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (assumed to be foot)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No foot contour found in image")
            
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        return final_mask
    
    def align_foot(self, image, mask):
        """Align foot to standard orientation."""
        # Find foot orientation using PCA
        y, x = np.nonzero(mask)
        coords = np.column_stack([x, y])
        pca = PCA(n_components=2)
        pca.fit(coords)
        
        # Calculate rotation angle
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle = np.degrees(angle)
        
        # Rotate image and mask
        center = tuple(np.mean(coords, axis=0).astype(int))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, M, image.shape[::-1])
        aligned_mask = cv2.warpAffine(mask, M, mask.shape[::-1])
        
        return aligned_image, aligned_mask
    
    def extract_rois(self, image, mask):
        """
        Extract anatomical regions of interest
        """
        # Find foot boundaries
        y, x = np.nonzero(mask)
        min_y, max_y = np.min(y), np.max(y)
        min_x, max_x = np.min(x), np.max(x)
        
        height = max_y - min_y
        width = max_x - min_x
        
        rois = {
            'toes': image[min_y:min_y + height//3, min_x:max_x],
            'forefoot': image[min_y + height//3:min_y + height//2, min_x:max_x],
            'midfoot': image[min_y + height//2:min_y + 2*height//3, min_x:max_x],
            'heel': image[min_y + 2*height//3:max_y, min_x:max_x],
            'lateral': image[min_y:max_y, min_x:min_x + width//3],
            'medial': image[min_y:max_y, max_x - width//3:max_x]
        }
        
        roi_masks = {key: mask[roi.shape[0], roi.shape[1]] for key, roi in rois.items()}
        
        return rois, roi_masks
    
    def compute_features(self, image, rois, temp_image):
        """
        Compute comprehensive thermal features
        """
        features = {}
        
        # Global features
        features.update({
            'mean_temp': np.mean(temp_image),
            'std_temp': np.std(temp_image),
            'skewness': skew(temp_image.flatten()),
            'kurtosis': kurtosis(temp_image.flatten()),
            'max_temp': np.max(temp_image),
            'min_temp': np.min(temp_image),
            'temp_range': np.max(temp_image) - np.min(temp_image)
        })
        
        # ROI-specific features
        for roi_name, roi in rois.items():
            roi_temp = temp_image[roi]
            features.update({
                f'{roi_name}_mean': np.mean(roi_temp),
                f'{roi_name}_std': np.std(roi_temp),
                f'{roi_name}_max': np.max(roi_temp),
                f'{roi_name}_min': np.min(roi_temp),
                f'{roi_name}_skew': skew(roi_temp.flatten()),
                f'{roi_name}_kurtosis': kurtosis(roi_temp.flatten())
            })
            
        # Temperature gradient features
        for roi_name in ['toes', 'forefoot', 'midfoot', 'heel']:
            roi_temp = temp_image[rois[roi_name]]
            gradient_y, gradient_x = np.gradient(roi_temp)
            features.update({
                f'{roi_name}_gradient_mean_x': np.mean(gradient_x),
                f'{roi_name}_gradient_mean_y': np.mean(gradient_y),
                f'{roi_name}_gradient_std_x': np.std(gradient_x),
                f'{roi_name}_gradient_std_y': np.std(gradient_y)
            })
        
        return features

class ThermalImageDataset:
    def __init__(self, base_path, preprocessing_config=None):
        self.base_path = Path(base_path)
        self.config = preprocessing_config or {}
        self.preprocessor = ThermalImagePreprocessor(**self.config)
        
    def load_dataset(self):
        """Load and preprocess complete dataset."""
        data = []
        labels = []
        file_paths = []
        failed_images = []
        
        logger.info(f"Loading dataset from {self.base_path}")
        
        for group in ['control', 'dm']:
            group_path = self.base_path / group
            if not group_path.exists():
                raise ValueError(f"Required folder '{group}' not found in {self.base_path}")
                
            logger.info(f"Processing {group} group...")
            
            for img_path in group_path.glob('*.tiff'):  # Adjust extension as needed
                try:
                    # Load and preprocess image
                    img = self.preprocessor.load_image(str(img_path))
                    cleaned_img = self.preprocessor.remove_artifacts(img)
                    normalized_img, temp_img = self.preprocessor.normalize_temperature(cleaned_img)
                    
                    # Segment and align foot
                    mask = self.preprocessor.segment_foot(normalized_img)
                    aligned_img, aligned_mask = self.preprocessor.align_foot(normalized_img, mask)
                    
                    # Extract ROIs and features
                    rois, roi_masks = self.preprocessor.extract_rois(aligned_img, aligned_mask)
                    features = self.preprocessor.compute_features(aligned_img, rois, temp_img)
                    
                    # Store results
                    data.append(features)
                    labels.append(1 if group == 'dm' else 0)
                    file_paths.append(str(img_path))
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
                    failed_images.append((str(img_path), str(e)))
        
        # Create DataFrame with features
        df = pd.DataFrame(data)
        
        # Add metadata
        df['file_path'] = file_paths
        df['label'] = labels
        
        # Log processing summary
        logger.info(f"Successfully processed {len(data)} images")
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")
            
        return df

class ThermalImageAnalyzer:
    def __init__(self, model_config=None):
        self.config = model_config or {}
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.cnn_model = None
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.scaler = StandardScaler()
        
    def build_cnn(self, input_shape):
        """Build CNN architecture."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.cnn_model = model
        return model
    
    def prepare_data(self, X, y=None):
        """Scale features and prepare data for modeling."""
        if isinstance(X, pd.DataFrame):
            # Separate features from metadata
            feature_cols = X.columns[~X.columns.isin(['file_path', 'label'])]
            X = X[feature_cols]
        
        # Scale features
        if y is None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
            
        return X_scaled
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train both RF and CNN models with cross-validation."""
        # Prepare data
        X_train_scaled = self.prepare_data(X_train, y_train)
        if X_val is not None:
            X_val_scaled = self.prepare_data(X_val)
        
        # Train Random Forest with cross-validation
        cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=5)
        logger.info(f"RF Cross-validation scores: {cv_scores}")
        logger.info(f"RF Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Final RF training
        self.rf_model.fit(X_train_scaled, y_train)
        
        if X_val is not None:
            rf_val_score = self.rf_model.score(X_val_scaled, y_val)
            logger.info(f"RF Validation Score: {rf_val_score:.3f}")
        
        # Train CNN if images are provided
        if self.cnn_model is not None and hasattr(X_train, 'shape') and len(X_train.shape) == 4:
            history = self.cnn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=20,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            return history
    
    def perform_clustering(self, X):
        """Perform unsupervised clustering analysis."""
        # Prepare data
        X_scaled = self.prepare_data(X)
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=42)
        
        X_pca = pca.fit_transform(X_scaled)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Perform clustering
        kmeans_clusters = self.kmeans.fit_predict(X_scaled)
        dbscan_clusters = self.dbscan.fit_predict(X_scaled)
        
        clustering_results = {
            'pca_coords': X_pca,
            'tsne_coords': X_tsne,
            'kmeans_labels': kmeans_clusters,
            'dbscan_labels': dbscan_clusters
        }
        
        return clustering_results
    
    def evaluate_model(self, X_test, y_test, output_dir=None):
        """Comprehensive model evaluation."""
        # Prepare data
        X_test_scaled = self.prepare_data(X_test)
        
        # Random Forest evaluation
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_pred_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        rf_report = classification_report(y_test, rf_pred, output_dict=True)
        rf_cm = confusion_matrix(y_test, rf_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, rf_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        evaluation_results = {
            'classification_report': rf_report,
            'confusion_matrix': rf_cm.tolist(),
            'roc_auc': roc_auc,
            'feature_importance': dict(zip(X_test.columns, self.rf_model.feature_importances_))
        }
        
        # Generate visualizations if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(output_dir / 'roc_curve.png')
            plt.close()
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            feature_importance = pd.Series(self.rf_model.feature_importances_, index=X_test.columns)
            feature_importance.sort_values(ascending=True).tail(20).plot(kind='barh')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png')
            plt.close()
            
            # Save evaluation results
            with open(output_dir / 'evaluation_results.json', 'w') as f:
                json.dump(evaluation_results, f, indent=4)
        
        return evaluation_results
    
    def save_models(self, output_dir):
        """Save trained models and scalers."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Random Forest model
        import joblib
        joblib.dump(self.rf_model, output_dir / 'random_forest_model.joblib')
        joblib.dump(self.scaler, output_dir / 'scaler.joblib')
        
        # Save CNN model if exists
        if self.cnn_model is not None:
            self.cnn_model.save(output_dir / 'cnn_model')
        
        # Save configuration
        with open(output_dir / 'model_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
            
def main():
    """Main execution script."""
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    logging.basicConfig(
        filename=output_dir / 'processing.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize dataset handlers
        logger.info("Initializing dataset handlers...")
        train_dataset = ThermalImageDataset('./train')
        val_dataset = ThermalImageDataset('./val')
        
        # Load and preprocess data
        logger.info("Loading and preprocessing training data...")
        train_df = train_dataset.load_dataset()
        logger.info("Loading and preprocessing validation data...")
        val_df = val_dataset.load_dataset()
        
        # Split features and labels
        X_train = train_df.drop(['label', 'file_path'], axis=1)
        y_train = train_df['label']
        X_val = val_df.drop(['label', 'file_path'], axis=1)
        y_val = val_df['label']
        
        # Initialize and train analyzer
        logger.info("Initializing thermal image analyzer...")
        analyzer = ThermalImageAnalyzer()
        
        # Perform clustering analysis
        logger.info("Performing clustering analysis...")
        clustering_results = analyzer.perform_clustering(X_train)
        
        # Train models
        logger.info("Training models...")
        analyzer.train_models(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        logger.info("Evaluating models...")
        evaluation_results = analyzer.evaluate_model(X_val, y_val, output_dir)
        
        # Save models
        logger.info("Saving models...")
        analyzer.save_models(output_dir / 'models')
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()