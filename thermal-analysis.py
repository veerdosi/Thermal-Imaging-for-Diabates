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
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Changed from IMREAD_ANYDEPTH
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
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
        """Convert PNG pixel values to temperature and normalize"""
        # For 8-bit PNG images
        max_val = 255
        temp_range = self.temp_range[1] - self.temp_range[0]
        temp_image = self.temp_range[0] + (image.astype(float) * temp_range / max_val)
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
        
        roi_masks = {key: mask[min_y:min_y+roi.shape[0], min_x:min_x+roi.shape[1]] for key, roi in rois.items()}
        
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

    def _calculate_asymmetry_stats(self, X):
        """Calculate temperature asymmetry statistics"""
        asymmetry_features = {
            'toes': abs(X['lateral_mean'] - X['medial_mean']),
            'forefoot': abs(X['forefoot_mean'] - X['heel_mean']),
            'midfoot': abs(X['midfoot_gradient_mean_x'] - X['midfoot_gradient_mean_y'])
        }
        return pd.DataFrame(asymmetry_features)

    def _get_risk_factors(self, X, predictions):
        """Compile risk factors with their importance"""
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def _generate_visualizations(self, results, output_dir):
        """Generate and save visualization plots"""
        output_dir = Path(output_dir)
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['Low', 'Moderate', 'High'],
                    yticklabels=['Low', 'Moderate', 'High'])
        plt.title('Risk Level Confusion Matrix')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
        
        # Feature importance
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': list(results['feature_importance'].keys()),
            'importance': list(results['feature_importance'].values())
        })
        importance_df.sort_values('importance', ascending=True).tail(20).plot(kind='barh')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png')
        plt.close()
        
    def build_cnn(self, input_shape):
        """Build CNN architecture for multi-class classification."""
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
            layers.Dense(3, activation='softmax')  # 3 classes: low, moderate, high risk
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
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
        """Comprehensive model evaluation for multi-class classification."""
        # Prepare data
        X_test_scaled = self.prepare_data(X_test)
        
        # Random Forest evaluation
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_pred_proba = self.rf_model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        rf_report = classification_report(y_test, rf_pred, output_dict=True)
        rf_cm = confusion_matrix(y_test, rf_pred)
        
        # Store results
        evaluation_results = {
            'classification_report': rf_report,
            'confusion_matrix': rf_cm.tolist(),
            'feature_importance': dict(zip(X_test.columns, self.rf_model.feature_importances_)),
            'risk_distribution': np.bincount(rf_pred, minlength=3),
            'temperature_asymmetry': self._calculate_asymmetry_stats(X_test),
            'risk_factors': self._get_risk_factors(X_test, rf_pred)
        }
        
        # Generate visualizations if output directory is provided
        if output_dir:
            self._generate_visualizations(evaluation_results, output_dir)

        # CNN evaluation if model exists
        if self.cnn_model is not None:
            cnn_pred = np.argmax(self.cnn_model.predict(X_test), axis=1)
            cnn_report = classification_report(y_test, cnn_pred, output_dict=True)
            evaluation_results['cnn_report'] = cnn_report
        
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

class RiskAssessmentCriteria:
    """Sophisticated risk assessment criteria for diabetic foot complications."""
    
    def __init__(self):
        # Temperature thresholds (in Celsius)
        self.temp_thresholds = {
            'critical_high': 35.0,
            'high': 33.0,
            'moderate': 31.0,
            'low': 29.0
        }
        
        # Asymmetry thresholds (in Celsius)
        self.asymmetry_thresholds = {
            'severe': 2.2,  # > 2.2°C difference indicates high risk
            'moderate': 1.5,  # 1.5-2.2°C difference indicates moderate risk
            'mild': 0.8     # < 0.8°C difference indicates low risk
        }
        
        # Gradient thresholds
        self.gradient_thresholds = {
            'severe': 1.5,   # °C/cm
            'moderate': 1.0,
            'mild': 0.5
        }
        
        # Pattern detection weights
        self.pattern_weights = {
            'temperature': 0.4,
            'asymmetry': 0.3,
            'gradient': 0.2,
            'pattern': 0.1
        }
    
    def calculate_regional_risk(self, region_features):
        """Calculate risk score for a specific foot region."""
        risk_score = 0
        
        # Temperature assessment
        mean_temp = region_features['mean']
        if mean_temp > self.temp_thresholds['critical_high']:
            risk_score += 3 * self.pattern_weights['temperature']
        elif mean_temp > self.temp_thresholds['high']:
            risk_score += 2 * self.pattern_weights['temperature']
        elif mean_temp > self.temp_thresholds['moderate']:
            risk_score += 1 * self.pattern_weights['temperature']
            
        # Temperature variation assessment
        temp_std = region_features['std']
        if temp_std > 1.5:
            risk_score += 2 * self.pattern_weights['pattern']
        elif temp_std > 1.0:
            risk_score += 1 * self.pattern_weights['pattern']
            
        return risk_score
    
    def calculate_asymmetry_risk(self, left_features, right_features):
        """Calculate risk based on left-right asymmetry."""
        asymmetry_score = 0
        
        # Compare mean temperatures
        temp_diff = abs(left_features['mean'] - right_features['mean'])
        if temp_diff > self.asymmetry_thresholds['severe']:
            asymmetry_score += 3
        elif temp_diff > self.asymmetry_thresholds['moderate']:
            asymmetry_score += 2
        elif temp_diff > self.asymmetry_thresholds['mild']:
            asymmetry_score += 1
            
        return asymmetry_score * self.pattern_weights['asymmetry']
    
    def calculate_gradient_risk(self, gradient_features):
        """Calculate risk based on temperature gradients."""
        gradient_score = 0
        
        max_gradient = max(abs(gradient_features['gradient_mean_x']), 
                         abs(gradient_features['gradient_mean_y']))
        
        if max_gradient > self.gradient_thresholds['severe']:
            gradient_score += 3
        elif max_gradient > self.gradient_thresholds['moderate']:
            gradient_score += 2
        elif max_gradient > self.gradient_thresholds['mild']:
            gradient_score += 1
            
        return gradient_score * self.pattern_weights['gradient']

class ThermalImageDataset:
    def __init__(self, base_path, preprocessing_config=None):
        self.base_path = Path(base_path)
        self.config = preprocessing_config or {}
        self.preprocessor = ThermalImagePreprocessor(**self.config)
        self.risk_assessor = RiskAssessmentCriteria()
        
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
            
            for img_path in group_path.glob('*.png'):
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
                    
                    # Determine risk level using the risk assessor
                    risk_level = self.determine_risk_level(features)
                    
                    # Store results
                    data.append(features)
                    labels.append(risk_level)  # Using risk level instead of binary classification
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
    
    def determine_risk_level(self, features):
        """
        Determine risk level based on extracted features.
        Returns: 0 (low), 1 (moderate), or 2 (high) risk
        """
        risk_score = 0
        
        # Temperature assessment
        if features['mean_temp'] > self.risk_assessor.temp_thresholds['critical_high']:
            risk_score += 3
        elif features['mean_temp'] > self.risk_assessor.temp_thresholds['high']:
            risk_score += 2
        
        # Asymmetry assessment
        for region in ['toes', 'forefoot', 'midfoot', 'heel']:
            grad_x = features[f'{region}_gradient_mean_x']
            grad_y = features[f'{region}_gradient_mean_y']
            if abs(grad_x - grad_y) > self.risk_assessor.gradient_thresholds['severe']:
                risk_score += 2
        
        # Convert score to risk level
        if risk_score >= 5:
            return 2  # High risk
        elif risk_score >= 3:
            return 1  # Moderate risk
        return 0     # Low risk

class RiskVisualization:
    """Tools for visualizing risk assessment results."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_risk_distribution(self, labels, title='Risk Level Distribution'):
        """Plot distribution of risk levels."""
        plt.figure(figsize=(10, 6))
        risk_counts = pd.Series(labels).value_counts().sort_index()
        
        sns.barplot(x=['Low Risk', 'Moderate Risk', 'High Risk'],
                   y=risk_counts.values)
        plt.title(title)
        plt.ylabel('Number of Cases')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_distribution.png')
        plt.close()
        
    def plot_temperature_heatmap(self, temp_data, risk_level, file_path):
        """Create heatmap visualization of foot temperature."""
        plt.figure(figsize=(8, 12))
        sns.heatmap(temp_data, cmap='RdYlBu_r')
        plt.title(f'Temperature Distribution (Risk Level: {risk_level})')
        save_path = self.output_dir / f'heatmap_{Path(file_path).stem}.png'
        plt.savefig(save_path)
        plt.close()
        
    def plot_risk_factors(self, feature_importances, n_top=20):
        """Plot top risk factors."""
        plt.figure(figsize=(12, 8))
        importance_df = pd.DataFrame({
            'feature': feature_importances.keys(),
            'importance': feature_importances.values()
        })
        importance_df = importance_df.sort_values('importance', ascending=True).tail(n_top)
        
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title('Top Risk Factors')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_factors.png')
        plt.close()
        
    def create_longitudinal_plot(self, patient_data, risk_scores):
        """Create longitudinal visualization of risk progression."""
        plt.figure(figsize=(12, 6))
        timestamps = pd.to_datetime(patient_data['timestamp'])
        plt.plot(timestamps, risk_scores, marker='o')
        plt.title('Risk Score Progression Over Time')
        plt.xlabel('Date')
        plt.ylabel('Risk Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_progression.png')
        plt.close()
        
    def create_summary_dashboard(self, results_dict):
        """Create a comprehensive risk assessment dashboard."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Risk distribution
        ax1 = fig.add_subplot(gs[0, 0])
        risk_dist = results_dict['risk_distribution']
        sns.barplot(x=['Low', 'Moderate', 'High'], y=risk_dist, ax=ax1)
        ax1.set_title('Risk Level Distribution')
        
        # Temperature asymmetry
        ax2 = fig.add_subplot(gs[0, 1])
        asymmetry_data = results_dict['temperature_asymmetry']
        sns.boxplot(data=asymmetry_data, ax=ax2)
        ax2.set_title('Temperature Asymmetry by Region')
        
        # Risk factors
        ax3 = fig.add_subplot(gs[1, :])
        risk_factors = results_dict['risk_factors']
        sns.barplot(data=risk_factors, x='importance', y='feature', ax=ax3)
        ax3.set_title('Key Risk Factors')

        if not all(key in results_dict for key in ['risk_distribution', 'temperature_asymmetry', 'risk_factors']):
            raise ValueError("Missing required keys in results dictionary")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_dashboard.png')
        plt.close()

def main():
    """Main execution script."""
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualization tools
    visualizer = RiskVisualization(output_dir / 'visualizations')
    
    try:
        # Load and process data
        train_dataset = ThermalImageDataset('./train')
        val_dataset = ThermalImageDataset('./val')
        
        train_df = train_dataset.load_dataset()
        val_df = val_dataset.load_dataset()
        
        # Create visualizations
        visualizer.plot_risk_distribution(train_df['label'])
        
        # Process each image
        for idx, row in train_df.iterrows():
            temp_data = cv2.imread(row['file_path'], cv2.IMREAD_GRAYSCALE)
            visualizer.plot_temperature_heatmap(temp_data, row['label'], row['file_path'])
        
        # Initialize and train analyzer
        analyzer = ThermalImageAnalyzer()
        analyzer.train_models(train_df.drop(['label', 'file_path'], axis=1), train_df['label'],
                            val_df.drop(['label', 'file_path'], axis=1), val_df['label'])
        
        # Evaluate model
        results = analyzer.evaluate_model(val_df.drop(['label', 'file_path'], axis=1), 
                                       val_df['label'], output_dir)
        
        # Create summary dashboard
        visualizer.create_summary_dashboard(results)
        
        logger.info("Analysis and visualization completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()