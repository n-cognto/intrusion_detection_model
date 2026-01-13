"""
IoT Intrusion Detection System
==============================
Machine Learning-based Intrusion Detection for IoT Networks

This system uses three classification algorithms:
1. Decision Tree Classifier
2. AdaBoost (Adaptive Boosting) Classifier
3. Support Vector Machine (SVM) Classifier

Dataset: IoT Network Intrusion Dataset from Kaggle
Features: 46 network flow features
Labels: 34 attack types + Benign Traffic
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils import class_weight
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class IoTIntrusionDetector:
    """
    IoT Intrusion Detection System using Machine Learning
    
    This class implements a complete pipeline for detecting network intrusions
    in IoT environments using Decision Tree, AdaBoost, and SVM classifiers.
    """
    
    def __init__(self, data_path='IoT_Intrusion.csv'):
        """
        Initialize the IoT Intrusion Detector
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV dataset file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = None
        
        # Create output directory for models and results
        self.output_dir = 'model_outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 70)
        print("IoT INTRUSION DETECTION SYSTEM")
        print("Machine Learning-based Network Security")
        print("=" * 70)
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def load_data(self, sample_size=None):
        """
        Load and explore the dataset
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of samples to use (for faster experimentation)
        """
        print("\n" + "=" * 50)
        print("STEP 1: LOADING DATA")
        print("=" * 50)
        
        start_time = time.time()
        
        # Load the dataset
        print(f"\nLoading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Sample data if specified
        if sample_size and sample_size < len(self.df):
            print(f"\nSampling {sample_size:,} records from {len(self.df):,} total records...")
            self.df = self.df.sample(n=sample_size, random_state=RANDOM_STATE)
        
        load_time = time.time() - start_time
        
        # Display dataset information
        print(f"\n✓ Data loaded successfully in {load_time:.2f} seconds")
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display column information
        print("\n" + "-" * 40)
        print("FEATURES (First 10):")
        print("-" * 40)
        for i, col in enumerate(self.df.columns[:10]):
            print(f"  {i+1:2d}. {col}")
        print(f"  ... and {len(self.df.columns) - 10} more columns")
        
        # Display label distribution
        print("\n" + "-" * 40)
        print("ATTACK TYPE DISTRIBUTION:")
        print("-" * 40)
        label_counts = self.df['label'].value_counts()
        for label, count in label_counts.head(10).items():
            percentage = (count / len(self.df)) * 100
            print(f"  {label:35s}: {count:8,} ({percentage:5.2f}%)")
        if len(label_counts) > 10:
            print(f"  ... and {len(label_counts) - 10} more attack types")
        
        print(f"\nTotal Attack Types: {len(label_counts)}")
        benign_count = label_counts.get('BenignTraffic', 0)
        attack_count = len(self.df) - benign_count
        print(f"Benign Traffic: {benign_count:,} ({benign_count/len(self.df)*100:.2f}%)")
        print(f"Attack Traffic: {attack_count:,} ({attack_count/len(self.df)*100:.2f}%)")
        
        return self.df
    
    def preprocess_data(self, binary_classification=True):
        """
        Preprocess the data for training
        
        Parameters:
        -----------
        binary_classification : bool
            If True, convert to binary classification (Attack vs Benign)
            If False, use multi-class classification
        """
        print("\n" + "=" * 50)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 50)
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Handle missing values
        print("\n1. Handling Missing Values...")
        missing_before = df_processed.isnull().sum().sum()
        df_processed = df_processed.dropna()
        missing_after = df_processed.isnull().sum().sum()
        print(f"   Removed {missing_before - missing_after} missing values")
        
        # Handle infinite values
        print("\n2. Handling Infinite Values...")
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df_processed[col]).sum()
            if inf_count > 0:
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        print("   ✓ Infinite values handled")
        
        # Separate features and target
        print("\n3. Separating Features and Target...")
        self.X = df_processed.drop('label', axis=1)
        self.y = df_processed['label'].copy()
        self.feature_names = list(self.X.columns)
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Samples: {self.X.shape[0]:,}")
        
        # Convert to binary classification if specified
        if binary_classification:
            print("\n4. Converting to Binary Classification...")
            self.y = self.y.apply(lambda x: 'Benign' if x == 'BenignTraffic' else 'Attack')
            print(f"   Classes: {self.y.unique()}")
            print(f"   Benign: {(self.y == 'Benign').sum():,}")
            print(f"   Attack: {(self.y == 'Attack').sum():,}")
        else:
            print("\n4. Using Multi-Class Classification...")
            print(f"   Total Classes: {self.y.nunique()}")
        
        # Encode labels
        print("\n5. Encoding Labels...")
        self.y = self.label_encoder.fit_transform(self.y)
        print(f"   Classes encoded: {list(self.label_encoder.classes_)}")
        
        # Feature scaling
        print("\n6. Scaling Features...")
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.feature_names
        )
        print("   ✓ StandardScaler applied")
        
        # Split data
        print("\n7. Splitting Data (80% Train, 20% Test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=self.y
        )
        print(f"   Training set: {len(self.X_train):,} samples")
        print(f"   Testing set:  {len(self.X_test):,} samples")
        
        print("\n✓ Preprocessing completed successfully!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self, max_depth=20, min_samples_split=10):
        """
        Train a Decision Tree Classifier
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        """
        print("\n" + "=" * 50)
        print("TRAINING: DECISION TREE CLASSIFIER")
        print("=" * 50)
        
        print(f"\nHyperparameters:")
        print(f"  - max_depth: {max_depth}")
        print(f"  - min_samples_split: {min_samples_split}")
        print(f"  - criterion: gini")
        
        # Initialize the model
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion='gini',
            random_state=RANDOM_STATE
        )
        
        # Train the model
        print("\nTraining in progress...")
        start_time = time.time()
        dt_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        # Store the model
        self.models['Decision Tree'] = dt_model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return dt_model
    
    def train_adaboost(self, n_estimators=100, learning_rate=1.0):
        """
        Train an AdaBoost Classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of weak learners (decision stumps)
        learning_rate : float
            Learning rate shrinks the contribution of each classifier
        """
        print("\n" + "=" * 50)
        print("TRAINING: ADABOOST CLASSIFIER")
        print("=" * 50)
        
        print(f"\nHyperparameters:")
        print(f"  - n_estimators: {n_estimators}")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - algorithm: SAMME")
        
        # Base estimator (Decision Tree with limited depth)
        base_estimator = DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE
        )
        
        # Initialize AdaBoost
        adaboost_model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm='SAMME',
            random_state=RANDOM_STATE
        )
        
        # Train the model
        print("\nTraining in progress...")
        print("(This may take a few minutes...)")
        start_time = time.time()
        adaboost_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        # Store the model
        self.models['AdaBoost'] = adaboost_model
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': adaboost_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return adaboost_model
    
    def train_svm(self, kernel='rbf', C=1.0, gamma='scale', max_samples=50000):
        """
        Train a Support Vector Machine Classifier
        
        Parameters:
        -----------
        kernel : str
            Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient
        max_samples : int
            Maximum samples to use (SVM is computationally expensive)
        """
        print("\n" + "=" * 50)
        print("TRAINING: SUPPORT VECTOR MACHINE (SVM)")
        print("=" * 50)
        
        print(f"\nHyperparameters:")
        print(f"  - kernel: {kernel}")
        print(f"  - C (regularization): {C}")
        print(f"  - gamma: {gamma}")
        
        # SVM is computationally expensive, so we may need to sample
        if len(self.X_train) > max_samples:
            print(f"\n⚠ Note: Using {max_samples:,} samples for SVM training")
            print(f"  (Full training set has {len(self.X_train):,} samples)")
            
            # Sample the training data
            indices = np.random.choice(
                len(self.X_train), 
                size=max_samples, 
                replace=False
            )
            X_train_svm = self.X_train.iloc[indices]
            y_train_svm = self.y_train[indices]
        else:
            X_train_svm = self.X_train
            y_train_svm = self.y_train
        
        # Initialize SVM
        svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=RANDOM_STATE,
            probability=True,  # Enable probability estimates
            cache_size=500     # Increase cache size for faster training
        )
        
        # Train the model
        print("\nTraining in progress...")
        print("(SVM training can be slow for large datasets...)")
        start_time = time.time()
        svm_model.fit(X_train_svm, y_train_svm)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        # Store the model
        self.models['SVM'] = svm_model
        
        return svm_model
    
    def evaluate_model(self, model_name):
        """
        Evaluate a trained model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found!")
            return None
        
        print(f"\n" + "-" * 50)
        print(f"EVALUATION: {model_name.upper()}")
        print("-" * 50)
        
        model = self.models[model_name]
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(self.X_test)
        predict_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': predict_time,
            'predictions': y_pred
        }
        
        # Display metrics
        print(f"\nPerformance Metrics:")
        print(f"  • Accuracy:  {accuracy * 100:.2f}%")
        print(f"  • Precision: {precision * 100:.2f}%")
        print(f"  • Recall:    {recall * 100:.2f}%")
        print(f"  • F1-Score:  {f1 * 100:.2f}%")
        print(f"\nPrediction Time: {predict_time:.4f} seconds for {len(self.X_test):,} samples")
        
        # Classification Report
        print(f"\nClassification Report:")
        print("-" * 40)
        target_names = [str(cls) for cls in self.label_encoder.classes_]
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        return self.results[model_name]
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models and compare results
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        for model_name in self.models:
            self.evaluate_model(model_name)
        
        # Comparison summary
        if len(self.results) > 0:
            print("\n" + "=" * 60)
            print("MODEL COMPARISON SUMMARY")
            print("=" * 60)
            
            comparison_data = []
            for model_name, metrics in self.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                    'Precision': f"{metrics['precision']*100:.2f}%",
                    'Recall': f"{metrics['recall']*100:.2f}%",
                    'F1-Score': f"{metrics['f1_score']*100:.2f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print(f"\n{comparison_df.to_string(index=False)}")
            
            # Find the best model
            best_model = max(self.results, key=lambda x: self.results[x]['f1_score'])
            best_f1 = self.results[best_model]['f1_score']
            print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1*100:.2f}%)")
    
    def plot_results(self):
        """
        Generate visualization plots for the results
        """
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)
        
        if len(self.results) == 0:
            print("No results to plot. Please train and evaluate models first.")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Performance Comparison Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[m][metric] * 100 for m in models]
            bars = ax.bar(models, values, color=color, edgecolor='black', alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel(f'{label} (%)', fontsize=12)
            ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.tick_params(axis='x', rotation=15)
        
        plt.suptitle('IoT Intrusion Detection - Model Performance Comparison',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/performance_comparison.png")
        
        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, len(self.models), figsize=(6*len(self.models), 5))
        if len(self.models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(self.models):
            y_pred = self.results[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/confusion_matrices.png")
        
        # 3. Radar Chart for metrics
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the polygon
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for model_name, color in zip(models, colors):
            values = [self.results[model_name][m] * 100 for m in metrics]
            values = values + [values[0]]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metric_labels)
        ax.set_ylim(0, 100)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/radar_chart.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir}/radar_chart.png")
        
        plt.close('all')
        print("\n✓ All visualizations generated successfully!")
    
    def save_models(self):
        """
        Save trained models to disk
        """
        print("\n" + "=" * 50)
        print("SAVING MODELS")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            filename = f"{self.output_dir}/{model_name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, filename)
            print(f"✓ Saved: {filename}")
        
        # Save the scaler and label encoder
        joblib.dump(self.scaler, f"{self.output_dir}/scaler.joblib")
        joblib.dump(self.label_encoder, f"{self.output_dir}/label_encoder.joblib")
        print(f"✓ Saved: {self.output_dir}/scaler.joblib")
        print(f"✓ Saved: {self.output_dir}/label_encoder.joblib")
        
        # Save results summary
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']*100:.2f}%",
                'Recall': f"{metrics['recall']*100:.2f}%",
                'F1-Score': f"{metrics['f1_score']*100:.2f}%"
            }
            for name, metrics in self.results.items()
        ])
        results_df.to_csv(f"{self.output_dir}/results_summary.csv", index=False)
        print(f"✓ Saved: {self.output_dir}/results_summary.csv")
    
    def predict(self, new_data, model_name='Decision Tree'):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            New data to classify
        model_name : str
            Name of the model to use for prediction
        
        Returns:
        --------
        predictions : array
            Predicted labels
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        # Preprocess new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = self.models[model_name].predict(new_data_scaled)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded


def main():
    """
    Main function to run the IoT Intrusion Detection System
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "IoT INTRUSION DETECTION SYSTEM" + " " * 22 + "║")
    print("║" + " " * 12 + "Decision Tree | AdaBoost | SVM" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Initialize the detector
    detector = IoTIntrusionDetector('IoT_Intrusion.csv')
    
    # Step 1: Load data
    # Using a sample for faster experimentation (adjust as needed)
    # For full dataset, set sample_size=None
    detector.load_data(sample_size=100000)  # Using 100k samples for faster training
    
    # Step 2: Preprocess data
    # binary_classification=True: Attack vs Benign
    # binary_classification=False: Multi-class (all 34 attack types)
    detector.preprocess_data(binary_classification=True)
    
    # Step 3: Train models
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)
    
    # Train Decision Tree
    detector.train_decision_tree(max_depth=20, min_samples_split=10)
    
    # Train AdaBoost
    detector.train_adaboost(n_estimators=50, learning_rate=1.0)
    
    # Train SVM
    # Note: SVM uses a subset of data due to computational constraints
    detector.train_svm(kernel='rbf', C=1.0, gamma='scale', max_samples=30000)
    
    # Step 4: Evaluate all models
    detector.evaluate_all_models()
    
    # Step 5: Generate visualizations
    detector.plot_results()
    
    # Step 6: Save models
    detector.save_models()
    
    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)
    print(f"\n✓ All models trained and evaluated successfully!")
    print(f"✓ Results saved to: {detector.output_dir}/")
    print(f"\nFiles generated:")
    print(f"  • performance_comparison.png")
    print(f"  • confusion_matrices.png")
    print(f"  • radar_chart.png")
    print(f"  • results_summary.csv")
    print(f"  • Model files (.joblib)")
    print()
    
    return detector


if __name__ == "__main__":
    detector = main()
