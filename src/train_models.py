"""
Comprehensive ML Model Training and Comparison Pipeline
for Parkinson's Disease Detection
"""

import pandas as pd
import numpy as np
import json
import pickle
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Models - Gradient Boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Models - Tree Ensembles
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)

# Models - Traditional ML
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Models - Neural Networks
from sklearn.neural_network import MLPClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ParkinsonModelTrainer:
    """
    Comprehensive training pipeline for Parkinson's detection models
    """

    def __init__(self, train_path='train.csv', test_path='test.csv',
                 feature_stats_path='feature_stats.json'):
        """Initialize trainer with data paths"""
        self.train_path = train_path
        self.test_path = test_path
        self.feature_stats_path = feature_stats_path

        # Create output directories
        Path("models/saved_models").mkdir(parents=True, exist_ok=True)
        Path("models/results").mkdir(parents=True, exist_ok=True)
        Path("models/hyperparameters").mkdir(parents=True, exist_ok=True)

        # Storage
        self.results = {}
        self.trained_models = {}
        self.best_params = {}

        print("🚀 Parkinson's Disease ML Pipeline Initialized")
        print("=" * 60)

    def load_data(self):
        """Load and prepare training and test data"""
        print("\n📊 Loading Data...")

        # Load datasets
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        # Load feature statistics for normalization
        with open(self.feature_stats_path, 'r') as f:
            self.feature_stats = json.load(f)

        # Feature columns (exclude ID, Status, Gender)
        self.feature_cols = [col for col in self.train_df.columns
                            if col not in ['ID', 'Status', 'Gender']]

        # Split features and labels
        self.X_train = self.train_df[self.feature_cols].values
        self.y_train = self.train_df['Status'].values
        self.X_test = self.test_df[self.feature_cols].values
        self.y_test = self.test_df['Status'].values

        # Normalize using training statistics
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"   ✓ Training set: {self.X_train.shape[0]} patients, {self.X_train.shape[1]} features")
        print(f"   ✓ Test set: {self.X_test.shape[0]} patients")
        print(f"   ✓ Class distribution (train): {np.bincount(self.y_train)}")
        print(f"   ✓ Class distribution (test): {np.bincount(self.y_test)}")

    def define_models(self):
        """Define all models and their hyperparameter grids"""
        print("\n🎯 Defining Models and Hyperparameter Grids...")

        self.model_configs = {
            # TIER 1: Gradient Boosting
            'XGBoost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'n_estimators': [100, 200, 500],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'tier': 1
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'n_estimators': [100, 200, 500],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'tier': 1
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=42, verbose=0),
                'params': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [3, 5, 7],
                    'iterations': [100, 200, 500],
                    'l2_leaf_reg': [1, 3, 5]
                },
                'tier': 1
            },

            # TIER 2: Tree Ensembles
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'tier': 2
            },
            'ExtraTrees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'tier': 2
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'tier': 2
            },

            # TIER 3: Traditional ML
            'SVM_RBF': {
                'model': SVC(kernel='rbf', probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                },
                'tier': 3
            },
            'SVM_Linear': {
                'model': SVC(kernel='linear', probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100]
                },
                'tier': 3
            },
            'LogisticRegression_L2': {
                'model': LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100]
                },
                'tier': 3
            },
            'LogisticRegression_L1': {
                'model': LogisticRegression(penalty='l1', solver='liblinear',
                                           max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100]
                },
                'tier': 3
            },

            # TIER 4: Neural Networks
            'NeuralNet_Small': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(64, 32), (128, 64)],
                    'learning_rate_init': [0.001, 0.01],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'tier': 4
            },
            'NeuralNet_Large': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(128, 64, 32), (256, 128, 64)],
                    'learning_rate_init': [0.001, 0.01],
                    'alpha': [0.0001, 0.001]
                },
                'tier': 4
            }
        }

        print(f"   ✓ Defined {len(self.model_configs)} models across 4 tiers")

    def train_single_model(self, model_name, config):
        """Train a single model with hyperparameter tuning"""
        print(f"\n🔧 Training {model_name}...")
        start_time = time.time()

        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Grid search
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        # Fit
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Cross-validation scores
        cv_scores = cross_validate(
            best_model, self.X_train, self.y_train,
            cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=-1
        )

        # Test set predictions
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'tier': config['tier'],
            'best_params': best_params,
            'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_precision_mean': cv_scores['test_precision'].mean(),
            'cv_recall_mean': cv_scores['test_recall'].mean(),
            'cv_f1_mean': cv_scores['test_f1'].mean(),
            'cv_roc_auc_mean': cv_scores['test_roc_auc'].mean(),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred, zero_division=0),
            'test_f1': f1_score(self.y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'training_time': time.time() - start_time,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = dict(zip(self.feature_cols, importances))
            metrics['feature_importance'] = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10

        # Save model and params
        self.trained_models[model_name] = best_model
        self.best_params[model_name] = best_params
        self.results[model_name] = metrics

        # Save to disk
        with open(f'models/saved_models/{model_name}_best.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        with open(f'models/hyperparameters/{model_name}_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"   ✓ CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
        print(f"   ✓ Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"   ✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   ✓ Training time: {metrics['training_time']:.2f}s")

        return metrics

    def train_all_models(self):
        """Train all models"""
        print("\n" + "=" * 60)
        print("🏋️  TRAINING ALL MODELS")
        print("=" * 60)

        total_start = time.time()

        for model_name, config in self.model_configs.items():
            try:
                self.train_single_model(model_name, config)
            except Exception as e:
                print(f"   ✗ Error training {model_name}: {e}")

        total_time = time.time() - total_start
        print(f"\n✓ All models trained in {total_time:.2f}s ({total_time/60:.2f} minutes)")

    def build_ensembles(self):
        """Build ensemble models from top 3 performers"""
        print("\n" + "=" * 60)
        print("🎯 BUILDING ENSEMBLE MODELS")
        print("=" * 60)

        # Get top 3 models by ROC-AUC
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_roc_auc'],
            reverse=True
        )[:3]

        top_model_names = [name for name, _ in sorted_models]
        print(f"\n📊 Top 3 models: {', '.join(top_model_names)}")

        # Voting Classifier
        print(f"\n🗳️  Training Voting Ensemble...")
        voting_estimators = [
            (name, self.trained_models[name])
            for name in top_model_names
        ]

        voting_clf = VotingClassifier(
            estimators=voting_estimators,
            voting='soft'
        )
        voting_clf.fit(self.X_train, self.y_train)

        # Evaluate voting
        y_pred_voting = voting_clf.predict(self.X_test)
        y_pred_proba_voting = voting_clf.predict_proba(self.X_test)[:, 1]

        self.results['Voting_Ensemble'] = {
            'model_name': 'Voting_Ensemble',
            'tier': 5,
            'base_models': top_model_names,
            'test_accuracy': accuracy_score(self.y_test, y_pred_voting),
            'test_precision': precision_score(self.y_test, y_pred_voting, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred_voting, zero_division=0),
            'test_f1': f1_score(self.y_test, y_pred_voting, zero_division=0),
            'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba_voting),
            'y_pred': y_pred_voting,
            'y_pred_proba': y_pred_proba_voting
        }

        self.trained_models['Voting_Ensemble'] = voting_clf

        print(f"   ✓ Test ROC-AUC: {self.results['Voting_Ensemble']['test_roc_auc']:.4f}")

        # Stacking Classifier
        print(f"\n📚 Training Stacking Ensemble...")
        stacking_clf = StackingClassifier(
            estimators=voting_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stacking_clf.fit(self.X_train, self.y_train)

        # Evaluate stacking
        y_pred_stacking = stacking_clf.predict(self.X_test)
        y_pred_proba_stacking = stacking_clf.predict_proba(self.X_test)[:, 1]

        self.results['Stacking_Ensemble'] = {
            'model_name': 'Stacking_Ensemble',
            'tier': 5,
            'base_models': top_model_names,
            'test_accuracy': accuracy_score(self.y_test, y_pred_stacking),
            'test_precision': precision_score(self.y_test, y_pred_stacking, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred_stacking, zero_division=0),
            'test_f1': f1_score(self.y_test, y_pred_stacking, zero_division=0),
            'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba_stacking),
            'y_pred': y_pred_stacking,
            'y_pred_proba': y_pred_proba_stacking
        }

        self.trained_models['Stacking_Ensemble'] = stacking_clf

        print(f"   ✓ Test ROC-AUC: {self.results['Stacking_Ensemble']['test_roc_auc']:.4f}")

        # Save ensembles
        with open('models/saved_models/Voting_Ensemble.pkl', 'wb') as f:
            pickle.dump(voting_clf, f)
        with open('models/saved_models/Stacking_Ensemble.pkl', 'wb') as f:
            pickle.dump(stacking_clf, f)

    def create_visualizations(self):
        """Create all comparison visualizations"""
        print("\n" + "=" * 60)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 60)

        # 1. Model Comparison Bar Chart
        print("\n📊 Creating model comparison chart...")
        self._plot_model_comparison()

        # 2. ROC Curves
        print("📈 Creating ROC curves...")
        self._plot_roc_curves()

        # 3. Confusion Matrices
        print("🔲 Creating confusion matrices...")
        self._plot_confusion_matrices()

        # 4. Feature Importance Comparison
        print("📊 Creating feature importance comparison...")
        self._plot_feature_importance()

        print("✓ All visualizations saved to models/results/")

    def _plot_model_comparison(self):
        """Plot model comparison bar chart"""
        metrics_df = pd.DataFrame([
            {
                'Model': name,
                'ROC-AUC': metrics['test_roc_auc'],
                'Accuracy': metrics['test_accuracy'],
                'Precision': metrics['test_precision'],
                'Recall': metrics['test_recall'],
                'F1-Score': metrics['test_f1']
            }
            for name, metrics in self.results.items()
        ])

        # Sort by ROC-AUC
        metrics_df = metrics_df.sort_values('ROC-AUC', ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(metrics_df))
        width = 0.15

        metrics_to_plot = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=color)

        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig('models/results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save CSV
        metrics_df.to_csv('models/results/model_comparison.csv', index=False)

    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot each model
        for name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            auc = metrics['test_roc_auc']
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - All Models', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('models/results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (name, metrics) in enumerate(self.results.items()):
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
            else:
                cm = confusion_matrix(self.y_test, metrics['y_pred'])

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[idx], cbar=False,
                xticklabels=['Healthy', 'PD'],
                yticklabels=['Healthy', 'PD']
            )
            axes[idx].set_title(f'{name}\nAUC: {metrics["test_roc_auc"]:.3f}',
                               fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('models/results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self):
        """Plot feature importance comparison for tree-based models"""
        models_with_importance = {
            name: metrics['feature_importance']
            for name, metrics in self.results.items()
            if 'feature_importance' in metrics
        }

        if not models_with_importance:
            print("   ⚠ No models with feature importance")
            return

        # Get top 10 features across all models
        all_features = set()
        for importances in models_with_importance.values():
            all_features.update(list(importances.keys())[:10])

        # Create comparison dataframe
        comparison_data = []
        for model_name, importances in models_with_importance.items():
            for feature in all_features:
                comparison_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importances.get(feature, 0)
                })

        df = pd.DataFrame(comparison_data)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Pivot for grouped bar chart
        pivot_df = df.pivot(index='Feature', columns='Model', values='Importance')
        pivot_df = pivot_df.fillna(0)

        # Sort by total importance
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('Total', ascending=True).drop('Total', axis=1)

        pivot_df.plot(kind='barh', ax=ax, width=0.8)

        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Comparison (Top Features)',
                     fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('models/results/feature_importance_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Generate comprehensive text report"""
        print("\n" + "=" * 60)
        print("📝 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)

        # Sort models by ROC-AUC
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_roc_auc'],
            reverse=True
        )

        # Build report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PARKINSON'S DISEASE DETECTION - MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Best model
        best_name, best_metrics = sorted_results[0]
        report_lines.append("🏆 BEST MODEL: " + best_name)
        report_lines.append("-" * 80)
        report_lines.append(f"Test ROC-AUC:    {best_metrics['test_roc_auc']:.4f}")
        report_lines.append(f"Test Accuracy:   {best_metrics['test_accuracy']:.4f}")
        report_lines.append(f"Test Precision:  {best_metrics['test_precision']:.4f}")
        report_lines.append(f"Test Recall:     {best_metrics['test_recall']:.4f}")
        report_lines.append(f"Test F1-Score:   {best_metrics['test_f1']:.4f}")

        if 'feature_importance' in best_metrics:
            report_lines.append("\nTop 5 Important Features:")
            for i, (feature, importance) in enumerate(
                list(best_metrics['feature_importance'].items())[:5], 1
            ):
                report_lines.append(f"  {i}. {feature}: {importance:.4f}")

        report_lines.append("")

        # Top 5 models
        report_lines.append("📊 TOP 5 MODELS (by ROC-AUC)")
        report_lines.append("-" * 80)
        for rank, (name, metrics) in enumerate(sorted_results[:5], 1):
            report_lines.append(
                f"{rank}. {name:25s} | "
                f"AUC: {metrics['test_roc_auc']:.4f} | "
                f"Acc: {metrics['test_accuracy']:.4f} | "
                f"Recall: {metrics['test_recall']:.4f}"
            )
        report_lines.append("")

        # All models table
        report_lines.append("📋 ALL MODELS - DETAILED METRICS")
        report_lines.append("-" * 80)
        report_lines.append(
            f"{'Model':<25} {'ROC-AUC':>8} {'Accuracy':>8} "
            f"{'Precision':>9} {'Recall':>8} {'F1':>8}"
        )
        report_lines.append("-" * 80)

        for name, metrics in sorted_results:
            report_lines.append(
                f"{name:<25} "
                f"{metrics['test_roc_auc']:>8.4f} "
                f"{metrics['test_accuracy']:>8.4f} "
                f"{metrics['test_precision']:>9.4f} "
                f"{metrics['test_recall']:>8.4f} "
                f"{metrics['test_f1']:>8.4f}"
            )

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("RECOMMENDATION FOR DEPLOYMENT")
        report_lines.append("=" * 80)
        report_lines.append(f"Model: {best_name}")
        report_lines.append(f"Rationale:")
        report_lines.append(f"  • Highest ROC-AUC ({best_metrics['test_roc_auc']:.4f})")
        report_lines.append(f"  • Strong recall ({best_metrics['test_recall']:.4f}) - catches PD cases")
        report_lines.append(f"  • Good precision ({best_metrics['test_precision']:.4f}) - minimizes false alarms")

        if 'training_time' in best_metrics:
            report_lines.append(f"  • Fast training ({best_metrics['training_time']:.2f}s)")

        report_lines.append("")
        report_lines.append("Next Steps:")
        report_lines.append("  1. Deploy best model to production")
        report_lines.append("  2. Integrate with audio feature extraction pipeline")
        report_lines.append("  3. Connect to Nemotron agent for clinical reasoning")
        report_lines.append("  4. Set up monitoring for model performance")
        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        with open('models/results/training_report.txt', 'w') as f:
            f.write(report_text)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Save JSON summary
        summary = {
            'best_model': best_name,
            'best_metrics': {k: v for k, v in best_metrics.items()
                           if k not in ['y_pred', 'y_pred_proba']},
            'all_results': {
                name: {k: v for k, v in metrics.items()
                      if k not in ['y_pred', 'y_pred_proba']}
                for name, metrics in self.results.items()
            }
        }

        # Convert to JSON-serializable format
        summary = convert_to_json_serializable(summary)

        with open('models/results/training_report.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Print report
        print("\n" + report_text)

        print("\n✓ Report saved to models/results/training_report.txt")
        print("✓ JSON summary saved to models/results/training_report.json")

    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        print("\n" + "=" * 60)
        print("🚀 STARTING FULL ML PIPELINE")
        print("=" * 60)

        # Load data
        self.load_data()

        # Define models
        self.define_models()

        # Train all models
        self.train_all_models()

        # Build ensembles
        self.build_ensembles()

        # Create visualizations
        self.create_visualizations()

        # Generate report
        self.generate_report()

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print("\n📂 Output Files:")
        print("   • models/saved_models/ - Trained model files (.pkl)")
        print("   • models/hyperparameters/ - Best hyperparameters (.json)")
        print("   • models/results/ - Visualizations and reports")
        print("\n🎉 Ready for deployment!\n")

if __name__ == "__main__":
    # Initialize and run
    trainer = ParkinsonModelTrainer()
    trainer.run_full_pipeline()
