"""
RSNA Aneurysm Detection - Metrics Module

Evaluation metrics and analysis functions for binary classification tasks.
Includes medical-specific metrics and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for binary classification
    
    Calculates various metrics commonly used in medical imaging competitions
    and provides detailed analysis of model performance.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
        """
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities [0, 1]
            threshold: Classification threshold
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.threshold = threshold
        self.y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input arrays"""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if not np.all(np.isin(self.y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")
        
        if not np.all((self.y_pred >= 0) & (self.y_pred <= 1)):
            raise ValueError("y_pred must contain probabilities in [0, 1]")
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive set of metrics
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        # ROC-AUC (primary metric for Kaggle competitions)
        metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred)
        
        # Precision-Recall metrics
        metrics['avg_precision'] = average_precision_score(self.y_true, self.y_pred)
        
        # Classification metrics at threshold
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred_binary)
        metrics['precision'] = precision_score(self.y_true, self.y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(self.y_true, self.y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(self.y_true, self.y_pred_binary, zero_division=0)
        
        # Additional metrics
        metrics['matthews_corr'] = matthews_corrcoef(self.y_true, self.y_pred_binary)
        metrics['cohen_kappa'] = cohen_kappa_score(self.y_true, self.y_pred_binary)
        
        # Probabilistic metrics
        metrics['log_loss'] = log_loss(self.y_true, self.y_pred, eps=1e-7)
        metrics['brier_score'] = brier_score_loss(self.y_true, self.y_pred)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred_binary).ravel()
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # Youden's J statistic
        metrics['youden_j'] = metrics['sensitivity'] + metrics['specificity'] - 1
        
        return metrics
    
    def calculate_optimal_thresholds(self) -> Dict[str, float]:
        """
        Calculate optimal thresholds using different methods
        
        Returns:
            Dictionary of method -> optimal threshold
        """
        thresholds = {}
        
        # Youden's J statistic (maximizes sensitivity + specificity - 1)
        fpr, tpr, thresh_roc = roc_curve(self.y_true, self.y_pred)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        thresholds['youden'] = thresh_roc[optimal_idx]
        
        # F1 score maximization
        precision, recall, thresh_pr = precision_recall_curve(self.y_true, self.y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        thresholds['f1'] = thresh_pr[optimal_idx] if optimal_idx < len(thresh_pr) else 0.5
        
        # Precision-Recall balance point
        diff = np.abs(precision - recall)
        optimal_idx = np.argmin(diff)
        thresholds['pr_balance'] = thresh_pr[optimal_idx] if optimal_idx < len(thresh_pr) else 0.5
        
        return thresholds
    
    def get_confusion_matrix_stats(self) -> Dict[str, Any]:
        """
        Get detailed confusion matrix statistics
        
        Returns:
            Dictionary with confusion matrix and derived statistics
        """
        cm = confusion_matrix(self.y_true, self.y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        
        stats = {
            'confusion_matrix': cm,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'total': int(total),
            'positive_samples': int(tp + fn),
            'negative_samples': int(tn + fp),
            'predicted_positive': int(tp + fp),
            'predicted_negative': int(tn + fn),
            'prevalence': (tp + fn) / total if total > 0 else 0,
        }
        
        return stats
    
    def plot_roc_curve(self, title: str = "ROC Curve", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        auc_score = roc_auc_score(self.y_true, self.y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(self, title: str = "Precision-Recall Curve", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)
        avg_precision = average_precision_score(self.y_true, self.y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})', linewidth=2)
        
        # Baseline (random classifier)
        baseline = np.sum(self.y_true) / len(self.y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.4f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_confusion_matrix(self, title: str = "Confusion Matrix", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(self.y_true, self.y_pred_binary)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        
        ax.set_title(f'{title} (Threshold = {self.threshold:.3f})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig
    
    def plot_calibration_curve(self, n_bins: int = 10, title: str = "Calibration Curve", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_true, self.y_pred, n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label='Model', linewidth=2)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", alpha=0.7)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_prediction_distribution(self, title: str = "Prediction Distribution", figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot prediction probability distribution by class"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram by class
        ax1.hist(self.y_pred[self.y_true == 0], bins=50, alpha=0.7, label='Negative', density=True)
        ax1.hist(self.y_pred[self.y_true == 1], bins=50, alpha=0.7, label='Positive', density=True)
        ax1.axvline(self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold:.3f}')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution by True Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [self.y_pred[self.y_true == 0], self.y_pred[self.y_true == 1]]
        ax2.boxplot(data_to_plot, labels=['Negative', 'Positive'])
        ax2.axhline(self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold:.3f}')
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title('Box Plot by True Class')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


class CrossValidationAnalyzer:
    """
    Analyze cross-validation results for consistency and reliability
    """
    
    def __init__(self, fold_predictions: List[np.ndarray], fold_targets: List[np.ndarray]):
        """
        Args:
            fold_predictions: List of prediction arrays for each fold
            fold_targets: List of target arrays for each fold
        """
        self.fold_predictions = fold_predictions
        self.fold_targets = fold_targets
        self.n_folds = len(fold_predictions)
    
    def calculate_fold_metrics(self) -> pd.DataFrame:
        """Calculate metrics for each fold"""
        fold_metrics = []
        
        for i, (y_pred, y_true) in enumerate(zip(self.fold_predictions, self.fold_targets)):
            calculator = MetricsCalculator(y_true, y_pred)
            metrics = calculator.calculate_all_metrics()
            metrics['fold'] = i + 1
            fold_metrics.append(metrics)
        
        return pd.DataFrame(fold_metrics)
    
    def analyze_consistency(self) -> Dict[str, Any]:
        """Analyze fold consistency"""
        fold_df = self.calculate_fold_metrics()
        
        # Calculate statistics for key metrics
        key_metrics = ['roc_auc', 'avg_precision', 'f1', 'accuracy']
        consistency_stats = {}
        
        for metric in key_metrics:
            values = fold_df[metric].values
            consistency_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            }
        
        # Overall consistency score
        roc_auc_cv = consistency_stats['roc_auc']['cv']
        consistency_score = max(0, 1 - roc_auc_cv * 10)  # Higher is better
        
        return {
            'fold_metrics': fold_df,
            'consistency_stats': consistency_stats,
            'consistency_score': consistency_score,
            'n_folds': self.n_folds
        }
    
    def plot_fold_comparison(self, metric: str = 'roc_auc', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot metric comparison across folds"""
        fold_df = self.calculate_fold_metrics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        ax1.bar(fold_df['fold'], fold_df[metric])
        ax1.set_xlabel('Fold')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'{metric.replace("_", " ").title()} by Fold')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot([fold_df[metric]], labels=[metric.replace('_', ' ').title()])
        ax2.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = fold_df[metric].mean()
        ax1.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean = {mean_val:.4f}')
        ax1.legend()
        
        plt.tight_layout()
        return fig


class ThresholdOptimizer:
    """
    Optimize classification threshold using various methods
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def optimize_threshold(self, method: str = 'youden') -> Dict[str, Any]:
        """
        Optimize threshold using specified method
        
        Args:
            method: Optimization method ('youden', 'f1', 'precision_recall_balance', 'roc_point')
        """
        if method == 'youden':
            return self._optimize_youden()
        elif method == 'f1':
            return self._optimize_f1()
        elif method == 'precision_recall_balance':
            return self._optimize_precision_recall_balance()
        elif method == 'roc_point':
            return self._optimize_roc_point()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_youden(self) -> Dict[str, Any]:
        """Optimize using Youden's J statistic"""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        
        return {
            'method': 'youden',
            'threshold': thresholds[optimal_idx],
            'score': youden_scores[optimal_idx],
            'sensitivity': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx]
        }
    
    def _optimize_f1(self) -> Dict[str, Any]:
        """Optimize F1 score"""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        
        threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return {
            'method': 'f1',
            'threshold': threshold,
            'score': f1_scores[optimal_idx],
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx]
        }
    
    def _optimize_precision_recall_balance(self) -> Dict[str, Any]:
        """Optimize for precision-recall balance"""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred)
        balance_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(balance_scores)
        
        threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return {
            'method': 'precision_recall_balance',
            'threshold': threshold,
            'score': balance_scores[optimal_idx],
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx]
        }
    
    def _optimize_roc_point(self, target_fpr: float = 0.1) -> Dict[str, Any]:
        """Optimize for specific point on ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)
        
        # Find threshold closest to target FPR
        optimal_idx = np.argmin(np.abs(fpr - target_fpr))
        
        return {
            'method': f'roc_point_fpr_{target_fpr}',
            'threshold': thresholds[optimal_idx],
            'fpr': fpr[optimal_idx],
            'tpr': tpr[optimal_idx],
            'target_fpr': target_fpr
        }
    
    def compare_methods(self) -> pd.DataFrame:
        """Compare all optimization methods"""
        methods = ['youden', 'f1', 'precision_recall_balance']
        results = []
        
        for method in methods:
            result = self.optimize_threshold(method)
            
            # Calculate metrics at this threshold
            calculator = MetricsCalculator(self.y_true, self.y_pred, result['threshold'])
            metrics = calculator.calculate_all_metrics()
            
            result.update(metrics)
            results.append(result)
        
        return pd.DataFrame(results)


def bootstrap_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_bootstrap: int = 1000, 
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed
    
    Returns:
        Dictionary with metrics and their confidence intervals
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Store bootstrap results
    bootstrap_scores = {
        'roc_auc': [],
        'avg_precision': [],
        'accuracy': [],
        'f1': []
    }
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        # Calculate metrics
        calculator = MetricsCalculator(y_true_boot, y_pred_boot)
        metrics = calculator.calculate_all_metrics()
        
        for metric in bootstrap_scores.keys():
            if metric in metrics:
                bootstrap_scores[metric].append(metrics[metric])
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {}
    for metric, scores in bootstrap_scores.items():
        if scores:  # Check if we have valid scores
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'lower_ci': np.percentile(scores, lower_percentile),
                'upper_ci': np.percentile(scores, upper_percentile)
            }
    
    return results


if __name__ == "__main__":
    # Test metrics calculation
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred = np.random.beta(2, 5, n_samples)  # Skewed towards 0
    
    # Make predictions somewhat correlated with truth
    y_pred[y_true == 1] += 0.3
    y_pred = np.clip(y_pred, 0, 1)
    
    print("Testing metrics calculation...")
    
    # Calculate metrics
    calculator = MetricsCalculator(y_true, y_pred)
    metrics = calculator.calculate_all_metrics()
    
    print("Calculated metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test optimal thresholds
    optimizer = ThresholdOptimizer(y_true, y_pred)
    comparison_df = optimizer.compare_methods()
    print(f"\nOptimal thresholds comparison:")
    print(comparison_df[['method', 'threshold', 'roc_auc', 'f1']].round(4))
    
    # Test bootstrap CI
    bootstrap_results = bootstrap_metrics(y_true, y_pred, n_bootstrap=100)
    print(f"\nBootstrap confidence intervals:")
    for metric, stats in bootstrap_results.items():
        print(f"  {metric}: {stats['mean']:.4f} [{stats['lower_ci']:.4f}, {stats['upper_ci']:.4f}]")
    
    print("\nAll tests passed!")