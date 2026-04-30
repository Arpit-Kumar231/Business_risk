"""
AI Agent-Based Classifier
A novel ensemble model that uses intelligent reasoning and multiple decision strategies
to predict business risk. Combines statistical analysis, feature patterns, and business logic.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class AgentBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    An intelligent agent-based classifier that combines multiple decision strategies:
    1. Statistical Anomaly Detection
    2. Feature Pattern Recognition
    3. Weighted Scoring System
    4. Adaptive Thresholding
    5. Ensemble Voting from Sub-agents
    
    This is novel because it reasons about features rather than just learning correlations.
    """
    
    def __init__(self, n_sub_agents=5, reasoning_depth='medium', adaptive=True):
        """
        Initialize the Agent-Based Classifier.
        
        Args:
            n_sub_agents: Number of sub-agent strategies to use
            reasoning_depth: 'shallow', 'medium', or 'deep' - affects complexity
            adaptive: Whether to adapt thresholds based on training data
        """
        self.n_sub_agents = n_sub_agents
        self.reasoning_depth = reasoning_depth
        self.adaptive = adaptive
        self.is_fitted = False
        self.feature_stats = None
        self.feature_means = None
        self.feature_stds = None
        self.risk_threshold = 0.5
        self.feature_weights = None
        
    def fit(self, X, y):
        """
        Train the agent by learning feature statistics and patterns from data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            self
        """
        if isinstance(X, np.ndarray):
            X_data = X
        else:
            X_data = X.values
            
        if isinstance(y, np.ndarray):
            y_data = y
        else:
            y_data = y.values
        
        # Learn feature statistics
        self.feature_means = np.mean(X_data, axis=0)
        self.feature_stds = np.std(X_data, axis=0)
        
        # Learn feature importance (correlation with target)
        correlations = []
        for i in range(X_data.shape[1]):
            corr = np.corrcoef(X_data[:, i], y_data)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # Normalize to weights
        self.feature_weights = np.array(correlations) / (np.sum(correlations) + 1e-10)
        
        # Learn optimal threshold based on training data
        if self.adaptive:
            proba = self._predict_proba_helper(X_data)
            from sklearn.metrics import f1_score
            best_f1 = 0
            best_thresh = 0.5
            
            for thresh in np.linspace(0.3, 0.7, 20):
                preds = (proba > thresh).astype(int)
                f1 = f1_score(y_data, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            self.risk_threshold = best_thresh
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict business risk class (0=Low Risk, 1=High Risk).
        
        Args:
            X: Features to predict on
            
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > self.risk_threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probability of high risk.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for both classes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, np.ndarray):
            X_data = X
        else:
            X_data = X.values
        
        # Get probabilities from helper
        proba_risk = self._predict_proba_helper(X_data)
        
        # Return in sklearn format: [[prob_class_0, prob_class_1], ...]
        proba_no_risk = 1 - proba_risk
        return np.column_stack([proba_no_risk, proba_risk])
    
    def _predict_proba_helper(self, X_data):
        """
        Internal method to compute risk probabilities using multiple agents.
        
        Args:
            X_data: Feature matrix (numpy array)
            
        Returns:
            Risk probability scores (0 to 1)
        """
        n_samples = X_data.shape[0]
        
        # Strategy 1: Statistical Anomaly Detection (Z-score based)
        anomaly_scores = self._agent_anomaly_detection(X_data)
        
        # Strategy 2: Feature Pattern Recognition
        pattern_scores = self._agent_pattern_recognition(X_data)
        
        # Strategy 3: Weighted Feature Scoring
        weighted_scores = self._agent_weighted_scoring(X_data)
        
        # Strategy 4: Outlier Detection
        outlier_scores = self._agent_outlier_detection(X_data)
        
        # Strategy 5: Distribution-based Scoring
        distribution_scores = self._agent_distribution_analysis(X_data)
        
        # Combine all strategies through ensemble voting
        if self.reasoning_depth == 'deep':
            # Deep reasoning: weighted average with adaptation
            ensemble_scores = (
                0.25 * anomaly_scores +
                0.25 * pattern_scores +
                0.20 * weighted_scores +
                0.20 * outlier_scores +
                0.10 * distribution_scores
            )
        elif self.reasoning_depth == 'medium':
            # Medium reasoning: equal weighted
            ensemble_scores = (
                anomaly_scores +
                pattern_scores +
                weighted_scores +
                outlier_scores +
                distribution_scores
            ) / 5.0
        else:
            # Shallow reasoning: simple averaging
            ensemble_scores = (anomaly_scores + pattern_scores + weighted_scores) / 3.0
        
        # Normalize to probability [0, 1]
        min_score = np.min(ensemble_scores)
        max_score = np.max(ensemble_scores)
        
        if max_score > min_score:
            proba = (ensemble_scores - min_score) / (max_score - min_score)
        else:
            proba = np.full(n_samples, 0.5)
        
        return proba
    
    def _agent_anomaly_detection(self, X_data):
        """
        Sub-agent 1: Detects anomalies using z-score analysis.
        High anomaly = high risk.
        """
        if self.feature_stds is None or np.all(self.feature_stds == 0):
            return np.zeros(len(X_data))
        
        z_scores = np.abs((X_data - self.feature_means) / (self.feature_stds + 1e-10))
        anomaly_score = np.max(z_scores, axis=1)
        
        # Normalize to [0, 1]
        anomaly_score = np.clip(anomaly_score / 5.0, 0, 1)
        return anomaly_score
    
    def _agent_pattern_recognition(self, X_data):
        """
        Sub-agent 2: Recognizes risky feature patterns.
        Combines feature importance with current values.
        """
        if self.feature_weights is None:
            return np.zeros(len(X_data))
        
        # Normalize features to [0, 1]
        X_normalized = (X_data - self.feature_means) / (self.feature_stds + 1e-10)
        X_normalized = np.clip((X_normalized + 3) / 6, 0, 1)  # Map roughly to [0,1]
        
        # Weight by importance and aggregate
        pattern_score = np.dot(X_normalized, self.feature_weights)
        return np.clip(pattern_score, 0, 1)
    
    def _agent_weighted_scoring(self, X_data):
        """
        Sub-agent 3: Uses weighted scoring based on feature importance.
        High-weight features with high values = high risk.
        """
        if self.feature_weights is None:
            return np.zeros(len(X_data))
        
        # Percentile-based scoring
        scores = []
        for i in range(X_data.shape[1]):
            feat_min = np.min(X_data[:, i])
            feat_max = np.max(X_data[:, i])
            
            if feat_max > feat_min:
                percentile = (X_data[:, i] - feat_min) / (feat_max - feat_min)
            else:
                percentile = np.zeros(len(X_data))
            
            scores.append(percentile * self.feature_weights[i])
        
        weighted_score = np.sum(scores, axis=0)
        return np.clip(weighted_score, 0, 1)
    
    def _agent_outlier_detection(self, X_data):
        """
        Sub-agent 4: Detects outliers using IQR-based method.
        Outliers often indicate risky companies.
        """
        outlier_scores = np.zeros(len(X_data))
        
        for i in range(X_data.shape[1]):
            feat = X_data[:, i]
            Q1 = np.percentile(feat, 25)
            Q3 = np.percentile(feat, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            is_outlier = (feat < lower_bound) | (feat > upper_bound)
            outlier_scores += is_outlier.astype(float)
        
        # Normalize by number of features
        outlier_scores = outlier_scores / X_data.shape[1]
        return np.clip(outlier_scores, 0, 1)
    
    def _agent_distribution_analysis(self, X_data):
        """
        Sub-agent 5: Analyzes distribution of features.
        Extreme values in important features = high risk.
        """
        if self.feature_weights is None:
            return np.zeros(len(X_data))
        
        distribution_score = np.zeros(len(X_data))
        
        for i in range(X_data.shape[1]):
            feat = X_data[:, i]
            
            # Calculate how extreme each value is in its distribution
            mean = np.mean(feat)
            std = np.std(feat)
            
            if std > 0:
                deviation = np.abs((feat - mean) / std)
                extremeness = np.clip(deviation / 3.0, 0, 1)
            else:
                extremeness = np.zeros(len(feat))
            
            distribution_score += extremeness * self.feature_weights[i]
        
        return np.clip(distribution_score, 0, 1)
    
    def get_agent_reasoning(self, X, sample_idx=0):
        """
        Get detailed reasoning from the agent for a specific sample.
        Useful for explainability.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with reasoning from each sub-agent
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reasoning")
        
        if isinstance(X, np.ndarray):
            X_data = X
        else:
            X_data = X.values
        
        sample = X_data[sample_idx:sample_idx+1]
        
        return {
            'sample_index': sample_idx,
            'anomaly_score': float(self._agent_anomaly_detection(sample)[0]),
            'pattern_score': float(self._agent_pattern_recognition(sample)[0]),
            'weighted_score': float(self._agent_weighted_scoring(sample)[0]),
            'outlier_score': float(self._agent_outlier_detection(sample)[0]),
            'distribution_score': float(self._agent_distribution_analysis(sample)[0]),
            'final_prediction': float(self.predict_proba(X)[sample_idx, 1]),
            'reasoning_depth': self.reasoning_depth,
            'adaptive_threshold': float(self.risk_threshold)
        }
