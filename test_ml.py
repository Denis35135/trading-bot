"""
test_ml.py - Tests pour le Machine Learning
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class TestFeatureEngineering:
    """Tests du feature engineering"""
    
    def test_create_features(self, sample_ohlcv_data):
        """Test crÃƒÂ©ation de features"""
        from ml.features import FeatureEngineer
        
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)
        
        assert features is not None
        assert not features.empty
        assert features.shape[1] >= 30  # Au moins 30 features
    
    def test_feature_names(self, sample_ohlcv_data):
        """Test noms des features"""
        from ml.features import FeatureEngineer
        
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)
        
        # VÃƒÂ©rifier prÃƒÂ©sence de features importantes
        expected_features = [
            'returns', 'volatility', 'rsi', 'macd',
            'volume_change', 'price_change'
        ]
        
        for feature in expected_features:
            assert any(feature in col for col in features.columns)
    
    def test_feature_scaling(self, sample_ohlcv_data):
        """Test normalisation des features"""
        from ml.features import FeatureEngineer
        
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)
        features_scaled = engineer.scale_features(features)
        
        # VÃƒÂ©rifier normalisation
        assert features_scaled.mean().mean() < 1
        assert features_scaled.std().mean() < 2


class TestMLModels:
    """Tests des modÃƒÂ¨les ML"""
    
    def test_model_training(self, sample_ohlcv_data):
        """Test entraÃƒÂ®nement du modÃƒÂ¨le"""
        from ml.models import MLPredictor
        from ml.features import FeatureEngineer
        
        # PrÃƒÂ©parer donnÃƒÂ©es
        engineer = FeatureEngineer()
        X = engineer.create_features(sample_ohlcv_data)
        y = (sample_ohlcv_data['close'].pct_change().shift(-1) > 0).astype(int)
        
        # Enlever NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) > 50:
            predictor = MLPredictor()
            predictor.train(X[:50], y[:50])
            
            assert predictor.model is not None
    
    def test_model_prediction(self, sample_ohlcv_data):
        """Test prÃƒÂ©dictions"""
        from ml.models import MLPredictor
        from ml.features import FeatureEngineer
        
        engineer = FeatureEngineer()
        X = engineer.create_features(sample_ohlcv_data)
        y = (sample_ohlcv_data['close'].pct_change().shift(-1) > 0).astype(int)
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) > 50:
            predictor = MLPredictor()
            predictor.train(X[:40], y[:40])
            
            predictions = predictor.predict(X[40:50])
            
            assert predictions is not None
            assert len(predictions) == 10
            assert all(p in [0, 1] for p in predictions)
    
    def test_model_confidence(self, sample_ohlcv_data):
        """Test confiance des prÃƒÂ©dictions"""
        from ml.models import MLPredictor
        from ml.features import FeatureEngineer
        
        engineer = FeatureEngineer()
        X = engineer.create_features(sample_ohlcv_data)
        y = (sample_ohlcv_data['close'].pct_change().shift(-1) > 0).astype(int)
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) > 50:
            predictor = MLPredictor()
            predictor.train(X[:40], y[:40])
            
            pred, confidence = predictor.predict_proba(X[40:41])
            
            assert confidence >= 0 and confidence <= 1


class TestModelEvaluation:
    """Tests d'ÃƒÂ©valuation des modÃƒÂ¨les"""
    
    def test_accuracy_calculation(self):
        """Test calcul accuracy"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy = (y_true == y_pred).mean()
        assert accuracy == 0.8
    
    def test_cross_validation(self, sample_ohlcv_data):
        """Test validation croisÃƒÂ©e"""
        from ml.models import MLPredictor
        from ml.features import FeatureEngineer
        from sklearn.model_selection import cross_val_score
        
        engineer = FeatureEngineer()
        X = engineer.create_features(sample_ohlcv_data)
        y = (sample_ohlcv_data['close'].pct_change().shift(-1) > 0).astype(int)
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) > 50:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            scores = cross_val_score(model, X, y, cv=3)
            
            assert len(scores) == 3
            assert all(s >= 0 and s <= 1 for s in scores)
