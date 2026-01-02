"""
Ensemble ML pour The Bot
Ensemble de 3 modeles optimises pour CPU: LightGBM, XGBoost, RandomForest
"""

import numpy as np
import joblib
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

# Imports ML
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class MLEnsemble:
    """
    Ensemble de 3 modeles legers mais performants
    
    Modeles:
    - LightGBM: Rapide et efficace
    - XGBoost: Precis avec tree_method='hist'
    - RandomForest: Robuste et stable
    
    Prediction par vote majoritaire avec seuil de confiance eleve
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise l'ensemble de modeles
        
        Args:
            config: Configuration des modeles
        """
        self.config = config or {}
        
        # Parametres par defaut optimises pour PC classique
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth_lgb = self.config.get('max_depth_lgb', 6)
        self.max_depth_xgb = self.config.get('max_depth_xgb', 5)
        self.max_depth_rf = self.config.get('max_depth_rf', 8)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.n_jobs = self.config.get('n_jobs', 4)
        
        # Seuil de confiance pour predictions
        self.confidence_threshold = self.config.get('confidence_threshold', 0.65)
        
        # Initialiser les modeles
        self.models = self._initialize_models()
        
        # Statistiques
        self.is_trained = False
        self.training_scores = {}
        self.feature_importance = None
        
        logger.info(f"""| ML Ensemble initialise (3 modeles, n_jobs={self.n_jobs})")
    
    def _initialize_models(self) -> Dict:
        """Initialise les 3 modeles"""
        models = {}
        
        # 1. LightGBM - Tres rapide
        models['lgb'] = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth_lgb,
            num_leaves=31,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=-1
        )
        
        # 2. XGBoost - Precis
        models['xgb'] = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth_xgb,
            learning_rate=self.learning_rate,
            tree_method='hist',  # Plus rapide pour CPU
            n_jobs=self.n_jobs,
            random_state=42,
            verbosity=0
        )
        
        # 3. RandomForest - Robuste
        models['rf'] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth_rf,
            min_samples_split=20,
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=0
        )
        
        return models
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
    """
    X_val: Optional[np.ndarray] = None,
    """
             y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Entrane tous les modeles
        
        Args:
            X_train: Features d'entranement
            y_train: Labels d'entranement
            X_val: Features de validation (optionnel)
            y_val: Labels de validation (optionnel)
            
        Returns:
            Dict avec les scores de chaque modele
        """
        logger.info(f"Entranement de l'ensemble sur {len(X_train)} samples...")
        
        scores = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"  Entranement {name}...")
                
                # Entraner
                model.fit(X_train, y_train)
                
                # Score sur train
                train_score = model.score(X_train, y_train)
                scores[f'{name}_train'] = train_score
                
                # Score sur validation si disponible
                if X_val is not None and y_val is not None:
                    val_score = model.score(X_val, y_val)
                    scores[f'{name}_val'] = val_score
                    logger.info(f"    {name}: Train={train_score:.2%}, Val={val_score:.2%}")
                else:
                    logger.info(f"    {name}: Train={train_score:.2%}")
                
            except Exception as e:
                logger.error(f"Erreur entranement {name}: {e}")
                scores[f'{name}_error'] = str(e)
        
        self.is_trained = True
        self.training_scores = scores
        
        # Calculer l'importance des features (moyenne des 3 modeles)
        self._calculate_feature_importance()
        
        logger.info("""| Entranement termine")
        
        return scores
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False) -> Tuple[int, float]:
        """
        Predit avec vote majoritaire
        
        Args:
            X: Features (1 sample, shape: [n_features])
            return_probabilities: Si True, retourne aussi les probas detaillees
            
        Returns:
            (signal, confidence) o:
            - signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            - confidence: Niveau de confiance (0-1)
        """
        if not self.is_trained:
            logger.warning("Modeles non entranes, retour HOLD")
            return 0, 0.0
        
        try:
            # Reshape si necessaire
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Predictions de chaque modele
            predictions = []
            probabilities = []
            
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(X)[0]
                    probabilities.append(proba)
                    predictions.append(proba)
                except Exception as e:
                    logger.error(f"Erreur prediction {name}: {e}")
                    # Prediction neutre en cas d'erreur
                    probabilities.append(np.array([0.5, 0.5]))
            
            # Moyenne des probabilites
            avg_proba = np.mean(probabilities, axis=0)
            
            # Decision avec seuil de confiance
            if avg_proba[1] > self.confidence_threshold:  # BUY
                signal = 1
                confidence = avg_proba[1]
            elif avg_proba[0] > self.confidence_threshold:  # SELL
                signal = -1
                confidence = avg_proba[0]
            else:  # HOLD
                signal = 0
                confidence = max(avg_proba)
            
            if return_probabilities:
                return signal, confidence, avg_proba
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Erreur prediction ensemble: {e}")
            return 0, 0.0
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predit sur un batch de samples
        
        Args:
            X: Features (shape: [n_samples, n_features])
            
        Returns:
            Array de signaux (1, -1, ou 0)
        """
        if not self.is_trained:
            return np.zeros(len(X), dtype=int)
        
        signals = []
        
        for i in range(len(X)):
            signal, _ = self.predict(X[i])
            signals.append(signal)
        
        return np.array(signals)
    
    def _calculate_feature_importance(self):
        """Calcule l'importance moyenne des features"""
        try:
            importances = []
            
            # LightGBM et XGBoost ont feature_importances_
            for name in ['lgb', 'xgb', 'rf']:
                if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                    importances.append(self.models[name].feature_importances_)
            
            if importances:
                self.feature_importance = np.mean(importances, axis=0)
                logger.debug(f"Feature importance calculee: {len(self.feature_importance)} features")
            
        except Exception as e:
            logger.error(f"Erreur calcul feature importance: {e}")
            self.feature_importance = None
    
    def get_feature_importance(self, feature_names: Optional[list] = None, 
                              top_n: int = 10) -> Dict:
        """
        Retourne l'importance des features
        
        Args:
            feature_names: Noms des features (optionnel)
            top_n: Nombre de top features  retourner
            
        Returns:
            Dict avec les features et leur importance
        """
        if self.feature_importance is None:
            return {}
        
        # Creer un dict avec importance
        if feature_names:
            importance_dict = dict(zip(feature_names, self.feature_importance))
        else:
            importance_dict = {f'feature_{i}': imp for i, imp in enumerate(self.feature_importance)}
        
        # Trier et garder top N
        sorted_importance = dict(sorted(importance_dict.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:top_n])
        
        return sorted_importance
    
    def save(self, filepath: str):
        """
        Sauvegarde les modeles
        
        Args:
            filepath: Chemin du fichier (sans extension)
        """
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder chaque modele
            for name, model in self.models.items():
                model_path = save_path.parent / f"{save_path.stem}_{name}.joblib"
                joblib.dump(model, model_path)
            
            # Sauvegarder les metadonnees
            metadata = {
                'is_trained': self.is_trained,
                'training_scores': self.training_scores,
                'feature_importance': self.feature_importance,
                'config': self.config
            }
            meta_path = save_path.parent / f"{save_path.stem}_meta.joblib"
            joblib.dump(metadata, meta_path)
            
            logger.info(f"""| Modeles sauvegardes: {save_path.parent}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modeles: {e}")
    
    def load(self, filepath: str):
        """
        Charge les modeles
        
        Args:
            filepath: Chemin du fichier (sans extension)
        """
        try:
            load_path = Path(filepath)
            
            # Charger chaque modele
            for name in self.models.keys():
                model_path = load_path.parent / f"{load_path.stem}_{name}.joblib"
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                else:
                    logger.warning(f"Modele {name} introuvable: {model_path}")
            
            # Charger les metadonnees
            meta_path = load_path.parent / f"{load_path.stem}_meta.joblib"
            if meta_path.exists():
                metadata = joblib.load(meta_path)
                self.is_trained = metadata.get('is_trained', False)
                self.training_scores = metadata.get('training_scores', {})
                self.feature_importance = metadata.get('feature_importance', None)
            
            logger.info(f"""| Modeles charges: {load_path.parent}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modeles: {e}")
    
    def get_model_info(self) -> Dict:
        """Retourne les informations sur les modeles"""
        return {
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'n_estimators': self.n_estimators,
            'confidence_threshold': self.confidence_threshold,
            'training_scores': self.training_scores,
            'feature_count': len(self.feature_importance) if self.feature_importance is not None else 0
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de l'ensemble ML"""
    
    print("\n=== Test ML Ensemble ===\n")
    
    # Donnees de test
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    # Generer des donnees synthetiques
    X = np.random.randn(n_samples, n_features)
    # Labels: 0 (SELL) ou 1 (BUY) base sur une combinaison de features
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    # Split train/val
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Features: {n_features}")
    
    # Creer l'ensemble
    ensemble = MLEnsemble({'confidence_threshold': 0.65})
    
    # Entraner
    import time
    start = time.time()
    scores = ensemble.train(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start
    
    print(f"\n  Temps d'entranement: {elapsed:.2f}s")
    print("\n" Scores:")
    for name, score in scores.items():
        if isinstance(score, float):
            print(f"  {name}: {score:.2%}")
    
    # Tester des predictions
    print("\n" Test de predictions:")
    for i in range(5):
        signal, confidence = ensemble.predict(X_val[i])
        signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[signal]
        print(f"  Sample {i}: {signal_name} (confidence: {confidence:.2%})")
    
    # Feature importance
    print("\n Top 10 features importantes:")
    importance = ensemble.get_feature_importance(top_n=10)
    for feature, imp in importance.items():
        print(f"  {feature}: {imp:.4f}")
    
    # Sauvegarder et recharger
    print("\n' Test sauvegarde/chargement...")
    ensemble.save('data/models/test_ensemble')
    
    ensemble2 = MLEnsemble()
    ensemble2.load('data/models/test_ensemble')
    
    # Verifier que a fonctionne apres chargement
    signal, confidence = ensemble2.predict(X_val[0])
    print(f"  Prediction apres chargement: {signal}, conf: {confidence:.2%}")
    
    print("\n""| Tests termines")
