"""
ML Trainer pour The Bot
EntraÃƒÂ®nement des modÃƒÂ¨les ML
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .feature_engineering import FeatureEngineer
from .ensemble import MLEnsemble

logger = logging.getLogger(__name__)


class MLTrainer:
    """
    EntraÃƒÂ®neur de modÃƒÂ¨les ML
    
    ResponsabilitÃƒÂ©s:
    - PrÃƒÂ©parer les donnÃƒÂ©es d'entraÃƒÂ®nement depuis l'historique
    - EntraÃƒÂ®ner l'ensemble de modÃƒÂ¨les
    - Valider les performances
    - Sauvegarder les modÃƒÂ¨les
    - Tracker l'historique d'entraÃƒÂ®nement
    """
    
    def __init__(self, 
                 feature_config: Optional[Dict] = None,
                 ensemble_config: Optional[Dict] = None):
        """
        Initialise le trainer
        
        Args:
            feature_config: Configuration du feature engineer
            ensemble_config: Configuration de l'ensemble
        """
        self.feature_engineer = FeatureEngineer(feature_config)
        self.ensemble = MLEnsemble(ensemble_config)
        
        self.training_history = []
        
        logger.info("Ã¢Å“â€¦ ML Trainer initialisÃƒÂ©")
    
    def prepare_training_data(self, 
                             trades_data: pd.DataFrame,
                             ohlcv_data: Dict[str, pd.DataFrame],
                             lookback_periods: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        PrÃƒÂ©pare les donnÃƒÂ©es d'entraÃƒÂ®nement ÃƒÂ  partir de l'historique des trades
        
        Args:
            trades_data: DataFrame avec colonnes: symbol, entry_time, exit_time, profit
            ohlcv_data: Dict {symbol: DataFrame OHLCV}
            lookback_periods: Nombre de pÃƒÂ©riodes ÃƒÂ  regarder en arriÃƒÂ¨re
            
        Returns:
            (X, y) Features et labels
        """
        logger.info(f"Ã°Å¸â€œÅ  PrÃƒÂ©paration donnÃƒÂ©es d'entraÃƒÂ®nement: {len(trades_data)} trades")
        
        X_list = []
        y_list = []
        skipped = 0
        
        for idx, trade in trades_data.iterrows():
            try:
                symbol = trade['symbol']
                entry_time = trade['entry_time']
                profit = trade['profit']
                
                # RÃƒÂ©cupÃƒÂ©rer les donnÃƒÂ©es OHLCV avant l'entrÃƒÂ©e
                if symbol not in ohlcv_data:
                    skipped += 1
                    continue
                
                df = ohlcv_data[symbol]
                
                # Filtrer jusqu'ÃƒÂ  l'heure d'entrÃƒÂ©e
                df_before = df[df['timestamp'] <= entry_time].tail(lookback_periods)
                
                if len(df_before) < lookback_periods:
                    skipped += 1
                    continue
                
                # Calculer les features
                features = self.feature_engineer.calculate_features(df_before)
                
                # Prendre la derniÃƒÂ¨re ligne (au moment de l'entrÃƒÂ©e)
                X_list.append(features[-1])
                
                # Label: 1 si profit positif, 0 sinon
                y_list.append(1 if profit > 0 else 0)
                
            except Exception as e:
                logger.debug(f"Erreur prÃƒÂ©paration trade {idx}: {e}")
                skipped += 1
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Ã¢Å“â€¦ DonnÃƒÂ©es prÃƒÂ©parÃƒÂ©es: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"   Skipped: {skipped} trades")
        logger.info(f"   Distribution: {np.sum(y)} wins ({np.sum(y)/len(y):.1%}), {len(y) - np.sum(y)} losses ({1 - np.sum(y)/len(y):.1%})")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray,
             test_size: float = 0.2,
             save_path: Optional[str] = None) -> Dict:
        """
        EntraÃƒÂ®ne les modÃƒÂ¨les
        
        Args:
            X: Features (shape: [n_samples, n_features])
            y: Labels binaires (0 ou 1)
            test_size: Taille du set de validation (0-1)
            save_path: Chemin pour sauvegarder les modÃƒÂ¨les
            
        Returns:
            Dict avec les rÃƒÂ©sultats d'entraÃƒÂ®nement
        """
        logger.info(f"Ã°Å¸Å½Â¯ DÃƒÂ©but entraÃƒÂ®nement: {len(X)} samples, {X.shape[1]} features")
        
        start_time = time.time()
        
        # Split train/val avec stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Split: {len(X_train)} train ({np.sum(y_train)/len(y_train):.1%} wins), "
                   f"{len(X_val)} val ({np.sum(y_val)/len(y_val):.1%} wins)")
        
        # EntraÃƒÂ®ner l'ensemble
        scores = self.ensemble.train(X_train, y_train, X_val, y_val)
        
        # Ãƒâ€°valuation dÃƒÂ©taillÃƒÂ©e sur validation
        evaluation = self.evaluate(X_val, y_val)
        
        elapsed = time.time() - start_time
        
        # RÃƒÂ©sultats
        results = {
            'timestamp': datetime.now(),
            'training_time_seconds': elapsed,
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val),
            'n_features': X.shape[1],
            'train_distribution': {
                'wins': int(np.sum(y_train)),
                'losses': int(len(y_train) - np.sum(y_train)),
                'win_rate': float(np.sum(y_train) / len(y_train))
            },
            'scores': scores,
            'evaluation': evaluation
        }
        
        # Sauvegarder l'historique
        self.training_history.append(results)
        
        # Sauvegarder les modÃƒÂ¨les
        if save_path:
            self.save_models(save_path)
            results['model_path'] = save_path
        
        # Logs
        logger.info(f"Ã¢Å“â€¦ EntraÃƒÂ®nement terminÃƒÂ© en {elapsed:.1f}s")
        logger.info(f"   Accuracy:  {evaluation['accuracy']:.2%}")
        logger.info(f"   Precision: {evaluation['precision']:.2%}")
        logger.info(f"   Recall:    {evaluation['recall']:.2%}")
        logger.info(f"   F1 Score:  {evaluation['f1_score']:.2%}")
        
        return results
    
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Ãƒâ€°value les performances sur le set de validation
        
        Args:
            X_val: Features de validation
            y_val: Labels de validation
            
        Returns:
            Dict avec les mÃƒÂ©triques de performance
        """
        # PrÃƒÂ©dictions
        y_pred = self.ensemble.predict_batch(X_val)
        
        # Convertir les signaux (-1, 0, 1) en labels binaires (0, 1)
        # Signal > 0 = BUY = prÃƒÂ©diction de profit = 1
        # Signal <= 0 = SELL/HOLD = prÃƒÂ©diction de perte = 0
        y_pred_binary = (y_pred > 0).astype(int)
        
        # Calculer les mÃƒÂ©triques
        accuracy = accuracy_score(y_val, y_pred_binary)
        precision = precision_score(y_val, y_pred_binary, zero_division=0)
        recall = recall_score(y_val, y_pred_binary, zero_division=0)
        f1 = f1_score(y_val, y_pred_binary, zero_division=0)
        
        # Matrice de confusion
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_val, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      n_splits: int = 5) -> Dict:
        """
        Validation croisÃƒÂ©e
        
        Args:
            X: Features
            y: Labels
            n_splits: Nombre de splits
            
        Returns:
            Dict avec les scores moyens
        """
        from sklearn.model_selection import StratifiedKFold
        
        logger.info(f"Ã°Å¸â€â€ž Validation croisÃƒÂ©e ({n_splits} splits)...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"  Fold {fold}/{n_splits}...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # EntraÃƒÂ®ner
            self.ensemble.train(X_train, y_train)
            
            # Ãƒâ€°valuer
            eval_results = self.evaluate(X_val, y_val)
            
            accuracies.append(eval_results['accuracy'])
            precisions.append(eval_results['precision'])
            recalls.append(eval_results['recall'])
            f1_scores.append(eval_results['f1_score'])
        
        results = {
            'n_splits': n_splits,
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'precision_mean': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_score_mean': float(np.mean(f1_scores)),
            'f1_score_std': float(np.std(f1_scores))
        }
        
        logger.info(f"Ã¢Å“â€¦ CV terminÃƒÂ©e: Accuracy={results['accuracy_mean']:.2%} Ã‚Â± {results['accuracy_std']:.2%}")
        
        return results
    
    def save_models(self, filepath: str):
        """
        Sauvegarde les modÃƒÂ¨les entraÃƒÂ®nÃƒÂ©s
        
        Args:
            filepath: Chemin pour sauvegarder
        """
        self.ensemble.save(filepath)
        logger.info(f"Ã°Å¸â€™Â¾ ModÃƒÂ¨les sauvegardÃƒÂ©s: {filepath}")
    
    def load_models(self, filepath: str):
        """
        Charge des modÃƒÂ¨les prÃƒÂ©-entraÃƒÂ®nÃƒÂ©s
        
        Args:
            filepath: Chemin des modÃƒÂ¨les
        """
        self.ensemble.load(filepath)
        logger.info(f"Ã°Å¸â€œâ€š ModÃƒÂ¨les chargÃƒÂ©s: {filepath}")
    
    def get_training_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retourne l'historique d'entraÃƒÂ®nement
        
        Args:
            limit: Nombre max d'entrÃƒÂ©es (None = toutes)
            
        Returns:
            Liste des entraÃƒÂ®nements
        """
        if limit:
            return self.training_history[-limit:]
        return self.training_history
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Retourne l'importance des features
        
        Args:
            top_n: Nombre de top features
            
        Returns:
            Dict avec les features importantes
        """
        feature_names = self.feature_engineer.get_feature_names()
        return self.ensemble.get_feature_importance(feature_names, top_n)


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du ML Trainer"""
    
    print("\n=== Test ML Trainer ===\n")
    
    # CrÃƒÂ©er des donnÃƒÂ©es synthÃƒÂ©tiques
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    # Labels basÃƒÂ©s sur une combinaison de features
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    print(f"DonnÃƒÂ©es: {n_samples} samples, {n_features} features")
    print(f"Distribution: {np.sum(y)} wins ({np.sum(y)/len(y):.1%})")
    
    # CrÃƒÂ©er le trainer
    trainer = MLTrainer()
    
    # EntraÃƒÂ®ner
    print("\nÃ°Å¸Å½Â¯ EntraÃƒÂ®nement...")
    results = trainer.train(X, y, test_size=0.2, save_path='data/models/test_trainer')
    
    print(f"\nÃ°Å¸â€œÅ  RÃƒÂ©sultats:")
    print(f"  Temps: {results['training_time_seconds']:.1f}s")
    print(f"  Accuracy: {results['evaluation']['accuracy']:.2%}")
    print(f"  Precision: {results['evaluation']['precision']:.2%}")
    print(f"  Recall: {results['evaluation']['recall']:.2%}")
    print(f"  F1 Score: {results['evaluation']['f1_score']:.2%}")
    
    # Feature importance
    print("\nÃ°Å¸Å½Â¯ Top 10 Features:")
    importance = trainer.get_feature_importance(top_n=10)
    for feature, imp in list(importance.items())[:10]:
        print(f"  {feature}: {imp:.4f}")
    
    # Validation croisÃƒÂ©e
    print("\nÃ°Å¸â€â€ž Test validation croisÃƒÂ©e...")
    cv_results = trainer.cross_validate(X, y, n_splits=3)
    print(f"  CV Accuracy: {cv_results['accuracy_mean']:.2%} Ã‚Â± {cv_results['accuracy_std']:.2%}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")