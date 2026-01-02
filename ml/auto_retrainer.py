"""
Auto Retrainer pour The Bot
Reentranement automatique des modeles ML
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .feature_engineering import FeatureEngineer
from .ensemble import MLEnsemble

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """
    Reentranement automatique des modeles
    
    Responsabilites:
    - Detecter quand reentraner (heure, performance, regime marche)
    - Charger les donnees recentes
    - Reentraner les modeles
    - Valider avant deploiement (ne deploie que si meilleur)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise l'auto-retrainer
        
        Args:
            config: Configuration du retrainer
        """
        self.config = config or {}
        
        # Parametres
        self.retrain_hour = self.config.get('retrain_hour', 3)  # 3h du matin par defaut
        self.min_samples = self.config.get('min_samples', 10000)
        self.performance_threshold = self.config.get('performance_threshold', 0.6)
        self.retrain_frequency_days = self.config.get('retrain_frequency_days', 7)
        
        # "tat
        self.last_retrain = None
        self.current_performance = 0.0
        self.retrain_history = []
        
        # Composants
        self.feature_engineer = FeatureEngineer()
        self.ensemble = None
        
        logger.info(f"""| Auto Retrainer initialise (heure: {self.retrain_hour}h, freq: {self.retrain_frequency_days}j)")
    
    def should_retrain(self) -> bool:
        """
        Determine si un reentranement est necessaire
        
        Criteres:
        1. Heure programmee atteinte
        2. Performance degradee
        3. Changement de regime de marche
        
        Returns:
            True si reentranement necessaire
        """
        now = datetime.now()
        
        # 1. Verifier l'heure programmee
        if now.hour == self.retrain_hour:
            # Verifier la frequence
            if self.last_retrain is None:
                logger.info(" Premier reentranement")
                return True
            
            days_since_last = (now - self.last_retrain).days
            if days_since_last >= self.retrain_frequency_days:
                logger.info(f" Heure de reentranement ({days_since_last} jours depuis le dernier)")
                return True
        
        # 2. Verifier la performance
        if self.current_performance > 0 and self.current_performance < self.performance_threshold:
            logger.warning(f" Performance degradee: {self.current_performance:.2%} < {self.performance_threshold:.2%}")
            return True
        
        # 3. Verifier changement de regime de marche
        if self.detect_regime_change():
            logger.info("" Changement de regime de marche detecte")
            return True
        
        return False
    
    def retrain(self, 
               trades_data: pd.DataFrame,
               ohlcv_data: Dict[str, pd.DataFrame],
               model_save_path: str,
               test_size: float = 0.2) -> Dict:
        """
        Reentrane les modeles
        
        Args:
            trades_data: DataFrame avec historique des trades
            ohlcv_data: Dict {symbol: DataFrame OHLCV}
            model_save_path: Chemin pour sauvegarder les modeles
            test_size: Taille du set de validation
            
        Returns:
            Dict avec les resultats du reentranement
        """
        logger.info(""" Debut reentranement automatique...")
        
        try:
            # 1. Verifier qu'on a assez de donnees
            if len(trades_data) < self.min_samples:
                logger.warning(f"Pas assez de donnees: {len(trades_data)} < {self.min_samples}")
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_data',
                    'trades_count': len(trades_data)
                }
            
            # 2. Preparer les donnees d'entranement
            logger.info("" Preparation des donnees..."")
            X, y = self._prepare_training_data(trades_data, ohlcv_data)
            
            if len(X) < self.min_samples:
                logger.warning(f"Pas assez de samples apres preparation: {len(X)}")
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_samples',
                    'samples_count': len(X)
                }
            
            # 3. Split train/val
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logger.info(f"Split: {len(X_train)} train, {len(X_val)} val")
            
            # 4. Creer un nouvel ensemble
            new_ensemble = MLEnsemble(self.config.get('ensemble_config', {}))
            
            # 5. Entraner
            logger.info(" Entranement...")
            scores = new_ensemble.train(X_train, y_train, X_val, y_val)
            
            # 6. "valuer la performance
            from sklearn.metrics import accuracy_score
            y_pred = new_ensemble.predict_batch(X_val)
            y_pred_binary = (y_pred > 0).astype(int)
            new_performance = accuracy_score(y_val, y_pred_binary)
            
            logger.info(f"Nouvelle performance: {new_performance:.2%}")
            logger.info(f"Performance actuelle: {self.current_performance:.2%}")
            
            # 7. Decider si on deploie
            if new_performance >= self.current_performance:
                # Sauvegarder les nouveaux modeles
                new_ensemble.save(model_save_path)
                
                # Mettre  jour l'etat
                self.ensemble = new_ensemble
                self.current_performance = new_performance
                self.last_retrain = datetime.now()
                
                status = 'success'
                logger.info(f"""| Nouveaux modeles deployes: {new_performance:.2%} >= {self.current_performance:.2%}")
            else:
                status = 'rejected'
                logger.warning(f" Nouveaux modeles rejetes: {new_performance:.2%} < {self.current_performance:.2%}")
                logger.warning("Anciens modeles conserves")
            
            # 8. Enregistrer dans l'historique
            result = {
                'status': status,
                'timestamp': datetime.now(),
                'new_performance': new_performance,
                'old_performance': self.current_performance,
                'n_samples': len(X),
                'n_train': len(X_train),
                'n_val': len(X_val),
                'scores': scores
            }
            
            self.retrain_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"' Erreur reentranement: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _prepare_training_data(self, 
    """
    trades_data: pd.DataFrame,
    """
                              ohlcv_data: Dict[str, pd.DataFrame],
                              lookback_periods: int = 50) -> tuple:
        """
        Prepare les donnees d'entranement  partir des trades
        
        Args:
            trades_data: DataFrame avec les trades
            ohlcv_data: Dict avec donnees OHLCV
            lookback_periods: Nombre de periodes en arriere
            
        Returns:
            (X, y) Features et labels
        """
        X_list = []
        y_list = []
        
        for idx, trade in trades_data.iterrows():
            try:
                symbol = trade['symbol']
                entry_time = trade['entry_time']
                profit = trade['profit']
                
                # Verifier qu'on a les donnees
                if symbol not in ohlcv_data:
                    continue
                
                df = ohlcv_data[symbol]
                
                # Filtrer jusqu' l'entree
                df_before = df[df['timestamp'] <= entry_time].tail(lookback_periods)
                
                if len(df_before) < lookback_periods:
                    continue
                
                # Calculer les features
                features = self.feature_engineer.calculate_features(df_before)
                
                # Prendre la derniere ligne
                X_list.append(features[-1])
                
                # Label: 1 si profit, 0 si perte
                y_list.append(1 if profit > 0 else 0)
                
            except Exception as e:
                logger.debug(f"Erreur trade {idx}: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Donnees preparees: {len(X)} samples")
        logger.info(f"Distribution: {np.sum(y)} wins ({np.sum(y)/len(y):.1%}), {len(y) - np.sum(y)} losses")
        
        return X, y
    
    def detect_regime_change(self) -> bool:
        """
        Detecte un changement de regime de marche
        
        Analyse:
        - Volatilite
        - Volume
        - Correlations
        - Tendance
        
        Returns:
            True si changement detecte
        """
        # TODO: Implementer la detection de changement de regime
        # Pour l'instant, retourner False
        # 
        # Dans une version complete:
        # - Analyser la volatilite recente vs historique
        # - Detecter des changements de correlations entre actifs
        # - Identifier des changements de volume
        # - Detecter des changements de tendance
        
        return False
    
    def update_performance(self, performance: float):
        """
        Met  jour la performance actuelle du modele
        
        Args:
            performance: Performance mesuree (accuracy, win rate, etc.)
        """
        self.current_performance = performance
        logger.debug(f"Performance mise  jour: {performance:.2%}")
    
    def load_current_models(self, model_path: str):
        """
        Charge les modeles actuels
        
        Args:
            model_path: Chemin vers les modeles
        """
        try:
            self.ensemble = MLEnsemble()
            self.ensemble.load(model_path)
            logger.info(f"""| Modeles actuels charges: {model_path}")
        except Exception as e:
            logger.error(f"Erreur chargement modeles: {e}")
    
    def get_status(self) -> Dict:
        """
        Retourne le statut de l'auto-retrainer
        
        Returns:
            Dict avec le statut
        """
        return {
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'current_performance': self.current_performance,
            'performance_threshold': self.performance_threshold,
            'retrain_frequency_days': self.retrain_frequency_days,
            'retrain_count': len(self.retrain_history),
            'next_scheduled': self._get_next_scheduled_retrain()
        }
    
    def _get_next_scheduled_retrain(self) -> Optional[str]:
        """Calcule la prochaine heure de reentranement programmee"""
        if self.last_retrain is None:
            return "Non programme (premier reentranement)"
        
        next_retrain = self.last_retrain + timedelta(days=self.retrain_frequency_days)
        next_retrain = next_retrain.replace(hour=self.retrain_hour, minute=0, second=0)
        
        return next_retrain.isoformat()
    
    def get_history(self, limit: int = 10) -> list:
        """
        Retourne l'historique des reentranements
        
        Args:
            limit: Nombre max d'entrees
            
        Returns:
            Liste des reentranements recents
        """
        return self.retrain_history[-limit:]


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de l'auto-retrainer"""
    
    print("\n=== Test Auto Retrainer ===\n")
    
    # Configuration
    config = {
        'retrain_hour': datetime.now().hour,  # Maintenant pour test
        'min_samples': 100,
        'performance_threshold': 0.6,
        'retrain_frequency_days': 1
    }
    
    # Creer l'auto-retrainer
    retrainer = AutoRetrainer(config)
    
    # Status
    print("" Status initial:")
    status = retrainer.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test should_retrain
    print(f"\n" Should retrain: {retrainer.should_retrain()}")
    
    # Simuler une performance
    retrainer.update_performance(0.55)
    print(f"\n  Performance degradee: {retrainer.should_retrain()}")
    
    # Creer des donnees de test
    print("\n"| Creation donnees de test...")
    trades_data = pd.DataFrame({
        'symbol': ['BTCUSDT'] * 200,
        'entry_time': pd.date_range(start='2024-01-01', periods=200, freq='1h'),
        'profit': np.random.randn(200) * 10
    })
    
    ohlcv_data = {
        'BTCUSDT': pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=500, freq='5min'),
            'open': 50000 + np.cumsum(np.random.randn(500) * 100),
            'high': 50100 + np.cumsum(np.random.randn(500) * 100),
            'low': 49900 + np.cumsum(np.random.randn(500) * 100),
            'close': 50000 + np.cumsum(np.random.randn(500) * 100),
            'volume': np.random.uniform(100, 1000, 500)
        })
    }
    
    # Test reentranement
    print("\n Test reentranement...")
    result = retrainer.retrain(
        trades_data, 
        ohlcv_data, 
        'data/models/test_retrain'
    )
    
    print(f"\nResultat: {result['status']}")
    if 'new_performance' in result:
        print(f"Performance: {result['new_performance']:.2%}")
    
    print("\n""| Test termine")
