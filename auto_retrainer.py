"""
Auto Retrainer pour The Bot
RÃƒÂ©entraÃƒÂ®nement automatique des modÃƒÂ¨les ML
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
    RÃƒÂ©entraÃƒÂ®nement automatique des modÃƒÂ¨les
    
    ResponsabilitÃƒÂ©s:
    - DÃƒÂ©tecter quand rÃƒÂ©entraÃƒÂ®ner (heure, performance, rÃƒÂ©gime marchÃƒÂ©)
    - Charger les donnÃƒÂ©es rÃƒÂ©centes
    - RÃƒÂ©entraÃƒÂ®ner les modÃƒÂ¨les
    - Valider avant dÃƒÂ©ploiement (ne dÃƒÂ©ploie que si meilleur)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise l'auto-retrainer
        
        Args:
            config: Configuration du retrainer
        """
        self.config = config or {}
        
        # ParamÃƒÂ¨tres
        self.retrain_hour = self.config.get('retrain_hour', 3)  # 3h du matin par dÃƒÂ©faut
        self.min_samples = self.config.get('min_samples', 10000)
        self.performance_threshold = self.config.get('performance_threshold', 0.6)
        self.retrain_frequency_days = self.config.get('retrain_frequency_days', 7)
        
        # Ãƒâ€°tat
        self.last_retrain = None
        self.current_performance = 0.0
        self.retrain_history = []
        
        # Composants
        self.feature_engineer = FeatureEngineer()
        self.ensemble = None
        
        logger.info(f"Ã¢Å“â€¦ Auto Retrainer initialisÃƒÂ© (heure: {self.retrain_hour}h, freq: {self.retrain_frequency_days}j)")
    
    def should_retrain(self) -> bool:
        """
        DÃƒÂ©termine si un rÃƒÂ©entraÃƒÂ®nement est nÃƒÂ©cessaire
        
        CritÃƒÂ¨res:
        1. Heure programmÃƒÂ©e atteinte
        2. Performance dÃƒÂ©gradÃƒÂ©e
        3. Changement de rÃƒÂ©gime de marchÃƒÂ©
        
        Returns:
            True si rÃƒÂ©entraÃƒÂ®nement nÃƒÂ©cessaire
        """
        now = datetime.now()
        
        # 1. VÃƒÂ©rifier l'heure programmÃƒÂ©e
        if now.hour == self.retrain_hour:
            # VÃƒÂ©rifier la frÃƒÂ©quence
            if self.last_retrain is None:
                logger.info("Ã¢ÂÂ° Premier rÃƒÂ©entraÃƒÂ®nement")
                return True
            
            days_since_last = (now - self.last_retrain).days
            if days_since_last >= self.retrain_frequency_days:
                logger.info(f"Ã¢ÂÂ° Heure de rÃƒÂ©entraÃƒÂ®nement ({days_since_last} jours depuis le dernier)")
                return True
        
        # 2. VÃƒÂ©rifier la performance
        if self.current_performance > 0 and self.current_performance < self.performance_threshold:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Performance dÃƒÂ©gradÃƒÂ©e: {self.current_performance:.2%} < {self.performance_threshold:.2%}")
            return True
        
        # 3. VÃƒÂ©rifier changement de rÃƒÂ©gime de marchÃƒÂ©
        if self.detect_regime_change():
            logger.info("Ã°Å¸â€œÅ  Changement de rÃƒÂ©gime de marchÃƒÂ© dÃƒÂ©tectÃƒÂ©")
            return True
        
        return False
    
    def retrain(self, 
               trades_data: pd.DataFrame,
               ohlcv_data: Dict[str, pd.DataFrame],
               model_save_path: str,
               test_size: float = 0.2) -> Dict:
        """
        RÃƒÂ©entraÃƒÂ®ne les modÃƒÂ¨les
        
        Args:
            trades_data: DataFrame avec historique des trades
            ohlcv_data: Dict {symbol: DataFrame OHLCV}
            model_save_path: Chemin pour sauvegarder les modÃƒÂ¨les
            test_size: Taille du set de validation
            
        Returns:
            Dict avec les rÃƒÂ©sultats du rÃƒÂ©entraÃƒÂ®nement
        """
        logger.info("Ã°Å¸â€â€ž DÃƒÂ©but rÃƒÂ©entraÃƒÂ®nement automatique...")
        
        try:
            # 1. VÃƒÂ©rifier qu'on a assez de donnÃƒÂ©es
            if len(trades_data) < self.min_samples:
                logger.warning(f"Pas assez de donnÃƒÂ©es: {len(trades_data)} < {self.min_samples}")
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_data',
                    'trades_count': len(trades_data)
                }
            
            # 2. PrÃƒÂ©parer les donnÃƒÂ©es d'entraÃƒÂ®nement
            logger.info("Ã°Å¸â€œÅ  PrÃƒÂ©paration des donnÃƒÂ©es...")
            X, y = self._prepare_training_data(trades_data, ohlcv_data)
            
            if len(X) < self.min_samples:
                logger.warning(f"Pas assez de samples aprÃƒÂ¨s prÃƒÂ©paration: {len(X)}")
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
            
            # 4. CrÃƒÂ©er un nouvel ensemble
            new_ensemble = MLEnsemble(self.config.get('ensemble_config', {}))
            
            # 5. EntraÃƒÂ®ner
            logger.info("Ã°Å¸Å½Â¯ EntraÃƒÂ®nement...")
            scores = new_ensemble.train(X_train, y_train, X_val, y_val)
            
            # 6. Ãƒâ€°valuer la performance
            from sklearn.metrics import accuracy_score
            y_pred = new_ensemble.predict_batch(X_val)
            y_pred_binary = (y_pred > 0).astype(int)
            new_performance = accuracy_score(y_val, y_pred_binary)
            
            logger.info(f"Nouvelle performance: {new_performance:.2%}")
            logger.info(f"Performance actuelle: {self.current_performance:.2%}")
            
            # 7. DÃƒÂ©cider si on dÃƒÂ©ploie
            if new_performance >= self.current_performance:
                # Sauvegarder les nouveaux modÃƒÂ¨les
                new_ensemble.save(model_save_path)
                
                # Mettre ÃƒÂ  jour l'ÃƒÂ©tat
                self.ensemble = new_ensemble
                self.current_performance = new_performance
                self.last_retrain = datetime.now()
                
                status = 'success'
                logger.info(f"Ã¢Å“â€¦ Nouveaux modÃƒÂ¨les dÃƒÂ©ployÃƒÂ©s: {new_performance:.2%} >= {self.current_performance:.2%}")
            else:
                status = 'rejected'
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Nouveaux modÃƒÂ¨les rejetÃƒÂ©s: {new_performance:.2%} < {self.current_performance:.2%}")
                logger.warning("Anciens modÃƒÂ¨les conservÃƒÂ©s")
            
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
            logger.error(f"Ã¢ÂÅ’ Erreur rÃƒÂ©entraÃƒÂ®nement: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _prepare_training_data(self, 
                              trades_data: pd.DataFrame,
                              ohlcv_data: Dict[str, pd.DataFrame],
                              lookback_periods: int = 50) -> tuple:
        """
        PrÃƒÂ©pare les donnÃƒÂ©es d'entraÃƒÂ®nement ÃƒÂ  partir des trades
        
        Args:
            trades_data: DataFrame avec les trades
            ohlcv_data: Dict avec donnÃƒÂ©es OHLCV
            lookback_periods: Nombre de pÃƒÂ©riodes en arriÃƒÂ¨re
            
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
                
                # VÃƒÂ©rifier qu'on a les donnÃƒÂ©es
                if symbol not in ohlcv_data:
                    continue
                
                df = ohlcv_data[symbol]
                
                # Filtrer jusqu'ÃƒÂ  l'entrÃƒÂ©e
                df_before = df[df['timestamp'] <= entry_time].tail(lookback_periods)
                
                if len(df_before) < lookback_periods:
                    continue
                
                # Calculer les features
                features = self.feature_engineer.calculate_features(df_before)
                
                # Prendre la derniÃƒÂ¨re ligne
                X_list.append(features[-1])
                
                # Label: 1 si profit, 0 si perte
                y_list.append(1 if profit > 0 else 0)
                
            except Exception as e:
                logger.debug(f"Erreur trade {idx}: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"DonnÃƒÂ©es prÃƒÂ©parÃƒÂ©es: {len(X)} samples")
        logger.info(f"Distribution: {np.sum(y)} wins ({np.sum(y)/len(y):.1%}), {len(y) - np.sum(y)} losses")
        
        return X, y
    
    def detect_regime_change(self) -> bool:
        """
        DÃƒÂ©tecte un changement de rÃƒÂ©gime de marchÃƒÂ©
        
        Analyse:
        - VolatilitÃƒÂ©
        - Volume
        - CorrÃƒÂ©lations
        - Tendance
        
        Returns:
            True si changement dÃƒÂ©tectÃƒÂ©
        """
        # TODO: ImplÃƒÂ©menter la dÃƒÂ©tection de changement de rÃƒÂ©gime
        # Pour l'instant, retourner False
        # 
        # Dans une version complÃƒÂ¨te:
        # - Analyser la volatilitÃƒÂ© rÃƒÂ©cente vs historique
        # - DÃƒÂ©tecter des changements de corrÃƒÂ©lations entre actifs
        # - Identifier des changements de volume
        # - DÃƒÂ©tecter des changements de tendance
        
        return False
    
    def update_performance(self, performance: float):
        """
        Met ÃƒÂ  jour la performance actuelle du modÃƒÂ¨le
        
        Args:
            performance: Performance mesurÃƒÂ©e (accuracy, win rate, etc.)
        """
        self.current_performance = performance
        logger.debug(f"Performance mise ÃƒÂ  jour: {performance:.2%}")
    
    def load_current_models(self, model_path: str):
        """
        Charge les modÃƒÂ¨les actuels
        
        Args:
            model_path: Chemin vers les modÃƒÂ¨les
        """
        try:
            self.ensemble = MLEnsemble()
            self.ensemble.load(model_path)
            logger.info(f"Ã¢Å“â€¦ ModÃƒÂ¨les actuels chargÃƒÂ©s: {model_path}")
        except Exception as e:
            logger.error(f"Erreur chargement modÃƒÂ¨les: {e}")
    
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
        """Calcule la prochaine heure de rÃƒÂ©entraÃƒÂ®nement programmÃƒÂ©e"""
        if self.last_retrain is None:
            return "Non programmÃƒÂ© (premier rÃƒÂ©entraÃƒÂ®nement)"
        
        next_retrain = self.last_retrain + timedelta(days=self.retrain_frequency_days)
        next_retrain = next_retrain.replace(hour=self.retrain_hour, minute=0, second=0)
        
        return next_retrain.isoformat()
    
    def get_history(self, limit: int = 10) -> list:
        """
        Retourne l'historique des rÃƒÂ©entraÃƒÂ®nements
        
        Args:
            limit: Nombre max d'entrÃƒÂ©es
            
        Returns:
            Liste des rÃƒÂ©entraÃƒÂ®nements rÃƒÂ©cents
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
    
    # CrÃƒÂ©er l'auto-retrainer
    retrainer = AutoRetrainer(config)
    
    # Status
    print("Ã°Å¸â€œÅ  Status initial:")
    status = retrainer.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test should_retrain
    print(f"\nÃ°Å¸â€Â Should retrain: {retrainer.should_retrain()}")
    
    # Simuler une performance
    retrainer.update_performance(0.55)
    print(f"\nÃ¢Å¡Â Ã¯Â¸Â  Performance dÃƒÂ©gradÃƒÂ©e: {retrainer.should_retrain()}")
    
    # CrÃƒÂ©er des donnÃƒÂ©es de test
    print("\nÃ°Å¸â€œÂ¦ CrÃƒÂ©ation donnÃƒÂ©es de test...")
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
    
    # Test rÃƒÂ©entraÃƒÂ®nement
    print("\nÃ°Å¸Å½Â¯ Test rÃƒÂ©entraÃƒÂ®nement...")
    result = retrainer.retrain(
        trades_data, 
        ohlcv_data, 
        'data/models/test_retrain'
    )
    
    print(f"\nRÃƒÂ©sultat: {result['status']}")
    if 'new_performance' in result:
        print(f"Performance: {result['new_performance']:.2%}")
    
    print("\nÃ¢Å“â€¦ Test terminÃƒÂ©")