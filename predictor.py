"""
ML Predictor pour The Bot
Interface haute performance pour les prÃƒÂ©dictions en temps rÃƒÂ©el
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
import logging

from .feature_engineering import FeatureEngineer
from .ensemble import MLEnsemble

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    PrÃƒÂ©dicteur ML optimisÃƒÂ© pour production
    
    ResponsabilitÃƒÂ©s:
    - Calculer les features
    - Faire les prÃƒÂ©dictions
    - Cacher les rÃƒÂ©sultats
    - Tracker la latence
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 feature_config: Optional[Dict] = None,
                 ensemble_config: Optional[Dict] = None):
        """
        Initialise le prÃƒÂ©dicteur
        
        Args:
            model_path: Chemin vers les modÃƒÂ¨les sauvegardÃƒÂ©s
            feature_config: Configuration du feature engineer
            ensemble_config: Configuration de l'ensemble
        """
        # Feature engineer
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # Ensemble de modÃƒÂ¨les
        self.ensemble = MLEnsemble(ensemble_config)
        
        # Charger les modÃƒÂ¨les si chemin fourni
        if model_path:
            self.load_models(model_path)
        
        # Cache pour ÃƒÂ©viter recalculs
        self.cache = {}
        self.cache_ttl = 60  # 60 secondes
        
        # Statistiques
        self.stats = {
            'total_predictions': 0,
            'avg_latency_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("Ã¢Å“â€¦ ML Predictor initialisÃƒÂ©")
    
    def predict(self, 
               df: pd.DataFrame,
               symbol: str,
               orderbook: Optional[Dict] = None,
               additional_data: Optional[Dict] = None,
               use_cache: bool = True) -> Dict:
        """
        PrÃƒÂ©dit le signal de trading
        
        Args:
            df: DataFrame avec donnÃƒÂ©es OHLCV
            symbol: Symbole tradÃƒÂ©
            orderbook: Orderbook optionnel
            additional_data: DonnÃƒÂ©es supplÃƒÂ©mentaires
            use_cache: Utiliser le cache ou non
            
        Returns:
            Dict avec:
            - signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            - confidence: Niveau de confiance (0-1)
            - latency_ms: Latence de la prÃƒÂ©diction
            - features: Features calculÃƒÂ©es (optionnel)
        """
        start_time = time.time()
        
        try:
            # VÃƒÂ©rifier le cache
            if use_cache:
                cache_key = self._get_cache_key(symbol, df)
                cached = self._get_from_cache(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    return cached
                self.stats['cache_misses'] += 1
            
            # Calculer les features
            features = self.feature_engineer.calculate_features(
                df, 
                orderbook=orderbook,
                additional_data=additional_data
            )
            
            # Prendre la derniÃƒÂ¨re ligne (dernier point)
            X = features[-1]
            
            # PrÃƒÂ©dire
            signal, confidence = self.ensemble.predict(X)
            
            # Calculer la latence
            latency_ms = (time.time() - start_time) * 1000
            
            # RÃƒÂ©sultat
            result = {
                'signal': signal,
                'confidence': confidence,
                'latency_ms': latency_ms,
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol
            }
            
            # Mettre en cache
            if use_cache:
                self._add_to_cache(cache_key, result)
            
            # Mettre ÃƒÂ  jour les stats
            self._update_stats(latency_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur prÃƒÂ©diction pour {symbol}: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'latency_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol
            }
    
    def predict_with_features(self, 
                             df: pd.DataFrame,
                             symbol: str,
                             return_features: bool = True) -> Dict:
        """
        PrÃƒÂ©dit et retourne aussi les features
        
        Args:
            df: DataFrame OHLCV
            symbol: Symbole
            return_features: Si True, inclut les features dans le rÃƒÂ©sultat
            
        Returns:
            Dict avec signal, confidence et features
        """
        result = self.predict(df, symbol, use_cache=False)
        
        if return_features:
            # Recalculer les features (dÃƒÂ©jÃƒÂ  fait dans predict mais on les retourne)
            features = self.feature_engineer.calculate_features(df)
            result['features'] = features[-1]
            result['feature_names'] = self.feature_engineer.get_feature_names()
        
        return result
    
    def batch_predict(self, 
                     data_dict: Dict[str, pd.DataFrame],
                     orderbooks: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
        """
        PrÃƒÂ©dit pour plusieurs symboles en batch
        
        Args:
            data_dict: Dict {symbol: DataFrame}
            orderbooks: Dict {symbol: orderbook}
            
        Returns:
            Dict {symbol: prediction_result}
        """
        results = {}
        
        for symbol, df in data_dict.items():
            orderbook = orderbooks.get(symbol) if orderbooks else None
            result = self.predict(df, symbol, orderbook=orderbook)
            results[symbol] = result
        
        return results
    
    def validate_prediction(self, prediction: Dict) -> bool:
        """
        Valide qu'une prÃƒÂ©diction est utilisable
        
        Args:
            prediction: RÃƒÂ©sultat de predict()
            
        Returns:
            True si valide
        """
        # VÃƒÂ©rifier les champs requis
        if 'signal' not in prediction or 'confidence' not in prediction:
            return False
        
        # VÃƒÂ©rifier les valeurs
        if prediction['signal'] not in [-1, 0, 1]:
            return False
        
        if not (0 <= prediction['confidence'] <= 1):
            return False
        
        # VÃƒÂ©rifier qu'il n'y a pas d'erreur
        if 'error' in prediction:
            return False
        
        return True
    
    def load_models(self, model_path: str):
        """
        Charge les modÃƒÂ¨les depuis un fichier
        
        Args:
            model_path: Chemin vers les modÃƒÂ¨les
        """
        try:
            self.ensemble.load(model_path)
            logger.info(f"Ã¢Å“â€¦ ModÃƒÂ¨les chargÃƒÂ©s: {model_path}")
        except Exception as e:
            logger.error(f"Erreur chargement modÃƒÂ¨les: {e}")
    
    def is_ready(self) -> bool:
        """VÃƒÂ©rifie si le prÃƒÂ©dicteur est prÃƒÂªt ÃƒÂ  faire des prÃƒÂ©dictions"""
        return self.ensemble.is_trained
    
    def _get_cache_key(self, symbol: str, df: pd.DataFrame) -> str:
        """GÃƒÂ©nÃƒÂ¨re une clÃƒÂ© de cache"""
        # Utiliser le timestamp de la derniÃƒÂ¨re bougie
        last_timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'].iloc[-1]
        return f"{symbol}_{last_timestamp}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """RÃƒÂ©cupÃƒÂ¨re du cache si disponible et valide"""
        if key in self.cache:
            item = self.cache[key]
            # VÃƒÂ©rifier TTL
            if (time.time() - item['cached_at']) < self.cache_ttl:
                return item['data']
            else:
                # Supprimer du cache si expirÃƒÂ©
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: Dict):
        """Ajoute au cache"""
        self.cache[key] = {
            'data': data,
            'cached_at': time.time()
        }
        
        # Limiter la taille du cache
        if len(self.cache) > 1000:
            # Supprimer les entrÃƒÂ©es les plus anciennes
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]['cached_at'])[:100]
            for k in oldest_keys:
                del self.cache[k]
    
    def _update_stats(self, latency_ms: float):
        """Met ÃƒÂ  jour les statistiques"""
        self.stats['total_predictions'] += 1
        
        # Moyenne mobile de la latence
        n = self.stats['total_predictions']
        current_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (current_avg * (n-1) + latency_ms) / n
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du prÃƒÂ©dicteur"""
        cache_hit_rate = 0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'total_predictions': self.stats['total_predictions'],
            'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
            'cache_hit_rate': round(cache_hit_rate, 3),
            'cache_size': len(self.cache),
            'is_ready': self.is_ready(),
            'feature_count': self.feature_engineer.get_feature_count()
        }
    
    def clear_cache(self):
        """Vide le cache"""
        self.cache.clear()
        logger.info("Cache vidÃƒÂ©")
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Retourne l'importance des features
        
        Args:
            top_n: Nombre de top features
            
        Returns:
            Dict avec les features et leur importance
        """
        feature_names = self.feature_engineer.get_feature_names()
        return self.ensemble.get_feature_importance(feature_names, top_n)


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du ML Predictor"""
    
    print("\n=== Test ML Predictor ===\n")
    
    # CrÃƒÂ©er des donnÃƒÂ©es de test
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='5min'),
        'open': 50000 + np.cumsum(np.random.randn(n) * 100),
        'high': 50100 + np.cumsum(np.random.randn(n) * 100),
        'low': 49900 + np.cumsum(np.random.randn(n) * 100),
        'close': 50000 + np.cumsum(np.random.randn(n) * 100),
        'volume': np.random.uniform(100, 1000, n)
    })
    
    # CrÃƒÂ©er le prÃƒÂ©dicteur
    predictor = MLPredictor()
    
    # Note: Le prÃƒÂ©dicteur n'est pas entraÃƒÂ®nÃƒÂ©, donc les prÃƒÂ©dictions seront HOLD
    print("Ã¢Å¡Â Ã¯Â¸Â  PrÃƒÂ©dicteur non entraÃƒÂ®nÃƒÂ©, les prÃƒÂ©dictions seront HOLD\n")
    
    # Test de prÃƒÂ©diction
    print("Ã°Å¸â€Â® Test de prÃƒÂ©diction:")
    result = predictor.predict(df, 'BTCUSDT')
    
    signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[result['signal']]
    print(f"  Signal: {signal_name}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    # Test de cache
    print("\nÃ°Å¸â€™Â¾ Test de cache:")
    result1 = predictor.predict(df, 'BTCUSDT', use_cache=True)
    result2 = predictor.predict(df, 'BTCUSDT', use_cache=True)
    
    print(f"  Latence 1ÃƒÂ¨re prÃƒÂ©diction: {result1['latency_ms']:.2f}ms")
    print(f"  Latence 2ÃƒÂ¨me prÃƒÂ©diction (cache): {result2['latency_ms']:.2f}ms")
    
    # Stats
    print("\nÃ°Å¸â€œÅ  Statistiques:")
    stats = predictor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")