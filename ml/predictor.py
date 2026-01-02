"""
ML Predictor pour The Bot
Interface haute performance pour les predictions en temps reel
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
    Predicteur ML optimise pour production
    
    Responsabilites:
    - Calculer les features
    - Faire les predictions
    - Cacher les resultats
    - Tracker la latence
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 feature_config: Optional[Dict] = None,
                 ensemble_config: Optional[Dict] = None):
        """
        Initialise le predicteur
        
        Args:
            model_path: Chemin vers les modeles sauvegardes
            feature_config: Configuration du feature engineer
            ensemble_config: Configuration de l'ensemble
        """
        # Feature engineer
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # Ensemble de modeles
        self.ensemble = MLEnsemble(ensemble_config)
        
        # Charger les modeles si chemin fourni
        if model_path:
            self.load_models(model_path)
        
        # Cache pour eviter recalculs
        self.cache = {}
        self.cache_ttl = 60  # 60 secondes
        
        # Statistiques
        self.stats = {
            'total_predictions': 0,
            'avg_latency_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("""| ML Predictor initialise")
    
    def predict(self, 
    """
    df: pd.DataFrame,
    """
               symbol: str,
               orderbook: Optional[Dict] = None,
               additional_data: Optional[Dict] = None,
               use_cache: bool = True) -> Dict:
        """
        Predit le signal de trading
        
        Args:
            df: DataFrame avec donnees OHLCV
            symbol: Symbole trade
            orderbook: Orderbook optionnel
            additional_data: Donnees supplementaires
            use_cache: Utiliser le cache ou non
            
        Returns:
            Dict avec:
            - signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            - confidence: Niveau de confiance (0-1)
            - latency_ms: Latence de la prediction
            - features: Features calculees (optionnel)
        """
        start_time = time.time()
        
        try:
            # Verifier le cache
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
            
            # Prendre la derniere ligne (dernier point)
            X = features[-1]
            
            # Predire
            signal, confidence = self.ensemble.predict(X)
            
            # Calculer la latence
            latency_ms = (time.time() - start_time) * 1000
            
            # Resultat
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
            
            # Mettre  jour les stats
            self._update_stats(latency_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur prediction pour {symbol}: {e}")
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
        Predit et retourne aussi les features
        
        Args:
            df: DataFrame OHLCV
            symbol: Symbole
            return_features: Si True, inclut les features dans le resultat
            
        Returns:
            Dict avec signal, confidence et features
        """
        result = self.predict(df, symbol, use_cache=False)
        
        if return_features:
            # Recalculer les features (dej fait dans predict mais on les retourne)
            features = self.feature_engineer.calculate_features(df)
            result['features'] = features[-1]
            result['feature_names'] = self.feature_engineer.get_feature_names()
        
        return result
    
    def batch_predict(self, 
                     data_dict: Dict[str, pd.DataFrame],
                     orderbooks: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
        """
        Predit pour plusieurs symboles en batch
        
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
        Valide qu'une prediction est utilisable
        
        Args:
            prediction: Resultat de predict()
            
        Returns:
            True si valide
        """
        # Verifier les champs requis
        if 'signal' not in prediction or 'confidence' not in prediction:
            return False
        
        # Verifier les valeurs
        if prediction['signal'] not in [-1, 0, 1]:
            return False
        
        if not (0 <= prediction['confidence'] <= 1):
            return False
        
        # Verifier qu'il n'y a pas d'erreur
        if 'error' in prediction:
            return False
        
        return True
    
    def load_models(self, model_path: str):
        """
        Charge les modeles depuis un fichier
        
        Args:
            model_path: Chemin vers les modeles
        """
        try:
            self.ensemble.load(model_path)
            logger.info(f"""| Modeles charges: {model_path}")
        except Exception as e:
            logger.error(f"Erreur chargement modeles: {e}")
    
    def is_ready(self) -> bool:
        """Verifie si le predicteur est pret  faire des predictions"""
        return self.ensemble.is_trained
    
    def _get_cache_key(self, symbol: str, df: pd.DataFrame) -> str:
        """Genere une cle de cache"""
        # Utiliser le timestamp de la derniere bougie
        last_timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'].iloc[-1]
        return f"{symbol}_{last_timestamp}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Recupere du cache si disponible et valide"""
        if key in self.cache:
            item = self.cache[key]
            # Verifier TTL
            if (time.time() - item['cached_at']) < self.cache_ttl:
                return item['data']
            else:
                # Supprimer du cache si expire
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
            # Supprimer les entrees les plus anciennes
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]['cached_at'])[:100]
            for k in oldest_keys:
                del self.cache[k]
    
    def _update_stats(self, latency_ms: float):
        """Met  jour les statistiques"""
        self.stats['total_predictions'] += 1
        
        # Moyenne mobile de la latence
        n = self.stats['total_predictions']
        current_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (current_avg * (n-1) + latency_ms) / n
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du predicteur"""
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
        logger.info("Cache vide")
    
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
    
    # Creer des donnees de test
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
    
    # Creer le predicteur
    predictor = MLPredictor()
    
    # Note: Le predicteur n'est pas entrane, donc les predictions seront HOLD
    print("  Predicteur non entrane, les predictions seront HOLD\n")
    
    # Test de prediction
    print("" Test de prediction:")
    result = predictor.predict(df, 'BTCUSDT')
    
    signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[result['signal']]
    print(f"  Signal: {signal_name}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    # Test de cache
    print("\n' Test de cache:")
    result1 = predictor.predict(df, 'BTCUSDT', use_cache=True)
    result2 = predictor.predict(df, 'BTCUSDT', use_cache=True)
    
    print(f"  Latence 1ere prediction: {result1['latency_ms']:.2f}ms")
    print(f"  Latence 2eme prediction (cache): {result2['latency_ms']:.2f}ms")
    
    # Stats
    print("\n" Statistiques:")
    stats = predictor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n""| Tests termines")
