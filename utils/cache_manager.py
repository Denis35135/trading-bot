"""
Cache Manager pour The Bot
Gestion du cache Redis et optimisation mÃƒÂ©moire
"""

import redis
import json
import pickle
import time
import psutil
import gc
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gestionnaire de cache avec Redis
    Ãƒâ€°vite les recalculs et optimise les performances
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le cache manager
        
        Args:
            config: Configuration du cache
        """
        self.config = config
        self.max_memory = getattr(config, 'MAX_MEMORY_MB', 2000)  # 2GB par dÃƒÂ©faut
        
        # Connexion Redis
        try:
            self.cache = redis.Redis(
                host=getattr(config, 'REDIS_HOST', 'localhost'),
                port=getattr(config, 'REDIS_PORT', 6379),
                db=getattr(config, 'REDIS_DB', 0),
                decode_responses=False,  # Pour supporter pickle
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connexion
            self.cache.ping()
            self.redis_available = True
            logger.info("Ã¢Å“â€¦ Cache Redis connectÃƒÂ©")
        except Exception as e:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Redis non disponible, cache dÃƒÂ©sactivÃƒÂ©: {e}")
            self.redis_available = False
            self.cache = None
        
        # Cache en mÃƒÂ©moire de secours
        self.memory_cache = {}
        self.cache_timestamps = {}
        
        # Buffer de donnÃƒÂ©es
        self.price_buffer = {}
        self.indicator_buffer = {}
        
        # Statistiques
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        
        logger.info("Cache Manager initialisÃƒÂ©")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        RÃƒÂ©cupÃƒÂ¨re une valeur du cache
        
        Args:
            key: ClÃƒÂ© du cache
            default: Valeur par dÃƒÂ©faut si non trouvÃƒÂ©
            
        Returns:
            Valeur du cache ou default
        """
        try:
            # Essayer Redis d'abord
            if self.redis_available:
                value = self.cache.get(key)
                if value is not None:
                    self.stats['hits'] += 1
                    return pickle.loads(value)
            
            # Sinon, cache mÃƒÂ©moire
            if key in self.memory_cache:
                # VÃƒÂ©rifier expiration
                if key in self.cache_timestamps:
                    if time.time() < self.cache_timestamps[key]:
                        self.stats['hits'] += 1
                        return self.memory_cache[key]
                    else:
                        # ExpirÃƒÂ©
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
            
            self.stats['misses'] += 1
            return default
            
        except Exception as e:
            logger.error(f"Erreur lecture cache {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: int = 60):
        """
        Stocke une valeur dans le cache
        
        Args:
            key: ClÃƒÂ© du cache
            value: Valeur ÃƒÂ  stocker
            ttl: DurÃƒÂ©e de vie en secondes (60s par dÃƒÂ©faut)
        """
        try:
            # Redis
            if self.redis_available:
                serialized = pickle.dumps(value)
                self.cache.setex(key, ttl, serialized)
            
            # Cache mÃƒÂ©moire aussi
            self.memory_cache[key] = value
            self.cache_timestamps[key] = time.time() + ttl
            
            self.stats['sets'] += 1
            
            # VÃƒÂ©rifier utilisation mÃƒÂ©moire
            self._check_memory()
            
        except Exception as e:
            logger.error(f"Erreur ÃƒÂ©criture cache {key}: {e}")
    
    def cache_indicators(self, symbol: str, timeframe: str, indicators: Dict, ttl: int = 60):
        """
        Cache les indicateurs techniques
        
        Args:
            symbol: Symbole (ex: BTCUSDT)
            timeframe: Timeframe (ex: 5m)
            indicators: Dictionnaire des indicateurs
            ttl: DurÃƒÂ©e de vie (60s par dÃƒÂ©faut)
        """
        key = f"indicators:{symbol}:{timeframe}"
        self.set(key, indicators, ttl)
    
    def get_indicators(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re les indicateurs du cache
        
        Args:
            symbol: Symbole
            timeframe: Timeframe
            
        Returns:
            Dictionnaire des indicateurs ou None
        """
        key = f"indicators:{symbol}:{timeframe}"
        return self.get(key)
    
    def cache_market_data(self, symbol: str, data: Dict, ttl: int = 10):
        """
        Cache les donnÃƒÂ©es de marchÃƒÂ©
        
        Args:
            symbol: Symbole
            data: DonnÃƒÂ©es (prix, volume, etc.)
            ttl: DurÃƒÂ©e de vie (10s par dÃƒÂ©faut pour donnÃƒÂ©es temps rÃƒÂ©el)
        """
        key = f"market:{symbol}"
        self.set(key, data, ttl)
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re les donnÃƒÂ©es de marchÃƒÂ© du cache
        
        Args:
            symbol: Symbole
            
        Returns:
            DonnÃƒÂ©es de marchÃƒÂ© ou None
        """
        key = f"market:{symbol}"
        return self.get(key)
    
    def cache_ml_prediction(self, symbol: str, prediction: Dict, ttl: int = 300):
        """
        Cache les prÃƒÂ©dictions ML
        
        Args:
            symbol: Symbole
            prediction: PrÃƒÂ©diction ML
            ttl: DurÃƒÂ©e de vie (5min par dÃƒÂ©faut)
        """
        key = f"ml:prediction:{symbol}"
        self.set(key, prediction, ttl)
    
    def get_ml_prediction(self, symbol: str) -> Optional[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re la prÃƒÂ©diction ML du cache
        
        Args:
            symbol: Symbole
            
        Returns:
            PrÃƒÂ©diction ou None
        """
        key = f"ml:prediction:{symbol}"
        return self.get(key)
    
    def add_to_price_buffer(self, symbol: str, price: float, timestamp: float):
        """
        Ajoute un prix au buffer (pour calculs rapides)
        
        Args:
            symbol: Symbole
            price: Prix
            timestamp: Timestamp
        """
        if symbol not in self.price_buffer:
            self.price_buffer[symbol] = []
        
        self.price_buffer[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Garder seulement les 1000 derniers
        if len(self.price_buffer[symbol]) > 1000:
            self.price_buffer[symbol] = self.price_buffer[symbol][-1000:]
    
    def get_price_buffer(self, symbol: str, max_items: int = 1000) -> List[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re le buffer de prix
        
        Args:
            symbol: Symbole
            max_items: Nombre max d'items
            
        Returns:
            Liste des prix rÃƒÂ©cents
        """
        if symbol not in self.price_buffer:
            return []
        
        return self.price_buffer[symbol][-max_items:]
    
    def clear_symbol_cache(self, symbol: str):
        """
        Nettoie tout le cache d'un symbole
        
        Args:
            symbol: Symbole ÃƒÂ  nettoyer
        """
        patterns = [
            f"indicators:{symbol}:*",
            f"market:{symbol}",
            f"ml:prediction:{symbol}"
        ]
        
        for pattern in patterns:
            try:
                if self.redis_available:
                    keys = self.cache.keys(pattern)
                    if keys:
                        self.cache.delete(*keys)
            except Exception as e:
                logger.error(f"Erreur nettoyage cache {pattern}: {e}")
        
        # Nettoyer buffer mÃƒÂ©moire
        if symbol in self.price_buffer:
            del self.price_buffer[symbol]
        if symbol in self.indicator_buffer:
            del self.indicator_buffer[symbol]
    
    def _check_memory(self):
        """
        VÃƒÂ©rifie l'utilisation mÃƒÂ©moire et nettoie si nÃƒÂ©cessaire
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â MÃƒÂ©moire ÃƒÂ©levÃƒÂ©e: {memory_mb:.0f}MB > {self.max_memory}MB")
                self.cleanup()
            
        except Exception as e:
            logger.error(f"Erreur vÃƒÂ©rification mÃƒÂ©moire: {e}")
    
    def cleanup(self):
        """
        Nettoie les anciennes donnÃƒÂ©es pour libÃƒÂ©rer de la mÃƒÂ©moire
        """
        logger.info("Ã°Å¸Â§Â¹ Nettoyage mÃƒÂ©moire en cours...")
        
        # Nettoyer buffers (garder seulement 1000 derniers)
        for symbol in list(self.price_buffer.keys()):
            if len(self.price_buffer[symbol]) > 1000:
                self.price_buffer[symbol] = self.price_buffer[symbol][-1000:]
        
        # Nettoyer cache mÃƒÂ©moire expirÃƒÂ©
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time > timestamp
        ]
        
        for key in expired_keys:
            if key in self.memory_cache:
                del self.memory_cache[key]
            del self.cache_timestamps[key]
            self.stats['evictions'] += 1
        
        # Force garbage collection
        gc.collect()
        
        # VÃƒÂ©rifier nouvelle utilisation
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Ã¢Å“â€¦ Nettoyage terminÃƒÂ© - MÃƒÂ©moire: {memory_mb:.0f}MB")
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques du cache
        
        Returns:
            Dictionnaire des stats
        """
        hit_rate = 0
        if self.stats['hits'] + self.stats['misses'] > 0:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
        
        # MÃƒÂ©moire actuelle
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.1%}",
            'memory_mb': f"{memory_mb:.0f}",
            'memory_usage': f"{(memory_mb/self.max_memory)*100:.1f}%",
            'redis_available': self.redis_available,
            'memory_cache_size': len(self.memory_cache),
            'price_buffers': len(self.price_buffer)
        }
    
    def clear_all(self):
        """Nettoie tout le cache"""
        try:
            if self.redis_available:
                self.cache.flushdb()
            
            self.memory_cache.clear()
            self.cache_timestamps.clear()
            self.price_buffer.clear()
            self.indicator_buffer.clear()
            
            logger.info("Ã°Å¸Â§Â¹ Cache entiÃƒÂ¨rement nettoyÃƒÂ©")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage complet: {e}")
    
    def close(self):
        """Ferme les connexions"""
        try:
            if self.redis_available and self.cache:
                self.cache.close()
            logger.info("Cache Manager fermÃƒÂ©")
        except Exception as e:
            logger.error(f"Erreur fermeture cache: {e}")


def cached(ttl: int = 60, key_prefix: str = ""):
    """
    DÃƒÂ©corateur pour mettre en cache les rÃƒÂ©sultats de fonctions
    
    Args:
        ttl: DurÃƒÂ©e de vie du cache
        key_prefix: PrÃƒÂ©fixe de la clÃƒÂ©
        
    Usage:
        @cached(ttl=60, key_prefix="strategy")
        def calculate_signals(symbol, data):
            # Calculs lourds...
            return signals
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Construire clÃƒÂ© de cache
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Essayer de rÃƒÂ©cupÃƒÂ©rer du cache
            if hasattr(self, 'cache_manager'):
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Calculer et mettre en cache
            result = func(self, *args, **kwargs)
            
            if hasattr(self, 'cache_manager'):
                self.cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class MemoryManager:
    """
    Gestionnaire de mÃƒÂ©moire pour ÃƒÂ©viter les memory leaks
    """
    
    def __init__(self, max_memory: int = 2000):
        """
        Initialise le memory manager
        
        Args:
            max_memory: MÃƒÂ©moire max en MB
        """
        self.max_memory = max_memory
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
        logger.info(f"Memory Manager initialisÃƒÂ© (max: {max_memory}MB)")
    
    def check_memory(self) -> bool:
        """
        VÃƒÂ©rifie si la mÃƒÂ©moire dÃƒÂ©passe la limite
        
        Returns:
            True si mÃƒÂ©moire OK, False si limite dÃƒÂ©passÃƒÂ©e
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Limite mÃƒÂ©moire dÃƒÂ©passÃƒÂ©e: {memory_mb:.0f}MB > {self.max_memory}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur vÃƒÂ©rification mÃƒÂ©moire: {e}")
            return True
    
    def auto_cleanup_if_needed(self, cleanup_func):
        """
        Nettoie automatiquement si nÃƒÂ©cessaire
        
        Args:
            cleanup_func: Fonction de nettoyage ÃƒÂ  appeler
        """
        current_time = time.time()
        
        # VÃƒÂ©rifier si cleanup nÃƒÂ©cessaire
        if not self.check_memory() or (current_time - self.last_cleanup > self.cleanup_interval):
            cleanup_func()
            self.last_cleanup = current_time
            gc.collect()
    
    def get_memory_info(self) -> Dict:
        """
        Retourne les infos mÃƒÂ©moire
        
        Returns:
            Dictionnaire avec les infos
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'usage_percent': f"{(memory_info.rss / 1024 / 1024 / self.max_memory) * 100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration info mÃƒÂ©moire: {e}")
            return {}
