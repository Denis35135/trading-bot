"""
Market Data Thread pour The Bot
Thread de collecte et distribution des donnees de marche
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import deque
import threading
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataThread:
    """
    Thread de collecte des donnees de marche
    
    Responsabilites:
    - Recevoir les donnees WebSocket
    - Maintenir des buffers de prix
    - Calculer les indicateurs techniques
    - Distribuer les donnees aux strategies
    - Gerer les reconnexions
    """
    
    def __init__(self, bot_instance, config: Dict):
        """
        Initialise le thread market data
        
        Args:
            bot_instance: Instance du bot principal
            config: Configuration
        """
        self.bot = bot_instance
        self.config = config
        self.is_running = False
        self.thread = None
        
        # Configuration
        self.update_interval = getattr(config, 'UPDATE_INTERVAL', 1)  # 1 seconde
        self.buffer_size = getattr(config, 'BUFFER_SIZE', 5000)
        self.symbols_to_watch = []
        
        # Buffers de donnees
        self.price_buffers = {}  # {symbol: deque of prices}
        self.orderbook_cache = {}  # {symbol: latest orderbook}
        self.ticker_cache = {}  # {symbol: latest ticker}
        self.klines_cache = {}  # {symbol: DataFrame}
        
        # Callbacks
        self.data_callbacks = []
        
        # Statistiques
        self.stats = {
            'ticks_received': 0,
            'data_updates_sent': 0,
            'last_update': None,
            'symbols_active': 0,
            'reconnections': 0
        }
        
        logger.info("Market Data Thread initialise")
    
    def start(self):
        """Demarre le thread"""
        if self.is_running:
            logger.warning("Market Data Thread deja en cours")
            return
        
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MarketDataThread"
        )
        self.thread.start()
        
        logger.info("Market Data Thread demarre")
    
    def stop(self):
        """Arrete le thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Market Data Thread arrete")
    
    def set_symbols(self, symbols: List[str]):
        """
        Definit les symboles a surveiller
        
        Args:
            symbols: Liste des symboles
        """
        self.symbols_to_watch = symbols
        self.stats['symbols_active'] = len(symbols)
        
        # Initialiser les buffers si necessaire
        for symbol in symbols:
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
        
        logger.info(f"Surveillance de {len(symbols)} symboles: {symbols}")
    
    def register_callback(self, callback: Callable):
        """
        Enregistre un callback pour recevoir les donnees
        
        Args:
            callback: Fonction appelee avec les donnees
        """
        self.data_callbacks.append(callback)
        logger.debug(f"Callback enregistre: {callback.__name__}")
    
    def _run(self):
        """Boucle principale du thread"""
        logger.info("Market Data Thread running...")
        
        while self.is_running:
            try:
                # Mettre a jour les symboles depuis le scanner
                self._update_symbols_from_scanner()
                
                # Collecter les donnees pour chaque symbole
                for symbol in self.symbols_to_watch:
                    if not self.is_running:
                        break
                    
                    self._collect_symbol_data(symbol)
                
                # Pause entre cycles
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans market data thread: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("Market Data Thread termine")
    
    def _update_symbols_from_scanner(self):
        """Met a jour la liste des symboles depuis le scanner"""
        try:
            if hasattr(self.bot, 'market_scanner'):
                top_symbols = self.bot.market_scanner.get_top_symbols()
                if top_symbols and top_symbols != self.symbols_to_watch:
                    logger.info(f"Mise a jour symboles: {top_symbols}")
                    self.set_symbols(top_symbols)
        except Exception as e:
            logger.error(f"Erreur mise a jour symboles: {e}")
    
    def _collect_symbol_data(self, symbol: str):
        """
        Collecte les donnees pour un symbole
        
        Args:
            symbol: Symbole a collecter
        """
        try:
            # 1. Recuperer le ticker
            ticker = self._get_ticker(symbol)
            if not ticker:
                return
            
            # 2. Ajouter au buffer de prix
            price = ticker.get('price')
            if price:
                self.price_buffers[symbol].append({
                    'price': price,
                    'timestamp': time.time(),
                    'volume': ticker.get('volume', 0)
                })
                self.stats['ticks_received'] += 1
            
            # 3. Recuperer l'orderbook (tous les 5 ticks)
            if self.stats['ticks_received'] % 5 == 0:
                orderbook = self._get_orderbook(symbol)
                if orderbook:
                    self.orderbook_cache[symbol] = orderbook
            
            # 4. Recuperer les klines (tous les 10 ticks)
            if self.stats['ticks_received'] % 10 == 0:
                klines = self._get_klines(symbol)
                if klines is not None and not klines.empty:
                    self.klines_cache[symbol] = klines
                    
                    # Calculer les indicateurs
                    self._calculate_indicators(symbol, klines)
                    
                    # Distribuer les donnees
                    self._distribute_market_data(symbol)
        
        except Exception as e:
            logger.error(f"Erreur collecte {symbol}: {e}")
    
    def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Recupere le ticker d'un symbole
        
        Args:
            symbol: Symbole
            
        Returns:
            Dict avec ticker ou None
        """
        try:
            if not hasattr(self.bot, 'exchange'):
                return None
            
            ticker = self.bot.exchange.get_symbol_ticker(symbol)
            if ticker:
                self.ticker_cache[symbol] = ticker
                return ticker
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur recuperation ticker {symbol}: {e}")
            return None
    
    def _get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Recupere l'orderbook d'un symbole
        
        Args:
            symbol: Symbole
            limit: Nombre de niveaux
            
        Returns:
            Dict avec orderbook ou None
        """
        try:
            if not hasattr(self.bot, 'exchange'):
                return None
            
            orderbook = self.bot.exchange.get_orderbook(symbol, limit=limit)
            return orderbook
            
        except Exception as e:
            logger.debug(f"Erreur recuperation orderbook {symbol}: {e}")
            return None
    
    def _get_klines(self, symbol: str, interval: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Recupere les klines d'un symbole
        
        Args:
            symbol: Symbole
            interval: Intervalle (5m, 15m, etc.)
            limit: Nombre de klines
            
        Returns:
            DataFrame ou None
        """
        try:
            if not hasattr(self.bot, 'exchange'):
                return None
            
            df = self.bot.exchange.get_klines(symbol, interval, limit=limit)
            return df
            
        except Exception as e:
            logger.debug(f"Erreur recuperation klines {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, symbol: str, df: pd.DataFrame):
        """
        Calcule les indicateurs techniques
        
        Args:
            symbol: Symbole
            df: DataFrame avec OHLCV
        """
        try:
            from utils.indicators import TechnicalIndicators
            
            # Calculer tous les indicateurs
            df_with_indicators = TechnicalIndicators.calculate_all(df)
            
            # Mettre a jour le cache
            self.klines_cache[symbol] = df_with_indicators
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs {symbol}: {e}")
    
    def _distribute_market_data(self, symbol: str):
        """
        Distribue les donnees de marche
        
        Args:
            symbol: Symbole
        """
        try:
            # Preparer le package de donnees
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'ticker': self.ticker_cache.get(symbol),
                'orderbook': self.orderbook_cache.get(symbol),
                'df': self.klines_cache.get(symbol),
                'price_buffer': list(self.price_buffers.get(symbol, []))
            }
            
            # Envoyer au strategy manager
            if hasattr(self.bot, 'strategy_manager'):
                self.bot.strategy_manager.process_market_data(symbol, market_data)
            
            # Appeler les callbacks
            for callback in self.data_callbacks:
                try:
                    callback(symbol, market_data)
                except Exception as e:
                    logger.error(f"Erreur callback: {e}")
            
            self.stats['data_updates_sent'] += 1
            self.stats['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Erreur distribution donnees {symbol}: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Retourne le dernier prix d'un symbole
        
        Args:
            symbol: Symbole
            
        Returns:
            Prix ou None
        """
        ticker = self.ticker_cache.get(symbol)
        if ticker:
            return ticker.get('price')
        
        # Fallback: buffer de prix
        if symbol in self.price_buffers and self.price_buffers[symbol]:
            return self.price_buffers[symbol][-1]['price']
        
        return None
    
    def get_price_buffer(self, symbol: str, max_items: int = 100) -> List[Dict]:
        """
        Retourne le buffer de prix d'un symbole
        
        Args:
            symbol: Symbole
            max_items: Nombre max d'items
            
        Returns:
            Liste des prix recents
        """
        if symbol not in self.price_buffers:
            return []
        
        buffer = list(self.price_buffers[symbol])
        return buffer[-max_items:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques
        
        Returns:
            Dict avec stats
        """
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['buffer_sizes'] = {
            symbol: len(buffer) 
            for symbol, buffer in self.price_buffers.items()
        }
        stats['symbols_watched'] = len(self.symbols_to_watch)
        
        return stats
    
    def clear_buffers(self):
        """Nettoie tous les buffers"""
        self.price_buffers.clear()
        self.orderbook_cache.clear()
        self.ticker_cache.clear()
        self.klines_cache.clear()
        logger.info("Buffers nettoyes")


class MarketDataCollector:
    """
    Collecteur de donnees de marche simplifie
    Peut etre utilise independamment du thread
    """
    
    def __init__(self, exchange_client):
        """
        Initialise le collecteur
        
        Args:
            exchange_client: Client d'exchange
        """
        self.exchange = exchange_client
        self.cache = {}
        self.last_update = {}
    
    def collect_symbol_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Collecte un snapshot complet pour un symbole
        
        Args:
            symbol: Symbole
            
        Returns:
            Dict avec toutes les donnees
        """
        try:
            snapshot = {
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
            # Ticker
            ticker = self.exchange.get_symbol_ticker(symbol)
            if ticker:
                snapshot['ticker'] = ticker
                snapshot['price'] = ticker.get('price')
                snapshot['volume'] = ticker.get('volume')
            
            # Orderbook
            orderbook = self.exchange.get_orderbook(symbol, limit=20)
            if orderbook:
                snapshot['orderbook'] = orderbook
                
                # Calculer spread
                if orderbook.get('bids') and orderbook.get('asks'):
                    best_bid = orderbook['bids'][0][0]
                    best_ask = orderbook['asks'][0][0]
                    snapshot['spread'] = best_ask - best_bid
                    snapshot['spread_pct'] = (best_ask - best_bid) / best_bid
            
            # Klines
            df = self.exchange.get_klines(symbol, '5m', limit=100)
            if df is not None and not df.empty:
                snapshot['df'] = df
                
                # Calculer indicateurs
                from utils.indicators import TechnicalIndicators
                df_with_indicators = TechnicalIndicators.calculate_all(df)
                snapshot['df'] = df_with_indicators
                
                # Metriques rapides
                snapshot['volatility'] = df['close'].pct_change().std()
                snapshot['trend'] = 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down'
            
            # Mettre en cache
            self.cache[symbol] = snapshot
            self.last_update[symbol] = time.time()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Erreur collecte snapshot {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def collect_multiple(self, symbols: List[str], parallel: bool = False) -> Dict[str, Dict]:
        """
        Collecte les donnees pour plusieurs symboles
        
        Args:
            symbols: Liste des symboles
            parallel: Collecte parallele (plus rapide)
            
        Returns:
            Dict {symbol: snapshot}
        """
        results = {}
        
        if parallel:
            # TODO: Implementer collecte parallele avec ThreadPoolExecutor
            pass
        else:
            for symbol in symbols:
                results[symbol] = self.collect_symbol_snapshot(symbol)
        
        return results
    
    def get_cached(self, symbol: str, max_age: int = 10) -> Optional[Dict]:
        """
        Recupere les donnees en cache
        
        Args:
            symbol: Symbole
            max_age: Age max du cache en secondes
            
        Returns:
            Donnees en cache ou None
        """
        if symbol not in self.cache:
            return None
        
        # Verifier l'age
        if symbol in self.last_update:
            age = time.time() - self.last_update[symbol]
            if age > max_age:
                return None
        
        return self.cache[symbol]
