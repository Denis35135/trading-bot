"""
Market Data pour The Bot
Gestion des donnÃƒÂ©es de marchÃƒÂ© en temps rÃƒÂ©el
"""

import time
from typing import Dict, Optional, List, Callable
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketData:
    """
    Gestionnaire de donnÃƒÂ©es de marchÃƒÂ©
    
    ResponsabilitÃƒÂ©s:
    - RÃƒÂ©cupÃƒÂ©rer les donnÃƒÂ©es OHLCV
    - Maintenir un buffer de donnÃƒÂ©es rÃƒÂ©centes
    - Fournir les donnÃƒÂ©es aux stratÃƒÂ©gies
    - Calculer des agrÃƒÂ©gations rapides
    """
    
    def __init__(self, exchange_client, config: Optional[Dict] = None):
        """
        Initialise le market data handler
        
        Args:
            exchange_client: Client Binance
            config: Configuration
        """
        self.client = exchange_client
        self.config = config or {}
        
        # ParamÃƒÂ¨tres
        self.buffer_size = self.config.get('buffer_size', 5000)
        self.default_interval = self.config.get('default_interval', '5m')
        self.cache_ttl = self.config.get('cache_ttl', 60)  # secondes
        
        # Buffers de donnÃƒÂ©es par symbole
        self.data_buffers = {}  # {symbol: deque of candles}
        self.last_update = {}   # {symbol: timestamp}
        
        # Cache
        self.cache = {}  # {key: (data, timestamp)}
        
        # Callbacks pour updates
        self.on_data_update: Optional[Callable] = None
        
        # Statistiques
        self.stats = {
            'total_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'symbols_tracked': 0
        }
        
        logger.info(f"Ã¢Å“â€¦ Market Data initialisÃƒÂ© (buffer: {self.buffer_size})")
    
    def get_klines(self, 
                   symbol: str,
                   interval: str = '5m',
                   limit: int = 500,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        RÃƒÂ©cupÃƒÂ¨re les klines (OHLCV)
        
        Args:
            symbol: Symbole
            interval: Intervalle (1m, 5m, 15m, 1h, etc.)
            limit: Nombre de bougies
            use_cache: Utiliser le cache ou non
            
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # VÃƒÂ©rifier le cache
        if use_cache and cache_key in self.cache:
            data, cached_at = self.cache[cache_key]
            age = time.time() - cached_at
            
            if age < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return data.copy()
        
        self.stats['cache_misses'] += 1
        
        try:
            # RÃƒÂ©cupÃƒÂ©rer depuis l'exchange
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convertir en DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Garder seulement les colonnes importantes
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convertir les types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Mettre en cache
            self.cache[cache_key] = (df, time.time())
            
            # Mettre ÃƒÂ  jour le buffer
            self._update_buffer(symbol, df)
            
            self.stats['total_updates'] += 1
            
            return df
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur rÃƒÂ©cupÃƒÂ©ration klines {symbol}: {e}")
            # Retourner DataFrame vide en cas d'erreur
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_ticker(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re le ticker (prix actuel, volume, etc.)
        
        Args:
            symbol: Symbole
            use_cache: Utiliser le cache
            
        Returns:
            Dict avec les infos du ticker
        """
        cache_key = f"ticker_{symbol}"
        
        # Cache trÃƒÂ¨s court pour les tickers (5 secondes)
        if use_cache and cache_key in self.cache:
            data, cached_at = self.cache[cache_key]
            if time.time() - cached_at < 5:
                self.stats['cache_hits'] += 1
                return data
        
        self.stats['cache_misses'] += 1
        
        try:
            ticker = self.client.get_symbol_ticker(symbol)
            
            # Mettre en cache
            self.cache[cache_key] = (ticker, time.time())
            
            return ticker
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur rÃƒÂ©cupÃƒÂ©ration ticker {symbol}: {e}")
            return None
    
    def get_orderbook(self, symbol: str, limit: int = 10) -> Optional[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re l'orderbook
        
        Args:
            symbol: Symbole
            limit: Profondeur (5, 10, 20, 50, 100, etc.)
            
        Returns:
            Dict avec bids et asks
        """
        try:
            orderbook = self.client.get_orderbook(symbol, limit=limit)
            return orderbook
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur rÃƒÂ©cupÃƒÂ©ration orderbook {symbol}: {e}")
            return None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> Optional[List]:
        """
        RÃƒÂ©cupÃƒÂ¨re les trades rÃƒÂ©cents
        
        Args:
            symbol: Symbole
            limit: Nombre de trades
            
        Returns:
            Liste des trades rÃƒÂ©cents
        """
        try:
            trades = self.client.get_recent_trades(symbol, limit=limit)
            return trades
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur rÃƒÂ©cupÃƒÂ©ration trades {symbol}: {e}")
            return None
    
    def _update_buffer(self, symbol: str, df: pd.DataFrame):
        """
        Met ÃƒÂ  jour le buffer de donnÃƒÂ©es pour un symbole
        
        Args:
            symbol: Symbole
            df: DataFrame avec les nouvelles donnÃƒÂ©es
        """
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.stats['symbols_tracked'] += 1
        
        buffer = self.data_buffers[symbol]
        
        # Ajouter les nouvelles bougies
        for _, row in df.iterrows():
            buffer.append(row.to_dict())
        
        self.last_update[symbol] = datetime.now()
        
        # Callback
        if self.on_data_update:
            self.on_data_update(symbol, df)
    
    def get_from_buffer(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        RÃƒÂ©cupÃƒÂ¨re les donnÃƒÂ©es depuis le buffer
        
        Args:
            symbol: Symbole
            limit: Nombre de bougies (None = toutes)
            
        Returns:
            DataFrame
        """
        if symbol not in self.data_buffers:
            return pd.DataFrame()
        
        buffer = self.data_buffers[symbol]
        
        if limit:
            data = list(buffer)[-limit:]
        else:
            data = list(buffer)
        
        return pd.DataFrame(data)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        RÃƒÂ©cupÃƒÂ¨re le dernier prix
        
        Args:
            symbol: Symbole
            
        Returns:
            Prix ou None
        """
        ticker = self.get_ticker(symbol)
        if ticker and 'price' in ticker:
            return float(ticker['price'])
        return None
    
    def calculate_vwap(self, symbol: str, periods: int = 20) -> Optional[float]:
        """
        Calcule le VWAP sur N pÃƒÂ©riodes
        
        Args:
            symbol: Symbole
            periods: Nombre de pÃƒÂ©riodes
            
        Returns:
            VWAP ou None
        """
        df = self.get_from_buffer(symbol, limit=periods)
        
        if df.empty or len(df) < periods:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        
        return float(vwap)
    
    def calculate_volatility(self, symbol: str, periods: int = 20) -> Optional[float]:
        """
        Calcule la volatilitÃƒÂ© sur N pÃƒÂ©riodes
        
        Args:
            symbol: Symbole
            periods: Nombre de pÃƒÂ©riodes
            
        Returns:
            VolatilitÃƒÂ© (ÃƒÂ©cart-type des returns) ou None
        """
        df = self.get_from_buffer(symbol, limit=periods + 1)
        
        if df.empty or len(df) < periods + 1:
            return None
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        return float(volatility)
    
    def get_price_change(self, symbol: str, periods: int = 1) -> Optional[float]:
        """
        Calcule le changement de prix sur N pÃƒÂ©riodes
        
        Args:
            symbol: Symbole
            periods: Nombre de pÃƒÂ©riodes
            
        Returns:
            Changement en % ou None
        """
        df = self.get_from_buffer(symbol, limit=periods + 1)
        
        if df.empty or len(df) < periods + 1:
            return None
        
        old_price = df['close'].iloc[0]
        new_price = df['close'].iloc[-1]
        
        if old_price == 0:
            return None
        
        change = (new_price - old_price) / old_price
        
        return float(change)
    
    def clear_cache(self):
        """Vide le cache"""
        self.cache.clear()
        logger.info("Ã°Å¸Â§Â¹ Cache vidÃƒÂ©")
    
    def clear_buffer(self, symbol: Optional[str] = None):
        """
        Vide le buffer
        
        Args:
            symbol: Symbole spÃƒÂ©cifique ou None pour tous
        """
        if symbol:
            if symbol in self.data_buffers:
                self.data_buffers[symbol].clear()
                logger.info(f"Ã°Å¸Â§Â¹ Buffer vidÃƒÂ©: {symbol}")
        else:
            self.data_buffers.clear()
            logger.info("Ã°Å¸Â§Â¹ Tous les buffers vidÃƒÂ©s")
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        cache_hit_rate = 0
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_requests > 0:
            cache_hit_rate = self.stats['cache_hits'] / total_requests
        
        return {
            'symbols_tracked': self.stats['symbols_tracked'],
            'total_updates': self.stats['total_updates'],
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'total_buffered_candles': sum(len(buf) for buf in self.data_buffers.values())
        }
    
    def get_symbols_tracked(self) -> List[str]:
        """Retourne la liste des symboles trackÃƒÂ©s"""
        return list(self.data_buffers.keys())


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Market Data"""
    
    print("\n=== Test Market Data ===\n")
    
    # Mock du client
    class MockBinanceClient:
        def get_historical_klines(self, symbol, interval, limit):
            # GÃƒÂ©nÃƒÂ©rer des donnÃƒÂ©es fictives
            now = int(time.time() * 1000)
            klines = []
            for i in range(limit):
                timestamp = now - (limit - i) * 300000  # 5 min intervals
                klines.append([
                    timestamp,
                    50000 + np.random.randn() * 100,  # open
                    50100 + np.random.randn() * 100,  # high
                    49900 + np.random.randn() * 100,  # low
                    50000 + np.random.randn() * 100,  # close
                    np.random.uniform(100, 1000),     # volume
                    timestamp + 299999,
                    0, 0, 0, 0, 0
                ])
            return klines
        
        def get_symbol_ticker(self, symbol):
            return {
                'symbol': symbol,
                'price': 50000.0,
                'volume': 1000000.0
            }
        
        def get_orderbook(self, symbol, limit):
            return {
                'bids': [[50000, 1.0], [49999, 2.0]],
                'asks': [[50001, 1.0], [50002, 2.0]]
            }
        
        def get_recent_trades(self, symbol, limit):
            return [{'price': 50000, 'qty': 0.1, 'time': int(time.time() * 1000)}]
    
    # CrÃƒÂ©er le market data
    client = MockBinanceClient()
    market_data = MarketData(client, {'buffer_size': 1000})
    
    # Test 1: RÃƒÂ©cupÃƒÂ©ration klines
    print("1Ã¯Â¸ÂÃ¢Æ’Â£ Test klines:")
    df = market_data.get_klines('BTCUSDT', interval='5m', limit=100)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Last price: ${df['close'].iloc[-1]:,.2f}")
    
    # Test 2: Cache
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ Test cache:")
    df1 = market_data.get_klines('BTCUSDT', interval='5m', limit=100, use_cache=True)
    df2 = market_data.get_klines('BTCUSDT', interval='5m', limit=100, use_cache=True)
    stats = market_data.get_stats()
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    # Test 3: Buffer
    print("\n3Ã¯Â¸ÂÃ¢Æ’Â£ Test buffer:")
    buffer_df = market_data.get_from_buffer('BTCUSDT', limit=10)
    print(f"   Buffer size: {len(buffer_df)}")
    
    # Test 4: Calculs
    print("\n4Ã¯Â¸ÂÃ¢Æ’Â£ Test calculs:")
    vwap = market_data.calculate_vwap('BTCUSDT', periods=20)
    vol = market_data.calculate_volatility('BTCUSDT', periods=20)
    change = market_data.get_price_change('BTCUSDT', periods=10)
    print(f"   VWAP: ${vwap:,.2f}" if vwap else "   VWAP: N/A")
    print(f"   Volatility: {vol:.4f}" if vol else "   Volatility: N/A")
    print(f"   Price change: {change:.2%}" if change else "   Price change: N/A")
    
    # Stats finales
    print("\nÃ°Å¸â€œÅ  Statistiques:")
    final_stats = market_data.get_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")
