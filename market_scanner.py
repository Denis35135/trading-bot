"""
Market Scanner pour The Bot
Scan et sÃƒÂ©lectionne les meilleures opportunitÃƒÂ©s de trading
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Scanner de marchÃƒÂ© intelligent
    
    ResponsabilitÃƒÂ©s:
    - Scanner tous les symboles disponibles
    - Filtrer selon critÃƒÂ¨res (volume, volatilitÃƒÂ©, etc.)
    - Calculer un score pour chaque symbole
    - SÃƒÂ©lectionner les top N symboles
    - Mettre ÃƒÂ  jour pÃƒÂ©riodiquement la sÃƒÂ©lection
    """
    
    def __init__(self, exchange_client, config: Dict):
        """
        Initialise le market scanner
        
        Args:
            exchange_client: Client de l'exchange
            config: Configuration du scanner
        """
        self.exchange = exchange_client
        self.config = config
        
        # Configuration
        self.min_volume_24h = getattr(config, 'MIN_VOLUME_24H', 10_000_000)
        self.max_spread = getattr(config, 'MAX_SPREAD_PERCENT', 0.002)
        self.min_volatility = getattr(config, 'VOLATILITY_RANGE', [0.015, 0.12])[0]
        self.max_volatility = getattr(config, 'VOLATILITY_RANGE', [0.015, 0.12])[1]
        self.symbols_to_scan = getattr(config, 'SYMBOLS_TO_SCAN', 100)
        self.symbols_to_trade = getattr(config, 'SYMBOLS_TO_TRADE', 20)
        self.scan_interval = getattr(config, 'SCAN_INTERVAL', 300)  # 5 minutes
        self.blacklist = getattr(config, 'BLACKLIST_SYMBOLS', ['USDCUSDT', 'BUSDUSDC'])
        
        # Ãƒâ€°tat
        self.all_symbols = []
        self.top_symbols = []
        self.symbol_scores = {}
        self.symbol_data = {}
        
        # Cache des donnÃƒÂ©es
        self.volume_cache = {}
        self.volatility_cache = {}
        self.correlation_matrix = None
        
        # Threading
        self.is_running = False
        self.scan_thread = None
        self.last_scan = None
        
        # Statistiques
        self.scan_stats = {
            'total_scans': 0,
            'last_scan_duration': 0,
            'symbols_analyzed': 0,
            'symbols_qualified': 0
        }
        
        logger.info("Market Scanner initialisÃƒÂ©")
        logger.info(f"Scan de {self.symbols_to_scan} symboles, sÃƒÂ©lection de {self.symbols_to_trade}")
    
    def start(self):
        """DÃƒÂ©marre le scanner"""
        if self.is_running:
            logger.warning("Scanner dÃƒÂ©jÃƒÂ  en cours")
            return
        
        self.is_running = True
        
        # Scan initial
        self.perform_scan()
        
        # DÃƒÂ©marrer le thread de scan pÃƒÂ©riodique
        self.scan_thread = threading.Thread(
            target=self._scan_loop,
            daemon=True
        )
        self.scan_thread.start()
        
        logger.info("Market Scanner dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte le scanner"""
        self.is_running = False
        
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        
        logger.info("Market Scanner arrÃƒÂªtÃƒÂ©")
    
    def perform_scan(self):
        """Effectue un scan complet du marchÃƒÂ©"""
        start_time = time.time()
        logger.info("Ã°Å¸â€Â DÃƒÂ©but du scan de marchÃƒÂ©...")
        
        try:
            # 1. RÃƒÂ©cupÃƒÂ©rer tous les symboles USDC
            self._fetch_all_symbols()
            
            # 2. Filtrer et collecter les donnÃƒÂ©es
            qualified_symbols = self._filter_symbols()
            
            # 3. Calculer les scores
            self._calculate_scores(qualified_symbols)
            
            # 4. SÃƒÂ©lectionner les top symboles
            self._select_top_symbols()
            
            # 5. Calculer les corrÃƒÂ©lations
            self._calculate_correlations()
            
            # Statistiques
            duration = time.time() - start_time
            self.scan_stats['total_scans'] += 1
            self.scan_stats['last_scan_duration'] = duration
            self.scan_stats['symbols_analyzed'] = len(self.all_symbols)
            self.scan_stats['symbols_qualified'] = len(qualified_symbols)
            
            self.last_scan = datetime.now()
            
            logger.info(f"Ã¢Å“â€¦ Scan terminÃƒÂ© en {duration:.1f}s")
            logger.info(f"   Symboles analysÃƒÂ©s: {len(self.all_symbols)}")
            logger.info(f"   Symboles qualifiÃƒÂ©s: {len(qualified_symbols)}")
            logger.info(f"   Top sÃƒÂ©lectionnÃƒÂ©s: {len(self.top_symbols)}")
            
        except Exception as e:
            logger.error(f"Erreur pendant le scan: {e}")
    
    def _scan_loop(self):
        """Boucle de scan pÃƒÂ©riodique"""
        while self.is_running:
            try:
                # Attendre l'intervalle
                time.sleep(self.scan_interval)
                
                # Effectuer le scan
                self.perform_scan()
                
            except Exception as e:
                logger.error(f"Erreur scan loop: {e}")
                time.sleep(60)  # Attendre 1 minute en cas d'erreur
    
    def _fetch_all_symbols(self):
        """RÃƒÂ©cupÃƒÂ¨re tous les symboles disponibles"""
        try:
            # RÃƒÂ©cupÃƒÂ©rer les stats 24h pour tous les symboles
            all_tickers = self.exchange.get_24h_stats()
            
            # Filtrer les paires USDC uniquement
            self.all_symbols = []
            self.volume_cache = {}
            
            for ticker in all_tickers:
                symbol = ticker['symbol']
                
                # Garder seulement les paires USDC
                if not symbol.endswith('USDC'):
                    continue
                
                # Ignorer la blacklist
                if symbol in self.blacklist:
                    continue
                
                # Ignorer les stablecoins pairs
                if any(stable in symbol for stable in ['USDT', 'BUSD', 'DAI', 'TUSD']):
                    continue
                
                self.all_symbols.append(symbol)
                self.volume_cache[symbol] = ticker['quote_volume']
            
            # Trier par volume et garder les top N
            self.all_symbols.sort(key=lambda s: self.volume_cache.get(s, 0), reverse=True)
            self.all_symbols = self.all_symbols[:self.symbols_to_scan]
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration symboles: {e}")
    
    def _filter_symbols(self) -> List[str]:
        """
        Filtre les symboles selon les critÃƒÂ¨res
        
        Returns:
            Liste des symboles qualifiÃƒÂ©s
        """
        qualified = []
        
        for symbol in self.all_symbols:
            try:
                # VÃƒÂ©rifier le volume
                volume = self.volume_cache.get(symbol, 0)
                if volume < self.min_volume_24h:
                    continue
                
                # RÃƒÂ©cupÃƒÂ©rer les donnÃƒÂ©es supplÃƒÂ©mentaires
                ticker = self.exchange.get_symbol_ticker(symbol)
                if not ticker:
                    continue
                
                # VÃƒÂ©rifier le spread
                if ticker['ask'] > 0 and ticker['bid'] > 0:
                    spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                    if spread > self.max_spread:
                        continue
                else:
                    continue
                
                # RÃƒÂ©cupÃƒÂ©rer les klines pour calculer la volatilitÃƒÂ©
                df = self.exchange.get_klines(symbol, '1h', limit=24)
                if df.empty:
                    continue
                
                # Calculer la volatilitÃƒÂ© (ATR / prix)
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values
                
                # ATR simplifiÃƒÂ© sur 24h
                ranges = high - low
                atr = np.mean(ranges)
                volatility = atr / np.mean(close) if np.mean(close) > 0 else 0
                
                # VÃƒÂ©rifier la volatilitÃƒÂ©
                if volatility < self.min_volatility or volatility > self.max_volatility:
                    continue
                
                # Stocker les donnÃƒÂ©es
                self.volatility_cache[symbol] = volatility
                self.symbol_data[symbol] = {
                    'price': ticker['price'],
                    'volume': volume,
                    'spread': spread,
                    'volatility': volatility,
                    'change_24h': ticker.get('change_24h', 0),
                    'df': df  # Garder pour analyse ultÃƒÂ©rieure
                }
                
                qualified.append(symbol)
                
            except Exception as e:
                logger.debug(f"Erreur filtrage {symbol}: {e}")
                continue
        
        return qualified
    
    def _calculate_scores(self, symbols: List[str]):
        """
        Calcule un score pour chaque symbole
        
        Args:
            symbols: Liste des symboles ÃƒÂ  scorer
        """
        self.symbol_scores = {}
        
        for symbol in symbols:
            try:
                data = self.symbol_data.get(symbol)
                if not data:
                    continue
                
                score = 0
                
                # 1. Score de volume (0-30 points)
                volume = data['volume']
                if volume > 100_000_000:
                    score += 30
                elif volume > 50_000_000:
                    score += 25
                elif volume > 20_000_000:
                    score += 20
                else:
                    score += 10
                
                # 2. Score de volatilitÃƒÂ© (0-25 points)
                volatility = data['volatility']
                if 0.02 < volatility < 0.06:  # Sweet spot
                    score += 25
                elif 0.015 < volatility < 0.08:
                    score += 20
                else:
                    score += 10
                
                # 3. Score de momentum (0-20 points)
                change_24h = abs(data['change_24h'])
                if change_24h > 5:
                    score += 20
                elif change_24h > 3:
                    score += 15
                elif change_24h > 1:
                    score += 10
                else:
                    score += 5
                
                # 4. Score de liquiditÃƒÂ©/spread (0-15 points)
                spread = data['spread']
                if spread < 0.0005:
                    score += 15
                elif spread < 0.001:
                    score += 10
                elif spread < 0.0015:
                    score += 5
                
                # 5. Score technique (0-10 points)
                technical_score = self._calculate_technical_score(data['df'])
                score += technical_score
                
                # Bonus/Malus
                # Bonus si trending
                if self._is_trending(data['df']):
                    score += 5
                
                # Malus si trop corrÃƒÂ©lÃƒÂ© ÃƒÂ  BTC (ÃƒÂ  implÃƒÂ©menter)
                # ...
                
                self.symbol_scores[symbol] = score
                
            except Exception as e:
                logger.debug(f"Erreur calcul score {symbol}: {e}")
                continue
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Calcule un score technique basÃƒÂ© sur les indicateurs
        
        Args:
            df: DataFrame avec OHLCV
            
        Returns:
            Score technique (0-10)
        """
        if len(df) < 20:
            return 5
        
        score = 5  # Score de base
        
        try:
            close = df['close'].values
            
            # RSI
            rsi = self._calculate_rsi(close, 14)
            if 40 < rsi < 60:  # Zone neutre, bon pour trading
                score += 2
            
            # Distance ÃƒÂ  la moyenne mobile
            sma_20 = np.mean(close[-20:])
            distance = abs(close[-1] - sma_20) / sma_20
            if distance < 0.02:  # Proche de la MA
                score += 2
            
            # Volume trend
            volume = df['volume'].values
            if len(volume) >= 5:
                recent_vol = np.mean(volume[-5:])
                older_vol = np.mean(volume[-20:-5]) if len(volume) >= 20 else np.mean(volume[:-5])
                if recent_vol > older_vol * 1.5:  # Volume en augmentation
                    score += 1
            
        except Exception as e:
            logger.debug(f"Erreur calcul technique: {e}")
        
        return min(score, 10)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcul RSI simple"""
        if len(prices) < period:
            return 50
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _is_trending(self, df: pd.DataFrame, threshold: float = 0.6) -> bool:
        """
        DÃƒÂ©termine si un symbole est en tendance
        
        Args:
            df: DataFrame avec OHLCV
            threshold: RÃ‚Â² minimum pour considÃƒÂ©rer une tendance
            
        Returns:
            True si en tendance
        """
        if len(df) < 20:
            return False
        
        try:
            close = df['close'].values[-20:]
            x = np.arange(len(close))
            
            # RÃƒÂ©gression linÃƒÂ©aire
            coeffs = np.polyfit(x, close, 1)
            y_pred = np.polyval(coeffs, x)
            
            # R-squared
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - np.mean(close)) ** 2)
            
            if ss_tot == 0:
                return False
            
            r_squared = 1 - (ss_res / ss_tot)
            
            return r_squared > threshold
            
        except Exception:
            return False
    
    def _select_top_symbols(self):
        """SÃƒÂ©lectionne les top N symboles"""
        # Trier par score
        sorted_symbols = sorted(
            self.symbol_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Prendre les top N
        self.top_symbols = [symbol for symbol, score in sorted_symbols[:self.symbols_to_trade]]
        
        # Log les top 10
        logger.info("Ã°Å¸Ââ€  Top 10 symboles:")
        for i, (symbol, score) in enumerate(sorted_symbols[:10]):
            data = self.symbol_data.get(symbol, {})
            logger.info(f"   {i+1}. {symbol}: Score={score:.0f}, "
                       f"Vol={data.get('volume', 0)/1e6:.1f}M, "
                       f"Volat={data.get('volatility', 0)*100:.1f}%")
    
    def _calculate_correlations(self):
        """Calcule la matrice de corrÃƒÂ©lation entre les top symboles"""
        if len(self.top_symbols) < 2:
            return
        
        try:
            # Collecter les prix de clÃƒÂ´ture
            price_data = {}
            min_length = float('inf')
            
            for symbol in self.top_symbols[:10]:  # Limiter pour performance
                if symbol in self.symbol_data:
                    df = self.symbol_data[symbol].get('df')
                    if df is not None and len(df) > 0:
                        price_data[symbol] = df['close'].values
                        min_length = min(min_length, len(price_data[symbol]))
            
            if len(price_data) < 2:
                return
            
            # Aligner les longueurs
            for symbol in price_data:
                price_data[symbol] = price_data[symbol][-min_length:]
            
            # CrÃƒÂ©er DataFrame et calculer corrÃƒÂ©lations
            df_prices = pd.DataFrame(price_data)
            self.correlation_matrix = df_prices.corr()
            
            # Identifier les paires trÃƒÂ¨s corrÃƒÂ©lÃƒÂ©es
            high_corr_pairs = []
            for i in range(len(self.correlation_matrix)):
                for j in range(i+1, len(self.correlation_matrix)):
                    corr = self.correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        symbol1 = self.correlation_matrix.index[i]
                        symbol2 = self.correlation_matrix.columns[j]
                        high_corr_pairs.append((symbol1, symbol2, corr))
            
            if high_corr_pairs:
                logger.warning(Ã¢Å¡Â Ã¯Â¸Â Paires fortement corrÃƒÂ©lÃƒÂ©es dÃƒÂ©tectÃƒÂ©es:")
                for s1, s2, corr in high_corr_pairs[:5]:
                    logger.warning(f"   {s1} <-> {s2}: {corr:.2f}")
            
        except Exception as e:
            logger.error(f"Erreur calcul corrÃƒÂ©lations: {e}")
    
    def get_top_symbols(self, n: Optional[int] = None) -> List[str]:
        """
        Retourne les top symboles
        
        Args:
            n: Nombre de symboles (dÃƒÂ©faut: tous)
            
        Returns:
            Liste des top symboles
        """
        if n:
            return self.top_symbols[:n]
        return self.top_symbols
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """
        Retourne les donnÃƒÂ©es d'un symbole
        
        Args:
            symbol: Le symbole
            
        Returns:
            Dict avec les donnÃƒÂ©es ou None
        """
        return self.symbol_data.get(symbol)
    
    def is_symbol_qualified(self, symbol: str) -> bool:
        """
        VÃƒÂ©rifie si un symbole est qualifiÃƒÂ© pour le trading
        
        Args:
            symbol: Le symbole ÃƒÂ  vÃƒÂ©rifier
            
        Returns:
            True si qualifiÃƒÂ©
        """
        return symbol in self.top_symbols
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Retourne la corrÃƒÂ©lation entre deux symboles
        
        Args:
            symbol1: Premier symbole
            symbol2: DeuxiÃƒÂ¨me symbole
            
        Returns:
            CorrÃƒÂ©lation ou None
        """
        if self.correlation_matrix is None:
            return None
        
        if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
            return self.correlation_matrix.loc[symbol1, symbol2]
        
        return None
    
    def force_rescan(self):
        """Force un nouveau scan immÃƒÂ©diatement"""
        logger.info("Rescan forcÃƒÂ© demandÃƒÂ©")
        self.perform_scan()
    
    def add_to_blacklist(self, symbol: str):
        """Ajoute un symbole ÃƒÂ  la blacklist"""
        if symbol not in self.blacklist:
            self.blacklist.append(symbol)
            logger.info(f"Symbole {symbol} ajoutÃƒÂ© ÃƒÂ  la blacklist")
            
            # Retirer des top symbols si prÃƒÂ©sent
            if symbol in self.top_symbols:
                self.top_symbols.remove(symbol)
    
    def remove_from_blacklist(self, symbol: str):
        """Retire un symbole de la blacklist"""
        if symbol in self.blacklist:
            self.blacklist.remove(symbol)
            logger.info(f"Symbole {symbol} retirÃƒÂ© de la blacklist")
    
    def get_scan_stats(self) -> Dict:
        """Retourne les statistiques de scan"""
        stats = self.scan_stats.copy()
        stats['last_scan'] = self.last_scan
        stats['top_symbols_count'] = len(self.top_symbols)
        stats['blacklist_size'] = len(self.blacklist)
        
        if self.last_scan:
            stats['time_since_last_scan'] = (datetime.now() - self.last_scan).seconds
        
        return stats


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du market scanner"""
    
    # Mock exchange pour tests
    class MockExchange:
        def get_24h_stats(self, symbol=None):
            # Simuler des donnÃƒÂ©es
            symbols = [
                'BTCUSDC', 'ETHUSDC', 'BNBUSDC', 'ADAUSDC', 'DOGEUSDC',
                'XRPUSDC', 'DOTUSDC', 'UNIUSDC', 'LTCUSDC', 'LINKUSDC',
                'MATICUSDC', 'ALGOUSDC', 'ATOMUSDC', 'AVAXUSDC', 'NEARUSDC'
            ]
            
            stats = []
            for i, sym in enumerate(symbols):
                stats.append({
                    'symbol': sym,
                    'volume': np.random.uniform(5e6, 100e6),
                    'quote_volume': np.random.uniform(5e6, 100e6),
                    'price_change': np.random.uniform(-10, 10),
                    'high': 100 + i,
                    'low': 95 + i
                })
            
            # Ajouter des symboles non USDC pour test
            stats.append({'symbol': 'BTCUSDT', 'volume': 500e6, 'quote_volume': 500e6})
            
            return stats
        
        def get_symbol_ticker(self, symbol):
            return {
                'symbol': symbol,
                'price': 100.0,
                'bid': 99.95,
                'ask': 100.05,
                'volume': 10000000,
                'change_24h': np.random.uniform(-5, 5)
            }
        
        def get_klines(self, symbol, interval, limit):
            # GÃƒÂ©nÃƒÂ©rer des donnÃƒÂ©es OHLCV fake
            size = limit
            close = 100 + np.cumsum(np.random.randn(size) * 0.5)
            
            df = pd.DataFrame({
                'open': close + np.random.randn(size) * 0.1,
                'high': close + abs(np.random.randn(size) * 0.3),
                'low': close - abs(np.random.randn(size) * 0.3),
                'close': close,
                'volume': np.random.randint(1000, 10000, size)
            })
            
            return df
    
    # Configuration
    config = {
        'min_volume_24h': 10_000_000,
        'max_spread_percent': 0.002,
        'volatility_range': [0.01, 0.10],
        'symbols_to_scan': 50,
        'symbols_to_trade': 10,
        'scan_interval': 300
    }
    
    # CrÃƒÂ©er et tester le scanner
    exchange = MockExchange()
    scanner = MarketScanner(exchange, config)
    
    print("=" * 60)
    print("TEST MARKET SCANNER")
    print("=" * 60)
    
    # Test 1: Scan initial
    print("\nÃ°Å¸â€œÅ  Test 1: Scan initial")
    scanner.perform_scan()
    
    # Afficher les rÃƒÂ©sultats
    print(f"\nTop symboles sÃƒÂ©lectionnÃƒÂ©s: {scanner.get_top_symbols()}")
    
    # Test 2: DonnÃƒÂ©es d'un symbole
    print("\nÃ°Å¸â€œÅ  Test 2: DonnÃƒÂ©es symbole")
    if scanner.top_symbols:
        symbol = scanner.top_symbols[0]
        data = scanner.get_symbol_data(symbol)
        if data:
            print(f"DonnÃƒÂ©es pour {symbol}:")
            print(f"  Prix: ${data['price']:.2f}")
            print(f"  Volume: ${data['volume']/1e6:.1f}M")
            print(f"  VolatilitÃƒÂ©: {data['volatility']*100:.1f}%")
            print(f"  Spread: {data['spread']*100:.3f}%")
    
    # Test 3: Statistiques
    print("\nÃ°Å¸â€œÅ  Test 3: Statistiques de scan")
    stats = scanner.get_scan_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test 4: Blacklist
    print("\nÃ°Å¸â€œÅ  Test 4: Blacklist")
    if scanner.top_symbols:
        symbol = scanner.top_symbols[0]
        print(f"Ajout de {symbol} ÃƒÂ  la blacklist")
        scanner.add_to_blacklist(symbol)
        print(f"Top symbols aprÃƒÂ¨s blacklist: {scanner.get_top_symbols()}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s!")