"""
Feature Engineering pour The Bot
Calcule les 30 features les plus importantes pour le ML
OptimisÃƒÂ© pour performance sur PC classique
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration des features"""
    use_price_features: bool = True
    use_volume_features: bool = True
    use_technical_features: bool = True
    use_market_structure: bool = True
    use_sentiment_features: bool = True
    
    # PÃƒÂ©riodes pour les features
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    atr_period: int = 14
    adx_period: int = 14


class FeatureEngineer:
    """
    Calculateur de features optimisÃƒÂ©
    
    30 features clÃƒÂ©s rÃƒÂ©parties en 5 catÃƒÂ©gories:
    - Prix (5 features)
    - Volume (5 features)
    - Indicateurs Techniques (10 features)
    - Structure de MarchÃƒÂ© (5 features)
    - Sentiment & Momentum (5 features)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialise le feature engineer
        
        Args:
            config: Configuration des features
        """
        self.config = config or FeatureConfig()
        self.feature_names = []
        self._initialize_feature_names()
        
        logger.info(f"Ã¢Å“â€¦ Feature Engineer initialisÃƒÂ© ({len(self.feature_names)} features)")
    
    def _initialize_feature_names(self):
        """Initialise la liste des noms de features"""
        self.feature_names = []
        
        if self.config.use_price_features:
            self.feature_names.extend([
                'price_change_5m',
                'price_change_15m',
                'price_change_1h',
                'price_position_24h',
                'distance_from_vwap'
            ])
        
        if self.config.use_volume_features:
            self.feature_names.extend([
                'volume_ratio',
                'buy_sell_ratio',
                'volume_trend',
                'large_trades_ratio',
                'volume_profile_score'
            ])
        
        if self.config.use_technical_features:
            self.feature_names.extend([
                'rsi_14',
                'rsi_divergence',
                'macd_signal',
                'bb_position',
                'atr_normalized',
                'ema_trend',
                'stoch_k',
                'adx',
                'obv_trend',
                'mfi'
            ])
        
        if self.config.use_market_structure:
            self.feature_names.extend([
                'support_distance',
                'resistance_distance',
                'orderbook_imbalance',
                'spread_ratio',
                'liquidity_score'
            ])
        
        if self.config.use_sentiment_features:
            self.feature_names.extend([
                'funding_rate',
                'long_short_ratio',
                'momentum_score',
                'trend_strength',
                'volatility_regime'
            ])
    
    def calculate_features(self, df: pd.DataFrame, 
                          orderbook: Optional[Dict] = None,
                          additional_data: Optional[Dict] = None) -> np.ndarray:
        """
        Calcule toutes les features pour un DataFrame
        
        Args:
            df: DataFrame avec colonnes OHLCV
            orderbook: Orderbook optionnel pour features avancÃƒÂ©es
            additional_data: DonnÃƒÂ©es supplÃƒÂ©mentaires (funding rate, etc.)
            
        Returns:
            Array numpy avec les features (shape: [len(df), n_features])
        """
        try:
            features = {}
            
            # 1. Prix Features
            if self.config.use_price_features:
                features.update(self._calculate_price_features(df))
            
            # 2. Volume Features
            if self.config.use_volume_features:
                features.update(self._calculate_volume_features(df))
            
            # 3. Indicateurs Techniques
            if self.config.use_technical_features:
                features.update(self._calculate_technical_features(df))
            
            # 4. Structure de MarchÃƒÂ©
            if self.config.use_market_structure:
                features.update(self._calculate_market_structure_features(df, orderbook))
            
            # 5. Sentiment & Momentum
            if self.config.use_sentiment_features:
                features.update(self._calculate_sentiment_features(df, additional_data))
            
            # Convertir en array numpy
            feature_array = np.column_stack([features[name] for name in self.feature_names])
            
            # Remplacer NaN et inf
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Erreur calcul features: {e}")
            # Retourner array de zÃƒÂ©ros en cas d'erreur
            return np.zeros((len(df), len(self.feature_names)))
    
    def _calculate_price_features(self, df: pd.DataFrame) -> Dict:
        """Calcule les 5 features de prix"""
        features = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 1. Price change 5m (environ 1 candle)
        features['price_change_5m'] = self._pct_change(close, 1)
        
        # 2. Price change 15m (environ 3 candles)
        features['price_change_15m'] = self._pct_change(close, 3)
        
        # 3. Price change 1h (environ 12 candles)
        features['price_change_1h'] = self._pct_change(close, 12)
        
        # 4. Position du prix dans la range 24h
        high_24h = pd.Series(high).rolling(288).max().values  # 288 * 5min = 24h
        low_24h = pd.Series(low).rolling(288).min().values
        range_24h = high_24h - low_24h
        features['price_position_24h'] = np.where(
            range_24h > 0,
            (close - low_24h) / range_24h,
            0.5
        )
        
        # 5. Distance from VWAP
        vwap = self._calculate_vwap(df)
        features['distance_from_vwap'] = (close - vwap) / vwap
        
        return features
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict:
        """Calcule les 5 features de volume"""
        features = {}
        
        volume = df['volume'].values
        close = df['close'].values
        
        # 1. Volume ratio (vs moyenne mobile)
        volume_ma = pd.Series(volume).rolling(20).mean().values
        features['volume_ratio'] = np.where(volume_ma > 0, volume / volume_ma, 1.0)
        
        # 2. Buy/Sell ratio (approximation avec close vs open)
        open_price = df['open'].values
        buy_volume = np.where(close > open_price, volume, 0)
        sell_volume = np.where(close <= open_price, volume, 0)
        buy_sum = pd.Series(buy_volume).rolling(20).sum().values
        sell_sum = pd.Series(sell_volume).rolling(20).sum().values
        features['buy_sell_ratio'] = np.where(sell_sum > 0, buy_sum / sell_sum, 1.0)
        
        # 3. Volume trend
        volume_ema_fast = pd.Series(volume).ewm(span=10).mean().values
        volume_ema_slow = pd.Series(volume).ewm(span=30).mean().values
        features['volume_trend'] = np.where(
            volume_ema_slow > 0,
            (volume_ema_fast - volume_ema_slow) / volume_ema_slow,
            0
        )
        
        # 4. Large trades ratio (volume > 2x moyenne)
        large_trades = (volume > 2 * volume_ma).astype(float)
        features['large_trades_ratio'] = pd.Series(large_trades).rolling(20).mean().values
        
        # 5. Volume profile score (volume concentrÃƒÂ©)
        vol_std = pd.Series(volume).rolling(20).std().values
        vol_mean = pd.Series(volume).rolling(20).mean().values
        features['volume_profile_score'] = np.where(vol_mean > 0, vol_std / vol_mean, 0)
        
        return features
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> Dict:
        """Calcule les 10 features d'indicateurs techniques"""
        features = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # 1. RSI 14
        features['rsi_14'] = self._calculate_rsi(close, self.config.rsi_period)
        
        # 2. RSI divergence (simplifiÃƒÂ©e)
        rsi = features['rsi_14']
        price_slope = self._calculate_slope(close, 14)
        rsi_slope = self._calculate_slope(rsi, 14)
        features['rsi_divergence'] = price_slope * rsi_slope  # NÃƒÂ©gatif = divergence
        
        # 3. MACD signal
        macd, signal, _ = self._calculate_macd(close)
        features['macd_signal'] = macd - signal
        
        # 4. Bollinger Bands position
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
        bb_range = bb_upper - bb_lower
        features['bb_position'] = np.where(
            bb_range > 0,
            (close - bb_lower) / bb_range,
            0.5
        )
        
        # 5. ATR normalized
        atr = self._calculate_atr(high, low, close, self.config.atr_period)
        features['atr_normalized'] = np.where(close > 0, atr / close, 0)
        
        # 6. EMA trend strength
        ema_fast = pd.Series(close).ewm(span=12).mean().values
        ema_slow = pd.Series(close).ewm(span=26).mean().values
        features['ema_trend'] = np.where(ema_slow > 0, (ema_fast - ema_slow) / ema_slow, 0)
        
        # 7. Stochastic K
        features['stoch_k'] = self._calculate_stochastic_k(high, low, close, 14)
        
        # 8. ADX
        features['adx'] = self._calculate_adx(high, low, close, self.config.adx_period)
        
        # 9. OBV trend
        obv = self._calculate_obv(close, volume)
        obv_ema = pd.Series(obv).ewm(span=20).mean().values
        features['obv_trend'] = self._calculate_slope(obv_ema, 14)
        
        # 10. MFI (Money Flow Index)
        features['mfi'] = self._calculate_mfi(high, low, close, volume, 14)
        
        return features
    
    def _calculate_market_structure_features(self, df: pd.DataFrame, 
                                            orderbook: Optional[Dict] = None) -> Dict:
        """Calcule les 5 features de structure de marchÃƒÂ©"""
        features = {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 1-2. Distance aux supports/rÃƒÂ©sistances
        supports, resistances = self._find_support_resistance(high, low, close)
        
        if supports:
            nearest_support = min(supports, key=lambda x: abs(x - close[-1]))
            features['support_distance'] = (close[-1] - nearest_support) / close[-1]
        else:
            features['support_distance'] = np.full(len(close), 0.0)
        
        if resistances:
            nearest_resistance = min(resistances, key=lambda x: abs(x - close[-1]))
            features['resistance_distance'] = (nearest_resistance - close[-1]) / close[-1]
        else:
            features['resistance_distance'] = np.full(len(close), 0.0)
        
        # RÃƒÂ©pliquer pour toutes les lignes
        if isinstance(features['support_distance'], float):
            features['support_distance'] = np.full(len(close), features['support_distance'])
        if isinstance(features['resistance_distance'], float):
            features['resistance_distance'] = np.full(len(close), features['resistance_distance'])
        
        # 3. Orderbook imbalance
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:10])
            ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:10])
            total = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
            features['orderbook_imbalance'] = np.full(len(close), imbalance)
        else:
            features['orderbook_imbalance'] = np.zeros(len(close))
        
        # 4. Spread ratio (approximation avec high-low)
        spread = high - low
        features['spread_ratio'] = np.where(close > 0, spread / close, 0)
        
        # 5. Liquidity score (volume relatif)
        volume_percentile = pd.Series(df['volume'].values).rolling(100).apply(
            lambda x: (x.iloc[-1] / x.quantile(0.5)) if len(x) > 0 and x.quantile(0.5) > 0 else 1.0,
            raw=False
        ).values
        features['liquidity_score'] = volume_percentile
        
        return features
    
    def _calculate_sentiment_features(self, df: pd.DataFrame,
                                     additional_data: Optional[Dict] = None) -> Dict:
        """Calcule les 5 features de sentiment et momentum"""
        features = {}
        
        close = df['close'].values
        
        # 1. Funding rate (si disponible)
        if additional_data and 'funding_rate' in additional_data:
            features['funding_rate'] = np.full(len(close), additional_data['funding_rate'])
        else:
            features['funding_rate'] = np.zeros(len(close))
        
        # 2. Long/Short ratio (si disponible)
        if additional_data and 'long_short_ratio' in additional_data:
            features['long_short_ratio'] = np.full(len(close), additional_data['long_short_ratio'])
        else:
            features['long_short_ratio'] = np.ones(len(close))
        
        # 3. Momentum score (combinaison de plusieurs pÃƒÂ©riodes)
        mom_5 = self._pct_change(close, 5)
        mom_10 = self._pct_change(close, 10)
        mom_20 = self._pct_change(close, 20)
        features['momentum_score'] = (mom_5 + mom_10 + mom_20) / 3
        
        # 4. Trend strength (ADX simplifiÃƒÂ© ou dÃƒÂ©rivÃƒÂ©)
        ema_12 = pd.Series(close).ewm(span=12).mean().values
        ema_26 = pd.Series(close).ewm(span=26).mean().values
        ema_50 = pd.Series(close).ewm(span=50).mean().values
        
        trend_score = np.zeros(len(close))
        trend_score += (ema_12 > ema_26).astype(float)
        trend_score += (ema_26 > ema_50).astype(float)
        trend_score += (close > ema_12).astype(float)
        features['trend_strength'] = trend_score / 3  # NormalisÃƒÂ© 0-1
        
        # 5. Volatility regime
        returns = np.diff(close, prepend=close[0]) / close
        vol_current = pd.Series(returns).rolling(20).std().values
        vol_ma = pd.Series(vol_current).rolling(100).mean().values
        features['volatility_regime'] = np.where(vol_ma > 0, vol_current / vol_ma, 1.0)
        
        return features
    
    # ============= Fonctions utilitaires =============
    
    def _pct_change(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Calcul du pourcentage de changement"""
        result = np.zeros_like(arr)
        result[period:] = (arr[period:] - arr[:-period]) / arr[:-period]
        return result
    
    def _calculate_slope(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Calcul de la pente"""
        result = np.zeros_like(arr)
        for i in range(period, len(arr)):
            x = np.arange(period)
            y = arr[i-period:i]
            if len(y) == period:
                slope = np.polyfit(x, y, 1)[0]
                result[i] = slope
        return result
    
    def _calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """Calcul du VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        return vwap.values
    
    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calcul du RSI"""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, close: np.ndarray) -> tuple:
        """Calcul du MACD"""
        ema_fast = pd.Series(close).ewm(span=self.config.macd_fast).mean().values
        ema_slow = pd.Series(close).ewm(span=self.config.macd_slow).mean().values
        macd = ema_fast - ema_slow
        signal = pd.Series(macd).ewm(span=self.config.macd_signal).mean().values
        histogram = macd - signal
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, close: np.ndarray) -> tuple:
        """Calcul des Bollinger Bands"""
        sma = pd.Series(close).rolling(self.config.bb_period).mean().values
        std = pd.Series(close).rolling(self.config.bb_period).std().values
        upper = sma + (std * self.config.bb_std)
        lower = sma - (std * self.config.bb_std)
        return upper, sma, lower
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int) -> np.ndarray:
        """Calcul de l'ATR"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(period).mean().values
        return atr
    
    def _calculate_stochastic_k(self, high: np.ndarray, low: np.ndarray,
                                close: np.ndarray, period: int) -> np.ndarray:
        """Calcul du Stochastic K"""
        lowest_low = pd.Series(low).rolling(period).min().values
        highest_high = pd.Series(high).rolling(period).max().values
        
        k = np.where(
            highest_high - lowest_low > 0,
            100 * (close - lowest_low) / (highest_high - lowest_low),
            50
        )
        return k
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int) -> np.ndarray:
        """Calcul de l'ADX (simplifiÃƒÂ©)"""
        tr = self._calculate_atr(high, low, close, 1)
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / pd.Series(tr).rolling(period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / pd.Series(tr).rolling(period).mean()
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(period).mean().values
        
        return adx
    
    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calcul de l'OBV"""
        obv = np.zeros_like(volume)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _calculate_mfi(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
        """Calcul du MFI"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = np.where(np.diff(typical_price, prepend=typical_price[0]) > 0, money_flow, 0)
        negative_flow = np.where(np.diff(typical_price, prepend=typical_price[0]) < 0, money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(period).sum().values
        negative_mf = pd.Series(negative_flow).rolling(period).sum().values
        
        mfi = 100 - (100 / (1 + positive_mf / np.where(negative_mf > 0, negative_mf, 1)))
        
        return mfi
    
    def _find_support_resistance(self, high: np.ndarray, low: np.ndarray,
                                 close: np.ndarray, window: int = 20) -> tuple:
        """Trouve les niveaux de support et rÃƒÂ©sistance"""
        supports = []
        resistances = []
        
        if len(close) < window * 2:
            return supports, resistances
        
        # Derniers 100 points seulement
        recent_high = high[-100:]
        recent_low = low[-100:]
        
        for i in range(window, len(recent_high) - window):
            # Resistance
            if recent_high[i] == max(recent_high[i-window:i+window+1]):
                resistances.append(recent_high[i])
            
            # Support
            if recent_low[i] == min(recent_low[i-window:i+window+1]):
                supports.append(recent_low[i])
        
        # Garder seulement les niveaux distincts
        supports = list(set([round(s, 2) for s in supports]))
        resistances = list(set([round(r, 2) for r in resistances]))
        
        return supports, resistances
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms des features"""
        return self.feature_names.copy()
    
    def get_feature_count(self) -> int:
        """Retourne le nombre de features"""
        return len(self.feature_names)


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du feature engineer"""
    
    # CrÃƒÂ©er des donnÃƒÂ©es de test
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='5min'),
        'open': 50000 + np.cumsum(np.random.randn(n) * 100),
        'high': 50100 + np.cumsum(np.random.randn(n) * 100),
        'low': 49900 + np.cumsum(np.random.randn(n) * 100),
        'close': 50000 + np.cumsum(np.random.randn(n) * 100),
        'volume': np.random.uniform(100, 1000, n)
    })
    
    print("\n=== Test Feature Engineering ===\n")
    
    # CrÃƒÂ©er le feature engineer
    engineer = FeatureEngineer()
    
    print(f"Nombre de features: {engineer.get_feature_count()}")
    print(f"\nFeatures calculÃƒÂ©es:")
    for i, name in enumerate(engineer.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
    
    # Calculer les features
    import time
    start = time.time()
    features = engineer.calculate_features(df)
    elapsed = time.time() - start
    
    print(f"\nÃ°Å¸â€œÅ  RÃƒÂ©sultats:")
    print(f"  Shape: {features.shape}")
    print(f"  Temps: {elapsed:.3f}s ({features.shape[0] / elapsed:.0f} rows/sec)")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  NaN: {np.isnan(features).sum()}")
    
    print("\nÃ¢Å“â€¦ Test terminÃƒÂ©")
