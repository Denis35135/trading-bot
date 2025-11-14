"""
Indicateurs techniques optimisÃƒÂ©s pour The Bot
Utilise NumPy pour performances maximales
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Classe contenant tous les indicateurs techniques optimisÃƒÂ©s
    Utilise NumPy pour des calculs ultra-rapides
    """
    
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
        """
        Simple Moving Average
        
        Args:
            data: Prix ou array
            period: PÃƒÂ©riode de la moyenne
            
        Returns:
            Array des SMA
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        sma = np.convolve(data, np.ones(period)/period, mode='valid')
        # Pad avec NaN pour garder la mÃƒÂªme longueur
        return np.concatenate([np.full(period-1, np.nan), sma])
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
        """
        Exponential Moving Average
        
        Args:
            data: Prix ou array
            period: PÃƒÂ©riode
            
        Returns:
            Array des EMA
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
        """
        Relative Strength Index
        
        Args:
            data: Prix de clÃƒÂ´ture
            period: PÃƒÂ©riode (dÃƒÂ©faut 14)
            
        Returns:
            Array des RSI (0-100)
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        # Calculer les changements
        deltas = np.diff(data)
        seed = deltas[:period+1]
        
        # SÃƒÂ©parer gains et pertes
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            rs = 100
        else:
            rs = up / down
            
        rsi = np.zeros_like(data)
        rsi[:period] = np.nan
        rsi[period] = 100 - (100 / (1 + rs))
        
        # Calculer le reste
        for i in range(period + 1, len(data)):
            delta = deltas[i-1]
            
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
            
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            if down == 0:
                rsi[i] = 100
            else:
                rs = up / down
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: Union[pd.Series, np.ndarray], 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Prix de clÃƒÂ´ture
            fast: PÃƒÂ©riode EMA rapide (12)
            slow: PÃƒÂ©riode EMA lente (26)
            signal: PÃƒÂ©riode signal (9)
            
        Returns:
            Tuple (macd_line, signal_line, histogram)
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        # Calculer les EMAs
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal)
        
        # Ajuster la longueur
        signal_full = np.full_like(macd_line, np.nan)
        signal_full[slow-1:] = signal_line
        
        # Histogram
        histogram = macd_line - signal_full
        
        return macd_line, signal_full, histogram
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                        period: int = 20, 
                        std_dev: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands
        
        Args:
            data: Prix de clÃƒÂ´ture
            period: PÃƒÂ©riode SMA (20)
            std_dev: Nombre d'ÃƒÂ©carts-types (2)
            
        Returns:
            Tuple (upper_band, middle_band, lower_band)
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        # Middle band (SMA)
        middle = TechnicalIndicators.sma(data, period)
        
        # Calcul de l'ÃƒÂ©cart-type
        std = np.zeros_like(data)
        for i in range(period-1, len(data)):
            std[i] = np.std(data[i-period+1:i+1])
        
        # Upper and Lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(high: np.ndarray, 
            low: np.ndarray, 
            close: np.ndarray, 
            period: int = 14) -> np.ndarray:
        """
        Average True Range (volatilitÃƒÂ©)
        
        Args:
            high: Prix hauts
            low: Prix bas
            close: Prix de clÃƒÂ´ture
            period: PÃƒÂ©riode (14)
            
        Returns:
            Array des ATR
        """
        # True Range
        hl = high - low
        hc = np.abs(high - np.roll(close, 1))
        lc = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(hl, np.maximum(hc, lc))
        tr[0] = hl[0]  # Premier ÃƒÂ©lÃƒÂ©ment
        
        # ATR
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        atr[:period-1] = np.nan
        
        return atr
    
    @staticmethod
    def stochastic(high: np.ndarray, 
                   low: np.ndarray, 
                   close: np.ndarray, 
                   period: int = 14, 
                   smooth_k: int = 3, 
                   smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator
        
        Args:
            high: Prix hauts
            low: Prix bas
            close: Prix de clÃƒÂ´ture
            period: PÃƒÂ©riode lookback (14)
            smooth_k: Lissage %K (3)
            smooth_d: Lissage %D (3)
            
        Returns:
            Tuple (%K, %D)
        """
        # %K raw
        k_raw = np.zeros_like(close)
        
        for i in range(period-1, len(close)):
            highest = np.max(high[i-period+1:i+1])
            lowest = np.min(low[i-period+1:i+1])
            
            if highest - lowest != 0:
                k_raw[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                k_raw[i] = 50
        
        k_raw[:period-1] = np.nan
        
        # %K smooth
        k_smooth = TechnicalIndicators.sma(k_raw[~np.isnan(k_raw)], smooth_k)
        k_full = np.full_like(k_raw, np.nan)
        k_full[period-1+smooth_k-1:] = k_smooth
        
        # %D
        d = TechnicalIndicators.sma(k_full[~np.isnan(k_full)], smooth_d)
        d_full = np.full_like(k_raw, np.nan)
        d_full[period-1+smooth_k-1+smooth_d-1:] = d
        
        return k_full, d_full
    
    @staticmethod
    def adx(high: np.ndarray, 
            low: np.ndarray, 
            close: np.ndarray, 
            period: int = 14) -> np.ndarray:
        """
        Average Directional Index (force de la tendance)
        
        Args:
            high: Prix hauts
            low: Prix bas
            close: Prix de clÃƒÂ´ture
            period: PÃƒÂ©riode (14)
            
        Returns:
            Array des ADX (0-100)
        """
        # Calcul +DM et -DM
        plus_dm = high - np.roll(high, 1)
        minus_dm = np.roll(low, 1) - low
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Si les deux sont positifs, garder le plus grand
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask & (plus_dm < minus_dm)] = 0
        minus_dm[mask & (minus_dm < plus_dm)] = 0
        
        # ATR
        atr_values = TechnicalIndicators.atr(high, low, close, period)
        
        # Smooth DM
        plus_di = 100 * TechnicalIndicators.ema(plus_dm, period) / atr_values
        minus_di = 100 * TechnicalIndicators.ema(minus_dm, period) / atr_values
        
        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(dx)] = 0
        
        # ADX
        adx = TechnicalIndicators.ema(dx, period)
        
        return adx
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        On Balance Volume
        
        Args:
            close: Prix de clÃƒÂ´ture
            volume: Volumes
            
        Returns:
            Array des OBV
        """
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    @staticmethod
    def mfi(high: np.ndarray, 
            low: np.ndarray, 
            close: np.ndarray, 
            volume: np.ndarray, 
            period: int = 14) -> np.ndarray:
        """
        Money Flow Index
        
        Args:
            high: Prix hauts
            low: Prix bas
            close: Prix de clÃƒÂ´ture
            volume: Volumes
            period: PÃƒÂ©riode (14)
            
        Returns:
            Array des MFI (0-100)
        """
        # Typical price
        tp = (high + low + close) / 3
        
        # Money flow
        mf = tp * volume
        
        # Positive and negative money flow
        mfi = np.zeros_like(close)
        
        for i in range(period, len(close)):
            positive_mf = 0
            negative_mf = 0
            
            for j in range(i-period+1, i+1):
                if tp[j] > tp[j-1]:
                    positive_mf += mf[j]
                else:
                    negative_mf += mf[j]
            
            if negative_mf == 0:
                mfi[i] = 100
            else:
                money_ratio = positive_mf / negative_mf
                mfi[i] = 100 - (100 / (1 + money_ratio))
        
        mfi[:period] = np.nan
        
        return mfi
    
    @staticmethod
    def vwap(high: np.ndarray, 
             low: np.ndarray, 
             close: np.ndarray, 
             volume: np.ndarray) -> np.ndarray:
        """
        Volume Weighted Average Price
        
        Args:
            high: Prix hauts
            low: Prix bas
            close: Prix de clÃƒÂ´ture
            volume: Volumes
            
        Returns:
            Array des VWAP
        """
        tp = (high + low + close) / 3
        cumulative_tp_volume = np.cumsum(tp * volume)
        cumulative_volume = np.cumsum(volume)
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        return vwap
    
    @staticmethod
    def support_resistance(high: np.ndarray, 
                          low: np.ndarray, 
                          close: np.ndarray, 
                          window: int = 20) -> Tuple[List[float], List[float]]:
        """
        DÃƒÂ©tecte les niveaux de support et rÃƒÂ©sistance
        
        Args:
            high: Prix hauts
            low: Prix bas  
            close: Prix de clÃƒÂ´ture
            window: FenÃƒÂªtre de dÃƒÂ©tection
            
        Returns:
            Tuple (supports, resistances)
        """
        supports = []
        resistances = []
        
        # DÃƒÂ©tection des pivots
        for i in range(window, len(close) - window):
            # Resistance: high local maximum
            if high[i] == np.max(high[i-window:i+window+1]):
                resistances.append(high[i])
            
            # Support: low local minimum
            if low[i] == np.min(low[i-window:i+window+1]):
                supports.append(low[i])
        
        # Regrouper les niveaux proches (Ã‚Â± 0.5%)
        def cluster_levels(levels, threshold=0.005):
            if not levels:
                return []
            
            sorted_levels = sorted(levels)
            clustered = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clustered.append(np.mean(current_cluster))
            return clustered
        
        supports = cluster_levels(supports)
        resistances = cluster_levels(resistances)
        
        return supports, resistances
    
    @staticmethod
    def fibonacci_retracement(high_point: float, low_point: float) -> dict:
        """
        Calcule les niveaux de retracement de Fibonacci
        
        Args:
            high_point: Point haut
            low_point: Point bas
            
        Returns:
            Dict avec les niveaux Fibonacci
        """
        diff = high_point - low_point
        
        levels = {
            '0.0%': high_point,
            '23.6%': high_point - diff * 0.236,
            '38.2%': high_point - diff * 0.382,
            '50.0%': high_point - diff * 0.5,
            '61.8%': high_point - diff * 0.618,
            '78.6%': high_point - diff * 0.786,
            '100.0%': low_point
        }
        
        return levels
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> dict:
        """
        Calcule les points pivots
        
        Args:
            high: High de la pÃƒÂ©riode prÃƒÂ©cÃƒÂ©dente
            low: Low de la pÃƒÂ©riode prÃƒÂ©cÃƒÂ©dente
            close: Close de la pÃƒÂ©riode prÃƒÂ©cÃƒÂ©dente
            
        Returns:
            Dict avec les niveaux pivots
        """
        pivot = (high + low + close) / 3
        
        levels = {
            'R3': pivot + 2 * (high - low),
            'R2': pivot + (high - low),
            'R1': 2 * pivot - low,
            'PP': pivot,
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low),
            'S3': pivot - 2 * (high - low)
        }
        
        return levels
    
    @staticmethod
    def calculate_all(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """
        Calcule tous les indicateurs sur un DataFrame
        
        Args:
            df: DataFrame avec OHLCV
            config: Configuration des pÃƒÂ©riodes (optionnel)
            
        Returns:
            DataFrame avec tous les indicateurs ajoutÃƒÂ©s
        """
        if config is None:
            config = {
                'rsi_period': 14,
                'ema_fast': 9,
                'ema_slow': 21,
                'bb_period': 20,
                'bb_std': 2,
                'atr_period': 14,
                'adx_period': 14
            }
        
        # Copie pour ne pas modifier l'original
        result = df.copy()
        
        # Prix arrays
        close = result['close'].values
        high = result['high'].values
        low = result['low'].values
        volume = result['volume'].values
        
        # Moving Averages
        result['sma_20'] = TechnicalIndicators.sma(close, 20)
        result['ema_9'] = TechnicalIndicators.ema(close, config['ema_fast'])
        result['ema_21'] = TechnicalIndicators.ema(close, config['ema_slow'])
        
        # RSI
        result['rsi'] = TechnicalIndicators.rsi(close, config['rsi_period'])
        
        # MACD
        macd, signal, hist = TechnicalIndicators.macd(close)
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close, config['bb_period'], config['bb_std'])
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        
        # ATR (volatilitÃƒÂ©)
        result['atr'] = TechnicalIndicators.atr(high, low, close, config['atr_period'])
        
        # Stochastic
        k, d = TechnicalIndicators.stochastic(high, low, close)
        result['stoch_k'] = k
        result['stoch_d'] = d
        
        # ADX (force de tendance)
        result['adx'] = TechnicalIndicators.adx(high, low, close, config['adx_period'])
        
        # OBV (volume)
        result['obv'] = TechnicalIndicators.obv(close, volume)
        
        # MFI (money flow)
        result['mfi'] = TechnicalIndicators.mfi(high, low, close, volume)
        
        # VWAP
        result['vwap'] = TechnicalIndicators.vwap(high, low, close, volume)
        
        logger.info(f"Ã¢Å“â€¦ {len(config)} indicateurs calculÃƒÂ©s")
        
        return result


# =============================================================
# FONCTIONS HELPERS
# =============================================================

def detect_divergence(price: np.ndarray, indicator: np.ndarray, window: int = 14) -> str:
    """
    DÃƒÂ©tecte les divergences entre prix et indicateur
    
    Args:
        price: Array des prix
        indicator: Array de l'indicateur
        window: FenÃƒÂªtre de dÃƒÂ©tection
        
    Returns:
        'bullish', 'bearish' ou 'none'
    """
    if len(price) < window * 2:
        return 'none'
    
    # Prix: lower low mais indicateur: higher low = divergence bullish
    price_lows = []
    indicator_lows = []
    
    for i in range(window, len(price) - window):
        if price[i] == np.min(price[i-window:i+window+1]):
            price_lows.append((i, price[i]))
            indicator_lows.append((i, indicator[i]))
    
    if len(price_lows) >= 2:
        # Compare les deux derniers lows
        if price_lows[-1][1] < price_lows[-2][1]:  # Lower low in price
            if indicator_lows[-1][1] > indicator_lows[-2][1]:  # Higher low in indicator
                return 'bullish'
    
    # Prix: higher high mais indicateur: lower high = divergence bearish
    price_highs = []
    indicator_highs = []
    
    for i in range(window, len(price) - window):
        if price[i] == np.max(price[i-window:i+window+1]):
            price_highs.append((i, price[i]))
            indicator_highs.append((i, indicator[i]))
    
    if len(price_highs) >= 2:
        # Compare les deux derniers highs
        if price_highs[-1][1] > price_highs[-2][1]:  # Higher high in price
            if indicator_highs[-1][1] < indicator_highs[-2][1]:  # Lower high in indicator
                return 'bearish'
    
    return 'none'


def trend_strength(prices: np.ndarray, period: int = 20) -> float:
    """
    Calcule la force de la tendance (0-1)
    
    Args:
        prices: Array des prix
        period: PÃƒÂ©riode d'analyse
        
    Returns:
        Score de 0 (pas de tendance) ÃƒÂ  1 (forte tendance)
    """
    if len(prices) < period:
        return 0.0
    
    recent = prices[-period:]
    
    # RÃƒÂ©gression linÃƒÂ©aire
    x = np.arange(period)
    coeffs = np.polyfit(x, recent, 1)
    slope = coeffs[0]
    
    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((recent - y_pred) ** 2)
    ss_tot = np.sum((recent - np.mean(recent)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    
    # Normaliser le slope
    normalized_slope = abs(slope) / np.mean(recent)
    
    # Combiner RÃ‚Â² et slope
    trend_score = (r_squared * 0.7 + min(normalized_slope * 10, 1) * 0.3)
    
    return max(0, min(1, trend_score))


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test des indicateurs"""
    
    # CrÃƒÂ©er des donnÃƒÂ©es de test
    np.random.seed(42)
    size = 100
    
    # Simuler des prix
    close = 100 + np.cumsum(np.random.randn(size) * 2)
    high = close + np.abs(np.random.randn(size))
    low = close - np.abs(np.random.randn(size))
    volume = np.random.randint(1000, 10000, size)
    
    # CrÃƒÂ©er DataFrame
    df = pd.DataFrame({
        'open': close + np.random.randn(size) * 0.5,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Calculer tous les indicateurs
    indicators = TechnicalIndicators()
    result = indicators.calculate_all(df)
    
    print("Indicateurs calculÃƒÂ©s:")
    print(result[['close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr']].tail(10))
    
    # Test divergence
    divergence = detect_divergence(close, indicators.rsi(close))
    print(f"\nDivergence dÃƒÂ©tectÃƒÂ©e: {divergence}")
    
    # Test trend strength
    strength = trend_strength(close)
    print(f"Force de tendance: {strength:.2%}")