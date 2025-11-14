"""
Pattern Recognition Strategy
DÃƒÂ©tecte les patterns chartistes classiques
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging
from scipy.signal import find_peaks, argrelextrema

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class PatternStrategy(BaseStrategy):
    """
    StratÃƒÂ©gie de reconnaissance de patterns (10% du capital)
    
    Patterns dÃƒÂ©tectÃƒÂ©s:
    - Double Bottom / Double Top
    - Head & Shoulders / Inverse H&S
    - Triangles (ascendant, descendant, symÃƒÂ©trique)
    - Flags & Pennants
    - Support/Resistance breakouts
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise la stratÃƒÂ©gie Pattern
        
        Args:
            config: Configuration de la stratÃƒÂ©gie
        """
        default_config = {
            'name': 'Pattern_Strategy',
            'allocation': 0.10,  # 10% du capital
            'min_confidence': 0.65,
            'lookback_period': 100,  # PÃƒÂ©riode pour dÃƒÂ©tecter patterns
            'tolerance': 0.02,  # TolÃƒÂ©rance pour pattern matching (2%)
            'min_pattern_bars': 20  # Minimum de barres pour un pattern
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Patterns supportÃƒÂ©s
        self.supported_patterns = [
            'double_bottom',
            'double_top',
            'head_shoulders',
            'inverse_head_shoulders',
            'triangle_ascending',
            'triangle_descending',
            'triangle_symmetric',
            'flag_bullish',
            'flag_bearish',
            'pennant'
        ]
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """
        Analyse et dÃƒÂ©tecte les patterns chartistes
        
        Args:
            data: Dict avec df, orderbook, trades, symbol
            
        Returns:
            Signal de trading ou None
        """
        try:
            if not self.is_active:
                return None
            
            df = data.get('df')
            symbol = data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < self.config['lookback_period']:
                return None
            
            # Calculer les indicateurs de base
            df = self.calculate_indicators(df)
            
            # Scanner tous les patterns
            detected_patterns = self._scan_all_patterns(df)
            
            if not detected_patterns:
                return None
            
            # Prendre le pattern avec la plus haute confiance
            best_pattern = max(detected_patterns, key=lambda x: x['confidence'])
            
            # GÃƒÂ©nÃƒÂ©rer le signal
            signal = self._generate_signal_from_pattern(best_pattern, df, symbol)
            
            if signal and self.validate_signal(signal):
                self.performance['total_signals'] += 1
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur Pattern analyze: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs pour pattern detection
        
        Args:
            df: DataFrame avec OHLCV
            
        Returns:
            DataFrame avec indicateurs
        """
        df = df.copy()
        
        # Volume moyen
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Moyennes mobiles
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # ATR pour stops
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
    
    def _scan_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scanne tous les patterns possibles
        
        Args:
            df: DataFrame avec prix
            
        Returns:
            Liste des patterns dÃƒÂ©tectÃƒÂ©s
        """
        detected = []
        
        # Trouver les pivots (highs et lows locaux)
        pivots = self._find_pivots(df)
        
        # Double Bottom/Top
        pattern = self._detect_double_bottom(df, pivots)
        if pattern:
            detected.append(pattern)
        
        pattern = self._detect_double_top(df, pivots)
        if pattern:
            detected.append(pattern)
        
        # Head & Shoulders
        pattern = self._detect_head_shoulders(df, pivots)
        if pattern:
            detected.append(pattern)
        
        pattern = self._detect_inverse_head_shoulders(df, pivots)
        if pattern:
            detected.append(pattern)
        
        # Triangles
        pattern = self._detect_triangle(df, pivots)
        if pattern:
            detected.append(pattern)
        
        # Flags & Pennants
        pattern = self._detect_flag(df, pivots)
        if pattern:
            detected.append(pattern)
        
        return detected
    
    def _find_pivots(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[int]]:
        """
        Trouve les pivots highs et lows
        
        Args:
            df: DataFrame avec prix
            window: FenÃƒÂªtre pour dÃƒÂ©tection
            
        Returns:
            Dict avec indices des pivots
        """
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Pivot High
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                highs.append(i)
            
            # Pivot Low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                lows.append(i)
        
        return {'highs': highs, 'lows': lows}
    
    def _detect_double_bottom(self, df: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        DÃƒÂ©tecte un pattern Double Bottom
        
        Args:
            df: DataFrame
            pivots: Pivots dÃƒÂ©tectÃƒÂ©s
            
        Returns:
            Pattern ou None
        """
        lows = pivots['lows']
        
        if len(lows) < 2:
            return None
        
        # Prendre les 2 derniers lows
        low1_idx = lows[-2]
        low2_idx = lows[-1]
        
        # VÃƒÂ©rifier qu'ils sont proches en prix (Ã‚Â±2%)
        low1_price = df['low'].iloc[low1_idx]
        low2_price = df['low'].iloc[low2_idx]
        
        price_diff = abs(low1_price - low2_price) / low1_price
        
        if price_diff > self.config['tolerance']:
            return None
        
        # VÃƒÂ©rifier qu'il y a un high entre les deux
        between_highs = [h for h in pivots['highs'] if low1_idx < h < low2_idx]
        
        if not between_highs:
            return None
        
        # VÃƒÂ©rifier que le prix actuel est au-dessus du neckline
        neckline = df['high'].iloc[between_highs].max()
        current_price = df['close'].iloc[-1]
        
        if current_price > neckline:
            # Breakout confirmÃƒÂ©
            return {
                'pattern': 'double_bottom',
                'confidence': 0.70,
                'direction': 'BUY',
                'entry_price': current_price,
                'target': current_price + (neckline - low2_price),  # Projection
                'stop': low2_price * 0.98,
                'neckline': neckline
            }
        
        return None
    
    def _detect_double_top(self, df: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        DÃƒÂ©tecte un pattern Double Top
        
        Args:
            df: DataFrame
            pivots: Pivots dÃƒÂ©tectÃƒÂ©s
            
        Returns:
            Pattern ou None
        """
        highs = pivots['highs']
        
        if len(highs) < 2:
            return None
        
        # Prendre les 2 derniers highs
        high1_idx = highs[-2]
        high2_idx = highs[-1]
        
        # VÃƒÂ©rifier qu'ils sont proches en prix (Ã‚Â±2%)
        high1_price = df['high'].iloc[high1_idx]
        high2_price = df['high'].iloc[high2_idx]
        
        price_diff = abs(high1_price - high2_price) / high1_price
        
        if price_diff > self.config['tolerance']:
            return None
        
        # VÃƒÂ©rifier qu'il y a un low entre les deux
        between_lows = [l for l in pivots['lows'] if high1_idx < l < high2_idx]
        
        if not between_lows:
            return None
        
        # VÃƒÂ©rifier que le prix actuel est en dessous du neckline
        neckline = df['low'].iloc[between_lows].min()
        current_price = df['close'].iloc[-1]
        
        if current_price < neckline:
            # Breakdown confirmÃƒÂ©
            return {
                'pattern': 'double_top',
                'confidence': 0.70,
                'direction': 'SELL',
                'entry_price': current_price,
                'target': current_price - (high2_price - neckline),  # Projection
                'stop': high2_price * 1.02,
                'neckline': neckline
            }
        
        return None
    
    def _detect_head_shoulders(self, df: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        DÃƒÂ©tecte un pattern Head & Shoulders
        
        Args:
            df: DataFrame
            pivots: Pivots dÃƒÂ©tectÃƒÂ©s
            
        Returns:
            Pattern ou None
        """
        highs = pivots['highs']
        
        if len(highs) < 3:
            return None
        
        # Prendre les 3 derniers highs
        left_shoulder_idx = highs[-3]
        head_idx = highs[-2]
        right_shoulder_idx = highs[-1]
        
        left_shoulder = df['high'].iloc[left_shoulder_idx]
        head = df['high'].iloc[head_idx]
        right_shoulder = df['high'].iloc[right_shoulder_idx]
        
        # VÃƒÂ©rifier structure: head > shoulders
        if not (head > left_shoulder and head > right_shoulder):
            return None
        
        # VÃƒÂ©rifier que les ÃƒÂ©paules sont similaires (Ã‚Â±2%)
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > self.config['tolerance']:
            return None
        
        # Trouver les lows entre les highs pour neckline
        lows_between = [l for l in pivots['lows'] if left_shoulder_idx < l < right_shoulder_idx]
        
        if len(lows_between) < 2:
            return None
        
        # Neckline approximative
        neckline = np.mean([df['low'].iloc[l] for l in lows_between[:2]])
        current_price = df['close'].iloc[-1]
        
        if current_price < neckline:
            # Breakdown
            return {
                'pattern': 'head_shoulders',
                'confidence': 0.75,
                'direction': 'SELL',
                'entry_price': current_price,
                'target': current_price - (head - neckline),
                'stop': right_shoulder * 1.02,
                'neckline': neckline
            }
        
        return None
    
    def _detect_inverse_head_shoulders(self, df: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        DÃƒÂ©tecte un Inverse Head & Shoulders
        
        Args:
            df: DataFrame
            pivots: Pivots dÃƒÂ©tectÃƒÂ©s
            
        Returns:
            Pattern ou None
        """
        lows = pivots['lows']
        
        if len(lows) < 3:
            return None
        
        # Prendre les 3 derniers lows
        left_shoulder_idx = lows[-3]
        head_idx = lows[-2]
        right_shoulder_idx = lows[-1]
        
        left_shoulder = df['low'].iloc[left_shoulder_idx]
        head = df['low'].iloc[head_idx]
        right_shoulder = df['low'].iloc[right_shoulder_idx]
        
        # VÃƒÂ©rifier structure: head < shoulders
        if not (head < left_shoulder and head < right_shoulder):
            return None
        
        # VÃƒÂ©rifier que les ÃƒÂ©paules sont similaires
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > self.config['tolerance']:
            return None
        
        # Trouver les highs pour neckline
        highs_between = [h for h in pivots['highs'] if left_shoulder_idx < h < right_shoulder_idx]
        
        if len(highs_between) < 2:
            return None
        
        neckline = np.mean([df['high'].iloc[h] for h in highs_between[:2]])
        current_price = df['close'].iloc[-1]
        
        if current_price > neckline:
            # Breakout
            return {
                'pattern': 'inverse_head_shoulders',
                'confidence': 0.75,
                'direction': 'BUY',
                'entry_price': current_price,
                'target': current_price + (neckline - head),
                'stop': right_shoulder * 0.98,
                'neckline': neckline
            }
        
        return None
    
    def _detect_triangle(self, df: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        DÃƒÂ©tecte un triangle (ascendant/descendant/symÃƒÂ©trique)
        
        Args:
            df: DataFrame
            pivots: Pivots dÃƒÂ©tectÃƒÂ©s
            
        Returns:
            Pattern ou None
        """
        lookback = min(50, len(df))
        recent_df = df.iloc[-lookback:]
        
        # Lignes de tendance
        highs_line = self._fit_trendline(recent_df, 'high', pivots['highs'])
        lows_line = self._fit_trendline(recent_df, 'low', pivots['lows'])
        
        if highs_line is None or lows_line is None:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Triangle ascendant: rÃƒÂ©sistance horizontale, support montant
        if abs(highs_line['slope']) < 0.001 and lows_line['slope'] > 0:
            resistance = highs_line['level']
            if current_price > resistance:
                return {
                    'pattern': 'triangle_ascending',
                    'confidence': 0.65,
                    'direction': 'BUY',
                    'entry_price': current_price,
                    'target': current_price + (current_price - lows_line['level']) * 0.5,
                    'stop': lows_line['level']
                }
        
        # Triangle descendant: support horizontal, rÃƒÂ©sistance descendante
        elif abs(lows_line['slope']) < 0.001 and highs_line['slope'] < 0:
            support = lows_line['level']
            if current_price < support:
                return {
                    'pattern': 'triangle_descending',
                    'confidence': 0.65,
                    'direction': 'SELL',
                    'entry_price': current_price,
                    'target': current_price - (highs_line['level'] - current_price) * 0.5,
                    'stop': highs_line['level']
                }
        
        # Triangle symÃƒÂ©trique: convergence
        elif lows_line['slope'] > 0 and highs_line['slope'] < 0:
            # Attendre le breakout
            resistance = highs_line['level']
            support = lows_line['level']
            
            if current_price > resistance:
                return {
                    'pattern': 'triangle_symmetric',
                    'confidence': 0.60,
                    'direction': 'BUY',
                    'entry_price': current_price,
                    'target': current_price + (resistance - support),
                    'stop': support
                }
            elif current_price < support:
                return {
                    'pattern': 'triangle_symmetric',
                    'confidence': 0.60,
                    'direction': 'SELL',
                    'entry_price': current_price,
                    'target': current_price - (resistance - support),
                    'stop': resistance
                }
        
        return None
    
    def _detect_flag(self, df: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        DÃƒÂ©tecte un flag (continuation pattern)
        
        Args:
            df: DataFrame
            pivots: Pivots dÃƒÂ©tectÃƒÂ©s
            
        Returns:
            Pattern ou None
        """
        # Flag = forte hausse/baisse suivie d'une consolidation
        lookback = 30
        if len(df) < lookback:
            return None
        
        recent = df.iloc[-lookback:]
        
        # DÃƒÂ©tecter le pole (mouvement fort)
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Flag haussier: forte hausse puis consolidation
        if price_change > 0.03:  # 3% hausse
            # VÃƒÂ©rifier consolidation (canal descendant lÃƒÂ©ger)
            consolidation_prices = recent['close'].iloc[-10:]
            if consolidation_prices.iloc[-1] > consolidation_prices.iloc[0]:
                # Breakout de la consolidation
                return {
                    'pattern': 'flag_bullish',
                    'confidence': 0.65,
                    'direction': 'BUY',
                    'entry_price': recent['close'].iloc[-1],
                    'target': recent['close'].iloc[-1] + abs(price_change * recent['close'].iloc[0]),
                    'stop': recent['low'].iloc[-10:].min()
                }
        
        # Flag baissier: forte baisse puis consolidation
        elif price_change < -0.03:
            consolidation_prices = recent['close'].iloc[-10:]
            if consolidation_prices.iloc[-1] < consolidation_prices.iloc[0]:
                return {
                    'pattern': 'flag_bearish',
                    'confidence': 0.65,
                    'direction': 'SELL',
                    'entry_price': recent['close'].iloc[-1],
                    'target': recent['close'].iloc[-1] - abs(price_change * recent['close'].iloc[0]),
                    'stop': recent['high'].iloc[-10:].max()
                }
        
        return None
    
    def _fit_trendline(self, df: pd.DataFrame, price_type: str, pivot_indices: List[int]) -> Optional[Dict]:
        """
        Fit une ligne de tendance sur les pivots
        
        Args:
            df: DataFrame
            price_type: 'high' ou 'low'
            pivot_indices: Indices des pivots
            
        Returns:
            Dict avec slope et level ou None
        """
        if len(pivot_indices) < 2:
            return None
        
        # Filtrer les pivots dans le range du df
        valid_pivots = [p for p in pivot_indices if p < len(df)]
        
        if len(valid_pivots) < 2:
            return None
        
        # Prendre les derniers pivots
        recent_pivots = valid_pivots[-3:]
        
        x = np.array(recent_pivots)
        y = np.array([df[price_type].iloc[i] for i in recent_pivots])
        
        # RÃƒÂ©gression linÃƒÂ©aire simple
        if len(x) >= 2:
            slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
            level = y[-1]
            
            return {
                'slope': slope,
                'level': level,
                'points': list(zip(x, y))
            }
        
        return None
    
    def _generate_signal_from_pattern(self, pattern: Dict, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        GÃƒÂ©nÃƒÂ¨re un signal de trading ÃƒÂ  partir d'un pattern
        
        Args:
            pattern: Pattern dÃƒÂ©tectÃƒÂ©
            df: DataFrame
            symbol: Symbole
            
        Returns:
            Signal ou None
        """
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
            
            # Ajuster les stops/targets avec ATR si nÃƒÂ©cessaire
            if 'stop' not in pattern or pattern['stop'] == 0:
                if pattern['direction'] == 'BUY':
                    pattern['stop'] = current_price - (atr * 2)
                else:
                    pattern['stop'] = current_price + (atr * 2)
            
            if 'target' not in pattern or pattern['target'] == 0:
                if pattern['direction'] == 'BUY':
                    pattern['target'] = current_price + (atr * 3)
                else:
                    pattern['target'] = current_price - (atr * 3)
            
            signal = {
                'type': 'ENTRY',
                'side': pattern['direction'],
                'price': pattern.get('entry_price', current_price),
                'confidence': pattern['confidence'],
                'stop_loss': pattern['stop'],
                'take_profit': pattern['target'],
                'reasons': [
                    f"Pattern dÃƒÂ©tectÃƒÂ©: {pattern['pattern']}",
                    f"Confiance: {pattern['confidence']:.2%}",
                    f"Direction: {pattern['direction']}"
                ],
                'metadata': {
                    'strategy': self.name,
                    'pattern_type': pattern['pattern'],
                    'pattern_data': pattern
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Erreur gÃƒÂ©nÃƒÂ©ration signal from pattern: {e}")
            return None


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de la stratÃƒÂ©gie Pattern"""
    
    config = {
        'min_confidence': 0.65
    }
    
    strategy = PatternStrategy(config)
    
    # DonnÃƒÂ©es de test avec un double bottom
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    
    # CrÃƒÂ©er un double bottom artificiel
    prices = np.ones(200) * 100
    prices[50] = 95  # Premier low
    prices[100] = 95  # DeuxiÃƒÂ¨me low
    prices[75] = 105  # High entre les deux
    prices[150:] = np.linspace(96, 106, 50)  # Breakout
    
    test_df = pd.DataFrame({
        'open': prices + np.random.randn(200) * 0.5,
        'high': prices + abs(np.random.randn(200) * 1),
        'low': prices - abs(np.random.randn(200) * 1),
        'close': prices + np.random.randn(200) * 0.5,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    data = {
        'df': test_df,
        'symbol': 'BTCUSDC'
    }
    
    print("Test Pattern Strategy")
    print("=" * 50)
    print(f"StratÃƒÂ©gie: {strategy.name}")
    print(f"Active: {strategy.is_active}")
    print(f"Patterns supportÃƒÂ©s: {len(strategy.supported_patterns)}")
    
    signal = strategy.analyze(data)
    
    if signal:
        print(f"\nÃ¢Å“â€¦ Signal dÃƒÂ©tectÃƒÂ©!")
        print(f"Type: {signal['side']}")
        print(f"Prix: {signal['price']:.2f}")
        print(f"Confiance: {signal['confidence']:.2%}")
        print(f"Stop Loss: {signal['stop_loss']:.2f}")
        print(f"Take Profit: {signal['take_profit']:.2f}")
        if 'pattern_type' in signal.get('metadata', {}):
            print(f"Pattern: {signal['metadata']['pattern_type']}")
    else:
        print("\nÃ¢ÂÅ’ Aucun pattern dÃƒÂ©tectÃƒÂ©")