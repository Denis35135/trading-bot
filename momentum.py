"""
StratÃƒÂ©gie Momentum pour The Bot
Capture les mouvements directionnels forts et les breakouts
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from strategies.base_strategy import BaseStrategy
from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    StratÃƒÂ©gie Momentum/Breakout
    
    CaractÃƒÂ©ristiques:
    - Identifie les breakouts de ranges
    - Suit les tendances fortes
    - Entre sur accÃƒÂ©lÃƒÂ©ration du momentum
    - Profits visÃƒÂ©s: 1-3%
    - Holding: 15min ÃƒÂ  plusieurs heures
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise la stratÃƒÂ©gie Momentum
        
        Args:
            config: Configuration personnalisÃƒÂ©e
        """
        # Configuration par dÃƒÂ©faut spÃƒÂ©cifique au momentum
        default_momentum_config = {
            'min_confidence': 0.70,
            'lookback_periods': 50,
            'breakout_periods': 20,
            'volume_threshold': 2.0,  # Volume 2x la moyenne
            'momentum_threshold': 0.015,  # 1.5% mouvement minimum
            'adx_threshold': 25,  # ADX minimum pour trend
            'rsi_range': [55, 80],  # RSI optimal pour momentum
            'profit_target': 0.02,  # 2% profit target
            'stop_loss': 0.01,  # 1% stop loss
            'trailing_stop': True,
            'trailing_stop_distance': 0.005,  # 0.5%
            'max_holding_hours': 4,
            'use_volume_confirmation': True,
            'use_multi_timeframe': True
        }
        
        # Merger avec config fournie
        if config:
            default_momentum_config.update(config)
        
        super().__init__("momentum", default_momentum_config)
        
        # Indicateurs techniques
        self.indicators = TechnicalIndicators()
        
        # Ãƒâ€°tat spÃƒÂ©cifique au momentum
        self.breakout_levels = {}
        self.trend_strength = {}
        self.momentum_score = {}
        
        logger.info("StratÃƒÂ©gie Momentum initialisÃƒÂ©e")
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """
        Analyse les donnÃƒÂ©es pour dÃƒÂ©tecter des opportunitÃƒÂ©s momentum
        
        Args:
            data: DonnÃƒÂ©es de marchÃƒÂ©
            
        Returns:
            Signal de trading ou None
        """
        try:
            df = data.get('df')
            if df is None or len(df) < self.config['lookback_periods']:
                return None
            
            symbol = data.get('symbol', 'UNKNOWN')
            orderbook = data.get('orderbook', {})
            
            # Calculer les indicateurs
            df = self.calculate_indicators(df)
            
            # DonnÃƒÂ©es actuelles
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # VÃƒÂ©rifier si on a dÃƒÂ©jÃƒÂ  une position
            if self.has_position(symbol):
                # GÃƒÂ©rer la position existante
                return self._check_exit_conditions(symbol, df, current)
            else:
                # Chercher une entrÃƒÂ©e
                signal = self._check_entry_conditions(df, current, prev, orderbook)
                
                if signal:
                    signal['symbol'] = symbol
                    
                    # Valider et logger
                    if self.validate_signal(signal):
                        self.log_signal(signal)
                        return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur analyse momentum: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs pour la stratÃƒÂ©gie momentum
        
        Args:
            df: DataFrame avec OHLCV
            
        Returns:
            DataFrame avec indicateurs
        """
        # Copier pour ne pas modifier l'original
        df = df.copy()
        
        # Prix
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Moving Averages
        df['sma_20'] = self.indicators.sma(close, 20)
        df['sma_50'] = self.indicators.sma(close, 50)
        df['ema_9'] = self.indicators.ema(close, 9)
        df['ema_21'] = self.indicators.ema(close, 21)
        
        # RSI
        df['rsi'] = self.indicators.rsi(close, 14)
        
        # MACD
        macd, signal, hist = self.indicators.macd(close)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # ADX (force de la tendance)
        df['adx'] = self.indicators.adx(high, low, close, 14)
        
        # ATR (volatilitÃƒÂ©)
        df['atr'] = self.indicators.atr(high, low, close, 14)
        
        # Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(close, 20, 2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = upper - lower
        
        # Volume indicators
        df['volume_ma'] = self.indicators.sma(volume, 20)
        df['volume_ratio'] = volume / df['volume_ma']
        df['obv'] = self.indicators.obv(close, volume)
        
        # Momentum indicators
        df['roc'] = self._calculate_roc(close, 10)  # Rate of Change
        df['momentum'] = close - pd.Series(close).shift(10)
        
        # Support/Resistance
        self._calculate_support_resistance(df)
        
        # Breakout detection
        df['highest_20'] = pd.Series(high).rolling(20).max()
        df['lowest_20'] = pd.Series(low).rolling(20).min()
        df['is_breakout_up'] = close > df['highest_20'].shift(1)
        df['is_breakout_down'] = close < df['lowest_20'].shift(1)
        
        return df
    
    def _check_entry_conditions(self, df: pd.DataFrame, current: pd.Series, 
                               prev: pd.Series, orderbook: Dict) -> Optional[Dict]:
        """
        VÃƒÂ©rifie les conditions d'entrÃƒÂ©e momentum
        
        Returns:
            Signal d'entrÃƒÂ©e ou None
        """
        signals = {
            'long': [],
            'short': []
        }
        
        # ===========================================
        # CONDITIONS LONG (Momentum haussier)
        # ===========================================
        
        # 1. Breakout haussier avec volume
        if current['is_breakout_up'] and current['volume_ratio'] > self.config['volume_threshold']:
            signals['long'].append(('breakout_volume', 0.85))
        
        # 2. Trend fort (ADX) avec momentum
        if current['adx'] > self.config['adx_threshold']:
            if current['close'] > current['sma_20'] > current['sma_50']:
                if current['roc'] > self.config['momentum_threshold'] * 100:
                    signals['long'].append(('strong_uptrend', 0.80))
        
        # 3. MACD crossover avec momentum
        if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
            if current['macd_hist'] > prev['macd_hist'] * 1.5:  # AccÃƒÂ©lÃƒÂ©ration
                signals['long'].append(('macd_momentum', 0.75))
        
        # 4. Squeeze Bollinger + Breakout
        bb_squeeze = current['bb_width'] < df['bb_width'].rolling(20).mean().iloc[-1] * 0.8
        if bb_squeeze and current['close'] > current['bb_upper']:
            signals['long'].append(('bb_squeeze_breakout', 0.70))
        
        # 5. Volume spike avec prix en hausse
        if current['volume_ratio'] > 3.0 and current['close'] > prev['close'] * 1.01:
            signals['long'].append(('volume_spike_up', 0.75))
        
        # 6. RSI momentum (pas oversold, momentum positif)
        if self.config['rsi_range'][0] < current['rsi'] < self.config['rsi_range'][1]:
            if current['rsi'] > prev['rsi'] + 5:  # RSI en accÃƒÂ©lÃƒÂ©ration
                signals['long'].append(('rsi_momentum', 0.65))
        
        # 7. Multiple MA alignment
        if (current['ema_9'] > current['ema_21'] > current['sma_20'] > current['sma_50']):
            signals['long'].append(('ma_alignment', 0.70))
        
        # ===========================================
        # CONDITIONS SHORT (Momentum baissier)
        # ===========================================
        
        # 1. Breakout baissier avec volume
        if current['is_breakout_down'] and current['volume_ratio'] > self.config['volume_threshold']:
            signals['short'].append(('breakdown_volume', 0.85))
        
        # 2. Trend fort baissier
        if current['adx'] > self.config['adx_threshold']:
            if current['close'] < current['sma_20'] < current['sma_50']:
                if current['roc'] < -self.config['momentum_threshold'] * 100:
                    signals['short'].append(('strong_downtrend', 0.80))
        
        # 3. MACD crossover baissier avec momentum
        if prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
            if current['macd_hist'] < prev['macd_hist'] * 1.5:
                signals['short'].append(('macd_momentum_down', 0.75))
        
        # 4. Rejet Bollinger upper avec volume
        if current['close'] < current['bb_upper'] and prev['close'] > prev['bb_upper']:
            if current['volume_ratio'] > 2.0:
                signals['short'].append(('bb_rejection', 0.70))
        
        # 5. Volume spike avec prix en baisse
        if current['volume_ratio'] > 3.0 and current['close'] < prev['close'] * 0.99:
            signals['short'].append(('volume_spike_down', 0.75))
        
        # ===========================================
        # GÃƒâ€°NÃƒâ€°RATION DU SIGNAL
        # ===========================================
        
        # Calculer les scores
        long_score = sum(score for _, score in signals['long']) / max(len(signals['long']), 1)
        short_score = sum(score for _, score in signals['short']) / max(len(signals['short']), 1)
        
        # Nombre minimum de confirmations
        min_confirmations = 2
        
        # Signal LONG
        if len(signals['long']) >= min_confirmations and long_score >= self.config['min_confidence']:
            entry_price = current['close']
            
            # Calculer les targets basÃƒÂ©s sur ATR
            atr = current['atr']
            take_profit = entry_price * (1 + self.config['profit_target'])
            stop_loss = entry_price - (1.5 * atr)  # Stop basÃƒÂ© sur volatilitÃƒÂ©
            
            # Ajuster si rÃƒÂ©sistance proche
            if 'resistance' in df.columns and not pd.isna(current['resistance']):
                if current['resistance'] < take_profit:
                    take_profit = current['resistance'] * 0.995
            
            return self.create_signal(
                signal_type='ENTRY',
                side='BUY',
                price=entry_price,
                confidence=long_score,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reasons=signals['long'],
                metadata={
                    'adx': current['adx'],
                    'volume_ratio': current['volume_ratio'],
                    'roc': current['roc']
                }
            )
        
        # Signal SHORT
        if len(signals['short']) >= min_confirmations and short_score >= self.config['min_confidence']:
            entry_price = current['close']
            
            # Calculer les targets
            atr = current['atr']
            take_profit = entry_price * (1 - self.config['profit_target'])
            stop_loss = entry_price + (1.5 * atr)
            
            # Ajuster si support proche
            if 'support' in df.columns and not pd.isna(current['support']):
                if current['support'] > take_profit:
                    take_profit = current['support'] * 1.005
            
            return self.create_signal(
                signal_type='ENTRY',
                side='SELL',
                price=entry_price,
                confidence=short_score,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reasons=signals['short'],
                metadata={
                    'adx': current['adx'],
                    'volume_ratio': current['volume_ratio'],
                    'roc': current['roc']
                }
            )
        
        return None
    
    def _check_exit_conditions(self, symbol: str, df: pd.DataFrame, current: pd.Series) -> Optional[Dict]:
        """
        VÃƒÂ©rifie les conditions de sortie pour une position momentum
        
        Returns:
            Signal de sortie ou None
        """
        position_key = f"{symbol}_{self.name}"
        if position_key not in self.positions:
            return None
        
        position = self.positions[position_key]
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        side = position['side']
        
        current_price = current['close']
        profit_pct = (current_price - entry_price) / entry_price if side == 'BUY' else (entry_price - current_price) / entry_price
        
        # Conditions de sortie
        exit_reasons = []
        
        # 1. Take Profit atteint
        if profit_pct >= self.config['profit_target']:
            exit_reasons.append(('take_profit', 1.0))
        
        # 2. Stop Loss atteint
        if profit_pct <= -self.config['stop_loss']:
            exit_reasons.append(('stop_loss', 1.0))
        
        # 3. Momentum s'essouffle
        if side == 'BUY':
            # Long: sortir si momentum devient nÃƒÂ©gatif
            if current['roc'] < 0 and current['macd_hist'] < 0:
                exit_reasons.append(('momentum_exhausted', 0.8))
            
            # RSI overbought
            if current['rsi'] > 85:
                exit_reasons.append(('rsi_overbought', 0.7))
            
            # Prix sous EMA9 (perte de momentum court terme)
            if current['close'] < current['ema_9']:
                exit_reasons.append(('below_ema9', 0.6))
        else:
            # Short: sortir si momentum devient positif
            if current['roc'] > 0 and current['macd_hist'] > 0:
                exit_reasons.append(('momentum_reversed', 0.8))
            
            # RSI oversold
            if current['rsi'] < 15:
                exit_reasons.append(('rsi_oversold', 0.7))
            
            # Prix au-dessus EMA9
            if current['close'] > current['ema_9']:
                exit_reasons.append(('above_ema9', 0.6))
        
        # 4. Temps maximum atteint
        hours_held = (datetime.now() - entry_time).seconds / 3600
        if hours_held >= self.config['max_holding_hours']:
            exit_reasons.append(('max_time', 0.9))
        
        # 5. Trailing stop (si activÃƒÂ© et en profit)
        if self.config['trailing_stop'] and profit_pct > self.config['trailing_stop_distance']:
            # Mettre ÃƒÂ  jour le peak profit
            if profit_pct > position.get('peak_profit', 0):
                position['peak_profit'] = profit_pct
            
            # VÃƒÂ©rifier trailing stop
            drawdown_from_peak = position['peak_profit'] - profit_pct
            if drawdown_from_peak >= self.config['trailing_stop_distance']:
                exit_reasons.append(('trailing_stop', 0.95))
        
        # DÃƒÂ©cision de sortie
        if exit_reasons:
            # Prendre la raison avec le score le plus ÃƒÂ©levÃƒÂ©
            main_reason = max(exit_reasons, key=lambda x: x[1])
            
            exit_side = 'SELL' if side == 'BUY' else 'BUY'
            
            return self.create_signal(
                signal_type='EXIT',
                side=exit_side,
                price=current_price,
                confidence=main_reason[1],
                symbol=symbol,
                reasons=exit_reasons,
                metadata={
                    'profit_pct': profit_pct,
                    'holding_time_hours': hours_held,
                    'exit_reason': main_reason[0]
                }
            )
        
        return None
    
    def _calculate_roc(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Calcule le Rate of Change (ROC)
        
        Args:
            prices: Array des prix
            period: PÃƒÂ©riode de calcul
            
        Returns:
            Array des ROC en %
        """
        roc = np.zeros_like(prices)
        roc[:period] = np.nan
        
        for i in range(period, len(prices)):
            if prices[i-period] != 0:
                roc[i] = ((prices[i] - prices[i-period]) / prices[i-period]) * 100
        
        return roc
    
    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20):
        """
        Calcule les niveaux de support et rÃƒÂ©sistance
        
        Args:
            df: DataFrame avec OHLCV
            window: FenÃƒÂªtre pour dÃƒÂ©tecter les pivots
        """
        high = df['high'].values
        low = df['low'].values
        
        # Initialiser les colonnes
        df['support'] = np.nan
        df['resistance'] = np.nan
        
        # Trouver les pivots
        for i in range(window, len(df) - window):
            # Resistance: high local maximum
            if high[i] == max(high[i-window:i+window+1]):
                df.loc[df.index[i:], 'resistance'] = high[i]
            
            # Support: low local minimum
            if low[i] == min(low[i-window:i+window+1]):
                df.loc[df.index[i:], 'support'] = low[i]
        
        # Forward fill pour avoir toujours une valeur
        df['support'].fillna(method='ffill', inplace=True)
        df['resistance'].fillna(method='ffill', inplace=True)