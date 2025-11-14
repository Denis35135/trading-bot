"""
StratÃƒÂ©gie Mean Reversion pour The Bot
Trade les retours ÃƒÂ  la moyenne aprÃƒÂ¨s des mouvements extrÃƒÂªmes
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats

from strategies.base_strategy import BaseStrategy
from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    StratÃƒÂ©gie Mean Reversion
    
    CaractÃƒÂ©ristiques:
    - Identifie les dÃƒÂ©viations extrÃƒÂªmes
    - Trade le retour vers la moyenne
    - Utilise Bollinger, RSI, et z-score
    - Profits visÃƒÂ©s: 0.5-1.5%
    - Holding: 30min ÃƒÂ  quelques heures
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise la stratÃƒÂ©gie Mean Reversion
        
        Args:
            config: Configuration personnalisÃƒÂ©e
        """
        # Configuration par dÃƒÂ©faut
        default_config = {
            'min_confidence': 0.68,
            'lookback_periods': 60,
            'bb_periods': 20,
            'bb_std_entry': 2.0,  # EntrÃƒÂ©e ÃƒÂ  2 std
            'bb_std_exit': 0.5,   # Sortie ÃƒÂ  0.5 std (proche moyenne)
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'z_score_threshold': 2.0,
            'min_deviation': 0.02,  # 2% minimum de la moyenne
            'profit_target': 0.01,   # 1% profit
            'stop_loss': 0.015,      # 1.5% stop
            'max_holding_hours': 6,
            'use_volume_filter': True,
            'volume_threshold': 1.5,  # Volume 1.5x moyenne
            'use_divergence': True,
            'mean_types': ['sma', 'ema', 'vwap'],  # Types de moyennes ÃƒÂ  utiliser
            'correlation_check': True  # VÃƒÂ©rifier corrÃƒÂ©lation avec BTC
        }
        
        # Merger avec config fournie
        if config:
            default_config.update(config)
        
        super().__init__("mean_reversion", default_config)
        
        # Indicateurs
        self.indicators = TechnicalIndicators()
        
        # Ãƒâ€°tat spÃƒÂ©cifique
        self.mean_levels = {}
        self.deviation_history = {}
        self.correlation_cache = {}
        
        logger.info("StratÃƒÂ©gie Mean Reversion initialisÃƒÂ©e")
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """
        Analyse les donnÃƒÂ©es pour dÃƒÂ©tecter des opportunitÃƒÂ©s de mean reversion
        
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
            
            # DonnÃƒÂ©es actuelles et prÃƒÂ©cÃƒÂ©dentes
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # VÃƒÂ©rifier position existante
            if self.has_position(symbol):
                return self._check_exit_conditions(symbol, df, current)
            else:
                signal = self._check_entry_conditions(df, current, prev, orderbook)
                
                if signal:
                    signal['symbol'] = symbol
                    
                    if self.validate_signal(signal):
                        self.log_signal(signal)
                        return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur analyse mean reversion: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs pour mean reversion
        
        Args:
            df: DataFrame avec OHLCV
            
        Returns:
            DataFrame avec indicateurs
        """
        df = df.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Moyennes mobiles
        df['sma_20'] = self.indicators.sma(close, 20)
        df['sma_50'] = self.indicators.sma(close, 50)
        df['ema_20'] = self.indicators.ema(close, 20)
        
        # VWAP
        df['vwap'] = self.indicators.vwap(high, low, close, volume)
        
        # Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(
            close, 
            self.config['bb_periods'], 
            self.config['bb_std_entry']
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = upper - lower
        
        # Position dans Bollinger (0 = lower, 1 = upper)
        df['bb_position'] = np.where(
            df['bb_width'] > 0,
            (close - lower) / df['bb_width'],
            0.5
        )
        
        # RSI
        df['rsi'] = self.indicators.rsi(close, 14)
        
        # Stochastic
        k, d = self.indicators.stochastic(high, low, close, 14)
        df['stoch_k'] = k
        df['stoch_d'] = d
        
        # Z-Score (ÃƒÂ©cart ÃƒÂ  la moyenne en ÃƒÂ©carts-types)
        df['z_score_20'] = self._calculate_z_score(close, 20)
        df['z_score_50'] = self._calculate_z_score(close, 50)
        
        # DÃƒÂ©viation de la moyenne
        df['deviation_sma20'] = (close - df['sma_20']) / df['sma_20']
        df['deviation_ema20'] = (close - df['ema_20']) / df['ema_20']
        df['deviation_vwap'] = (close - df['vwap']) / df['vwap']
        
        # Mean reversion indicator (custom)
        df['mean_reversion_score'] = self._calculate_mean_reversion_score(df)
        
        # Volume
        df['volume_ma'] = self.indicators.sma(volume, 20)
        df['volume_ratio'] = volume / df['volume_ma']
        
        # ATR pour volatilitÃƒÂ©
        df['atr'] = self.indicators.atr(high, low, close, 14)
        df['atr_percent'] = df['atr'] / close
        
        # Divergences
        if self.config['use_divergence']:
            df['rsi_divergence'] = self._detect_divergence(close, df['rsi'].values)
            df['stoch_divergence'] = self._detect_divergence(close, df['stoch_k'].values)
        
        return df
    
    def _check_entry_conditions(self, df: pd.DataFrame, current: pd.Series,
                               prev: pd.Series, orderbook: Dict) -> Optional[Dict]:
        """
        VÃƒÂ©rifie les conditions d'entrÃƒÂ©e mean reversion
        """
        signals = {
            'long': [],
            'short': []
        }
        
        # ===========================================
        # CONDITIONS LONG (Oversold Ã¢â€ â€™ Retour haussier)
        # ===========================================
        
        # 1. Prix sous Bollinger lower
        if current['close'] <= current['bb_lower']:
            if current['close'] > prev['close']:  # DÃƒÂ©but de rebond
                signals['long'].append(('bb_oversold_bounce', 0.85))
        
        # 2. RSI oversold avec divergence
        if current['rsi'] < self.config['rsi_oversold']:
            if current['rsi'] > prev['rsi']:  # RSI remonte
                signals['long'].append(('rsi_oversold_reversal', 0.80))
            
            if self.config['use_divergence'] and current.get('rsi_divergence') == 'bullish':
                signals['long'].append(('rsi_bullish_divergence', 0.90))
        
        # 3. Z-score extrÃƒÂªme nÃƒÂ©gatif
        if current['z_score_20'] < -self.config['z_score_threshold']:
            signals['long'].append(('extreme_z_score_low', 0.75))
        
        # 4. DÃƒÂ©viation extrÃƒÂªme de la moyenne
        max_deviation = max(
            abs(current['deviation_sma20']),
            abs(current['deviation_ema20']),
            abs(current['deviation_vwap'])
        )
        
        if current['deviation_sma20'] < -self.config['min_deviation']:
            if current['close'] > prev['close']:  # DÃƒÂ©but de retour
                signals['long'].append(('mean_deviation_reversal', 0.70))
        
        # 5. Stochastic oversold
        if current['stoch_k'] < 20 and current['stoch_d'] < 20:
            if current['stoch_k'] > current['stoch_d']:  # K croise D vers le haut
                signals['long'].append(('stoch_oversold_cross', 0.65))
        
        # 6. Volume spike en oversold (capitulation)
        if current['volume_ratio'] > self.config['volume_threshold']:
            if current['bb_position'] < 0.2:  # Dans le bas des BB
                signals['long'].append(('capitulation_volume', 0.75))
        
        # 7. Support bounce
        support_level = df['low'].rolling(20).min().iloc[-1]
        if abs(current['close'] - support_level) / support_level < 0.005:  # Proche support
            signals['long'].append(('support_bounce', 0.70))
        
        # ===========================================
        # CONDITIONS SHORT (Overbought Ã¢â€ â€™ Retour baissier)
        # ===========================================
        
        # 1. Prix au-dessus Bollinger upper
        if current['close'] >= current['bb_upper']:
            if current['close'] < prev['close']:  # DÃƒÂ©but de retournement
                signals['short'].append(('bb_overbought_reversal', 0.85))
        
        # 2. RSI overbought avec divergence
        if current['rsi'] > self.config['rsi_overbought']:
            if current['rsi'] < prev['rsi']:  # RSI descend
                signals['short'].append(('rsi_overbought_reversal', 0.80))
            
            if self.config['use_divergence'] and current.get('rsi_divergence') == 'bearish':
                signals['short'].append(('rsi_bearish_divergence', 0.90))
        
        # 3. Z-score extrÃƒÂªme positif
        if current['z_score_20'] > self.config['z_score_threshold']:
            signals['short'].append(('extreme_z_score_high', 0.75))
        
        # 4. DÃƒÂ©viation extrÃƒÂªme positive
        if current['deviation_sma20'] > self.config['min_deviation']:
            if current['close'] < prev['close']:  # DÃƒÂ©but de retour
                signals['short'].append(('mean_deviation_reversal_down', 0.70))
        
        # 5. Stochastic overbought
        if current['stoch_k'] > 80 and current['stoch_d'] > 80:
            if current['stoch_k'] < current['stoch_d']:  # K croise D vers le bas
                signals['short'].append(('stoch_overbought_cross', 0.65))
        
        # 6. Volume spike en overbought (euphorie)
        if current['volume_ratio'] > self.config['volume_threshold']:
            if current['bb_position'] > 0.8:  # Dans le haut des BB
                signals['short'].append(('euphoria_volume', 0.75))
        
        # 7. Resistance rejection
        resistance_level = df['high'].rolling(20).max().iloc[-1]
        if abs(current['close'] - resistance_level) / resistance_level < 0.005:
            if current['close'] < prev['high']:  # Rejet
                signals['short'].append(('resistance_rejection', 0.70))
        
        # ===========================================
        # FILTRES ADDITIONNELS
        # ===========================================
        
        # Filtre de volatilitÃƒÂ© (ÃƒÂ©viter les marchÃƒÂ©s trop volatils)
        if current['atr_percent'] > 0.05:  # ATR > 5%
            # RÃƒÂ©duire la confiance si trop volatil
            for direction in ['long', 'short']:
                signals[direction] = [(reason, score * 0.8) for reason, score in signals[direction]]
        
        # Filtre de volume
        if self.config['use_volume_filter'] and current['volume_ratio'] < 0.5:
            # Pas assez de volume, rÃƒÂ©duire confiance
            for direction in ['long', 'short']:
                signals[direction] = [(reason, score * 0.7) for reason, score in signals[direction]]
        
        # ===========================================
        # GÃƒâ€°NÃƒâ€°RATION DU SIGNAL
        # ===========================================
        
        long_score = sum(score for _, score in signals['long']) / max(len(signals['long']), 1)
        short_score = sum(score for _, score in signals['short']) / max(len(signals['short']), 1)
        
        min_confirmations = 2
        
        # Signal LONG
        if len(signals['long']) >= min_confirmations and long_score >= self.config['min_confidence']:
            entry_price = current['close']
            
            # Targets conservateurs pour mean reversion
            take_profit = min(
                entry_price * (1 + self.config['profit_target']),
                current['bb_middle']  # Viser le retour ÃƒÂ  la moyenne
            )
            stop_loss = entry_price * (1 - self.config['stop_loss'])
            
            return self.create_signal(
                signal_type='ENTRY',
                side='BUY',
                price=entry_price,
                confidence=long_score,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reasons=signals['long'],
                metadata={
                    'bb_position': current['bb_position'],
                    'z_score': current['z_score_20'],
                    'deviation': current['deviation_sma20'],
                    'rsi': current['rsi']
                }
            )
        
        # Signal SHORT
        if len(signals['short']) >= min_confirmations and short_score >= self.config['min_confidence']:
            entry_price = current['close']
            
            take_profit = max(
                entry_price * (1 - self.config['profit_target']),
                current['bb_middle']  # Viser le retour ÃƒÂ  la moyenne
            )
            stop_loss = entry_price * (1 + self.config['stop_loss'])
            
            return self.create_signal(
                signal_type='ENTRY',
                side='SELL',
                price=entry_price,
                confidence=short_score,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reasons=signals['short'],
                metadata={
                    'bb_position': current['bb_position'],
                    'z_score': current['z_score_20'],
                    'deviation': current['deviation_sma20'],
                    'rsi': current['rsi']
                }
            )
        
        return None
    
    def _check_exit_conditions(self, symbol: str, df: pd.DataFrame, current: pd.Series) -> Optional[Dict]:
        """
        VÃƒÂ©rifie les conditions de sortie mean reversion
        """
        position_key = f"{symbol}_{self.name}"
        if position_key not in self.positions:
            return None
        
        position = self.positions[position_key]
        entry_price = position['entry_price']
        side = position['side']
        entry_time = position['entry_time']
        
        current_price = current['close']
        profit_pct = (current_price - entry_price) / entry_price if side == 'BUY' else (entry_price - current_price) / entry_price
        
        exit_reasons = []
        
        # 1. Take profit atteint
        if profit_pct >= self.config['profit_target']:
            exit_reasons.append(('take_profit', 1.0))
        
        # 2. Stop loss atteint
        if profit_pct <= -self.config['stop_loss']:
            exit_reasons.append(('stop_loss', 1.0))
        
        # 3. Retour ÃƒÂ  la moyenne accompli
        if side == 'BUY':
            # Long: sortir quand on approche/dÃƒÂ©passe la moyenne
            if current['bb_position'] >= 0.45:  # Proche du milieu
                exit_reasons.append(('mean_reached', 0.9))
            
            if current['deviation_sma20'] >= 0:  # Au-dessus de la moyenne
                exit_reasons.append(('above_mean', 0.85))
            
            # RSI n'est plus oversold
            if current['rsi'] > 50:
                exit_reasons.append(('rsi_neutral', 0.7))
                
        else:  # SHORT
            # Short: sortir quand on approche/descend sous la moyenne
            if current['bb_position'] <= 0.55:
                exit_reasons.append(('mean_reached', 0.9))
            
            if current['deviation_sma20'] <= 0:  # En dessous de la moyenne
                exit_reasons.append(('below_mean', 0.85))
            
            # RSI n'est plus overbought
            if current['rsi'] < 50:
                exit_reasons.append(('rsi_neutral', 0.7))
        
        # 4. Z-score normalisÃƒÂ©
        if abs(current['z_score_20']) < 0.5:
            exit_reasons.append(('z_score_normalized', 0.8))
        
        # 5. Temps maximum
        hours_held = (datetime.now() - entry_time).seconds / 3600
        if hours_held >= self.config['max_holding_hours']:
            exit_reasons.append(('max_time', 0.95))
        
        # 6. Momentum inverse (le prix continue dans la mauvaise direction)
        if side == 'BUY' and current['close'] < position.get('worst_price', entry_price):
            position['worst_price'] = current['close']
            adverse_move = (entry_price - current['close']) / entry_price
            if adverse_move > self.config['stop_loss'] * 0.8:  # Proche du stop
                exit_reasons.append(('adverse_momentum', 0.85))
                
        elif side == 'SELL' and current['close'] > position.get('worst_price', entry_price):
            position['worst_price'] = current['close']
            adverse_move = (current['close'] - entry_price) / entry_price
            if adverse_move > self.config['stop_loss'] * 0.8:
                exit_reasons.append(('adverse_momentum', 0.85))
        
        # DÃƒÂ©cision de sortie
        if exit_reasons:
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
                    'exit_reason': main_reason[0],
                    'final_bb_position': current['bb_position'],
                    'final_z_score': current['z_score_20']
                }
            )
        
        return None
    
    def _calculate_z_score(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calcule le z-score (nombre d'ÃƒÂ©carts-types de la moyenne)
        """
        z_scores = np.zeros_like(prices)
        z_scores[:period] = np.nan
        
        for i in range(period, len(prices)):
            window = prices[i-period+1:i+1]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                z_scores[i] = (prices[i] - mean) / std
            else:
                z_scores[i] = 0
        
        return z_scores
    
    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcule un score composite de mean reversion
        """
        scores = np.zeros(len(df))
        
        for i in range(20, len(df)):
            score = 0
            count = 0
            
            # BB position
            if not np.isnan(df['bb_position'].iloc[i]):
                if df['bb_position'].iloc[i] < 0.2:
                    score += (0.2 - df['bb_position'].iloc[i]) * 5
                elif df['bb_position'].iloc[i] > 0.8:
                    score += (df['bb_position'].iloc[i] - 0.8) * 5
                count += 1
            
            # RSI
            if not np.isnan(df['rsi'].iloc[i]):
                if df['rsi'].iloc[i] < 30:
                    score += (30 - df['rsi'].iloc[i]) / 30
                elif df['rsi'].iloc[i] > 70:
                    score += (df['rsi'].iloc[i] - 70) / 30
                count += 1
            
            # Z-score
            if not np.isnan(df['z_score_20'].iloc[i]):
                score += min(abs(df['z_score_20'].iloc[i]) / 3, 1)
                count += 1
            
            scores[i] = score / max(count, 1)
        
        return scores
    
    def _detect_divergence(self, prices: np.ndarray, indicator: np.ndarray, 
                          window: int = 14) -> np.ndarray:
        """
        DÃƒÂ©tecte les divergences prix/indicateur
        
        Returns:
            Array avec 'bullish', 'bearish', ou 'none'
        """
        divergence = np.full(len(prices), 'none', dtype=object)
        
        if len(prices) < window * 2:
            return divergence
        
        for i in range(window * 2, len(prices)):
            # Chercher les lows pour divergence bullish
            price_window = prices[i-window:i]
            ind_window = indicator[i-window:i]
            
            # Skip si NaN
            if np.isnan(ind_window).any():
                continue
            
            # Trouver les minima locaux
            price_min_idx = np.argmin(price_window)
            
            if price_min_idx > 2 and price_min_idx < window - 2:
                # VÃƒÂ©rifier si c'est un vrai minimum local
                if price_window[price_min_idx] == min(price_window[price_min_idx-2:price_min_idx+3]):
                    # Chercher un minimum prÃƒÂ©cÃƒÂ©dent
                    prev_window = prices[i-window*2:i-window]
                    prev_min_idx = np.argmin(prev_window)
                    
                    if prev_min_idx > 2 and prev_min_idx < window - 2:
                        # Comparer prix et indicateur
                        if price_window[price_min_idx] < prev_window[prev_min_idx]:  # Lower low in price
                            ind_current = ind_window[price_min_idx]
                            ind_prev = indicator[i-window*2+prev_min_idx]
                            
                            if ind_current > ind_prev:  # Higher low in indicator
                                divergence[i] = 'bullish'
            
            # Chercher les highs pour divergence bearish (similaire)
            price_max_idx = np.argmax(price_window)
            
            if price_max_idx > 2 and price_max_idx < window - 2:
                if price_window[price_max_idx] == max(price_window[price_max_idx-2:price_max_idx+3]):
                    prev_window = prices[i-window*2:i-window]
                    prev_max_idx = np.argmax(prev_window)
                    
                    if prev_max_idx > 2 and prev_max_idx < window - 2:
                        if price_window[price_max_idx] > prev_window[prev_max_idx]:  # Higher high in price
                            ind_current = ind_window[price_max_idx]
                            ind_prev = indicator[i-window*2+prev_max_idx]
                            
                            if ind_current < ind_prev:  # Lower high in indicator
                                divergence[i] = 'bearish'
        
        return divergence
