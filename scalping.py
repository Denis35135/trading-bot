"""
StratÃƒÂ©gie de Scalping Intelligent pour The Bot
Vise des profits rapides de 0.3-0.5% avec haute frÃƒÂ©quence
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import logging

# Import des modules du bot
from utils.indicators import TechnicalIndicators, detect_divergence, trend_strength

logger = logging.getLogger(__name__)


class ScalpingStrategy:
    """
    StratÃƒÂ©gie de scalping optimisÃƒÂ©e pour Binance
    
    CaractÃƒÂ©ristiques:
    - Trades courts (1-15 minutes)
    - Profits visÃƒÂ©s: 0.3-0.5%
    - Stop loss serrÃƒÂ©: 0.2-0.3%
    - Haute frÃƒÂ©quence: 20-50 trades/jour
    - Win rate visÃƒÂ©: 65-70%
    """
    
    def __init__(self, config: dict = None):
        """
        Initialise la stratÃƒÂ©gie de scalping
        
        Args:
            config: Configuration personnalisÃƒÂ©e
        """
        # Configuration par dÃƒÂ©faut
        self.config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2,
            'volume_factor': 1.5,  # Volume > 1.5x moyenne
            'min_profit_percent': 0.003,  # 0.3% minimum
            'stop_loss_percent': 0.003,  # 0.3% stop loss
            'max_holding_minutes': 15,
            'use_divergence': True,
            'use_orderbook': True,
            'min_confidence': 0.65
        }
        
        if config:
            self.config.update(config)
        
        # Ãƒâ€°tat interne
        self.positions = {}
        self.signals_history = []
        self.performance_stats = {
            'total_signals': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0
        }
        
        # Indicateurs
        self.indicators = TechnicalIndicators()
        
        logger.info("StratÃƒÂ©gie Scalping initialisÃƒÂ©e")
        logger.info(f"Config: {self.config}")
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """
        Analyse les donnÃƒÂ©es et gÃƒÂ©nÃƒÂ¨re un signal de trading
        
        Args:
            data: Dict contenant:
                - df: DataFrame avec OHLCV et indicateurs
                - orderbook: Orderbook actuel
                - recent_trades: Trades rÃƒÂ©cents
                - symbol: Le symbole
                
        Returns:
            Signal de trading ou None
        """
        try:
            df = data.get('df')
            if df is None or len(df) < 50:
                return None
            
            symbol = data.get('symbol', 'UNKNOWN')
            orderbook = data.get('orderbook', {})
            
            # Calculer indicateurs si pas dÃƒÂ©jÃƒÂ  fait
            if 'rsi' not in df.columns:
                df = self.indicators.calculate_all(df, self.config)
            
            # DonnÃƒÂ©es actuelles
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Analyse multi-critÃƒÂ¨res
            signal = self._analyze_entry_conditions(df, current, prev, orderbook)
            
            if signal:
                signal['symbol'] = symbol
                signal['strategy'] = 'scalping'
                signal['timestamp'] = datetime.now()
                
                # Log et sauvegarde
                self._log_signal(signal)
                
                return signal
            
            # VÃƒÂ©rifier les positions existantes pour sortie
            exit_signal = self._check_exit_conditions(symbol, current)
            if exit_signal:
                return exit_signal
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur analyse scalping: {e}")
            return None
    
    def _analyze_entry_conditions(self, df: pd.DataFrame, current: pd.Series, 
                                 prev: pd.Series, orderbook: Dict) -> Optional[Dict]:
        """
        Analyse les conditions d'entrÃƒÂ©e
        
        Returns:
            Signal d'entrÃƒÂ©e ou None
        """
        signals = {
            'long': [],
            'short': []
        }
        
        # ===========================================
        # CONDITIONS LONG (Achat)
        # ===========================================
        
        # 1. RSI Oversold avec momentum shift
        if current['rsi'] < self.config['rsi_oversold']:
            if current['rsi'] > prev['rsi']:  # RSI commence ÃƒÂ  remonter
                signals['long'].append(('rsi_oversold_reversal', 0.8))
        
        # 2. Bollinger Bands squeeze et breakout
        bb_width = current['bb_upper'] - current['bb_lower']
        bb_width_ma = df['bb_upper'].rolling(20).mean() - df['bb_lower'].rolling(20).mean()
        
        if current['close'] <= current['bb_lower']:
            if bb_width < bb_width_ma.iloc[-1] * 0.8:  # Squeeze
                signals['long'].append(('bb_squeeze_long', 0.7))
        
        # 3. Volume spike avec prix en hausse
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        if current['volume'] > volume_ma * self.config['volume_factor']:
            if current['close'] > prev['close']:
                signals['long'].append(('volume_breakout_long', 0.75))
        
        # 4. MACD crossover bullish
        if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
            signals['long'].append(('macd_bullish_cross', 0.7))
        
        # 5. Support bounce
        support = self._find_nearest_support(df)
        if support and abs(current['close'] - support) / support < 0.002:  # Proche du support (0.2%)
            if current['close'] > prev['close']:
                signals['long'].append(('support_bounce', 0.8))
        
        # 6. Divergence bullish
        if self.config['use_divergence']:
            divergence = detect_divergence(df['close'].values, df['rsi'].values)
            if divergence == 'bullish':
                signals['long'].append(('bullish_divergence', 0.85))
        
        # 7. Orderbook imbalance (plus d'acheteurs)
        if self.config['use_orderbook'] and orderbook:
            imbalance = self._calculate_orderbook_imbalance(orderbook)
            if imbalance > 0.6:  # 60% cÃƒÂ´tÃƒÂ© achat
                signals['long'].append(('orderbook_buy_pressure', 0.65))
        
        # ===========================================
        # CONDITIONS SHORT (Vente)
        # ===========================================
        
        # 1. RSI Overbought avec momentum shift
        if current['rsi'] > self.config['rsi_overbought']:
            if current['rsi'] < prev['rsi']:  # RSI commence ÃƒÂ  descendre
                signals['short'].append(('rsi_overbought_reversal', 0.8))
        
        # 2. Bollinger Bands upper rejection
        if current['close'] >= current['bb_upper']:
            if current['close'] < prev['high']:  # Rejet
                signals['short'].append(('bb_upper_rejection', 0.7))
        
        # 3. Volume spike avec prix en baisse
        if current['volume'] > volume_ma * self.config['volume_factor']:
            if current['close'] < prev['close']:
                signals['short'].append(('volume_breakdown_short', 0.75))
        
        # 4. MACD crossover bearish
        if prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
            signals['short'].append(('macd_bearish_cross', 0.7))
        
        # 5. Resistance rejection
        resistance = self._find_nearest_resistance(df)
        if resistance and abs(current['close'] - resistance) / resistance < 0.002:
            if current['close'] < prev['close']:
                signals['short'].append(('resistance_rejection', 0.8))
        
        # 6. Divergence bearish
        if self.config['use_divergence']:
            divergence = detect_divergence(df['close'].values, df['rsi'].values)
            if divergence == 'bearish':
                signals['short'].append(('bearish_divergence', 0.85))
        
        # 7. Orderbook imbalance (plus de vendeurs)
        if self.config['use_orderbook'] and orderbook:
            imbalance = self._calculate_orderbook_imbalance(orderbook)
            if imbalance < 0.4:  # 40% cÃƒÂ´tÃƒÂ© achat = 60% vente
                signals['short'].append(('orderbook_sell_pressure', 0.65))
        
        # ===========================================
        # GÃƒâ€°NÃƒâ€°RATION DU SIGNAL
        # ===========================================
        
        # Calculer le score pour chaque direction
        long_score = sum(score for _, score in signals['long']) / max(len(signals['long']), 1)
        short_score = sum(score for _, score in signals['short']) / max(len(signals['short']), 1)
        
        # Nombre minimum de confirmations
        min_confirmations = 2
        
        # Signal LONG
        if len(signals['long']) >= min_confirmations and long_score >= self.config['min_confidence']:
            return self._create_signal(
                side='BUY',
                price=current['close'],
                confidence=long_score,
                reasons=signals['long'],
                data=current
            )
        
        # Signal SHORT
        if len(signals['short']) >= min_confirmations and short_score >= self.config['min_confidence']:
            return self._create_signal(
                side='SELL',
                price=current['close'],
                confidence=short_score,
                reasons=signals['short'],
                data=current
            )
        
        return None
    
    def _check_exit_conditions(self, symbol: str, current: pd.Series) -> Optional[Dict]:
        """
        VÃƒÂ©rifie les conditions de sortie pour les positions ouvertes
        
        Returns:
            Signal de sortie ou None
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        side = position['side']
        
        current_price = current['close']
        profit_pct = (current_price - entry_price) / entry_price
        
        if side == 'BUY':
            # Conditions de sortie pour position LONG
            
            # Take Profit atteint
            if profit_pct >= self.config['min_profit_percent']:
                return self._create_exit_signal(symbol, 'SELL', current_price, 'take_profit', profit_pct)
            
            # Stop Loss atteint
            if profit_pct <= -self.config['stop_loss_percent']:
                return self._create_exit_signal(symbol, 'SELL', current_price, 'stop_loss', profit_pct)
            
            # Time stop (position trop longue)
            if (datetime.now() - entry_time).seconds > self.config['max_holding_minutes'] * 60:
                return self._create_exit_signal(symbol, 'SELL', current_price, 'time_stop', profit_pct)
            
            # Trailing stop si en profit
            if profit_pct > 0.002:  # En profit de 0.2%+
                if current['rsi'] > 70 or current['close'] < current['ema_9']:
                    return self._create_exit_signal(symbol, 'SELL', current_price, 'trailing_stop', profit_pct)
        
        else:  # SHORT position
            profit_pct = -profit_pct  # Inverser pour short
            
            # Take Profit atteint
            if profit_pct >= self.config['min_profit_percent']:
                return self._create_exit_signal(symbol, 'BUY', current_price, 'take_profit', profit_pct)
            
            # Stop Loss atteint
            if profit_pct <= -self.config['stop_loss_percent']:
                return self._create_exit_signal(symbol, 'BUY', current_price, 'stop_loss', profit_pct)
            
            # Time stop
            if (datetime.now() - entry_time).seconds > self.config['max_holding_minutes'] * 60:
                return self._create_exit_signal(symbol, 'BUY', current_price, 'time_stop', profit_pct)
            
            # Trailing stop si en profit
            if profit_pct > 0.002:
                if current['rsi'] < 30 or current['close'] > current['ema_9']:
                    return self._create_exit_signal(symbol, 'BUY', current_price, 'trailing_stop', profit_pct)
        
        return None
    
    def _create_signal(self, side: str, price: float, confidence: float, 
                      reasons: List[Tuple], data: pd.Series) -> Dict:
        """CrÃƒÂ©e un signal de trading formatÃƒÂ©"""
        
        # Calcul des targets
        if side == 'BUY':
            take_profit = price * (1 + self.config['min_profit_percent'])
            stop_loss = price * (1 - self.config['stop_loss_percent'])
        else:
            take_profit = price * (1 - self.config['min_profit_percent'])
            stop_loss = price * (1 + self.config['stop_loss_percent'])
        
        signal = {
            'type': 'ENTRY',
            'side': side,
            'price': price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'confidence': confidence,
            'reasons': reasons,
            'indicators': {
                'rsi': data['rsi'],
                'macd': data['macd'],
                'volume_ratio': data['volume'] / data['volume'].rolling(20).mean().iloc[-1] if 'volume' in data else 1,
                'bb_position': (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower']) if 'bb_upper' in data else 0.5
            }
        }
        
        return signal
    
    def _create_exit_signal(self, symbol: str, side: str, price: float, 
                           reason: str, profit_pct: float) -> Dict:
        """CrÃƒÂ©e un signal de sortie"""
        
        signal = {
            'type': 'EXIT',
            'symbol': symbol,
            'side': side,
            'price': price,
            'reason': reason,
            'profit_pct': profit_pct,
            'confidence': 1.0,
            'strategy': 'scalping',
            'timestamp': datetime.now()
        }
        
        # Mise ÃƒÂ  jour des stats
        if profit_pct > 0:
            self.performance_stats['winning_trades'] += 1
        else:
            self.performance_stats['losing_trades'] += 1
        
        self.performance_stats['total_profit'] += profit_pct
        
        # Retirer de positions
        del self.positions[symbol]
        
        return signal
    
    def _find_nearest_support(self, df: pd.DataFrame) -> Optional[float]:
        """Trouve le support le plus proche"""
        current_price = df['close'].iloc[-1]
        recent_lows = df['low'].rolling(20).min()
        
        supports = []
        for i in range(len(recent_lows) - 20, len(recent_lows)):
            if recent_lows.iloc[i] < current_price:
                supports.append(recent_lows.iloc[i])
        
        if supports:
            return max(supports)  # Support le plus proche
        return None
    
    def _find_nearest_resistance(self, df: pd.DataFrame) -> Optional[float]:
        """Trouve la rÃƒÂ©sistance la plus proche"""
        current_price = df['close'].iloc[-1]
        recent_highs = df['high'].rolling(20).max()
        
        resistances = []
        for i in range(len(recent_highs) - 20, len(recent_highs)):
            if recent_highs.iloc[i] > current_price:
                resistances.append(recent_highs.iloc[i])
        
        if resistances:
            return min(resistances)  # RÃƒÂ©sistance la plus proche
        return None
    
    def _calculate_orderbook_imbalance(self, orderbook: Dict) -> float:
        """
        Calcule le dÃƒÂ©sÃƒÂ©quilibre de l'orderbook
        
        Returns:
            Ratio entre 0 (que des vendeurs) et 1 (que des acheteurs)
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return 0.5
        
        # Calculer volume pondÃƒÂ©rÃƒÂ© des 10 premiers niveaux
        bid_volume = sum(price * qty for price, qty in orderbook['bids'][:10])
        ask_volume = sum(price * qty for price, qty in orderbook['asks'][:10])
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.5
        
        return bid_volume / total_volume
    
    def _log_signal(self, signal: Dict):
        """Log et sauvegarde le signal"""
        self.signals_history.append(signal)
        self.performance_stats['total_signals'] += 1
        
        # Garder seulement les 100 derniers signaux
        if len(self.signals_history) > 100:
            self.signals_history.pop(0)
        
        logger.info(f"Ã°Å¸â€œÅ  Signal Scalping: {signal['side']} @ {signal['price']:.4f}")
        logger.info(f"   Confidence: {signal['confidence']:.2%}")
        logger.info(f"   Raisons: {[r[0] for r in signal['reasons']]}")
    
    def register_position(self, symbol: str, side: str, entry_price: float, quantity: float):
        """
        Enregistre une nouvelle position
        
        Args:
            symbol: Le symbole
            side: BUY ou SELL
            entry_price: Prix d'entrÃƒÂ©e
            quantity: QuantitÃƒÂ©
        """
        self.positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': datetime.now()
        }
        
        logger.info(f"Position enregistrÃƒÂ©e: {symbol} {side} @ {entry_price}")
    
    def get_performance_stats(self) -> Dict:
        """Retourne les statistiques de performance"""
        total_trades = self.performance_stats['winning_trades'] + self.performance_stats['losing_trades']
        
        if total_trades == 0:
            win_rate = 0
            avg_profit = 0
        else:
            win_rate = self.performance_stats['winning_trades'] / total_trades
            avg_profit = self.performance_stats['total_profit'] / total_trades
        
        return {
            'total_signals': self.performance_stats['total_signals'],
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': self.performance_stats['total_profit']
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de la stratÃƒÂ©gie de scalping"""
    
    import sys
    sys.path.append('..')
    
    # Configuration de test
    config = {
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'min_profit_percent': 0.003,
        'stop_loss_percent': 0.002
    }
    
    # Initialiser la stratÃƒÂ©gie
    strategy = ScalpingStrategy(config)
    
    # CrÃƒÂ©er des donnÃƒÂ©es de test
    np.random.seed(42)
    size = 100
    
    close = 100 + np.cumsum(np.random.randn(size) * 0.5)
    high = close + np.abs(np.random.randn(size) * 0.2)
    low = close - np.abs(np.random.randn(size) * 0.2)
    volume = np.random.randint(1000, 10000, size)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(size) * 0.1,
        'high': high,
        'low': low, 
        'close': close,
        'volume': volume
    })
    
    # Calculer les indicateurs
    indicators = TechnicalIndicators()
    df = indicators.calculate_all(df)
    
    # Test analyse
    data = {
        'df': df,
        'symbol': 'BTCUSDC',
        'orderbook': {
            'bids': [[99.5, 10], [99.4, 20], [99.3, 30]],
            'asks': [[100.5, 10], [100.6, 20], [100.7, 30]]
        }
    }
    
    # Analyser plusieurs fois
    for i in range(5):
        signal = strategy.analyze(data)
        if signal:
            print(f"\nÃ¢Å“â€¦ Signal dÃƒÂ©tectÃƒÂ©!")
            print(f"Type: {signal['type']}")
            print(f"Side: {signal['side']}")
            print(f"Prix: {signal['price']:.2f}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Raisons: {[r[0] for r in signal['reasons']]}")
            
            # Simuler position
            if signal['type'] == 'ENTRY':
                strategy.register_position('BTCUSDC', signal['side'], signal['price'], 0.01)
    
    # Afficher stats
    stats = strategy.get_performance_stats()
    print(f"\nÃ°Å¸â€œÅ  Statistiques:")
    print(f"Signaux gÃƒÂ©nÃƒÂ©rÃƒÂ©s: {stats['total_signals']}")
    print(f"Trades: {stats['total_trades']}")
    print(f"Win rate: {stats['win_rate']:.1%}")
    print(f"Profit moyen: {stats['avg_profit']:.3%}")