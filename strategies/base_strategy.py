"""
Base Strategy pour The Bot
Classe abstraite dont toutes les stratÃƒÂ©gies hÃƒÂ©ritent
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Classe de base pour toutes les stratÃƒÂ©gies de trading
    
    Toute nouvelle stratÃƒÂ©gie doit hÃƒÂ©riter de cette classe
    et implÃƒÂ©menter les mÃƒÂ©thodes abstraites
    """
    
    def __init__(self, name: str, config: Dict = None):
        """
        Initialise la stratÃƒÂ©gie de base
        
        Args:
            name: Nom de la stratÃƒÂ©gie
            config: Configuration spÃƒÂ©cifique ÃƒÂ  la stratÃƒÂ©gie
        """
        self.name = name
        self.config = config or {}
        
        # Configuration par dÃƒÂ©faut
        self.default_config = {
            'min_confidence': 0.65,
            'risk_reward_ratio': 1.5,
            'max_positions': 5,
            'timeframe': '5m',
            'lookback_periods': 100,
            'use_ml': False,
            'use_volume_confirmation': True,
            'use_multi_timeframe': False
        }
        
        # Merger avec config spÃƒÂ©cifique
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Ãƒâ€°tat de la stratÃƒÂ©gie
        self.is_active = True
        self.positions = {}
        self.pending_signals = []
        self.last_signal_time = {}
        
        # Performance tracking
        self.performance = {
            'total_signals': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_profit': 0,
            'max_loss': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'best_trade': None,
            'worst_trade': None,
            'average_win': 0,
            'average_loss': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'recovery_factor': 0
        }
        
        # Historique
        self.signal_history = []
        self.trade_history = []
        
        logger.info(f"StratÃƒÂ©gie '{name}' initialisÃƒÂ©e avec config: {self.config}")
    
    @abstractmethod
    def analyze(self, data: Dict) -> Optional[Dict]:
        """
        Analyse les donnÃƒÂ©es et gÃƒÂ©nÃƒÂ¨re un signal de trading
        
        Cette mÃƒÂ©thode DOIT ÃƒÂªtre implÃƒÂ©mentÃƒÂ©e par chaque stratÃƒÂ©gie
        
        Args:
            data: Dict contenant:
                - df: DataFrame avec OHLCV et indicateurs
                - orderbook: Orderbook actuel (optionnel)
                - trades: Trades rÃƒÂ©cents (optionnel)
                - symbol: Le symbole
                
        Returns:
            Signal de trading ou None si pas de signal
            Format du signal:
            {
                'type': 'ENTRY' ou 'EXIT',
                'side': 'BUY' ou 'SELL',
                'price': float,
                'confidence': float (0-1),
                'take_profit': float,
                'stop_loss': float,
                'reasons': list,
                'metadata': dict
            }
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs spÃƒÂ©cifiques ÃƒÂ  la stratÃƒÂ©gie
        
        Args:
            df: DataFrame avec OHLCV
            
        Returns:
            DataFrame avec indicateurs ajoutÃƒÂ©s
        """
        pass
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Valide un signal avant de le retourner
        
        Args:
            signal: Le signal ÃƒÂ  valider
            
        Returns:
            True si valide, False sinon
        """
        # VÃƒÂ©rifications de base
        required_fields = ['type', 'side', 'price', 'confidence']
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Signal invalide: champ '{field}' manquant")
                return False
        
        # VÃƒÂ©rifier la confidence minimum
        if signal['confidence'] < self.config['min_confidence']:
            logger.debug(f"Signal rejetÃƒÂ©: confidence {signal['confidence']:.2%} "
                        f"< minimum {self.config['min_confidence']:.2%}")
            return False
        
        # VÃƒÂ©rifier le type
        if signal['type'] not in ['ENTRY', 'EXIT']:
            logger.warning(f"Type de signal invalide: {signal['type']}")
            return False
        
        # VÃƒÂ©rifier le side
        if signal['side'] not in ['BUY', 'SELL']:
            logger.warning(f"Side invalide: {signal['side']}")
            return False
        
        # VÃƒÂ©rifier les prix
        if signal['price'] <= 0:
            logger.warning(f"Prix invalide: {signal['price']}")
            return False
        
        # VÃƒÂ©rifier le risk/reward si signal d'entrÃƒÂ©e
        if signal['type'] == 'ENTRY' and 'take_profit' in signal and 'stop_loss' in signal:
            reward = abs(signal['take_profit'] - signal['price'])
            risk = abs(signal['price'] - signal['stop_loss'])
            
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.config['risk_reward_ratio']:
                    logger.debug(f"Signal rejetÃƒÂ©: R/R ratio {rr_ratio:.2f} "
                               f"< minimum {self.config['risk_reward_ratio']:.2f}")
                    return False
        
        # Anti-spam : ÃƒÂ©viter signaux trop frÃƒÂ©quents
        symbol = signal.get('symbol', 'UNKNOWN')
        if symbol in self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time[symbol]).seconds
            if time_since_last < 60:  # Minimum 1 minute entre signaux
                logger.debug(f"Signal rejetÃƒÂ©: trop rÃƒÂ©cent ({time_since_last}s)")
                return False
        
        return True
    
    def register_position(self, symbol: str, side: str, entry_price: float, 
                         quantity: float, signal: Dict = None):
        """
        Enregistre une nouvelle position
        
        Args:
            symbol: Le symbole
            side: BUY ou SELL
            entry_price: Prix d'entrÃƒÂ©e
            quantity: QuantitÃƒÂ©
            signal: Signal original (optionnel)
        """
        position_key = f"{symbol}_{self.name}"
        
        self.positions[position_key] = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': datetime.now(),
            'signal': signal,
            'peak_profit': 0,
            'max_drawdown': 0
        }
        
        logger.info(f"[{self.name}] Position enregistrÃƒÂ©e: {symbol} {side} @ {entry_price}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "signal"):
        """
        Ferme une position et met ÃƒÂ  jour les statistiques
        
        Args:
            symbol: Le symbole
            exit_price: Prix de sortie
            reason: Raison de la fermeture
        """
        position_key = f"{symbol}_{self.name}"
        
        if position_key not in self.positions:
            logger.warning(f"Position {position_key} non trouvÃƒÂ©e")
            return
        
        position = self.positions[position_key]
        
        # Calculer le profit
        if position['side'] == 'BUY':
            profit_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            profit_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        profit_usdc = position['quantity'] * position['entry_price'] * profit_pct
        
        # CrÃƒÂ©er l'entrÃƒÂ©e d'historique
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'profit_pct': profit_pct,
            'profit_usdc': profit_usdc,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'duration': (datetime.now() - position['entry_time']).seconds,
            'reason': reason,
            'strategy': self.name
        }
        
        self.trade_history.append(trade)
        
        # Mettre ÃƒÂ  jour les statistiques
        self._update_performance(trade)
        
        # Retirer la position
        del self.positions[position_key]
        
        logger.info(f"[{self.name}] Position fermÃƒÂ©e: {symbol} @ {exit_price} "
                   f"({profit_pct:+.2%} = ${profit_usdc:+.2f})")
    
    def _update_performance(self, trade: Dict):
        """Met ÃƒÂ  jour les statistiques de performance"""
        profit = trade['profit_usdc']
        profit_pct = trade['profit_pct']
        
        # Stats gÃƒÂ©nÃƒÂ©rales
        self.performance['total_profit'] += profit
        
        if profit > 0:
            self.performance['winning_trades'] += 1
            self.performance['consecutive_wins'] += 1
            self.performance['consecutive_losses'] = 0
            
            if profit > self.performance['max_profit']:
                self.performance['max_profit'] = profit
                self.performance['best_trade'] = trade
        else:
            self.performance['losing_trades'] += 1
            self.performance['consecutive_losses'] += 1
            self.performance['consecutive_wins'] = 0
            
            if profit < self.performance['max_loss']:
                self.performance['max_loss'] = profit
                self.performance['worst_trade'] = trade
        
        # Calculer les moyennes
        total_trades = self.performance['winning_trades'] + self.performance['losing_trades']
        
        if total_trades > 0:
            # Win rate
            self.performance['win_rate'] = self.performance['winning_trades'] / total_trades
            
            # Average win/loss
            if self.performance['winning_trades'] > 0:
                wins = [t['profit_usdc'] for t in self.trade_history if t['profit_usdc'] > 0]
                self.performance['average_win'] = np.mean(wins)
            
            if self.performance['losing_trades'] > 0:
                losses = [abs(t['profit_usdc']) for t in self.trade_history if t['profit_usdc'] < 0]
                self.performance['average_loss'] = np.mean(losses)
            
            # Profit factor
            if self.performance['average_loss'] > 0:
                self.performance['profit_factor'] = (
                    self.performance['average_win'] * self.performance['winning_trades']
                ) / (
                    self.performance['average_loss'] * self.performance['losing_trades']
                )
            
            # Sharpe ratio (simplifiÃƒÂ©)
            returns = [t['profit_pct'] for t in self.trade_history[-30:]]  # 30 derniers trades
            if len(returns) > 1:
                self.performance['sharpe_ratio'] = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                    if np.std(returns) > 0 else 0
                )
        
        # Garder max 1000 trades en historique
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def get_active_positions(self) -> Dict:
        """Retourne les positions actives"""
        return self.positions.copy()
    
    def has_position(self, symbol: str) -> bool:
        """VÃƒÂ©rifie si une position existe pour un symbole"""
        position_key = f"{symbol}_{self.name}"
        return position_key in self.positions
    
    def get_performance_summary(self) -> Dict:
        """Retourne un rÃƒÂ©sumÃƒÂ© des performances"""
        total_trades = self.performance['winning_trades'] + self.performance['losing_trades']
        
        return {
            'strategy': self.name,
            'total_trades': total_trades,
            'win_rate': self.performance['win_rate'],
            'profit_factor': self.performance['profit_factor'],
            'total_profit': self.performance['total_profit'],
            'average_win': self.performance['average_win'],
            'average_loss': self.performance['average_loss'],
            'sharpe_ratio': self.performance['sharpe_ratio'],
            'best_trade': self.performance['best_trade']['profit_pct'] if self.performance['best_trade'] else 0,
            'worst_trade': self.performance['worst_trade']['profit_pct'] if self.performance['worst_trade'] else 0,
            'consecutive_wins': self.performance['consecutive_wins'],
            'consecutive_losses': self.performance['consecutive_losses']
        }
    
    def reset_performance(self):
        """RÃƒÂ©initialise les statistiques de performance"""
        self.performance = {
            'total_signals': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_profit': 0,
            'max_loss': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'best_trade': None,
            'worst_trade': None,
            'average_win': 0,
            'average_loss': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'recovery_factor': 0
        }
        self.trade_history = []
        self.signal_history = []
    
    def enable(self):
        """Active la stratÃƒÂ©gie"""
        self.is_active = True
        logger.info(f"StratÃƒÂ©gie '{self.name}' activÃƒÂ©e")
    
    def disable(self):
        """DÃƒÂ©sactive la stratÃƒÂ©gie"""
        self.is_active = False
        logger.info(f"StratÃƒÂ©gie '{self.name}' dÃƒÂ©sactivÃƒÂ©e")
    
    def is_enabled(self) -> bool:
        """VÃƒÂ©rifie si la stratÃƒÂ©gie est active"""
        return self.is_active
    
    def log_signal(self, signal: Dict):
        """Enregistre un signal dans l'historique"""
        signal['timestamp'] = datetime.now()
        signal['strategy'] = self.name
        
        self.signal_history.append(signal)
        self.performance['total_signals'] += 1
        
        # Update last signal time
        if 'symbol' in signal:
            self.last_signal_time[signal['symbol']] = datetime.now()
        
        # Garder max 500 signaux en historique
        if len(self.signal_history) > 500:
            self.signal_history = self.signal_history[-500:]
        
        logger.info(f"[{self.name}] Signal gÃƒÂ©nÃƒÂ©rÃƒÂ©: {signal['type']} {signal.get('symbol', 'N/A')} "
                   f"@ {signal.get('price', 0):.4f} (conf: {signal.get('confidence', 0):.2%})")
    
    def create_signal(self,
                     signal_type: str,
                     side: str,
                     price: float,
                     confidence: float,
                     symbol: str = None,
                     take_profit: float = None,
                     stop_loss: float = None,
                     reasons: List = None,
                     metadata: Dict = None) -> Dict:
        """
        Helper pour crÃƒÂ©er un signal formatÃƒÂ©
        
        Args:
            signal_type: 'ENTRY' ou 'EXIT'
            side: 'BUY' ou 'SELL'
            price: Prix du signal
            confidence: Confiance (0-1)
            symbol: Symbole (optionnel)
            take_profit: Prix TP (optionnel)
            stop_loss: Prix SL (optionnel)
            reasons: Liste des raisons (optionnel)
            metadata: Metadata additionnelle (optionnel)
            
        Returns:
            Signal formatÃƒÂ©
        """
        signal = {
            'type': signal_type,
            'side': side,
            'price': price,
            'confidence': confidence,
            'strategy': self.name,
            'timestamp': datetime.now()
        }
        
        if symbol:
            signal['symbol'] = symbol
        
        if take_profit:
            signal['take_profit'] = take_profit
        
        if stop_loss:
            signal['stop_loss'] = stop_loss
        
        if reasons:
            signal['reasons'] = reasons
        
        if metadata:
            signal['metadata'] = metadata
        
        return signal
