"""
Strategy Manager pour The Bot - IMPORTS CORRIGÃƒâ€°S
Coordonne toutes les stratÃƒÂ©gies et gÃƒÂ¨re l'allocation du capital
"""

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from queue import Queue
from collections import defaultdict
import logging

# CrÃƒÂ©er l'enum OrderPriority localement puisqu'elle n'existe pas dans order_manager
class OrderPriority(Enum):
    """Niveaux de prioritÃƒÂ© des ordres"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

# Import des composants du bot (CORRIGÃƒâ€°S)
from risk.position_sizing import PositionSizer
from risk.risk_monitor import RiskMonitor, RiskLevel  # RiskLevel existe bien dans risk_monitor.py
from exchange.order_manager import OrderManager  # Sans OrderPriority
from strategies.scalping import ScalpingStrategy

logger = logging.getLogger(__name__)

# Reste du code de StrategyManager...
# Note: Dans les mÃƒÂ©thodes qui utilisent OrderPriority, elle est maintenant dÃƒÂ©finie localement

class StrategyManager:
    """
    Gestionnaire central des stratÃƒÂ©gies de trading
    
    ResponsabilitÃƒÂ©s:
    - Charger et initialiser les stratÃƒÂ©gies
    - Distribuer les donnÃƒÂ©es aux stratÃƒÂ©gies
    - Collecter et filtrer les signaux
    - GÃƒÂ©rer l'allocation du capital entre stratÃƒÂ©gies
    - Coordonner avec risk monitor et order manager
    - Tracker la performance par stratÃƒÂ©gie
    """
    
    def __init__(self, 
                 config: Dict,
                 exchange_client,
                 order_manager: OrderManager,
                 position_sizer: PositionSizer,
                 risk_monitor: RiskMonitor):
        """
        Initialise le strategy manager
        
        Args:
            config: Configuration globale
            exchange_client: Client exchange (Binance)
            order_manager: Gestionnaire d'ordres
            position_sizer: Calculateur de taille de positions
            risk_monitor: Moniteur de risque
        """
        self.config = config
        self.exchange = exchange_client
        self.order_manager = order_manager
        self.position_sizer = position_sizer
        self.risk_monitor = risk_monitor
        
        # StratÃƒÂ©gies configurÃƒÂ©es
        self.strategy_configs = getattr(config, 'STRATEGIES', [])
        self.active_strategies = {}
        self.strategy_allocations = {}
        
        # Ãƒâ€°tat
        self.is_running = False
        self.trading_enabled = True
        self.positions = {}  # Positions ouvertes par symbole
        self.pending_signals = Queue()
        
        # Performance tracking
        self.performance_by_strategy = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'last_signal': None
        })
        
        # Threads
        self.signal_processor_thread = None
        self.performance_tracker_thread = None
        
        # Market data cache
        self.market_data_cache = {}
        self.last_data_update = {}
        
        # Initialiser les stratÃƒÂ©gies
        self._initialize_strategies()
        
        logger.info("Strategy Manager initialisÃƒÂ©")
        logger.info(f"StratÃƒÂ©gies actives: {list(self.active_strategies.keys())}")
    
    def _initialize_strategies(self):
        """Initialise toutes les stratÃƒÂ©gies configurÃƒÂ©es"""
        total_allocation = 0
        
        for strat_config in self.strategy_configs:
            name = strat_config['name']
            allocation = strat_getattr(config, 'ALLOCATION', 0.2)
            enabled = strat_getattr(config, 'ENABLED', True)
            
            if not enabled:
                logger.info(f"StratÃƒÂ©gie {name} dÃƒÂ©sactivÃƒÂ©e")
                continue
            
            try:
                # Charger la stratÃƒÂ©gie dynamiquement
                strategy = self._load_strategy(name, strat_config)
                
                if strategy:
                    self.active_strategies[name] = strategy
                    self.strategy_allocations[name] = allocation
                    total_allocation += allocation
                    
                    logger.info(f"Ã¢Å“â€¦ StratÃƒÂ©gie {name} chargÃƒÂ©e (allocation: {allocation:.1%})")
                
            except Exception as e:
                logger.error(f"Erreur chargement stratÃƒÂ©gie {name}: {e}")
        
        # Normaliser les allocations
        if total_allocation > 0 and total_allocation != 1.0:
            for name in self.strategy_allocations:
                self.strategy_allocations[name] /= total_allocation
            logger.info(f"Allocations normalisÃƒÂ©es (total ÃƒÂ©tait {total_allocation:.1%})")
    
    def _load_strategy(self, name: str, config: Dict):
        """
        Charge une stratÃƒÂ©gie par son nom
        
        Args:
            name: Nom de la stratÃƒÂ©gie
            config: Configuration spÃƒÂ©cifique
            
        Returns:
            Instance de la stratÃƒÂ©gie
        """
        # Pour l'instant, seulement scalping implÃƒÂ©mentÃƒÂ©
        # Ajouter les autres stratÃƒÂ©gies au fur et ÃƒÂ  mesure
        
        if name == 'scalping':
            return ScalpingStrategy(getattr(config, 'PARAMS', {}))
        
        # elif name == 'momentum':
        #     from strategies.momentum import MomentumStrategy
        #     return MomentumStrategy(getattr(config, 'PARAMS', {}))
        
        # elif name == 'mean_reversion':
        #     from strategies.mean_reversion import MeanReversionStrategy
        #     return MeanReversionStrategy(getattr(config, 'PARAMS', {}))
        
        else:
            logger.warning(f"StratÃƒÂ©gie {name} non trouvÃƒÂ©e")
            return None
    
    def start(self):
        """DÃƒÂ©marre le strategy manager"""
        if self.is_running:
            logger.warning("Strategy Manager dÃƒÂ©jÃƒÂ  en cours")
            return
        
        self.is_running = True
        
        # DÃƒÂ©marrer le processeur de signaux
        self.signal_processor_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True
        )
        self.signal_processor_thread.start()
        
        # DÃƒÂ©marrer le tracker de performance
        self.performance_tracker_thread = threading.Thread(
            target=self._performance_tracking_loop,
            daemon=True
        )
        self.performance_tracker_thread.start()
        
        logger.info("Strategy Manager dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte le strategy manager"""
        self.is_running = False
        
        # Attendre l'arrÃƒÂªt des threads
        if self.signal_processor_thread:
            self.signal_processor_thread.join(timeout=5)
        if self.performance_tracker_thread:
            self.performance_tracker_thread.join(timeout=5)
        
        logger.info("Strategy Manager arrÃƒÂªtÃƒÂ©")
    
    def process_market_data(self, symbol: str, data: Dict):
        """
        Traite les donnÃƒÂ©es de marchÃƒÂ© et gÃƒÂ©nÃƒÂ¨re des signaux
        
        Args:
            symbol: Le symbole
            data: DonnÃƒÂ©es de marchÃƒÂ© (OHLCV, orderbook, etc.)
        """
        if not self.trading_enabled:
            return
        
        # Mettre en cache
        self.market_data_cache[symbol] = data
        self.last_data_update[symbol] = datetime.now()
        
        # VÃƒÂ©rifier qu'on a assez de donnÃƒÂ©es
        if 'df' not in data or len(data['df']) < 50:
            return
        
        # Envoyer aux stratÃƒÂ©gies pour analyse
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # VÃƒÂ©rifier l'allocation
                if self.strategy_allocations[strategy_name] <= 0:
                    continue
                
                # VÃƒÂ©rifier si pas dÃƒÂ©jÃƒÂ  en position
                if self._has_position(symbol, strategy_name):
                    # VÃƒÂ©rifier conditions de sortie seulement
                    signal = strategy.analyze(data)
                    if signal and signal.get('type') == 'EXIT':
                        self._queue_signal(signal, strategy_name, symbol)
                else:
                    # Chercher signal d'entrÃƒÂ©e
                    signal = strategy.analyze(data)
                    if signal and signal.get('type') == 'ENTRY':
                        self._queue_signal(signal, strategy_name, symbol)
                        
            except Exception as e:
                logger.error(f"Erreur analyse {strategy_name} sur {symbol}: {e}")
    
    def _queue_signal(self, signal: Dict, strategy_name: str, symbol: str):
        """
        Ajoute un signal ÃƒÂ  la queue de traitement
        
        Args:
            signal: Le signal gÃƒÂ©nÃƒÂ©rÃƒÂ©
            strategy_name: Nom de la stratÃƒÂ©gie
            symbol: Symbole concernÃƒÂ©
        """
        # Enrichir le signal
        signal['strategy'] = strategy_name
        signal['symbol'] = symbol
        signal['timestamp'] = datetime.now()
        signal['allocation'] = self.strategy_allocations[strategy_name]
        
        # Ajouter ÃƒÂ  la queue
        self.pending_signals.put(signal)
        
        # Logger
        logger.info(f"Ã°Å¸â€œÂ¡ Signal reÃƒÂ§u de {strategy_name}: {signal['side']} {symbol} "
                   f"(confidence: {signal.get('confidence', 0):.2%})")
    
    def _signal_processing_loop(self):
        """Boucle de traitement des signaux"""
        while self.is_running:
            try:
                # RÃƒÂ©cupÃƒÂ©rer le prochain signal
                if not self.pending_signals.empty():
                    signal = self.pending_signals.get(timeout=0.1)
                    self._process_signal(signal)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Erreur processing loop: {e}")
                time.sleep(1)
    
    def _process_signal(self, signal: Dict):
        """
        Traite un signal de trading
        
        Args:
            signal: Le signal ÃƒÂ  traiter
        """
        try:
            strategy_name = signal['strategy']
            symbol = signal['symbol']
            
            # Log
            logger.info(f"Traitement signal {strategy_name}: {signal['type']} {symbol}")
            
            # VÃƒÂ©rifier le risk level
            if self.risk_monitor.current_risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                logger.warning("Risk level trop ÃƒÂ©levÃƒÂ©, signal ignorÃƒÂ©")
                return
            
            if signal['type'] == 'ENTRY':
                self._process_entry_signal(signal)
            elif signal['type'] == 'EXIT':
                self._process_exit_signal(signal)
                
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")
    
    def _process_entry_signal(self, signal: Dict):
        """Traite un signal d'entrÃƒÂ©e"""
        symbol = signal['symbol']
        strategy = signal['strategy']
        
        # VÃƒÂ©rifier qu'on n'a pas dÃƒÂ©jÃƒÂ  une position
        if self._has_position(symbol, strategy):
            logger.warning(f"Position dÃƒÂ©jÃƒÂ  ouverte pour {symbol} avec {strategy}")
            return
        
        # RÃƒÂ©cupÃƒÂ©rer le prix actuel
        ticker = self.exchange.get_symbol_ticker(symbol)
        if not ticker:
            logger.error(f"Impossible de rÃƒÂ©cupÃƒÂ©rer le prix pour {symbol}")
            return
        
        current_price = ticker['price']
        
        # Calculer la taille de position
        stop_loss_price = signal.get('stop_loss', current_price * 0.98)
        
        position_size = self.position_sizer.calculate_position_size(
            signal=signal,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            market_conditions=self._get_market_conditions()
        )
        
        if not position_size or position_size['position_size_usdc'] < 50:
            logger.warning(f"Position size trop petite ou nulle")
            return
        
        # Appliquer l'allocation de la stratÃƒÂ©gie
        allocated_size = position_size['position_size_usdc'] * self.strategy_allocations[strategy]
        
        # VÃƒÂ©rifier avec le risk monitor
        approved, adjusted_size, reason = self.risk_monitor.approve_new_trade(
            signal, 
            allocated_size
        )
        
        if not approved:
            logger.warning(f"Trade rejetÃƒÂ© par risk monitor: {reason}")
            return
        
        # Calculer la quantitÃƒÂ©
        quantity = self.exchange.calculate_quantity_from_usdc(symbol, adjusted_size)
        
        if quantity <= 0:
            logger.error("QuantitÃƒÂ© calculÃƒÂ©e <= 0")
            return
        
        # Soumettre l'ordre
        logger.info(f"Ã°Å¸â€œË† EXECUTION: {signal['side']} {quantity:.6f} {symbol} "
                   f"@ {current_price:.4f} (${adjusted_size:.2f})")
        
        # DÃƒÂ©terminer le type d'ordre
        if signal.get('confidence', 0) > 0.8:
            order_type = "MARKET"  # Haute confiance = market order
            price = None
        else:
            order_type = "LIMIT"   # Confiance moyenne = limit order
            if signal['side'] == 'BUY':
                price = current_price * 0.999  # LÃƒÂ©gÃƒÂ¨rement en dessous
            else:
                price = current_price * 1.001  # LÃƒÂ©gÃƒÂ¨rement au dessus
        
        # Soumettre l'ordre principal
        order = self.order_manager.submit_order(
            symbol=symbol,
            side=signal['side'],
            quantity=quantity,
            order_type=order_type,
            price=price,
            priority=self._get_order_priority(signal),
            strategy=strategy
        )
        
        # Si ordre soumis, configurer SL/TP
        if order:
            # Enregistrer la position
            self._register_position(
                symbol=symbol,
                strategy=strategy,
                order=order,
                signal=signal,
                size_usdc=adjusted_size
            )
            
            # Soumettre OCO pour SL/TP
            take_profit = signal.get('take_profit', current_price * 1.005)
            stop_loss = signal.get('stop_loss', current_price * 0.995)
            
            # Inverser pour ordre de sortie
            exit_side = 'SELL' if signal['side'] == 'BUY' else 'BUY'
            
            self.order_manager.submit_oco_order(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
                strategy=strategy
            )
            
            # Mettre ÃƒÂ  jour les stats
            self.performance_by_strategy[strategy]['total_trades'] += 1
            self.performance_by_strategy[strategy]['last_signal'] = datetime.now()
    
    def _process_exit_signal(self, signal: Dict):
        """Traite un signal de sortie"""
        symbol = signal['symbol']
        strategy = signal['strategy']
        
        # VÃƒÂ©rifier qu'on a une position
        position = self._get_position(symbol, strategy)
        if not position:
            logger.warning(f"Pas de position ÃƒÂ  fermer pour {symbol} avec {strategy}")
            return
        
        # Soumettre l'ordre de sortie
        logger.info(f"Ã°Å¸â€œâ€° FERMETURE: {symbol} pour {strategy} (raison: {signal.get('reason', 'signal')})")
        
        order = self.order_manager.submit_order(
            symbol=symbol,
            side=signal['side'],
            quantity=position['quantity'],
            order_type="MARKET",  # Toujours market pour sortie
            priority=OrderPriority.HIGH,
            strategy=strategy
        )
        
        if order:
            # Calculer le P&L
            ticker = self.exchange.get_symbol_ticker(symbol)
            if ticker:
                exit_price = ticker['price']
                entry_price = position['entry_price']
                
                if position['side'] == 'BUY':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                pnl_usdc = position['size_usdc'] * pnl_pct
                
                # Mettre ÃƒÂ  jour les stats
                if pnl_pct > 0:
                    self.performance_by_strategy[strategy]['winning_trades'] += 1
                
                self.performance_by_strategy[strategy]['total_pnl'] += pnl_usdc
                
                # Informer le risk monitor
                self.risk_monitor.register_trade_close(
                    symbol=symbol,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size_usdc=position['size_usdc'],
                    side=position['side']
                )
                
                # Informer le position sizer
                self.position_sizer.close_position(
                    symbol=symbol,
                    exit_price=exit_price,
                    profit_pct=pnl_pct
                )
                
                logger.info(f"P&L: {pnl_usdc:+.2f} USDC ({pnl_pct:+.2%})")
            
            # Retirer la position
            self._remove_position(symbol, strategy)
    
    def _has_position(self, symbol: str, strategy: str) -> bool:
        """VÃƒÂ©rifie si une position existe"""
        key = f"{symbol}_{strategy}"
        return key in self.positions
    
    def _get_position(self, symbol: str, strategy: str) -> Optional[Dict]:
        """RÃƒÂ©cupÃƒÂ¨re une position"""
        key = f"{symbol}_{strategy}"
        return self.positions.get(key)
    
    def _register_position(self, symbol: str, strategy: str, order, signal: Dict, size_usdc: float):
        """Enregistre une nouvelle position"""
        key = f"{symbol}_{strategy}"
        
        self.positions[key] = {
            'symbol': symbol,
            'strategy': strategy,
            'side': signal['side'],
            'quantity': order.quantity,
            'entry_price': order.price or signal['price'],
            'size_usdc': size_usdc,
            'entry_time': datetime.now(),
            'order_id': order.client_order_id,
            'signal': signal
        }
        
        # Informer la stratÃƒÂ©gie
        if hasattr(self.active_strategies[strategy], 'register_position'):
            self.active_strategies[strategy].register_position(
                symbol=symbol,
                side=signal['side'],
                entry_price=order.price or signal['price'],
                quantity=order.quantity
            )
        
        logger.info(f"Position enregistrÃƒÂ©e: {key}")
    
    def _remove_position(self, symbol: str, strategy: str):
        """Retire une position"""
        key = f"{symbol}_{strategy}"
        if key in self.positions:
            del self.positions[key]
            logger.info(f"Position retirÃƒÂ©e: {key}")
    
    def _get_order_priority(self, signal: Dict) -> OrderPriority:
        """DÃƒÂ©termine la prioritÃƒÂ© d'un ordre"""
        confidence = signal.get('confidence', 0.5)
        
        if confidence > 0.9:
            return OrderPriority.HIGH
        elif confidence > 0.75:
            return OrderPriority.NORMAL
        else:
            return OrderPriority.LOW
    
    def _get_market_conditions(self) -> Dict:
        """RÃƒÂ©cupÃƒÂ¨re les conditions de marchÃƒÂ© actuelles"""
        # Simplified - ÃƒÂ  amÃƒÂ©liorer avec vraies mÃƒÂ©triques
        conditions = {
            'volatility': 0.02,  # Ãƒâ‚¬ calculer
            'trend_strength': 0.5,  # Ãƒâ‚¬ calculer
            'volume_ratio': 1.0,  # Ãƒâ‚¬ calculer
            'market_regime': 'NEUTRAL'  # Ãƒâ‚¬ dÃƒÂ©terminer
        }
        
        return conditions
    
    def _performance_tracking_loop(self):
        """Boucle de suivi de performance"""
        while self.is_running:
            try:
                # Calculer les mÃƒÂ©triques toutes les 60 secondes
                time.sleep(60)
                
                for strategy_name in self.active_strategies:
                    self._update_strategy_performance(strategy_name)
                
                # Log summary
                self._log_performance_summary()
                
            except Exception as e:
                logger.error(f"Erreur tracking performance: {e}")
    
    def _update_strategy_performance(self, strategy_name: str):
        """Met ÃƒÂ  jour les mÃƒÂ©triques de performance d'une stratÃƒÂ©gie"""
        perf = self.performance_by_strategy[strategy_name]
        
        # Calculer win rate
        if perf['total_trades'] > 0:
            win_rate = perf['winning_trades'] / perf['total_trades']
        else:
            win_rate = 0
        
        # Calculer Sharpe (simplifiÃƒÂ©)
        # TODO: ImplÃƒÂ©menter calcul Sharpe complet
        
        perf['win_rate'] = win_rate
    
    def _log_performance_summary(self):
        """Log un rÃƒÂ©sumÃƒÂ© de performance"""
        total_pnl = sum(p['total_pnl'] for p in self.performance_by_strategy.values())
        total_trades = sum(p['total_trades'] for p in self.performance_by_strategy.values())
        
        summary = f"\n{'='*60}\n"
        summary += f"PERFORMANCE SUMMARY - {datetime.now().strftime('%H:%M:%S')}\n"
        summary += f"{'='*60}\n"
        summary += f"Total P&L: ${total_pnl:+.2f}\n"
        summary += f"Total Trades: {total_trades}\n"
        summary += f"Active Positions: {len(self.positions)}\n"
        
        for strategy, perf in self.performance_by_strategy.items():
            if perf['total_trades'] > 0:
                summary += f"\n{strategy}:\n"
                summary += f"  Trades: {perf['total_trades']}\n"
                summary += f"  Win Rate: {perf.get('win_rate', 0):.1%}\n"
                summary += f"  P&L: ${perf['total_pnl']:+.2f}\n"
        
        summary += f"{'='*60}\n"
        
        logger.info(summary)
    
    def enable_trading(self):
        """Active le trading"""
        self.trading_enabled = True
        logger.info("Trading activÃƒÂ©")
    
    def disable_trading(self):
        """DÃƒÂ©sactive le trading (positions existantes continuent)"""
        self.trading_enabled = False
        logger.info("Trading dÃƒÂ©sactivÃƒÂ© (nouvelles positions interdites)")
    
    def close_all_positions(self, reason: str = "manual"):
        """Ferme toutes les positions ouvertes"""
        logger.warning(f"Fermeture de toutes les positions: {reason}")
        
        for position_key in list(self.positions.keys()):
            position = self.positions[position_key]
            
            # CrÃƒÂ©er un signal de sortie
            exit_signal = {
                'type': 'EXIT',
                'side': 'SELL' if position['side'] == 'BUY' else 'BUY',
                'symbol': position['symbol'],
                'strategy': position['strategy'],
                'reason': reason
            }
            
            self._process_exit_signal(exit_signal)
    
    def get_status(self) -> Dict:
        """Retourne le statut du strategy manager"""
        return {
            'is_running': self.is_running,
            'trading_enabled': self.trading_enabled,
            'active_strategies': list(self.active_strategies.keys()),
            'allocations': self.strategy_allocations,
            'open_positions': len(self.positions),
            'positions': [
                {
                    'symbol': p['symbol'],
                    'strategy': p['strategy'],
                    'side': p['side'],
                    'entry_price': p['entry_price'],
                    'size_usdc': p['size_usdc']
                }
                for p in self.positions.values()
            ],
            'performance': {
                name: {
                    'trades': perf['total_trades'],
                    'pnl': perf['total_pnl'],
                    'win_rate': perf.get('win_rate', 0)
                }
                for name, perf in self.performance_by_strategy.items()
            }
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du strategy manager"""
    
    # Configuration de test
    config = {
        'strategies': [
            {
                'name': 'scalping',
                'enabled': True,
                'allocation': 0.6,
                'params': {
                    'rsi_oversold': 30,
                    'min_profit_percent': 0.003
                }
            },
            # Ajouter d'autres stratÃƒÂ©gies quand implÃƒÂ©mentÃƒÂ©es
        ]
    }
    
    # Mocks pour test
    class MockExchange:
        def get_symbol_ticker(self, symbol):
            return {'price': 100.0, 'bid': 99.9, 'ask': 100.1}
        
        def calculate_quantity_from_usdc(self, symbol, usdc):
            return usdc / 100.0  # Prix fixe pour test
    
    class MockOrderManager:
        def submit_order(self, **kwargs):
            print(f"Mock order: {kwargs}")
            order = type('Order', (), {})()
            order.quantity = kwargs['quantity']
            order.price = kwargs.get('price', 100)
            order.client_order_id = f"TEST_{time.time()}"
            return order
        
        def submit_oco_order(self, **kwargs):
            print(f"Mock OCO: {kwargs}")
    
    class MockPositionSizer:
        def calculate_position_size(self, **kwargs):
            return {'position_size_usdc': 100.0}
    
    class MockRiskMonitor:
        current_risk_level = RiskLevel.NORMAL
        
        def approve_new_trade(self, signal, size):
            return True, size, "Approved"
        
        def register_trade_close(self, **kwargs):
            pass
    
    # Initialiser avec mocks
    exchange = MockExchange()
    order_manager = MockOrderManager(self.exchange_client, self.config)
    position_sizer = MockPositionSizer(self.config)
    risk_monitor = MockRiskMonitor()
    
    manager = StrategyManager(
        config=config,
        exchange_client=exchange,
        order_manager=order_manager,
        position_sizer=position_sizer,
        risk_monitor=risk_monitor
    )
    
    # DÃƒÂ©marrer
    manager.start()
    
    # Simuler des donnÃƒÂ©es de marchÃƒÂ©
    import numpy as np
    size = 100
    close_prices = 100 + np.cumsum(np.random.randn(size) * 0.5)
    
    df = pd.DataFrame({
        'close': close_prices,
        'open': close_prices + np.random.randn(size) * 0.1,
        'high': close_prices + abs(np.random.randn(size) * 0.2),
        'low': close_prices - abs(np.random.randn(size) * 0.2),
        'volume': np.random.randint(1000, 10000, size)
    })
    
    # Ajouter indicateurs basiques
    df['rsi'] = 50 + np.random.randn(size) * 20
    df['rsi'] = np.clip(df['rsi'], 0, 100)
    
    market_data = {
        'df': df,
        'orderbook': {
            'bids': [[99.9, 100], [99.8, 200]],
            'asks': [[100.1, 100], [100.2, 200]]
        }
    }
    
    # Envoyer les donnÃƒÂ©es
    print("=" * 60)
    print("TEST: Envoi de donnÃƒÂ©es de marchÃƒÂ©")
    manager.process_market_data('BTCUSDC', market_data)
    
    # Attendre un peu
    time.sleep(2)
    
    # Afficher statut
    print("\n" + "=" * 60)
    print("STATUT:")
    status = manager.get_status()
    for key, value in status.items():
        if key != 'positions':
            print(f"  {key}: {value}")
    
    # ArrÃƒÂªter
    manager.stop()
    print("\nÃ¢Å“â€¦ Test terminÃƒÂ©")
