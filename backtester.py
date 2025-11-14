#!/usr/bin/env python3
"""
Backtester pour The Bot
Test les stratÃƒÂ©gies sur donnÃƒÂ©es historiques avec simulation rÃƒÂ©aliste
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration du backtest"""
    initial_capital: float = 1000.0
    commission: float = 0.0007  # 0.07% Binance
    slippage: float = 0.001  # 0.1% slippage moyen
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    timeframe: str = "5m"
    risk_per_trade: float = 0.02  # 2% risque par trade
    max_positions: int = 10
    max_drawdown: float = 0.08  # 8% drawdown max
    use_stop_loss: bool = True
    use_take_profit: bool = True
    pyramiding: bool = False  # Permet positions multiples mÃƒÂªme symbole
    realistic_fills: bool = True  # Simule fills rÃƒÂ©alistes
    save_results: bool = True
    plot_results: bool = True


@dataclass
class Trade:
    """ReprÃƒÂ©sente un trade"""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    side: str  # 'LONG' ou 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission_paid: float
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_reason: Optional[str] = None
    is_open: bool = True
    metadata: Dict = None


class Backtester:
    """
    Backtester principal avec simulation rÃƒÂ©aliste
    
    Features:
    - Multi-stratÃƒÂ©gie
    - Gestion rÃƒÂ©aliste des commissions et slippage
    - Risk management intÃƒÂ©grÃƒÂ©
    - MÃƒÂ©triques complÃƒÂ¨tes
    - Visualisations
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialise le backtester
        
        Args:
            config: Configuration du backtest
        """
        self.config = config or BacktestConfig()
        
        # Ãƒâ€°tat du portfolio
        self.initial_capital = self.config.initial_capital
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.peak_capital = self.initial_capital
        
        # Positions et trades
        self.open_positions = {}
        self.closed_trades = []
        self.all_trades = []
        
        # Historique
        self.equity_curve = []
        self.drawdown_curve = []
        self.signals_history = []
        
        # MÃƒÂ©triques
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_duration': 0,
            'total_return': 0,
            'annual_return': 0,
            'volatility': 0,
            'best_trade': None,
            'worst_trade': None,
            'avg_trade_duration': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'expectancy': 0,
            'kelly_criterion': 0
        }
        
        logger.info(f"Backtester initialisÃƒÂ© - Capital: ${self.initial_capital:,.2f}")
    
    def run(self, 
            data: pd.DataFrame, 
            strategy,
            symbols: List[str] = None) -> Dict:
        """
        Lance le backtest
        
        Args:
            data: DataFrame avec donnÃƒÂ©es OHLCV
            strategy: Instance de stratÃƒÂ©gie ÃƒÂ  tester
            symbols: Liste des symboles ÃƒÂ  trader (None = tous)
            
        Returns:
            Dictionnaire avec rÃƒÂ©sultats et mÃƒÂ©triques
        """
        logger.info(f"DÃƒÂ©but backtest: {self.config.start_date} -> {self.config.end_date}")
        
        # Filtrer les donnÃƒÂ©es par date
        data = self._filter_data(data)
        
        if symbols:
            data = data[data['symbol'].isin(symbols)]
        
        # Reset ÃƒÂ©tat
        self._reset_state()
        
        # Progress bar
        total_bars = len(data)
        pbar = tqdm(total=total_bars, desc="Backtesting")
        
        # ItÃƒÂ©rer sur chaque barre
        for timestamp, bar_data in data.groupby('timestamp'):
            # Update positions avec nouveaux prix
            self._update_open_positions(bar_data)
            
            # VÃƒÂ©rifier stop loss / take profit
            if self.config.use_stop_loss or self.config.use_take_profit:
                self._check_exit_conditions(bar_data)
            
            # GÃƒÂ©nÃƒÂ©rer signaux avec la stratÃƒÂ©gie
            signals = self._generate_signals(strategy, bar_data)
            
            # ExÃƒÂ©cuter les signaux
            for signal in signals:
                self._process_signal(signal, bar_data)
            
            # Enregistrer l'ÃƒÂ©tat
            self._record_state(timestamp)
            
            # Check drawdown limit
            if self._check_drawdown_limit():
                logger.warning(f"Drawdown max atteint: {self.metrics['max_drawdown']:.2%}")
                break
            
            pbar.update(len(bar_data))
        
        pbar.close()
        
        # Fermer positions restantes
        self._close_all_positions(data.iloc[-1])
        
        # Calculer mÃƒÂ©triques finales
        self._calculate_metrics()
        
        # GÃƒÂ©nÃƒÂ©rer rapport
        results = self._generate_results()
        
        # Sauvegarder si demandÃƒÂ©
        if self.config.save_results:
            self._save_results(results)
        
        # Plots si demandÃƒÂ©
        if self.config.plot_results:
            self._plot_results()
        
        logger.info(f"Backtest terminÃƒÂ© - Return: {self.metrics['total_return']:.2%}")
        
        return results
    
    def _reset_state(self):
        """Reset l'ÃƒÂ©tat pour un nouveau backtest"""
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.peak_capital = self.initial_capital
        self.open_positions = {}
        self.closed_trades = []
        self.all_trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.signals_history = []
    
    def _filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtre les donnÃƒÂ©es par date"""
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        
        mask = (data['timestamp'] >= start) & (data['timestamp'] <= end)
        return data[mask].copy()
    
    def _generate_signals(self, strategy, bar_data: pd.DataFrame) -> List[Dict]:
        """
        GÃƒÂ©nÃƒÂ¨re les signaux de la stratÃƒÂ©gie
        
        Args:
            strategy: Instance de stratÃƒÂ©gie
            bar_data: DonnÃƒÂ©es de la barre courante
            
        Returns:
            Liste des signaux
        """
        signals = []
        
        for symbol in bar_data['symbol'].unique():
            symbol_data = bar_data[bar_data['symbol'] == symbol]
            
            # PrÃƒÂ©parer donnÃƒÂ©es pour stratÃƒÂ©gie
            data_dict = {
                'symbol': symbol,
                'df': symbol_data,
                'orderbook': self._simulate_orderbook(symbol_data)
            }
            
            # Analyser avec la stratÃƒÂ©gie
            signal = strategy.analyze(data_dict)
            
            if signal and signal.get('confidence', 0) >= strategy.min_confidence:
                signals.append(signal)
                self.signals_history.append(signal)
        
        return signals
    
    def _process_signal(self, signal: Dict, bar_data: pd.DataFrame):
        """
        Traite un signal de trading
        
        Args:
            signal: Le signal ÃƒÂ  traiter
            bar_data: DonnÃƒÂ©es de marchÃƒÂ© actuelles
        """
        symbol = signal['symbol']
        side = signal['side']
        
        # VÃƒÂ©rifier si on a dÃƒÂ©jÃƒÂ  une position
        if not self.config.pyramiding and symbol in self.open_positions:
            return
        
        # VÃƒÂ©rifier nombre max de positions
        if len(self.open_positions) >= self.config.max_positions:
            return
        
        # Calculer la taille de position
        position_size = self._calculate_position_size(signal)
        
        if position_size * signal['price'] > self.cash:
            # Pas assez de cash
            return
        
        # CrÃƒÂ©er le trade
        trade = self._open_trade(symbol, side, signal['price'], position_size, signal)
        
        if trade:
            self.open_positions[symbol] = trade
            self.all_trades.append(trade)
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """
        Calcule la taille de position optimale
        
        Args:
            signal: Signal avec infos de trade
            
        Returns:
            Taille de position en unitÃƒÂ©s
        """
        # Risque en dollars
        risk_amount = self.capital * self.config.risk_per_trade
        
        # Ajuster par confidence du signal
        confidence = signal.get('confidence', 0.5)
        risk_amount *= min(1.5, max(0.5, confidence))
        
        # Calculer avec stop loss
        entry_price = signal['price']
        stop_loss = signal.get('stop_loss', entry_price * 0.98)
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # Position size
        position_value = risk_amount / stop_distance
        position_size = position_value / entry_price
        
        # Limiter ÃƒÂ  25% du capital
        max_position = self.capital * 0.25 / entry_price
        position_size = min(position_size, max_position)
        
        return position_size
    
    def _open_trade(self, symbol: str, side: str, price: float, 
                   quantity: float, signal: Dict) -> Optional[Trade]:
        """
        Ouvre un nouveau trade
        
        Args:
            symbol: Symbole ÃƒÂ  trader
            side: Direction (LONG/SHORT)
            price: Prix d'entrÃƒÂ©e
            quantity: QuantitÃƒÂ©
            signal: Signal original
            
        Returns:
            Trade crÃƒÂ©ÃƒÂ© ou None
        """
        # Appliquer slippage
        if self.config.realistic_fills:
            if side == 'BUY':
                price *= (1 + self.config.slippage)
            else:
                price *= (1 - self.config.slippage)
        
        # Calculer commission
        trade_value = price * quantity
        commission = trade_value * self.config.commission
        
        # VÃƒÂ©rifier le cash disponible
        total_cost = trade_value + commission
        if total_cost > self.cash:
            return None
        
        # CrÃƒÂ©er le trade
        trade = Trade(
            symbol=symbol,
            entry_time=datetime.now(),
            exit_time=None,
            side=side,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            commission_paid=commission,
            metadata={
                'signal': signal,
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit')
            }
        )
        
        # Mettre ÃƒÂ  jour le cash
        self.cash -= total_cost
        
        logger.debug(f"Trade ouvert: {symbol} {side} @ {price:.4f} x {quantity:.6f}")
        
        return trade
    
    def _close_trade(self, trade: Trade, exit_price: float, exit_reason: str = "signal"):
        """
        Ferme un trade
        
        Args:
            trade: Trade ÃƒÂ  fermer
            exit_price: Prix de sortie
            exit_reason: Raison de sortie
        """
        # Appliquer slippage
        if self.config.realistic_fills:
            if trade.side == 'BUY':
                exit_price *= (1 - self.config.slippage)
            else:
                exit_price *= (1 + self.config.slippage)
        
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = exit_reason
        trade.is_open = False
        
        # Calculer P&L
        if trade.side == 'BUY':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Commission de sortie
        exit_commission = exit_price * trade.quantity * self.config.commission
        trade.commission_paid += exit_commission
        trade.pnl -= trade.commission_paid
        
        # P&L en pourcentage
        trade.pnl_percent = trade.pnl / (trade.entry_price * trade.quantity)
        
        # Mettre ÃƒÂ  jour cash
        self.cash += (exit_price * trade.quantity - exit_commission)
        
        # Enregistrer
        self.closed_trades.append(trade)
        
        # Retirer des positions ouvertes
        if trade.symbol in self.open_positions:
            del self.open_positions[trade.symbol]
        
        logger.debug(f"Trade fermÃƒÂ©: {trade.symbol} @ {exit_price:.4f} | PnL: {trade.pnl_percent:.2%}")
    
    def _update_open_positions(self, bar_data: pd.DataFrame):
        """Met ÃƒÂ  jour la valeur des positions ouvertes"""
        for symbol, trade in self.open_positions.items():
            if symbol in bar_data['symbol'].values:
                current_price = bar_data[bar_data['symbol'] == symbol]['close'].iloc[0]
                
                # Calculer P&L non rÃƒÂ©alisÃƒÂ©
                if trade.side == 'BUY':
                    unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                
                trade.metadata['unrealized_pnl'] = unrealized_pnl
    
    def _check_exit_conditions(self, bar_data: pd.DataFrame):
        """VÃƒÂ©rifie les conditions de sortie (SL/TP)"""
        for symbol, trade in list(self.open_positions.items()):
            if symbol not in bar_data['symbol'].values:
                continue
            
            bar = bar_data[bar_data['symbol'] == symbol].iloc[0]
            
            # RÃƒÂ©cupÃƒÂ©rer SL/TP du metadata
            stop_loss = trade.metadata.get('stop_loss')
            take_profit = trade.metadata.get('take_profit')
            
            should_exit = False
            exit_price = None
            exit_reason = None
            
            if trade.side == 'BUY':
                # Check Stop Loss
                if stop_loss and bar['low'] <= stop_loss:
                    should_exit = True
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                # Check Take Profit
                elif take_profit and bar['high'] >= take_profit:
                    should_exit = True
                    exit_price = take_profit
                    exit_reason = "take_profit"
            
            else:  # SHORT
                # Check Stop Loss
                if stop_loss and bar['high'] >= stop_loss:
                    should_exit = True
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                # Check Take Profit
                elif take_profit and bar['low'] <= take_profit:
                    should_exit = True
                    exit_price = take_profit
                    exit_reason = "take_profit"
            
            if should_exit:
                self._close_trade(trade, exit_price, exit_reason)
    
    def _close_all_positions(self, last_bar):
        """Ferme toutes les positions ouvertes"""
        for symbol, trade in list(self.open_positions.items()):
            # Utiliser le dernier prix connu
            exit_price = last_bar['close']
            self._close_trade(trade, exit_price, "end_of_backtest")
    
    def _record_state(self, timestamp):
        """Enregistre l'ÃƒÂ©tat actuel du portfolio"""
        # Calculer valeur totale
        positions_value = sum(
            trade.metadata.get('unrealized_pnl', 0) + trade.entry_price * trade.quantity
            for trade in self.open_positions.values()
        )
        
        total_equity = self.cash + positions_value
        
        # Mettre ÃƒÂ  jour capital
        self.capital = total_equity
        self.peak_capital = max(self.peak_capital, total_equity)
        
        # Calculer drawdown
        drawdown = 0
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - total_equity) / self.peak_capital
        
        # Enregistrer
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.cash,
            'positions_value': positions_value,
            'drawdown': drawdown,
            'num_positions': len(self.open_positions)
        })
        
        self.drawdown_curve.append(drawdown)
    
    def _check_drawdown_limit(self) -> bool:
        """VÃƒÂ©rifie si on a atteint le drawdown max"""
        if not self.drawdown_curve:
            return False
        
        current_dd = self.drawdown_curve[-1]
        return current_dd >= self.config.max_drawdown
    
    def _calculate_metrics(self):
        """Calcule toutes les mÃƒÂ©triques de performance"""
        if not self.closed_trades:
            logger.warning("Aucun trade fermÃƒÂ© pour calculer les mÃƒÂ©triques")
            return
        
        # Trades gagnants/perdants
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        self.metrics['total_trades'] = len(self.closed_trades)
        self.metrics['winning_trades'] = len(winning_trades)
        self.metrics['losing_trades'] = len(losing_trades)
        
        # Win rate
        self.metrics['win_rate'] = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
        
        # Average win/loss
        self.metrics['avg_win'] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        self.metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        self.metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Best/worst trade
        self.metrics['best_trade'] = max(self.closed_trades, key=lambda t: t.pnl_percent)
        self.metrics['worst_trade'] = min(self.closed_trades, key=lambda t: t.pnl_percent)
        
        # Returns
        self.metrics['total_return'] = (self.capital - self.initial_capital) / self.initial_capital
        
        # Drawdown
        self.metrics['max_drawdown'] = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Sharpe Ratio (simplified)
        if self.equity_curve:
            returns = pd.Series([e['equity'] for e in self.equity_curve]).pct_change().dropna()
            if len(returns) > 1:
                self.metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                self.metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Expectancy
        if self.metrics['win_rate'] > 0 and self.metrics['avg_loss'] != 0:
            self.metrics['expectancy'] = (
                self.metrics['win_rate'] * self.metrics['avg_win'] + 
                (1 - self.metrics['win_rate']) * self.metrics['avg_loss']
            )
        
        # Kelly Criterion
        if self.metrics['avg_loss'] != 0:
            W = self.metrics['win_rate']
            R = abs(self.metrics['avg_win'] / self.metrics['avg_loss'])
            self.metrics['kelly_criterion'] = (W - (1-W)/R) if R > 0 else 0
        
        # Commissions totales
        self.metrics['total_commission'] = sum(t.commission_paid for t in self.closed_trades)
    
    def _simulate_orderbook(self, data: pd.DataFrame) -> Dict:
        """
        Simule un orderbook basique
        
        Args:
            data: DonnÃƒÂ©es OHLCV
            
        Returns:
            Orderbook simulÃƒÂ©
        """
        if data.empty:
            return {}
        
        last_price = data['close'].iloc[-1]
        spread = last_price * 0.001  # 0.1% spread
        
        return {
            'bids': [[last_price - spread/2, 1000]],  # Prix, Volume
            'asks': [[last_price + spread/2, 1000]],
            'spread': spread,
            'mid_price': last_price
        }
    
    def _generate_results(self) -> Dict:
        """GÃƒÂ©nÃƒÂ¨re le rapport de rÃƒÂ©sultats"""
        return {
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'trades': [
                {
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'exit_reason': t.exit_reason
                }
                for t in self.closed_trades
            ],
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_return': self.metrics['total_return'],
                'win_rate': self.metrics['win_rate'],
                'profit_factor': self.metrics['profit_factor'],
                'max_drawdown': self.metrics['max_drawdown'],
                'sharpe_ratio': self.metrics['sharpe_ratio'],
                'total_trades': self.metrics['total_trades']
            }
        }
    
    def _save_results(self, results: Dict):
        """Sauvegarde les rÃƒÂ©sultats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"RÃƒÂ©sultats sauvegardÃƒÂ©s dans {filename}")
    
    def _plot_results(self):
        """GÃƒÂ©nÃƒÂ¨re les graphiques de rÃƒÂ©sultats"""
        if not self.equity_curve:
            return
        
        # Setup style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Equity Curve
        equity_df = pd.DataFrame(self.equity_curve)
        axes[0, 0].plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2)
        axes[0, 0].fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'], alpha=0.3)
        axes[0, 0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[0, 1].fill_between(equity_df['timestamp'], 0, -equity_df['drawdown']*100, color='red', alpha=0.5)
        axes[0, 1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution des returns
        if self.closed_trades:
            returns = [t.pnl_percent * 100 for t in self.closed_trades]
            axes[1, 0].hist(returns, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Distribution des Returns', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Return (%)')
            axes[1, 0].set_ylabel('FrÃƒÂ©quence')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Win/Loss ratio
        if self.metrics['winning_trades'] > 0 or self.metrics['losing_trades'] > 0:
            sizes = [self.metrics['winning_trades'], self.metrics['losing_trades']]
            labels = [f"Wins ({self.metrics['winning_trades']})", 
                     f"Losses ({self.metrics['losing_trades']})"]
            colors = ['#2ecc71', '#e74c3c']
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')
        
        # 5. Monthly Returns Heatmap
        if len(self.equity_curve) > 30:
            monthly_returns = self._calculate_monthly_returns()
            if not monthly_returns.empty:
                sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', 
                          center=0, ax=axes[2, 0], cbar_kws={'label': 'Return (%)'})
                axes[2, 0].set_title('Returns Mensuels (%)', fontsize=14, fontweight='bold')
        
        # 6. MÃƒÂ©triques clÃƒÂ©s
        metrics_text = f"""
        Total Return: {self.metrics['total_return']:.2%}
        Win Rate: {self.metrics['win_rate']:.1%}
        Profit Factor: {self.metrics['profit_factor']:.2f}
        Max Drawdown: {self.metrics['max_drawdown']:.2%}
        Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
        Total Trades: {self.metrics['total_trades']}
        Avg Win: ${self.metrics['avg_win']:.2f}
        Avg Loss: ${self.metrics['avg_loss']:.2f}
        Expectancy: ${self.metrics['expectancy']:.2f}
        """
        
        axes[2, 1].text(0.1, 0.5, metrics_text, transform=axes[2, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 1].set_title('MÃƒÂ©triques ClÃƒÂ©s', fontsize=14, fontweight='bold')
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_chart_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        logger.info(f"Graphique sauvegardÃƒÂ© dans {filename}")
        
        plt.show()
    
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calcule les returns mensuels pour heatmap"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Resample monthly
        monthly = equity_df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Reshape pour heatmap
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
        
        # CrÃƒÂ©er matrice annÃƒÂ©e x mois
        years = sorted(set(pd.to_datetime(monthly_returns.index).year))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        matrix = pd.DataFrame(index=years, columns=months)
        
        for date, ret in monthly_returns.items():
            dt = pd.to_datetime(date)
            matrix.loc[dt.year, months[dt.month-1]] = ret
        
        return matrix.astype(float)
    
    def print_summary(self):
        """Affiche un rÃƒÂ©sumÃƒÂ© des rÃƒÂ©sultats"""
        print("\n" + "="*60)
        print("Ã°Å¸â€œÅ  RÃƒâ€°SUMÃƒâ€° DU BACKTEST")
        print("="*60)
        
        print(f"\nÃ°Å¸â€™Â° CAPITAL:")
        print(f"  Ã¢â‚¬Â¢ Initial: ${self.initial_capital:,.2f}")
        print(f"  Ã¢â‚¬Â¢ Final: ${self.capital:,.2f}")
        print(f"  Ã¢â‚¬Â¢ Return: {self.metrics['total_return']:.2%}")
        
        print(f"\nÃ°Å¸â€œË† PERFORMANCE:")
        print(f"  Ã¢â‚¬Â¢ Total Trades: {self.metrics['total_trades']}")
        print(f"  Ã¢â‚¬Â¢ Win Rate: {self.metrics['win_rate']:.1%}")
        print(f"  Ã¢â‚¬Â¢ Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"  Ã¢â‚¬Â¢ Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        
        print(f"\nÃ°Å¸â€œâ€° RISQUE:")
        print(f"  Ã¢â‚¬Â¢ Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        print(f"  Ã¢â‚¬Â¢ Volatility: {self.metrics['volatility']:.2%}")
        
        print(f"\nÃ°Å¸â€™Â¸ COÃƒâ€ºTS:")
        print(f"  Ã¢â‚¬Â¢ Total Commission: ${self.metrics['total_commission']:.2f}")
        
        if self.metrics['best_trade']:
            print(f"\nÃ°Å¸Ââ€  MEILLEUR TRADE:")
            print(f"  Ã¢â‚¬Â¢ Symbol: {self.metrics['best_trade'].symbol}")
            print(f"  Ã¢â‚¬Â¢ Return: {self.metrics['best_trade'].pnl_percent:.2%}")
        
        if self.metrics['worst_trade']:
            print(f"\nÃ°Å¸â€™â‚¬ PIRE TRADE:")
            print(f"  Ã¢â‚¬Â¢ Symbol: {self.metrics['worst_trade'].symbol}")
            print(f"  Ã¢â‚¬Â¢ Return: {self.metrics['worst_trade'].pnl_percent:.2%}")
        
        print("="*60 + "\n")