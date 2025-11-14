"""
Performance Tracker pour The Bot
Calcule et suit les mÃƒÂ©triques de performance avancÃƒÂ©es
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot de performance ÃƒÂ  un instant T"""
    timestamp: datetime
    capital: float
    total_pnl: float
    daily_pnl: float
    positions_count: int
    trades_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    recovery_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceTracker:
    """
    Tracker de performance avancÃƒÂ©
    
    ResponsabilitÃƒÂ©s:
    - Calculer les mÃƒÂ©triques de performance (Sharpe, Sortino, etc.)
    - Suivre le win rate, profit factor
    - Analyser les distributions de profits/pertes
    - DÃƒÂ©tecter les patterns de performance
    - GÃƒÂ©nÃƒÂ©rer des rapports dÃƒÂ©taillÃƒÂ©s
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le performance tracker
        
        Args:
            config: Configuration
        """
        self.config = config
        self.initial_capital = getattr(config, 'INITIAL_CAPITAL', 1000)
        
        # Ãƒâ€°tat actuel
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.daily_starting_capital = self.initial_capital
        
        # Historique des trades
        self.trades_history = []
        self.closed_trades = []  # Tous les trades fermÃƒÂ©s
        self.winning_trades = []
        self.losing_trades = []
        
        # Historique P&L
        self.pnl_history = deque(maxlen=10000)  # Buffer circulaire
        self.daily_pnl_history = []
        
        # Historique drawdown
        self.drawdown_history = deque(maxlen=10000)
        
        # Performance par pÃƒÂ©riode
        self.hourly_performance = defaultdict(list)
        self.daily_performance = defaultdict(list)
        
        # Performance par stratÃƒÂ©gie
        self.strategy_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0
        })
        
        # Performance par symbole
        self.symbol_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0
        })
        
        # Statistiques temps rÃƒÂ©el
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0
        }
        
        # Timestamps
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.last_daily_reset = datetime.now()
        
        logger.info(f"Ã¢Å“â€¦ Performance Tracker initialisÃƒÂ© (capital: ${self.initial_capital:,.2f})")
    
    def record_trade(self,
                    symbol: str,
                    strategy: str,
                    side: str,
                    entry_price: float,
                    exit_price: float,
                    quantity: float,
                    profit: float,
                    entry_time: datetime,
                    exit_time: datetime,
                    metadata: Optional[Dict] = None):
        """
        Enregistre un trade fermÃƒÂ©
        
        Args:
            symbol: Symbole tradÃƒÂ©
            strategy: StratÃƒÂ©gie utilisÃƒÂ©e
            side: BUY ou SELL
            entry_price: Prix d'entrÃƒÂ©e
            exit_price: Prix de sortie
            quantity: QuantitÃƒÂ©
            profit: Profit rÃƒÂ©alisÃƒÂ©
            entry_time: Timestamp d'entrÃƒÂ©e
            exit_time: Timestamp de sortie
            metadata: MÃƒÂ©tadonnÃƒÂ©es optionnelles
        """
        trade = {
            'timestamp': exit_time,
            'symbol': symbol,
            'strategy': strategy,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'profit': profit,
            'profit_pct': (profit / (entry_price * quantity)) if entry_price * quantity > 0 else 0,
            'duration': (exit_time - entry_time).total_seconds(),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'metadata': metadata or {}
        }
        
        # Ajouter ÃƒÂ  l'historique
        self.trades_history.append(trade)
        self.closed_trades.append(trade)
        
        # Classer win/loss
        if profit > 0:
            self.winning_trades.append(trade)
            self.strategy_performance[strategy]['wins'] += 1
            self.symbol_performance[symbol]['wins'] += 1
        else:
            self.losing_trades.append(trade)
            self.strategy_performance[strategy]['losses'] += 1
        
        # Mettre ÃƒÂ  jour les performances par stratÃƒÂ©gie
        self.strategy_performance[strategy]['trades'] += 1
        self.strategy_performance[strategy]['total_pnl'] += profit
        
        # Mettre ÃƒÂ  jour les performances par symbole
        self.symbol_performance[symbol]['trades'] += 1
        self.symbol_performance[symbol]['total_pnl'] += profit
        
        # Mettre ÃƒÂ  jour les stats
        self.stats['total_trades'] += 1
        if profit > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        self.stats['total_pnl'] += profit
        self.stats['daily_pnl'] += profit
        
        # Recalculer les mÃƒÂ©triques
        self._update_statistics()
        
        logger.debug(f"Trade enregistrÃƒÂ©: {symbol} {side} P&L: ${profit:+.2f}")
    
    def update_capital(self, new_capital: float):
        """
        Met ÃƒÂ  jour le capital
        
        Args:
            new_capital: Nouveau capital
        """
        # Enregistrer dans l'historique
        self.pnl_history.append({
            'timestamp': datetime.now(),
            'capital': new_capital,
            'pnl': new_capital - self.current_capital
        })
        
        self.current_capital = new_capital
        
        # Mettre ÃƒÂ  jour le pic
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        # Calculer le drawdown
        self._update_drawdown()
        
        # Reset journalier
        self._check_daily_reset()
    
    def get_snapshot(self) -> PerformanceSnapshot:
        """
        Retourne un snapshot de performance actuel
        
        Returns:
            PerformanceSnapshot
        """
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            capital=self.current_capital,
            total_pnl=self.stats['total_pnl'],
            daily_pnl=self.stats['daily_pnl'],
            positions_count=0,  # Ãƒâ‚¬ remplir depuis l'extÃƒÂ©rieur
            trades_count=self.stats['total_trades'],
            win_rate=self.stats['win_rate'],
            profit_factor=self.stats['profit_factor'],
            sharpe_ratio=self.stats['sharpe_ratio'],
            max_drawdown=self.stats['max_drawdown'],
            current_drawdown=self.stats['current_drawdown'],
            recovery_factor=self._calculate_recovery_factor(),
            avg_win=self.stats['avg_win'],
            avg_loss=self.stats['avg_loss'],
            largest_win=self.stats['largest_win'],
            largest_loss=self.stats['largest_loss']
        )
    
    def get_detailed_report(self) -> Dict:
        """
        GÃƒÂ©nÃƒÂ¨re un rapport dÃƒÂ©taillÃƒÂ© de performance
        
        Returns:
            Dict avec toutes les mÃƒÂ©triques
        """
        # DurÃƒÂ©e de trading
        trading_duration = datetime.now() - self.start_time
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': {
                'days': trading_duration.days,
                'hours': trading_duration.seconds // 3600,
                'total_seconds': trading_duration.total_seconds()
            },
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            },
            'pnl': {
                'total': self.stats['total_pnl'],
                'daily': self.stats['daily_pnl'],
                'avg_per_trade': self.stats['total_pnl'] / self.stats['total_trades'] if self.stats['total_trades'] > 0 else 0
            },
            'trades': {
                'total': self.stats['total_trades'],
                'winning': self.stats['winning_trades'],
                'losing': self.stats['losing_trades'],
                'win_rate': self.stats['win_rate'],
                'avg_per_day': self.stats['total_trades'] / max(trading_duration.days, 1)
            },
            'metrics': {
                'profit_factor': self.stats['profit_factor'],
                'sharpe_ratio': self.stats['sharpe_ratio'],
                'sortino_ratio': self.stats['sortino_ratio'],
                'max_drawdown': self.stats['max_drawdown'],
                'current_drawdown': self.stats['current_drawdown'],
                'recovery_factor': self._calculate_recovery_factor()
            },
            'wins_losses': {
                'avg_win': self.stats['avg_win'],
                'avg_loss': self.stats['avg_loss'],
                'largest_win': self.stats['largest_win'],
                'largest_loss': self.stats['largest_loss'],
                'win_loss_ratio': abs(self.stats['avg_win'] / self.stats['avg_loss']) if self.stats['avg_loss'] != 0 else 0
            },
            'by_strategy': self._get_strategy_breakdown(),
            'by_symbol': self._get_symbol_breakdown(),
            'distribution': self._get_pnl_distribution()
        }
        
        return report
    
    def get_performance_metrics(self) -> Dict:
        """
        Retourne les mÃƒÂ©triques de performance principales
        
        Returns:
            Dict avec les mÃƒÂ©triques clÃƒÂ©s
        """
        return {
            'win_rate': self.stats['win_rate'],
            'profit_factor': self.stats['profit_factor'],
            'sharpe_ratio': self.stats['sharpe_ratio'],
            'sortino_ratio': self.stats['sortino_ratio'],
            'max_drawdown': self.stats['max_drawdown'],
            'current_drawdown': self.stats['current_drawdown'],
            'total_pnl': self.stats['total_pnl'],
            'daily_pnl': self.stats['daily_pnl'],
            'total_trades': self.stats['total_trades'],
            'capital': self.current_capital,
            'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        }
    
    def _update_statistics(self):
        """Met ÃƒÂ  jour toutes les statistiques"""
        if not self.closed_trades:
            return
        
        # Win rate
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = self.stats['winning_trades'] / self.stats['total_trades']
        
        # Profit factor
        total_wins = sum(t['profit'] for t in self.winning_trades)
        total_losses = abs(sum(t['profit'] for t in self.losing_trades))
        
        if total_losses > 0:
            self.stats['profit_factor'] = total_wins / total_losses
        else:
            self.stats['profit_factor'] = float('inf') if total_wins > 0 else 0
        
        # Average win/loss
        if self.winning_trades:
            self.stats['avg_win'] = np.mean([t['profit'] for t in self.winning_trades])
            self.stats['largest_win'] = max(t['profit'] for t in self.winning_trades)
        
        if self.losing_trades:
            self.stats['avg_loss'] = np.mean([t['profit'] for t in self.losing_trades])
            self.stats['largest_loss'] = min(t['profit'] for t in self.losing_trades)
        
        # Sharpe ratio
        self.stats['sharpe_ratio'] = self._calculate_sharpe_ratio()
        
        # Sortino ratio
        self.stats['sortino_ratio'] = self._calculate_sortino_ratio()
        
        # Mise ÃƒÂ  jour performance par stratÃƒÂ©gie
        for strategy, perf in self.strategy_performance.items():
            if perf['trades'] > 0:
                perf['win_rate'] = perf['wins'] / perf['trades']
                perf['avg_pnl'] = perf['total_pnl'] / perf['trades']
    
    def _update_drawdown(self):
        """Met ÃƒÂ  jour le drawdown"""
        if self.current_capital < self.peak_capital:
            current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        else:
            current_dd = 0
        
        self.stats['current_drawdown'] = current_dd
        
        # Max drawdown
        if current_dd > self.stats['max_drawdown']:
            self.stats['max_drawdown'] = current_dd
        
        # Historique
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'drawdown': current_dd,
            'capital': self.current_capital,
            'peak': self.peak_capital
        })
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcule le Sharpe Ratio
        
        Args:
            risk_free_rate: Taux sans risque annualisÃƒÂ©
            
        Returns:
            Sharpe Ratio
        """
        if len(self.closed_trades) < 2:
            return 0.0
        
        # Calculer les returns
        returns = [t['profit_pct'] for t in self.closed_trades]
        
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualiser (supposons 250 jours de trading par an)
        sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(250)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcule le Sortino Ratio (comme Sharpe mais avec downside deviation)
        
        Args:
            risk_free_rate: Taux sans risque annualisÃƒÂ©
            
        Returns:
            Sortino Ratio
        """
        if len(self.closed_trades) < 2:
            return 0.0
        
        returns = [t['profit_pct'] for t in self.closed_trades]
        
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        
        # Downside deviation (seulement les returns nÃƒÂ©gatifs)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (avg_return - risk_free_rate) / downside_std * np.sqrt(250)
        
        return sortino
    
    def _calculate_recovery_factor(self) -> float:
        """
        Calcule le Recovery Factor (Total Profit / Max Drawdown)
        
        Returns:
            Recovery Factor
        """
        if self.stats['max_drawdown'] == 0:
            return float('inf') if self.stats['total_pnl'] > 0 else 0.0
        
        max_dd_value = self.stats['max_drawdown'] * self.peak_capital
        
        if max_dd_value == 0:
            return 0.0
        
        return self.stats['total_pnl'] / max_dd_value
    
    def _get_strategy_breakdown(self) -> Dict:
        """
        Retourne la performance par stratÃƒÂ©gie
        
        Returns:
            Dict avec les stats par stratÃƒÂ©gie
        """
        breakdown = {}
        
        for strategy, perf in self.strategy_performance.items():
            breakdown[strategy] = {
                'trades': perf['trades'],
                'wins': perf['wins'],
                'losses': perf['losses'],
                'win_rate': perf['win_rate'],
                'total_pnl': perf['total_pnl'],
                'avg_pnl': perf['avg_pnl'],
                'pnl_per_trade': perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
            }
        
        return breakdown
    
    def _get_symbol_breakdown(self) -> Dict:
        """
        Retourne la performance par symbole
        
        Returns:
            Dict avec les stats par symbole
        """
        breakdown = {}
        
        for symbol, perf in self.symbol_performance.items():
            breakdown[symbol] = {
                'trades': perf['trades'],
                'wins': perf['wins'],
                'win_rate': perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0,
                'total_pnl': perf['total_pnl'],
                'avg_pnl': perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
            }
        
        return breakdown
    
    def _get_pnl_distribution(self) -> Dict:
        """
        Analyse la distribution des P&L
        
        Returns:
            Dict avec les statistiques de distribution
        """
        if not self.closed_trades:
            return {}
        
        profits = [t['profit'] for t in self.closed_trades]
        
        return {
            'mean': np.mean(profits),
            'median': np.median(profits),
            'std': np.std(profits),
            'min': np.min(profits),
            'max': np.max(profits),
            'q25': np.percentile(profits, 25),
            'q75': np.percentile(profits, 75),
            'skewness': self._calculate_skewness(profits),
            'kurtosis': self._calculate_kurtosis(profits)
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calcule le skewness (asymÃƒÂ©trie)"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        skew = (n / ((n-1) * (n-2))) * sum(((x - mean) / std) ** 3 for x in data)
        
        return skew
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calcule le kurtosis (aplatissement)"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * sum(((x - mean) / std) ** 4 for x in data)
        kurt -= 3 * ((n-1) ** 2) / ((n-2) * (n-3))
        
        return kurt
    
    def _check_daily_reset(self):
        """VÃƒÂ©rifie et effectue le reset journalier si nÃƒÂ©cessaire"""
        now = datetime.now()
        
        # Si on a changÃƒÂ© de jour (UTC)
        if now.date() > self.last_daily_reset.date():
            logger.info(f"Reset journalier - P&L hier: ${self.stats['daily_pnl']:+.2f}")
            
            # Sauvegarder la performance de la journÃƒÂ©e
            self.daily_performance[self.last_daily_reset.date()] = {
                'pnl': self.stats['daily_pnl'],
                'trades': self.stats['total_trades'],
                'win_rate': self.stats['win_rate']
            }
            
            # Reset
            self.stats['daily_pnl'] = 0.0
            self.daily_starting_capital = self.current_capital
            self.last_daily_reset = now
    
    def get_trade_history(self, 
                         limit: Optional[int] = None,
                         strategy: Optional[str] = None,
                         symbol: Optional[str] = None) -> List[Dict]:
        """
        Retourne l'historique des trades
        
        Args:
            limit: Nombre max de trades ÃƒÂ  retourner
            strategy: Filtrer par stratÃƒÂ©gie
            symbol: Filtrer par symbole
            
        Returns:
            Liste des trades
        """
        trades = self.closed_trades.copy()
        
        # Filtres
        if strategy:
            trades = [t for t in trades if t['strategy'] == strategy]
        
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        # Limiter
        if limit:
            trades = trades[-limit:]
        
        return trades
    
    def get_best_worst_trades(self, n: int = 5) -> Dict:
        """
        Retourne les meilleurs et pires trades
        
        Args:
            n: Nombre de trades ÃƒÂ  retourner
            
        Returns:
            Dict avec best et worst trades
        """
        if not self.closed_trades:
            return {'best': [], 'worst': []}
        
        sorted_trades = sorted(self.closed_trades, key=lambda t: t['profit'], reverse=True)
        
        return {
            'best': sorted_trades[:n],
            'worst': sorted_trades[-n:]
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Exporte l'historique des trades en DataFrame
        
        Returns:
            DataFrame pandas
        """
        if not self.closed_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.closed_trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        return df


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du performance tracker"""
    
    # Configuration de test
    config = {
        'initial_capital': 1000.0
    }
    
    tracker = PerformanceTracker(config)
    
    print("\n=== Test Performance Tracker ===\n")
    
    # Simuler quelques trades
    now = datetime.now()
    
    # Trade gagnant
    tracker.record_trade(
        symbol='BTCUSDT',
        strategy='scalping',
        side='BUY',
        entry_price=50000,
        exit_price=50100,
        quantity=0.01,
        profit=10.0,
        entry_time=now - timedelta(minutes=5),
        exit_time=now
    )
    
    # Trade perdant
    tracker.record_trade(
        symbol='ETHUSDT',
        strategy='momentum',
        side='SELL',
        entry_price=3000,
        exit_price=3010,
        quantity=0.1,
        profit=-5.0,
        entry_time=now - timedelta(minutes=10),
        exit_time=now - timedelta(minutes=2)
    )
    
    # Trade gagnant
    tracker.record_trade(
        symbol='BTCUSDT',
        strategy='scalping',
        side='BUY',
        entry_price=50100,
        exit_price=50200,
        quantity=0.01,
        profit=15.0,
        entry_time=now - timedelta(minutes=3),
        exit_time=now
    )
    
    # Mettre ÃƒÂ  jour le capital
    tracker.update_capital(1020.0)
    
    # Snapshot
    snapshot = tracker.get_snapshot()
    print(f"Capital: ${snapshot.capital:,.2f}")
    print(f"Total P&L: ${snapshot.total_pnl:+,.2f}")
    print(f"Win Rate: {snapshot.win_rate:.1%}")
    print(f"Profit Factor: {snapshot.profit_factor:.2f}")
    print(f"Sharpe Ratio: {snapshot.sharpe_ratio:.2f}")
    
    # Rapport dÃƒÂ©taillÃƒÂ©
    print("\n--- Rapport DÃƒÂ©taillÃƒÂ© ---")
    report = tracker.get_detailed_report()
    print(f"Total Trades: {report['trades']['total']}")
    print(f"Return: {report['capital']['return_pct']:+.2f}%")
    
    # Par stratÃƒÂ©gie
    print("\n--- Performance par StratÃƒÂ©gie ---")
    for strategy, perf in report['by_strategy'].items():
        print(f"{strategy}: {perf['trades']} trades, Win Rate: {perf['win_rate']:.1%}, P&L: ${perf['total_pnl']:+.2f}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")