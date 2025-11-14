#!/usr/bin/env python3
"""
📊 Risk Metrics Calculator
Calcule toutes les métriques de risque pour le bot
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskMetrics:
    """Calcule et suit les métriques de risque"""
    
    def __init__(self, initial_capital: float):
        """
        Args:
            initial_capital: Capital initial en USDC
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Historique
        self.equity_curve = [initial_capital]
        self.returns = []
        self.trades_history = []
        
        # Métriques
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def update_capital(self, new_capital: float):
        """Met à jour le capital"""
        self.current_capital = new_capital
        self.equity_curve.append(new_capital)
        
        # Met à jour le peak
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            
        # Calcule le return
        if len(self.equity_curve) > 1:
            last_capital = self.equity_curve[-2]
            ret = (new_capital - last_capital) / last_capital if last_capital > 0 else 0
            self.returns.append(ret)
            
    def add_trade(self, pnl: float, entry_price: float, exit_price: float, 
                  symbol: str, side: str, size: float):
        """
        Enregistre un trade
        
        Args:
            pnl: Profit/Loss du trade
            entry_price: Prix d'entrée
            exit_price: Prix de sortie
            symbol: Symbole tradé
            side: 'long' ou 'short'
            size: Taille de la position
        """
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'return': pnl / (entry_price * size) if entry_price * size > 0 else 0
        }
        
        self.trades_history.append(trade)
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
            
    def calculate_drawdown(self) -> float:
        """
        Calcule le drawdown actuel
        
        Returns:
            Drawdown en % (0.0 à 1.0)
        """
        if self.peak_capital <= 0:
            return 0.0
            
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return max(0.0, drawdown)
        
    def calculate_max_drawdown(self) -> float:
        """
        Calcule le drawdown maximum historique
        
        Returns:
            Max drawdown en %
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
            
        equity = np.array(self.equity_curve)
        
        # Calcule le running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calcule les drawdowns
        drawdowns = (running_max - equity) / running_max
        
        # Max drawdown
        max_dd = np.max(drawdowns)
        
        return max_dd
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcule le Sharpe Ratio
        
        Args:
            risk_free_rate: Taux sans risque annuel (0.0 par défaut)
            
        Returns:
            Sharpe ratio (annualisé)
        """
        if not self.returns or len(self.returns) < 2:
            return 0.0
            
        returns = np.array(self.returns)
        
        # Moyenne et écart-type
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
            
        # Sharpe ratio (annualisé pour trading journalier)
        # On suppose 365 jours de trading
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(365)
        
        return sharpe
        
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcule le Sortino Ratio (comme Sharpe mais seulement downside volatility)
        
        Returns:
            Sortino ratio
        """
        if not self.returns or len(self.returns) < 2:
            return 0.0
            
        returns = np.array(self.returns)
        
        # Moyenne
        mean_return = np.mean(returns)
        
        # Downside deviation (seulement returns négatifs)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # Pas de pertes!
            
        downside_std = np.std(negative_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
            
        # Sortino ratio annualisé
        sortino = (mean_return - risk_free_rate) / downside_std * np.sqrt(365)
        
        return sortino
        
    def calculate_win_rate(self) -> float:
        """
        Calcule le win rate
        
        Returns:
            Win rate en % (0.0 à 1.0)
        """
        if self.total_trades == 0:
            return 0.0
            
        return self.winning_trades / self.total_trades
        
    def calculate_profit_factor(self) -> float:
        """
        Calcule le profit factor (gains totaux / pertes totales)
        
        Returns:
            Profit factor (> 1 = profitable)
        """
        if not self.trades_history:
            return 0.0
            
        total_gains = sum(t['pnl'] for t in self.trades_history if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trades_history if t['pnl'] < 0))
        
        if total_losses == 0:
            return float('inf') if total_gains > 0 else 0.0
            
        return total_gains / total_losses
        
    def calculate_average_win(self) -> float:
        """Calcule le gain moyen"""
        winning = [t['pnl'] for t in self.trades_history if t['pnl'] > 0]
        return np.mean(winning) if winning else 0.0
        
    def calculate_average_loss(self) -> float:
        """Calcule la perte moyenne"""
        losing = [t['pnl'] for t in self.trades_history if t['pnl'] < 0]
        return np.mean(losing) if losing else 0.0
        
    def calculate_expectancy(self) -> float:
        """
        Calcule l'expectancy (gain moyen par trade)
        
        Returns:
            Expectancy en USDC
        """
        if not self.trades_history:
            return 0.0
            
        total_pnl = sum(t['pnl'] for t in self.trades_history)
        return total_pnl / len(self.trades_history)
        
    def calculate_recovery_factor(self) -> float:
        """
        Calcule le recovery factor (profit net / max drawdown)
        
        Returns:
            Recovery factor
        """
        max_dd = self.calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf')
            
        net_profit = self.current_capital - self.initial_capital
        max_dd_amount = self.initial_capital * max_dd
        
        if max_dd_amount == 0:
            return 0.0
            
        return net_profit / max_dd_amount
        
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calcule la Value at Risk (VaR)
        
        Args:
            confidence_level: Niveau de confiance (0.95 = 95%)
            
        Returns:
            VaR en USDC (perte maximale à X% de confiance)
        """
        if not self.returns or len(self.returns) < 10:
            return 0.0
            
        returns = np.array(self.returns)
        
        # Calcule le percentile
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Convertit en USDC
        var_usdc = abs(var_percentile * self.current_capital)
        
        return var_usdc
        
    def calculate_calmar_ratio(self) -> float:
        """
        Calcule le Calmar Ratio (return annuel / max drawdown)
        
        Returns:
            Calmar ratio
        """
        max_dd = self.calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf')
            
        # Return annualisé
        if not self.returns or len(self.returns) < 2:
            return 0.0
            
        mean_daily_return = np.mean(self.returns)
        annual_return = mean_daily_return * 365
        
        return annual_return / max_dd
        
    def get_all_metrics(self) -> Dict:
        """
        Retourne toutes les métriques de risque
        
        Returns:
            Dict avec toutes les métriques
        """
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'net_pnl': self.current_capital - self.initial_capital,
                'return_pct': (self.current_capital - self.initial_capital) / self.initial_capital * 100
            },
            'risk': {
                'current_drawdown': self.calculate_drawdown(),
                'max_drawdown': self.calculate_max_drawdown(),
                'var_95': self.calculate_var(0.95),
                'var_99': self.calculate_var(0.99)
            },
            'performance': {
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'sortino_ratio': self.calculate_sortino_ratio(),
                'calmar_ratio': self.calculate_calmar_ratio(),
                'recovery_factor': self.calculate_recovery_factor()
            },
            'trading': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.calculate_win_rate(),
                'profit_factor': self.calculate_profit_factor(),
                'average_win': self.calculate_average_win(),
                'average_loss': self.calculate_average_loss(),
                'expectancy': self.calculate_expectancy()
            }
        }
        
    def print_metrics(self):
        """Affiche toutes les métriques"""
        metrics = self.get_all_metrics()
        
        print("\n" + "="*60)
        print("📊 RISK METRICS REPORT")
        print("="*60)
        
        print("\n💰 CAPITAL:")
        print(f"  Initial:  ${metrics['capital']['initial']:,.2f}")
        print(f"  Current:  ${metrics['capital']['current']:,.2f}")
        print(f"  Peak:     ${metrics['capital']['peak']:,.2f}")
        print(f"  Net P&L:  ${metrics['capital']['net_pnl']:+,.2f} ({metrics['capital']['return_pct']:+.2f}%)")
        
        print("\n⚠️  RISK:")
        print(f"  Current DD:  {metrics['risk']['current_drawdown']:.2%}")
        print(f"  Max DD:      {metrics['risk']['max_drawdown']:.2%}")
        print(f"  VaR 95%:     ${metrics['risk']['var_95']:,.2f}")
        print(f"  VaR 99%:     ${metrics['risk']['var_99']:,.2f}")
        
        print("\n📈 PERFORMANCE:")
        sharpe = metrics['performance']['sharpe_ratio']
        print(f"  Sharpe:      {sharpe:.2f}")
        sortino = metrics['performance']['sortino_ratio']
        print(f"  Sortino:     {sortino:.2f}")
        calmar = metrics['performance']['calmar_ratio']
        print(f"  Calmar:      {calmar:.2f}")
        recovery = metrics['performance']['recovery_factor']
        print(f"  Recovery:    {recovery:.2f}")
        
        print("\n📊 TRADING:")
        print(f"  Total Trades:    {metrics['trading']['total_trades']}")
        print(f"  Winning:         {metrics['trading']['winning_trades']}")
        print(f"  Losing:          {metrics['trading']['losing_trades']}")
        print(f"  Win Rate:        {metrics['trading']['win_rate']:.1%}")
        print(f"  Profit Factor:   {metrics['trading']['profit_factor']:.2f}")
        print(f"  Avg Win:         ${metrics['trading']['average_win']:,.2f}")
        print(f"  Avg Loss:        ${metrics['trading']['average_loss']:,.2f}")
        print(f"  Expectancy:      ${metrics['trading']['expectancy']:,.2f}")
        
        print("\n" + "="*60 + "\n")


# Test du module
if __name__ == "__main__":
    print("📊 Test du Risk Metrics Calculator...\n")
    
    # Simule des trades
    rm = RiskMetrics(initial_capital=1000.0)
    
    # Quelques trades gagnants
    rm.add_trade(pnl=50, entry_price=100, exit_price=105, symbol='BTC/USDC', side='long', size=1)
    rm.update_capital(1050)
    
    rm.add_trade(pnl=30, entry_price=200, exit_price=205, symbol='ETH/USDC', side='long', size=1)
    rm.update_capital(1080)
    
    # Un trade perdant
    rm.add_trade(pnl=-20, entry_price=150, exit_price=145, symbol='SOL/USDC', side='long', size=1)
    rm.update_capital(1060)
    
    # Affiche les métriques
    rm.print_metrics()
