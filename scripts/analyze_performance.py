#!/usr/bin/env python3
"""
Script d'analyse de performance pour The Bot
GÃƒÂ©nÃƒÂ¨re des statistiques dÃƒÂ©taillÃƒÂ©es et des graphiques
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json

# Ajouter le rÃƒÂ©pertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from typing import Dict, List, Optional
except ImportError as e:
    print(f"Ã¢ÂÅ’ DÃƒÂ©pendance manquante: {e}")
    print("Ã°Å¸â€™Â¡ Installez: pip install pandas numpy")
    sys.exit(1)


class PerformanceAnalyzer:
    """Analyse les performances du bot"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.trades_file = self.data_dir / "trades.json"
        self.positions_file = self.data_dir / "positions.json"
        self.metrics_file = self.data_dir / "metrics.json"
        
        self.trades = []
        self.positions = []
        self.metrics = {}
    
    def load_data(self):
        """Charge les donnÃƒÂ©es de trading"""
        print("Ã°Å¸â€œÅ  Chargement des donnÃƒÂ©es...\n")
        
        # Charger les trades
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                self.trades = json.load(f)
            print(f"   Ã¢Å“â€¦ {len(self.trades)} trades chargÃƒÂ©s")
        else:
            print("   Ã¢Å¡Â Ã¯Â¸Â  Aucun fichier de trades trouvÃƒÂ©")
        
        # Charger les positions
        if self.positions_file.exists():
            with open(self.positions_file, 'r') as f:
                self.positions = json.load(f)
            print(f"   Ã¢Å“â€¦ {len(self.positions)} positions chargÃƒÂ©es")
        else:
            print("   Ã¢Å¡Â Ã¯Â¸Â  Aucun fichier de positions trouvÃƒÂ©")
        
        # Charger les mÃƒÂ©triques
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
            print(f"   Ã¢Å“â€¦ MÃƒÂ©triques chargÃƒÂ©es")
        else:
            print("   Ã¢Å¡Â Ã¯Â¸Â  Aucun fichier de mÃƒÂ©triques trouvÃƒÂ©")
        
        print()
    
    def calculate_statistics(self) -> Dict:
        """Calcule les statistiques globales"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # Conversions
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['profit_usdc'] = df['profit_usdc'].astype(float)
        df['profit_pct'] = df['profit_pct'].astype(float)
        
        # Statistiques de base
        total_trades = len(df)
        winning_trades = len(df[df['profit_usdc'] > 0])
        losing_trades = len(df[df['profit_usdc'] < 0])
        breakeven_trades = len(df[df['profit_usdc'] == 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L
        total_profit = df['profit_usdc'].sum()
        avg_profit = df['profit_usdc'].mean()
        
        winning_df = df[df['profit_usdc'] > 0]
        losing_df = df[df['profit_usdc'] < 0]
        
        avg_win = winning_df['profit_usdc'].mean() if len(winning_df) > 0 else 0
        avg_loss = losing_df['profit_usdc'].mean() if len(losing_df) > 0 else 0
        
        best_trade = df.loc[df['profit_usdc'].idxmax()] if total_trades > 0 else None
        worst_trade = df.loc[df['profit_usdc'].idxmin()] if total_trades > 0 else None
        
        # Profit Factor
        gross_profit = winning_df['profit_usdc'].sum() if len(winning_df) > 0 else 0
        gross_loss = abs(losing_df['profit_usdc'].sum()) if len(losing_df) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe Ratio
        returns = df['profit_pct'] / 100
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # DurÃƒÂ©e moyenne
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['duration'] = pd.to_datetime(df['exit_time']) - pd.to_datetime(df['entry_time'])
            avg_duration = df['duration'].mean()
        else:
            avg_duration = None
        
        # Par stratÃƒÂ©gie
        strategy_stats = {}
        if 'strategy' in df.columns:
            for strategy in df['strategy'].unique():
                strategy_df = df[df['strategy'] == strategy]
                strategy_stats[strategy] = {
                    'trades': len(strategy_df),
                    'win_rate': len(strategy_df[strategy_df['profit_usdc'] > 0]) / len(strategy_df),
                    'total_profit': strategy_df['profit_usdc'].sum(),
                    'avg_profit': strategy_df['profit_usdc'].mean()
                }
        
        # Par symbole
        symbol_stats = {}
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                symbol_stats[symbol] = {
                    'trades': len(symbol_df),
                    'win_rate': len(symbol_df[symbol_df['profit_usdc'] > 0]) / len(symbol_df),
                    'total_profit': symbol_df['profit_usdc'].sum()
                }
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': {
                'profit': float(best_trade['profit_usdc']),
                'symbol': best_trade['symbol'],
                'strategy': best_trade.get('strategy', 'N/A')
            } if best_trade is not None else None,
            'worst_trade': {
                'profit': float(worst_trade['profit_usdc']),
                'symbol': worst_trade['symbol'],
                'strategy': worst_trade.get('strategy', 'N/A')
            } if worst_trade is not None else None,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_duration': str(avg_duration) if avg_duration else 'N/A',
            'strategy_stats': strategy_stats,
            'symbol_stats': symbol_stats
        }
    
    def print_summary(self, stats: Dict):
        """Affiche le rÃƒÂ©sumÃƒÂ© des performances"""
        print("\n" + "="*70)
        print("Ã°Å¸â€œÅ  RÃƒâ€°SUMÃƒâ€° DES PERFORMANCES")
        print("="*70 + "\n")
        
        # Vue d'ensemble
        print("Ã°Å¸â€œË† VUE D'ENSEMBLE")
        print("-" * 70)
        print(f"Total Trades:        {stats['total_trades']:,}")
        print(f"Gagnants:           {stats['winning_trades']:,} ({stats['win_rate']:.1%})")
        print(f"Perdants:           {stats['losing_trades']:,}")
        print(f"Breakeven:          {stats['breakeven_trades']:,}")
        print()
        
        # P&L
        print("Ã°Å¸â€™Â° PROFIT & LOSS")
        print("-" * 70)
        profit_color = "+" if stats['total_profit'] > 0 else ""
        print(f"P&L Total:          {profit_color}${stats['total_profit']:,.2f}")
        print(f"P&L Moyen/Trade:    {profit_color}${stats['avg_profit']:,.2f}")
        print(f"Gain Moyen:         +${stats['avg_win']:,.2f}")
        print(f"Perte Moyenne:      ${stats['avg_loss']:,.2f}")
        print()
        
        # Meilleurs/Pires trades
        if stats['best_trade']:
            print("Ã°Å¸Ââ€  MEILLEURS/PIRES TRADES")
            print("-" * 70)
            best = stats['best_trade']
            print(f"Meilleur:           +${best['profit']:,.2f} ({best['symbol']}, {best['strategy']})")
            
            worst = stats['worst_trade']
            print(f"Pire:               ${worst['profit']:,.2f} ({worst['symbol']}, {worst['strategy']})")
            print()
        
        # Ratios
        print("Ã°Å¸â€œÅ  RATIOS")
        print("-" * 70)
        print(f"Profit Factor:      {stats['profit_factor']:.2f}")
        print(f"Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {stats['max_drawdown']:.2%}")
        print(f"DurÃƒÂ©e Moy/Trade:    {stats['avg_duration']}")
        print()
        
        # Par stratÃƒÂ©gie
        if stats['strategy_stats']:
            print("Ã°Å¸Å½Â¯ PAR STRATÃƒâ€°GIE")
            print("-" * 70)
            for strategy, s_stats in stats['strategy_stats'].items():
                print(f"\n{strategy.upper()}:")
                print(f"  Trades:     {s_stats['trades']:,}")
                print(f"  Win Rate:   {s_stats['win_rate']:.1%}")
                print(f"  P&L Total:  ${s_stats['total_profit']:,.2f}")
                print(f"  P&L Moyen:  ${s_stats['avg_profit']:,.2f}")
        
        # Top symboles
        if stats['symbol_stats']:
            print("\n\nÃ°Å¸â€™Å½ TOP 10 SYMBOLES")
            print("-" * 70)
            sorted_symbols = sorted(
                stats['symbol_stats'].items(),
                key=lambda x: x[1]['total_profit'],
                reverse=True
            )[:10]
            
            for symbol, s_stats in sorted_symbols:
                profit_sign = "+" if s_stats['total_profit'] > 0 else ""
                print(f"{symbol:12} | Trades: {s_stats['trades']:4} | "
                      f"Win Rate: {s_stats['win_rate']:5.1%} | "
                      f"P&L: {profit_sign}${s_stats['total_profit']:,.2f}")
        
        print("\n" + "="*70 + "\n")
    
    def export_to_csv(self, output_file: str = "performance_report.csv"):
        """Exporte les rÃƒÂ©sultats en CSV"""
        if not self.trades:
            print("Ã¢ÂÅ’ Aucune donnÃƒÂ©e ÃƒÂ  exporter")
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(output_file, index=False)
        print(f"Ã¢Å“â€¦ Rapport exportÃƒÂ©: {output_file}")
    
    def generate_plots(self):
        """GÃƒÂ©nÃƒÂ¨re des graphiques (si matplotlib disponible)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("Ã¢Å¡Â Ã¯Â¸Â  matplotlib non disponible pour les graphiques")
            print("Ã°Å¸â€™Â¡ Installez: pip install matplotlib")
            return
        
        if not self.trades:
            print("Ã¢ÂÅ’ Aucune donnÃƒÂ©e pour les graphiques")
            return
        
        df = pd.DataFrame(self.trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['profit_usdc'] = df['profit_usdc'].astype(float)
        
        # CrÃƒÂ©er les subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('THE BOT - Analyse de Performance', fontsize=16, fontweight='bold')
        
        # 1. Courbe de P&L cumulÃƒÂ©
        cumulative_pnl = df['profit_usdc'].cumsum()
        axes[0, 0].plot(df['timestamp'], cumulative_pnl, linewidth=2, color='#2E86AB')
        axes[0, 0].fill_between(df['timestamp'], cumulative_pnl, alpha=0.3, color='#2E86AB')
        axes[0, 0].set_title('P&L CumulÃƒÂ©', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('P&L ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        
        # 2. Distribution des profits
        axes[0, 1].hist(df['profit_usdc'], bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Distribution des Profits', fontweight='bold')
        axes[0, 1].set_xlabel('Profit ($)')
        axes[0, 1].set_ylabel('FrÃƒÂ©quence')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Win Rate par stratÃƒÂ©gie
        if 'strategy' in df.columns:
            strategy_wr = df.groupby('strategy').apply(
                lambda x: (x['profit_usdc'] > 0).sum() / len(x) * 100
            )
            axes[1, 0].bar(strategy_wr.index, strategy_wr.values, color='#F18F01', edgecolor='black')
            axes[1, 0].set_title('Win Rate par StratÃƒÂ©gie', fontweight='bold')
            axes[1, 0].set_xlabel('StratÃƒÂ©gie')
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Trades par jour
        df['date'] = df['timestamp'].dt.date
        trades_per_day = df.groupby('date').size()
        axes[1, 1].bar(range(len(trades_per_day)), trades_per_day.values, color='#6A994E', edgecolor='black')
        axes[1, 1].set_title('Nombre de Trades par Jour', fontweight='bold')
        axes[1, 1].set_xlabel('Jours')
        axes[1, 1].set_ylabel('Nombre de Trades')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "performance_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Ã¢Å“â€¦ Graphiques sauvegardÃƒÂ©s: {output_file}")
        
        # Afficher
        plt.show()


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Analyse de performance The Bot")
    parser.add_argument('--data-dir', default='data', help='Dossier des donnÃƒÂ©es')
    parser.add_argument('--export-csv', action='store_true', help='Exporter en CSV')
    parser.add_argument('--no-plots', action='store_true', help='Ne pas gÃƒÂ©nÃƒÂ©rer les graphiques')
    parser.add_argument('--output', default='performance_report', help='Nom du fichier de sortie')
    
    args = parser.parse_args()
    
    # En-tÃƒÂªte
    print("\n" + "="*70)
    print("Ã°Å¸â€œÅ  THE BOT - ANALYSE DE PERFORMANCE")
    print("="*70 + "\n")
    
    # CrÃƒÂ©er l'analyseur
    analyzer = PerformanceAnalyzer(data_dir=args.data_dir)
    
    # Charger les donnÃƒÂ©es
    analyzer.load_data()
    
    if not analyzer.trades:
        print("Ã¢ÂÅ’ Aucune donnÃƒÂ©e de trading trouvÃƒÂ©e")
        print("Ã°Å¸â€™Â¡ Assurez-vous que le bot a effectuÃƒÂ© des trades")
        return
    
    # Calculer les statistiques
    print("Ã°Å¸â€Â¢ Calcul des statistiques...\n")
    stats = analyzer.calculate_statistics()
    
    # Afficher le rÃƒÂ©sumÃƒÂ©
    analyzer.print_summary(stats)
    
    # Exporter CSV si demandÃƒÂ©
    if args.export_csv:
        analyzer.export_to_csv(f"{args.output}.csv")
    
    # GÃƒÂ©nÃƒÂ©rer les graphiques
    if not args.no_plots:
        print("\nÃ°Å¸â€œË† GÃƒÂ©nÃƒÂ©ration des graphiques...")
        analyzer.generate_plots()
    
    print("\nÃ¢Å“â€¦ Analyse terminÃƒÂ©e!\n")


if __name__ == "__main__":
    main()
