"""
Dashboard
Dashboard en temps rÃƒÂ©el pour monitoring du bot
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Dashboard en temps rÃƒÂ©el du bot
    
    Affiche:
    - Statut global
    - Capital et P&L
    - Positions ouvertes
    - Performance (win rate, drawdown)
    - Derniers trades
    - SantÃƒÂ© du systÃƒÂ¨me
    """
    
    def __init__(self):
        """Initialise le dashboard"""
        self.last_update = None
        self.update_interval = 5  # secondes
        
        logger.info("Ã°Å¸â€œÅ  Dashboard initialisÃƒÂ©")
    
    def display(self, data: Dict):
        """
        Affiche le dashboard
        
        Args:
            data: DonnÃƒÂ©es ÃƒÂ  afficher
        """
        # Effacer l'ÃƒÂ©cran (cross-platform)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        self._print_header()
        
        # Section 1: Status
        self._print_status(data.get('status', {}))
        
        # Section 2: Capital & P&L
        self._print_capital(data.get('portfolio', {}))
        
        # Section 3: Positions
        self._print_positions(data.get('positions', {}))
        
        # Section 4: Performance
        self._print_performance(data.get('performance', {}))
        
        # Section 5: Derniers trades
        self._print_recent_trades(data.get('recent_trades', []))
        
        # Section 6: SantÃƒÂ© systÃƒÂ¨me
        self._print_health(data.get('health', {}))
        
        # Footer
        self._print_footer()
        
        self.last_update = datetime.now()
    
    def _print_header(self):
        """Affiche l'en-tÃƒÂªte"""
        print("Ã¢â€¢â€" + "Ã¢â€¢Â" * 78 + "Ã¢â€¢â€”")
        print("Ã¢â€¢â€˜" + " " * 25 + "Ã°Å¸Â¤â€“ THE BOT - DASHBOARD" + " " * 31 + "Ã¢â€¢â€˜")
        print("Ã¢â€¢â€˜" + " " * 18 + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 37 + "Ã¢â€¢â€˜")
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_status(self, status: Dict):
        """Affiche le statut"""
        is_running = status.get('is_running', False)
        mode = status.get('mode', 'unknown')
        uptime = status.get('uptime_hours', 0)
        
        status_icon = "Ã°Å¸Å¸Â¢" if is_running else "Ã°Å¸â€Â´"
        status_text = "RUNNING" if is_running else "STOPPED"
        
        print("Ã¢â€¢â€˜ STATUS")
        print("Ã¢â€¢â€˜" + "-" * 78)
        print(f"Ã¢â€¢â€˜ {status_icon} {status_text:<20} Mode: {mode:<15} Uptime: {uptime:.1f}h" + " " * 18 + "Ã¢â€¢â€˜")
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_capital(self, portfolio: Dict):
        """Affiche capital et P&L"""
        initial = portfolio.get('initial_capital', 0)
        current = portfolio.get('current_capital', 0)
        pnl = portfolio.get('total_pnl', 0)
        pnl_pct = portfolio.get('total_pnl_pct', 0)
        unrealized = portfolio.get('unrealized_pnl', 0)
        
        # Couleur pour P&L
        pnl_color = "+" if pnl >= 0 else ""
        
        print("Ã¢â€¢â€˜ CAPITAL & P&L")
        print("Ã¢â€¢â€˜" + "-" * 78)
        print(f"Ã¢â€¢â€˜ Initial:     ${initial:>12,.2f}                                              Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Current:     ${current:>12,.2f}                                              Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ P&L:         ${pnl:>{pnl_color}12,.2f} ({pnl_pct:>{pnl_color}.2%})                                   Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Unrealized:  ${unrealized:>12,.2f}                                              Ã¢â€¢â€˜")
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_positions(self, positions: Dict):
        """Affiche les positions ouvertes"""
        open_count = positions.get('open_count', 0)
        max_positions = positions.get('max_positions', 20)
        exposure_pct = positions.get('exposure_pct', 0)
        
        print("Ã¢â€¢â€˜ POSITIONS")
        print("Ã¢â€¢â€˜" + "-" * 78)
        print(f"Ã¢â€¢â€˜ Open:        {open_count}/{max_positions}                                                           Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Exposure:    {exposure_pct:.1%}                                                      Ã¢â€¢â€˜")
        
        # Lister les positions
        position_list = positions.get('list', [])
        if position_list:
            print("Ã¢â€¢â€˜")
            print("Ã¢â€¢â€˜ Symbol       Side    Size        Entry       Current     P&L                Ã¢â€¢â€˜")
            print("Ã¢â€¢â€˜" + "-" * 78)
            
            for pos in position_list[:5]:  # Max 5 positions affichÃƒÂ©es
                symbol = pos.get('symbol', '')[:10]
                side = pos.get('side', '')
                size = pos.get('size', 0)
                entry = pos.get('entry_price', 0)
                current = pos.get('current_price', 0)
                pnl_pct = pos.get('unrealized_pnl_pct', 0)
                
                pnl_symbol = "+" if pnl_pct >= 0 else ""
                
                print(f"Ã¢â€¢â€˜ {symbol:<12} {side:<7} {size:<11.4f} {entry:<11,.2f} {current:<11,.2f} {pnl_symbol}{pnl_pct:.2%}" + " " * 8 + "Ã¢â€¢â€˜")
            
            if len(position_list) > 5:
                print(f"Ã¢â€¢â€˜ ... et {len(position_list) - 5} autres positions" + " " * 43 + "Ã¢â€¢â€˜")
        else:
            print("Ã¢â€¢â€˜ Aucune position ouverte                                                      Ã¢â€¢â€˜")
        
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_performance(self, performance: Dict):
        """Affiche les performances"""
        total_trades = performance.get('total_trades', 0)
        win_rate = performance.get('win_rate', 0)
        profit_factor = performance.get('profit_factor', 0)
        sharpe = performance.get('sharpe_ratio', 0)
        drawdown = performance.get('max_drawdown', 0)
        
        print("Ã¢â€¢â€˜ PERFORMANCE")
        print("Ã¢â€¢â€˜" + "-" * 78)
        print(f"Ã¢â€¢â€˜ Total Trades:    {total_trades:<10}                                                Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Win Rate:        {win_rate:<10.1%}                                                Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Profit Factor:   {profit_factor:<10.2f}                                                Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Sharpe Ratio:    {sharpe:<10.2f}                                                Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Max Drawdown:    {drawdown:<10.2%}                                                Ã¢â€¢â€˜")
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_recent_trades(self, trades: List[Dict]):
        """Affiche les derniers trades"""
        print("Ã¢â€¢â€˜ RECENT TRADES")
        print("Ã¢â€¢â€˜" + "-" * 78)
        
        if trades:
            print("Ã¢â€¢â€˜ Time     Symbol       Side    P&L          %                                Ã¢â€¢â€˜")
            print("Ã¢â€¢â€˜" + "-" * 78)
            
            for trade in trades[-5:]:  # 5 derniers trades
                time = trade.get('close_time', datetime.now()).strftime('%H:%M')
                symbol = trade.get('symbol', '')[:10]
                side = trade.get('side', '')
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                
                pnl_symbol = "+" if pnl >= 0 else ""
                
                print(f"Ã¢â€¢â€˜ {time}   {symbol:<12} {side:<7} ${pnl:>{pnl_symbol}9,.2f}   {pnl_symbol}{pnl_pct:.2%}" + " " * 26 + "Ã¢â€¢â€˜")
        else:
            print("Ã¢â€¢â€˜ Aucun trade rÃƒÂ©cent                                                           Ã¢â€¢â€˜")
        
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_health(self, health: Dict):
        """Affiche la santÃƒÂ© du systÃƒÂ¨me"""
        overall = health.get('overall', 'unknown')
        api_status = health.get('api_status', 'unknown')
        memory_usage = health.get('memory_usage_pct', 0)
        cpu_usage = health.get('cpu_usage_pct', 0)
        
        # IcÃƒÂ´ne selon statut
        status_icons = {
            'healthy': 'Ã°Å¸Å¸Â¢',
            'degraded': 'Ã°Å¸Å¸Â¡',
            'unhealthy': 'Ã°Å¸â€Â´',
            'unknown': 'Ã¢Å¡Âª'
        }
        
        overall_icon = status_icons.get(overall, 'Ã¢Å¡Âª')
        api_icon = status_icons.get(api_status, 'Ã¢Å¡Âª')
        
        print("Ã¢â€¢â€˜ SYSTEM HEALTH")
        print("Ã¢â€¢â€˜" + "-" * 78)
        print(f"Ã¢â€¢â€˜ Overall:     {overall_icon} {overall.upper():<20}                                    Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ API:         {api_icon} {api_status.upper():<20}                                    Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ Memory:      {memory_usage:.1f}%                                                         Ã¢â€¢â€˜")
        print(f"Ã¢â€¢â€˜ CPU:         {cpu_usage:.1f}%                                                         Ã¢â€¢â€˜")
        print("Ã¢â€¢Â " + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â£")
    
    def _print_footer(self):
        """Affiche le pied de page"""
        print("Ã¢â€¢â€˜" + " " * 20 + "Press Ctrl+C to stop" + " " * 38 + "Ã¢â€¢â€˜")
        print("Ã¢â€¢Å¡" + "Ã¢â€¢Â" * 78 + "Ã¢â€¢Â")
    
    def display_simple_summary(self, data: Dict):
        """
        Affiche un rÃƒÂ©sumÃƒÂ© simplifiÃƒÂ© (1 ligne)
        
        Args:
            data: DonnÃƒÂ©es
        """
        portfolio = data.get('portfolio', {})
        performance = data.get('performance', {})
        
        capital = portfolio.get('current_capital', 0)
        pnl_pct = portfolio.get('total_pnl_pct', 0)
        positions = data.get('positions', {}).get('open_count', 0)
        win_rate = performance.get('win_rate', 0)
        drawdown = performance.get('max_drawdown', 0)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        pnl_symbol = "+" if pnl_pct >= 0 else ""
        
        print(f"[{timestamp}] Capital: ${capital:,.0f} ({pnl_symbol}{pnl_pct:.2%}) | "
              f"Positions: {positions} | Win Rate: {win_rate:.1%} | DD: {drawdown:.2%}")


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Dashboard"""
    import time
    import random
    
    dashboard = Dashboard()
    
    # DonnÃƒÂ©es de test
    test_data = {
        'status': {
            'is_running': True,
            'mode': 'live',
            'uptime_hours': 12.5
        },
        'portfolio': {
            'initial_capital': 10000,
            'current_capital': 10500,
            'total_pnl': 500,
            'total_pnl_pct': 0.05,
            'unrealized_pnl': 50
        },
        'positions': {
            'open_count': 3,
            'max_positions': 20,
            'exposure_pct': 0.45,
            'list': [
                {'symbol': 'BTCUSDC', 'side': 'BUY', 'size': 0.1, 'entry_price': 50000, 'current_price': 50500, 'unrealized_pnl_pct': 0.01},
                {'symbol': 'ETHUSDC', 'side': 'BUY', 'size': 1.0, 'entry_price': 3000, 'current_price': 3050, 'unrealized_pnl_pct': 0.0167},
                {'symbol': 'BNBUSDC', 'side': 'SELL', 'size': 10, 'entry_price': 400, 'current_price': 395, 'unrealized_pnl_pct': 0.0125}
            ]
        },
        'performance': {
            'total_trades': 150,
            'win_rate': 0.68,
            'profit_factor': 2.3,
            'sharpe_ratio': 2.1,
            'max_drawdown': 0.035
        },
        'recent_trades': [
            {'close_time': datetime.now(), 'symbol': 'BTCUSDC', 'side': 'BUY', 'pnl': 150, 'pnl_pct': 0.03},
            {'close_time': datetime.now(), 'symbol': 'ETHUSDC', 'side': 'SELL', 'pnl': -50, 'pnl_pct': -0.015},
            {'close_time': datetime.now(), 'symbol': 'ADAUSDC', 'side': 'BUY', 'pnl': 75, 'pnl_pct': 0.02}
        ],
        'health': {
            'overall': 'healthy',
            'api_status': 'healthy',
            'memory_usage_pct': 45.2,
            'cpu_usage_pct': 23.8
        }
    }
    
    print("Test Dashboard - Appuyez sur Ctrl+C pour arrÃƒÂªter\n")
    time.sleep(2)
    
    try:
        while True:
            # Simuler des changements
            test_data['portfolio']['current_capital'] += random.uniform(-50, 100)
            test_data['portfolio']['total_pnl'] = test_data['portfolio']['current_capital'] - 10000
            test_data['portfolio']['total_pnl_pct'] = test_data['portfolio']['total_pnl'] / 10000
            
            # Afficher
            dashboard.display(test_data)
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nDashboard arrÃƒÂªtÃƒÂ©")
