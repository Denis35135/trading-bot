"""
Portfolio Manager
GÃƒÂ¨re le portfolio et l'allocation du capital entre positions
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Gestionnaire de portfolio
    
    ResponsabilitÃƒÂ©s:
    - Allocation du capital
    - Suivi des positions ouvertes
    - Calcul de l'exposition totale
    - Diversification
    - RÃƒÂ©ÃƒÂ©quilibrage
    """
    
    def __init__(self, initial_capital: float, config: Dict = None):
        """
        Initialise le portfolio manager
        
        Args:
            initial_capital: Capital initial
            config: Configuration
        """
        default_config = {
            'max_positions': 20,
            'max_position_size_pct': 0.25,  # 25% max par position
            'max_total_exposure_pct': 0.95,  # 95% du capital max
            'min_position_size': 50,  # 50$ minimum
            'reserve_cash_pct': 0.05,  # 5% de rÃƒÂ©serve
            'max_correlation_exposure': 0.5  # Max 50% dans actifs corrÃƒÂ©lÃƒÂ©s
        }
        
        if config:
            # Gestion objet Config ou dict
if hasattr(config, '__dict__'):
    default_config.update(vars(config))
elif isinstance(config, dict):
    default_config.update(config)
else:
    default_config.update(config if isinstance(config, dict) else {})
        
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        
        # Positions
        self.positions = {}  # {symbol: position_data}
        self.closed_positions = []
        
        # Allocation par stratÃƒÂ©gie
        self.strategy_allocations = {
            'scalping': 0.40,
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'pattern': 0.10,
            'ml': 0.05
        }
        
        # Stats
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'total_loss': 0,
            'max_open_positions': 0
        }
        
        logger.info("Ã°Å¸â€™Â¼ Portfolio Manager initialisÃƒÂ©")
        logger.info(f"   Capital initial: ${initial_capital:,.0f}")
        logger.info(f"   Max positions: {self.config['max_positions']}")
    
    def can_open_position(self, symbol: str, required_capital: float) -> Tuple[bool, str]:
        """
        VÃƒÂ©rifie si une position peut ÃƒÂªtre ouverte
        
        Args:
            symbol: Le symbole
            required_capital: Capital requis
            
        Returns:
            Tuple (can_open, reason)
        """
        # 1. VÃƒÂ©rifier si position existe dÃƒÂ©jÃƒÂ 
        if symbol in self.positions:
            return False, "Position dÃƒÂ©jÃƒÂ  ouverte sur ce symbole"
        
        # 2. VÃƒÂ©rifier nombre max de positions
        if len(self.positions) >= self.config['max_positions']:
            return False, f"Nombre max de positions atteint ({self.config['max_positions']})"
        
        # 3. VÃƒÂ©rifier capital disponible
        if required_capital > self.available_capital:
            return False, f"Capital insuffisant (requis: ${required_capital:.0f}, dispo: ${self.available_capital:.0f})"
        
        # 4. VÃƒÂ©rifier taille minimum
        if required_capital < self.config['min_position_size']:
            return False, f"Taille position trop petite (min: ${self.config['min_position_size']})"
        
        # 5. VÃƒÂ©rifier taille maximum par position
        max_size = self.current_capital * self.config['max_position_size_pct']
        if required_capital > max_size:
            return False, f"Taille position trop grande (max: ${max_size:.0f})"
        
        # 6. VÃƒÂ©rifier exposition totale
        total_exposure = self.get_total_exposure() + required_capital
        max_exposure = self.current_capital * self.config['max_total_exposure_pct']
        
        if total_exposure > max_exposure:
            return False, f"Exposition totale trop ÃƒÂ©levÃƒÂ©e ({total_exposure/self.current_capital:.1%})"
        
        return True, "OK"
    
    def open_position(self, position_data: Dict) -> bool:
        """
        Ouvre une nouvelle position
        
        Args:
            position_data: DonnÃƒÂ©es de la position
            
        Returns:
            True si succÃƒÂ¨s
        """
        try:
            symbol = position_data['symbol']
            size = position_data['size']
            entry_price = position_data['entry_price']
            capital_used = size * entry_price
            
            # VÃƒÂ©rifier si possible
            can_open, reason = self.can_open_position(symbol, capital_used)
            if not can_open:
                logger.warning(f"Impossible d'ouvrir position {symbol}: {reason}")
                return False
            
            # CrÃƒÂ©er la position
            position = {
                'symbol': symbol,
                'side': position_data['side'],
                'size': size,
                'entry_price': entry_price,
                'capital_used': capital_used,
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profit'),
                'strategy': position_data.get('strategy', 'unknown'),
                'open_time': datetime.now(),
                'current_price': entry_price,
                'unrealized_pnl': 0
            }
            
            self.positions[symbol] = position
            self.available_capital -= capital_used
            
            # Mettre ÃƒÂ  jour stats
            if len(self.positions) > self.stats['max_open_positions']:
                self.stats['max_open_positions'] = len(self.positions)
            
            logger.info(f"Ã¢Å“â€¦ Position ouverte: {symbol}")
            logger.info(f"   Side: {position['side']}")
            logger.info(f"   Size: {size:.4f} @ ${entry_price:.2f}")
            logger.info(f"   Capital: ${capital_used:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur ouverture position: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """
        Ferme une position
        
        Args:
            symbol: Le symbole
            exit_price: Prix de sortie
            
        Returns:
            Dict avec rÃƒÂ©sultat ou None
        """
        if symbol not in self.positions:
            logger.warning(f"Position {symbol} non trouvÃƒÂ©e")
            return None
        
        try:
            position = self.positions[symbol]
            
            # Calculer le P&L
            size = position['size']
            entry_price = position['entry_price']
            
            if position['side'] == 'BUY':
                pnl = (exit_price - entry_price) * size
            else:  # SELL
                pnl = (entry_price - exit_price) * size
            
            pnl_pct = pnl / position['capital_used']
            
            # LibÃƒÂ©rer le capital
            self.available_capital += position['capital_used'] + pnl
            self.current_capital += pnl
            
            # Enregistrer le rÃƒÂ©sultat
            result = {
                'symbol': symbol,
                'side': position['side'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'capital_used': position['capital_used'],
                'strategy': position['strategy'],
                'open_time': position['open_time'],
                'close_time': datetime.now(),
                'holding_time': (datetime.now() - position['open_time']).seconds / 60  # minutes
            }
            
            self.closed_positions.append(result)
            
            # Mettre ÃƒÂ  jour les stats
            self.stats['total_trades'] += 1
            if pnl > 0:
                self.stats['winning_trades'] += 1
                self.stats['total_profit'] += pnl
            else:
                self.stats['losing_trades'] += 1
                self.stats['total_loss'] += abs(pnl)
            
            # Supprimer la position
            del self.positions[symbol]
            
            logger.info(f"Ã°Å¸â€â€™ Position fermÃƒÂ©e: {symbol}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2%})")
            logger.info(f"   Nouveau capital: ${self.current_capital:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur fermeture position: {e}")
            return None
    
    def update_position_price(self, symbol: str, current_price: float):
        """
        Met ÃƒÂ  jour le prix actuel d'une position
        
        Args:
            symbol: Le symbole
            current_price: Prix actuel
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position['current_price'] = current_price
        
        # Calculer le P&L non rÃƒÂ©alisÃƒÂ©
        size = position['size']
        entry_price = position['entry_price']
        
        if position['side'] == 'BUY':
            position['unrealized_pnl'] = (current_price - entry_price) * size
        else:
            position['unrealized_pnl'] = (entry_price - current_price) * size
        
        position['unrealized_pnl_pct'] = position['unrealized_pnl'] / position['capital_used']
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Retourne une position"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """Retourne toutes les positions"""
        return self.positions.copy()
    
    def get_total_exposure(self) -> float:
        """Retourne l'exposition totale en capital"""
        return sum(p['capital_used'] for p in self.positions.values())
    
    def get_total_unrealized_pnl(self) -> float:
        """Retourne le P&L non rÃƒÂ©alisÃƒÂ© total"""
        return sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
    
    def get_positions_by_strategy(self, strategy: str) -> List[Dict]:
        """Retourne les positions d'une stratÃƒÂ©gie"""
        return [p for p in self.positions.values() if p.get('strategy') == strategy]
    
    def calculate_strategy_capital(self, strategy: str) -> float:
        """
        Calcule le capital disponible pour une stratÃƒÂ©gie
        
        Args:
            strategy: Nom de la stratÃƒÂ©gie
            
        Returns:
            Capital disponible
        """
        allocation = self.strategy_allocations.get(strategy, 0)
        return self.current_capital * allocation
    
    def get_portfolio_value(self) -> float:
        """Calcule la valeur totale du portfolio"""
        return self.current_capital + self.get_total_unrealized_pnl()
    
    def get_portfolio_summary(self) -> Dict:
        """
        Retourne un rÃƒÂ©sumÃƒÂ© du portfolio
        
        Returns:
            Dict avec mÃƒÂ©triques
        """
        total_pnl = self.current_capital - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        exposure_pct = self.get_total_exposure() / self.current_capital if self.current_capital > 0 else 0
        
        unrealized_pnl = self.get_total_unrealized_pnl()
        portfolio_value = self.get_portfolio_value()
        
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades']
                   if self.stats['total_trades'] > 0 else 0)
        
        profit_factor = (self.stats['total_profit'] / abs(self.stats['total_loss'])
                        if self.stats['total_loss'] != 0 else 0)
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'available_capital': self.available_capital,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'unrealized_pnl': unrealized_pnl,
            'open_positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'exposure_pct': exposure_pct,
            'total_trades': self.stats['total_trades'],
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return self.stats.copy()


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Portfolio Manager"""
    
    manager = PortfolioManager(initial_capital=10000)
    
    print("Test Portfolio Manager")
    print("=" * 50)
    
    # Test 1: Ouvrir une position
    print("\n1. Ouverture de position:")
    position1 = {
        'symbol': 'BTCUSDC',
        'side': 'BUY',
        'size': 0.1,
        'entry_price': 50000,
        'stop_loss': 49000,
        'take_profit': 52000,
        'strategy': 'momentum'
    }
    
    success = manager.open_position(position1)
    print(f"   SuccÃƒÂ¨s: {success}")
    
    # Test 2: Ouvrir plusieurs positions
    print("\n2. Ouverture de positions supplÃƒÂ©mentaires:")
    for symbol, price in [('ETHUSDC', 3000), ('BNBUSDC', 400)]:
        position = {
            'symbol': symbol,
            'side': 'BUY',
            'size': 1.0,
            'entry_price': price,
            'strategy': 'scalping'
        }
        manager.open_position(position)
    
    # Test 3: RÃƒÂ©sumÃƒÂ© du portfolio
    print("\n3. RÃƒÂ©sumÃƒÂ© du portfolio:")
    summary = manager.get_portfolio_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            if 'pct' in key or 'rate' in key or 'factor' in key:
                print(f"   {key}: {value:.2%}")
            else:
                print(f"   {key}: ${value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Test 4: Fermer une position
    print("\n4. Fermeture de position:")
    result = manager.close_position('BTCUSDC', 51000)
    if result:
        print(f"   P&L: ${result['pnl']:.2f} ({result['pnl_pct']:+.2%})")
    
    # Test 5: RÃƒÂ©sumÃƒÂ© final
    print("\n5. RÃƒÂ©sumÃƒÂ© final:")
    final_summary = manager.get_portfolio_summary()
    print(f"   Capital: ${final_summary['current_capital']:.2f}")
    print(f"   P&L Total: ${final_summary['total_pnl']:.2f} ({final_summary['total_pnl_pct']:+.2%})")
    print(f"   Positions ouvertes: {final_summary['open_positions']}")