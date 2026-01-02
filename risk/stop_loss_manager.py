"""
Stop Loss Manager
Gere intelligemment les stop loss et trailing stops
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StopType(Enum):
    """Types de stops"""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    BREAK_EVEN = "break_even"
    TIME_BASED = "time_based"


class StopLossManager:
    """
    Gestionnaire de stop loss intelligent
    
    Fonctionnalites:
    - Stop loss fixe
    - Trailing stop
    - Stop base sur ATR
    - Break-even automatique
    - Time-based stop
    - Ajustement dynamique
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le stop loss manager
        
        Args:
            config: Configuration
        """
        default_config = {
            'default_stop_pct': 0.02,  # 2% stop par defaut
            'trailing_stop_activation': 0.015,  # Active  +1.5%
            'trailing_stop_distance': 0.01,  # 1% de trailing
            'break_even_activation': 0.01,  # Break-even  +1%
            'atr_multiplier': 2.0,  # 2x ATR pour stop
            'max_stop_distance': 0.05,  # 5% maximum
            'min_stop_distance': 0.005,  # 0.5% minimum
            'max_time_in_position': 24  # heures
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.stops = {}  # {position_id: stop_data}
        
        # Stats
        self.stats = {
            'stops_hit': 0,
            'trailing_stops_hit': 0,
            'break_even_stops_hit': 0,
            'time_stops_hit': 0,
            'avg_stop_distance': 0
        }
        
        logger.info("" Stop Loss Manager initialise")
    
    def create_stop(
        self,
        position_id: str,
        entry_price: float,
        side: str,
        stop_type: StopType = StopType.FIXED,
        custom_distance: float = None,
        atr: float = None
    ) -> Dict:
        """
        Cree un stop loss pour une position
        
        Args:
            position_id: ID de la position
            entry_price: Prix d'entree
            side: BUY ou SELL
            stop_type: Type de stop
            custom_distance: Distance custom (si None, utilise config)
            atr: ATR pour calcul (si stop ATR-based)
            
        Returns:
            Dict avec donnees du stop
        """
        # Calculer la distance du stop
        if stop_type == StopType.ATR_BASED and atr:
            stop_distance = atr * self.config['atr_multiplier']
            stop_distance_pct = stop_distance / entry_price
        elif custom_distance:
            stop_distance_pct = custom_distance
        else:
            stop_distance_pct = self.config['default_stop_pct']
        
        # Limiter la distance
        stop_distance_pct = max(
            self.config['min_stop_distance'],
            min(stop_distance_pct, self.config['max_stop_distance'])
        )
        
        # Calculer le prix du stop
        if side == 'BUY':
            stop_price = entry_price * (1 - stop_distance_pct)
        else:  # SELL
            stop_price = entry_price * (1 + stop_distance_pct)
        
        # Creer le stop
        stop_data = {
            'position_id': position_id,
            'type': stop_type,
            'side': side,
            'entry_price': entry_price,
            'initial_stop': stop_price,
            'current_stop': stop_price,
            'stop_distance_pct': stop_distance_pct,
            'highest_price': entry_price if side == 'BUY' else 0,
            'lowest_price': entry_price if side == 'SELL' else float('inf'),
            'is_break_even': False,
            'is_trailing': stop_type == StopType.TRAILING,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        
        self.stops[position_id] = stop_data
        
        logger.info(f"" Stop cree pour {position_id}")
        logger.info(f"   Type: {stop_type.value}")
        logger.info(f"   Entry: ${entry_price:.2f}")
        logger.info(f"   Stop: ${stop_price:.2f} ({stop_distance_pct:.2%})")
        
        return stop_data
    
    def update_stop(
        self,
        position_id: str,
        current_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Met  jour un stop et verifie s'il est touche
        
        Args:
            position_id: ID de la position
            current_price: Prix actuel
            
        Returns:
            Tuple (is_triggered, reason)
        """
        if position_id not in self.stops:
            return False, None
        
        stop = self.stops[position_id]
        side = stop['side']
        entry_price = stop['entry_price']
        
        # Mettre  jour les extremes
        if side == 'BUY':
            if current_price > stop['highest_price']:
                stop['highest_price'] = current_price
        else:  # SELL
            if current_price < stop['lowest_price']:
                stop['lowest_price'] = current_price
        
        # Calculer le profit actuel
        if side == 'BUY':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # 1. Verifier break-even
        if not stop['is_break_even'] and profit_pct >= self.config['break_even_activation']:
            self._activate_break_even(position_id, entry_price)
            stop = self.stops[position_id]  # Recharger apres modification
        
        # 2. Verifier trailing stop
        if stop['is_trailing'] and profit_pct >= self.config['trailing_stop_activation']:
            self._update_trailing_stop(position_id, current_price)
            stop = self.stops[position_id]  # Recharger
        
        # 3. Verifier time-based stop
        time_in_position = (datetime.now() - stop['created_at']).seconds / 3600  # heures
        if time_in_position > self.config['max_time_in_position']:
            self.stats['time_stops_hit'] += 1
            return True, f"Time stop ({time_in_position:.1f}h)"
        
        # 4. Verifier si stop touche
        triggered, reason = self._check_stop_hit(position_id, current_price)
        
        if triggered:
            self._handle_stop_hit(position_id, reason)
        
        stop['last_updated'] = datetime.now()
        
        return triggered, reason
    
    def _activate_break_even(self, position_id: str, entry_price: float):
        """Active le break-even"""
        stop = self.stops[position_id]
        
        # Mettre le stop au break-even (legerement au-dessus/en-dessous)
        if stop['side'] == 'BUY':
            stop['current_stop'] = entry_price * 1.001  # +0.1% pour fees
        else:
            stop['current_stop'] = entry_price * 0.999
        
        stop['is_break_even'] = True
        
        logger.info(f"""| Break-even active pour {position_id}")
        logger.info(f"   Nouveau stop: ${stop['current_stop']:.2f}")
    
    def _update_trailing_stop(self, position_id: str, current_price: float):
        """Met  jour le trailing stop"""
        stop = self.stops[position_id]
        distance = self.config['trailing_stop_distance']
        
        if stop['side'] == 'BUY':
            # Trailing stop suit le prix  la hausse
            new_stop = current_price * (1 - distance)
            if new_stop > stop['current_stop']:
                stop['current_stop'] = new_stop
                logger.debug(f"" Trailing stop ajuste: ${new_stop:.2f}"")
        else:  # SELL
            # Trailing stop suit le prix  la baisse
            new_stop = current_price * (1 + distance)
            if new_stop < stop['current_stop']:
                stop['current_stop'] = new_stop
                logger.debug(f""" Trailing stop ajuste: ${new_stop:.2f}")
    
    def _check_stop_hit(
    """
    self,
    """
        position_id: str,
        current_price: float
    ) -> Tuple[bool, Optional[str]]:
        """Verifie si le stop est touche"""
        stop = self.stops[position_id]
        
        if stop['side'] == 'BUY':
            if current_price <= stop['current_stop']:
                if stop['is_break_even']:
                    return True, "Break-even stop hit"
                elif stop['is_trailing']:
                    return True, "Trailing stop hit"
                else:
                    return True, "Stop loss hit"
        else:  # SELL
            if current_price >= stop['current_stop']:
                if stop['is_break_even']:
                    return True, "Break-even stop hit"
                elif stop['is_trailing']:
                    return True, "Trailing stop hit"
                else:
                    return True, "Stop loss hit"
        
        return False, None
    
    def _handle_stop_hit(self, position_id: str, reason: str):
        """Gere un stop touche"""
        self.stats['stops_hit'] += 1
        
        if 'trailing' in reason.lower():
            self.stats['trailing_stops_hit'] += 1
        elif 'break-even' in reason.lower():
            self.stats['break_even_stops_hit'] += 1
        
        logger.warning(f" Stop touche: {position_id}")
        logger.warning(f"   Raison: {reason}")
    
    def modify_stop(
    """
    self,
    """
        position_id: str,
        new_stop_price: float
    ) -> bool:
        """
        Modifie manuellement le stop
        
        Args:
            position_id: ID de la position
            new_stop_price: Nouveau prix de stop
            
        Returns:
            True si succes
        """
        if position_id not in self.stops:
            return False
        
        stop = self.stops[position_id]
        old_stop = stop['current_stop']
        stop['current_stop'] = new_stop_price
        
        logger.info(f"" Stop modifie pour {position_id}")
        logger.info(f"   Ancien: ${old_stop:.2f}")
        logger.info(f"   Nouveau: ${new_stop_price:.2f}")
        
        return True
    
    def remove_stop(self, position_id: str):
        """Retire un stop"""
        if position_id in self.stops:
            del self.stops[position_id]
            logger.info(f"""" Stop retire: {position_id}")
    
    def get_stop(self, position_id: str) -> Optional[Dict]:
        """Retourne les donnees d'un stop"""
        return self.stops.get(position_id)
    
    def get_all_stops(self) -> Dict:
        """Retourne tous les stops"""
        return self.stops.copy()
    
    def calculate_stop_price(
    """
    self,
    """
        entry_price: float,
        side: str,
        atr: float = None,
        custom_pct: float = None
    ) -> float:
        """
        Calcule un prix de stop optimal
        
        Args:
            entry_price: Prix d'entree
            side: BUY ou SELL
            atr: ATR (optionnel)
            custom_pct: Distance custom (optionnel)
            
        Returns:
            Prix de stop
        """
        if atr:
            distance_pct = (atr * self.config['atr_multiplier']) / entry_price
        elif custom_pct:
            distance_pct = custom_pct
        else:
            distance_pct = self.config['default_stop_pct']
        
        # Limiter
        distance_pct = max(
            self.config['min_stop_distance'],
            min(distance_pct, self.config['max_stop_distance'])
        )
        
        if side == 'BUY':
            return entry_price * (1 - distance_pct)
        else:
            return entry_price * (1 + distance_pct)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return self.stats.copy()


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Stop Loss Manager"""
    
    manager = StopLossManager()
    
    print("Test Stop Loss Manager")
    print("=" * 50)
    
    # Test 1: Creer un stop fixe
    print("\n1. Creation d'un stop fixe:")
    stop = manager.create_stop(
        position_id='BTC_001',
        entry_price=50000,
        side='BUY',
        stop_type=StopType.FIXED
    )
    print(f"   Stop cree  ${stop['current_stop']:.2f}")
    
    # Test 2: Simuler mouvement de prix
    print("\n2. Simulation de prix:")
    prices = [50000, 50500, 51000, 50800, 50200, 49800, 49000]
    
    for price in prices:
        triggered, reason = manager.update_stop('BTC_001', price)
        print(f"   Prix: ${price:,.0f} - Triggered: {triggered}", end="")
        if reason:
            print(f" ({reason})")
        else:
            stop = manager.get_stop('BTC_001')
            print(f" - Stop: ${stop['current_stop']:.2f}")
        
        if triggered:
            break
    
    # Test 3: Trailing stop
    print("\n3. Test trailing stop:")
    manager.create_stop(
        position_id='ETH_001',
        entry_price=3000,
        side='BUY',
        stop_type=StopType.TRAILING
    )
    
    trailing_prices = [3000, 3050, 3100, 3150, 3100, 3050]
    for price in trailing_prices:
        triggered, reason = manager.update_stop('ETH_001', price)
        stop = manager.get_stop('ETH_001')
        print(f"   Prix: ${price:,.0f} - Stop: ${stop['current_stop']:.2f} - BE: {stop['is_break_even']}")
    
    # Test 4: Stats
    print("\n4. Statistiques:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
