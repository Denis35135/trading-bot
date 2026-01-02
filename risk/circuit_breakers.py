"""
Circuit Breakers
Disjoncteurs d'urgence pour arreter le trading en cas de conditions extremes
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class BreakLevel(Enum):
    """Niveaux de gravite des circuit breakers"""
    WARNING = 1
    PAUSE = 2
    HALT = 3
    EMERGENCY = 4


class CircuitBreaker:
    """
    Systeme de disjoncteurs pour protection du capital
    
    Declencheurs:
    - Drawdown maximum depasse
    - Pertes journalieres excessives
    - Serie de trades perdants
    - Volatilite de marche extreme
    - Erreurs systeme repetees
    - Liquidite insuffisante
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le circuit breaker
        
        Args:
            config: Configuration
        """
        default_config = {
            'max_daily_loss_pct': 0.05,  # 5% perte max/jour
            'max_drawdown_pct': 0.08,  # 8% drawdown max
            'max_consecutive_losses': 5,
            'min_win_rate_threshold': 0.35,  # Arret si <35% win rate
            'max_position_loss_pct': 0.03,  # 3% perte max par position
            'cooldown_period': 3600,  # 1h de pause apres declenchement
            'volatility_spike_threshold': 3.0,  # 3x volatilite normale
            'max_errors_per_hour': 10,
            'min_liquidity_threshold': 100000  # 100k$ minimum
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.is_active = True
        self.breakers_triggered = []
        self.last_trigger_time = None
        self.cooldown_until = None
        
        # Compteurs
        self.consecutive_losses = 0
        self.daily_losses = 0
        self.errors_count = 0
        self.error_timestamps = []
        
        # Stats
        self.stats = {
            'total_triggers': 0,
            'triggers_by_type': {},
            'false_positives': 0,
            'prevented_losses': 0
        }
        
        logger.info("" Circuit Breaker initialise")
        logger.info(f"   Max daily loss: {self.config['max_daily_loss_pct']:.1%}")
        logger.info(f"   Max drawdown: {self.config['max_drawdown_pct']:.1%}")
    
    def check_conditions(
        self,
        portfolio_value: float,
        daily_pnl: float,
        drawdown: float,
        recent_trades: List[Dict]
    ) -> Dict:
        """
        Verifie toutes les conditions de declenchement
        
        Args:
            portfolio_value: Valeur actuelle du portfolio
            daily_pnl: P&L du jour
            drawdown: Drawdown actuel
            recent_trades: Trades recents
            
        Returns:
            Dict avec statut et actions
        """
        # Verifier si en cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds
            return {
                'status': 'cooldown',
                'can_trade': False,
                'reason': f'En pause (cooldown: {remaining}s restants)',
                'level': BreakLevel.PAUSE
            }
        
        # Liste des conditions declenchees
        triggered = []
        max_level = BreakLevel.WARNING
        
        # 1. Verifier perte journaliere
        daily_loss_check = self._check_daily_loss(portfolio_value, daily_pnl)
        if daily_loss_check['triggered']:
            triggered.append(daily_loss_check)
            max_level = max(max_level, daily_loss_check['level'])
        
        # 2. Verifier drawdown
        drawdown_check = self._check_drawdown(drawdown)
        if drawdown_check['triggered']:
            triggered.append(drawdown_check)
            max_level = max(max_level, drawdown_check['level'])
        
        # 3. Verifier serie de pertes
        losing_streak_check = self._check_losing_streak(recent_trades)
        if losing_streak_check['triggered']:
            triggered.append(losing_streak_check)
            max_level = max(max_level, losing_streak_check['level'])
        
        # 4. Verifier win rate
        win_rate_check = self._check_win_rate(recent_trades)
        if win_rate_check['triggered']:
            triggered.append(win_rate_check)
            max_level = max(max_level, win_rate_check['level'])
        
        # 5. Verifier erreurs systeme
        error_check = self._check_system_errors()
        if error_check['triggered']:
            triggered.append(error_check)
            max_level = max(max_level, error_check['level'])
        
        # Decision finale
        if triggered:
            return self._handle_triggers(triggered, max_level)
        
        return {
            'status': 'ok',
            'can_trade': True,
            'reason': 'Toutes les conditions sont normales',
            'level': BreakLevel.WARNING
        }
    
    def _check_daily_loss(self, portfolio_value: float, daily_pnl: float) -> Dict:
        """Verifie la perte journaliere"""
        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        
        if daily_loss_pct >= self.config['max_daily_loss_pct']:
            return {
                'triggered': True,
                'type': 'daily_loss',
                'level': BreakLevel.HALT,
                'message': f'Perte journaliere excessive: {daily_loss_pct:.2%}',
                'value': daily_loss_pct
            }
        elif daily_loss_pct >= self.config['max_daily_loss_pct'] * 0.8:
            return {
                'triggered': True,
                'type': 'daily_loss_warning',
                'level': BreakLevel.WARNING,
                'message': f'Perte journaliere elevee: {daily_loss_pct:.2%}',
                'value': daily_loss_pct
            }
        
        return {'triggered': False}
    
    def _check_drawdown(self, drawdown: float) -> Dict:
        """Verifie le drawdown"""
        if drawdown >= self.config['max_drawdown_pct']:
            return {
                'triggered': True,
                'type': 'max_drawdown',
                'level': BreakLevel.EMERGENCY,
                'message': f'Drawdown maximum atteint: {drawdown:.2%}',
                'value': drawdown
            }
        elif drawdown >= self.config['max_drawdown_pct'] * 0.75:
            return {
                'triggered': True,
                'type': 'drawdown_warning',
                'level': BreakLevel.PAUSE,
                'message': f'Drawdown eleve: {drawdown:.2%}',
                'value': drawdown
            }
        
        return {'triggered': False}
    
    def _check_losing_streak(self, recent_trades: List[Dict]) -> Dict:
        """Verifie la serie de trades perdants"""
        if not recent_trades:
            return {'triggered': False}
        
        # Compter les pertes consecutives recentes
        consecutive_losses = 0
        for trade in reversed(recent_trades[-20:]):  # 20 derniers trades
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        self.consecutive_losses = consecutive_losses
        
        if consecutive_losses >= self.config['max_consecutive_losses']:
            return {
                'triggered': True,
                'type': 'losing_streak',
                'level': BreakLevel.PAUSE,
                'message': f'{consecutive_losses} pertes consecutives',
                'value': consecutive_losses
            }
        
        return {'triggered': False}
    
    def _check_win_rate(self, recent_trades: List[Dict]) -> Dict:
        """Verifie le win rate recent"""
        if len(recent_trades) < 20:
            return {'triggered': False}
        
        # Calculer win rate sur les 20 derniers trades
        recent = recent_trades[-20:]
        wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
        win_rate = wins / len(recent)
        
        if win_rate < self.config['min_win_rate_threshold']:
            return {
                'triggered': True,
                'type': 'low_win_rate',
                'level': BreakLevel.PAUSE,
                'message': f'Win rate trop faible: {win_rate:.1%}',
                'value': win_rate
            }
        
        return {'triggered': False}
    
    def _check_system_errors(self) -> Dict:
        """Verifie les erreurs systeme"""
        # Nettoyer les anciennes erreurs (>1h)
        cutoff = datetime.now() - timedelta(hours=1)
        self.error_timestamps = [t for t in self.error_timestamps if t > cutoff]
        
        if len(self.error_timestamps) >= self.config['max_errors_per_hour']:
            return {
                'triggered': True,
                'type': 'system_errors',
                'level': BreakLevel.HALT,
                'message': f'{len(self.error_timestamps)} erreurs en 1h',
                'value': len(self.error_timestamps)
            }
        
        return {'triggered': False}
    
    def _handle_triggers(self, triggered: List[Dict], max_level: BreakLevel) -> Dict:
        """
        Gere les declenchements
        
        Args:
            triggered: Liste des conditions declenchees
            max_level: Niveau maximum de gravite
            
        Returns:
            Dict avec actions  prendre
        """
        self.breakers_triggered = triggered
        self.last_trigger_time = datetime.now()
        self.stats['total_triggers'] += 1
        
        # Compter par type
        for trigger in triggered:
            trigger_type = trigger['type']
            self.stats['triggers_by_type'][trigger_type] = \
                self.stats['triggers_by_type'].get(trigger_type, 0) + 1
        
        # Log detaille
        logger.warning(" CIRCUIT BREAKER D"CLENCH"!")
        logger.warning(f"   Niveau: {max_level.name}")
        for trigger in triggered:
            logger.warning(f"   - {trigger['message']}")
        
        # Determiner les actions
        if max_level == BreakLevel.EMERGENCY:
            # ARRT D'URGENCE TOTAL
            self.is_active = False
            self.cooldown_until = datetime.now() + timedelta(days=1)  # 24h
            
            return {
                'status': 'emergency',
                'can_trade': False,
                'must_close_all': True,
                'reason': 'ARRT D\'URGENCE - Conditions critiques',
                'level': max_level,
                'triggers': triggered,
                'cooldown_hours': 24
            }
        
        elif max_level == BreakLevel.HALT:
            # ARRT COMPLET
            self.cooldown_until = datetime.now() + timedelta(hours=4)
            
            return {
                'status': 'halt',
                'can_trade': False,
                'must_close_all': True,
                'reason': 'ARRT - Fermeture de toutes les positions',
                'level': max_level,
                'triggers': triggered,
                'cooldown_hours': 4
            }
        
        elif max_level == BreakLevel.PAUSE:
            # PAUSE
            self.cooldown_until = datetime.now() + timedelta(
                seconds=self.config['cooldown_period']
            )
            
            return {
                'status': 'pause',
                'can_trade': False,
                'must_close_all': False,
                'reduce_positions': True,
                'reason': 'PAUSE - Reduction des positions',
                'level': max_level,
                'triggers': triggered,
                'cooldown_hours': self.config['cooldown_period'] / 3600
            }
        
        else:  # WARNING
            return {
                'status': 'warning',
                'can_trade': True,
                'reduce_risk': True,
                'reason': 'ALERTE - Reduire l\'exposition',
                'level': max_level,
                'triggers': triggered
            }
    
    def report_trade(self, trade_result: Dict):
        """
        Enregistre le resultat d'un trade
        
        Args:
            trade_result: Dict avec resultat du trade
        """
        pnl = trade_result.get('pnl', 0)
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_losses += abs(pnl)
        else:
            self.consecutive_losses = 0
    
    def report_error(self, error: Exception):
        """
        Enregistre une erreur systeme
        
        Args:
            error: L'exception
        """
        self.error_timestamps.append(datetime.now())
        self.errors_count += 1
        logger.error(f"Erreur enregistree par Circuit Breaker: {error}")
    
    def reset_daily_counters(self):
        """Reinitialise les compteurs journaliers"""
        self.daily_losses = 0
        logger.info(""" Compteurs journaliers reinitialises")
    
    def force_reset(self):
        """Force la reinitialisation complete"""
        self.is_active = True
        self.breakers_triggered = []
        self.cooldown_until = None
        self.consecutive_losses = 0
        self.daily_losses = 0
        self.error_timestamps = []
        
        logger.warning(" Circuit Breaker FORCE RESET")
    
    def get_status(self) -> Dict:
        """
        Retourne le statut actuel
        
        Returns:
            Dict avec statut
        """
        status = {
            'is_active': self.is_active,
            'in_cooldown': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'consecutive_losses': self.consecutive_losses,
            'daily_losses': self.daily_losses,
            'errors_last_hour': len(self.error_timestamps),
            'last_trigger': self.last_trigger_time,
            'active_triggers': len(self.breakers_triggered)
        }
        
        if self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds
            status['cooldown_remaining_seconds'] = max(0, remaining)
        
        return status
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return self.stats.copy()


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Circuit Breaker"""
    
    breaker = CircuitBreaker()
    
    print("Test Circuit Breaker")
    print("=" * 50)
    
    # Simuler des trades perdants
    recent_trades = [
        {'pnl': -50, 'symbol': 'BTCUSDC'},
        {'pnl': -30, 'symbol': 'ETHUSDC'},
        {'pnl': -40, 'symbol': 'BNBUSDC'},
        {'pnl': -25, 'symbol': 'ADAUSDC'},
        {'pnl': -35, 'symbol': 'DOGEUSDC'},
        {'pnl': -20, 'symbol': 'XRPUSDC'}  # 6 pertes consecutives
    ]
    
    # Test 1: Serie de pertes
    print("\n1. Test serie de pertes:")
    result = breaker.check_conditions(
        portfolio_value=10000,
        daily_pnl=-200,
        drawdown=0.03,
        recent_trades=recent_trades
    )
    print(f"   Status: {result['status']}")
    print(f"   Can trade: {result['can_trade']}")
    print(f"   Reason: {result['reason']}")
    
    # Test 2: Perte journaliere excessive
    print("\n2. Test perte journaliere:")
    result = breaker.check_conditions(
        portfolio_value=10000,
        daily_pnl=-600,  # 6% de perte
        drawdown=0.06,
        recent_trades=recent_trades
    )
    print(f"   Status: {result['status']}")
    print(f"   Level: {result['level'].name}")
    print(f"   Must close all: {result.get('must_close_all', False)}")
    
    # Test 3: Drawdown maximum
    print("\n3. Test drawdown maximum:")
    result = breaker.check_conditions(
        portfolio_value=10000,
        daily_pnl=-300,
        drawdown=0.09,  # 9% drawdown (> limite)
        recent_trades=recent_trades
    )
    print(f"   Status: {result['status']}")
    print(f"   Level: {result['level'].name}")
    
    # Statut
    print("\n4. Statut actuel:")
    status = breaker.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Stats
    print("\n5. Statistiques:")
    stats = breaker.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
