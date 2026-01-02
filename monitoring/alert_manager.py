"""
Alert Manager
Gere les alertes et notifications du systeme
"""

import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveaux d'alerte"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertManager:
    """
    Gestionnaire d'alertes centralise
    
    Fonctionnalites:
    - Alertes multi-niveaux
    - Rate limiting (pas de spam)
    - Historique des alertes
    - Notifications multiples canaux
    - Groupement d'alertes similaires
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'alert manager
        
        Args:
            config: Configuration
        """
        default_config = {
            'rate_limit_seconds': 300,  # 5 min entre alertes similaires
            'max_history': 1000,
            'enable_console': True,
            'enable_file': True,
            'enable_email': False,  #  activer avec config SMTP
            'enable_telegram': False  #  activer avec bot token
        }
        
        if config:
            # Gestion objet Config ou dict
if hasattr(config, '__dict__'):
    default_config.update(vars(config))
elif isinstance(config, dict):
    default_config.update(config)
else:
    default_config.update(config if isinstance(config, dict) else {})
        
        self.config = default_config
        
        # Historique
        self.alert_history = deque(maxlen=self.config['max_history'])
        self.last_alert_times = {}  # {alert_key: timestamp}
        
        # Handlers personnalises
        self.custom_handlers = []
        
        # Stats
        self.stats = {
            'total_alerts': 0,
            'alerts_by_level': {
                AlertLevel.INFO: 0,
                AlertLevel.WARNING: 0,
                AlertLevel.ERROR: 0,
                AlertLevel.CRITICAL: 0
            },
            'suppressed_alerts': 0
        }
        
        logger.info(""" Alert Manager initialise")
    
    def send_alert(
    """
    self,
    """
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        category: str = 'general',
        data: Dict = None
    ):
        """
        Envoie une alerte
        
        Args:
            title: Titre de l'alerte
            message: Message detaille
            level: Niveau d'alerte
            category: Categorie (trading, system, risk, etc.)
            data: Donnees supplementaires
        """
        # Creer une cle pour rate limiting
        alert_key = f"{category}_{title}"
        
        # Verifier rate limiting
        if self._should_suppress(alert_key, level):
            self.stats['suppressed_alerts'] += 1
            logger.debug(f"Alerte supprimee (rate limit): {title}")
            return
        
        # Creer l'alerte
        alert = {
            'timestamp': datetime.now(),
            'title': title,
            'message': message,
            'level': level,
            'category': category,
            'data': data or {}
        }
        
        # Ajouter  l'historique
        self.alert_history.append(alert)
        
        # Mettre  jour les stats
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_level'][level] += 1
        
        # Mettre  jour le rate limiting
        self.last_alert_times[alert_key] = datetime.now()
        
        # Envoyer via les differents canaux
        self._send_to_handlers(alert)
        
        # Log selon le niveau
        log_message = f"[{category.upper()}] {title}: {message}"
        
        if level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif level == AlertLevel.ERROR:
            logger.error(log_message)
        elif level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _should_suppress(self, alert_key: str, level: AlertLevel) -> bool:
        """
        Verifie si l'alerte doit etre supprimee (rate limiting)
        
        Args:
            alert_key: Cle de l'alerte
            level: Niveau
            
        Returns:
            True si doit etre supprimee
        """
        # Les alertes CRITICAL ne sont jamais supprimees
        if level == AlertLevel.CRITICAL:
            return False
        
        if alert_key not in self.last_alert_times:
            return False
        
        last_time = self.last_alert_times[alert_key]
        elapsed = (datetime.now() - last_time).total_seconds()
        
        return elapsed < self.config['rate_limit_seconds']
    
    def _send_to_handlers(self, alert: Dict):
        """Envoie l'alerte  tous les handlers actives"""
        # Console (toujours active)
        if self.config['enable_console']:
            self._send_to_console(alert)
        
        # Handlers personnalises
        for handler in self.custom_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Erreur handler personnalise: {e}")
    
    def _send_to_console(self, alert: Dict):
        """Affiche l'alerte en console avec formatage"""
        level = alert['level']
        
        # Emoji selon le niveau
        emoji_map = {
            AlertLevel.INFO: "',
            AlertLevel.WARNING: '',
            AlertLevel.ERROR: ''',
            AlertLevel.CRITICAL: ''
        }
        
        emoji = emoji_map.get(level, '"')
        timestamp = alert['timestamp'].strftime('%H:%M:%S')
        
        print(f"\n{emoji} [{timestamp}] {alert['title']}")
        print(f"   {alert['message']}")
        
        if alert['data']:
            print(f"   Data: {alert['data']}")
    
    def add_handler(self, handler: Callable):
        """
        Ajoute un handler personnalise
        
        Args:
            handler: Fonction qui prend un dict alert en parametre
        """
        self.custom_handlers.append(handler)
        logger.info(f"Handler personnalise ajoute ({len(self.custom_handlers)} total)")
    
    def get_recent_alerts(
        self,
        count: int = 10,
        level: AlertLevel = None,
        category: str = None
    ) -> List[Dict]:
        """
        Retourne les alertes recentes
        
        Args:
            count: Nombre d'alertes
            level: Filtrer par niveau
            category: Filtrer par categorie
            
        Returns:
            Liste d'alertes
        """
        alerts = list(self.alert_history)
        
        # Filtrer par niveau
        if level:
            alerts = [a for a in alerts if a['level'] == level]
        
        # Filtrer par categorie
        if category:
            alerts = [a for a in alerts if a['category'] == category]
        
        # Retourner les N plus recentes
        return alerts[-count:]
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        Resume des alertes sur une periode
        
        Args:
            hours: Nombre d'heures
            
        Returns:
            Dict avec resume
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [a for a in self.alert_history if a['timestamp'] > cutoff]
        
        summary = {
            'total': len(recent),
            'by_level': {
                'info': sum(1 for a in recent if a['level'] == AlertLevel.INFO),
                'warning': sum(1 for a in recent if a['level'] == AlertLevel.WARNING),
                'error': sum(1 for a in recent if a['level'] == AlertLevel.ERROR),
                'critical': sum(1 for a in recent if a['level'] == AlertLevel.CRITICAL)
            },
            'by_category': {}
        }
        
        # Compter par categorie
        for alert in recent:
            cat = alert['category']
            summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1
        
        return summary
    
    def clear_history(self):
        """Efface l'historique des alertes"""
        self.alert_history.clear()
        self.last_alert_times.clear()
        logger.info("Historique des alertes efface")
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return {
            **self.stats,
            'history_size': len(self.alert_history),
            'custom_handlers': len(self.custom_handlers)
        }


# Fonctions utilitaires pour alertes courantes

def alert_trade_opened(manager: AlertManager, symbol: str, side: str, size: float, price: float):
    """Alerte pour ouverture de trade"""
    manager.send_alert(
        title="Trade Opened",
        message=f"{side} {size} {symbol} @ ${price:,.2f}",
        level=AlertLevel.INFO,
        category='trading',
        data={'symbol': symbol, 'side': side, 'size': size, 'price': price}
    )


def alert_trade_closed(manager: AlertManager, symbol: str, pnl: float, pnl_pct: float):
    """Alerte pour fermeture de trade"""
    level = AlertLevel.INFO if pnl > 0 else AlertLevel.WARNING
    
    manager.send_alert(
        title="Trade Closed",
        message=f"{symbol}: P&L ${pnl:+.2f} ({pnl_pct:+.2%})",
        level=level,
        category='trading',
        data={'symbol': symbol, 'pnl': pnl, 'pnl_pct': pnl_pct}
    )


def alert_drawdown_warning(manager: AlertManager, drawdown: float):
    """Alerte pour drawdown eleve"""
    if drawdown > 0.08:
        level = AlertLevel.CRITICAL
    elif drawdown > 0.06:
        level = AlertLevel.ERROR
    else:
        level = AlertLevel.WARNING
    
    manager.send_alert(
        title="High Drawdown",
        message=f"Drawdown actuel: {drawdown:.2%}",
        level=level,
        category='risk',
        data={'drawdown': drawdown}
    )


def alert_system_error(manager: AlertManager, error: str, details: str = None):
    """Alerte pour erreur systeme"""
    manager.send_alert(
        title="System Error",
        message=f"{error}" + (f" - {details}" if details else ""),
        level=AlertLevel.ERROR,
        category='system',
        data={'error': error, 'details': details}
    )


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de l'Alert Manager"""
    
    manager = AlertManager()
    
    print("Test Alert Manager")
    print("=" * 50)
    
    # Test 1: Alertes de differents niveaux
    print("\n1. Test alertes multi-niveaux:")
    manager.send_alert(
        "Test Info",
        "Ceci est une alerte INFO",
        AlertLevel.INFO,
        'test'
    )
    
    manager.send_alert(
        "Test Warning",
        "Ceci est une alerte WARNING",
        AlertLevel.WARNING,
        'test'
    )
    
    manager.send_alert(
        "Test Critical",
        "Ceci est une alerte CRITICAL",
        AlertLevel.CRITICAL,
        'test'
    )
    
    # Test 2: Rate limiting
    print("\n2. Test rate limiting:")
    for i in range(3):
        manager.send_alert(
            "Same Alert",
            f"Tentative {i+1}",
            AlertLevel.INFO,
            'test'
        )
    
    # Test 3: Alertes de trading
    print("\n3. Test alertes trading:")
    alert_trade_opened(manager, 'BTCUSDC', 'BUY', 0.1, 50000)
    alert_trade_closed(manager, 'BTCUSDC', 150, 0.03)
    
    # Test 4: Resume
    print("\n4. Resume des alertes:")
    summary = manager.get_alert_summary(24)
    print(f"   Total: {summary['total']}")
    print(f"   Par niveau: {summary['by_level']}")
    
    # Test 5: Stats
    print("\n5. Statistiques:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
