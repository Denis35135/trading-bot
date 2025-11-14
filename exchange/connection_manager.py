"""
Connection Manager pour The Bot
GÃƒÂ¨re la connexion ÃƒÂ  Binance avec reconnexion automatique
"""

import time
from typing import Optional, Dict, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Statut de la connexion"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ConnectionManager:
    """
    Gestionnaire de connexion avec reconnexion automatique
    
    ResponsabilitÃƒÂ©s:
    - Maintenir la connexion ÃƒÂ  Binance
    - Reconnexion automatique en cas de dÃƒÂ©connexion
    - Monitoring de la santÃƒÂ© de la connexion
    - Circuit breaker si trop d'erreurs
    """
    
    def __init__(self, exchange_client, config: Optional[Dict] = None):
        """
        Initialise le connection manager
        
        Args:
            exchange_client: Client Binance
            config: Configuration de reconnexion
        """
        self.client = exchange_client
        self.config = config or {}
        
        # ParamÃƒÂ¨tres de reconnexion
        self.max_retries = self.config.get('max_retries', 5)
        self.retry_delay = self.config.get('retry_delay', 5)  # secondes
        self.max_retry_delay = self.config.get('max_retry_delay', 60)
        self.backoff_multiplier = self.config.get('backoff_multiplier', 2)
        
        # ParamÃƒÂ¨tres de circuit breaker
        self.circuit_breaker_threshold = self.config.get('circuit_breaker_threshold', 10)
        self.circuit_breaker_timeout = self.config.get('circuit_breaker_timeout', 300)  # 5 min
        
        # Ãƒâ€°tat
        self.status = ConnectionStatus.DISCONNECTED
        self.last_connected = None
        self.connection_attempts = 0
        self.consecutive_errors = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        # Statistiques
        self.stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_reconnections': 0,
            'total_errors': 0,
            'uptime_seconds': 0
        }
        
        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info("Ã¢Å“â€¦ Connection Manager initialisÃƒÂ©")
    
    def connect(self) -> bool:
        """
        Ãƒâ€°tablit la connexion
        
        Returns:
            True si connexion rÃƒÂ©ussie
        """
        if self.circuit_breaker_active:
            if datetime.now() < self.circuit_breaker_until:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Circuit breaker actif jusqu'ÃƒÂ  {self.circuit_breaker_until}")
                return False
            else:
                logger.info("Circuit breaker rÃƒÂ©initialisÃƒÂ©")
                self.circuit_breaker_active = False
                self.consecutive_errors = 0
        
        self.status = ConnectionStatus.CONNECTING
        logger.info("Ã°Å¸â€Å’ Connexion ÃƒÂ  Binance...")
        
        try:
            # Test de connexion avec ping
            self.client.ping()
            
            # VÃƒÂ©rifier le temps du serveur
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            if time_diff > 5000:  # Plus de 5 secondes de diffÃƒÂ©rence
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â DiffÃƒÂ©rence de temps: {time_diff}ms")
            
            # Connexion rÃƒÂ©ussie
            self.status = ConnectionStatus.CONNECTED
            self.last_connected = datetime.now()
            self.connection_attempts = 0
            self.consecutive_errors = 0
            self.stats['total_connections'] += 1
            
            logger.info(f"Ã¢Å“â€¦ ConnectÃƒÂ© ÃƒÂ  Binance (ping: {time_diff}ms)")
            
            # Callback
            if self.on_connected:
                self.on_connected()
            
            return True
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur de connexion: {e}")
            self.status = ConnectionStatus.ERROR
            self.consecutive_errors += 1
            self.stats['total_errors'] += 1
            
            # VÃƒÂ©rifier circuit breaker
            if self.consecutive_errors >= self.circuit_breaker_threshold:
                self._activate_circuit_breaker()
            
            # Callback
            if self.on_error:
                self.on_error(e)
            
            return False
    
    def disconnect(self):
        """DÃƒÂ©connecte proprement"""
        logger.info("Ã°Å¸â€Å’ DÃƒÂ©connexion...")
        
        self.status = ConnectionStatus.DISCONNECTED
        self.stats['total_disconnections'] += 1
        
        # Callback
        if self.on_disconnected:
            self.on_disconnected()
        
        logger.info("Ã¢Å“â€¦ DÃƒÂ©connectÃƒÂ©")
    
    def reconnect(self) -> bool:
        """
        Tente de se reconnecter avec backoff exponentiel
        
        Returns:
            True si reconnexion rÃƒÂ©ussie
        """
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker actif, reconnexion impossible")
            return False
        
        self.status = ConnectionStatus.RECONNECTING
        delay = self.retry_delay
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Ã°Å¸â€â€ž Tentative de reconnexion {attempt}/{self.max_retries}")
            
            if self.connect():
                logger.info("Ã¢Å“â€¦ Reconnexion rÃƒÂ©ussie")
                self.stats['total_reconnections'] += 1
                return True
            
            # Attendre avant retry avec backoff exponentiel
            if attempt < self.max_retries:
                wait_time = min(delay, self.max_retry_delay)
                logger.info(f"Ã¢ÂÂ³ Attente {wait_time}s avant retry...")
                time.sleep(wait_time)
                delay *= self.backoff_multiplier
        
        logger.error("Ã¢ÂÅ’ Ãƒâ€°chec de reconnexion aprÃƒÂ¨s tous les essais")
        return False
    
    def check_connection(self) -> bool:
        """
        VÃƒÂ©rifie que la connexion est active
        
        Returns:
            True si connexion OK
        """
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            # Ping rapide pour vÃƒÂ©rifier
            self.client.ping()
            return True
            
        except Exception as e:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Connexion perdue: {e}")
            self.status = ConnectionStatus.DISCONNECTED
            self.consecutive_errors += 1
            return False
    
    def ensure_connected(self) -> bool:
        """
        S'assure que la connexion est active, reconnecte si nÃƒÂ©cessaire
        
        Returns:
            True si connectÃƒÂ©
        """
        if self.check_connection():
            return True
        
        logger.warning("Ã¢Å¡Â Ã¯Â¸Â Connexion perdue, tentative de reconnexion...")
        return self.reconnect()
    
    def _activate_circuit_breaker(self):
        """Active le circuit breaker"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
        
        logger.error(f"Ã°Å¸Å¡Â¨ CIRCUIT BREAKER ACTIVÃƒâ€° jusqu'ÃƒÂ  {self.circuit_breaker_until}")
        logger.error(f"   Raison: {self.consecutive_errors} erreurs consÃƒÂ©cutives")
    
    def reset_circuit_breaker(self):
        """RÃƒÂ©initialise le circuit breaker manuellement"""
        if self.circuit_breaker_active:
            logger.info("Ã°Å¸â€â€œ Circuit breaker rÃƒÂ©initialisÃƒÂ© manuellement")
            self.circuit_breaker_active = False
            self.circuit_breaker_until = None
            self.consecutive_errors = 0
    
    def get_status(self) -> Dict:
        """
        Retourne le statut de la connexion
        
        Returns:
            Dict avec les infos de statut
        """
        uptime = 0
        if self.status == ConnectionStatus.CONNECTED and self.last_connected:
            uptime = (datetime.now() - self.last_connected).total_seconds()
        
        return {
            'status': self.status.value,
            'connected': self.status == ConnectionStatus.CONNECTED,
            'last_connected': self.last_connected.isoformat() if self.last_connected else None,
            'uptime_seconds': uptime,
            'connection_attempts': self.connection_attempts,
            'consecutive_errors': self.consecutive_errors,
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_until': self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None,
            'stats': self.stats.copy()
        }
    
    def get_uptime(self) -> float:
        """
        Retourne l'uptime en secondes
        
        Returns:
            Secondes depuis la derniÃƒÂ¨re connexion
        """
        if self.status == ConnectionStatus.CONNECTED and self.last_connected:
            return (datetime.now() - self.last_connected).total_seconds()
        return 0
    
    def is_healthy(self) -> bool:
        """
        VÃƒÂ©rifie si la connexion est saine
        
        Returns:
            True si pas de problÃƒÂ¨mes
        """
        return (
            self.status == ConnectionStatus.CONNECTED and
            not self.circuit_breaker_active and
            self.consecutive_errors == 0
        )


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Connection Manager"""
    
    print("\n=== Test Connection Manager ===\n")
    
    # Mock du client Binance
    class MockBinanceClient:
        def __init__(self, should_fail=False):
            self.should_fail = should_fail
            self.call_count = 0
        
        def ping(self):
            self.call_count += 1
            if self.should_fail and self.call_count < 3:
                raise Exception("Connection error")
            return True
        
        def get_server_time(self):
            return int(time.time() * 1000)
    
    # Test connexion rÃƒÂ©ussie
    print("1Ã¯Â¸ÂÃ¢Æ’Â£ Test connexion rÃƒÂ©ussie:")
    client = MockBinanceClient(should_fail=False)
    manager = ConnectionManager(client, {'max_retries': 3})
    
    success = manager.connect()
    print(f"   Connexion: {'Ã¢Å“â€¦ OK' if success else 'Ã¢ÂÅ’ FAIL'}")
    
    status = manager.get_status()
    print(f"   Status: {status['status']}")
    print(f"   Connected: {status['connected']}")
    
    # Test vÃƒÂ©rification connexion
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ Test check_connection:")
    is_connected = manager.check_connection()
    print(f"   Check: {'Ã¢Å“â€¦ OK' if is_connected else 'Ã¢ÂÅ’ FAIL'}")
    
    # Test reconnexion
    print("\n3Ã¯Â¸ÂÃ¢Æ’Â£ Test reconnexion:")
    client_fail = MockBinanceClient(should_fail=True)
    manager_fail = ConnectionManager(client_fail, {'max_retries': 3, 'retry_delay': 1})
    
    success = manager_fail.reconnect()
    print(f"   Reconnexion: {'Ã¢Å“â€¦ OK' if success else 'Ã¢ÂÅ’ FAIL'}")
    
    final_status = manager_fail.get_status()
    print(f"   Tentatives: {final_status['stats']['total_reconnections']}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")
