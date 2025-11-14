"""
Thread Manager pour The Bot
Gestion optimisÃƒÂ©e des threads pour exÃƒÂ©cution parallÃƒÂ¨le
"""

import threading
import time
import queue
from typing import Dict, Callable, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import psutil

logger = logging.getLogger(__name__)


class ThreadPriority(Enum):
    """PrioritÃƒÂ©s des threads"""
    CRITICAL = 1  # Market data, execution
    HIGH = 2      # Risk monitoring
    NORMAL = 3    # Strategy, analysis
    LOW = 4       # Logging, stats


class ThreadStatus(Enum):
    """Status des threads"""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class ManagedThread:
    """Classe reprÃƒÂ©sentant un thread gÃƒÂ©rÃƒÂ©"""
    
    def __init__(self, 
                 name: str,
                 target: Callable,
                 priority: ThreadPriority = ThreadPriority.NORMAL,
                 daemon: bool = True,
                 restart_on_error: bool = True):
        """
        CrÃƒÂ©e un thread gÃƒÂ©rÃƒÂ©
        
        Args:
            name: Nom du thread
            target: Fonction ÃƒÂ  exÃƒÂ©cuter
            priority: PrioritÃƒÂ© du thread
            daemon: Thread daemon ou non
            restart_on_error: RedÃƒÂ©marrer en cas d'erreur
        """
        self.name = name
        self.target = target
        self.priority = priority
        self.daemon = daemon
        self.restart_on_error = restart_on_error
        
        # Ãƒâ€°tat
        self.thread = None
        self.status = ThreadStatus.IDLE
        self.start_time = None
        self.error_count = 0
        self.last_error = None
        self.restart_count = 0
        
        # ContrÃƒÂ´le
        self.should_run = threading.Event()
        self.is_running = threading.Event()


class ThreadManager:
    """
    Gestionnaire central des threads
    
    ResponsabilitÃƒÂ©s:
    - CrÃƒÂ©er et dÃƒÂ©marrer les threads
    - Monitorer leur santÃƒÂ©
    - RedÃƒÂ©marrer en cas d'erreur
    - GÃƒÂ©rer les prioritÃƒÂ©s
    - Optimiser l'utilisation CPU
    """
    
    def __init__(self, config: Dict, bot_instance):
        """
        Initialise le thread manager
        
        Args:
            config: Configuration
            bot_instance: Instance du bot principal
        """
        self.config = config
        self.bot = bot_instance
        
        # Configuration
        self.max_threads = getattr(config, 'MAX_THREADS', 4)
        self.monitor_interval = getattr(config, 'MONITOR_INTERVAL', 5)
        self.max_restart_attempts = getattr(config, 'MAX_RESTART_ATTEMPTS', 3)
        
        # Threads gÃƒÂ©rÃƒÂ©s
        self.threads = {}
        
        # Ãƒâ€°tat
        self.is_running = False
        self.monitor_thread = None
        
        # Statistiques
        self.stats = {
            'threads_created': 0,
            'threads_restarted': 0,
            'total_errors': 0,
            'uptime_seconds': 0
        }
        
        self.start_time = None
        
        # Initialiser les threads
        self._initialize_threads()
        
        logger.info(f"Thread Manager initialisÃƒÂ© avec {self.max_threads} threads max")
    
    def _initialize_threads(self):
        """Initialise tous les threads nÃƒÂ©cessaires"""
        
        # Thread 1: Market Data Collection (CRITICAL)
        self.register_thread(
            name="market_data",
            target=self._market_data_thread,
            priority=ThreadPriority.CRITICAL,
            restart_on_error=True
        )
        
        # Thread 2: Strategy Processing (NORMAL)
        self.register_thread(
            name="strategy",
            target=self._strategy_thread,
            priority=ThreadPriority.NORMAL,
            restart_on_error=True
        )
        
        # Thread 3: Risk Monitoring (HIGH)
        self.register_thread(
            name="risk_monitor",
            target=self._risk_monitor_thread,
            priority=ThreadPriority.HIGH,
            restart_on_error=True
        )
        
        # Thread 4: Performance Tracking (LOW)
        self.register_thread(
            name="performance",
            target=self._performance_thread,
            priority=ThreadPriority.LOW,
            restart_on_error=False
        )
    
    def register_thread(self,
                       name: str,
                       target: Callable,
                       priority: ThreadPriority = ThreadPriority.NORMAL,
                       daemon: bool = True,
                       restart_on_error: bool = True):
        """
        Enregistre un nouveau thread
        
        Args:
            name: Nom du thread
            target: Fonction ÃƒÂ  exÃƒÂ©cuter
            priority: PrioritÃƒÂ©
            daemon: Thread daemon
            restart_on_error: RedÃƒÂ©marrer si erreur
        """
        if name in self.threads:
            logger.warning(f"Thread {name} dÃƒÂ©jÃƒÂ  enregistrÃƒÂ©")
            return
        
        managed_thread = ManagedThread(
            name=name,
            target=target,
            priority=priority,
            daemon=daemon,
            restart_on_error=restart_on_error
        )
        
        self.threads[name] = managed_thread
        self.stats['threads_created'] += 1
        
        logger.info(f"Thread '{name}' enregistrÃƒÂ© (prioritÃƒÂ©: {priority.name})")
    
    def start_all(self):
        """DÃƒÂ©marre tous les threads"""
        if self.is_running:
            logger.warning("Thread Manager dÃƒÂ©jÃƒÂ  en cours")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # DÃƒÂ©marrer les threads par ordre de prioritÃƒÂ©
        sorted_threads = sorted(
            self.threads.values(),
            key=lambda t: t.priority.value
        )
        
        for managed_thread in sorted_threads:
            self._start_thread(managed_thread)
            time.sleep(0.1)  # Petit dÃƒÂ©lai entre dÃƒÂ©marrages
        
        # DÃƒÂ©marrer le monitor
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Ã¢Å“â€¦ {len(self.threads)} threads dÃƒÂ©marrÃƒÂ©s")
    
    def stop_all(self):
        """ArrÃƒÂªte tous les threads"""
        logger.info("ArrÃƒÂªt de tous les threads...")
        
        self.is_running = False
        
        # Signaler l'arrÃƒÂªt ÃƒÂ  tous les threads
        for managed_thread in self.threads.values():
            managed_thread.should_run.clear()
        
        # Attendre l'arrÃƒÂªt (avec timeout)
        timeout = 5
        for managed_thread in self.threads.values():
            if managed_thread.thread and managed_thread.thread.is_alive():
                logger.info(f"Attente arrÃƒÂªt thread '{managed_thread.name}'...")
                managed_thread.thread.join(timeout=timeout)
                
                if managed_thread.thread.is_alive():
                    logger.warning(f"Thread '{managed_thread.name}' ne s'est pas arrÃƒÂªtÃƒÂ© proprement")
        
        # ArrÃƒÂªter le monitor
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        logger.info("Tous les threads arrÃƒÂªtÃƒÂ©s")
    
    def _start_thread(self, managed_thread: ManagedThread):
        """
        DÃƒÂ©marre un thread individuel
        
        Args:
            managed_thread: Le thread ÃƒÂ  dÃƒÂ©marrer
        """
        try:
            # Wrapper pour gÃƒÂ©rer les erreurs
            def thread_wrapper():
                managed_thread.is_running.set()
                managed_thread.status = ThreadStatus.RUNNING
                
                try:
                    managed_thread.target()
                except Exception as e:
                    logger.error(f"Erreur dans thread '{managed_thread.name}': {e}")
                    managed_thread.last_error = str(e)
                    managed_thread.error_count += 1
                    managed_thread.status = ThreadStatus.ERROR
                    self.stats['total_errors'] += 1
                finally:
                    managed_thread.is_running.clear()
                    if not self.is_running:
                        managed_thread.status = ThreadStatus.STOPPED
            
            # CrÃƒÂ©er et dÃƒÂ©marrer le thread
            managed_thread.should_run.set()
            managed_thread.thread = threading.Thread(
                target=thread_wrapper,
                name=managed_thread.name,
                daemon=managed_thread.daemon
            )
            managed_thread.thread.start()
            managed_thread.start_time = datetime.now()
            
            logger.info(f"Thread '{managed_thread.name}' dÃƒÂ©marrÃƒÂ©")
            
        except Exception as e:
            logger.error(f"Erreur dÃƒÂ©marrage thread '{managed_thread.name}': {e}")
    
    def _monitor_loop(self):
        """Boucle de monitoring des threads"""
        while self.is_running:
            try:
                # Check tous les threads
                for name, managed_thread in self.threads.items():
                    self._check_thread_health(managed_thread)
                
                # Mise ÃƒÂ  jour stats
                if self.start_time:
                    self.stats['uptime_seconds'] = (datetime.now() - self.start_time).seconds
                
                # Log pÃƒÂ©riodique
                if self.stats['uptime_seconds'] % 60 == 0:  # Toutes les minutes
                    self._log_status()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Erreur monitor loop: {e}")
                time.sleep(10)
    
    def _check_thread_health(self, managed_thread: ManagedThread):
        """
        VÃƒÂ©rifie la santÃƒÂ© d'un thread
        
        Args:
            managed_thread: Thread ÃƒÂ  vÃƒÂ©rifier
        """
        # Thread devrait ÃƒÂªtre en cours mais ne l'est pas
        if managed_thread.should_run.is_set() and not managed_thread.is_running.is_set():
            
            # Si le thread est en erreur et doit redÃƒÂ©marrer
            if managed_thread.status == ThreadStatus.ERROR and managed_thread.restart_on_error:
                if managed_thread.restart_count < self.max_restart_attempts:
                    logger.warning(f"RedÃƒÂ©marrage thread '{managed_thread.name}' "
                                 f"(tentative {managed_thread.restart_count + 1})")
                    
                    # Attendre un peu avant redÃƒÂ©marrage
                    time.sleep(2 ** managed_thread.restart_count)  # Backoff exponentiel
                    
                    # RedÃƒÂ©marrer
                    self._start_thread(managed_thread)
                    managed_thread.restart_count += 1
                    self.stats['threads_restarted'] += 1
                else:
                    logger.error(f"Thread '{managed_thread.name}' - "
                               f"max tentatives de redÃƒÂ©marrage atteintes")
    
    def _log_status(self):
        """Log le statut des threads"""
        active_count = sum(1 for t in self.threads.values() 
                          if t.is_running.is_set())
        
        logger.info(f"Thread Status: {active_count}/{len(self.threads)} actifs, "
                   f"Errors: {self.stats['total_errors']}, "
                   f"Restarts: {self.stats['threads_restarted']}")
    
    # =============================================================
    # THREADS MÃƒâ€°TIER
    # =============================================================
    
    def _market_data_thread(self):
        """Thread de collecte des donnÃƒÂ©es de marchÃƒÂ©"""
        logger.info("Market Data Thread dÃƒÂ©marrÃƒÂ©")
        
        managed_thread = self.threads.get('market_data')
        if not managed_thread:
            return
        
        while managed_thread.should_run.is_set() and self.bot.is_running:
            try:
                # RÃƒÂ©cupÃƒÂ©rer les symboles ÃƒÂ  surveiller
                if hasattr(self.bot, 'scanner') and self.bot.scanner:
                    symbols = self.bot.scanner.get_top_symbols()
                else:
                    symbols = ['BTCUSDC', 'ETHUSDC']  # DÃƒÂ©faut
                
                # Collecter les donnÃƒÂ©es pour chaque symbole
                for symbol in symbols:
                    if not managed_thread.should_run.is_set():
                        break
                    
                    # RÃƒÂ©cupÃƒÂ©rer les donnÃƒÂ©es
                    ticker = self.bot.exchange.get_symbol_ticker(symbol)
                    if ticker:
                        # RÃƒÂ©cupÃƒÂ©rer l'orderbook
                        orderbook = self.bot.exchange.get_orderbook(symbol, limit=20)
                        
                        # RÃƒÂ©cupÃƒÂ©rer les klines
                        df = self.bot.exchange.get_klines(symbol, '5m', limit=100)
                        
                        if not df.empty:
                            # Calculer les indicateurs
                            from utils.indicators import TechnicalIndicators
                            indicators = TechnicalIndicators()
                            df = indicators.calculate_all(df)
                            
                            # PrÃƒÂ©parer les donnÃƒÂ©es
                            market_data = {
                                'df': df,
                                'ticker': ticker,
                                'orderbook': orderbook,
                                'symbol': symbol,
                                'timestamp': datetime.now()
                            }
                            
                            # Envoyer au strategy manager
                            if hasattr(self.bot, 'strategy_manager'):
                                self.bot.strategy_manager.process_market_data(symbol, market_data)
                
                # Pause entre cycles
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur market data thread: {e}")
                time.sleep(5)
        
        logger.info("Market Data Thread arrÃƒÂªtÃƒÂ©")
    
    def _strategy_thread(self):
        """Thread de traitement des stratÃƒÂ©gies"""
        logger.info("Strategy Thread dÃƒÂ©marrÃƒÂ©")
        
        managed_thread = self.threads.get('strategy')
        if not managed_thread:
            return
        
        while managed_thread.should_run.is_set() and self.bot.is_running:
            try:
                # Le strategy manager a son propre processing loop
                # Ce thread surveille juste son ÃƒÂ©tat
                
                if hasattr(self.bot, 'strategy_manager'):
                    status = self.bot.strategy_manager.get_status()
                    
                    # Log pÃƒÂ©riodique
                    if int(time.time()) % 30 == 0:  # Toutes les 30 secondes
                        logger.debug(f"StratÃƒÂ©gies actives: {status['active_strategies']}, "
                                   f"Positions: {status['open_positions']}")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Erreur strategy thread: {e}")
                time.sleep(10)
        
        logger.info("Strategy Thread arrÃƒÂªtÃƒÂ©")
    
    def _risk_monitor_thread(self):
        """Thread de surveillance des risques"""
        logger.info("Risk Monitor Thread dÃƒÂ©marrÃƒÂ©")
        
        managed_thread = self.threads.get('risk_monitor')
        if not managed_thread:
            return
        
        while managed_thread.should_run.is_set() and self.bot.is_running:
            try:
                if hasattr(self.bot, 'risk_monitor'):
                    # RÃƒÂ©cupÃƒÂ©rer les positions actuelles
                    positions = {}
                    if hasattr(self.bot, 'strategy_manager'):
                        positions = self.bot.strategy_manager.positions
                    
                    # Mettre ÃƒÂ  jour le risk monitor
                    report = self.bot.risk_monitor.update(
                        current_capital=self.bot.capital,
                        positions=positions
                    )
                    
                    # RÃƒÂ©agir selon le niveau de risque
                    if report['risk_level'] == 'EMERGENCY':
                        logger.critical("Ã°Å¸Å¡Â¨ NIVEAU D'URGENCE - Fermeture de toutes les positions!")
                        if hasattr(self.bot, 'strategy_manager'):
                            self.bot.strategy_manager.close_all_positions('emergency')
                    
                    elif report['risk_level'] == 'CRITICAL':
                        logger.error("Ã¢ÂÅ’ Niveau critique - Trading suspendu")
                        if hasattr(self.bot, 'strategy_manager'):
                            self.bot.strategy_manager.disable_trading()
                
                # Check toutes les 5 secondes
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Erreur risk monitor thread: {e}")
                time.sleep(10)
        
        logger.info("Risk Monitor Thread arrÃƒÂªtÃƒÂ©")
    
    def _performance_thread(self):
        """Thread de tracking de performance"""
        logger.info("Performance Thread dÃƒÂ©marrÃƒÂ©")
        
        managed_thread = self.threads.get('performance')
        if not managed_thread:
            return
        
        while managed_thread.should_run.is_set() and self.bot.is_running:
            try:
                # Collecter les mÃƒÂ©triques
                metrics = {
                    'timestamp': datetime.now(),
                    'capital': self.bot.capital
                }
                
                # MÃƒÂ©triques du strategy manager
                if hasattr(self.bot, 'strategy_manager'):
                    status = self.bot.strategy_manager.get_status()
                    metrics['positions'] = status['open_positions']
                    metrics['performance'] = status['performance']
                
                # MÃƒÂ©triques du risk monitor
                if hasattr(self.bot, 'risk_monitor'):
                    metrics['risk_level'] = self.bot.risk_monitor.current_risk_level.value
                    metrics['drawdown'] = self.bot.risk_monitor.current_drawdown
                
                # MÃƒÂ©triques d'exÃƒÂ©cution
                if hasattr(self.bot, 'order_manager'):
                    metrics['orders'] = self.bot.order_manager.get_stats()
                
                # Sauvegarder ou logger
                if int(time.time()) % 60 == 0:  # Chaque minute
                    self._log_performance(metrics)
                
                # Mise ÃƒÂ  jour du dashboard si disponible
                if hasattr(self.bot, 'dashboard'):
                    self.bot.dashboard.update()
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Erreur performance thread: {e}")
                time.sleep(30)
        
        logger.info("Performance Thread arrÃƒÂªtÃƒÂ©")
    
    def _log_performance(self, metrics: Dict):
        """Log les mÃƒÂ©triques de performance"""
        perf_msg = f"\n{'='*60}\n"
        perf_msg += f"PERFORMANCE UPDATE - {metrics['timestamp'].strftime('%H:%M:%S')}\n"
        perf_msg += f"{'='*60}\n"
        perf_msg += f"Capital: ${metrics['capital']:,.2f}\n"
        perf_msg += f"Positions: {metrics.get('positions', 0)}\n"
        perf_msg += f"Risk Level: {metrics.get('risk_level', 'N/A')}\n"
        perf_msg += f"Drawdown: {metrics.get('drawdown', 0):.2%}\n"
        
        if 'orders' in metrics:
            perf_msg += f"Orders: {metrics['orders']['successful']}/{metrics['orders']['total_orders']}\n"
        
        perf_msg += f"{'='*60}\n"
        
        logger.info(perf_msg)
    
    def get_thread_status(self, name: str) -> Optional[Dict]:
        """
        Retourne le statut d'un thread
        
        Args:
            name: Nom du thread
            
        Returns:
            Dict avec le statut ou None
        """
        if name not in self.threads:
            return None
        
        managed_thread = self.threads[name]
        
        return {
            'name': name,
            'status': managed_thread.status.value,
            'is_running': managed_thread.is_running.is_set(),
            'priority': managed_thread.priority.name,
            'error_count': managed_thread.error_count,
            'restart_count': managed_thread.restart_count,
            'last_error': managed_thread.last_error,
            'uptime': (datetime.now() - managed_thread.start_time).seconds if managed_thread.start_time else 0
        }
    
    def get_all_status(self) -> Dict:
        """Retourne le statut de tous les threads"""
        return {
            'is_running': self.is_running,
            'thread_count': len(self.threads),
            'active_threads': sum(1 for t in self.threads.values() if t.is_running.is_set()),
            'stats': self.stats,
            'threads': {
                name: self.get_thread_status(name)
                for name in self.threads.keys()
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'thread_count_system': threading.active_count()
            }
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du thread manager"""
    
    # Mock bot pour test
    class MockBot:
        def __init__(self):
            self.running = True
            self.capital = 1000
            self.exchange = None
            self.scanner = None
            self.strategy_manager = None
            self.risk_monitor = None
            self.order_manager = None
    
    # Configuration
    config = {
        'max_threads': 4,
        'monitor_interval': 2,
        'max_restart_attempts': 3
    }
    
    # CrÃƒÂ©er le bot et le manager
    bot = MockBot()
    manager = ThreadManager(config, bot)
    
    print("=" * 60)
    print("TEST THREAD MANAGER")
    print("=" * 60)
    
    # DÃƒÂ©marrer
    print("\nÃ¢â€“Â¶Ã¯Â¸Â DÃƒÂ©marrage des threads...")
    manager.start_all()
    
    # Attendre un peu
    time.sleep(5)
    
    # Afficher le statut
    print("\nÃ°Å¸â€œÅ  Statut des threads:")
    status = manager.get_all_status()
    print(f"Threads actifs: {status['active_threads']}/{status['thread_count']}")
    print(f"CPU: {status['system']['cpu_percent']:.1f}%")
    print(f"RAM: {status['system']['memory_percent']:.1f}%")
    
    for name, thread_status in status['threads'].items():
        print(f"\n  {name}:")
        print(f"    Status: {thread_status['status']}")
        print(f"    Running: {thread_status['is_running']}")
        print(f"    Errors: {thread_status['error_count']}")
    
    # Attendre encore
    time.sleep(5)
    
    # ArrÃƒÂªter
    print("\nÃ¢ÂÂ¹Ã¯Â¸Â ArrÃƒÂªt des threads...")
    bot.running = False
    manager.stop_all()
    
    print("\nÃ¢Å“â€¦ Test terminÃƒÂ©!")
