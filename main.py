#!/usr/bin/env python3
"""
 AUTOBOT ULTIMATE - Main Entry Point
Bot de trading automatise haute performance
Mode: PAPER TRADING
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime
from queue import Queue
import logging

# Configuration du logging
from utils.logger import setup_logger
logger = setup_logger('main', 'logs/main.log')

# Import des threads
from threads.market_data_thread import MarketDataThread
from threads.strategy_thread import StrategyThread
from threads.execution_thread import ExecutionThread
from threads.risk_thread import RiskThread

# Import des managers
from monitoring.performance_tracker import PerformanceTracker
from utils.database import DatabaseManager
# Sinon utilisez : from utils.database import DatabaseManager
from risk.risk_monitor import RiskMonitor
from exchange.binance_client import BinanceClient

# Import config
import config


class TradingBot:
    """
    Bot de trading principal avec architecture multi-thread
    4 threads: market_data, strategy, execution, risk monitoring
    """
    
    def __init__(self):
        """Initialisation du bot"""
        logger.info("="*80)
        logger.info(" INITIALISATION DU BOT ULTIME")
        logger.info("="*80)
        
        self.is_running = False
        self.mode = config.MODE  # 'paper' ou 'live'
        
        # Queues pour communication inter-threads
        self.data_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=100)
        self.order_queue = Queue(maxsize=50)
        
        # Etat du bot
        self.capital = config.INITIAL_CAPITAL
        self.positions = {}
        self.daily_pnl = 0.0
        self.trades_today = 0
        
        # Composants principaux
        self.db = DatabaseManager(config)
        self.binance = BinanceClient(
            api_key=config.BINANCE_API_KEY,
            secret_key=config.BINANCE_API_SECRET,
            testnet=config.BINANCE_TESTNET
        )
        self.performance_tracker = PerformanceTracker(config)
        self.risk_monitor = RiskMonitor(config)
        
        # Threads
        self.threads = {}
        self.thread_objects = {}
        
        # Signal handler pour arret propre
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Capital initial: {self.capital} USDC")
        logger.info(f"Testnet: {config.BINANCE_TESTNET}")
        
    def signal_handler(self, sig, frame):
        """Gestion des signaux pour arret propre"""
        logger.warning(f"\n Signal {sig} recu - Arret propre du bot...")
        self.stop()
        
    def initialize_threads(self):
        """Initialise les 4 threads selon Documentation.docx"""
        logger.info("Initialisation des threads...")
        
        try:
            # Thread 1: Market Data Handler
            # MarketDataThread(exchange_client, symbols, update_interval=5)
            self.thread_objects['market_data'] = MarketDataThread(
                self.binance,
                config.SYMBOLS_TO_TRADE
            )
            
            # Thread 2: Strategy Engine
            # StrategyThread(data_queue, signal_queue)
            self.thread_objects['strategy'] = StrategyThread(
                self.data_queue,
                self.signal_queue
            )
            
            # Thread 3: Execution Engine  
            # ExecutionThread(exchange_client, signal_queue, risk_manager)
            self.thread_objects['execution'] = ExecutionThread(
                self.binance,
                self.signal_queue,
                self.risk_monitor
            )
            
            # Thread 4: Risk Monitor
            # RiskThread(risk_monitor, config)
            self.thread_objects['risk'] = RiskThread(
                self.risk_monitor,
                config
            )
            
            logger.info("[OK] 4 threads initialises")
            return True
            
        except Exception as e:
            logger.error(f"[ERREUR] Erreur initialisation threads: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def start_threads(self):
        """Demarre tous les threads"""
        logger.info("Demarrage des threads...")
        
        for name, thread_obj in self.thread_objects.items():
            try:
                # Demarre le thread interne de chaque objet
                thread_obj.start()
                logger.info(f"[OK] Thread '{name}' demarre")
                time.sleep(0.5)  # Petit delai entre chaque thread
                
            except Exception as e:
                logger.error(f"[ERREUR] Erreur demarrage thread '{name}': {e}")
                return False
                
        return True
        
    def monitor_threads(self):
        """Surveille l'etat des threads"""
        for name, thread_obj in self.thread_objects.items():
            try:
                # Verifie si le thread est toujours en cours
                if hasattr(thread_obj, 'is_running') and not thread_obj.is_running:
                    logger.warning(f"[ALERT] Thread '{name}' arrete - tentative de redemarrage...")
                    thread_obj.start()
                    logger.info(f"[OK] Thread '{name}' redemarre")
            except Exception as e:
                logger.error(f"[ERREUR] Monitoring thread '{name}': {e}")
                    
    def print_status(self):
        """Affiche le statut du bot (style Documentation.docx)"""
        try:
            # Calculs metriques
            win_rate = self.performance_tracker.stats.get('win_rate', 0.0)
            drawdown = self.risk_monitor.current_drawdown
            daily_pnl_pct = (self.daily_pnl / self.capital) * 100 if self.capital > 0 else 0
            
            # Verification etat threads
            thread_status = {}
            for name, thread_obj in self.thread_objects.items():
                if hasattr(thread_obj, 'is_running'):
                    thread_status[name] = thread_obj.is_running
                else:
                    thread_status[name] = False
            
            status = f"""

============================================
         AUTOBOT STATUS - {datetime.now().strftime('%H:%M:%S')}
============================================
Mode: {self.mode.upper():8s}                    
Capital: ${self.capital:,.2f}              
P&L Today: ${self.daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%)      
Drawdown: {drawdown:.2%}                   
Win Rate: {win_rate:.1%}                  
Positions: {len(self.positions)}/20               
Trades/Day: {self.trades_today}                  

Threads:                             
  * Market Data: {'[OK]' if thread_status.get('market_data', False) else '[ERREUR]'}               
  * Strategy: {'[OK]' if thread_status.get('strategy', False) else '[ERREUR]'}                  
  * Execution: {'[OK]' if thread_status.get('execution', False) else '[ERREUR]'}                 
  * Risk: {'[OK]' if thread_status.get('risk', False) else '[ERREUR]'}                      

Status: {'[RUNNING]' if self.is_running else '[STOPPED]'}    
============================================
"""
            print(status)
            
        except Exception as e:
            logger.error(f"Erreur affichage status: {e}")
            
    def run(self):
        """Boucle principale du bot - 100% autonome H24"""
        logger.info("\n" + "="*80)
        logger.info(" DEMARRAGE DU BOT")
        logger.info("="*80 + "\n")
        
        # Verifications pre-demarrage
        if not self.pre_flight_checks():
            logger.error("[ERREUR] Pre-flight checks echoues")
            return False
            
        # Initialise et demarre les threads
        if not self.initialize_threads():
            logger.error("[ERREUR] Echec initialisation threads")
            return False
            
        if not self.start_threads():
            logger.error("[ERREUR] Echec demarrage threads")
            return False
            
        self.is_running = True
        logger.info("[OK] Bot demarre avec succes!\n")
        
        # Compteurs pour monitoring
        status_counter = 0
        health_counter = 0
        save_counter = 0
        
        # BOUCLE PRINCIPALE H24
        try:
            while self.is_running:
                time.sleep(1)  # 1 seconde entre chaque cycle
                
                # Affiche status toutes les 60 secondes
                status_counter += 1
                if status_counter >= 60:
                    self.print_status()
                    status_counter = 0
                    
                # Health check toutes les 60 secondes
                health_counter += 1
                if health_counter >= config.HEALTH_CHECK_INTERVAL:
                    self.monitor_threads()
                    self.update_metrics()
                    health_counter = 0
                    
                # Sauvegarde donnees toutes les 5 minutes
                save_counter += 1
                if save_counter >= config.SAVE_INTERVAL:
                    self.save_state()
                    save_counter = 0
                    
        except KeyboardInterrupt:
            logger.info("\n[ALERT] Interruption clavier detectee")
        except Exception as e:
            logger.error(f"[ERREUR] Erreur dans boucle principale: {e}")
        finally:
            self.stop()
            
        return True
        
    def pre_flight_checks(self):
        """Verifications avant demarrage"""
        logger.info("Pre-flight checks...")
        
        checks = []
        
        # Check 1: Connection Binance
        try:
            if self.binance.test_connection():
                checks.append(('Binance Connection', True))
                logger.info("[OK] Connexion Binance OK")
            else:
                checks.append(('Binance Connection', False))
                logger.error("[ERREUR] Connexion Binance echouee")
        except Exception as e:
            checks.append(('Binance Connection', False))
            logger.error(f"[ERREUR] Connexion Binance echouee: {e}")
            
        # Check 2: Database
        try:
            stats = self.db.get_database_stats()
            checks.append(('Database', True))
            logger.info(f"[OK] Database OK ({stats['total_trades']} trades)")
        except Exception as e:
            checks.append(('Database', False))
            logger.error(f"[ERREUR] Database echouee: {e}")
            
        # Check 3: Capital suffisant
        if self.capital >= config.MIN_ORDER_SIZE:
            checks.append(('Capital', True))
            logger.info(f"[OK] Capital OK: {self.capital} USDC")
        else:
            checks.append(('Capital', False))
            logger.error(f"[ERREUR] Capital insuffisant: {self.capital} < {config.MIN_ORDER_SIZE}")
            
        # Check 4: Dossiers data
        required_dirs = ['logs', 'data', 'data/backups', 'cache']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        checks.append(('Data Directories', True))
        logger.info("[OK] Dossiers data OK")
        
        # Resultat
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            logger.info("\n[OK] Tous les pre-flight checks passes!\n")
        else:
            logger.error("\n[ERREUR] Certains pre-flight checks ont echoue:")
            for name, passed in checks:
                if not passed:
                    logger.error(f"  {name}: [ECHEC]")
            logger.error("")
            
        return all_passed
        
    def update_metrics(self):
        """Met a jour les metriques du bot"""
        try:
            # Met a jour le capital depuis les positions
            total_value = self.capital
            for symbol, position in self.positions.items():
                total_value += position.get('unrealized_pnl', 0)
                
            # Update performance tracker
            self.performance_tracker.update_capital(total_value)
            
        except Exception as e:
            logger.error(f"Erreur update metrics: {e}")
            
    def save_state(self):
        """Sauvegarde l'etat du bot"""
        try:
            # Sauvegarde performance snapshot
            snapshot_data = {
                'timestamp': datetime.now(),
                'total_capital': self.capital,
                'available_capital': self.capital - sum(p.get('size_usdc', 0) for p in self.positions.values()),
                'total_exposure': sum(p.get('size_usdc', 0) for p in self.positions.values()),
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': (self.daily_pnl / self.capital) * 100 if self.capital > 0 else 0,
                'total_pnl': self.performance_tracker.stats.get('total_pnl', 0),
                'total_pnl_pct': (self.performance_tracker.stats.get('total_pnl', 0) / config.INITIAL_CAPITAL) * 100,
                'total_trades': self.performance_tracker.stats.get('total_trades', 0),
                'winning_trades': self.performance_tracker.stats.get('winning_trades', 0),
                'losing_trades': self.performance_tracker.stats.get('losing_trades', 0),
                'win_rate': self.performance_tracker.stats.get('win_rate', 0),
                'open_positions': len(self.positions),
                'current_drawdown': self.risk_monitor.current_drawdown,
                'max_drawdown': self.risk_monitor.stats.get('max_drawdown', 0),
                'sharpe_ratio': self.performance_tracker.stats.get('sharpe_ratio', 0),
                'profit_factor': self.performance_tracker.stats.get('profit_factor', 0)
            }
            
            self.db.save_performance_snapshot(snapshot_data)
            logger.debug("[OK] Etat sauvegarde")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde etat: {e}")
            
    def stop(self):
        """Arret propre du bot"""
        if not self.is_running:
            return
            
        logger.info("\n" + "="*80)
        logger.info(" ARRET DU BOT")
        logger.info("="*80)
        
        self.is_running = False
        
        # Arrete tous les threads
        for name, thread_obj in self.thread_objects.items():
            try:
                logger.info(f"Arret thread '{name}'...")
                thread_obj.stop()
            except Exception as e:
                logger.error(f"Erreur arret thread '{name}': {e}")
                
        # Attendre un peu pour que les threads se terminent
        time.sleep(2)
                
        # Ferme les positions en mode live
        if self.mode == 'live' and len(self.positions) > 0:
            logger.warning("[ALERT] Fermeture des positions ouvertes...")
            # TODO: Implementer fermeture positions
            
        # Sauvegarde finale
        self.save_state()
        
        # Ferme database
        try:
            self.db.close()
        except:
            pass
            
        logger.info("\n[OK] Bot arrete proprement")
        logger.info("="*80 + "\n")
        

def main():
    """Point d'entree principal"""
    print("""
    
====================================================
           AUTOBOT ULTIMATE v1.0              
====================================================
      Bot de Trading Automatise Haute Perf        
             Mode: PAPER TRADING                 
====================================================
    
    """)
    
    # Cree et lance le bot
    bot = TradingBot()
    
    try:
        success = bot.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"[ERREUR] Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())