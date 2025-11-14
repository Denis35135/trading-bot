#!/usr/bin/env python3
"""
🚀 AUTOBOT ULTIMATE - Main Entry Point
Bot de trading automatisé haute performance
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
from utils.database import Database
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
        logger.info("🚀 INITIALISATION DU BOT ULTIME")
        logger.info("="*80)
        
        self.running = False
        self.mode = config.MODE  # 'paper' ou 'live'
        
        # Queues pour communication inter-threads
        self.data_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=100)
        self.order_queue = Queue(maxsize=50)
        
        # État du bot
        self.capital = config.INITIAL_CAPITAL
        self.positions = {}
        self.daily_pnl = 0.0
        self.trades_today = 0
        
        # Composants principaux
        self.db = Database(config)
        self.binance = BinanceClient(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
            testnet=config.BINANCE_TESTNET
        )
        self.performance_tracker = PerformanceTracker(self.db)
        self.risk_monitor = RiskMonitor(self.capital)
        
        # Threads
        self.threads = {}
        self.thread_objects = {}
        
        # Signal handler pour arrêt propre
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Capital initial: {self.capital} USDC")
        logger.info(f"Testnet: {config.BINANCE_TESTNET}")
        
    def signal_handler(self, sig, frame):
        """Gestion des signaux pour arrêt propre"""
        logger.warning(f"\n⚠️  Signal {sig} reçu - Arrêt propre du bot...")
        self.stop()
        
    def initialize_threads(self):
        """Initialise les 4 threads selon Documentation.docx"""
        logger.info("Initialisation des threads...")
        
        try:
            # Thread 1: Market Data Handler
            self.thread_objects['market_data'] = MarketDataThread(
                self.binance,
                self.data_queue,
                config.SYMBOLS_TO_TRADE
            )
            
            # Thread 2: Strategy Engine
            self.thread_objects['strategy'] = StrategyThread(
                self.data_queue,
                self.signal_queue,
                config.STRATEGIES
            )
            
            # Thread 3: Execution Engine
            self.thread_objects['execution'] = ExecutionThread(
                self.binance,
                self.signal_queue,
                self.order_queue,
                self.risk_monitor,
                self.capital
            )
            
            # Thread 4: Risk Monitor
            self.thread_objects['risk'] = RiskThread(
                self.risk_monitor,
                self.positions,
                self.capital
            )
            
            logger.info("✅ 4 threads initialisés")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation threads: {e}")
            return False
            
    def start_threads(self):
        """Démarre tous les threads"""
        logger.info("Démarrage des threads...")
        
        for name, thread_obj in self.thread_objects.items():
            try:
                thread = threading.Thread(target=thread_obj.run, name=name)
                thread.daemon = True
                thread.start()
                self.threads[name] = thread
                logger.info(f"✅ Thread '{name}' démarré")
                time.sleep(0.5)  # Petit délai entre chaque thread
                
            except Exception as e:
                logger.error(f"❌ Erreur démarrage thread '{name}': {e}")
                return False
                
        return True
        
    def monitor_threads(self):
        """Surveille l'état des threads"""
        for name, thread in self.threads.items():
            if not thread.is_alive():
                logger.error(f"❌ Thread '{name}' est mort ! Redémarrage...")
                try:
                    # Redémarre le thread
                    thread_obj = self.thread_objects[name]
                    new_thread = threading.Thread(target=thread_obj.run, name=name)
                    new_thread.daemon = True
                    new_thread.start()
                    self.threads[name] = new_thread
                    logger.info(f"✅ Thread '{name}' redémarré")
                except Exception as e:
                    logger.error(f"❌ Impossible de redémarrer '{name}': {e}")
                    
    def print_status(self):
        """Affiche le statut du bot (style Documentation.docx)"""
        try:
            # Calculs métriques
            win_rate = self.performance_tracker.get_win_rate()
            drawdown = self.risk_monitor.get_current_drawdown()
            daily_pnl_pct = (self.daily_pnl / self.capital) * 100 if self.capital > 0 else 0
            
            status = f"""
╔══════════════════════════════════════╗
║ AUTOBOT STATUS - {datetime.now().strftime('%H:%M:%S')} ║
╠══════════════════════════════════════╣
║ Mode: {self.mode.upper():8s}                    ║
║ Capital: ${self.capital:,.2f}              ║
║ P&L Today: ${self.daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%)      ║
║ Drawdown: {drawdown:.2%}                   ║
║ Win Rate: {win_rate:.1%}                  ║
║ Positions: {len(self.positions)}/20               ║
║ Trades/Day: {self.trades_today}                  ║
╠══════════════════════════════════════╣
║ Threads:                             ║
║  • Market Data: {'🟢' if self.threads.get('market_data', None) and self.threads['market_data'].is_alive() else '🔴'}               ║
║  • Strategy: {'🟢' if self.threads.get('strategy', None) and self.threads['strategy'].is_alive() else '🔴'}                  ║
║  • Execution: {'🟢' if self.threads.get('execution', None) and self.threads['execution'].is_alive() else '🔴'}                 ║
║  • Risk: {'🟢' if self.threads.get('risk', None) and self.threads['risk'].is_alive() else '🔴'}                      ║
╠══════════════════════════════════════╣
║ Status: {'🟢 RUNNING' if self.running else '🔴 STOPPED':28s}    ║
╚══════════════════════════════════════╝
"""
            print(status)
            
        except Exception as e:
            logger.error(f"Erreur affichage status: {e}")
            
    def run(self):
        """Boucle principale du bot - 100% autonome H24"""
        logger.info("\n" + "="*80)
        logger.info("🚀 DÉMARRAGE DU BOT")
        logger.info("="*80 + "\n")
        
        # Vérifications pré-démarrage
        if not self.pre_flight_checks():
            logger.error("❌ Pre-flight checks échoués")
            return False
            
        # Initialise et démarre les threads
        if not self.initialize_threads():
            logger.error("❌ Échec initialisation threads")
            return False
            
        if not self.start_threads():
            logger.error("❌ Échec démarrage threads")
            return False
            
        self.running = True
        logger.info("✅ Bot démarré avec succès!\n")
        
        # Compteurs pour monitoring
        status_counter = 0
        health_counter = 0
        save_counter = 0
        
        # BOUCLE PRINCIPALE H24
        try:
            while self.running:
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
                    
                # Sauvegarde données toutes les 5 minutes
                save_counter += 1
                if save_counter >= config.SAVE_INTERVAL:
                    self.save_state()
                    save_counter = 0
                    
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interruption clavier détectée")
        except Exception as e:
            logger.error(f"❌ Erreur dans boucle principale: {e}")
        finally:
            self.stop()
            
        return True
        
    def pre_flight_checks(self):
        """Vérifications avant démarrage"""
        logger.info("🔍 Pre-flight checks...")
        
        checks = []
        
        # Check 1: Connection Binance
        try:
            server_time = self.binance.get_server_time()
            checks.append(('Binance Connection', True))
            logger.info(f"✅ Connexion Binance OK (server time: {server_time})")
        except Exception as e:
            checks.append(('Binance Connection', False))
            logger.error(f"❌ Connexion Binance échouée: {e}")
            
        # Check 2: Database
        try:
            self.db.connect()
            checks.append(('Database', True))
            logger.info("✅ Database OK")
        except Exception as e:
            checks.append(('Database', False))
            logger.error(f"❌ Database échouée: {e}")
            
        # Check 3: Capital suffisant
        if self.capital >= config.MIN_ORDER_SIZE:
            checks.append(('Capital', True))
            logger.info(f"✅ Capital OK: {self.capital} USDC")
        else:
            checks.append(('Capital', False))
            logger.error(f"❌ Capital insuffisant: {self.capital} < {config.MIN_ORDER_SIZE}")
            
        # Check 4: Dossiers data
        required_dirs = ['data/logs', 'data/models', 'data/cache', 'data/backtest']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        checks.append(('Data Directories', True))
        logger.info("✅ Dossiers data OK")
        
        # Résultat
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            logger.info("\n✅ Tous les pre-flight checks passés!\n")
        else:
            logger.error("\n❌ Certains pre-flight checks ont échoué:")
            for name, passed in checks:
                logger.error(f"  {name}: {'✅' if passed else '❌'}")
            logger.error("")
            
        return all_passed
        
    def update_metrics(self):
        """Met à jour les métriques du bot"""
        try:
            # Met à jour le capital depuis les positions
            total_value = self.capital
            for symbol, position in self.positions.items():
                total_value += position.get('unrealized_pnl', 0)
                
            # Update performance tracker
            self.performance_tracker.update(
                capital=total_value,
                positions=self.positions,
                trades_today=self.trades_today
            )
            
        except Exception as e:
            logger.error(f"Erreur update metrics: {e}")
            
    def save_state(self):
        """Sauvegarde l'état du bot"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'capital': self.capital,
                'positions': self.positions,
                'daily_pnl': self.daily_pnl,
                'trades_today': self.trades_today
            }
            
            # Sauvegarde dans DB
            self.db.save_state(state)
            logger.debug("💾 État sauvegardé")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde état: {e}")
            
    def stop(self):
        """Arrêt propre du bot"""
        if not self.running:
            return
            
        logger.info("\n" + "="*80)
        logger.info("🛑 ARRÊT DU BOT")
        logger.info("="*80)
        
        self.running = False
        
        # Arrête tous les threads
        for name, thread_obj in self.thread_objects.items():
            try:
                logger.info(f"Arrêt thread '{name}'...")
                thread_obj.stop()
            except Exception as e:
                logger.error(f"Erreur arrêt thread '{name}': {e}")
                
        # Attend que les threads se terminent
        for name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Attente thread '{name}'...")
                thread.join(timeout=5)
                
        # Ferme les positions en mode live
        if self.mode == 'live' and len(self.positions) > 0:
            logger.warning("⚠️  Fermeture des positions ouvertes...")
            # TODO: Implémenter fermeture positions
            
        # Sauvegarde finale
        self.save_state()
        
        # Ferme database
        try:
            self.db.close()
        except:
            pass
            
        logger.info("\n✅ Bot arrêté proprement")
        logger.info("="*80 + "\n")
        

def main():
    """Point d'entrée principal"""
    print("""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║         🚀 AUTOBOT ULTIMATE v1.0 🚀             ║
    ║                                                  ║
    ║     Bot de Trading Automatisé Haute Perf        ║
    ║              Mode: PAPER TRADING                 ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    # Crée et lance le bot
    bot = TradingBot()
    
    try:
        success = bot.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
