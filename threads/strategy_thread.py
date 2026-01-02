"""
Strategy Thread pour The Bot
Thread de traitement des strategies de trading
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)


class StrategyThread:
    """
    Thread de traitement des strategies
    
    Responsabilites:
    - Recevoir les donnees de marche
    - Analyser avec toutes les strategies actives
    - Generer des signaux de trading
    - Envoyer les signaux au thread d'execution
    - Gerer les allocations par strategie
    """
    
    def __init__(self, bot_instance, config: Dict):
        """
        Initialise le thread de strategies
        
        Args:
            bot_instance: Instance du bot principal
            config: Configuration
        """
        self.bot = bot_instance
        self.config = config
        self.is_running = False
        self.thread = None
        
        # Configuration
        self.process_interval = getattr(config, 'PROCESS_INTERVAL', 1)  # 1 seconde
        self.min_signal_confidence = getattr(config, 'MIN_CONFIDENCE', 0.65)
        
        # Queues
        self.data_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=100)
        
        # Etat
        self.active_strategies = {}
        self.strategy_allocations = {}
        self.last_signals = {}
        
        # Statistiques
        self.stats = {
            'data_processed': 0,
            'signals_generated': 0,
            'signals_filtered': 0,
            'by_strategy': {},
            'last_update': None
        }
        
        logger.info("Strategy Thread initialise")
    
    def start(self):
        """Demarre le thread"""
        if self.is_running:
            logger.warning("Strategy Thread deja en cours")
            return
        
        # Charger les strategies
        self._initialize_strategies()
        
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="StrategyThread"
        )
        self.thread.start()
        
        logger.info("[OK] Strategy Thread demarre")
    
    def stop(self):
        """Arrete le thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Strategy Thread arrete")
    
    def _initialize_strategies(self):
        """Initialise les strategies depuis le strategy manager"""
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                logger.warning("Strategy manager non disponible")
                return
            
            self.active_strategies = self.bot.strategy_manager.active_strategies
            self.strategy_allocations = self.bot.strategy_manager.strategy_allocations
            
            # Initialiser les stats par strategie
            for strategy_name in self.active_strategies:
                self.stats['by_strategy'][strategy_name] = {
                    'signals_generated': 0,
                    'signals_accepted': 0,
                    'avg_confidence': 0
                }
            
            logger.info(f"{len(self.active_strategies)} strategies chargees: {list(self.active_strategies.keys())}")
        
        except Exception as e:
            logger.error(f"Erreur initialisation strategies: {e}")
    
    def add_market_data(self, symbol: str, market_data: Dict):
        """
        Ajoute des donnees de marche a traiter
        
        Args:
            symbol: Symbole
            market_data: Donnees de marche
        """
        try:
            self.data_queue.put({
                'symbol': symbol,
                'data': market_data,
                'timestamp': datetime.now()
            }, timeout=1)
        except Exception as e:
            logger.debug(f"Queue pleine, donnees ignorees: {symbol}")
    
    def _run(self):
        """Boucle principale du thread"""
        logger.info("Strategy Thread running...")
        
        while self.is_running:
            try:
                # Recuperer des donnees (timeout 1 seconde)
                try:
                    item = self.data_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Traiter les donnees
                self._process_market_data(item['symbol'], item['data'])
                
            except Exception as e:
                logger.error(f"Erreur dans strategy thread: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("Strategy Thread termine")
    
    def _process_market_data(self, symbol: str, market_data: Dict):
        """
        Traite les donnees de marche avec toutes les strategies
        
        Args:
            symbol: Symbole
            market_data: Donnees de marche
        """
        try:
            self.stats['data_processed'] += 1
            self.stats['last_update'] = datetime.now()
            
            # Verifier que les donnees sont completes
            if not self._validate_market_data(market_data):
                return
            
            # Analyser avec chaque strategie
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    # Analyser
                    signal = strategy.analyze(market_data)
                    
                    if signal:
                        # Ajouter des metadonnees
                        signal['strategy'] = strategy_name
                        signal['symbol'] = symbol
                        signal['timestamp'] = datetime.now()
                        
                        # Stats
                        self.stats['by_strategy'][strategy_name]['signals_generated'] += 1
                        
                        # Valider et filtrer le signal
                        if self._validate_signal(signal):
                            # Envoyer au thread d'execution
                            self._send_signal(signal)
                            
                            self.stats['signals_generated'] += 1
                            self.stats['by_strategy'][strategy_name]['signals_accepted'] += 1
                        else:
                            self.stats['signals_filtered'] += 1
                
                except Exception as e:
                    logger.error(f"Erreur analyse {strategy_name} pour {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Erreur process_market_data: {e}")
    
    def _validate_market_data(self, market_data: Dict) -> bool:
        """
        Valide que les donnees de marche sont completes
        
        Args:
            market_data: Donnees a valider
            
        Returns:
            True si valide
        """
        required_keys = ['df', 'ticker']
        
        for key in required_keys:
            if key not in market_data:
                return False
        
        # Verifier que le DataFrame n'est pas vide
        df = market_data.get('df')
        if df is None or df.empty:
            return False
        
        return True
    
    def _validate_signal(self, signal: Dict) -> bool:
        """
        Valide un signal de trading
        
        Args:
            signal: Signal a valider
            
        Returns:
            True si valide
        """
        try:
            # Verifier les champs requis
            required_fields = ['type', 'side', 'price', 'confidence', 'symbol']
            for field in required_fields:
                if field not in signal:
                    logger.debug(f"Signal invalide: champ '{field}' manquant")
                    return False
            
            # Verifier la confiance minimum
            confidence = signal.get('confidence', 0)
            if confidence < self.min_signal_confidence:
                logger.debug(
                    f"Signal filtre: confiance {confidence:.2%} < "
                    f"minimum {self.min_signal_confidence:.2%}"
                )
                return False
            
            # Verifier que le type est valide
            if signal['type'] not in ['ENTRY', 'EXIT']:
                logger.debug(f"Signal invalide: type '{signal['type']}' inconnu")
                return False
            
            # Verifier que le side est valide
            if signal['side'] not in ['BUY', 'SELL']:
                logger.debug(f"Signal invalide: side '{signal['side']}' inconnu")
                return False
            
            # Verifier le prix
            price = signal.get('price')
            if not price or price <= 0:
                logger.debug(f"Signal invalide: prix {price}")
                return False
            
            # Verifier les doublons (meme signal recent)
            if self._is_duplicate_signal(signal):
                logger.debug(f"Signal duplique ignore: {signal['symbol']}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur validation signal: {e}")
            return False
    
    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """
        Verifie si un signal est un doublon
        
        Args:
            signal: Signal a verifier
            
        Returns:
            True si doublon
        """
        try:
            key = f"{signal['symbol']}_{signal['side']}_{signal['strategy']}"
            
            if key in self.last_signals:
                last_time = self.last_signals[key]
                elapsed = (datetime.now() - last_time).total_seconds()
                
                # Considerer comme doublon si < 60 secondes
                if elapsed < 60:
                    return True
            
            # Enregistrer ce signal
            self.last_signals[key] = datetime.now()
            
            return False
        
        except Exception as e:
            logger.error(f"Erreur check duplicate: {e}")
            return False
    
    def _send_signal(self, signal: Dict):
        """
        Envoie un signal au thread d'execution
        
        Args:
            signal: Signal a envoyer
        """
        try:
            # Ajouter a la queue locale
            self.signal_queue.put(signal, timeout=1)
            
            # Envoyer aussi au execution thread
            if hasattr(self.bot, 'execution_thread'):
                self.bot.execution_thread.add_signal(signal)
            
            logger.info(
                f"[SIGNAL] Signal genere: {signal['symbol']} {signal['side']} "
                f"par {signal['strategy']} (confiance: {signal['confidence']:.2%})"
            )
        
        except Exception as e:
            logger.error(f"Erreur envoi signal: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques
        
        Returns:
            Dict avec stats
        """
        stats = self.stats.copy()
        
        # Calculer moyennes
        for strategy_name, strategy_stats in stats['by_strategy'].items():
            total = strategy_stats['signals_generated']
            if total > 0:
                strategy_stats['acceptance_rate'] = (
                    strategy_stats['signals_accepted'] / total
                )
            else:
                strategy_stats['acceptance_rate'] = 0
        
        # Stats globales
        stats['is_running'] = self.is_running
        stats['active_strategies_count'] = len(self.active_strategies)
        stats['data_queue_size'] = self.data_queue.qsize()
        stats['signal_queue_size'] = self.signal_queue.qsize()
        
        if stats['signals_generated'] > 0:
            stats['signal_filter_rate'] = (
                stats['signals_filtered'] / 
                (stats['signals_generated'] + stats['signals_filtered'])
            )
        else:
            stats['signal_filter_rate'] = 0
        
        return stats
    
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """
        Retourne la performance par strategie
        
        Returns:
            Dict {strategy_name: performance}
        """
        performance = {}
        
        try:
            if hasattr(self.bot, 'strategy_manager'):
                for strategy_name in self.active_strategies:
                    perf = self.bot.strategy_manager.performance_by_strategy.get(strategy_name, {})
                    
                    performance[strategy_name] = {
                        'signals_generated': self.stats['by_strategy'][strategy_name]['signals_generated'],
                        'signals_accepted': self.stats['by_strategy'][strategy_name]['signals_accepted'],
                        'total_trades': perf.get('total_trades', 0),
                        'winning_trades': perf.get('winning_trades', 0),
                        'total_pnl': perf.get('total_pnl', 0),
                        'win_rate': perf.get('win_rate', 0),
                        'allocation': self.strategy_allocations.get(strategy_name, 0)
                    }
        
        except Exception as e:
            logger.error(f"Erreur get_strategy_performance: {e}")
        
        return performance
    
    def enable_strategy(self, strategy_name: str):
        """
        Active une strategie
        
        Args:
            strategy_name: Nom de la strategie
        """
        try:
            if hasattr(self.bot, 'strategy_manager'):
                # Activer dans le strategy manager
                if strategy_name in self.bot.strategy_manager.active_strategies:
                    self.active_strategies[strategy_name] = (
                        self.bot.strategy_manager.active_strategies[strategy_name]
                    )
                    logger.info(f"[OK] Strategie '{strategy_name}' activee")
        except Exception as e:
            logger.error(f"Erreur enable_strategy: {e}")
    
    def disable_strategy(self, strategy_name: str):
        """
        Desactive une strategie
        
        Args:
            strategy_name: Nom de la strategie
        """
        try:
            if strategy_name in self.active_strategies:
                del self.active_strategies[strategy_name]
                logger.info(f"Strategie '{strategy_name}' desactivee")
        except Exception as e:
            logger.error(f"Erreur disable_strategy: {e}")
    
    def clear_stats(self):
        """Reinitialise les statistiques"""
        self.stats = {
            'data_processed': 0,
            'signals_generated': 0,
            'signals_filtered': 0,
            'by_strategy': {
                name: {
                    'signals_generated': 0,
                    'signals_accepted': 0,
                    'avg_confidence': 0
                }
                for name in self.active_strategies
            },
            'last_update': None
        }
        logger.info("Statistiques de strategie reinitialisees")


class StrategyProcessor:
    """
    Processeur de strategies simplifie
    Peut etre utilise independamment du thread
    """
    
    def __init__(self, strategies: Dict):
        """
        Initialise le processeur
        
        Args:
            strategies: Dict des strategies {name: instance}
        """
        self.strategies = strategies
    
    def process_symbol(self, symbol: str, market_data: Dict) -> List[Dict]:
        """
        Traite un symbole avec toutes les strategies
        
        Args:
            symbol: Symbole
            market_data: Donnees de marche
            
        Returns:
            Liste des signaux generes
        """
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.analyze(market_data)
                
                if signal:
                    signal['strategy'] = strategy_name
                    signal['symbol'] = symbol
                    signal['timestamp'] = datetime.now()
                    signals.append(signal)
            
            except Exception as e:
                logger.error(f"Erreur {strategy_name} pour {symbol}: {e}")
        
        return signals
    
    def process_multiple(self, symbols: List[str], market_data_dict: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        Traite plusieurs symboles
        
        Args:
            symbols: Liste des symboles
            market_data_dict: Dict {symbol: market_data}
            
        Returns:
            Dict {symbol: [signals]}
        """
        results = {}
        
        for symbol in symbols:
            if symbol in market_data_dict:
                results[symbol] = self.process_symbol(symbol, market_data_dict[symbol])
        
        return results
