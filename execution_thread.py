"""
Execution Thread pour The Bot
Thread d'exÃƒÂ©cution des ordres avec gestion intelligente
"""

import time
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)


class ExecutionThread:
    """
    Thread d'exÃƒÂ©cution des ordres
    
    ResponsabilitÃƒÂ©s:
    - Consommer les signaux de trading depuis la queue
    - Valider les signaux avec le risk manager
    - Calculer la taille des positions
    - ExÃƒÂ©cuter les ordres avec retry
    - Tracker les positions ouvertes
    """
    
    def __init__(self, bot_instance):
        """
        Initialise le thread d'exÃƒÂ©cution
        
        Args:
            bot_instance: Instance du bot principal
        """
        self.bot = bot_instance
        self.is_running = False
        self.thread = None
        
        # Queues
        self.signal_queue = Queue(maxsize=100)
        
        # Ãƒâ€°tat
        self.pending_orders = {}
        self.execution_stats = {
            'total_signals': 0,
            'signals_approved': 0,
            'signals_rejected': 0,
            'orders_executed': 0,
            'orders_failed': 0,
            'total_execution_time_ms': 0
        }
        
        logger.info("Execution Thread initialisÃƒÂ©")
    
    def start(self):
        """DÃƒÂ©marre le thread"""
        if self.is_running:
            logger.warning("Execution Thread dÃƒÂ©jÃƒÂ  en cours")
            return
        
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ExecutionThread"
        )
        self.thread.start()
        
        logger.info("Ã¢Å“â€¦ Execution Thread dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte le thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Execution Thread arrÃƒÂªtÃƒÂ©")
    
    def add_signal(self, signal: Dict):
        """
        Ajoute un signal ÃƒÂ  la queue d'exÃƒÂ©cution
        
        Args:
            signal: Signal de trading ÃƒÂ  exÃƒÂ©cuter
        """
        try:
            self.signal_queue.put(signal, timeout=1)
            logger.debug(f"Signal ajoutÃƒÂ© ÃƒÂ  la queue: {signal['symbol']} {signal['side']}")
        except Exception as e:
            logger.error(f"Erreur ajout signal: {e}")
    
    def _run(self):
        """Boucle principale du thread"""
        logger.info("Ã°Å¸â€â€ž Execution Thread running...")
        
        while self.is_running:
            try:
                # RÃƒÂ©cupÃƒÂ©rer un signal (timeout 1 seconde)
                try:
                    signal = self.signal_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Traiter le signal
                self._process_signal(signal)
                
            except Exception as e:
                logger.error(f"Erreur dans execution thread: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("Execution Thread terminÃƒÂ©")
    
    def _process_signal(self, signal: Dict):
        """
        Traite un signal de trading
        
        Args:
            signal: Signal ÃƒÂ  traiter
        """
        start_time = time.time()
        self.execution_stats['total_signals'] += 1
        
        try:
            symbol = signal['symbol']
            side = signal['side']
            
            logger.info(f"Ã°Å¸â€œÅ  Traitement signal: {symbol} {side} (confiance: {signal['confidence']:.2%})")
            
            # 1. VÃƒÂ©rification avec Risk Manager
            if not self._validate_with_risk_manager(signal):
                logger.warning(f"Ã¢ÂÅ’ Signal rejetÃƒÂ© par risk manager: {symbol}")
                self.execution_stats['signals_rejected'] += 1
                return
            
            self.execution_stats['signals_approved'] += 1
            
            # 2. Calcul de la taille de position
            position_size = self._calculate_position_size(signal)
            if not position_size or position_size <= 0:
                logger.warning(f"Ã¢ÂÅ’ Taille de position invalide: {position_size}")
                return
            
            # 3. ExÃƒÂ©cution de l'ordre
            order_result = self._execute_order(signal, position_size)
            
            if order_result['success']:
                logger.info(
                    f"Ã¢Å“â€¦ Ordre exÃƒÂ©cutÃƒÂ©: {symbol} {side} "
                    f"{position_size:.6f} @ ${order_result['price']:.2f}"
                )
                self.execution_stats['orders_executed'] += 1
                
                # Notifier le succÃƒÂ¨s
                self._notify_execution_success(signal, order_result)
            else:
                logger.error(f"Ã¢ÂÅ’ Ãƒâ€°chec exÃƒÂ©cution: {order_result.get('error')}")
                self.execution_stats['orders_failed'] += 1
        
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}", exc_info=True)
            self.execution_stats['orders_failed'] += 1
        
        finally:
            # Mesurer le temps d'exÃƒÂ©cution
            elapsed_ms = (time.time() - start_time) * 1000
            self.execution_stats['total_execution_time_ms'] += elapsed_ms
            logger.debug(f"Ã¢ÂÂ±Ã¯Â¸Â Signal traitÃƒÂ© en {elapsed_ms:.0f}ms")
    
    def _validate_with_risk_manager(self, signal: Dict) -> bool:
        """
        Valide un signal avec le risk manager
        
        Args:
            signal: Signal ÃƒÂ  valider
            
        Returns:
            True si approuvÃƒÂ©
        """
        try:
            if not hasattr(self.bot, 'risk_monitor'):
                logger.warning("Risk monitor non disponible, validation bypassÃƒÂ©e")
                return True
            
            # Demander l'approbation
            approved = self.bot.risk_monitor.approve_trade(signal)
            
            if not approved:
                logger.info(f"Signal rejetÃƒÂ©: {signal['symbol']} - Raison: limites de risque")
            
            return approved
            
        except Exception as e:
            logger.error(f"Erreur validation risk manager: {e}")
            return False
    
    def _calculate_position_size(self, signal: Dict) -> Optional[float]:
        """
        Calcule la taille de position
        
        Args:
            signal: Signal de trading
            
        Returns:
            Taille de la position ou None
        """
        try:
            if not hasattr(self.bot, 'position_sizer'):
                logger.error("Position sizer non disponible")
                return None
            
            # Calculer la taille
            position_size = self.bot.position_sizer.calculate(
                symbol=signal['symbol'],
                side=signal['side'],
                entry_price=signal['price'],
                stop_loss=signal.get('stop_loss'),
                confidence=signal.get('confidence', 1.0)
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Erreur calcul position size: {e}")
            return None
    
    def _execute_order(self, signal: Dict, quantity: float) -> Dict[str, Any]:
        """
        ExÃƒÂ©cute un ordre
        
        Args:
            signal: Signal de trading
            quantity: QuantitÃƒÂ© ÃƒÂ  trader
            
        Returns:
            Dict avec rÃƒÂ©sultat
        """
        try:
            if not hasattr(self.bot, 'order_manager'):
                logger.error("Order manager non disponible")
                return {'success': False, 'error': 'No order manager'}
            
            # PrÃƒÂ©parer les paramÃƒÂ¨tres d'ordre
            order_params = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'quantity': quantity,
                'order_type': signal.get('order_type', 'MARKET'),
                'price': signal.get('price') if signal.get('order_type') == 'LIMIT' else None,
                'strategy': signal.get('strategy')
            }
            
            # Soumettre l'ordre
            order = self.bot.order_manager.submit_order(**order_params)
            
            # Attendre la confirmation (max 5 secondes)
            max_wait = 5
            waited = 0
            while waited < max_wait:
                if order.status in ['FILLED', 'REJECTED', 'CANCELED', 'EXPIRED']:
                    break
                time.sleep(0.1)
                waited += 0.1
            
            # RÃƒÂ©sultat
            if order.status == 'FILLED':
                return {
                    'success': True,
                    'order_id': order.client_order_id,
                    'price': order.average_price or signal['price'],
                    'quantity': order.filled_quantity,
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'success': False,
                    'error': f"Order status: {order.status}",
                    'order_id': order.client_order_id
                }
        
        except Exception as e:
            logger.error(f"Erreur exÃƒÂ©cution ordre: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _notify_execution_success(self, signal: Dict, result: Dict):
        """
        Notifie le succÃƒÂ¨s d'une exÃƒÂ©cution
        
        Args:
            signal: Signal exÃƒÂ©cutÃƒÂ©
            result: RÃƒÂ©sultat de l'exÃƒÂ©cution
        """
        try:
            # Notifier le strategy manager
            if hasattr(self.bot, 'strategy_manager'):
                self.bot.strategy_manager.on_order_filled(
                    signal=signal,
                    execution_result=result
                )
            
            # Envoyer notification si configurÃƒÂ©
            if hasattr(self.bot, 'notification_manager'):
                self.bot.notification_manager.notify_trade(
                    symbol=signal['symbol'],
                    side=signal['side'],
                    price=result['price'],
                    quantity=result['quantity']
                )
        
        except Exception as e:
            logger.error(f"Erreur notification: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'exÃƒÂ©cution
        
        Returns:
            Dict avec stats
        """
        stats = self.execution_stats.copy()
        
        # Calculer moyennes
        if stats['orders_executed'] > 0:
            stats['avg_execution_time_ms'] = (
                stats['total_execution_time_ms'] / stats['orders_executed']
            )
        else:
            stats['avg_execution_time_ms'] = 0
        
        if stats['total_signals'] > 0:
            stats['approval_rate'] = stats['signals_approved'] / stats['total_signals']
            stats['success_rate'] = stats['orders_executed'] / stats['total_signals']
        else:
            stats['approval_rate'] = 0
            stats['success_rate'] = 0
        
        stats['queue_size'] = self.signal_queue.qsize()
        stats['is_running'] = self.is_running
        
        return stats
    
    def get_pending_orders_count(self) -> int:
        """Retourne le nombre d'ordres en attente"""
        return len(self.pending_orders)
    
    def clear_stats(self):
        """RÃƒÂ©initialise les statistiques"""
        self.execution_stats = {
            'total_signals': 0,
            'signals_approved': 0,
            'signals_rejected': 0,
            'orders_executed': 0,
            'orders_failed': 0,
            'total_execution_time_ms': 0
        }
        logger.info("Statistiques d'exÃƒÂ©cution rÃƒÂ©initialisÃƒÂ©es")


class OrderExecutor:
    """
    ExÃƒÂ©cuteur d'ordres avec retry et gestion d'erreurs
    UtilisÃƒÂ© par ExecutionThread
    """
    
    def __init__(self, exchange_client, max_retries: int = 3):
        """
        Initialise l'exÃƒÂ©cuteur
        
        Args:
            exchange_client: Client d'exchange
            max_retries: Nombre max de tentatives
        """
        self.exchange = exchange_client
        self.max_retries = max_retries
    
    def execute_with_retry(self, order_params: Dict) -> Dict[str, Any]:
        """
        ExÃƒÂ©cute un ordre avec retry automatique
        
        Args:
            order_params: ParamÃƒÂ¨tres de l'ordre
            
        Returns:
            Dict avec rÃƒÂ©sultat
        """
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Tentative {attempt}/{self.max_retries}")
                
                # ExÃƒÂ©cuter l'ordre
                result = self.exchange.create_order(**order_params)
                
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Tentative {attempt} ÃƒÂ©chouÃƒÂ©e: {e}")
                
                if attempt < self.max_retries:
                    # Attendre avant retry (backoff exponentiel)
                    wait_time = 2 ** (attempt - 1)
                    time.sleep(wait_time)
        
        # Toutes les tentatives ont ÃƒÂ©chouÃƒÂ©
        return {
            'success': False,
            'error': str(last_error),
            'attempts': self.max_retries
        }