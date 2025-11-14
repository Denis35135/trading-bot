"""
Order Manager pour The Bot
Gestion complÃƒÂ¨te de l'exÃƒÂ©cution des ordres avec retry, slippage control, et smart routing
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue, PriorityQueue
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types d'ordres supportÃƒÂ©s"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"  # Post-only
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    OCO = "OCO"  # One-Cancels-Other


class OrderStatus(Enum):
    """Statuts des ordres"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderPriority(Enum):
    """PrioritÃƒÂ©s d'exÃƒÂ©cution"""
    CRITICAL = 1  # Stop loss, urgence
    HIGH = 2      # Signaux forts
    NORMAL = 3    # Ordres normaux
    LOW = 4       # Ordres non urgents


class Order:
    """Classe reprÃƒÂ©sentant un ordre"""
    
    def __init__(self,
                 symbol: str,
                 side: str,
                 order_type: OrderType,
                 quantity: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 client_order_id: Optional[str] = None):
        """
        CrÃƒÂ©e un ordre
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            side: BUY ou SELL
            order_type: Type d'ordre
            quantity: QuantitÃƒÂ©
            price: Prix limite (si applicable)
            stop_price: Prix stop (si applicable)
            client_order_id: ID client personnalisÃƒÂ©
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.client_order_id = client_order_id or f"{symbol}_{int(time.time()*1000)}"
        
        # Ãƒâ€°tat
        self.status = OrderStatus.PENDING
        self.exchange_order_id = None
        self.filled_quantity = 0
        self.average_price = 0
        self.commission = 0
        
        # Timing
        self.created_at = datetime.now()
        self.submitted_at = None
        self.filled_at = None
        
        # Metadata
        self.strategy = None
        self.priority = OrderPriority.NORMAL
        self.retries = 0
        self.max_retries = 3
        self.error_message = None


class OrderManager:
    """
    Gestionnaire d'ordres avec exÃƒÂ©cution robuste
    
    Features:
    - File de prioritÃƒÂ© pour exÃƒÂ©cution
    - Retry automatique avec backoff
    - Slippage control
    - Smart routing (market vs limit)
    - Order tracking et monitoring
    - Protection contre les erreurs
    """
    
    def __init__(self, exchange_client, config: Dict):
        """
        Initialise l'order manager
        
        Args:
            exchange_client: Client de l'exchange (Binance)
            config: Configuration
        """
        self.exchange = exchange_client
        self.config = config
        
        # Configuration exÃƒÂ©cution
        self.max_slippage = config.get('slippage_tolerance', 0.002)  # 0.2%
        self.order_timeout = config.get('order_timeout', 5000)  # 5 secondes
        self.retry_delay = config.get('retry_delay', 1000)  # 1 seconde
        self.max_retries = config.get('retry_attempts', 3)
        self.use_limit_orders = config.get('prefer_limit_orders', True)
        
        # Files d'ordres
        self.order_queue = PriorityQueue()
        self.pending_orders = {}
        self.active_orders = {}
        self.completed_orders = []
        
        # Ãƒâ€°tat
        self.is_running = False
        self.execution_thread = None
        self.monitoring_thread = None
        
        # Statistiques
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_slippage': 0,
            'average_execution_time': 0
        }
        
        # Callbacks
        self.callbacks = {
            'on_fill': [],
            'on_partial_fill': [],
            'on_cancel': [],
            'on_error': []
        }
        
        logger.info("Order Manager initialisÃƒÂ©")
    
    def start(self):
        """DÃƒÂ©marre l'order manager"""
        if self.is_running:
            logger.warning("Order Manager dÃƒÂ©jÃƒÂ  en cours d'exÃƒÂ©cution")
            return
        
        self.is_running = True
        
        # Thread d'exÃƒÂ©cution
        self.execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True
        )
        self.execution_thread.start()
        
        # Thread de monitoring
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Order Manager dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte l'order manager"""
        self.is_running = False
        
        # Attendre que les threads se terminent
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Annuler les ordres pending
        self._cancel_all_pending_orders()
        
        logger.info("Order Manager arrÃƒÂªtÃƒÂ©")
    
    def submit_order(self,
                     symbol: str,
                     side: str,
                     quantity: float,
                     order_type: str = "MARKET",
                     price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     priority: OrderPriority = OrderPriority.NORMAL,
                     strategy: Optional[str] = None) -> Order:
        """
        Soumet un ordre pour exÃƒÂ©cution
        
        Args:
            symbol: Symbole
            side: BUY ou SELL
            quantity: QuantitÃƒÂ©
            order_type: Type d'ordre (MARKET, LIMIT, etc.)
            price: Prix limite
            stop_price: Prix stop
            priority: PrioritÃƒÂ© d'exÃƒÂ©cution
            strategy: Nom de la stratÃƒÂ©gie
            
        Returns:
            Objet Order crÃƒÂ©ÃƒÂ©
        """
        # CrÃƒÂ©er l'ordre
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType[order_type],
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        order.priority = priority
        order.strategy = strategy
        
        # Validation
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            order.error_message = "Validation ÃƒÂ©chouÃƒÂ©e"
            logger.error(f"Ordre rejetÃƒÂ©: {order.client_order_id}")
            return order
        
        # Smart routing
        if self.use_limit_orders and order.order_type == OrderType.MARKET:
            order = self._convert_to_limit_order(order)
        
        # Ajouter ÃƒÂ  la queue
        self.order_queue.put((order.priority.value, time.time(), order))
        self.pending_orders[order.client_order_id] = order
        
        self.stats['total_orders'] += 1
        
        logger.info(f"Ordre soumis: {order.side} {order.quantity:.6f} {order.symbol} "
                   f"@ {order.price if order.price else 'MARKET'}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre
        
        Args:
            order_id: ID de l'ordre (client ou exchange)
            
        Returns:
            True si annulation rÃƒÂ©ussie
        """
        try:
            # Trouver l'ordre
            order = None
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
            elif order_id in self.active_orders:
                order = self.active_orders[order_id]
            else:
                # Chercher par exchange ID
                for o in self.active_orders.values():
                    if o.exchange_order_id == order_id:
                        order = o
                        break
            
            if not order:
                logger.error(f"Ordre {order_id} non trouvÃƒÂ©")
                return False
            
            # Annuler sur l'exchange
            if order.exchange_order_id:
                success = self.exchange.cancel_order(
                    symbol=order.symbol,
                    order_id=order.exchange_order_id
                )
                
                if success:
                    order.status = OrderStatus.CANCELLED
                    self._move_to_completed(order)
                    self._trigger_callbacks('on_cancel', order)
                    logger.info(f"Ordre {order_id} annulÃƒÂ©")
                    return True
            else:
                # Ordre pas encore soumis
                order.status = OrderStatus.CANCELLED
                if order.client_order_id in self.pending_orders:
                    del self.pending_orders[order.client_order_id]
                return True
                
        except Exception as e:
            logger.error(f"Erreur annulation ordre {order_id}: {e}")
        
        return False
    
    def submit_oco_order(self,
                        symbol: str,
                        side: str,
                        quantity: float,
                        take_profit_price: float,
                        stop_loss_price: float,
                        strategy: Optional[str] = None) -> Tuple[Order, Order]:
        """
        Soumet un ordre OCO (One-Cancels-Other)
        
        Args:
            symbol: Symbole
            side: BUY ou SELL
            quantity: QuantitÃƒÂ©
            take_profit_price: Prix take profit
            stop_loss_price: Prix stop loss
            strategy: StratÃƒÂ©gie
            
        Returns:
            Tuple (ordre TP, ordre SL)
        """
        # CrÃƒÂ©er les deux ordres
        tp_order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT,
            quantity=quantity,
            price=take_profit_price
        )
        tp_order.strategy = strategy
        
        sl_order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_loss_price
        )
        sl_order.strategy = strategy
        sl_order.priority = OrderPriority.CRITICAL
        
        # Lier les ordres
        tp_order.linked_order = sl_order.client_order_id
        sl_order.linked_order = tp_order.client_order_id
        
        # Soumettre
        self.order_queue.put((tp_order.priority.value, time.time(), tp_order))
        self.order_queue.put((sl_order.priority.value, time.time(), sl_order))
        
        self.pending_orders[tp_order.client_order_id] = tp_order
        self.pending_orders[sl_order.client_order_id] = sl_order
        
        logger.info(f"OCO soumis: {symbol} TP@{take_profit_price:.2f} SL@{stop_loss_price:.2f}")
        
        return tp_order, sl_order
    
    def _execution_loop(self):
        """Boucle principale d'exÃƒÂ©cution des ordres"""
        while self.is_running:
            try:
                # RÃƒÂ©cupÃƒÂ©rer le prochain ordre
                if not self.order_queue.empty():
                    priority, timestamp, order = self.order_queue.get(timeout=0.1)
                    
                    # ExÃƒÂ©cuter l'ordre
                    self._execute_order(order)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Erreur execution loop: {e}")
                time.sleep(1)
    
    def _execute_order(self, order: Order):
        """
        ExÃƒÂ©cute un ordre
        
        Args:
            order: L'ordre ÃƒÂ  exÃƒÂ©cuter
        """
        try:
            logger.info(f"ExÃƒÂ©cution ordre: {order.client_order_id}")
            
            # Mettre ÃƒÂ  jour le statut
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            # DÃƒÂ©placer vers ordres actifs
            if order.client_order_id in self.pending_orders:
                del self.pending_orders[order.client_order_id]
            self.active_orders[order.client_order_id] = order
            
            # ExÃƒÂ©cuter selon le type
            if order.order_type == OrderType.MARKET:
                result = self._execute_market_order(order)
            elif order.order_type in [OrderType.LIMIT, OrderType.LIMIT_MAKER]:
                result = self._execute_limit_order(order)
            elif order.order_type == OrderType.STOP_LOSS:
                result = self._execute_stop_loss_order(order)
            elif order.order_type == OrderType.TAKE_PROFIT:
                result = self._execute_take_profit_order(order)
            else:
                logger.error(f"Type d'ordre non supportÃƒÂ©: {order.order_type}")
                result = None
            
            # Traiter le rÃƒÂ©sultat
            if result:
                self._process_execution_result(order, result)
            else:
                self._handle_execution_failure(order)
                
        except Exception as e:
            logger.error(f"Erreur exÃƒÂ©cution ordre {order.client_order_id}: {e}")
            order.error_message = str(e)
            self._handle_execution_failure(order)
    
    def _execute_market_order(self, order: Order) -> Optional[Dict]:
        """ExÃƒÂ©cute un ordre au marchÃƒÂ©"""
        try:
            result = self.exchange.place_market_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity
            )
            
            if result:
                order.exchange_order_id = result.get('order_id')
                return result
                
        except Exception as e:
            logger.error(f"Erreur market order: {e}")
            order.error_message = str(e)
        
        return None
    
    def _execute_limit_order(self, order: Order) -> Optional[Dict]:
        """ExÃƒÂ©cute un ordre limite"""
        try:
            # Si pas de prix, utiliser le meilleur prix disponible
            if not order.price:
                ticker = self.exchange.get_symbol_ticker(order.symbol)
                if order.side == "BUY":
                    order.price = ticker['bid']
                else:
                    order.price = ticker['ask']
            
            result = self.exchange.place_limit_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price
            )
            
            if result:
                order.exchange_order_id = result.get('order_id')
                return result
                
        except Exception as e:
            logger.error(f"Erreur limit order: {e}")
            order.error_message = str(e)
        
        return None
    
    def _execute_stop_loss_order(self, order: Order) -> Optional[Dict]:
        """
        ExÃƒÂ©cute un ordre stop loss
        
        Pour Binance, on peut soit:
        1. Utiliser un ordre STOP_LOSS_LIMIT natif
        2. Surveiller le prix et dÃƒÂ©clencher un market order
        
        Pour l'instant, implÃƒÂ©mentation basique avec surveillance
        """
        try:
            # Option 1: ImplÃƒÂ©menter plus tard avec surveillance
            logger.warning(f"Stop loss orders en dÃƒÂ©veloppement pour {order.symbol}")
            
            # Pour l'instant, stocker pour monitoring manuel
            order.status = OrderStatus.SUBMITTED
            order.monitoring_price = order.stop_price
            
            # Ajouter ÃƒÂ  une liste de surveillance
            if not hasattr(self, 'stop_loss_orders'):
                self.stop_loss_orders = []
            self.stop_loss_orders.append(order)
            
            logger.info(f"Stop Loss configurÃƒÂ©: {order.symbol} @ {order.stop_price}")
            
            return {
                'order_id': order.client_order_id,
                'status': 'MONITORING',
                'stop_price': order.stop_price,
                'message': 'Stop loss en surveillance'
            }
            
        except Exception as e:
            logger.error(f"Erreur configuration stop loss: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return None
    
    def _execute_take_profit_order(self, order: Order) -> Optional[Dict]:
        """ExÃƒÂ©cute un ordre take profit"""
        # Similaire au stop loss
        order.status = OrderStatus.SUBMITTED
        logger.info(f"Take Profit configurÃƒÂ©: {order.symbol} @ {order.price}")
        
        return {'order_id': order.client_order_id, 'status': 'MONITORING'}
    
    def _monitoring_loop(self):
        """Boucle de monitoring des ordres actifs"""
        while self.is_running:
            try:
                # Copier la liste pour ÃƒÂ©viter modifications concurrentes
                active_orders = list(self.active_orders.values())
                
                for order in active_orders:
                    # VÃƒÂ©rifier statut
                    if order.exchange_order_id:
                        self._check_order_status(order)
                    
                    # VÃƒÂ©rifier stop loss / take profit
                    if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                        self._check_trigger_conditions(order)
                    
                    # VÃƒÂ©rifier timeout
                    if order.status == OrderStatus.SUBMITTED:
                        self._check_order_timeout(order)
                
                time.sleep(1)  # Check toutes les secondes
                
            except Exception as e:
                logger.error(f"Erreur monitoring loop: {e}")
                time.sleep(1)
    
    def _check_order_status(self, order: Order):
        """VÃƒÂ©rifie le statut d'un ordre sur l'exchange"""
        try:
            if not order.exchange_order_id:
                return
            
            status = self.exchange.get_order_status(
                symbol=order.symbol,
                order_id=order.exchange_order_id
            )
            
            if status:
                # Mettre ÃƒÂ  jour l'ordre
                if status['status'] == 'FILLED':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = status.get('filled_qty', order.quantity)
                    order.filled_at = datetime.now()
                    
                    # Calculer slippage si market order
                    if order.order_type == OrderType.MARKET and order.price:
                        actual_price = status.get('average_price', order.price)
                        slippage = abs(actual_price - order.price) / order.price
                        self.stats['total_slippage'] += slippage
                    
                    self._move_to_completed(order)
                    self._trigger_callbacks('on_fill', order)
                    
                elif status['status'] == 'PARTIALLY_FILLED':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = status.get('filled_qty', 0)
                    self._trigger_callbacks('on_partial_fill', order)
                    
                elif status['status'] == 'CANCELLED':
                    order.status = OrderStatus.CANCELLED
                    self._move_to_completed(order)
                    self._trigger_callbacks('on_cancel', order)
                    
        except Exception as e:
            logger.error(f"Erreur check status {order.client_order_id}: {e}")
    
    def _check_trigger_conditions(self, order: Order):
        """VÃƒÂ©rifie les conditions de dÃƒÂ©clenchement pour SL/TP"""
        try:
            # RÃƒÂ©cupÃƒÂ©rer le prix actuel
            ticker = self.exchange.get_symbol_ticker(order.symbol)
            if not ticker:
                return
            
            current_price = ticker['price']
            
            # Stop Loss
            if order.order_type == OrderType.STOP_LOSS:
                if order.side == "SELL" and current_price <= order.stop_price:
                    # DÃƒÂ©clencher vente
                    self._trigger_stop_order(order, current_price)
                elif order.side == "BUY" and current_price >= order.stop_price:
                    # DÃƒÂ©clencher achat
                    self._trigger_stop_order(order, current_price)
            
            # Take Profit
            elif order.order_type == OrderType.TAKE_PROFIT:
                if order.side == "SELL" and current_price >= order.price:
                    # DÃƒÂ©clencher vente
                    self._trigger_stop_order(order, current_price)
                elif order.side == "BUY" and current_price <= order.price:
                    # DÃƒÂ©clencher achat
                    self._trigger_stop_order(order, current_price)
                    
        except Exception as e:
            logger.error(f"Erreur check trigger {order.client_order_id}: {e}")
    
    def _trigger_stop_order(self, order: Order, trigger_price: float):
        """DÃƒÂ©clenche un ordre stop"""
        logger.info(f"DÃƒÂ©clenchement {order.order_type.value}: {order.symbol} @ {trigger_price}")
        
        # Convertir en market order
        market_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.MARKET,
            quantity=order.quantity
        )
        market_order.priority = OrderPriority.CRITICAL
        market_order.strategy = order.strategy
        
        # ExÃƒÂ©cuter immÃƒÂ©diatement
        result = self._execute_market_order(market_order)
        
        if result:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.average_price = trigger_price
            
            # Si OCO, annuler l'ordre liÃƒÂ©
            if hasattr(order, 'linked_order'):
                self.cancel_order(order.linked_order)
            
            self._move_to_completed(order)
            self._trigger_callbacks('on_fill', order)
    
    def _check_order_timeout(self, order: Order):
        """VÃƒÂ©rifie si un ordre a timeout"""
        if not order.submitted_at:
            return
        
        elapsed = (datetime.now() - order.submitted_at).total_seconds() * 1000
        
        if elapsed > self.order_timeout:
            logger.warning(f"Timeout ordre {order.client_order_id}")
            
            # Retry ou annuler
            if order.retries < self.max_retries:
                order.retries += 1
                logger.info(f"Retry {order.retries}/{self.max_retries}")
                self._execute_order(order)
            else:
                self.cancel_order(order.client_order_id)
    
    def _validate_order(self, order: Order) -> bool:
        """Valide un ordre avant exÃƒÂ©cution"""
        # VÃƒÂ©rifier symbole
        if order.symbol not in self.exchange.symbols_info:
            logger.error(f"Symbole invalide: {order.symbol}")
            return False
        
        # VÃƒÂ©rifier quantitÃƒÂ© minimum
        min_qty = self.exchange.symbols_info[order.symbol]['min_qty']
        if order.quantity < min_qty:
            logger.error(f"QuantitÃƒÂ© {order.quantity} < minimum {min_qty}")
            return False
        
        # VÃƒÂ©rifier notional minimum
        if order.price:
            notional = order.quantity * order.price
            min_notional = self.exchange.symbols_info[order.symbol]['min_notional']
            if notional < min_notional:
                logger.error(f"Notional {notional} < minimum {min_notional}")
                return False
        
        return True
    
    def _convert_to_limit_order(self, order: Order) -> Order:
        """Convertit un ordre market en limit pour ÃƒÂ©viter le slippage"""
        try:
            ticker = self.exchange.get_symbol_ticker(order.symbol)
            
            if order.side == "BUY":
                # Utiliser ask + petit premium
                order.price = ticker['ask'] * (1 + self.max_slippage)
            else:
                # Utiliser bid - petit discount
                order.price = ticker['bid'] * (1 - self.max_slippage)
            
            order.order_type = OrderType.LIMIT
            logger.info(f"Conversion MARKET Ã¢â€ â€™ LIMIT @ {order.price:.4f}")
            
        except Exception as e:
            logger.error(f"Erreur conversion limit: {e}")
        
        return order
    
    def _process_execution_result(self, order: Order, result: Dict):
        """Traite le rÃƒÂ©sultat d'une exÃƒÂ©cution"""
        if result.get('status') == 'FILLED':
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_at = datetime.now()
            
            # Extraire prix moyen des fills
            if 'fills' in result:
                total_qty = 0
                total_value = 0
                for fill in result['fills']:
                    qty = float(fill.get('qty', 0))
                    price = float(fill.get('price', 0))
                    total_qty += qty
                    total_value += qty * price
                
                if total_qty > 0:
                    order.average_price = total_value / total_qty
            
            self.stats['successful_orders'] += 1
            self._move_to_completed(order)
            self._trigger_callbacks('on_fill', order)
            
            logger.info(f"Ã¢Å“â€¦ Ordre exÃƒÂ©cutÃƒÂ©: {order.client_order_id}")
        else:
            # Ordre soumis mais pas encore rempli
            logger.info(f"Ordre soumis: {order.client_order_id}")
    
    def _handle_execution_failure(self, order: Order):
        """GÃƒÂ¨re l'ÃƒÂ©chec d'exÃƒÂ©cution d'un ordre"""
        order.retries += 1
        
        if order.retries < self.max_retries:
            # Retry aprÃƒÂ¨s dÃƒÂ©lai
            logger.info(f"Retry {order.retries}/{self.max_retries} aprÃƒÂ¨s {self.retry_delay}ms")
            time.sleep(self.retry_delay / 1000)
            
            # Remettre dans la queue
            self.order_queue.put((order.priority.value, time.time(), order))
        else:
            # Ãƒâ€°chec dÃƒÂ©finitif
            order.status = OrderStatus.REJECTED
            self.stats['failed_orders'] += 1
            self._move_to_completed(order)
            self._trigger_callbacks('on_error', order)
            
            logger.error(f"Ã¢ÂÅ’ Ordre ÃƒÂ©chouÃƒÂ© aprÃƒÂ¨s {self.max_retries} tentatives: {order.client_order_id}")
    
    def _move_to_completed(self, order: Order):
        """DÃƒÂ©place un ordre vers les ordres complÃƒÂ©tÃƒÂ©s"""
        if order.client_order_id in self.active_orders:
            del self.active_orders[order.client_order_id]
        
        self.completed_orders.append(order)
        
        # Limiter l'historique
        if len(self.completed_orders) > 1000:
            self.completed_orders = self.completed_orders[-1000:]
    
    def _cancel_all_pending_orders(self):
        """Annule tous les ordres en attente"""
        for order in list(self.pending_orders.values()):
            self.cancel_order(order.client_order_id)
    
    def _trigger_callbacks(self, event: str, order: Order):
        """DÃƒÂ©clenche les callbacks pour un ÃƒÂ©vÃƒÂ©nement"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Erreur callback {event}: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """
        Enregistre un callback pour un ÃƒÂ©vÃƒÂ©nement
        
        Args:
            event: Nom de l'ÃƒÂ©vÃƒÂ©nement (on_fill, on_error, etc.)
            callback: Fonction ÃƒÂ  appeler
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques d'exÃƒÂ©cution"""
        success_rate = 0
        if self.stats['total_orders'] > 0:
            success_rate = self.stats['successful_orders'] / self.stats['total_orders']
        
        avg_slippage = 0
        if self.stats['successful_orders'] > 0:
            avg_slippage = self.stats['total_slippage'] / self.stats['successful_orders']
        
        return {
            'total_orders': self.stats['total_orders'],
            'successful': self.stats['successful_orders'],
            'failed': self.stats['failed_orders'],
            'success_rate': success_rate,
            'average_slippage': avg_slippage,
            'pending': len(self.pending_orders),
            'active': len(self.active_orders),
            'completed': len(self.completed_orders)
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de l'order manager"""
    
    # Mock exchange client
    class MockExchange:
        def __init__(self):
            self.symbols_info = {
                'BTCUSDC': {
                    'min_qty': 0.00001,
                    'min_notional': 10
                }
            }
        
        def get_symbol_ticker(self, symbol):
            return {
                'price': 50000,
                'bid': 49999,
                'ask': 50001
            }
        
        def place_market_order(self, **kwargs):
            return {
                'order_id': f"TEST_{int(time.time())}",
                'status': 'FILLED'
            }
        
        def place_limit_order(self, **kwargs):
            return {
                'order_id': f"TEST_{int(time.time())}",
                'status': 'SUBMITTED'
            }
        
        def get_order_status(self, **kwargs):
            return {
                'status': 'FILLED',
                'filled_qty': 0.001
            }
        
        def cancel_order(self, **kwargs):
            return True
    
    # Configuration
    config = {
        'slippage_tolerance': 0.002,
        'order_timeout': 5000,
        'retry_attempts': 3
    }
    
    # Initialiser
    exchange = MockExchange()
    manager = OrderManager(exchange, config)
    
    # Callbacks
    def on_fill(order):
        print(f"Ã¢Å“â€¦ Ordre rempli: {order.client_order_id}")
    
    def on_error(order):
        print(f"Ã¢ÂÅ’ Erreur ordre: {order.client_order_id} - {order.error_message}")
    
    manager.register_callback('on_fill', on_fill)
    manager.register_callback('on_error', on_error)
    
    # DÃƒÂ©marrer
    manager.start()
    
    # Test 1: Ordre market
    print("=" * 50)
    print("TEST 1: Ordre Market")
    order1 = manager.submit_order(
        symbol="BTCUSDC",
        side="BUY",
        quantity=0.001,
        order_type="MARKET"
    )
    print(f"Ordre soumis: {order1.client_order_id}")
    
    # Test 2: Ordre limit
    print("\n" + "=" * 50)
    print("TEST 2: Ordre Limit")
    order2 = manager.submit_order(
        symbol="BTCUSDC",
        side="SELL",
        quantity=0.001,
        order_type="LIMIT",
        price=51000
    )
    print(f"Ordre soumis: {order2.client_order_id}")
    
    # Test 3: OCO
    print("\n" + "=" * 50)
    print("TEST 3: Ordre OCO")
    tp_order, sl_order = manager.submit_oco_order(
        symbol="BTCUSDC",
        side="SELL",
        quantity=0.001,
        take_profit_price=52000,
        stop_loss_price=49000
    )
    print(f"TP: {tp_order.client_order_id}, SL: {sl_order.client_order_id}")
    
    # Attendre un peu
    time.sleep(2)
    
    # Stats
    print("\n" + "=" * 50)
    print("STATISTIQUES:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # ArrÃƒÂªter
    manager.stop()
    print("\nÃ¢Å“â€¦ Test terminÃƒÂ©")
