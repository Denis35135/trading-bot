"""
Execution Engine pour The Bot
Moteur d'exÃƒÂ©cution intelligent des ordres
"""

import time
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Statut d'exÃƒÂ©cution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class ExecutionEngine:
    """
    Moteur d'exÃƒÂ©cution intelligent
    
    ResponsabilitÃƒÂ©s:
    - ExÃƒÂ©cuter les ordres avec retry intelligent
    - GÃƒÂ©rer le slippage
    - Split des ordres importants
    - Tracking de l'exÃƒÂ©cution
    - Circuit breaker si trop d'ÃƒÂ©checs
    """
    
    def __init__(self, order_manager, config: Optional[Dict] = None):
        """
        Initialise l'execution engine
        
        Args:
            order_manager: OrderManager pour passer les ordres
            config: Configuration de l'exÃƒÂ©cution
        """
        self.order_manager = order_manager
        self.config = config or {}
        
        # ParamÃƒÂ¨tres d'exÃƒÂ©cution
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        self.max_slippage = self.config.get('max_slippage', 0.002)  # 0.2%
        self.order_timeout = self.config.get('order_timeout', 5)  # secondes
        
        # Split orders
        self.enable_order_splitting = self.config.get('enable_order_splitting', True)
        self.min_split_size = self.config.get('min_split_size', 1000)  # USDC
        self.max_chunk_size = self.config.get('max_chunk_size', 500)  # USDC
        
        # Ãƒâ€°tat
        self.active_executions = {}  # {execution_id: execution_data}
        self.execution_history = []
        
        # Statistiques
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'cancelled_executions': 0,
            'total_retries': 0,
            'avg_execution_time_ms': 0,
            'total_slippage_pct': 0
        }
        
        logger.info("Ã¢Å“â€¦ Execution Engine initialisÃƒÂ©")
    
    def execute(self, 
               symbol: str,
               side: str,
               quantity: float,
               order_type: str = 'MARKET',
               price: Optional[float] = None,
               stop_loss: Optional[float] = None,
               take_profit: Optional[float] = None,
               metadata: Optional[Dict] = None) -> Dict:
        """
        ExÃƒÂ©cute un ordre avec gestion intelligente
        
        Args:
            symbol: Symbole ÃƒÂ  trader
            side: BUY ou SELL
            quantity: QuantitÃƒÂ©
            order_type: MARKET ou LIMIT
            price: Prix limite (si LIMIT)
            stop_loss: Prix de stop loss
            take_profit: Prix de take profit
            metadata: MÃƒÂ©tadonnÃƒÂ©es optionnelles
            
        Returns:
            Dict avec le rÃƒÂ©sultat de l'exÃƒÂ©cution
        """
        execution_id = self._generate_execution_id()
        start_time = time.time()
        
        execution = {
            'id': execution_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': ExecutionStatus.PENDING,
            'start_time': datetime.now(),
            'metadata': metadata or {},
            'orders': [],
            'retries': 0
        }
        
        self.active_executions[execution_id] = execution
        self.stats['total_executions'] += 1
        
        logger.info(f"Ã°Å¸Å½Â¯ ExÃƒÂ©cution {execution_id}: {side} {quantity} {symbol}")
        
        try:
            # DÃƒÂ©cider si on doit splitter l'ordre
            if self._should_split_order(quantity, price or 0):
                result = self._execute_with_splitting(execution)
            else:
                result = self._execute_single_order(execution)
            
            # Calculer le temps d'exÃƒÂ©cution
            execution_time = (time.time() - start_time) * 1000  # ms
            result['execution_time_ms'] = execution_time
            
            # Mettre ÃƒÂ  jour les stats
            self._update_stats(result, execution_time)
            
            # Sauvegarder dans l'historique
            execution['result'] = result
            execution['end_time'] = datetime.now()
            self.execution_history.append(execution)
            
            # Retirer des exÃƒÂ©cutions actives
            del self.active_executions[execution_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur exÃƒÂ©cution {execution_id}: {e}")
            
            execution['status'] = ExecutionStatus.FAILED
            execution['error'] = str(e)
            execution['end_time'] = datetime.now()
            
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            self.stats['failed_executions'] += 1
            
            return {
                'success': False,
                'execution_id': execution_id,
                'error': str(e),
                'status': ExecutionStatus.FAILED.value
            }
    
    def _execute_single_order(self, execution: Dict) -> Dict:
        """
        ExÃƒÂ©cute un ordre unique avec retry
        
        Args:
            execution: DonnÃƒÂ©es d'exÃƒÂ©cution
            
        Returns:
            RÃƒÂ©sultat de l'exÃƒÂ©cution
        """
        execution['status'] = ExecutionStatus.EXECUTING
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"  Tentative {attempt}/{self.max_retries}")
                
                # Passer l'ordre
                order = self.order_manager.place_order(
                    symbol=execution['symbol'],
                    side=execution['side'],
                    quantity=execution['quantity'],
                    order_type=execution['order_type'],
                    price=execution['price']
                )
                
                execution['orders'].append(order)
                
                # VÃƒÂ©rifier le slippage
                if execution['price']:
                    actual_price = order.price or execution['price']
                    slippage = abs(actual_price - execution['price']) / execution['price']
                    
                    if slippage > self.max_slippage:
                        logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Slippage ÃƒÂ©levÃƒÂ©: {slippage:.2%}")
                
                # Ordre placÃƒÂ© avec succÃƒÂ¨s
                execution['status'] = ExecutionStatus.COMPLETED
                
                logger.info(f"Ã¢Å“â€¦ Ordre exÃƒÂ©cutÃƒÂ©: {order.client_order_id}")
                
                return {
                    'success': True,
                    'execution_id': execution['id'],
                    'order': order,
                    'status': ExecutionStatus.COMPLETED.value,
                    'attempts': attempt
                }
                
            except Exception as e:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Tentative {attempt} ÃƒÂ©chouÃƒÂ©e: {e}")
                execution['retries'] += 1
                self.stats['total_retries'] += 1
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        # Ãƒâ€°chec aprÃƒÂ¨s tous les essais
        execution['status'] = ExecutionStatus.FAILED
        raise Exception(f"Ãƒâ€°chec aprÃƒÂ¨s {self.max_retries} tentatives")
    
    def _execute_with_splitting(self, execution: Dict) -> Dict:
        """
        ExÃƒÂ©cute un ordre en le divisant en plusieurs chunks
        
        Args:
            execution: DonnÃƒÂ©es d'exÃƒÂ©cution
            
        Returns:
            RÃƒÂ©sultat de l'exÃƒÂ©cution
        """
        logger.info(f"Ã°Å¸â€œÂ¦ Split de l'ordre en chunks de {self.max_chunk_size} USDC")
        
        execution['status'] = ExecutionStatus.EXECUTING
        
        # Calculer les chunks
        total_size_usdc = execution['quantity'] * (execution['price'] or 0)
        num_chunks = int(total_size_usdc / self.max_chunk_size) + 1
        chunk_quantity = execution['quantity'] / num_chunks
        
        completed_orders = []
        failed_orders = []
        
        for i in range(num_chunks):
            try:
                logger.info(f"  Chunk {i+1}/{num_chunks}: {chunk_quantity:.6f}")
                
                # CrÃƒÂ©er une mini-exÃƒÂ©cution pour ce chunk
                chunk_execution = execution.copy()
                chunk_execution['quantity'] = chunk_quantity
                
                result = self._execute_single_order(chunk_execution)
                
                if result['success']:
                    completed_orders.append(result['order'])
                else:
                    failed_orders.append(result)
                
                # Petit dÃƒÂ©lai entre les chunks
                if i < num_chunks - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Ãƒâ€°chec chunk {i+1}: {e}")
                failed_orders.append({'error': str(e)})
        
        # DÃƒÂ©terminer le statut final
        if len(completed_orders) == num_chunks:
            execution['status'] = ExecutionStatus.COMPLETED
            status = ExecutionStatus.COMPLETED
        elif len(completed_orders) > 0:
            execution['status'] = ExecutionStatus.PARTIAL
            status = ExecutionStatus.PARTIAL
        else:
            execution['status'] = ExecutionStatus.FAILED
            status = ExecutionStatus.FAILED
        
        execution['orders'] = completed_orders
        
        logger.info(f"Ã°Å¸â€œÂ¦ Split terminÃƒÂ©: {len(completed_orders)}/{num_chunks} rÃƒÂ©ussis")
        
        return {
            'success': len(completed_orders) > 0,
            'execution_id': execution['id'],
            'status': status.value,
            'total_chunks': num_chunks,
            'completed_chunks': len(completed_orders),
            'failed_chunks': len(failed_orders),
            'orders': completed_orders
        }
    
    def _should_split_order(self, quantity: float, price: float) -> bool:
        """
        DÃƒÂ©termine si un ordre doit ÃƒÂªtre divisÃƒÂ©
        
        Args:
            quantity: QuantitÃƒÂ©
            price: Prix
            
        Returns:
            True si l'ordre doit ÃƒÂªtre divisÃƒÂ©
        """
        if not self.enable_order_splitting:
            return False
        
        if price == 0:
            return False
        
        total_size_usdc = quantity * price
        
        return total_size_usdc > self.min_split_size
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Annule une exÃƒÂ©cution en cours
        
        Args:
            execution_id: ID de l'exÃƒÂ©cution
            
        Returns:
            True si annulÃƒÂ©e
        """
        if execution_id not in self.active_executions:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â ExÃƒÂ©cution {execution_id} introuvable")
            return False
        
        execution = self.active_executions[execution_id]
        
        # Annuler les ordres placÃƒÂ©s
        for order in execution['orders']:
            try:
                self.order_manager.cancel_order(
                    execution['symbol'],
                    order.client_order_id
                )
            except Exception as e:
                logger.error(f"Erreur annulation ordre: {e}")
        
        execution['status'] = ExecutionStatus.CANCELLED
        execution['end_time'] = datetime.now()
        
        self.execution_history.append(execution)
        del self.active_executions[execution_id]
        
        self.stats['cancelled_executions'] += 1
        
        logger.info(f"Ã°Å¸Å¡Â« ExÃƒÂ©cution {execution_id} annulÃƒÂ©e")
        
        return True
    
    def _generate_execution_id(self) -> str:
        """GÃƒÂ©nÃƒÂ¨re un ID unique pour l'exÃƒÂ©cution"""
        timestamp = int(time.time() * 1000)
        return f"exec_{timestamp}_{self.stats['total_executions']}"
    
    def _update_stats(self, result: Dict, execution_time: float):
        """Met ÃƒÂ  jour les statistiques"""
        if result['success']:
            self.stats['successful_executions'] += 1
        else:
            self.stats['failed_executions'] += 1
        
        # Moyenne mobile de l'execution time
        n = self.stats['successful_executions']
        if n > 0:
            current_avg = self.stats['avg_execution_time_ms']
            self.stats['avg_execution_time_ms'] = (current_avg * (n-1) + execution_time) / n
    
    def get_status(self) -> Dict:
        """Retourne le statut de l'execution engine"""
        return {
            'active_executions': len(self.active_executions),
            'total_executions': self.stats['total_executions'],
            'success_rate': (self.stats['successful_executions'] / self.stats['total_executions']
                           if self.stats['total_executions'] > 0 else 0),
            'avg_execution_time_ms': self.stats['avg_execution_time_ms'],
            'stats': self.stats.copy()
        }
    
    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Retourne l'historique des exÃƒÂ©cutions"""
        return self.execution_history[-limit:]


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de l'Execution Engine"""
    
    print("\n=== Test Execution Engine ===\n")
    
    # Mock de l'order manager
    class MockOrderManager:
        def __init__(self):
            self.orders = []
        
        def place_order(self, symbol, side, quantity, order_type='MARKET', price=None):
            from dataclasses import dataclass
            
            @dataclass
            class Order:
                client_order_id: str
                symbol: str
                side: str
                quantity: float
                price: float
            
            order = Order(
                client_order_id=f"order_{len(self.orders)}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price or 50000
            )
            self.orders.append(order)
            return order
        
        def cancel_order(self, symbol, order_id):
            return True
    
    # CrÃƒÂ©er l'engine
    order_mgr = MockOrderManager()
    engine = ExecutionEngine(order_mgr, {
        'max_retries': 2,
        'enable_order_splitting': True,
        'max_chunk_size': 200
    })
    
    # Test 1: Ordre simple
    print("1Ã¯Â¸ÂÃ¢Æ’Â£ Test ordre simple:")
    result = engine.execute(
        symbol='BTCUSDT',
        side='BUY',
        quantity=0.01,
        price=50000
    )
    print(f"   Success: {result['success']}")
    print(f"   Status: {result['status']}")
    print(f"   Time: {result.get('execution_time_ms', 0):.2f}ms")
    
    # Test 2: Ordre avec splitting
    print("\n2Ã¯Â¸ÂÃ¢Æ’Â£ Test ordre avec splitting:")
    result = engine.execute(
        symbol='BTCUSDT',
        side='BUY',
        quantity=0.02,  # 1000 USDC -> sera splittÃƒÂ©
        price=50000
    )
    print(f"   Success: {result['success']}")
    print(f"   Status: {result['status']}")
    if 'total_chunks' in result:
        print(f"   Chunks: {result['completed_chunks']}/{result['total_chunks']}")
    
    # Stats
    print("\nÃ°Å¸â€œÅ  Statistiques:")
    status = engine.get_status()
    print(f"   Total: {status['total_executions']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Avg time: {status['avg_execution_time_ms']:.2f}ms")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")
