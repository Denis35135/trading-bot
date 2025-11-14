"""
Decorators pour The Bot
DÃƒÂ©corateurs utilitaires pour retry, timeout, logging, performance, etc.
"""

import time
import logging
import threading
from functools import wraps
from typing import Callable, Any, Optional, List, Type
from datetime import datetime, timedelta
import signal

logger = logging.getLogger(__name__)


# ============================================================================
# RETRY DECORATOR
# ============================================================================

def retry(max_attempts: int = 3, 
          delay: float = 1.0, 
          backoff: float = 2.0,
          exceptions: tuple = (Exception,),
          on_retry: Optional[Callable] = None):
    """
    DÃƒÂ©corateur pour retry automatique avec backoff exponentiel
    
    Args:
        max_attempts: Nombre maximum de tentatives
        delay: DÃƒÂ©lai initial entre tentatives (secondes)
        backoff: Multiplicateur pour backoff exponentiel
        exceptions: Tuple des exceptions ÃƒÂ  capturer
        on_retry: Fonction callback appelÃƒÂ©e ÃƒÂ  chaque retry
        
    Usage:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def risky_operation():
            # Code qui peut ÃƒÂ©chouer
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Tentative {attempt}/{max_attempts} ÃƒÂ©chouÃƒÂ©e pour {func.__name__}: {e}. "
                            f"Retry dans {current_delay:.1f}s..."
                        )
                        
                        # Callback optionnel
                        if on_retry:
                            try:
                                on_retry(attempt, e)
                            except Exception as callback_error:
                                logger.error(f"Erreur callback retry: {callback_error}")
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"Ãƒâ€°chec dÃƒÂ©finitif de {func.__name__} aprÃƒÂ¨s {max_attempts} tentatives: {e}"
                        )
            
            # Relancer la derniÃƒÂ¨re exception si tous les essais ont ÃƒÂ©chouÃƒÂ©
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# TIMEOUT DECORATOR
# ============================================================================

class TimeoutError(Exception):
    """Exception levÃƒÂ©e en cas de timeout"""
    pass


def timeout(seconds: int):
    """
    DÃƒÂ©corateur pour limiter le temps d'exÃƒÂ©cution d'une fonction
    
    Args:
        seconds: Temps maximum en secondes
        
    Usage:
        @timeout(5)
        def slow_operation():
            # Code qui ne doit pas dÃƒÂ©passer 5 secondes
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                logger.error(f"Timeout de {seconds}s dÃƒÂ©passÃƒÂ© pour {func.__name__}")
                raise TimeoutError(f"Function {func.__name__} exceeded timeout of {seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


# ============================================================================
# LOGGING DECORATOR
# ============================================================================

def log_execution(level: str = "INFO", 
                  log_args: bool = False,
                  log_result: bool = False,
                  log_time: bool = True):
    """
    DÃƒÂ©corateur pour logger l'exÃƒÂ©cution d'une fonction
    
    Args:
        level: Niveau de log (INFO/DEBUG/WARNING)
        log_args: Logger les arguments
        log_result: Logger le rÃƒÂ©sultat
        log_time: Logger le temps d'exÃƒÂ©cution
        
    Usage:
        @log_execution(level="INFO", log_time=True)
        def important_operation(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            
            # Log dÃƒÂ©but
            log_msg = f"ExÃƒÂ©cution de {func_name}"
            if log_args:
                log_msg += f" avec args={args}, kwargs={kwargs}"
            
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(log_msg)
            
            # ExÃƒÂ©cution
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Log succÃƒÂ¨s
                end_time = time.time()
                success_msg = f"Ã¢Å“â€¦ {func_name} terminÃƒÂ©"
                
                if log_time:
                    elapsed = end_time - start_time
                    success_msg += f" en {elapsed:.3f}s"
                
                if log_result:
                    success_msg += f" - RÃƒÂ©sultat: {result}"
                
                log_method(success_msg)
                return result
                
            except Exception as e:
                # Log erreur
                end_time = time.time()
                elapsed = end_time - start_time
                logger.error(f"Ã¢ÂÅ’ {func_name} ÃƒÂ©chouÃƒÂ© aprÃƒÂ¨s {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# ============================================================================
# PERFORMANCE DECORATOR
# ============================================================================

def measure_time(threshold_ms: Optional[float] = None):
    """
    DÃƒÂ©corateur pour mesurer le temps d'exÃƒÂ©cution
    
    Args:
        threshold_ms: Seuil en ms, log warning si dÃƒÂ©passÃƒÂ©
        
    Usage:
        @measure_time(threshold_ms=100)
        def operation():
            # Code ÃƒÂ  mesurer
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            
            if threshold_ms and elapsed_ms > threshold_ms:
                logger.warning(
                    f"Ã¢Å¡Â Ã¯Â¸Â {func.__name__} a pris {elapsed_ms:.2f}ms "
                    f"(seuil: {threshold_ms}ms)"
                )
            else:
                logger.debug(f"Ã¢ÂÂ±Ã¯Â¸Â {func.__name__}: {elapsed_ms:.2f}ms")
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# RATE LIMITING DECORATOR
# ============================================================================

class RateLimiter:
    """Gestionnaire de rate limiting"""
    
    def __init__(self, max_calls: int, period: float):
        """
        Args:
            max_calls: Nombre max d'appels
            period: PÃƒÂ©riode en secondes
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with self.lock:
                now = time.time()
                
                # Nettoyer les vieux appels
                self.calls = [call_time for call_time in self.calls 
                             if now - call_time < self.period]
                
                # VÃƒÂ©rifier la limite
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        logger.warning(
                            f"Rate limit atteint pour {func.__name__}. "
                            f"Attente de {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        
                        # Re-nettoyer aprÃƒÂ¨s sleep
                        now = time.time()
                        self.calls = [call_time for call_time in self.calls 
                                     if now - call_time < self.period]
                
                # Enregistrer l'appel
                self.calls.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper


def rate_limit(max_calls: int, period: float):
    """
    DÃƒÂ©corateur pour limiter le taux d'appels
    
    Args:
        max_calls: Nombre maximum d'appels
        period: PÃƒÂ©riode en secondes
        
    Usage:
        @rate_limit(max_calls=10, period=1.0)  # 10 appels max par seconde
        def api_call():
            pass
    """
    return RateLimiter(max_calls, period)


# ============================================================================
# ERROR HANDLER DECORATOR
# ============================================================================

def error_handler(default_return: Any = None,
                  log_error: bool = True,
                  raise_error: bool = False,
                  on_error: Optional[Callable] = None):
    """
    DÃƒÂ©corateur pour gÃƒÂ©rer les erreurs ÃƒÂ©lÃƒÂ©gamment
    
    Args:
        default_return: Valeur par dÃƒÂ©faut en cas d'erreur
        log_error: Logger l'erreur
        raise_error: Relancer l'erreur aprÃƒÂ¨s handling
        on_error: Callback en cas d'erreur
        
    Usage:
        @error_handler(default_return=None, log_error=True)
        def risky_function():
            # Code qui peut ÃƒÂ©chouer
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Erreur dans {func.__name__}: {e}", exc_info=True)
                
                if on_error:
                    try:
                        on_error(e)
                    except Exception as callback_error:
                        logger.error(f"Erreur callback error_handler: {callback_error}")
                
                if raise_error:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


# ============================================================================
# VALIDATION DECORATOR
# ============================================================================

def validate_params(**validators):
    """
    DÃƒÂ©corateur pour valider les paramÃƒÂ¨tres d'une fonction
    
    Args:
        **validators: Dict de validateurs {param_name: validator_func}
        
    Usage:
        @validate_params(
            symbol=lambda x: isinstance(x, str) and len(x) > 0,
            price=lambda x: isinstance(x, (int, float)) and x > 0
        )
        def place_order(symbol, price):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Obtenir les noms des paramÃƒÂ¨tres
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Valider chaque paramÃƒÂ¨tre
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    if not validator(value):
                        raise ValueError(
                            f"Validation ÃƒÂ©chouÃƒÂ©e pour le paramÃƒÂ¨tre '{param_name}' "
                            f"de {func.__name__}: valeur={value}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# THREAD SAFETY DECORATOR
# ============================================================================

def thread_safe(lock: Optional[threading.Lock] = None):
    """
    DÃƒÂ©corateur pour rendre une fonction thread-safe
    
    Args:
        lock: Lock ÃƒÂ  utiliser (ou crÃƒÂ©er un nouveau)
        
    Usage:
        my_lock = threading.Lock()
        
        @thread_safe(my_lock)
        def critical_section():
            # Code thread-safe
            pass
    """
    if lock is None:
        lock = threading.Lock()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with lock:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# CACHE DECORATOR (Simple)
# ============================================================================

def simple_cache(ttl_seconds: int = 60):
    """
    DÃƒÂ©corateur pour mettre en cache les rÃƒÂ©sultats
    
    Args:
        ttl_seconds: DurÃƒÂ©e de vie du cache en secondes
        
    Usage:
        @simple_cache(ttl_seconds=60)
        def expensive_calculation(x, y):
            return x * y
    """
    cache = {}
    cache_times = {}
    lock = threading.Lock()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # CrÃƒÂ©er une clÃƒÂ© de cache
            cache_key = str(args) + str(kwargs)
            
            with lock:
                now = time.time()
                
                # VÃƒÂ©rifier si en cache et valide
                if cache_key in cache:
                    if now - cache_times[cache_key] < ttl_seconds:
                        logger.debug(f"Cache hit pour {func.__name__}")
                        return cache[cache_key]
                    else:
                        # ExpirÃƒÂ©
                        del cache[cache_key]
                        del cache_times[cache_key]
            
            # Calculer et mettre en cache
            result = func(*args, **kwargs)
            
            with lock:
                cache[cache_key] = result
                cache_times[cache_key] = time.time()
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# TRADING SPECIFIC DECORATORS
# ============================================================================

def require_market_open(exchange_checker: Optional[Callable] = None):
    """
    DÃƒÂ©corateur pour vÃƒÂ©rifier que le marchÃƒÂ© est ouvert
    
    Args:
        exchange_checker: Fonction qui vÃƒÂ©rifie si marchÃƒÂ© ouvert
        
    Usage:
        @require_market_open()
        def place_trade():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if exchange_checker:
                if not exchange_checker():
                    logger.warning(f"MarchÃƒÂ© fermÃƒÂ©, {func.__name__} ignorÃƒÂ©")
                    return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def check_balance(min_balance: float = 0):
    """
    DÃƒÂ©corateur pour vÃƒÂ©rifier le solde avant exÃƒÂ©cution
    
    Args:
        min_balance: Solde minimum requis
        
    Usage:
        @check_balance(min_balance=100)
        def place_order():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            if hasattr(self, 'capital') and self.capital < min_balance:
                logger.warning(
                    f"Solde insuffisant pour {func.__name__}: "
                    f"{self.capital} < {min_balance}"
                )
                return None
            
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def require_risk_approval(risk_manager_attr: str = 'risk_monitor'):
    """
    DÃƒÂ©corateur pour obtenir l'approbation du risk manager
    
    Args:
        risk_manager_attr: Nom de l'attribut risk manager
        
    Usage:
        @require_risk_approval()
        def execute_trade(self, signal):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            if hasattr(self, risk_manager_attr):
                risk_manager = getattr(self, risk_manager_attr)
                
                # VÃƒÂ©rifier approbation (assume une mÃƒÂ©thode approve_trade)
                if hasattr(risk_manager, 'approve_trade'):
                    if not risk_manager.approve_trade(*args, **kwargs):
                        logger.warning(f"Trade refusÃƒÂ© par risk manager: {func.__name__}")
                        return None
            
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# ASYNC DECORATOR
# ============================================================================

def async_executor(max_workers: int = 4):
    """
    DÃƒÂ©corateur pour exÃƒÂ©cuter une fonction de maniÃƒÂ¨re asynchrone
    
    Args:
        max_workers: Nombre max de workers
        
    Usage:
        @async_executor(max_workers=4)
        def heavy_computation():
            pass
    """
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return executor.submit(func, *args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# DEPRECATION DECORATOR
# ============================================================================

def deprecated(reason: str = "", version: str = ""):
    """
    DÃƒÂ©corateur pour marquer une fonction comme dÃƒÂ©prÃƒÂ©ciÃƒÂ©e
    
    Args:
        reason: Raison de la dÃƒÂ©prÃƒÂ©ciation
        version: Version ÃƒÂ  partir de laquelle c'est dÃƒÂ©prÃƒÂ©ciÃƒÂ©
        
    Usage:
        @deprecated(reason="Use new_function instead", version="2.0")
        def old_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            msg = f"Ã¢Å¡Â Ã¯Â¸Â {func.__name__} est dÃƒÂ©prÃƒÂ©ciÃƒÂ©"
            if version:
                msg += f" depuis la version {version}"
            if reason:
                msg += f". {reason}"
            
            logger.warning(msg)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator