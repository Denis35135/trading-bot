"""
Logger pour The Bot
SystÃƒÂ¨me de logging colorÃƒÂ© et structurÃƒÂ© avec rotation de fichiers
"""

import os
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False
    print("Ã¢Å¡Â Ã¯Â¸Â colorlog non disponible, logs en noir et blanc")


# RÃƒÂ©pertoire des logs
LOGS_DIR = "data/logs"
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5


# Format des logs
LOG_FORMAT = "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Format colorÃƒÂ© pour la console
if COLORLOG_AVAILABLE:
    CONSOLE_LOG_FORMAT = "%(log_color)s%(asctime)s%(reset)s - %(cyan)s%(name)-20s%(reset)s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s"
else:
    CONSOLE_LOG_FORMAT = LOG_FORMAT

# Couleurs par niveau
LOG_COLORS = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


def setup_logger(name: str = 'TheBot', 
                level: str = 'INFO',
                log_to_file: bool = True,
                log_to_console: bool = True,
                file_level: Optional[str] = None,
                console_level: Optional[str] = None) -> logging.Logger:
    """
    Configure et retourne un logger
    
    Args:
        name: Nom du logger
        level: Niveau de log par dÃƒÂ©faut (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_to_file: Logger dans un fichier
        log_to_console: Logger sur la console
        file_level: Niveau pour le fichier (si diffÃƒÂ©rent de level)
        console_level: Niveau pour la console (si diffÃƒÂ©rent de level)
        
    Returns:
        Logger configurÃƒÂ©
        
    Usage:
        logger = setup_logger('MyModule', 'INFO')
        logger.info("Message info")
        logger.error("Message erreur")
    """
    # CrÃƒÂ©er le logger
    logger = logging.getLogger(name)
    
    # Ãƒâ€°viter les doublons si dÃƒÂ©jÃƒÂ  configurÃƒÂ©
    if logger.handlers:
        return logger
    
    # Niveau global
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # CrÃƒÂ©er le rÃƒÂ©pertoire de logs
    if log_to_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
    
    # ========================================================================
    # HANDLER CONSOLE
    # ========================================================================
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            getattr(logging, console_level.upper()) if console_level 
            else log_level
        )
        
        # Formatter avec couleurs si disponible
        if COLORLOG_AVAILABLE:
            console_formatter = colorlog.ColoredFormatter(
                CONSOLE_LOG_FORMAT,
                datefmt=LOG_DATE_FORMAT,
                log_colors=LOG_COLORS,
                reset=True,
                style='%'
            )
        else:
            console_formatter = logging.Formatter(
                LOG_FORMAT,
                datefmt=LOG_DATE_FORMAT
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # ========================================================================
    # HANDLER FICHIER (avec rotation par taille)
    # ========================================================================
    if log_to_file:
        # Fichier principal avec rotation
        log_filename = os.path.join(LOGS_DIR, f"{name.lower()}.log")
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=LOG_FILE_MAX_SIZE,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(
            getattr(logging, file_level.upper()) if file_level 
            else log_level
        )
        
        file_formatter = logging.Formatter(
            LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # ====================================================================
        # HANDLER FICHIER ERREURS (seulement ERROR et CRITICAL)
        # ====================================================================
        error_log_filename = os.path.join(LOGS_DIR, f"{name.lower()}_errors.log")
        error_handler = RotatingFileHandler(
            error_log_filename,
            maxBytes=LOG_FILE_MAX_SIZE,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        # ====================================================================
        # HANDLER FICHIER QUOTIDIEN (rotation par jour)
        # ====================================================================
        daily_log_filename = os.path.join(
            LOGS_DIR, 
            f"{name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        daily_handler = TimedRotatingFileHandler(
            daily_log_filename,
            when='midnight',
            interval=1,
            backupCount=30,  # Garder 30 jours
            encoding='utf-8'
        )
        daily_handler.setLevel(log_level)
        daily_handler.setFormatter(file_formatter)
        
        # Ajouter suffix de date aux backups
        daily_handler.suffix = "%Y%m%d"
        
        logger.addHandler(daily_handler)
    
    # Ne pas propager aux loggers parents
    logger.propagate = False
    
    # Message de confirmation
    logger.debug(f"Logger '{name}' configurÃƒÂ© (niveau: {level})")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    RÃƒÂ©cupÃƒÂ¨re un logger existant ou en crÃƒÂ©e un nouveau
    
    Args:
        name: Nom du logger
        
    Returns:
        Logger
        
    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    logger = logging.getLogger(name)
    
    # Si pas de handlers, configurer avec les paramÃƒÂ¨tres par dÃƒÂ©faut
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def set_log_level(logger: logging.Logger, level: str):
    """
    Change le niveau de log d'un logger
    
    Args:
        logger: Logger ÃƒÂ  modifier
        level: Nouveau niveau (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Mettre ÃƒÂ  jour tous les handlers
    for handler in logger.handlers:
        handler.setLevel(log_level)
    
    logger.info(f"Niveau de log changÃƒÂ© ÃƒÂ  {level}")


def disable_external_loggers():
    """
    DÃƒÂ©sactive ou rÃƒÂ©duit les logs des librairies externes bruyantes
    """
    # Binance
    logging.getLogger('binance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)
    
    # Requests
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    
    # SQLAlchemy
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    # Matplotlib (si utilisÃƒÂ©)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Asyncio
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def cleanup_old_logs(days: int = 7):
    """
    Nettoie les vieux fichiers de logs
    
    Args:
        days: Nombre de jours ÃƒÂ  conserver
    """
    if not os.path.exists(LOGS_DIR):
        return
    
    cutoff_time = datetime.now().timestamp() - (days * 86400)
    deleted_count = 0
    
    for filename in os.listdir(LOGS_DIR):
        filepath = os.path.join(LOGS_DIR, filename)
        
        if os.path.isfile(filepath):
            file_time = os.path.getmtime(filepath)
            
            if file_time < cutoff_time:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"Erreur suppression {filename}: {e}")
    
    if deleted_count > 0:
        logger = get_logger('LogCleaner')
        logger.info(f"Ã°Å¸Â§Â¹ {deleted_count} vieux fichiers de logs supprimÃƒÂ©s (>{days} jours)")


def get_log_stats() -> dict:
    """
    Retourne des statistiques sur les logs
    
    Returns:
        Dict avec stats
    """
    if not os.path.exists(LOGS_DIR):
        return {
            'total_files': 0,
            'total_size_mb': 0,
            'files': []
        }
    
    total_size = 0
    files_info = []
    
    for filename in os.listdir(LOGS_DIR):
        filepath = os.path.join(LOGS_DIR, filename)
        
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            
            files_info.append({
                'name': filename,
                'size_mb': size / (1024 * 1024),
                'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
            })
    
    # Trier par date de modification
    files_info.sort(key=lambda x: x['modified'], reverse=True)
    
    return {
        'total_files': len(files_info),
        'total_size_mb': total_size / (1024 * 1024),
        'files': files_info
    }


def print_log_stats():
    """Affiche les statistiques des logs"""
    stats = get_log_stats()
    
    print("\n" + "=" * 60)
    print("STATISTIQUES DES LOGS")
    print("=" * 60)
    print(f"Nombre de fichiers: {stats['total_files']}")
    print(f"Taille totale: {stats['total_size_mb']:.2f} MB")
    
    if stats['files']:
        print("\nFichiers rÃƒÂ©cents:")
        for file_info in stats['files'][:10]:  # Top 10
            print(
                f"  Ã¢â‚¬Â¢ {file_info['name']:<40} "
                f"{file_info['size_mb']:>6.2f} MB  "
                f"{file_info['modified'].strftime('%Y-%m-%d %H:%M')}"
            )
    
    print("=" * 60 + "\n")


class LoggerContext:
    """
    Context manager pour changer temporairement le niveau de log
    
    Usage:
        with LoggerContext(logger, 'DEBUG'):
            # Code avec niveau DEBUG
            logger.debug("Debug message visible")
        # Retour au niveau prÃƒÂ©cÃƒÂ©dent
    """
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Args:
            logger: Logger ÃƒÂ  modifier
            level: Niveau temporaire
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
        self.old_handler_levels = []
    
    def __enter__(self):
        # Sauvegarder les niveaux actuels
        self.old_handler_levels = [h.level for h in self.logger.handlers]
        
        # Changer les niveaux
        self.logger.setLevel(self.new_level)
        for handler in self.logger.handlers:
            handler.setLevel(self.new_level)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurer les niveaux
        self.logger.setLevel(self.old_level)
        for handler, level in zip(self.logger.handlers, self.old_handler_levels):
            handler.setLevel(level)


class PerformanceLogger:
    """
    Logger de performance pour mesurer le temps d'exÃƒÂ©cution
    
    Usage:
        with PerformanceLogger(logger, "Calcul indicateurs"):
            # Code ÃƒÂ  mesurer
            calculate_indicators()
    """
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = "INFO"):
        """
        Args:
            logger: Logger ÃƒÂ  utiliser
            operation: Nom de l'opÃƒÂ©ration
            level: Niveau de log
        """
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level.upper())
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Ã¢ÂÂ±Ã¯Â¸Â DÃƒÂ©but: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(
                self.level, 
                f"Ã¢Å“â€¦ Fin: {self.operation} (temps: {elapsed:.3f}s)"
            )
        else:
            self.logger.log(
                logging.ERROR,
                f"Ã¢ÂÅ’ Ãƒâ€°chec: {self.operation} aprÃƒÂ¨s {elapsed:.3f}s - {exc_val}"
            )


# ============================================================================
# CONFIGURATION PAR DÃƒâ€°FAUT AU CHARGEMENT DU MODULE
# ============================================================================

# DÃƒÂ©sactiver les loggers externes bruyants
disable_external_loggers()

# Logger par dÃƒÂ©faut pour le module utils
_module_logger = get_logger('utils')

# Nettoyer les vieux logs au dÃƒÂ©marrage (>30 jours)
try:
    cleanup_old_logs(days=30)
except Exception as e:
    _module_logger.warning(f"Impossible de nettoyer les vieux logs: {e}")


# ============================================================================
# HELPERS
# ============================================================================

def log_system_info(logger: Optional[logging.Logger] = None):
    """
    Log les informations systÃƒÂ¨me
    
    Args:
        logger: Logger ÃƒÂ  utiliser (ou logger par dÃƒÂ©faut)
    """
    if logger is None:
        logger = get_logger('SystemInfo')
    
    import platform
    import psutil
    
    logger.info("=" * 60)
    logger.info("INFORMATIONS SYSTÃƒË†ME")
    logger.info("=" * 60)
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"Processeur: {platform.processor()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info("=" * 60)


def log_config(config_dict: dict, logger: Optional[logging.Logger] = None, mask_keys: list = None):
    """
    Log une configuration en masquant les clÃƒÂ©s sensibles
    
    Args:
        config_dict: Configuration ÃƒÂ  logger
        logger: Logger ÃƒÂ  utiliser
        mask_keys: Liste des clÃƒÂ©s ÃƒÂ  masquer (ex: ['api_key', 'secret'])
    """
    if logger is None:
        logger = get_logger('Config')
    
    if mask_keys is None:
        mask_keys = ['api_key', 'api_secret', 'secret', 'password', 'token', 'key']
    
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    
    for key, value in config_dict.items():
        # Masquer les valeurs sensibles
        if any(sensitive in key.lower() for sensitive in mask_keys):
            display_value = "***MASKED***"
        else:
            display_value = value
        
        logger.info(f"{key}: {display_value}")
    
    logger.info("=" * 60)


# Export principal
__all__ = [
    'setup_logger',
    'get_logger',
    'set_log_level',
    'disable_external_loggers',
    'cleanup_old_logs',
    'get_log_stats',
    'print_log_stats',
    'LoggerContext',
    'PerformanceLogger',
    'log_system_info',
    'log_config',
]