"""
Configuration Template pour The Bot
Copiez ce fichier en config.py et ajustez les parametres
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class Config:
    """Configuration principale du bot"""
    
    # ===================================================================
    # BINANCE API
    # ===================================================================
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key_here')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'your_secret_key_here')
    TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'
    
    # ===================================================================
    # CAPITAL & RISQUE
    # ===================================================================
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))  # USDC
    
    # Risque par trade (2% = 0.02)
    RISK_PER_TRADE = 0.02
    
    # Perte maximale par jour (5% = 0.05)
    MAX_DAILY_LOSS = 0.05
    
    # Drawdown maximum avant arret (8% = 0.08)
    MAX_DRAWDOWN = 0.08
    
    # Taille minimale d'ordre (minimum Binance)
    MIN_ORDER_SIZE = 50  # USDC
    
    # Taille maximale de position (25% du capital)
    MAX_POSITION_SIZE = 0.25
    
    # ===================================================================
    # STRAT"GIES
    # ===================================================================
    ACTIVE_STRATEGIES = [
        {
            'name': 'scalping',
            'enabled': True,
            'allocation': 0.40,  # 40% du capital
            'min_confidence': 0.65,
            'timeframe': '5m'
        },
        {
            'name': 'momentum',
            'enabled': True,
            'allocation': 0.25,  # 25%
            'min_confidence': 0.70,
            'timeframe': '15m'
        },
        {
            'name': 'mean_reversion',
            'enabled': True,
            'allocation': 0.20,  # 20%
            'min_confidence': 0.70,
            'timeframe': '5m'
        },
        {
            'name': 'pattern',
            'enabled': True,
            'allocation': 0.10,  # 10%
            'min_confidence': 0.65,
            'timeframe': '15m'
        },
        {
            'name': 'ml',
            'enabled': True,
            'allocation': 0.05,  # 5% (test)
            'min_confidence': 0.75,
            'timeframe': '5m'
        }
    ]
    
    # ===================================================================
    # MARKET SCANNER
    # ===================================================================
    
    # Nombre de symboles  scanner
    SYMBOLS_TO_SCAN = 100
    
    # Nombre de symboles  trader
    SYMBOLS_TO_TRADE = 20
    
    # Intervalle de scan (secondes)
    SCAN_INTERVAL = 300  # 5 minutes
    
    # Volume minimum 24h (USDC)
    MIN_VOLUME_24H = 10_000_000  # 10M
    
    # Spread maximum (%)
    MAX_SPREAD_PERCENT = 0.002  # 0.2%
    
    # Range de volatilite acceptable (%)
    VOLATILITY_RANGE = (0.02, 0.08)  # 2% - 8%
    
    # Blacklist de symboles
    BLACKLISTED_SYMBOLS = [
        # Ajoutez les symboles  ignorer
        # 'LUNAUSDC',  # Exemple
    ]
    
    # Forcer certains symboles (optionnel)
    FORCED_SYMBOLS = [
        # 'BTCUSDC',
        # 'ETHUSDC',
    ]
    
    # ===================================================================
    # EX"CUTION DES ORDRES
    # ===================================================================
    
    # Tolerance au slippage (0.2% = 0.002)
    SLIPPAGE_TOLERANCE = 0.002
    
    # Timeout pour les ordres (ms)
    ORDER_TIMEOUT = 5000  # 5 secondes
    
    # Nombre de tentatives en cas d'echec
    RETRY_ATTEMPTS = 3
    
    # Delai entre les tentatives (secondes)
    RETRY_DELAY = 1
    
    # ===================================================================
    # MACHINE LEARNING
    # ===================================================================
    
    # Seuil de confiance minimum
    ML_CONFIDENCE_THRESHOLD = 0.65
    
    # Nombre de features
    FEATURE_COUNT = 30
    
    # Frequence de reentranement (secondes)
    RETRAIN_FREQUENCY = 86400  # 24h
    
    # Taille minimale du dataset pour entranement
    MIN_TRAINING_SAMPLES = 10000
    
    # ===================================================================
    # CIRCUIT BREAKERS
    # ===================================================================
    
    # Niveaux de circuit breakers
    CIRCUIT_BREAKER_LEVELS = {
        'warning': {
            'drawdown': 0.03,  # 3%
            'daily_loss': 0.03,  # 3%
            'action': 'reduce_positions'  # Reduire de 50%
        },
        'critical': {
            'drawdown': 0.05,  # 5%
            'daily_loss': 0.05,  # 5%
            'action': 'close_losing'  # Fermer les positions perdantes
        },
        'emergency': {
            'drawdown': 0.08,  # 8%
            'daily_loss': 0.08,  # 8%
            'action': 'close_all'  # Tout fermer
        }
    }
    
    # ===================================================================
    # PERFORMANCE & OPTIMISATION
    # ===================================================================
    
    # Nombre de threads maximum
    MAX_THREADS = 4
    
    # Taille du buffer de ticks
    TICK_BUFFER_SIZE = 5000
    
    # Memoire maximum (MB)
    MAX_MEMORY_MB = 2000  # 2GB
    
    # Utiliser Redis pour le cache
    USE_REDIS = True
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    # Duree du cache (secondes)
    CACHE_TTL = 60
    
    # ===================================================================
    # MONITORING & LOGGING
    # ===================================================================
    
    # Niveau de log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Intervalle de sauvegarde (secondes)
    SAVE_INTERVAL = 300  # 5 minutes
    
    # Intervalle de health check (secondes)
    HEALTH_CHECK_INTERVAL = 60  # 1 minute
    
    # Rotation des logs (jours)
    LOG_ROTATION_DAYS = 7
    
    # Taille maximale des logs (MB)
    LOG_MAX_SIZE_MB = 100
    
    # ===================================================================
    # NOTIFICATIONS (optionnel)
    # ===================================================================
    
    # Telegram
    TELEGRAM_ENABLED = False
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Discord
    DISCORD_ENABLED = False
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    
    # Email
    EMAIL_ENABLED = False
    EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    EMAIL_FROM = os.getenv('EMAIL_FROM', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
    EMAIL_TO = os.getenv('EMAIL_TO', '')
    
    # ===================================================================
    # INDICATEURS TECHNIQUES
    # ===================================================================
    
    # RSI
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # MACD
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD = 2
    
    # EMA
    EMA_FAST = 9
    EMA_MEDIUM = 21
    EMA_SLOW = 50
    
    # ATR
    ATR_PERIOD = 14
    
    # ADX
    ADX_PERIOD = 14
    ADX_THRESHOLD = 25
    
    # ===================================================================
    # BASE DE DONN"ES (optionnel)
    # ===================================================================
    
    # PostgreSQL
    USE_POSTGRES = False
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'thebot')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'thebot')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    
    # ===================================================================
    # BACKTESTING
    # ===================================================================
    
    # Capital initial pour backtests
    BACKTEST_INITIAL_CAPITAL = 10000
    
    # Frais de trading (%)
    BACKTEST_COMMISSION = 0.001  # 0.1%
    
    # Slippage simule (%)
    BACKTEST_SLIPPAGE = 0.0005  # 0.05%
    
    # ===================================================================
    # AVANC"
    # ===================================================================
    
    # Activer le mode debug (plus de logs)
    DEBUG_MODE = False
    
    # Sauvegarder tous les signaux (pour analyse)
    SAVE_ALL_SIGNALS = True
    
    # Intervalle de sauvegarde des metriques (secondes)
    METRICS_SAVE_INTERVAL = 60
    
    # Activer le profiling de performance
    ENABLE_PROFILING = False
    
    # Limiter le nombre de positions simultanees
    MAX_CONCURRENT_POSITIONS = 20
    
    # Temps minimum entre deux trades sur le meme symbole (secondes)
    MIN_TIME_BETWEEN_TRADES = 60  # 1 minute
    
    # ===================================================================
    # VALIDATION
    # ===================================================================
    
    @classmethod
    def validate(cls):
        """Valide la configuration"""
        errors = []
        
        # Verifier les cles API
        if cls.BINANCE_API_KEY == 'your_api_key_here':
            errors.append("BINANCE_API_KEY non configuree")
        
        if cls.BINANCE_SECRET_KEY == 'your_secret_key_here':
            errors.append("BINANCE_SECRET_KEY non configuree")
        
        # Verifier le capital
        if cls.INITIAL_CAPITAL < 100:
            errors.append("INITIAL_CAPITAL trop faible (minimum 100 USDC)")
        
        # Verifier les allocations de strategies
        total_allocation = sum(s['allocation'] for s in cls.ACTIVE_STRATEGIES if s['enabled'])
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Total allocation strategies doit etre 1.0 (actuellement: {total_allocation})")
        
        # Verifier les risques
        if cls.RISK_PER_TRADE > 0.05:
            errors.append("RISK_PER_TRADE trop eleve (maximum 5%)")
        
        if cls.MAX_DRAWDOWN > 0.15:
            errors.append("MAX_DRAWDOWN trop eleve (maximum 15%)")
        
        # Verifier Redis si active
        if cls.USE_REDIS:
            try:
                import redis
                r = redis.Redis(host=cls.REDIS_HOST, port=cls.REDIS_PORT)
                r.ping()
            except Exception as e:
                errors.append(f"Redis non accessible: {e}")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """Affiche la configuration actuelle"""
        print("\n" + "="*60)
        print("  CONFIGURATION THE BOT")
        print("="*60 + "\n")
        
        print("" CAPITAL & RISQUE")
        print(f"   Capital Initial:    ${cls.INITIAL_CAPITAL:,.2f}")
        print(f"   Risk/Trade:         {cls.RISK_PER_TRADE:.1%}")
        print(f"   Max Daily Loss:     {cls.MAX_DAILY_LOSS:.1%}")
        print(f"   Max Drawdown:       {cls.MAX_DRAWDOWN:.1%}")
        
        print("\n STRAT"GIES ACTIVES")
        for strategy in cls.ACTIVE_STRATEGIES:
            if strategy['enabled']:
                print(f"   {strategy['name']:15} {strategy['allocation']:5.1%} "
                      f"(confidence: {strategy['min_confidence']:.0%})")
        
        print("\n" MARKET SCANNER")
        print(f"   Symboles scannes:   {cls.SYMBOLS_TO_SCAN}")
        print(f"   Symboles trades:    {cls.SYMBOLS_TO_TRADE}")
        print(f"   Volume min 24h:     ${cls.MIN_VOLUME_24H:,.0f}")
        print(f"   Spread max:         {cls.MAX_SPREAD_PERCENT:.2%}")
        
        print("\n' PERFORMANCE")
        print(f"   Max Threads:        {cls.MAX_THREADS}")
        print(f"   Max Memory:         {cls.MAX_MEMORY_MB} MB")
        print(f"   Redis:              {'""| Active' if cls.USE_REDIS else '' Desactive'}")
        
        print("\n"" NOTIFICATIONS")
        notifications = []
        if cls.TELEGRAM_ENABLED:
            notifications.append("Telegram")
        if cls.DISCORD_ENABLED:
            notifications.append("Discord")
        if cls.EMAIL_ENABLED:
            notifications.append("Email")
        
        if notifications:
            print(f"   Actives:            {', '.join(notifications)}")
        else:
            print("   Actives:            Aucune")
        
        print("\n" + "="*60 + "\n")


# ===================================================================
# CONFIGURATIONS PR"D"FINIES
# ===================================================================

class ConservativeConfig(Config):
    """Configuration conservatrice (faible risque)"""
    RISK_PER_TRADE = 0.01  # 1%
    MAX_DAILY_LOSS = 0.03  # 3%
    MAX_DRAWDOWN = 0.05    # 5%
    MAX_POSITION_SIZE = 0.15  # 15%
    
    ACTIVE_STRATEGIES = [
        {'name': 'scalping', 'enabled': True, 'allocation': 0.30, 'min_confidence': 0.75, 'timeframe': '5m'},
        {'name': 'momentum', 'enabled': True, 'allocation': 0.30, 'min_confidence': 0.75, 'timeframe': '15m'},
        {'name': 'mean_reversion', 'enabled': True, 'allocation': 0.30, 'min_confidence': 0.75, 'timeframe': '5m'},
        {'name': 'pattern', 'enabled': True, 'allocation': 0.10, 'min_confidence': 0.70, 'timeframe': '15m'},
        {'name': 'ml', 'enabled': False, 'allocation': 0.00, 'min_confidence': 0.80, 'timeframe': '5m'}
    ]


class AggressiveConfig(Config):
    """Configuration aggressive (haut risque)"""
    RISK_PER_TRADE = 0.03  # 3%
    MAX_DAILY_LOSS = 0.10  # 10%
    MAX_DRAWDOWN = 0.15    # 15%
    MAX_POSITION_SIZE = 0.35  # 35%
    
    ACTIVE_STRATEGIES = [
        {'name': 'scalping', 'enabled': True, 'allocation': 0.50, 'min_confidence': 0.60, 'timeframe': '5m'},
        {'name': 'momentum', 'enabled': True, 'allocation': 0.30, 'min_confidence': 0.65, 'timeframe': '15m'},
        {'name': 'mean_reversion', 'enabled': True, 'allocation': 0.10, 'min_confidence': 0.65, 'timeframe': '5m'},
        {'name': 'pattern', 'enabled': True, 'allocation': 0.05, 'min_confidence': 0.60, 'timeframe': '15m'},
        {'name': 'ml', 'enabled': True, 'allocation': 0.05, 'min_confidence': 0.70, 'timeframe': '5m'}
    ]


class ScalpingOnlyConfig(Config):
    """Configuration scalping uniquement"""
    RISK_PER_TRADE = 0.015  # 1.5%
    SYMBOLS_TO_TRADE = 10  # Moins de symboles, plus de focus
    
    ACTIVE_STRATEGIES = [
        {'name': 'scalping', 'enabled': True, 'allocation': 1.00, 'min_confidence': 0.65, 'timeframe': '5m'},
        {'name': 'momentum', 'enabled': False, 'allocation': 0.00, 'min_confidence': 0.70, 'timeframe': '15m'},
        {'name': 'mean_reversion', 'enabled': False, 'allocation': 0.00, 'min_confidence': 0.70, 'timeframe': '5m'},
        {'name': 'pattern', 'enabled': False, 'allocation': 0.00, 'min_confidence': 0.65, 'timeframe': '15m'},
        {'name': 'ml', 'enabled': False, 'allocation': 0.00, 'min_confidence': 0.75, 'timeframe': '5m'}
    ]


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def load_config(config_name: str = 'default'):
    """
    Charge une configuration specifique
    
    Args:
        config_name: 'default', 'conservative', 'aggressive', 'scalping'
    
    Returns:
        Config object
    """
    configs = {
        'default': Config,
        'conservative': ConservativeConfig,
        'aggressive': AggressiveConfig,
        'scalping': ScalpingOnlyConfig
    }
    
    config_class = configs.get(config_name, Config)
    
    # Valider la configuration
    errors = config_class.validate()
    if errors:
        print("\n' ERREURS DE CONFIGURATION:")
        for error in errors:
            print(f"   " {error}")
        print("\n' Corrigez ces erreurs avant de continuer\n")
        return None
    
    return config_class


def get_config_summary():
    """Retourne un resume de la configuration"""
    return {
        'capital': Config.INITIAL_CAPITAL,
        'risk_per_trade': Config.RISK_PER_TRADE,
        'max_drawdown': Config.MAX_DRAWDOWN,
        'strategies_count': len([s for s in Config.ACTIVE_STRATEGIES if s['enabled']]),
        'symbols_to_trade': Config.SYMBOLS_TO_TRADE,
        'testnet': Config.TESTNET
    }


# ===================================================================
# MAIN (pour test)
# ===================================================================

if __name__ == "__main__":
    print("\n Test de la configuration\n")
    
    # Charger la config par defaut
    config = load_config('default')
    
    if config:
        # Afficher la configuration
        config.print_config()
        
        # Afficher le resume
        summary = get_config_summary()
        print(""" R"SUM":")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("\n""| Configuration valide!\n")
    else:
        print("\n' Configuration invalide\n")
