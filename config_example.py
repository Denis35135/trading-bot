"""
Configuration Template pour The Bot
Copiez ce fichier en config.py et ajustez les paramÃƒÂ¨tres
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
    
    # Drawdown maximum avant arrÃƒÂªt (8% = 0.08)
    MAX_DRAWDOWN = 0.08
    
    # Taille minimale d'ordre (minimum Binance)
    MIN_ORDER_SIZE = 50  # USDC
    
    # Taille maximale de position (25% du capital)
    MAX_POSITION_SIZE = 0.25
    
    # ===================================================================
    # STRATÃƒâ€°GIES
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
    
    # Nombre de symboles ÃƒÂ  scanner
    SYMBOLS_TO_SCAN = 100
    
    # Nombre de symboles ÃƒÂ  trader
    SYMBOLS_TO_TRADE = 20
    
    # Intervalle de scan (secondes)
    SCAN_INTERVAL = 300  # 5 minutes
    
    # Volume minimum 24h (USDC)
    MIN_VOLUME_24H = 10_000_000  # 10M
    
    # Spread maximum (%)
    MAX_SPREAD_PERCENT = 0.002  # 0.2%
    
    # Range de volatilitÃƒÂ© acceptable (%)
    VOLATILITY_RANGE = (0.02, 0.08)  # 2% - 8%
    
    # Blacklist de symboles
    BLACKLISTED_SYMBOLS = [
        # Ajoutez les symboles ÃƒÂ  ignorer
        # 'LUNAUSDC',  # Exemple
    ]
    
    # Forcer certains symboles (optionnel)
    FORCED_SYMBOLS = [
        # 'BTCUSDC',
        # 'ETHUSDC',
    ]
    
    # ===================================================================
    # EXÃƒâ€°CUTION DES ORDRES
    # ===================================================================
    
    # TolÃƒÂ©rance au slippage (0.2% = 0.002)
    SLIPPAGE_TOLERANCE = 0.002
    
    # Timeout pour les ordres (ms)
    ORDER_TIMEOUT = 5000  # 5 secondes
    
    # Nombre de tentatives en cas d'ÃƒÂ©chec
    RETRY_ATTEMPTS = 3
    
    # DÃƒÂ©lai entre les tentatives (secondes)
    RETRY_DELAY = 1
    
    # ===================================================================
    # MACHINE LEARNING
    # ===================================================================
    
    # Seuil de confiance minimum
    ML_CONFIDENCE_THRESHOLD = 0.65
    
    # Nombre de features
    FEATURE_COUNT = 30
    
    # FrÃƒÂ©quence de rÃƒÂ©entraÃƒÂ®nement (secondes)
    RETRAIN_FREQUENCY = 86400  # 24h
    
    # Taille minimale du dataset pour entraÃƒÂ®nement
    MIN_TRAINING_SAMPLES = 10000
    
    # ===================================================================
    # CIRCUIT BREAKERS
    # ===================================================================
    
    # Niveaux de circuit breakers
    CIRCUIT_BREAKER_LEVELS = {
        'warning': {
            'drawdown': 0.03,  # 3%
            'daily_loss': 0.03,  # 3%
            'action': 'reduce_positions'  # RÃƒÂ©duire de 50%
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
    
    # MÃƒÂ©moire maximum (MB)
    MAX_MEMORY_MB = 2000  # 2GB
    
    # Utiliser Redis pour le cache
    USE_REDIS = True
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    # DurÃƒÂ©e du cache (secondes)
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
    # BASE DE DONNÃƒâ€°ES (optionnel)
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
    
    # Slippage simulÃƒÂ© (%)
    BACKTEST_SLIPPAGE = 0.0005  # 0.05%
    
    # ===================================================================
    # AVANCÃƒâ€°
    # ===================================================================
    
    # Activer le mode debug (plus de logs)
    DEBUG_MODE = False
    
    # Sauvegarder tous les signaux (pour analyse)
    SAVE_ALL_SIGNALS = True
    
    # Intervalle de sauvegarde des mÃƒÂ©triques (secondes)
    METRICS_SAVE_INTERVAL = 60
    
    # Activer le profiling de performance
    ENABLE_PROFILING = False
    
    # Limiter le nombre de positions simultanÃƒÂ©es
    MAX_CONCURRENT_POSITIONS = 20
    
    # Temps minimum entre deux trades sur le mÃƒÂªme symbole (secondes)
    MIN_TIME_BETWEEN_TRADES = 60  # 1 minute
    
    # ===================================================================
    # VALIDATION
    # ===================================================================
    
    @classmethod
    def validate(cls):
        """Valide la configuration"""
        errors = []
        
        # VÃƒÂ©rifier les clÃƒÂ©s API
        if cls.BINANCE_API_KEY == 'your_api_key_here':
            errors.append("BINANCE_API_KEY non configurÃƒÂ©e")
        
        if cls.BINANCE_SECRET_KEY == 'your_secret_key_here':
            errors.append("BINANCE_SECRET_KEY non configurÃƒÂ©e")
        
        # VÃƒÂ©rifier le capital
        if cls.INITIAL_CAPITAL < 100:
            errors.append("INITIAL_CAPITAL trop faible (minimum 100 USDC)")
        
        # VÃƒÂ©rifier les allocations de stratÃƒÂ©gies
        total_allocation = sum(s['allocation'] for s in cls.ACTIVE_STRATEGIES if s['enabled'])
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Total allocation stratÃƒÂ©gies doit ÃƒÂªtre 1.0 (actuellement: {total_allocation})")
        
        # VÃƒÂ©rifier les risques
        if cls.RISK_PER_TRADE > 0.05:
            errors.append("RISK_PER_TRADE trop ÃƒÂ©levÃƒÂ© (maximum 5%)")
        
        if cls.MAX_DRAWDOWN > 0.15:
            errors.append("MAX_DRAWDOWN trop ÃƒÂ©levÃƒÂ© (maximum 15%)")
        
        # VÃƒÂ©rifier Redis si activÃƒÂ©
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
        print("Ã¢Å¡â„¢Ã¯Â¸Â  CONFIGURATION THE BOT")
        print("="*60 + "\n")
        
        print("Ã°Å¸â€œÅ  CAPITAL & RISQUE")
        print(f"   Capital Initial:    ${cls.INITIAL_CAPITAL:,.2f}")
        print(f"   Risk/Trade:         {cls.RISK_PER_TRADE:.1%}")
        print(f"   Max Daily Loss:     {cls.MAX_DAILY_LOSS:.1%}")
        print(f"   Max Drawdown:       {cls.MAX_DRAWDOWN:.1%}")
        
        print("\nÃ°Å¸Å½Â¯ STRATÃƒâ€°GIES ACTIVES")
        for strategy in cls.ACTIVE_STRATEGIES:
            if strategy['enabled']:
                print(f"   {strategy['name']:15} {strategy['allocation']:5.1%} "
                      f"(confidence: {strategy['min_confidence']:.0%})")
        
        print("\nÃ°Å¸â€œË† MARKET SCANNER")
        print(f"   Symboles scannÃƒÂ©s:   {cls.SYMBOLS_TO_SCAN}")
        print(f"   Symboles tradÃƒÂ©s:    {cls.SYMBOLS_TO_TRADE}")
        print(f"   Volume min 24h:     ${cls.MIN_VOLUME_24H:,.0f}")
        print(f"   Spread max:         {cls.MAX_SPREAD_PERCENT:.2%}")
        
        print("\nÃ°Å¸â€™Â» PERFORMANCE")
        print(f"   Max Threads:        {cls.MAX_THREADS}")
        print(f"   Max Memory:         {cls.MAX_MEMORY_MB} MB")
        print(f"   Redis:              {'Ã¢Å“â€¦ ActivÃƒÂ©' if cls.USE_REDIS else 'Ã¢ÂÅ’ DÃƒÂ©sactivÃƒÂ©'}")
        
        print("\nÃ°Å¸â€â€ NOTIFICATIONS")
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
# CONFIGURATIONS PRÃƒâ€°DÃƒâ€°FINIES
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
    Charge une configuration spÃƒÂ©cifique
    
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
        print("\nÃ¢ÂÅ’ ERREURS DE CONFIGURATION:")
        for error in errors:
            print(f"   Ã¢â‚¬Â¢ {error}")
        print("\nÃ°Å¸â€™Â¡ Corrigez ces erreurs avant de continuer\n")
        return None
    
    return config_class


def get_config_summary():
    """Retourne un rÃƒÂ©sumÃƒÂ© de la configuration"""
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
    print("\nÃ°Å¸Â§Âª Test de la configuration\n")
    
    # Charger la config par dÃƒÂ©faut
    config = load_config('default')
    
    if config:
        # Afficher la configuration
        config.print_config()
        
        # Afficher le rÃƒÂ©sumÃƒÂ©
        summary = get_config_summary()
        print("Ã°Å¸â€œâ€¹ RÃƒâ€°SUMÃƒâ€°:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("\nÃ¢Å“â€¦ Configuration valide!\n")
    else:
        print("\nÃ¢ÂÅ’ Configuration invalide\n")