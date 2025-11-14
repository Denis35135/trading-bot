"""
Configuration du bot - Charge depuis .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charge .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================
# MODE & BINANCE
# ============================================

MODE = os.getenv('MODE', 'paper')  # 'paper' ou 'live'
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

# ============================================
# CAPITAL
# ============================================

INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))
MIN_ORDER_SIZE = 50  # Minimum Binance
MAX_POSITION_SIZE = 0.25  # 25% max par trade

# ============================================
# TRADING
# ============================================

SYMBOLS_TO_SCAN = int(os.getenv('SYMBOLS_TO_SCAN', '100'))
SYMBOLS_TO_TRADE = int(os.getenv('SYMBOLS_TO_TRADE', '20'))
KLINE_INTERVAL = os.getenv('KLINE_INTERVAL', '5m')

# ============================================
# RISK MANAGEMENT
# ============================================

RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2%
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.05'))  # 5%
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.08'))  # 8%

# ============================================
# EXECUTION
# ============================================

SLIPPAGE_TOLERANCE = float(os.getenv('SLIPPAGE_TOLERANCE', '0.002'))  # 0.2%
ORDER_TIMEOUT = int(os.getenv('ORDER_TIMEOUT', '5000'))  # ms
RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', '3'))

# ============================================
# STRATEGIES
# ============================================

STRATEGIES = [
    {
        'name': 'scalping',
        'allocation': float(os.getenv('STRATEGY_SCALPING_ALLOCATION', '0.40')),
        'enabled': True
    },
    {
        'name': 'momentum',
        'allocation': float(os.getenv('STRATEGY_MOMENTUM_ALLOCATION', '0.25')),
        'enabled': True
    },
    {
        'name': 'mean_reversion',
        'allocation': float(os.getenv('STRATEGY_MEAN_REVERSION_ALLOCATION', '0.20')),
        'enabled': True
    },
    {
        'name': 'pattern',
        'allocation': float(os.getenv('STRATEGY_PATTERN_ALLOCATION', '0.10')),
        'enabled': True
    },
    {
        'name': 'ml',
        'allocation': float(os.getenv('STRATEGY_ML_ALLOCATION', '0.05')),
        'enabled': True
    }
]

# ============================================
# ML
# ============================================

ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.65'))
FEATURE_COUNT = int(os.getenv('FEATURE_COUNT', '30'))
RETRAIN_FREQUENCY = int(os.getenv('RETRAIN_FREQUENCY', '86400'))  # 24h
MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '1000'))

# ============================================
# MONITORING
# ============================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SAVE_INTERVAL = int(os.getenv('SAVE_INTERVAL', '300'))  # 5 min
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))  # 1 min

# ============================================
# PERFORMANCE
# ============================================

MAX_THREADS = int(os.getenv('MAX_THREADS', '4'))
TICK_BUFFER_SIZE = int(os.getenv('TICK_BUFFER_SIZE', '5000'))
MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB', '2000'))
MAX_CPU_PERCENT = float(os.getenv('MAX_CPU_PERCENT', '80'))

# ============================================
# DATABASE
# ============================================

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')

# ============================================
# REDIS (optionnel)
# ============================================

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# ============================================
# NOTIFICATIONS (optionnel)
# ============================================

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', '')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
EMAIL_FROM = os.getenv('EMAIL_FROM', '')
EMAIL_TO = os.getenv('EMAIL_TO', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

# ============================================
# VALIDATION
# ============================================

def validate_config():
    """Valide la configuration"""
    errors = []
    
    # Check API keys
    if not BINANCE_API_KEY or BINANCE_API_KEY == 'VOTRE_CLE_API_ICI':
        errors.append("BINANCE_API_KEY non configurée dans .env")
    
    if not BINANCE_API_SECRET or BINANCE_API_SECRET == 'VOTRE_SECRET_API_ICI':
        errors.append("BINANCE_API_SECRET non configurée dans .env")
    
    # Check capital
    if INITIAL_CAPITAL < MIN_ORDER_SIZE:
        errors.append(f"INITIAL_CAPITAL ({INITIAL_CAPITAL}) < MIN_ORDER_SIZE ({MIN_ORDER_SIZE})")
    
    # Check strategies allocation
    total_allocation = sum(s['allocation'] for s in STRATEGIES)
    if abs(total_allocation - 1.0) > 0.01:
        errors.append(f"Total allocation stratégies = {total_allocation} (devrait être 1.0)")
    
    return errors

# ============================================
# PATHS
# ============================================

# Dossiers data
DATA_DIR = Path('data')
LOGS_DIR = DATA_DIR / 'logs'
MODELS_DIR = DATA_DIR / 'models'
CACHE_DIR = DATA_DIR / 'cache'
BACKTEST_DIR = DATA_DIR / 'backtest'
HISTORICAL_DIR = DATA_DIR / 'historical'
CONFIGS_DIR = DATA_DIR / 'configs'

# Créer dossiers si nécessaire
for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR, BACKTEST_DIR, HISTORICAL_DIR, CONFIGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# DEBUG INFO
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("CONFIGURATION DU BOT")
    print("="*60)
    print(f"\nMODE: {MODE}")
    print(f"TESTNET: {BINANCE_TESTNET}")
    print(f"CAPITAL: {INITIAL_CAPITAL} USDC")
    print(f"\nAPI KEY: {'✅ Configurée' if BINANCE_API_KEY and BINANCE_API_KEY != 'VOTRE_CLE_API_ICI' else '❌ Non configurée'}")
    print(f"API SECRET: {'✅ Configurée' if BINANCE_API_SECRET and BINANCE_API_SECRET != 'VOTRE_SECRET_API_ICI' else '❌ Non configurée'}")
    print(f"\nSTRATÉGIES:")
    for s in STRATEGIES:
        print(f"  - {s['name']}: {s['allocation']*100}%")
    print(f"\nVALIDATION:")
    errors = validate_config()
    if errors:
        print("❌ ERREURS:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration OK")
    print("="*60)
