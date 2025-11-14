"""
Constants pour The Bot
Toutes les constantes et valeurs fixes utilisÃƒÂ©es dans le systÃƒÂ¨me
"""

from enum import Enum


# ============================================================================
# CAPITAL ET TRADING
# ============================================================================

# Capital initial
INITIAL_CAPITAL = 1000  # USDC

# Tailles de position
MIN_ORDER_SIZE = 50  # Minimum Binance (USDC)
MAX_POSITION_SIZE = 0.25  # 25% du capital max par trade
MIN_POSITION_SIZE = 0.01  # 1% du capital min par trade

# Limites de positions
MAX_OPEN_POSITIONS = 20  # Maximum de positions simultanÃƒÂ©es
MAX_POSITIONS_PER_SYMBOL = 1  # 1 position max par symbole

# Constantes pour position_sizing.py
MAX_CORRELATION_ALLOWED = 0.70  # CorrÃƒÂ©lation max entre positions (70%)
MIN_POSITION_SIZE_PCT = 0.01    # Taille minimum en % du capital (1%)
MAX_POSITION_SIZE_PCT = 0.25    # Taille maximum en % du capital (25%)


# ============================================================================
# GESTION DU RISQUE
# ============================================================================

# Risque par trade
RISK_PER_TRADE = 0.02  # 2% du capital max par trade
MAX_DAILY_LOSS = 0.05  # 5% de perte max par jour
MAX_DRAWDOWN = 0.08  # 8% de drawdown max global

# Stop Loss et Take Profit
DEFAULT_STOP_LOSS_PERCENT = 0.02  # 2% stop loss
DEFAULT_TAKE_PROFIT_PERCENT = 0.04  # 4% take profit (ratio 1:2)
MIN_RISK_REWARD_RATIO = 1.5  # Ratio risque/reward minimum

# Trailing stops
TRAILING_STOP_ACTIVATION = 0.015  # Active aprÃƒÂ¨s 1.5% de profit
TRAILING_STOP_CALLBACK = 0.01  # Callback de 1%


# ============================================================================
# EXECUTION DES ORDRES
# ============================================================================

# Timeouts et retry
ORDER_TIMEOUT = 5000  # 5 secondes
ORDER_RETRY_ATTEMPTS = 3
ORDER_RETRY_DELAY = 1  # 1 seconde entre retry

# Slippage et fees
SLIPPAGE_TOLERANCE = 0.002  # 0.2% de slippage tolÃƒÂ©rÃƒÂ©
BINANCE_MAKER_FEE = 0.001  # 0.1% maker fee
BINANCE_TAKER_FEE = 0.001  # 0.1% taker fee

# Types d'ordres
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_STOP_LOSS = "STOP_LOSS_LIMIT"
ORDER_TYPE_TAKE_PROFIT = "TAKE_PROFIT_LIMIT"


# ============================================================================
# STRATEGIES
# ============================================================================

# Allocation des stratÃƒÂ©gies (doit sommer ÃƒÂ  1.0)
STRATEGY_ALLOCATIONS = {
    'scalping': 0.40,
    'momentum': 0.25,
    'mean_reversion': 0.20,
    'pattern': 0.10,
    'ml': 0.05
}

# ParamÃƒÂ¨tres Scalping
SCALPING_MIN_SPREAD = 0.0005  # 0.05% spread minimum
SCALPING_TARGET_PROFIT = 0.008  # 0.8% profit cible
SCALPING_MAX_HOLD_TIME = 300  # 5 minutes max

# ParamÃƒÂ¨tres Momentum
MOMENTUM_LOOKBACK_PERIOD = 20  # PÃƒÂ©riode de lookback
MOMENTUM_MIN_STRENGTH = 0.6  # Force minimum du momentum
MOMENTUM_TREND_CONFIRMATION = 3  # 3 bougies de confirmation

# ParamÃƒÂ¨tres Mean Reversion
MEAN_REVERSION_BB_PERIOD = 20  # PÃƒÂ©riode Bollinger Bands
MEAN_REVERSION_BB_STD = 2  # Ãƒâ€°carts-types BB
MEAN_REVERSION_RSI_OVERSOLD = 30  # RSI survendu
MEAN_REVERSION_RSI_OVERBOUGHT = 70  # RSI surachetÃƒÂ©

# ParamÃƒÂ¨tres Pattern Recognition
PATTERN_MIN_CONFIDENCE = 0.65  # Confiance minimum pour un pattern
PATTERN_LOOKBACK = 50  # Bougies ÃƒÂ  analyser


# ============================================================================
# MACHINE LEARNING
# ============================================================================

# Seuils de confiance
ML_CONFIDENCE_THRESHOLD = 0.65  # Confiance minimum pour trade
ML_MIN_SAMPLES = 1000  # Minimum de samples pour entraÃƒÂ®nement

# Features
FEATURE_COUNT = 30  # Nombre de features utilisÃƒÂ©es
FEATURE_ENGINEERING_PERIODS = [5, 10, 20, 50, 100]  # PÃƒÂ©riodes pour features

# RÃƒÂ©entraÃƒÂ®nement
RETRAIN_FREQUENCY = 86400  # 24h entre rÃƒÂ©entraÃƒÂ®nements
MIN_TRADES_BEFORE_RETRAIN = 100  # Minimum de trades avant rÃƒÂ©entraÃƒÂ®nement

# ModÃƒÂ¨les
ML_MODELS = ['xgboost', 'lightgbm', 'random_forest']
ML_ENSEMBLE_VOTING = 'soft'  # Soft voting pour ensemble


# ============================================================================
# SYMBOLES ET MARCHÃƒâ€°S
# ============================================================================

# SÃƒÂ©lection des symboles
SYMBOLS_TO_SCAN = 100  # Top 100 par volume ÃƒÂ  scanner
SYMBOLS_TO_TRADE = 20  # Top 20 aprÃƒÂ¨s scoring ÃƒÂ  trader
MIN_VOLUME_24H = 10_000_000  # 10M$ de volume 24h minimum

# Filtres de qualitÃƒÂ©
MAX_SPREAD_PERCENT = 0.002  # 0.2% spread maximum
MIN_VOLATILITY = 0.01  # 1% volatilitÃƒÂ© minimum
MAX_VOLATILITY = 0.10  # 10% volatilitÃƒÂ© maximum

# Quote assets acceptÃƒÂ©s
ACCEPTED_QUOTE_ASSETS = ['USDT', 'USDC', 'BUSD']
PREFERRED_QUOTE_ASSET = 'USDC'


# ============================================================================
# TIMEFRAMES
# ============================================================================

# Timeframe principal
DEFAULT_TIMEFRAME = '5m'

# Timeframes disponibles
TIMEFRAMES = {
    '1m': 60,
    '3m': 180,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}

# Multi-timeframe analysis
MTF_TIMEFRAMES = ['5m', '15m', '1h']  # Timeframes pour analyse multi-TF


# ============================================================================
# INDICATEURS TECHNIQUES
# ============================================================================

# PÃƒÂ©riodes par dÃƒÂ©faut
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 200
SMA_PERIOD = 20

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR
ATR_PERIOD = 14

# ADX
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25  # ADX > 25 = tendance forte

# Stochastic
STOCH_PERIOD = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3


# ============================================================================
# PERFORMANCE ET OPTIMISATION
# ============================================================================

# Threads
MAX_THREADS = 4  # Maximum de threads worker
THREAD_POOL_SIZE = 4

# Buffers
TICK_BUFFER_SIZE = 5000  # Taille du buffer de ticks
KLINE_BUFFER_SIZE = 1000  # Taille du buffer de klines

# MÃƒÂ©moire
MAX_MEMORY_MB = 2000  # 2GB maximum
MEMORY_CHECK_INTERVAL = 60  # VÃƒÂ©rifier toutes les 60 secondes

# Cache
CACHE_TTL_MARKET_DATA = 10  # 10 secondes pour market data
CACHE_TTL_INDICATORS = 60  # 60 secondes pour indicateurs
CACHE_TTL_ML_PREDICTION = 300  # 5 minutes pour ML


# ============================================================================
# WEBSOCKETS ET API
# ============================================================================

# Reconnexion
WS_RECONNECT_DELAY = 5  # 5 secondes
WS_MAX_RECONNECT_ATTEMPTS = 10
WS_PING_INTERVAL = 30  # Ping toutes les 30 secondes

# Rate limiting
API_RATE_LIMIT_PER_MINUTE = 1200
API_WEIGHT_LIMIT_PER_MINUTE = 6000
API_ORDER_LIMIT_PER_10S = 100

# Timeouts
API_TIMEOUT = 10  # 10 secondes
WS_TIMEOUT = 30  # 30 secondes


# ============================================================================
# MONITORING ET LOGS
# ============================================================================

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# Sauvegarde des donnÃƒÂ©es
SAVE_INTERVAL = 300  # 5 minutes
BACKUP_INTERVAL = 3600  # 1 heure

# Health checks
HEALTH_CHECK_INTERVAL = 60  # 1 minute
MAX_MISSED_HEARTBEATS = 3

# Performance tracking
PERFORMANCE_WINDOW = 1000  # Derniers 1000 trades
STATS_UPDATE_INTERVAL = 60  # Mise ÃƒÂ  jour stats chaque minute


# ============================================================================
# RÃƒâ€°GIMES DE MARCHÃƒâ€°
# ============================================================================

class MarketRegime(Enum):
    """Types de rÃƒÂ©gimes de marchÃƒÂ©"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    UNKNOWN = "unknown"


# Seuils pour dÃƒÂ©tection de rÃƒÂ©gime
TRENDING_ADX_THRESHOLD = 25
RANGING_ADX_THRESHOLD = 20
VOLATILITY_THRESHOLD_HIGH = 0.05  # 5%
VOLATILITY_THRESHOLD_LOW = 0.01  # 1%


# ============================================================================
# SIGNAUX
# ============================================================================

class SignalType(Enum):
    """Types de signaux"""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NO_SIGNAL = "no_signal"


class SignalStrength(Enum):
    """Force des signaux"""
    WEAK = 0.3
    MEDIUM = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


# Seuils de signaux
MIN_SIGNAL_STRENGTH = 0.6  # Force minimum pour agir
SIGNAL_EXPIRY_SECONDS = 60  # Signal expire aprÃƒÂ¨s 60s


# ============================================================================
# Ãƒâ€°TATS DU SYSTÃƒË†ME
# ============================================================================

class BotStatus(Enum):
    """Ãƒâ€°tats possibles du bot"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class OrderStatus(Enum):
    """Ãƒâ€°tats des ordres"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ============================================================================
# CHEMINS ET FICHIERS
# ============================================================================

# RÃƒÂ©pertoires
DATA_DIR = "data"
LOGS_DIR = "data/logs"
MODELS_DIR = "data/models"
CACHE_DIR = "data/cache"
BACKUPS_DIR = "data/backups"

# Fichiers
STATE_FILE = "data/bot_state.json"
TRADES_FILE = "data/trades.csv"
POSITIONS_FILE = "data/positions.json"
PERFORMANCE_FILE = "data/performance.json"


# ============================================================================
# NOTIFICATIONS (Optionnel)
# ============================================================================

# Seuils pour notifications
NOTIFY_ON_TRADE = True
NOTIFY_ON_ERROR = True
NOTIFY_ON_DRAWDOWN_PERCENT = 0.05  # Notifier si drawdown > 5%
NOTIFY_ON_PROFIT_PERCENT = 0.10  # Notifier si profit > 10%


# ============================================================================
# BACKTESTING
# ============================================================================

# ParamÃƒÂ¨tres de backtest
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_INITIAL_CAPITAL = 1000
BACKTEST_COMMISSION = 0.001  # 0.1%


# ============================================================================
# PAPER TRADING
# ============================================================================

# Simulation
PAPER_TRADING_ENABLED = True
PAPER_TRADING_SLIPPAGE = 0.001  # 0.1% slippage simulÃƒÂ©


# ============================================================================
# SÃƒâ€°CURITÃƒâ€°
# ============================================================================

# Limites de sÃƒÂ©curitÃƒÂ©
MAX_DAILY_TRADES = 500  # Maximum de trades par jour
MAX_LOSS_STREAK = 10  # Pause aprÃƒÂ¨s 10 pertes consÃƒÂ©cutives
CIRCUIT_BREAKER_LOSS = 0.15  # ArrÃƒÂªt d'urgence si perte > 15%

# Cooldowns
TRADE_COOLDOWN_SECONDS = 1  # 1 seconde entre trades
SYMBOL_COOLDOWN_SECONDS = 10  # 10 secondes avant re-trade mÃƒÂªme symbole


# ============================================================================
# DÃƒâ€°VELOPPEMENT
# ============================================================================

# Modes
DEBUG_MODE = False
DRY_RUN_MODE = True  # Mode simulation par dÃƒÂ©faut
VERBOSE_LOGGING = False

# Tests
TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
TEST_CAPITAL = 1000
