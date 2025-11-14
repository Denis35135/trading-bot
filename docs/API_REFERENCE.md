# Ã°Å¸â€œâ€“ API Reference - The Bot

Documentation complÃƒÂ¨te de l'API interne de The Bot.

## Ã°Å¸â€œÅ¡ Table des MatiÃƒÂ¨res

- [Strategies](#strategies)
- [ML Module](#ml-module)
- [Risk Management](#risk-management)
- [Exchange](#exchange)
- [Monitoring](#monitoring)
- [Utils](#utils)

## Ã°Å¸Å½Â¯ Strategies

### BaseStrategy

Classe de base pour toutes les stratÃƒÂ©gies.

```python
from strategies.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__('custom', config)
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """Analyse et gÃƒÂ©nÃƒÂ¨re un signal"""
        # Votre logique ici
        return {
            'type': 'ENTRY',  # ou 'EXIT'
            'side': 'BUY',    # ou 'SELL'
            'price': 50000,
            'confidence': 0.75,
            'take_profit': 51000,
            'stop_loss': 49500,
            'reasons': ['RSI oversold', 'Volume spike']
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs spÃƒÂ©cifiques"""
        df['custom_indicator'] = ...
        return df
```

### ScalpingStrategy

```python
from strategies.scalping import ScalpingStrategy

scalping = ScalpingStrategy(config={
    'min_profit_percent': 0.003,
    'max_holding_time': 300,
    'rsi_oversold': 30
})

# Analyser
signal = scalping.analyze(data)
```

## Ã°Å¸Â¤â€“ ML Module

### FeatureEngineer

```python
from ml.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Calculer les features
features = engineer.calculate_features(df)
# Returns: ndarray shape (n_samples, 30)

# Obtenir les noms
names = engineer.get_feature_names()
# Returns: List of 30 feature names
```

### MLEnsemble

```python
from ml.ensemble import MLEnsemble

ensemble = MLEnsemble({
    'n_estimators': 100,
    'confidence_threshold': 0.65
})

# EntraÃƒÂ®ner
ensemble.train(X_train, y_train, X_val, y_val)

# PrÃƒÂ©dire
signal, confidence = ensemble.predict(X)
# Returns: (1, -1, or 0), confidence (0-1)

# Sauvegarder/Charger
ensemble.save('path/to/models')
ensemble.load('path/to/models')
```

### MLPredictor

```python
from ml.predictor import MLPredictor

predictor = MLPredictor(model_path='data/models/v1')

# PrÃƒÂ©dire
result = predictor.predict(df, symbol='BTCUSDT')
# Returns: {
#     'signal': 1,
#     'confidence': 0.72,
#     'latency_ms': 45.3,
#     'timestamp': ...
# }

# Stats
stats = predictor.get_stats()
```

## Ã°Å¸â€ºÂ¡Ã¯Â¸Â Risk Management

### PositionSizer

```python
from risk.position_sizing import PositionSizer

sizer = PositionSizer(config={
    'method': 'kelly',
    'kelly_fraction': 0.25
})

# Calculer taille
size = sizer.calculate_position_size(
    signal={'confidence': 0.75},
    current_price=50000,
    stop_loss_price=49500,
    market_conditions={'volatility': 0.02}
)
# Returns: {
#     'position_size_usdc': 250,
#     'quantity': 0.005,
#     'risk_amount': 12.5,
#     'risk_reward_ratio': 2.0
# }
```

### RiskMonitor

```python
from risk.risk_monitor import RiskMonitor

monitor = RiskMonitor(config={
    'initial_capital': 1000,
    'max_drawdown': 0.08
})

# Mettre ÃƒÂ  jour
report = monitor.update(
    capital=1050,
    positions={'BTCUSDT': {...}}
)

# Returns: {
#     'risk_level': 'NORMAL',
#     'current_drawdown': 0.02,
#     'required_actions': [],
#     ...
# }
```

## Ã°Å¸â€Å’ Exchange

### BinanceClient

```python
from exchange.binance_client import BinanceClient

client = BinanceClient(
    api_key='your_key',
    api_secret='your_secret'
)

# RÃƒÂ©cupÃƒÂ©rer prix
ticker = client.get_symbol_ticker('BTCUSDT')

# RÃƒÂ©cupÃƒÂ©rer klines
klines = client.get_historical_klines(
    symbol='BTCUSDT',
    interval='5m',
    limit=100
)

# Account info
account = client.get_account()
balance = client.get_asset_balance('USDC')
```

### OrderManager

```python
from exchange.order_manager import OrderManager

order_mgr = OrderManager(client)

# Placer ordre
order = order_mgr.place_order(
    symbol='BTCUSDT',
    side='BUY',
    quantity=0.01,
    order_type='MARKET'
)

# Annuler ordre
order_mgr.cancel_order('BTCUSDT', order_id)

# Obtenir ordres ouverts
open_orders = order_mgr.get_open_orders('BTCUSDT')
```

### MarketData

```python
from exchange.market_data import MarketData

market_data = MarketData(client)

# Klines
df = market_data.get_klines('BTCUSDT', interval='5m', limit=100)

# VWAP
vwap = market_data.calculate_vwap('BTCUSDT', periods=20)

# VolatilitÃƒÂ©
vol = market_data.calculate_volatility('BTCUSDT', periods=20)
```

### WebSocketHandler

```python
from exchange.websocket_handler import WebSocketHandler

ws = WebSocketHandler()

# Callback
def on_kline(data):
    print(f"New candle: {data['close']}")

ws.register_callback('kline', on_kline)
ws.subscribe_kline('BTCUSDT', '5m')
ws.start()
```

## Ã°Å¸â€œÅ  Monitoring

### MetricsCollector

```python
from monitoring.metrics_collector import MetricsCollector, MetricType

collector = MetricsCollector(config={'buffer_size': 10000})

# Enregistrer mÃƒÂ©trique
collector.record(
    MetricType.PNL,
    'daily_pnl',
    50.0,
    metadata={'strategy': 'scalping'}
)

# Snapshot
snapshot = collector.get_snapshot()

# Statistiques
stats = collector.calculate_statistics(
    MetricType.PNL,
    'daily_pnl',
    window_minutes=60
)
```

### PerformanceTracker

```python
from monitoring.performance_tracker import PerformanceTracker

tracker = PerformanceTracker(config={'initial_capital': 1000})

# Enregistrer trade
tracker.record_trade(
    symbol='BTCUSDT',
    strategy='scalping',
    side='BUY',
    entry_price=50000,
    exit_price=50150,
    quantity=0.01,
    profit=1.5,
    entry_time=datetime.now(),
    exit_time=datetime.now()
)

# Snapshot
snapshot = tracker.get_snapshot()
# Returns: PerformanceSnapshot with all metrics

# Rapport dÃƒÂ©taillÃƒÂ©
report = tracker.get_detailed_report()
```

### ReportGenerator

```python
from monitoring.report_generator import ReportGenerator

generator = ReportGenerator(config={'reports_dir': 'data/reports'})

# Dashboard live
dashboard = generator.generate_live_dashboard(
    capital=1050,
    initial_capital=1000,
    positions=[...],
    performance={...},
    risk_metrics={...},
    system_health={...}
)
print(dashboard)

# Rapport quotidien
generator.generate_and_save_daily_report(
    date=datetime.now(),
    performance={...},
    trades=[...],
    top_performers={...},
    alerts=[...]
)
```

## Ã°Å¸â€Â§ Utils

### Indicators

```python
from utils.indicators import Indicators

# RSI
rsi = Indicators.rsi(close_prices, period=14)

# MACD
macd, signal, hist = Indicators.macd(
    close_prices,
    fast=12,
    slow=26,
    signal_period=9
)

# Bollinger Bands
upper, middle, lower = Indicators.bollinger_bands(
    close_prices,
    period=20,
    std_dev=2
)

# ATR
atr = Indicators.atr(high, low, close, period=14)

# Support/Resistance
supports, resistances = Indicators.find_support_resistance(
    high, low, close,
    window=20
)
```

### Logger

```python
from utils.logger import setup_logger

logger = setup_logger(
    name='MyModule',
    level='INFO',
    log_file='data/logs/mymodule.log'
)

logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.debug("Debug message")
```

## Ã°Å¸Ââ€”Ã¯Â¸Â Exemples Complets

### CrÃƒÂ©er une StratÃƒÂ©gie PersonnalisÃƒÂ©e

```python
from strategies.base_strategy import BaseStrategy
from utils.indicators import Indicators
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    """StratÃƒÂ©gie personnalisÃƒÂ©e basÃƒÂ©e sur EMA crossover"""
    
    def __init__(self, config=None):
        super().__init__('my_custom', config or {
            'ema_fast': 12,
            'ema_slow': 26,
            'min_confidence': 0.65
        })
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule EMA fast et slow"""
        df['ema_fast'] = df['close'].ewm(
            span=self.config['ema_fast']
        ).mean()
        df['ema_slow'] = df['close'].ewm(
            span=self.config['ema_slow']
        ).mean()
        return df
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """DÃƒÂ©tecte les crossovers"""
        df = data['df']
        
        if len(df) < 2:
            return None
        
        # Calculer indicateurs
        df = self.calculate_indicators(df)
        
        # VÃƒÂ©rifier crossover
        ema_fast_now = df['ema_fast'].iloc[-1]
        ema_fast_prev = df['ema_fast'].iloc[-2]
        ema_slow_now = df['ema_slow'].iloc[-1]
        ema_slow_prev = df['ema_slow'].iloc[-2]
        
        # Bullish crossover
        if ema_fast_prev < ema_slow_prev and ema_fast_now > ema_slow_now:
            return {
                'type': 'ENTRY',
                'side': 'BUY',
                'price': df['close'].iloc[-1],
                'confidence': 0.70,
                'take_profit': df['close'].iloc[-1] * 1.02,
                'stop_loss': df['close'].iloc[-1] * 0.98,
                'reasons': ['EMA bullish crossover']
            }
        
        # Bearish crossover
        if ema_fast_prev > ema_slow_prev and ema_fast_now < ema_slow_now:
            return {
                'type': 'ENTRY',
                'side': 'SELL',
                'price': df['close'].iloc[-1],
                'confidence': 0.70,
                'take_profit': df['close'].iloc[-1] * 0.98,
                'stop_loss': df['close'].iloc[-1] * 1.02,
                'reasons': ['EMA bearish crossover']
            }
        
        return None

# Utilisation
strategy = MyCustomStrategy()
signal = strategy.analyze({'df': df})
if signal:
    print(f"Signal: {signal['side']} at {signal['price']}")
```

### Pipeline ML Complet

```python
from ml.feature_engineering import FeatureEngineer
from ml.trainer import MLTrainer
from ml.predictor import MLPredictor

# 1. PrÃƒÂ©parer les features
engineer = FeatureEngineer()
features = engineer.calculate_features(df)

# 2. EntraÃƒÂ®ner
trainer = MLTrainer()
X, y = trainer.prepare_training_data(trades_data, ohlcv_data)
results = trainer.train(X, y, save_path='data/models/v1')

print(f"Accuracy: {results['evaluation']['accuracy']:.2%}")

# 3. PrÃƒÂ©dire
predictor = MLPredictor(model_path='data/models/v1')
result = predictor.predict(df, 'BTCUSDT')

if result['signal'] == 1:
    print(f"BUY signal with {result['confidence']:.1%} confidence")
```

### Gestion ComplÃƒÂ¨te d'un Trade

```python
from exchange.binance_client import BinanceClient
from risk.position_sizing import PositionSizer
from exchange.order_manager import OrderManager

# Setup
client = BinanceClient(api_key, api_secret)
sizer = PositionSizer(config={'method': 'kelly'})
order_mgr = OrderManager(client)

# Signal
signal = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'confidence': 0.75,
    'entry_price': 50000,
    'stop_loss': 49500,
    'take_profit': 51000
}

# Calculer taille
position = sizer.calculate_position_size(
    signal=signal,
    current_price=signal['entry_price'],
    stop_loss_price=signal['stop_loss'],
    market_conditions={'volatility': 0.02}
)

# Placer ordre
if position['position_size_usdc'] >= 50:  # Min Binance
    order = order_mgr.place_order(
        symbol=signal['symbol'],
        side=signal['side'],
        quantity=position['quantity'],
        order_type='MARKET'
    )
    
    print(f"Order placed: {order.client_order_id}")
    print(f"Quantity: {order.quantity}")
    print(f"Price: {order.price}")
```

## Ã°Å¸â€Â Types & Enums

### OrderStatus

```python
from exchange.order_manager import OrderStatus

OrderStatus.NEW          # Nouvel ordre
OrderStatus.FILLED       # Rempli
OrderStatus.PARTIALLY_FILLED  # Partiellement rempli
OrderStatus.CANCELED     # AnnulÃƒÂ©
OrderStatus.REJECTED     # RejetÃƒÂ©
```

### RiskLevel

```python
from risk.risk_monitor import RiskLevel

RiskLevel.NORMAL         # Risque normal
RiskLevel.ELEVATED       # Risque ÃƒÂ©levÃƒÂ©
RiskLevel.HIGH           # Risque haut
RiskLevel.CRITICAL       # Risque critique
RiskLevel.EMERGENCY      # Urgence
```

### MetricType

```python
from monitoring.metrics_collector import MetricType

MetricType.CAPITAL       # MÃƒÂ©triques de capital
MetricType.PNL           # P&L
MetricType.POSITION      # Positions
MetricType.TRADE         # Trades
MetricType.RISK          # Risque
MetricType.EXECUTION     # ExÃƒÂ©cution
MetricType.STRATEGY      # StratÃƒÂ©gies
MetricType.SYSTEM        # SystÃƒÂ¨me
```

## Ã°Å¸â€œÂ Callbacks & Events

### Strategy Events

```python
from strategies.strategy_manager import StrategyManager

mgr = StrategyManager(...)

# Callback sur nouveau signal
def on_signal(signal):
    print(f"New signal: {signal['side']} {signal['symbol']}")

mgr.on_signal = on_signal
```

### WebSocket Events

```python
from exchange.websocket_handler import WebSocketHandler

ws = WebSocketHandler()

# Callbacks
ws.register_callback('kline', lambda data: print(f"Kline: {data}"))
ws.register_callback('trade', lambda data: print(f"Trade: {data}"))
ws.register_callback('ticker', lambda data: print(f"Ticker: {data}"))
```

## Ã°Å¸Â§Âª Testing

### Mock Objects

```python
from tests.mocks import MockBinanceClient, MockOrderManager

# Client mock
client = MockBinanceClient(should_fail=False)
ticker = client.get_symbol_ticker('BTCUSDT')

# Order manager mock
order_mgr = MockOrderManager()
order = order_mgr.place_order('BTCUSDT', 'BUY', 0.01)
```

## Ã°Å¸â€œÅ¡ Ressources SupplÃƒÂ©mentaires

- [Configuration Guide](CONFIGURATION.md) - Configuration dÃƒÂ©taillÃƒÂ©e
- [Strategies Guide](STRATEGIES.md) - Guide des stratÃƒÂ©gies
- [Troubleshooting](TROUBLESHOOTING.md) - RÃƒÂ©solution de problÃƒÂ¨mes

## Ã°Å¸â€™Â¡ Best Practices

### Error Handling

```python
try:
    result = some_function()
except Exception as e:
    logger.error(f"Error: {e}")
    # Handle gracefully
```

### Logging

```python
logger.info("Normal operation")
logger.warning("Unusual but not critical")
logger.error("Error occurred")
logger.debug("Detailed debug info")
```

### Resource Management

```python
# Always use context managers
with open('file.txt', 'r') as f:
    data = f.read()

# Clean up resources
def cleanup():
    ws.stop()
    client.close()
```

---

**API complÃƒÂ¨te pour ÃƒÂ©tendre The Bot ! Ã°Å¸Å¡â‚¬**

*DerniÃƒÂ¨re mise ÃƒÂ  jour : Octobre 2024*
