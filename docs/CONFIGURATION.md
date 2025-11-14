\# Ã¢Å¡â„¢Ã¯Â¸Â Guide de Configuration - The Bot



Guide complet pour configurer The Bot selon vos besoins.



\## Ã°Å¸â€œâ€¹ Fichiers de Configuration



The Bot utilise 2 fichiers de configuration :



1\. \*\*`.env`\*\* - Variables d'environnement (clÃƒÂ©s API, secrets)

2\. \*\*`config.py`\*\* - ParamÃƒÂ¨tres du bot (risque, stratÃƒÂ©gies, etc.)



\## Ã°Å¸â€Â Configuration .env



\### Structure du Fichier



```env

\# ===================================

\# BINANCE API

\# ===================================

BINANCE\_API\_KEY=votre\_api\_key

BINANCE\_API\_SECRET=votre\_secret

BINANCE\_TESTNET=false



\# ===================================

\# MODE DE TRADING

\# ===================================

TRADING\_MODE=paper  # paper ou live



\# ===================================

\# CAPITAL

\# ===================================

INITIAL\_CAPITAL=1000

MAX\_CAPITAL=10000



\# ===================================

\# NOTIFICATIONS (Optionnel)

\# ===================================

TELEGRAM\_ENABLED=false

TELEGRAM\_TOKEN=

TELEGRAM\_CHAT\_ID=



DISCORD\_ENABLED=false

DISCORD\_WEBHOOK=



\# ===================================

\# BASE DE DONNÃƒâ€°ES (Optionnel)

\# ===================================

DATABASE\_URL=sqlite:///data/bot.db

REDIS\_URL=redis://localhost:6379/0



\# ===================================

\# LOGGING

\# ===================================

LOG\_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

LOG\_TO\_FILE=true

LOG\_TO\_CONSOLE=true

```



\### Variables Importantes



| Variable | Description | Valeurs | DÃƒÂ©faut |

|----------|-------------|---------|--------|

| `BINANCE\_API\_KEY` | ClÃƒÂ© API Binance | String | Requis |

| `BINANCE\_API\_SECRET` | Secret API Binance | String | Requis |

| `TRADING\_MODE` | Mode de trading | paper/live | paper |

| `INITIAL\_CAPITAL` | Capital de dÃƒÂ©part | Number (USDC) | 1000 |

| `LOG\_LEVEL` | Niveau de logs | DEBUG/INFO/WARNING/ERROR | INFO |



\## Ã¢Å¡â„¢Ã¯Â¸Â Configuration config.py



\### Structure Globale



```python

class Config:

&nbsp;   """Configuration globale du bot"""

&nbsp;   

&nbsp;   # CAPITAL \& RISQUE

&nbsp;   INITIAL\_CAPITAL = 1000

&nbsp;   RISK\_PER\_TRADE = 0.02          # 2% par trade

&nbsp;   MAX\_DAILY\_LOSS = 0.05          # 5% perte max/jour

&nbsp;   MAX\_DRAWDOWN = 0.08            # 8% drawdown max

&nbsp;   

&nbsp;   # STRATÃƒâ€°GIES

&nbsp;   ACTIVE\_STRATEGIES = \[

&nbsp;       {'name': 'scalping', 'enabled': True, 'allocation': 0.40},

&nbsp;       {'name': 'momentum', 'enabled': True, 'allocation': 0.25},

&nbsp;       {'name': 'mean\_reversion', 'enabled': True, 'allocation': 0.20},

&nbsp;       {'name': 'pattern', 'enabled': True, 'allocation': 0.10},

&nbsp;       {'name': 'ml', 'enabled': True, 'allocation': 0.05}

&nbsp;   ]

&nbsp;   

&nbsp;   # EXÃƒâ€°CUTION

&nbsp;   MIN\_ORDER\_SIZE = 50            # USDC minimum

&nbsp;   MAX\_POSITION\_SIZE = 0.25       # 25% max du capital

&nbsp;   SLIPPAGE\_TOLERANCE = 0.002     # 0.2%

&nbsp;   ORDER\_TIMEOUT = 5              # secondes

&nbsp;   

&nbsp;   # MACHINE LEARNING

&nbsp;   ML\_CONFIDENCE\_THRESHOLD = 0.65

&nbsp;   FEATURE\_COUNT = 30

&nbsp;   RETRAIN\_FREQUENCY = 86400      # 24h en secondes

&nbsp;   

&nbsp;   # MARCHÃƒâ€°

&nbsp;   SYMBOLS\_TO\_SCAN = 100          # Top 100 par volume

&nbsp;   SYMBOLS\_TO\_TRADE = 20          # Top 20 aprÃƒÂ¨s scoring

&nbsp;   MIN\_VOLUME\_24H = 10000000      # $10M minimum

&nbsp;   MAX\_SPREAD\_PERCENT = 0.005     # 0.5% max

&nbsp;   

&nbsp;   # PERFORMANCE

&nbsp;   MAX\_THREADS = 4

&nbsp;   TICK\_BUFFER\_SIZE = 5000

&nbsp;   MAX\_MEMORY\_MB = 2000           # 2GB max

```



\## Ã°Å¸Å½Â¯ Configuration des StratÃƒÂ©gies



\### Scalping (40% allocation)



```python

SCALPING\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'allocation': 0.40,

&nbsp;   'min\_profit\_percent': 0.003,    # 0.3% minimum

&nbsp;   'max\_holding\_time': 300,        # 5 minutes max

&nbsp;   'rsi\_oversold': 30,

&nbsp;   'rsi\_overbought': 70,

&nbsp;   'use\_vwap': True,

&nbsp;   'use\_orderflow': True

}

```



\### Momentum (25% allocation)



```python

MOMENTUM\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'allocation': 0.25,

&nbsp;   'breakout\_threshold': 0.02,     # 2% breakout

&nbsp;   'volume\_multiplier': 2.0,       # 2x volume normal

&nbsp;   'confirmation\_candles': 2,

&nbsp;   'use\_multiple\_timeframes': True

}

```



\### Mean Reversion (20% allocation)



```python

MEAN\_REVERSION\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'allocation': 0.20,

&nbsp;   'bb\_std': 2.0,                  # Bollinger Bands

&nbsp;   'rsi\_extreme': 25,              # RSI < 25 ou > 75

&nbsp;   'z\_score\_threshold': 2.0,

&nbsp;   'max\_distance\_from\_mean': 0.05  # 5%

}

```



\### Pattern Recognition (10% allocation)



```python

PATTERN\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'allocation': 0.10,

&nbsp;   'patterns': \[

&nbsp;       'double\_bottom', 'double\_top',

&nbsp;       'head\_shoulders', 'triangle'

&nbsp;   ],

&nbsp;   'min\_confidence': 0.70,

&nbsp;   'lookback\_periods': 50

}

```



\### Machine Learning (5% allocation)



```python

ML\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'allocation': 0.05,

&nbsp;   'models': \['lgb', 'xgb', 'rf'],

&nbsp;   'confidence\_threshold': 0.65,

&nbsp;   'retrain\_hour': 3,              # 3h du matin

&nbsp;   'min\_samples': 10000,

&nbsp;   'use\_ensemble': True

}

```



\## Ã°Å¸â€ºÂ¡Ã¯Â¸Â Configuration du Risque



\### Position Sizing



```python

POSITION\_SIZING = {

&nbsp;   'method': 'kelly',              # kelly, fixed, volatility

&nbsp;   'kelly\_fraction': 0.25,         # 25% du Kelly

&nbsp;   'min\_position\_pct': 0.01,       # 1% minimum

&nbsp;   'max\_position\_pct': 0.25,       # 25% maximum

&nbsp;   'adjust\_for\_volatility': True,

&nbsp;   'adjust\_for\_correlation': True

}

```



\### Stop Loss \& Take Profit



```python

STOP\_LOSS\_CONFIG = {

&nbsp;   'type': 'trailing',             # fixed, trailing, volatility

&nbsp;   'default\_percent': 0.02,        # 2%

&nbsp;   'trailing\_percent': 0.015,      # 1.5%

&nbsp;   'use\_atr': True,

&nbsp;   'atr\_multiplier': 2.0

}



TAKE\_PROFIT\_CONFIG = {

&nbsp;   'type': 'dynamic',              # fixed, dynamic

&nbsp;   'default\_percent': 0.03,        # 3%

&nbsp;   'scale\_out': True,

&nbsp;   'scale\_out\_levels': \[0.02, 0.04, 0.06]

}

```



\### Circuit Breakers



```python

CIRCUIT\_BREAKER\_LEVELS = {

&nbsp;   'level\_1': {

&nbsp;       'drawdown': 0.03,           # 3%

&nbsp;       'action': 'reduce\_positions',

&nbsp;       'reduce\_by': 0.5            # 50%

&nbsp;   },

&nbsp;   'level\_2': {

&nbsp;       'drawdown': 0.05,           # 5%

&nbsp;       'action': 'halt\_new\_trades',

&nbsp;       'resume\_after': 3600        # 1h

&nbsp;   },

&nbsp;   'level\_3': {

&nbsp;       'drawdown': 0.08,           # 8%

&nbsp;       'action': 'close\_all',

&nbsp;       'notify': True

&nbsp;   }

}

```



\## Ã°Å¸â€œÅ  Configuration du Monitoring



\### Dashboard



```python

DASHBOARD\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'refresh\_interval': 10,         # secondes

&nbsp;   'show\_positions': True,

&nbsp;   'show\_performance': True,

&nbsp;   'show\_risk': True,

&nbsp;   'compact\_mode': False

}

```



\### Logs



```python

LOGGING\_CONFIG = {

&nbsp;   'level': 'INFO',

&nbsp;   'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',

&nbsp;   'file': 'data/logs/bot.log',

&nbsp;   'max\_bytes': 10485760,          # 10MB

&nbsp;   'backup\_count': 5,

&nbsp;   'console': True,

&nbsp;   'colorized': True

}

```



\### MÃƒÂ©triques



```python

METRICS\_CONFIG = {

&nbsp;   'collect\_interval': 60,         # secondes

&nbsp;   'buffer\_size': 10000,

&nbsp;   'export\_interval': 300,         # 5 minutes

&nbsp;   'export\_format': 'json',

&nbsp;   'track\_latency': True,

&nbsp;   'track\_memory': True

}

```



\## Ã°Å¸â€â€ Configuration des Notifications



\### Telegram



```python

TELEGRAM\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'token': 'votre\_token',

&nbsp;   'chat\_id': 'votre\_chat\_id',

&nbsp;   'notify\_on': \[

&nbsp;       'trade\_opened',

&nbsp;       'trade\_closed',

&nbsp;       'high\_profit',              # > 5%

&nbsp;       'high\_loss',                # > 3%

&nbsp;       'circuit\_breaker',

&nbsp;       'error'

&nbsp;   ],

&nbsp;   'quiet\_hours': {

&nbsp;       'enabled': True,

&nbsp;       'start': '23:00',

&nbsp;       'end': '08:00'

&nbsp;   }

}

```



\### Discord



```python

DISCORD\_CONFIG = {

&nbsp;   'enabled': False,

&nbsp;   'webhook\_url': 'votre\_webhook',

&nbsp;   'notify\_on': \['trade\_closed', 'circuit\_breaker'],

&nbsp;   'embed\_color': 0x00ff00

}

```



\## Ã°Å¸Å’Â Configuration Exchange



\### Binance



```python

BINANCE\_CONFIG = {

&nbsp;   'api\_key': os.getenv('BINANCE\_API\_KEY'),

&nbsp;   'api\_secret': os.getenv('BINANCE\_API\_SECRET'),

&nbsp;   'testnet': False,

&nbsp;   'futures': False,

&nbsp;   'margin': False,

&nbsp;   'timeout': 10,

&nbsp;   'recvWindow': 5000,

&nbsp;   'enable\_rate\_limit': True

}

```



\### WebSocket



```python

WEBSOCKET\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'base\_url': 'wss://stream.binance.com:9443/ws',

&nbsp;   'ping\_interval': 20,

&nbsp;   'reconnect\_delay': 5,

&nbsp;   'max\_reconnect\_attempts': 10,

&nbsp;   'streams': \['kline', 'trade', 'ticker']

}

```



\## Ã°Å¸â€™Â¾ Configuration Base de DonnÃƒÂ©es



\### SQLite (Par dÃƒÂ©faut)



```python

DATABASE\_CONFIG = {

&nbsp;   'type': 'sqlite',

&nbsp;   'path': 'data/bot.db',

&nbsp;   'echo': False,

&nbsp;   'pool\_size': 5,

&nbsp;   'backup\_interval': 86400        # 24h

}

```



\### PostgreSQL (Production)



```python

DATABASE\_CONFIG = {

&nbsp;   'type': 'postgresql',

&nbsp;   'host': 'localhost',

&nbsp;   'port': 5432,

&nbsp;   'database': 'thebot',

&nbsp;   'user': 'bot\_user',

&nbsp;   'password': 'secure\_password',

&nbsp;   'pool\_size': 20

}

```



\## Ã°Å¸â€Â§ Configurations AvancÃƒÂ©es



\### Performance



```python

PERFORMANCE\_CONFIG = {

&nbsp;   'use\_numba': True,              # Compilation JIT

&nbsp;   'use\_cython': False,

&nbsp;   'cache\_indicators': True,

&nbsp;   'cache\_ttl': 60,

&nbsp;   'optimize\_memory': True,

&nbsp;   'gc\_interval': 300              # Garbage collection

}

```



\### Backtesting



```python

BACKTEST\_CONFIG = {

&nbsp;   'enabled': False,

&nbsp;   'start\_date': '2023-01-01',

&nbsp;   'end\_date': '2024-01-01',

&nbsp;   'initial\_capital': 10000,

&nbsp;   'commission': 0.001,            # 0.1%

&nbsp;   'slippage': 0.0005              # 0.05%

}

```



\## Ã°Å¸â€œÂ Exemples de Configuration



\### Configuration Conservative



```python

\# Profil: DÃƒÂ©butant / Risk-Averse

RISK\_PER\_TRADE = 0.01              # 1%

MAX\_DAILY\_LOSS = 0.03              # 3%

MAX\_DRAWDOWN = 0.05                # 5%

ML\_CONFIDENCE\_THRESHOLD = 0.75     # Plus exigeant

```



\### Configuration Aggressive



```python

\# Profil: ExpÃƒÂ©rimentÃƒÂ© / Risk-Taker

RISK\_PER\_TRADE = 0.03              # 3%

MAX\_DAILY\_LOSS = 0.08              # 8%

MAX\_DRAWDOWN = 0.12                # 12%

ML\_CONFIDENCE\_THRESHOLD = 0.60     # Moins exigeant

```



\### Configuration Scalping Only



```python

\# Focus sur scalping uniquement

ACTIVE\_STRATEGIES = \[

&nbsp;   {'name': 'scalping', 'enabled': True, 'allocation': 1.0},

]

SYMBOLS\_TO\_TRADE = 30              # Plus de symboles

MIN\_PROFIT\_PERCENT = 0.002         # 0.2% seulement

MAX\_HOLDING\_TIME = 180             # 3 minutes max

```



\## Ã¢Å“â€¦ Validation de la Configuration



\### Script de Test



```bash

\# Tester la configuration

python test\_config.py

```



\*\*VÃƒÂ©rifications effectuÃƒÂ©es :\*\*

\- Ã¢Å“â€¦ Fichier .env existe et est lisible

\- Ã¢Å“â€¦ ClÃƒÂ©s API valides

\- Ã¢Å“â€¦ ParamÃƒÂ¨tres de risque cohÃƒÂ©rents

\- Ã¢Å“â€¦ Allocations des stratÃƒÂ©gies = 100%

\- Ã¢Å“â€¦ Connexion ÃƒÂ  la base de donnÃƒÂ©es

\- Ã¢Å“â€¦ Connexion aux APIs externes



\### Checklist de Configuration



\- \[ ] Fichier `.env` crÃƒÂ©ÃƒÂ© avec clÃƒÂ©s API

\- \[ ] Fichier `config.py` paramÃƒÂ©trÃƒÂ©

\- \[ ] Mode `paper` activÃƒÂ© pour les tests

\- \[ ] Capital initial dÃƒÂ©fini (ex: 1000 USDC)

\- \[ ] StratÃƒÂ©gies activÃƒÂ©es et allouÃƒÂ©es

\- \[ ] ParamÃƒÂ¨tres de risque configurÃƒÂ©s

\- \[ ] Notifications configurÃƒÂ©es (optionnel)

\- \[ ] Tests de connexion passÃƒÂ©s



\## Ã°Å¸â€â€ž Changement de Configuration



\### Passer de Paper ÃƒÂ  Live



```bash

\# 1. Ãƒâ€°diter .env

TRADING\_MODE=live



\# 2. Confirmer dans config.py

PAPER\_TRADING = False



\# 3. RedÃƒÂ©marrer le bot

python main.py --mode live

```



Ã¢Å¡Â Ã¯Â¸Â \*\*ATTENTION\*\* : VÃƒÂ©rifiez 3 fois avant de passer en mode Live !



\### Ajuster le Risque en Direct



```python

\# Pendant que le bot tourne, crÃƒÂ©er un nouveau config

\# Le bot rechargera automatiquement



\# Exemple: RÃƒÂ©duire le risque

RISK\_PER\_TRADE = 0.01  # De 2% ÃƒÂ  1%



\# Sauvegarder et le bot dÃƒÂ©tectera le changement

```



\## Ã°Å¸â€œÅ  Configuration par Environnement



\### DÃƒÂ©veloppement



```python

ENV = 'development'

LOG\_LEVEL = 'DEBUG'

TRADING\_MODE = 'paper'

COLLECT\_METRICS = True

ENABLE\_PROFILING = True

```



\### Production



```python

ENV = 'production'

LOG\_LEVEL = 'INFO'

TRADING\_MODE = 'live'

COLLECT\_METRICS = True

ENABLE\_PROFILING = False

DATABASE\_TYPE = 'postgresql'

```



\## Ã°Å¸â€ºÂ Ã¯Â¸Â Variables d'Environnement SystÃƒÂ¨me



\### Linux/macOS



```bash

\# Ajouter au ~/.bashrc ou ~/.zshrc

export THE\_BOT\_ENV=production

export THE\_BOT\_CONFIG=/path/to/config.py

```



\### Windows



```powershell

\# PowerShell

$env:THE\_BOT\_ENV = "production"

$env:THE\_BOT\_CONFIG = "C:\\path\\to\\config.py"

```



\## Ã°Å¸â€œâ€“ Bonnes Pratiques



\### SÃƒÂ©curitÃƒÂ©

1\. Ã¢Å“â€¦ \*\*Jamais\*\* de clÃƒÂ©s API dans config.py

2\. Ã¢Å“â€¦ Toujours dans `.env`

3\. Ã¢Å“â€¦ `.env` dans `.gitignore`

4\. Ã¢Å“â€¦ Permissions restrictives sur `.env` (600)

5\. Ã¢Å“â€¦ Backup rÃƒÂ©gulier de `.env` (sÃƒÂ©curisÃƒÂ©)



\### Performance

1\. Ã¢Å“â€¦ Ajuster `MAX\_THREADS` selon votre CPU

2\. Ã¢Å“â€¦ Limiter `SYMBOLS\_TO\_TRADE` pour ÃƒÂ©conomiser RAM

3\. Ã¢Å“â€¦ Activer le cache pour les indicateurs

4\. Ã¢Å“â€¦ Surveiller `MAX\_MEMORY\_MB`



\### StratÃƒÂ©gies

1\. Ã¢Å“â€¦ Commencer avec 1-2 stratÃƒÂ©gies seulement

2\. Ã¢Å“â€¦ Tester chaque stratÃƒÂ©gie sÃƒÂ©parÃƒÂ©ment

3\. Ã¢Å“â€¦ Augmenter progressivement les allocations

4\. Ã¢Å“â€¦ Surveiller les performances par stratÃƒÂ©gie



\### Risque

1\. Ã¢Å“â€¦ Commencer conservateur (1% par trade)

2\. Ã¢Å“â€¦ Augmenter progressivement si performances bonnes

3\. Ã¢Å“â€¦ Respecter TOUJOURS le max drawdown

4\. Ã¢Å“â€¦ Ne jamais dÃƒÂ©sactiver les circuit breakers



\## Ã°Å¸â€Â Debugging de Configuration



\### Afficher la Configuration Active



```python

\# Lancer le bot avec --show-config

python main.py --show-config

```



\### VÃƒÂ©rifier les Overrides



```python

\# Voir quelles valeurs ont ÃƒÂ©tÃƒÂ© surchargÃƒÂ©es

python main.py --check-overrides

```



\### Valider la Configuration



```python

\# Valider sans lancer le bot

python main.py --validate-config

```



\## Ã°Å¸â€™Â¡ Tips \& Astuces



\### Capital Progressif

```python

\# Augmenter le capital progressivement

\# Semaine 1: 1000 USDC

\# Semaine 2: Si profit, ajouter 500 USDC

\# Etc.

```



\### Profils de Risque

```python

\# CrÃƒÂ©er plusieurs fichiers config

\# config\_conservative.py

\# config\_moderate.py

\# config\_aggressive.py



\# Lancer avec

python main.py --config config\_conservative.py

```



\### Hot Reload

```python

\# Activer le rechargement auto de config

HOT\_RELOAD\_CONFIG = True

RELOAD\_INTERVAL = 60  # VÃƒÂ©rifier toutes les 60s

```



\## Ã°Å¸â€ Ëœ ProblÃƒÂ¨mes Courants



\### "Invalid configuration"

Ã¢â€ â€™ VÃƒÂ©rifier que toutes les allocations = 100%

Ã¢â€ â€™ VÃƒÂ©rifier que les pourcentages sont entre 0 et 1



\### "Cannot connect to Binance"

Ã¢â€ â€™ VÃƒÂ©rifier les clÃƒÂ©s API dans .env

Ã¢â€ â€™ VÃƒÂ©rifier la connexion internet

Ã¢â€ â€™ VÃƒÂ©rifier que Binance n'est pas en maintenance



\### "Insufficient capital"

Ã¢â€ â€™ Augmenter INITIAL\_CAPITAL

Ã¢â€ â€™ Ou rÃƒÂ©duire MIN\_ORDER\_SIZE



\## Ã°Å¸â€œÅ¡ Ressources



\- \[API Reference](API\_REFERENCE.md) - Documentation de l'API

\- \[Strategies Guide](STRATEGIES.md) - Guide des stratÃƒÂ©gies

\- \[Troubleshooting](TROUBLESHOOTING.md) - RÃƒÂ©solution de problÃƒÂ¨mes



\## Ã°Å¸Å¡â‚¬ Prochaines Ãƒâ€°tapes



1\. Ã¢Å“â€¦ Configuration terminÃƒÂ©e

2\. Ã°Å¸â€œâ€“ Lire le \[Guide des StratÃƒÂ©gies](STRATEGIES.md)

3\. Ã¢â€“Â¶Ã¯Â¸Â  Lancer en mode Paper Trading

4\. Ã°Å¸â€œÅ  Surveiller les performances 24-48h

5\. Ã°Å¸â€Â§ Ajuster la configuration si nÃƒÂ©cessaire

6\. Ã°Å¸â€™Â° Passer en Live (avec prudence)



---



\*\*Configuration optimale = Performances optimales ! Ã¢Å¡â„¢Ã¯Â¸Â\*\*



\*DerniÃƒÂ¨re mise ÃƒÂ  jour : Octobre 2024\*

