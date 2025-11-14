\# Ã¢Å¡Â¡ PERFORMANCE.md - Guide d'Optimisation



Guide complet pour optimiser les performances de \*\*The Bot\*\* sur votre systÃƒÂ¨me.



---



\## Ã°Å¸â€œâ€¹ Table des MatiÃƒÂ¨res



1\. \[Benchmarks](#-benchmarks)

2\. \[Configuration MatÃƒÂ©rielle](#-configuration-matÃƒÂ©rielle)

3\. \[Optimisations SystÃƒÂ¨me](#-optimisations-systÃƒÂ¨me)

4\. \[Optimisations Python](#-optimisations-python)

5\. \[Optimisations Bot](#-optimisations-bot)

6\. \[Monitoring Performance](#-monitoring-performance)

7\. \[Troubleshooting](#-troubleshooting)



---



\## Ã°Å¸â€œÅ  Benchmarks



\### Configuration de RÃƒÂ©fÃƒÂ©rence



\*\*Setup testÃƒÂ©:\*\*

```

CPU: Intel i7-9700K (8 cores @ 3.6GHz)

RAM: 16GB DDR4 3200MHz

Disque: SSD NVMe 500GB

OS: Windows 11 / Ubuntu 22.04

Python: 3.11

```



\### MÃƒÂ©triques Attendues



| MÃƒÂ©trique | Valeur | Note |

|----------|--------|------|

| \*\*Utilisation CPU\*\* | 15-30% | En moyenne |

| \*\*Utilisation RAM\*\* | 1-2 GB | Stable |

| \*\*Latence rÃƒÂ©seau\*\* | 50-200ms | Vers Binance |

| \*\*Scan cycle\*\* | 2-5s | Pour 100 symboles |

| \*\*Order execution\*\* | 100-300ms | Average |

| \*\*Startup time\*\* | 10-20s | Initialisation |



\### Performances par Configuration



\#### Configuration Minimale (8GB RAM, 4 cores)

```python

SYMBOLS\_TO\_SCAN = 50

SYMBOLS\_TO\_TRADE = 10

MAX\_THREADS = 2

MAX\_MEMORY\_MB = 1000

USE\_REDIS = False

```

\- CPU: 20-35%

\- RAM: 800MB-1.2GB

\- Trades/jour: 50-150



\#### Configuration RecommandÃƒÂ©e (16GB RAM, 8 cores)

```python

SYMBOLS\_TO\_SCAN = 100

SYMBOLS\_TO\_TRADE = 20

MAX\_THREADS = 4

MAX\_MEMORY\_MB = 2000

USE\_REDIS = True

```

\- CPU: 15-30%

\- RAM: 1-2GB

\- Trades/jour: 100-300



\#### Configuration Haute Performance (32GB RAM, 16 cores)

```python

SYMBOLS\_TO\_SCAN = 200

SYMBOLS\_TO\_TRADE = 50

MAX\_THREADS = 8

MAX\_MEMORY\_MB = 4000

USE\_REDIS = True

```

\- CPU: 10-25%

\- RAM: 2-4GB

\- Trades/jour: 200-500



---



\## Ã°Å¸â€™Â» Configuration MatÃƒÂ©rielle



\### CPU



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­Â (Critique)



\*\*Recommandations:\*\*

\- \*\*Minimum\*\*: 4 cores physiques

\- \*\*Optimal\*\*: 8 cores physiques

\- \*\*Clock speed\*\*: 3.0GHz+ prÃƒÂ©fÃƒÂ©rÃƒÂ©



\*\*Optimisations:\*\*

```bash

\# Linux: DÃƒÂ©sactiver CPU throttling

sudo cpupower frequency-set -g performance



\# Windows: Mode "Performances ÃƒÂ©levÃƒÂ©es"

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

```



\### RAM



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­Â (Important)



\*\*Recommandations:\*\*

\- \*\*Minimum\*\*: 8GB

\- \*\*RecommandÃƒÂ©\*\*: 16GB

\- \*\*Optimal\*\*: 32GB



\*\*Configuration systÃƒÂ¨me:\*\*

```bash

\# Linux: Configurer le swap

sudo sysctl vm.swappiness=10



\# VÃƒÂ©rifier l'utilisation

free -h

```



\### Disque



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­Â (ModÃƒÂ©rÃƒÂ©)



\*\*Recommandations:\*\*

\- Ã¢Å“â€¦ \*\*SSD NVMe\*\* (idÃƒÂ©al)

\- Ã¢Å“â€¦ \*\*SSD SATA\*\* (bon)

\- Ã¢ÂÅ’ \*\*HDD\*\* (lent, non recommandÃƒÂ©)



\*\*Impact sur performance:\*\*

\- SSD vs HDD: \*\*3-5x plus rapide\*\* pour I/O

\- Logs, cache, et donnÃƒÂ©es historiques



\### RÃƒÂ©seau



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­Â (Critique)



\*\*Recommandations:\*\*

\- \*\*Minimum\*\*: 10 Mbps stable

\- \*\*RecommandÃƒÂ©\*\*: 50+ Mbps

\- \*\*Latence\*\*: < 100ms vers Binance



\*\*Test de latence:\*\*

```bash

\# Ping Binance API

ping api.binance.com



\# Latence attendue:

\# France/Europe: 20-50ms

\# USA: 150-200ms

\# Asie: 100-150ms

```



---



\## Ã°Å¸â€Â§ Optimisations SystÃƒÂ¨me



\### Linux



\#### 1. Limites du systÃƒÂ¨me

```bash

\# Augmenter les file descriptors

sudo nano /etc/security/limits.conf



\# Ajouter:

\* soft nofile 65535

\* hard nofile 65535

```



\#### 2. Network tuning

```bash

\# Optimiser TCP

sudo nano /etc/sysctl.conf



\# Ajouter:

net.core.rmem\_max=16777216

net.core.wmem\_max=16777216

net.ipv4.tcp\_rmem=4096 87380 16777216

net.ipv4.tcp\_wmem=4096 65536 16777216

net.ipv4.tcp\_congestion\_control=bbr



\# Appliquer

sudo sysctl -p

```



\#### 3. DÃƒÂ©sactiver services inutiles

```bash

\# Lister les services

systemctl list-unit-files --type=service --state=enabled



\# DÃƒÂ©sactiver (exemples)

sudo systemctl disable bluetooth

sudo systemctl disable cups

```



\### Windows



\#### 1. Mode Haute Performance

```cmd

\# Activer mode haute performance

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c



\# DÃƒÂ©sactiver veille disque

powercfg /change disk-timeout-ac 0

```



\#### 2. DÃƒÂ©sactiver services inutiles

```

Services.msc:

\- Windows Search (Indexation)

\- Superfetch

\- Windows Defender (si antivirus tiers)

```



\#### 3. PrioritÃƒÂ© processus

```cmd

\# Lancer avec haute prioritÃƒÂ©

start /HIGH python main.py --mode paper

```



\### macOS



\#### 1. DÃƒÂ©sactiver animations

```bash

\# RÃƒÂ©duire animations

defaults write com.apple.dock autohide-delay -float 0

defaults write com.apple.dock autohide-time-modifier -float 0

killall Dock

```



\#### 2. Optimiser network

```bash

\# Augmenter buffer sizes

sudo sysctl -w net.inet.tcp.sendspace=1048576

sudo sysctl -w net.inet.tcp.recvspace=1048576

```



---



\## Ã°Å¸ÂÂ Optimisations Python



\### 1. Utiliser PyPy (optionnel)



\*\*PyPy\*\* = JIT compiler pour Python



```bash

\# Installation

sudo apt install pypy3  # Linux

brew install pypy3      # macOS



\# CrÃƒÂ©er venv avec PyPy

pypy3 -m venv venv-pypy

source venv-pypy/bin/activate

pip install -r requirements.txt



\# Lancer le bot

pypy3 main.py --mode paper

```



\*\*Gain attendu\*\*: 20-40% plus rapide



\### 2. Compiler avec Cython



\*\*modules critiques:\*\*

```python

\# CrÃƒÂ©er setup\_cython.py

from setuptools import setup

from Cython.Build import cythonize



setup(

&nbsp;   ext\_modules=cythonize(\[

&nbsp;       "utils/indicators.py",

&nbsp;       "strategies/scalping.py",

&nbsp;       "risk/position\_sizing.py"

&nbsp;   ])

)



\# Compiler

python setup\_cython.py build\_ext --inplace

```



\*\*Gain attendu\*\*: 30-60% sur modules compilÃƒÂ©s



\### 3. Numba JIT



DÃƒÂ©jÃƒÂ  intÃƒÂ©grÃƒÂ© dans le code pour fonctions critiques:



```python

from numba import jit



@jit(nopython=True)

def calculate\_indicators\_fast(prices, volumes):

&nbsp;   # Code optimisÃƒÂ© automatiquement

&nbsp;   pass

```



\### 4. Garbage Collection



```python

\# Dans config.py

import gc



\# DÃƒÂ©sactiver GC automatique

gc.disable()



\# Forcer manuellement toutes les 5 min

\# (dÃƒÂ©jÃƒÂ  implÃƒÂ©mentÃƒÂ© dans memory\_manager.py)

```



---



\## Ã°Å¸Â¤â€“ Optimisations Bot



\### 1. RÃƒÂ©duire le Scan



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­Â



```python

\# config.py

SYMBOLS\_TO\_SCAN = 50      # Au lieu de 100

SYMBOLS\_TO\_TRADE = 10     # Au lieu de 20

SCAN\_INTERVAL = 600       # 10 min au lieu de 5

```



\*\*Gain\*\*: -30% CPU, -20% RAM



\### 2. DÃƒÂ©sactiver StratÃƒÂ©gies



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­Â



```python

\# DÃƒÂ©sactiver ML (plus lourd)

{

&nbsp;   'name': 'ml',

&nbsp;   'enabled': False,  # Ã¢â€ Â Mettre False

&nbsp;   'allocation': 0.00,

}

```



\*\*Gain\*\*: -15% CPU, -10% RAM



\### 3. Optimiser Redis



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­ÂÃ¢Â­ÂÃ¢Â­Â



```bash

\# redis.conf

maxmemory 256mb

maxmemory-policy allkeys-lru

save ""  # DÃƒÂ©sactiver persistence



\# RedÃƒÂ©marrer

sudo systemctl restart redis

```



```python

\# config.py

CACHE\_TTL = 120  # 2 min au lieu de 1

```



\*\*Gain\*\*: +40% vitesse indicateurs



\### 4. RÃƒÂ©duire les Logs



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­Â



```python

\# config.py

LOG\_LEVEL = 'WARNING'  # Au lieu de 'INFO'

```



\*\*Gain\*\*: -10% I/O disque



\### 5. Buffer Size



\*\*Impact\*\*: Ã¢Â­ÂÃ¢Â­Â



```python

\# config.py

TICK\_BUFFER\_SIZE = 2000  # Au lieu de 5000

```



\*\*Gain\*\*: -15% RAM



---



\## Ã°Å¸â€œË† Monitoring Performance



\### 1. Script de Monitoring



CrÃƒÂ©er `monitor\_performance.py`:



```python

\#!/usr/bin/env python3

import psutil

import time



def monitor():

&nbsp;   process = psutil.Process()

&nbsp;   

&nbsp;   while True:

&nbsp;       cpu = process.cpu\_percent(interval=1)

&nbsp;       mem = process.memory\_info().rss / 1024 / 1024  # MB

&nbsp;       threads = process.num\_threads()

&nbsp;       

&nbsp;       print(f"CPU: {cpu:5.1f}% | RAM: {mem:6.1f}MB | Threads: {threads}")

&nbsp;       time.sleep(5)



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   monitor()

```



\### 2. Logs de Performance



```bash

\# Analyser les logs

grep "performance" data/logs/thebot.log



\# Temps d'exÃƒÂ©cution par fonction

grep "took" data/logs/thebot.log | sort -k5 -n

```



\### 3. Profiling (si activÃƒÂ©)



```python

\# Dans config.py

ENABLE\_PROFILING = True

```



```bash

\# Analyser les rÃƒÂ©sultats

python -m pstats data/profiling\_results.prof

```



---



\## Ã°Å¸â€Â Troubleshooting



\### ProblÃƒÂ¨me: CPU ÃƒÂ  100%



\*\*Causes possibles:\*\*

1\. Trop de symboles scannÃƒÂ©s

2\. ML trop lourd

3\. Boucle infinie quelque part



\*\*Solutions:\*\*

```python

\# 1. RÃƒÂ©duire scan

SYMBOLS\_TO\_SCAN = 30

SYMBOLS\_TO\_TRADE = 5



\# 2. DÃƒÂ©sactiver ML

\# Dans ACTIVE\_STRATEGIES, mettre ml: enabled=False



\# 3. VÃƒÂ©rifier les logs

tail -f data/logs/thebot.log

```



\### ProblÃƒÂ¨me: RAM qui augmente



\*\*Causes:\*\*

1\. Memory leak

2\. Buffer trop grand

3\. Pas de garbage collection



\*\*Solutions:\*\*

```python

\# config.py

MAX\_MEMORY\_MB = 1500  # Limite plus stricte

TICK\_BUFFER\_SIZE = 1000  # Buffer plus petit



\# Forcer cleanup

import gc

gc.collect()

```



\### ProblÃƒÂ¨me: Bot lent



\*\*Diagnostic:\*\*

```python

\# Activer le profiling

ENABLE\_PROFILING = True

LOG\_LEVEL = 'DEBUG'

```



\*\*Solutions selon la cause:\*\*

\- \*\*RÃƒÂ©seau lent\*\*: Changer de FAI, utiliser VPN

\- \*\*Disque lent\*\*: Migrer vers SSD

\- \*\*Redis lent\*\*: Optimiser config Redis

\- \*\*Indicateurs lents\*\*: Activer Numba JIT



\### ProblÃƒÂ¨me: Latence ÃƒÂ©levÃƒÂ©e



\*\*Test:\*\*

```bash

\# Tester latence Binance

for i in {1..10}; do

&nbsp;   curl -w "%{time\_total}\\n" -o /dev/null -s https://api.binance.com/api/v3/ping

done

```



\*\*Solutions:\*\*

\- Utiliser VPN proche de Binance (Singapour, Tokyo)

\- Changer de FAI

\- VÃƒÂ©rifier firewall/antivirus



---



\## Ã°Å¸â€œÅ  Tableau RÃƒÂ©capitulatif



| Optimisation | Impact CPU | Impact RAM | DifficultÃƒÂ© | RecommandÃƒÂ© |

|--------------|------------|------------|------------|------------|

| Redis | -10% | +50MB | Facile | Ã¢Å“â€¦ Oui |

| RÃƒÂ©duire scan | -30% | -20% | Facile | Ã¢Å“â€¦ Oui |

| DÃƒÂ©sactiver ML | -15% | -10% | Facile | Ã¢Å¡Â Ã¯Â¸Â  Si CPU limitÃƒÂ© |

| SSD | -5% | 0% | Moyen | Ã¢Å“â€¦ Oui |

| Numba JIT | -20% | +10MB | Facile | Ã¢Å“â€¦ Oui |

| PyPy | -30% | +50MB | Moyen | Ã¢Å¡Â Ã¯Â¸Â  ExpÃƒÂ©rimental |

| Cython | -40% | 0% | Difficile | Ã¢Å¡Â Ã¯Â¸Â  AvancÃƒÂ© |

| RÃƒÂ©duire logs | -5% | 0% | Facile | Ã¢Å¡Â Ã¯Â¸Â  Si stockage limitÃƒÂ© |



---



\## Ã°Å¸Å½Â¯ Configuration Optimale par Cas d'Usage



\### Trading Intensif (Max Trades)

```python

SYMBOLS\_TO\_SCAN = 200

SYMBOLS\_TO\_TRADE = 50

MAX\_THREADS = 8

USE\_REDIS = True

ENABLE\_PROFILING = False

LOG\_LEVEL = 'WARNING'

```



\### PC LimitÃƒÂ© (8GB RAM, 4 cores)

```python

SYMBOLS\_TO\_SCAN = 30

SYMBOLS\_TO\_TRADE = 5

MAX\_THREADS = 2

USE\_REDIS = False

TICK\_BUFFER\_SIZE = 1000

MAX\_MEMORY\_MB = 1000

```



\### Ãƒâ€°quilibrÃƒÂ© (Recommended)

```python

SYMBOLS\_TO\_SCAN = 100

SYMBOLS\_TO\_TRADE = 20

MAX\_THREADS = 4

USE\_REDIS = True

TICK\_BUFFER\_SIZE = 5000

MAX\_MEMORY\_MB = 2000

```



---



\## Ã°Å¸â€œÅ¾ Support Performance



Si aprÃƒÂ¨s toutes ces optimisations vous rencontrez encore des problÃƒÂ¨mes:



1\. \*\*Collectez les mÃƒÂ©triques\*\*:

&nbsp;  ```bash

&nbsp;  python monitor\_performance.py > perf.log

&nbsp;  ```



2\. \*\*Partagez sur Discord/Support\*\* avec:

&nbsp;  - Configuration matÃƒÂ©rielle

&nbsp;  - Config du bot (config.py)

&nbsp;  - Logs de performance

&nbsp;  - RÃƒÂ©sultat de `test\_connection.py`



---



\*Guide mis ÃƒÂ  jour le: 2025-01-15\*  

\*Version: 1.0.0\*



\*\*Performance is key! Ã°Å¸Å¡â‚¬Ã¢Å¡Â¡\*\*

