\# Ã°Å¸â€œÂ¦ GUIDE D'INSTALLATION - THE BOT



Guide complet pour installer et configurer \*\*The Bot\*\* sur votre systÃƒÂ¨me.



---



\## Ã°Å¸â€œâ€¹ Table des MatiÃƒÂ¨res



1\. \[PrÃƒÂ©requis](#prÃƒÂ©requis)

2\. \[Installation Python et DÃƒÂ©pendances](#installation-python-et-dÃƒÂ©pendances)

3\. \[Installation TA-Lib](#installation-ta-lib)

4\. \[Installation Redis (Optionnel)](#installation-redis)

5\. \[Configuration Binance](#configuration-binance)

6\. \[Installation The Bot](#installation-the-bot)

7\. \[Configuration](#configuration)

8\. \[Test de Connexion](#test-de-connexion)

9\. \[Premier Lancement](#premier-lancement)

10\. \[DÃƒÂ©pannage](#dÃƒÂ©pannage)



---



\## Ã°Å¸â€Â§ PrÃƒÂ©requis



\### Configuration MatÃƒÂ©rielle Minimale



| Composant | Minimum | RecommandÃƒÂ© |

|-----------|---------|------------|

| \*\*CPU\*\* | 4 cores | 8 cores |

| \*\*RAM\*\* | 8 GB | 16 GB |

| \*\*Stockage\*\* | 10 GB libre | 50 GB SSD |

| \*\*Internet\*\* | 10 Mbps | 50+ Mbps (stable) |



\### SystÃƒÂ¨mes d'Exploitation SupportÃƒÂ©s



\- Ã¢Å“â€¦ \*\*Windows 10/11\*\* (64-bit)

\- Ã¢Å“â€¦ \*\*macOS\*\* 11+ (Big Sur ou plus rÃƒÂ©cent)

\- Ã¢Å“â€¦ \*\*Linux\*\* (Ubuntu 20.04+, Debian 10+, CentOS 8+)



\### Logiciels Requis



\- \*\*Python 3.9\*\* ou supÃƒÂ©rieur (3.9, 3.10, 3.11 testÃƒÂ©s)

\- \*\*Git\*\* (pour cloner le repository)

\- \*\*pip\*\* (gestionnaire de packages Python)

\- \*\*Compte Binance\*\* avec API activÃƒÂ©e



---



\## Ã°Å¸ÂÂ Installation Python et DÃƒÂ©pendances



\### Windows



\#### Option 1: Installation depuis python.org (RecommandÃƒÂ©)



1\. \*\*TÃƒÂ©lÃƒÂ©charger Python\*\*

&nbsp;  ```

&nbsp;  https://www.python.org/downloads/

&nbsp;  ```

&nbsp;  - Choisir Python 3.9+ (version 3.11 recommandÃƒÂ©e)



2\. \*\*Installer Python\*\*

&nbsp;  - Ã¢Å¡Â Ã¯Â¸Â \*\*IMPORTANT\*\*: Cocher "Add Python to PATH"

&nbsp;  - Cliquer sur "Install Now"



3\. \*\*VÃƒÂ©rifier l'installation\*\*

&nbsp;  ```cmd

&nbsp;  python --version

&nbsp;  pip --version

&nbsp;  ```



\#### Option 2: Via Microsoft Store



```cmd

\# Rechercher "Python 3.11" dans Microsoft Store

\# Installer directement

```



\### macOS



\#### Via Homebrew (RecommandÃƒÂ©)



```bash

\# Installer Homebrew si pas dÃƒÂ©jÃƒÂ  fait

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"



\# Installer Python

brew install python@3.11



\# VÃƒÂ©rifier

python3 --version

pip3 --version

```



\### Linux (Ubuntu/Debian)



```bash

\# Mettre ÃƒÂ  jour les packages

sudo apt update \&\& sudo apt upgrade -y



\# Installer Python et pip

sudo apt install python3.11 python3.11-venv python3-pip -y



\# Installer build tools (nÃƒÂ©cessaire pour TA-Lib)

sudo apt install build-essential -y



\# VÃƒÂ©rifier

python3.11 --version

pip3 --version

```



\### Linux (CentOS/RHEL)



```bash

\# Activer EPEL

sudo yum install epel-release -y



\# Installer Python

sudo yum install python39 python39-pip python39-devel -y



\# Installer build tools

sudo yum groupinstall "Development Tools" -y



\# VÃƒÂ©rifier

python3.9 --version

pip3 --version

```



---



\## Ã°Å¸â€œÅ  Installation TA-Lib



TA-Lib est \*\*ESSENTIEL\*\* pour les indicateurs techniques.



\### Windows



\#### MÃƒÂ©thode Simple (Wheel prÃƒÂ©compilÃƒÂ©)



1\. \*\*TÃƒÂ©lÃƒÂ©charger le wheel correspondant\*\*

&nbsp;  ```

&nbsp;  https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

&nbsp;  ```

&nbsp;  - Choisir selon votre Python:

&nbsp;    - Python 3.9: `TA\_LibÃ¢â‚¬â€˜0.4.24Ã¢â‚¬â€˜cp39Ã¢â‚¬â€˜cp39Ã¢â‚¬â€˜win\_amd64.whl`

&nbsp;    - Python 3.10: `TA\_LibÃ¢â‚¬â€˜0.4.24Ã¢â‚¬â€˜cp310Ã¢â‚¬â€˜cp310Ã¢â‚¬â€˜win\_amd64.whl`

&nbsp;    - Python 3.11: `TA\_LibÃ¢â‚¬â€˜0.4.24Ã¢â‚¬â€˜cp311Ã¢â‚¬â€˜cp311Ã¢â‚¬â€˜win\_amd64.whl`



2\. \*\*Installer le wheel\*\*

&nbsp;  ```cmd

&nbsp;  cd Downloads

&nbsp;  pip install TA\_LibÃ¢â‚¬â€˜0.4.24Ã¢â‚¬â€˜cp311Ã¢â‚¬â€˜cp311Ã¢â‚¬â€˜win\_amd64.whl

&nbsp;  ```



\#### MÃƒÂ©thode Compilation (AvancÃƒÂ©)



```cmd

\# Installer Visual Studio Build Tools

\# https://visualstudio.microsoft.com/downloads/



\# TÃƒÂ©lÃƒÂ©charger TA-Lib C

\# https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/



\# Extraire dans C:\\ta-lib



\# Installer

pip install TA-Lib

```



\### macOS



```bash

\# Installer TA-Lib C library

brew install ta-lib



\# Installer le wrapper Python

pip3 install TA-Lib



\# VÃƒÂ©rifier

python3 -c "import talib; print(talib.\_\_version\_\_)"

```



\### Linux (Ubuntu/Debian)



```bash

\# TÃƒÂ©lÃƒÂ©charger et compiler TA-Lib C

cd /tmp

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -xzf ta-lib-0.4.0-src.tar.gz

cd ta-lib/



\# Compiler et installer

./configure --prefix=/usr

make

sudo make install



\# Installer wrapper Python

pip3 install TA-Lib



\# VÃƒÂ©rifier

python3 -c "import talib; print(talib.\_\_version\_\_)"

```



\### Linux (CentOS/RHEL)



```bash

\# MÃƒÂªme procÃƒÂ©dure qu'Ubuntu

cd /tmp

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -xzf ta-lib-0.4.0-src.tar.gz

cd ta-lib/

./configure --prefix=/usr

make

sudo make install

sudo ldconfig



pip3 install TA-Lib

```



---



\## Ã°Å¸â€Â´ Installation Redis (Optionnel mais RecommandÃƒÂ©)



Redis est utilisÃƒÂ© pour le cache local et amÃƒÂ©liore les performances.



\### Windows



```cmd

\# Option 1: Via Memurai (Redis pour Windows)

\# https://www.memurai.com/get-memurai



\# Option 2: Via WSL2 + Ubuntu

wsl --install

\# Puis suivre instructions Linux

```



\### macOS



```bash

\# Via Homebrew

brew install redis



\# DÃƒÂ©marrer Redis

brew services start redis



\# VÃƒÂ©rifier

redis-cli ping

\# Devrait rÃƒÂ©pondre: PONG

```



\### Linux



```bash

\# Ubuntu/Debian

sudo apt install redis-server -y



\# CentOS/RHEL

sudo yum install redis -y



\# DÃƒÂ©marrer Redis

sudo systemctl start redis

sudo systemctl enable redis



\# VÃƒÂ©rifier

redis-cli ping

```



---



\## Ã°Å¸â€Â Configuration Binance



\### 1. CrÃƒÂ©er un Compte Binance



Si vous n'avez pas de compte:

```

https://www.binance.com/fr/register

```



Ã¢Å¡Â Ã¯Â¸Â \*\*Important\*\*:

\- Activer l'authentification 2FA (obligatoire)

\- ComplÃƒÂ©ter la vÃƒÂ©rification KYC

\- DÃƒÂ©poser du capital (minimum 1000 USDC recommandÃƒÂ©)



\### 2. CrÃƒÂ©er une API Key



1\. \*\*Se connecter ÃƒÂ  Binance\*\*



2\. \*\*Aller dans API Management\*\*

&nbsp;  ```

&nbsp;  Profil > API Management

&nbsp;  ```



3\. \*\*CrÃƒÂ©er une nouvelle API Key\*\*

&nbsp;  - Label: "The Bot Trading"

&nbsp;  - Cliquer sur "Create API"



4\. \*\*Configurer les Permissions\*\*

&nbsp;  - Ã¢Å“â€¦ Enable Reading

&nbsp;  - Ã¢Å“â€¦ Enable Spot \& Margin Trading

&nbsp;  - Ã¢ÂÅ’ Enable Withdrawals (DÃƒâ€°SACTIVER!)

&nbsp;  - Ã¢ÂÅ’ Enable Internal Transfer (DÃƒâ€°SACTIVER!)



5\. \*\*Restriction IP (Fortement RecommandÃƒÂ©)\*\*

&nbsp;  - Obtenir votre IP: `https://whatismyipaddress.com/`

&nbsp;  - Ajouter votre IP dans "Restrict access to trusted IPs only"



6\. \*\*Sauvegarder les ClÃƒÂ©s\*\*

&nbsp;  ```

&nbsp;  API Key: XXXXXXXXXXXXXXXXXXXXXXX

&nbsp;  Secret Key: YYYYYYYYYYYYYYYYYYYY

&nbsp;  ```

&nbsp;  Ã¢Å¡Â Ã¯Â¸Â \*\*NE JAMAIS PARTAGER CES CLÃƒâ€°S!\*\*



\### 3. Testnet (Pour DÃƒÂ©buter)



Le Testnet permet de tester sans risque:



```

https://testnet.binance.vision/

```



1\. Se connecter avec GitHub

2\. CrÃƒÂ©er une API Key testnet

3\. Obtenir des fonds de test (faucet)



---



\## Ã°Å¸â€œÂ¥ Installation The Bot



\### 1. Cloner le Repository



```bash

\# Via HTTPS

git clone https://github.com/your-username/the-bot.git



\# Ou via SSH (si configurÃƒÂ©)

git clone git@github.com:your-username/the-bot.git



\# Aller dans le dossier

cd the-bot

```



\### 2. CrÃƒÂ©er l'Environnement Virtuel



\*\*Windows:\*\*

```cmd

python -m venv venv

venv\\Scripts\\activate

```



\*\*macOS/Linux:\*\*

```bash

python3 -m venv venv

source venv/bin/activate

```



Votre terminal devrait maintenant afficher `(venv)` au dÃƒÂ©but.



\### 3. Installer les DÃƒÂ©pendances



```bash

\# Mettre ÃƒÂ  jour pip

pip install --upgrade pip



\# Installer toutes les dÃƒÂ©pendances

pip install -r requirements.txt



\# VÃƒÂ©rifier l'installation

pip list

```



\*\*DÃƒÂ©pendances installÃƒÂ©es:\*\*

```

pandas==1.4.0

numpy==1.22.0

scikit-learn==1.0.2

xgboost==1.5.1

lightgbm==3.3.2

python-binance==1.0.16

websocket-client==1.3.1

TA-Lib==0.4.24

ccxt==2.5.0

redis==4.1.0

psutil==5.9.0

```



---



\## Ã¢Å¡â„¢Ã¯Â¸Â Configuration



\### 1. Fichier .env



```bash

\# Copier le template

cp .env.example .env



\# Ãƒâ€°diter avec votre ÃƒÂ©diteur

\# Windows: notepad .env

\# macOS/Linux: nano .env

```



\*\*Contenu du .env:\*\*

```bash

\# BINANCE API

BINANCE\_API\_KEY=your\_api\_key\_here

BINANCE\_SECRET\_KEY=your\_secret\_key\_here



\# MODE

TESTNET=True  # True pour testnet, False pour production



\# CAPITAL

INITIAL\_CAPITAL=1000  # En USDC



\# RISK MANAGEMENT

RISK\_PER\_TRADE=0.02  # 2% par trade

MAX\_DAILY\_LOSS=0.05  # 5% perte max/jour

MAX\_DRAWDOWN=0.08    # 8% drawdown max



\# PERFORMANCE

MAX\_THREADS=4

MAX\_MEMORY\_MB=2000



\# LOGGING

LOG\_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

```



\### 2. Fichier config.py



```bash

\# Copier le template

cp config.example.py config.py



\# Le fichier config.py charge les variables depuis .env

\# Pas besoin de l'ÃƒÂ©diter sauf pour personnalisation avancÃƒÂ©e

```



\### 3. VÃƒÂ©rifier la Structure



```bash

\# Votre structure devrait ressembler ÃƒÂ :

The Bot/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ .env                    # Ã¢Å“â€¦ Vos clÃƒÂ©s API

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ .env.example           # Template

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config.py              # Ã¢Å“â€¦ Configuration

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config.example.py      # Template

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ main.py                # Point d'entrÃƒÂ©e

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt       # DÃƒÂ©pendances

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ venv/                  # Ã¢Å“â€¦ Environnement virtuel

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ strategies/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ risk/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ ml/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ exchange/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ scanner/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ threads/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ monitoring/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ utils/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ data/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tests/

Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ docs/

```



---



\## Ã°Å¸Â§Âª Test de Connexion



\### 1. Script de Test



CrÃƒÂ©er `test\_connection.py`:



```python

\#!/usr/bin/env python3

"""Test de connexion ÃƒÂ  Binance"""



import os

from dotenv import load\_dotenv

from binance.client import Client



\# Charger les variables d'environnement

load\_dotenv()



API\_KEY = os.getenv('BINANCE\_API\_KEY')

SECRET\_KEY = os.getenv('BINANCE\_SECRET\_KEY')

TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'



print("Ã°Å¸Â§Âª Test de connexion ÃƒÂ  Binance...")

print(f"Mode: {'TESTNET' if TESTNET else 'PRODUCTION'}")

print("-" \* 50)



try:

&nbsp;   # CrÃƒÂ©er le client

&nbsp;   if TESTNET:

&nbsp;       client = Client(API\_KEY, SECRET\_KEY, testnet=True)

&nbsp;   else:

&nbsp;       client = Client(API\_KEY, SECRET\_KEY)

&nbsp;   

&nbsp;   # Test 1: Ping

&nbsp;   print("1Ã¯Â¸ÂÃ¢Æ’Â£ Test ping...", end=" ")

&nbsp;   client.ping()

&nbsp;   print("Ã¢Å“â€¦ OK")

&nbsp;   

&nbsp;   # Test 2: Server time

&nbsp;   print("2Ã¯Â¸ÂÃ¢Æ’Â£ Test server time...", end=" ")

&nbsp;   time = client.get\_server\_time()

&nbsp;   print(f"Ã¢Å“â€¦ OK ({time\['serverTime']})")

&nbsp;   

&nbsp;   # Test 3: Account info

&nbsp;   print("3Ã¯Â¸ÂÃ¢Æ’Â£ Test account info...", end=" ")

&nbsp;   account = client.get\_account()

&nbsp;   print("Ã¢Å“â€¦ OK")

&nbsp;   

&nbsp;   # Test 4: Balance

&nbsp;   print("4Ã¯Â¸ÂÃ¢Æ’Â£ Test balance...", end=" ")

&nbsp;   balances = {b\['asset']: float(b\['free']) 

&nbsp;               for b in account\['balances'] 

&nbsp;               if float(b\['free']) > 0}

&nbsp;   print(f"Ã¢Å“â€¦ OK")

&nbsp;   print(f"   Balances: {balances}")

&nbsp;   

&nbsp;   # Test 5: Ticker price

&nbsp;   print("5Ã¯Â¸ÂÃ¢Æ’Â£ Test ticker price...", end=" ")

&nbsp;   ticker = client.get\_symbol\_ticker(symbol="BTCUSDC")

&nbsp;   print(f"Ã¢Å“â€¦ OK (BTC = ${float(ticker\['price']):,.2f})")

&nbsp;   

&nbsp;   print("\\n" + "=" \* 50)

&nbsp;   print("Ã¢Å“â€¦ TOUS LES TESTS RÃƒâ€°USSIS!")

&nbsp;   print("=" \* 50)

&nbsp;   print("\\nÃ°Å¸Å¡â‚¬ Vous pouvez lancer The Bot!")

&nbsp;   

except Exception as e:

&nbsp;   print(f"\\nÃ¢ÂÅ’ ERREUR: {e}")

&nbsp;   print("\\nÃ°Å¸â€Â§ VÃƒÂ©rifiez:")

&nbsp;   print("  1. Vos clÃƒÂ©s API dans .env")

&nbsp;   print("  2. Les permissions de l'API")

&nbsp;   print("  3. Votre connexion internet")

&nbsp;   print("  4. Le mode TESTNET/PRODUCTION")

```



\### 2. Lancer le Test



```bash

python test\_connection.py

```



\*\*RÃƒÂ©sultat attendu:\*\*

```

Ã°Å¸Â§Âª Test de connexion ÃƒÂ  Binance...

Mode: TESTNET

--------------------------------------------------

1Ã¯Â¸ÂÃ¢Æ’Â£ Test ping... Ã¢Å“â€¦ OK

2Ã¯Â¸ÂÃ¢Æ’Â£ Test server time... Ã¢Å“â€¦ OK (1705334400000)

3Ã¯Â¸ÂÃ¢Æ’Â£ Test account info... Ã¢Å“â€¦ OK

4Ã¯Â¸ÂÃ¢Æ’Â£ Test balance... Ã¢Å“â€¦ OK

&nbsp;  Balances: {'USDC': 1000.0, 'BTC': 0.05}

5Ã¯Â¸ÂÃ¢Æ’Â£ Test ticker price... Ã¢Å“â€¦ OK (BTC = $45,234.50)



==================================================

Ã¢Å“â€¦ TOUS LES TESTS RÃƒâ€°USSIS!

==================================================



Ã°Å¸Å¡â‚¬ Vous pouvez lancer The Bot!

```



---



\## Ã°Å¸Å¡â‚¬ Premier Lancement



\### Mode Paper Trading (RecommandÃƒÂ© pour dÃƒÂ©buter)



```bash

\# Activer l'environnement virtuel

\# Windows: venv\\Scripts\\activate

\# macOS/Linux: source venv/bin/activate



\# Lancer en mode paper

python main.py --mode paper

```



\*\*Vous devriez voir:\*\*

```

Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”

Ã¢â€¢â€˜                                                  Ã¢â€¢â€˜

Ã¢â€¢â€˜                  Ã°Å¸Â¤â€“ THE BOT Ã°Å¸Â¤â€“                   Ã¢â€¢â€˜

Ã¢â€¢â€˜                                                  Ã¢â€¢â€˜

Ã¢â€¢â€˜      Bot de Trading Algorithmique AvancÃƒÂ©         Ã¢â€¢â€˜

Ã¢â€¢â€˜                Version 1.0.0                     Ã¢â€¢â€˜

Ã¢â€¢â€˜                                                  Ã¢â€¢â€˜

Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â



Ã¢Å“â€¦ DÃƒÂ©pendances vÃƒÂ©rifiÃƒÂ©es



Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”

Ã¢â€¢â€˜           Ã°Å¸Â¤â€“ THE BOT - INITIALISATION Ã°Å¸Â¤â€“         Ã¢â€¢â€˜

Ã¢â€¢Â Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â£

Ã¢â€¢â€˜  Mode:         PAPER                             Ã¢â€¢â€˜

Ã¢â€¢â€˜  Capital:      $1,000.00                         Ã¢â€¢â€˜

Ã¢â€¢â€˜  Risk/Trade:   2.0%                              Ã¢â€¢â€˜

Ã¢â€¢â€˜  Max Drawdown: 8.0%                              Ã¢â€¢â€˜

Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â



Initialisation des composants...

1/7 - Connexion ÃƒÂ  Binance...

Ã¢Å“â€¦ Connexion Binance ÃƒÂ©tablie

2/7 - Initialisation Order Manager...

Ã¢Å“â€¦ Order Manager prÃƒÂªt

...

```



\### Mode Live Trading (Production)



Ã¢Å¡Â Ã¯Â¸Â \*\*ATTENTION: Argent rÃƒÂ©el!\*\*



```bash

python main.py --mode live

```



Vous devrez confirmer:

```

Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â



Capital configurÃƒÂ©: $1,000.00

Exchange: Binance (PRODUCTION)



ÃƒÅ tes-vous ABSOLUMENT SÃƒâ€ºR? Tapez 'OUI JE COMPRENDS' pour confirmer:

```



---



\## Ã°Å¸â€Â DÃƒÂ©pannage



\### ProblÃƒÂ¨mes Courants



\#### 1. ImportError: No module named 'talib'



\*\*Cause\*\*: TA-Lib pas installÃƒÂ© correctement



\*\*Solution\*\*:

```bash

\# Windows: TÃƒÂ©lÃƒÂ©charger le wheel et installer

pip install TA\_Lib-0.4.24-cp311-cp311-win\_amd64.whl



\# macOS:

brew install ta-lib

pip install TA-Lib



\# Linux:

\# Recompiler TA-Lib (voir section Installation TA-Lib)

```



\#### 2. BinanceAPIException: Invalid API-key



\*\*Cause\*\*: ClÃƒÂ©s API incorrectes ou permissions insuffisantes



\*\*Solution\*\*:

1\. VÃƒÂ©rifier les clÃƒÂ©s dans `.env`

2\. VÃƒÂ©rifier les permissions sur Binance:

&nbsp;  - Ã¢Å“â€¦ Enable Reading

&nbsp;  - Ã¢Å“â€¦ Enable Spot \& Margin Trading

3\. VÃƒÂ©rifier la restriction IP si activÃƒÂ©e

4\. RÃƒÂ©gÃƒÂ©nÃƒÂ©rer les clÃƒÂ©s si nÃƒÂ©cessaire



\#### 3. ConnectionError: Max retries exceeded



\*\*Cause\*\*: ProblÃƒÂ¨me de connexion internet ou Binance down



\*\*Solution\*\*:

```bash

\# Tester la connexion

ping api.binance.com



\# VÃƒÂ©rifier le status Binance

\# https://www.binance.com/en/support/announcement



\# VÃƒÂ©rifier votre pare-feu/antivirus

```



\#### 4. MemoryError ou SystÃƒÂ¨me Lent



\*\*Cause\*\*: RAM insuffisante ou trop de processus



\*\*Solution\*\*:

```python

\# Dans config.py, rÃƒÂ©duire:

MAX\_MEMORY\_MB = 1000  # Au lieu de 2000

SYMBOLS\_TO\_SCAN = 50   # Au lieu de 100

SYMBOLS\_TO\_TRADE = 10  # Au lieu de 20

```



\#### 5. ModuleNotFoundError: No module named 'config'



\*\*Cause\*\*: Fichier config.py manquant



\*\*Solution\*\*:

```bash

cp config.example.py config.py

\# Puis ÃƒÂ©diter config.py avec vos paramÃƒÂ¨tres

```



\#### 6. Redis Connection Error



\*\*Cause\*\*: Redis non dÃƒÂ©marrÃƒÂ© (optionnel mais recommandÃƒÂ©)



\*\*Solution\*\*:

```bash

\# macOS/Linux:

sudo systemctl start redis

\# ou

brew services start redis



\# Windows:

\# DÃƒÂ©marrer Memurai depuis le menu dÃƒÂ©marrer



\# Ou dÃƒÂ©sactiver Redis dans config.py:

USE\_REDIS = False

```



\#### 7. Permission Denied sur Linux/macOS



\*\*Cause\*\*: Droits insuffisants sur les fichiers



\*\*Solution\*\*:

```bash

\# Donner les droits d'exÃƒÂ©cution

chmod +x main.py

chmod +x test\_connection.py



\# Ou lancer avec python explicitement

python main.py --mode paper

```



\#### 8. SSL Certificate Error



\*\*Cause\*\*: ProblÃƒÂ¨me de certificats SSL



\*\*Solution\*\*:

```bash

\# Mettre ÃƒÂ  jour certifi

pip install --upgrade certifi



\# macOS: Installer les certificats Python

/Applications/Python\\ 3.11/Install\\ Certificates.command

```



\### Logs et Debug



\#### Activer le Mode Debug



```bash

\# Dans .env:

LOG\_LEVEL=DEBUG



\# Ou en ligne de commande:

python main.py --mode paper --log-level DEBUG

```



\#### Consulter les Logs



```bash

\# Les logs sont dans:

cd data/logs/



\# Voir les derniers logs:

tail -f thebot.log



\# Chercher des erreurs:

grep ERROR thebot.log

grep CRITICAL thebot.log

```



\#### VÃƒÂ©rifier la SantÃƒÂ© du SystÃƒÂ¨me



```python

\# CrÃƒÂ©er check\_system.py

import psutil



print(f"CPU: {psutil.cpu\_percent()}%")

print(f"RAM: {psutil.virtual\_memory().percent}%")

print(f"Disk: {psutil.disk\_usage('/').percent}%")

```



---



\## Ã°Å¸â€œÅ  VÃƒÂ©rification Post-Installation



\### Checklist ComplÃƒÂ¨te



Avant de lancer en production, vÃƒÂ©rifier:



\- \[ ] Ã¢Å“â€¦ Python 3.9+ installÃƒÂ©

\- \[ ] Ã¢Å“â€¦ TA-Lib installÃƒÂ© et fonctionnel

\- \[ ] Ã¢Å“â€¦ Redis installÃƒÂ© et dÃƒÂ©marrÃƒÂ© (optionnel)

\- \[ ] Ã¢Å“â€¦ Toutes les dÃƒÂ©pendances pip installÃƒÂ©es

\- \[ ] Ã¢Å“â€¦ Fichier `.env` configurÃƒÂ© avec API keys

\- \[ ] Ã¢Å“â€¦ Fichier `config.py` prÃƒÂ©sent

\- \[ ] Ã¢Å“â€¦ Test de connexion Binance rÃƒÂ©ussi

\- \[ ] Ã¢Å“â€¦ Mode Paper Trading testÃƒÂ©

\- \[ ] Ã¢Å“â€¦ Logs crÃƒÂ©ÃƒÂ©s dans `data/logs/`

\- \[ ] Ã¢Å“â€¦ Au moins 8GB RAM disponible

\- \[ ] Ã¢Å“â€¦ Connexion internet stable (>10 Mbps)

\- \[ ] Ã¢Å“â€¦ 2FA activÃƒÂ© sur Binance

\- \[ ] Ã¢Å“â€¦ Restrictions IP configurÃƒÂ©es (recommandÃƒÂ©)

\- \[ ] Ã¢Å“â€¦ Permissions API vÃƒÂ©rifiÃƒÂ©es

\- \[ ] Ã¢Å“â€¦ Capital suffisant (>1000 USDC recommandÃƒÂ©)



\### Test de Performance



```bash

\# Lancer un test de charge

python -m pytest tests/ -v



\# Ou crÃƒÂ©er test\_performance.py

python test\_performance.py

```



---



\## Ã°Å¸Å½â€œ Prochaines Ãƒâ€°tapes



\### 1. Comprendre les StratÃƒÂ©gies



Lisez la documentation des stratÃƒÂ©gies:

```bash

cat docs/STRATEGIES.md

```



\### 2. Configurer le Risk Management



```python

\# Dans config.py, ajuster:

RISK\_PER\_TRADE = 0.01  # 1% si vous ÃƒÂªtes conservateur

MAX\_DAILY\_LOSS = 0.03  # 3% perte max/jour

MAX\_DRAWDOWN = 0.05    # 5% drawdown max

```



\### 3. Backtesting



```bash

\# Tester sur donnÃƒÂ©es historiques

python backtest.py --symbol BTCUSDC --days 30

```



\### 4. Paper Trading Extended



```bash

\# Laisser tourner 7 jours en paper trading

python main.py --mode paper



\# Analyser les rÃƒÂ©sultats:

python analyze\_performance.py

```



\### 5. Monitoring



```bash

\# Installer le dashboard (optionnel)

pip install streamlit plotly



\# Lancer le dashboard

streamlit run monitoring/dashboard\_web.py

```



---



\## Ã°Å¸â€â€ž Mise ÃƒÂ  Jour



\### Mise ÃƒÂ  jour du Code



```bash

\# Sauvegarder votre configuration

cp config.py config.backup.py

cp .env .env.backup



\# Pull les derniÃƒÂ¨res modifications

git pull origin main



\# Mettre ÃƒÂ  jour les dÃƒÂ©pendances

pip install -r requirements.txt --upgrade



\# Restaurer votre configuration si ÃƒÂ©crasÃƒÂ©e

\# (vÃƒÂ©rifier d'abord les changements)

```



\### Mise ÃƒÂ  jour des DÃƒÂ©pendances



```bash

\# Voir les packages outdated

pip list --outdated



\# Mettre ÃƒÂ  jour tout

pip install --upgrade -r requirements.txt



\# Ou un package spÃƒÂ©cifique

pip install --upgrade python-binance

```



---



\## Ã°Å¸â€”â€˜Ã¯Â¸Â DÃƒÂ©sinstallation



Si vous souhaitez dÃƒÂ©sinstaller The Bot:



```bash

\# 1. ArrÃƒÂªter le bot (Ctrl+C si en cours)



\# 2. DÃƒÂ©sactiver l'environnement virtuel

deactivate



\# 3. Supprimer le dossier

cd ..

rm -rf "The Bot"  # Linux/macOS

\# ou

rmdir /s "The Bot"  # Windows



\# 4. (Optionnel) Supprimer Redis

\# macOS:

brew uninstall redis



\# Linux:

sudo apt remove redis-server



\# Windows: DÃƒÂ©sinstaller Memurai

```



---



\## Ã°Å¸â€œÅ¾ Support et Aide



\### Documentation



\- \*\*Guide Principal\*\*: `README.md`

\- \*\*Changelog\*\*: `CHANGELOG.md`

\- \*\*API Documentation\*\*: `docs/API.md`

\- \*\*StratÃƒÂ©gies\*\*: `docs/STRATEGIES.md`

\- \*\*FAQ\*\*: `docs/FAQ.md`



\### CommunautÃƒÂ©



\- \*\*GitHub Issues\*\*: Pour les bugs et feature requests

\- \*\*Discord\*\*: \[Lien vers serveur Discord]

\- \*\*Telegram\*\*: \[Lien vers groupe Telegram]

\- \*\*Email\*\*: support@thebot.trading



\### Ressources Utiles



\*\*Binance\*\*

\- API Documentation: https://binance-docs.github.io/apidocs/spot/en/

\- Testnet: https://testnet.binance.vision/

\- Status: https://www.binance.com/en/support/announcement



\*\*Python\*\*

\- Documentation: https://docs.python.org/3/

\- Package Index: https://pypi.org/



\*\*TA-Lib\*\*

\- Documentation: https://mrjbq7.github.io/ta-lib/

\- Indicators: https://ta-lib.org/function.html



---



\## Ã°Å¸Å½Â¯ Tips pour DÃƒÂ©butants



\### Commencer Petit



```python

\# Recommandations pour dÃƒÂ©buter:

INITIAL\_CAPITAL = 100      # Commencer avec 100 USDC

RISK\_PER\_TRADE = 0.01     # 1% seulement

MAX\_POSITION\_SIZE = 0.1   # 10% max par position

```



\### Observer Avant d'Agir



1\. \*\*Semaine 1\*\*: Paper trading, observer les signaux

2\. \*\*Semaine 2\*\*: Paper trading, analyser les performances

3\. \*\*Semaine 3\*\*: Paper trading, ajuster la configuration

4\. \*\*Semaine 4\*\*: Si rÃƒÂ©sultats positifs, dÃƒÂ©marrer petit en live



\### Surveiller Quotidiennement



```bash

\# Checker les logs tous les jours

tail -100 data/logs/thebot.log



\# Analyser les performances

python analyze\_performance.py

```



\### Ne Jamais



\- Ã¢ÂÅ’ Trader plus que ce que vous pouvez perdre

\- Ã¢ÂÅ’ DÃƒÂ©sactiver le risk management

\- Ã¢ÂÅ’ Modifier la config en cours de trade

\- Ã¢ÂÅ’ Ignorer les alertes de drawdown

\- Ã¢ÂÅ’ Partager vos API keys



\### Toujours



\- Ã¢Å“â€¦ Commencer en Paper Trading

\- Ã¢Å“â€¦ Activer 2FA sur Binance

\- Ã¢Å“â€¦ Utiliser IP whitelisting

\- Ã¢Å“â€¦ Surveiller rÃƒÂ©guliÃƒÂ¨rement

\- Ã¢Å“â€¦ Garder des backups de config

\- Ã¢Å“â€¦ Lire les logs quotidiennement

\- Ã¢Å“â€¦ Comprendre chaque stratÃƒÂ©gie



---



\## Ã¢Å¡â€“Ã¯Â¸Â Disclaimer



\*\*AVERTISSEMENT IMPORTANT\*\*



Le trading de cryptomonnaies comporte des risques ÃƒÂ©levÃƒÂ©s, y compris la perte totale du capital investi. 



\- Ce bot est fourni "tel quel" sans garantie

\- Les performances passÃƒÂ©es ne garantissent pas les rÃƒÂ©sultats futurs

\- L'auteur n'est pas responsable des pertes financiÃƒÂ¨res

\- Utilisez uniquement de l'argent que vous pouvez perdre

\- Consultez un conseiller financier avant de trader



\*\*Vous ÃƒÂªtes seul responsable de vos dÃƒÂ©cisions de trading.\*\*



---



\## Ã°Å¸â€œâ€ž Licence



The Bot - PropriÃƒÂ©taire - Tous droits rÃƒÂ©servÃƒÂ©s



Copyright Ã‚Â© 2025



---



\*\*FÃƒÂ©licitations! Ã°Å¸Å½â€°\*\*



Vous avez terminÃƒÂ© l'installation de The Bot. 



Prochaine ÃƒÂ©tape: Lancer votre premier test en Paper Trading!



```bash

python main.py --mode paper

```



\*\*Bon trading! Ã°Å¸Å¡â‚¬Ã°Å¸â€œË†\*\*



---



\*Guide crÃƒÂ©ÃƒÂ© le: 2025-01-15\*  

\*DerniÃƒÂ¨re mise ÃƒÂ  jour: 2025-01-15\*  

\*Version: 1.0.0\*Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â

Ã¢Å¡Â Ã¯Â¸Â  ATTENTION: MODE LIVE AVEC ARGENT RÃƒâ€°EL!

Ã¢Å¡Â Ã¯Â¸Â  Vous allez trader avec de vrais fonds!

Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â Ã¢Å¡Â Ã¯Â¸Â ...

