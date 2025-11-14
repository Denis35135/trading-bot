\# ðŸ”§ Troubleshooting - The Bot



Guide de rÃ©solution des problÃ¨mes courants.



\## ðŸš¨ ProblÃ¨mes de Connexion



\### "Cannot connect to Binance"



\*\*SymptÃ´mes\*\* : Le bot ne peut pas se connecter Ã  Binance



\*\*Solutions\*\* :

1\. VÃ©rifier votre connexion internet

2\. VÃ©rifier que Binance n'est pas en maintenance

3\. Tester manuellement :

```bash

curl https://api.binance.com/api/v3/ping

```

4\. Utiliser un VPN si Binance est bloquÃ© dans votre pays

5\. VÃ©rifier les paramÃ¨tres firewall



\### "Invalid API key"



\*\*Causes\*\* :

\- ClÃ©s API incorrectes dans `.env`

\- Espaces avant/aprÃ¨s les clÃ©s

\- ClÃ©s rÃ©voquÃ©es sur Binance



\*\*Solutions\*\* :

```bash

\# 1. VÃ©rifier .env

cat .env | grep BINANCE



\# 2. RÃ©gÃ©nÃ©rer les clÃ©s sur Binance

\# Profil > SÃ©curitÃ© > Gestion API > CrÃ©er nouvelle clÃ©



\# 3. Copier EXACTEMENT (sans espaces)

BINANCE\_API\_KEY=votre\_clÃ©\_ici

BINANCE\_API\_SECRET=votre\_secret\_ici

```



\### "Timestamp for this request is outside of the recvWindow"



\*\*Cause\*\* : Horloge systÃ¨me dÃ©synchronisÃ©e



\*\*Solutions\*\* :

```bash

\# Windows

net stop w32time \&\& net start w32time



\# Linux

sudo ntpdate -s time.nist.gov



\# macOS

sudo sntp -sS time.apple.com

```



\## ðŸ’¾ ProblÃ¨mes d'Installation



\### "TA-Lib not found"



\*\*Windows\*\* :

```bash

\# TÃ©lÃ©charger le wheel depuis

\# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

pip install TA\_Libâ€‘0.4.24â€‘cp310â€‘cp310â€‘win\_amd64.whl

```



\*\*Linux\*\* :

```bash

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -xzf ta-lib-0.4.0-src.tar.gz

cd ta-lib/

./configure --prefix=/usr

make

sudo make install

pip install TA-Lib

```



\### "Module not found"



\*\*Solution\*\* :

```bash

\# VÃ©rifier que l'environnement virtuel est activÃ©

which python  # Doit pointer vers venv/



\# RÃ©installer les dÃ©pendances

pip install -r requirements.txt --force-reinstall

```



\### "Permission denied"



\*\*Linux/macOS\*\* :

```bash

chmod +x \*.sh

chmod 600 .env

```



\*\*Windows\*\* : ExÃ©cuter PowerShell en tant qu'administrateur



\## ðŸ“Š ProblÃ¨mes de Trading



\### Le bot ne passe aucun trade



\*\*VÃ©rifications\*\* :

1\. \*\*Mode Paper\*\* : VÃ©rifier `TRADING\_MODE=paper` ou `live`

2\. \*\*Capital\*\* : VÃ©rifier `INITIAL\_CAPITAL >= 1000`

3\. \*\*StratÃ©gies\*\* : Au moins une stratÃ©gie activÃ©e

4\. \*\*Symboles\*\* : VÃ©rifier les symboles disponibles

5\. \*\*Logs\*\* : Consulter `data/logs/bot.log`



```bash

\# VÃ©rifier la config

python main.py --show-config



\# Mode debug

python main.py --mode paper --debug

```



\### Trades non profitables



\*\*Analyse\*\* :

1\. Consulter les performances par stratÃ©gie

2\. VÃ©rifier le win rate (doit Ãªtre > 60%)

3\. Ajuster les paramÃ¨tres de risque

4\. DÃ©sactiver les stratÃ©gies sous-performantes



```python

\# Dans config.py

\# RÃ©duire le risque

RISK\_PER\_TRADE = 0.01  # De 2% Ã  1%



\# DÃ©sactiver stratÃ©gie

ACTIVE\_STRATEGIES = \[

&nbsp;   {'name': 'scalping', 'enabled': True, 'allocation': 0.70},

&nbsp;   {'name': 'momentum', 'enabled': False, 'allocation': 0.0},  # DÃ©sactivÃ©

]

```



\### "Insufficient balance"



\*\*Solutions\*\* :

1\. VÃ©rifier le solde sur Binance

2\. TransfÃ©rer des fonds depuis Spot vers Futures (si applicable)

3\. RÃ©duire `MIN\_ORDER\_SIZE` dans config.py

4\. Augmenter le capital



\## ðŸ¤– ProblÃ¨mes ML



\### "Model not trained"



\*\*Solution\*\* :

```bash

\# EntraÃ®ner les modÃ¨les

python -m ml.trainer --train-initial



\# Ou lancer le bot qui entraÃ®nera automatiquement

python main.py --mode paper

```



\### "Insufficient training data"



\*\*Cause\*\* : Pas assez de trades historiques



\*\*Solution\*\* :

```python

\# RÃ©duire le minimum dans config.py

ML\_CONFIG = {

&nbsp;   'min\_samples': 1000,  # Au lieu de 10000

}



\# Ou utiliser des donnÃ©es synthÃ©tiques pour tests

python scripts/generate\_training\_data.py

```



\### ModÃ¨les ne prÃ©disent que HOLD



\*\*Causes\*\* :

\- Confidence threshold trop Ã©levÃ©

\- ModÃ¨les mal entraÃ®nÃ©s

\- Features manquantes



\*\*Solutions\*\* :

```python

\# RÃ©duire le threshold

ML\_CONFIDENCE\_THRESHOLD = 0.60  # Au lieu de 0.65



\# RÃ©entraÃ®ner

python -m ml.auto\_retrainer --force-retrain

```



\## ðŸ’» ProblÃ¨mes de Performance



\### Bot trop lent / Lag



\*\*Solutions\*\* :

1\. RÃ©duire le nombre de symboles

```python

SYMBOLS\_TO\_TRADE = 10  # Au lieu de 20

```



2\. DÃ©sactiver certaines stratÃ©gies

3\. Augmenter les intervalles de calcul

4\. VÃ©rifier l'utilisation CPU/RAM



```bash

\# Monitorer les ressources

python -m utils.monitor\_resources

```



\### "Out of memory"



\*\*Solutions\*\* :

```python

\# config.py

MAX\_MEMORY\_MB = 1500  # RÃ©duire la limite

TICK\_BUFFER\_SIZE = 2000  # RÃ©duire le buffer

```



```bash

\# RedÃ©marrer pÃ©riodiquement

python scripts/scheduled\_restart.py --interval 12h

```



\### High CPU usage



\*\*Causes\*\* : Trop de threads, calculs intensifs



\*\*Solutions\*\* :

```python

MAX\_THREADS = 2  # RÃ©duire de 4 Ã  2

ML\_ENABLED = False  # DÃ©sactiver temporairement ML

```



\## ðŸ”’ ProblÃ¨mes de SÃ©curitÃ©



\### "API key compromised"



\*\*Actions immÃ©diates\*\* :

1\. RÃ©voquer immÃ©diatement la clÃ© sur Binance

2\. CrÃ©er une nouvelle clÃ©

3\. Activer 2FA si pas dÃ©jÃ  fait

4\. VÃ©rifier l'historique des transactions

5\. Changer le mot de passe Binance



\### "Unauthorized access"



\*\*PrÃ©vention\*\* :

```bash

\# Restreindre les permissions .env

chmod 600 .env



\# Ne jamais committer .env

echo ".env" >> .gitignore



\# Utiliser IP whitelist sur Binance

```



\## ðŸ“ ProblÃ¨mes de Fichiers



\### "Config file not found"



\*\*Solution\*\* :

```bash

\# Copier depuis l'exemple

cp config.example.py config.py

cp .env.example .env



\# VÃ©rifier les chemins

ls -la config.py .env

```



\### "Cannot write to logs"



\*\*Solution\*\* :

```bash

\# CrÃ©er les dossiers

mkdir -p data/logs data/models data/cache



\# Permissions

chmod 755 data/

chmod 755 data/logs/

```



\### Database locked



\*\*Cause\*\* : SQLite utilisÃ© par plusieurs processus



\*\*Solutions\*\* :

```bash

\# ArrÃªter tous les processus

pkill -f "python main.py"



\# Supprimer le lock

rm data/bot.db-journal



\# Ou utiliser PostgreSQL pour production

```



\## ðŸŒ ProblÃ¨mes de WebSocket



\### WebSocket disconnects frequently



\*\*Causes\*\* :

\- Connexion internet instable

\- Firewall bloque les WebSockets

\- Proxy incompatible



\*\*Solutions\*\* :

```python

\# config.py

WEBSOCKET\_CONFIG = {

&nbsp;   'ping\_interval': 30,  # Augmenter

&nbsp;   'reconnect\_delay': 10,  # Augmenter

&nbsp;   'max\_reconnect\_attempts': 20  # Augmenter

}

```



\### "WebSocket connection failed"



\*\*Solution\*\* :

```bash

\# Tester manuellement

python -c "

from websocket import create\_connection

ws = create\_connection('wss://stream.binance.com:9443/ws/btcusdt@trade')

print(ws.recv())

"

```



\## ðŸ“Š ProblÃ¨mes de Dashboard



\### Dashboard ne s'affiche pas



\*\*Solutions\*\* :

```python

\# VÃ©rifier la config

DASHBOARD\_CONFIG = {

&nbsp;   'enabled': True,

&nbsp;   'refresh\_interval': 10

}

```



```bash

\# Lancer avec verbose

python main.py --mode paper --verbose

```



\### DonnÃ©es incorrectes affichÃ©es



\*\*Solution\*\* :

```bash

\# Vider le cache

rm -rf data/cache/\*



\# RedÃ©marrer le bot

```



\## ðŸ” Debugging AvancÃ©



\### Activer le Mode Debug



```bash

\# Logs dÃ©taillÃ©s

export LOG\_LEVEL=DEBUG

python main.py --mode paper --debug

```



\### Consulter les Logs



```bash

\# Logs en temps rÃ©el

tail -f data/logs/bot.log



\# Erreurs seulement

grep ERROR data/logs/bot.log



\# Filtrer par module

grep "strategies" data/logs/bot.log

```



\### Profiling Performance



```python

\# Dans le code

import cProfile

import pstats



profiler = cProfile.Profile()

profiler.enable()



\# Votre code ici



profiler.disable()

stats = pstats.Stats(profiler)

stats.sort\_stats('cumulative')

stats.print\_stats(20)

```



\## ðŸ†˜ Obtenir de l'Aide



\### Checklist Avant de Demander de l'Aide



\- \[ ] ConsultÃ© la documentation

\- \[ ] VÃ©rifiÃ© les logs dans `data/logs/`

\- \[ ] TestÃ© en mode debug

\- \[ ] VÃ©rifiÃ© la configuration

\- \[ ] RecherchÃ© l'erreur sur Google/GitHub Issues



\### Informations Ã  Fournir



```bash

\# GÃ©nÃ©rer un rapport de debug

python scripts/generate\_debug\_report.py



\# Inclure:

\# - Version Python

\# - Version du bot

\# - OS

\# - Logs rÃ©cents (sans clÃ©s API!)

\# - Configuration (sans secrets!)

```



\### OÃ¹ Demander de l'Aide



1\. \*\*GitHub Issues\*\* : https://github.com/votre-repo/issues

2\. \*\*Discord\*\* : Lien du serveur

3\. \*\*Email Support\*\* : support@thebot.com



\## ðŸ“ FAQ



\### Q: Le bot peut-il perdre tout mon argent ?

\*\*R\*\*: Oui, le trading comporte des risques. C'est pourquoi le bot a des protections (stop loss, max drawdown, circuit breakers). Commencez TOUJOURS en Paper Trading.



\### Q: Combien de temps avant d'Ãªtre profitable ?

\*\*R\*\*: GÃ©nÃ©ralement 1-4 semaines pour que le bot s'adapte au marchÃ© actuel. Les premiers jours servent au rodage.



\### Q: Puis-je modifier les stratÃ©gies pendant que le bot tourne ?

\*\*R\*\*: Oui, modifiez `config.py` et le bot rechargera automatiquement (avec HOT\_RELOAD\_CONFIG=True).



\### Q: Combien de capital minimum ?

\*\*R\*\*: 1000 USDC recommandÃ©. Minimum absolu: 500 USDC.



\### Q: Le bot fonctionne sur Mac/Linux ?

\*\*R\*\*: Oui, compatible Windows, Linux et macOS.



---



\*\*Toujours pas rÃ©solu ? Contactez le support ! ðŸ†˜\*\*



\*DerniÃ¨re mise Ã  jour : Octobre 2024\*

