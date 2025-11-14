# 🚀 AUTOBOT ULTIMATE

Bot de trading automatisé haute performance pour Binance

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

## 📊 Performances Cibles

| Métrique | Objectif |
|----------|----------|
| **ROI Mensuel** | 30-60% |
| **Sharpe Ratio** | 2.0-3.0 |
| **Win Rate** | 65-75% |
| **Max Drawdown** | < 8% |
| **Trades/Jour** | 100-300 |
| **Latence** | 50-200ms |

## ✨ Caractéristiques

### Architecture Multi-Thread (4 Threads)
- 🔄 **Market Data Handler** - Gestion WebSocket temps réel
- 🎯 **Strategy Engine** - 5 stratégies complémentaires
- ⚡ **Execution Engine** - Exécution optimisée des ordres
- 🛡️ **Risk Monitor** - Surveillance continue du risque

### 5 Stratégies de Trading
1. **Scalping Intelligent** (40%) - Profits rapides 0.3-0.5%
2. **Momentum Breakout** (25%) - Capture des mouvements forts
3. **Mean Reversion** (20%) - Retours à la moyenne
4. **Pattern Recognition** (10%) - Patterns chartistes
5. **ML Prediction** (5%) - Machine Learning léger

### Gestion des Risques
- ✅ Position sizing intelligent
- ✅ Stop loss adaptatif
- ✅ Circuit breakers (3 niveaux)
- ✅ Max drawdown 8%
- ✅ VaR monitoring

## 🖥️ Prérequis Système

### Minimum
- **OS:** Windows 10/11, Linux, macOS
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Disque:** 10 GB libre
- **Internet:** Connexion stable

### Recommandé
- **CPU:** 8+ cores
- **RAM:** 16 GB
- **SSD:** 50 GB libre

## 📦 Installation

### Windows

1. **Cloner le repository**
```bash
git clone https://github.com/votre-repo/autobot-ultimate.git
cd autobot-ultimate
```

2. **Lancer l'installation automatique**
```bash
install.bat
```

3. **Configurer les API Keys**
- Copier `.env.example` vers `.env`
- Éditer `.env` avec vos clés Binance

4. **Lancer le bot**
```bash
python main.py
```

### Linux / macOS

1. **Cloner et installer**
```bash
git clone https://github.com/votre-repo/autobot-ultimate.git
cd autobot-ultimate

# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt --break-system-packages
```

2. **Configurer**
```bash
cp .env.example .env
nano .env  # Éditer avec vos clés API
```

3. **Lancer**
```bash
python main.py
```

## 🔑 Configuration Binance API

### Obtenir les clés API

1. Se connecter sur [Binance](https://www.binance.com/)
2. Aller dans **Compte** > **API Management**
3. Créer une nouvelle API Key
4. **Permissions requises:**
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Enable Withdrawals (DÉSACTIVER pour sécurité)

### Mode Paper Trading (Testnet)

Pour tester sans risque:

1. Créer un compte sur [Binance Testnet](https://testnet.binance.vision/)
2. Obtenir des clés API testnet
3. Dans `.env`, mettre `BINANCE_TESTNET=true`

## ⚙️ Configuration

### Fichier .env

```env
# Mode
MODE=paper  # 'paper' ou 'live'

# Binance
BINANCE_API_KEY=votre_cle_api
BINANCE_API_SECRET=votre_secret_api
BINANCE_TESTNET=true

# Capital
INITIAL_CAPITAL=1000
```

### Fichiers de configuration (data/configs/)

- **default_config.json** - Paramètres principaux
- **risk_config.json** - Gestion des risques
- **strategies_config.json** - Configuration des stratégies

## 🚀 Utilisation

### Démarrage Standard

```bash
python main.py
```

### Mode Debug

```bash
LOG_LEVEL=DEBUG python main.py
```

### Vérification Système

```bash
python check_requirements.py --system
```

### Tests

```bash
# Test de connexion
python tests/test_connection.py

# Test des stratégies
python tests/test_strategies.py

# Tests complets
pytest tests/
```

## 📊 Monitoring

Le bot affiche son statut toutes les 60 secondes:

```
╔══════════════════════════════════════╗
║ AUTOBOT STATUS - 14:30:45           ║
╠══════════════════════════════════════╣
║ Mode: PAPER                          ║
║ Capital: $1,250.00                   ║
║ P&L Today: +$50.00 (+4.17%)         ║
║ Drawdown: 2.30%                      ║
║ Win Rate: 72.5%                      ║
║ Positions: 8/20                      ║
║ Trades/Day: 145                      ║
╠══════════════════════════════════════╣
║ Threads:                             ║
║  • Market Data: 🟢                   ║
║  • Strategy: 🟢                      ║
║  • Execution: 🟢                     ║
║  • Risk: 🟢                          ║
╠══════════════════════════════════════╣
║ Status: 🟢 RUNNING                   ║
╚══════════════════════════════════════╝
```

## 🛠️ Maintenance

### Backup des Données

```bash
python scripts/backup_data.py
```

### Nettoyage des Logs

```bash
python scripts/clean_logs.py
```

### Réentraînement des Modèles ML

```bash
python scripts/train_models.py
```

### Optimisation des Paramètres

```bash
python scripts/optimize_parameters.py
```

## 📈 Backtesting

```bash
python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31
```

## ⚠️ Avertissements

- ⚠️ **RISQUE**: Le trading comporte des risques. Ne tradez qu'avec de l'argent que vous pouvez vous permettre de perdre.
- ⚠️ **TESTEZ D'ABORD**: Toujours tester en mode PAPER TRADING avant le mode LIVE
- ⚠️ **SÉCURITÉ**: Ne partagez JAMAIS vos clés API
- ⚠️ **SURVEILLANCE**: Surveillez régulièrement le bot

## 🐛 Dépannage

### Le bot ne démarre pas

```bash
# Vérifier les dépendances
python check_requirements.py

# Vérifier la connexion Binance
python tests/test_connection.py
```

### Erreur de mémoire

Réduire `MAX_MEMORY_MB` dans `.env` ou `config.py`

### Latence élevée

- Vérifier votre connexion internet
- Se rapprocher d'un serveur Binance (VPS)

## 📚 Documentation Complète

Voir le fichier `Documentation.docx` pour:
- Architecture détaillée
- Explication des stratégies
- Configuration avancée
- Optimisations

## 🤝 Support

- 📧 Email: support@autobot-ultimate.com
- 💬 Discord: [Rejoindre](https://discord.gg/autobot)
- 📖 Wiki: [Documentation](https://wiki.autobot-ultimate.com)

## 📝 License

MIT License - voir [LICENSE](LICENSE)

## 🎯 Roadmap

- [ ] Interface web (dashboard)
- [ ] Stratégies DeFi
- [ ] Support multi-exchanges
- [ ] Mobile app
- [ ] Cloud deployment

---

**⚠️ Disclaimer**: Ce bot est fourni "tel quel" sans garantie. L'utilisation est à vos risques et périls.

**Made with ❤️ for traders**
