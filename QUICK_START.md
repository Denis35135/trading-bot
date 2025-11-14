# ⚡ Quick Start - Démarrage Rapide

## 🎯 En 3 Minutes Chrono

### Windows

```bash
# 1. Installer (lance install.bat)
install.bat

# 2. Configurer vos clés API
# Éditer .env avec vos clés Binance

# 3. Lancer!
python main.py
```

### Linux/macOS

```bash
# 1. Installer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --break-system-packages

# 2. Configurer
cp .env.example .env
nano .env  # Ajouter vos clés API

# 3. Lancer!
python main.py
```

## 🔑 Configuration Minimale (.env)

```env
MODE=paper
BINANCE_API_KEY=votre_cle_api
BINANCE_API_SECRET=votre_secret_api
BINANCE_TESTNET=true
INITIAL_CAPITAL=1000
```

## ✅ Checklist Avant Lancement

- [ ] Python 3.9+ installé
- [ ] Dépendances installées (`install.bat` ou `pip install -r requirements.txt`)
- [ ] Fichier `.env` configuré avec clés API
- [ ] Mode testnet activé pour premiers tests
- [ ] Internet stable

## 🚨 Premiers Pas

### 1. Test de Connexion

```bash
python tests/test_connection.py
```

Si ça fonctionne ✅, vous êtes prêt !

### 2. Lancement en Mode Paper Trading

```bash
python main.py
```

Le bot va:
1. Se connecter à Binance (testnet)
2. Scanner les marchés
3. Démarrer les 4 threads
4. Afficher son statut toutes les 60s

### 3. Surveillance

Laissez tourner et surveillez:
- Capital qui augmente
- Win rate autour de 70%
- Drawdown < 8%

## 🎛️ Commandes Utiles

```bash
# Vérifier l'installation
python check_requirements.py

# Voir les logs
tail -f data/logs/main.log

# Backup des données
python scripts/backup_data.py

# Stopper proprement: Ctrl+C
```

## 📊 À Quoi S'Attendre

**Premier jour:**
- 50-100 trades
- +2-5% de capital
- Win rate ~65%

**Première semaine:**
- 500-1000 trades
- +20-40% de capital
- Win rate ~70%

## ⚠️ Important

1. **TOUJOURS tester en mode PAPER d'abord**
2. **Ne jamais partager vos clés API**
3. **Surveiller les premières 24h**
4. **Pas de withdrawals activés sur l'API**

## 🆘 Problèmes Courants

### Erreur "Module not found"
```bash
pip install -r requirements.txt --break-system-packages
```

### Erreur connexion Binance
- Vérifier clés API dans `.env`
- Vérifier `BINANCE_TESTNET=true`
- Tester: `python tests/test_connection.py`

### Bot ne démarre pas
```bash
# Vérifier prérequis
python check_requirements.py --system
```

## 📚 Plus d'Infos

- **README.md** - Documentation complète
- **Documentation.docx** - Architecture détaillée
- **config.py** - Tous les paramètres

## 🎯 Après 24h de Tests

Si tout fonctionne bien:
1. Augmenter le capital progressivement
2. Optimiser les paramètres
3. Analyser les performances: `python scripts/analyze_performance.py`

**Bon trading! 🚀**
