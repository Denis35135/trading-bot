#!/usr/bin/env python3
"""
Script de test de connexion ÃƒÂ  Binance
VÃƒÂ©rifie que tout est correctement configurÃƒÂ© avant de lancer le bot
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Ajouter le rÃƒÂ©pertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def print_header():
    """Affiche l'en-tÃƒÂªte"""
    print("\n" + "=" * 60)
    print("Ã°Å¸Â§Âª TEST DE CONNEXION - THE BOT")
    print("=" * 60 + "\n")

def test_imports():
    """Test 1: VÃƒÂ©rifier les imports Python"""
    print("Ã°Å¸â€œÂ¦ Test 1/8: VÃƒÂ©rification des dÃƒÂ©pendances Python...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'ccxt': 'ccxt',
        'binance': 'python-binance',
        'redis': 'redis',
        'psutil': 'psutil',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"   Ã¢Å“â€¦ {pip_name}")
        except ImportError:
            print(f"   Ã¢ÂÅ’ {pip_name} - MANQUANT")
            missing.append(pip_name)
    
    if missing:
        print(f"\nÃ¢Å¡Â Ã¯Â¸Â  Packages manquants: {', '.join(missing)}")
        print(f"Ã°Å¸â€™Â¡ Installez-les avec: pip install {' '.join(missing)}")
        return False
    
    print("   Ã¢Å“â€¦ Toutes les dÃƒÂ©pendances sont installÃƒÂ©es\n")
    return True

def test_talib():
    """Test 2: VÃƒÂ©rifier TA-Lib"""
    print("Ã°Å¸â€œÅ  Test 2/8: VÃƒÂ©rification de TA-Lib...")
    
    try:
        import talib
        print(f"   Ã¢Å“â€¦ TA-Lib version {talib.__version__} installÃƒÂ©")
        
        # Test rapide
        import numpy as np
        test_data = np.random.random(100)
        sma = talib.SMA(test_data, timeperiod=14)
        print("   Ã¢Å“â€¦ TA-Lib fonctionnel\n")
        return True
    except ImportError:
        print("   Ã¢ÂÅ’ TA-Lib non installÃƒÂ©")
        print("   Ã°Å¸â€™Â¡ Guide d'installation:")
        print("      - Windows: TÃƒÂ©lÃƒÂ©charger le wheel depuis")
        print("        https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("      - macOS: brew install ta-lib && pip install TA-Lib")
        print("      - Linux: Voir docs/INSTALLATION.md\n")
        return False
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur TA-Lib: {e}\n")
        return False

def test_env_file():
    """Test 3: VÃƒÂ©rifier le fichier .env"""
    print("Ã°Å¸â€Â Test 3/8: VÃƒÂ©rification du fichier .env...")
    
    if not os.path.exists('.env'):
        print("   Ã¢ÂÅ’ Fichier .env non trouvÃƒÂ©")
        print("   Ã°Å¸â€™Â¡ CrÃƒÂ©ez-le avec: cp .env.example .env")
        print("   Ã°Å¸â€™Â¡ Puis ÃƒÂ©ditez-le avec vos clÃƒÂ©s API Binance\n")
        return False
    
    print("   Ã¢Å“â€¦ Fichier .env trouvÃƒÂ©")
    
    # Charger les variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or api_key == 'your_api_key_here':
            print("   Ã¢Å¡Â Ã¯Â¸Â  BINANCE_API_KEY non configurÃƒÂ©e")
            return False
        
        if not secret_key or secret_key == 'your_secret_key_here':
            print("   Ã¢Å¡Â Ã¯Â¸Â  BINANCE_SECRET_KEY non configurÃƒÂ©e")
            return False
        
        print(f"   Ã¢Å“â€¦ API Key configurÃƒÂ©e (commence par: {api_key[:8]}...)")
        print(f"   Ã¢Å“â€¦ Secret Key configurÃƒÂ©e\n")
        return True
        
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur lecture .env: {e}\n")
        return False

def test_config_file():
    """Test 4: VÃƒÂ©rifier le fichier config.py"""
    print("Ã¢Å¡â„¢Ã¯Â¸Â  Test 4/8: VÃƒÂ©rification du fichier config.py...")
    
    if not os.path.exists('config.py'):
        print("   Ã¢ÂÅ’ Fichier config.py non trouvÃƒÂ©")
        print("   Ã°Å¸â€™Â¡ CrÃƒÂ©ez-le avec: cp config.example.py config.py\n")
        return False
    
    try:
        from config import Config
        config = Config()
        
        print("   Ã¢Å“â€¦ config.py chargÃƒÂ© avec succÃƒÂ¨s")
        print(f"   Ã¢Å“â€¦ Capital initial: ${config.INITIAL_CAPITAL:,.2f}")
        print(f"   Ã¢Å“â€¦ Risk per trade: {config.RISK_PER_TRADE:.1%}")
        print(f"   Ã¢Å“â€¦ Max drawdown: {config.MAX_DRAWDOWN:.1%}\n")
        return True
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur config.py: {e}\n")
        return False

def test_binance_connection():
    """Test 5: Tester la connexion ÃƒÂ  Binance"""
    print("Ã°Å¸Å’Â Test 5/8: Connexion ÃƒÂ  Binance...")
    
    try:
        from dotenv import load_dotenv
        from binance.client import Client
        
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        testnet = os.getenv('TESTNET', 'True').lower() == 'true'
        
        # CrÃƒÂ©er le client
        if testnet:
            print("   Ã°Å¸â€Â§ Mode: TESTNET")
            client = Client(api_key, secret_key, testnet=True)
        else:
            print("   Ã°Å¸â€Â§ Mode: PRODUCTION")
            client = Client(api_key, secret_key)
        
        # Test 1: Ping
        print("   Ã°Å¸â€œÂ¡ Test ping...", end=" ")
        client.ping()
        print("Ã¢Å“â€¦")
        
        # Test 2: Server time
        print("   Ã°Å¸â€¢Â Test server time...", end=" ")
        server_time = client.get_server_time()
        print(f"Ã¢Å“â€¦ ({datetime.fromtimestamp(server_time['serverTime']/1000).strftime('%H:%M:%S')})")
        
        # Test 3: Account info
        print("   Ã°Å¸â€˜Â¤ Test account info...", end=" ")
        account = client.get_account()
        print("Ã¢Å“â€¦")
        
        # Test 4: Balances
        print("   Ã°Å¸â€™Â° Test balances...", end=" ")
        balances = {b['asset']: float(b['free']) 
                   for b in account['balances'] 
                   if float(b['free']) > 0}
        print("Ã¢Å“â€¦")
        
        if balances:
            print("\n   Ã°Å¸â€™Âµ Soldes disponibles:")
            for asset, amount in list(balances.items())[:5]:  # Top 5
                print(f"      Ã¢â‚¬Â¢ {asset}: {amount:,.4f}")
        else:
            print("   Ã¢Å¡Â Ã¯Â¸Â  Aucun solde (normal pour testnet)")
        
        # Test 5: Ticker price
        print("\n   Ã°Å¸â€œÅ  Test ticker price...", end=" ")
        ticker = client.get_symbol_ticker(symbol="BTCUSDC")
        btc_price = float(ticker['price'])
        print(f"Ã¢Å“â€¦ (BTC = ${btc_price:,.2f})")
        
        # Test 6: Klines
        print("   Ã°Å¸â€œË† Test donnÃƒÂ©es historiques...", end=" ")
        klines = client.get_klines(symbol="BTCUSDC", interval="5m", limit=10)
        print(f"Ã¢Å“â€¦ ({len(klines)} bougies rÃƒÂ©cupÃƒÂ©rÃƒÂ©es)")
        
        print("\n   Ã¢Å“â€¦ Connexion Binance opÃƒÂ©rationnelle!\n")
        return True
        
    except Exception as e:
        print(f"\n   Ã¢ÂÅ’ Erreur connexion Binance: {e}")
        print("\n   Ã°Å¸â€Â§ VÃƒÂ©rifiez:")
        print("      1. Vos clÃƒÂ©s API dans .env")
        print("      2. Les permissions de l'API sur Binance")
        print("      3. Votre connexion internet")
        print("      4. Le mode TESTNET/PRODUCTION\n")
        return False

def test_redis():
    """Test 6: Tester Redis (optionnel)"""
    print("Ã°Å¸â€Â´ Test 6/8: Connexion Redis (optionnel)...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("   Ã¢Å“â€¦ Redis connectÃƒÂ© et opÃƒÂ©rationnel")
        
        # Test lecture/ÃƒÂ©criture
        r.setex("test_key", 5, "test_value")
        value = r.get("test_key")
        if value == "test_value":
            print("   Ã¢Å“â€¦ Redis lecture/ÃƒÂ©criture OK\n")
            return True
        
    except Exception as e:
        print("   Ã¢Å¡Â Ã¯Â¸Â  Redis non disponible (optionnel)")
        print(f"      Raison: {e}")
        print("   Ã°Å¸â€™Â¡ Le bot peut fonctionner sans Redis,")
        print("      mais les performances seront rÃƒÂ©duites\n")
        return None  # None = optionnel

def test_directory_structure():
    """Test 7: VÃƒÂ©rifier la structure des dossiers"""
    print("Ã°Å¸â€œÂ Test 7/8: VÃƒÂ©rification de la structure...")
    
    required_dirs = [
        'strategies',
        'risk',
        'ml',
        'exchange',
        'scanner',
        'threads',
        'monitoring',
        'utils',
        'data',
        'data/logs',
        'data/models',
        'data/cache',
        'tests',
        'docs',
        'scripts'
    ]
    
    missing = []
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"   Ã¢Å“â€¦ {directory}/")
        else:
            print(f"   Ã¢Å¡Â Ã¯Â¸Â  {directory}/ - MANQUANT")
            missing.append(directory)
    
    if missing:
        print(f"\n   Ã°Å¸â€™Â¡ CrÃƒÂ©ez les dossiers manquants:")
        for d in missing:
            print(f"      mkdir -p {d}")
    
    print()
    return len(missing) == 0

def test_system_resources():
    """Test 8: VÃƒÂ©rifier les ressources systÃƒÂ¨me"""
    print("Ã°Å¸â€™Â» Test 8/8: VÃƒÂ©rification des ressources systÃƒÂ¨me...")
    
    try:
        import psutil
        
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   Ã°Å¸â€Â· CPU: {cpu_count} cores ({cpu_percent}% utilisÃƒÂ©s)")
        
        if cpu_count < 4:
            print("   Ã¢Å¡Â Ã¯Â¸Â  Minimum 4 cores recommandÃƒÂ©")
        else:
            print("   Ã¢Å“â€¦ CPU suffisant")
        
        # RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        print(f"   Ã°Å¸â€Â· RAM: {ram_gb:.1f} GB total ({ram_available_gb:.1f} GB disponible)")
        
        if ram_gb < 8:
            print("   Ã¢ÂÅ’ Minimum 8 GB requis")
            return False
        elif ram_gb < 16:
            print("   Ã¢Å¡Â Ã¯Â¸Â  16 GB recommandÃƒÂ© pour performances optimales")
        else:
            print("   Ã¢Å“â€¦ RAM suffisante")
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        print(f"   Ã°Å¸â€Â· Disque: {disk_free_gb:.1f} GB disponible")
        
        if disk_free_gb < 10:
            print("   Ã¢Å¡Â Ã¯Â¸Â  Minimum 10 GB recommandÃƒÂ©")
        else:
            print("   Ã¢Å“â€¦ Espace disque suffisant")
        
        print()
        return True
        
    except Exception as e:
        print(f"   Ã¢Å¡Â Ã¯Â¸Â  Impossible de vÃƒÂ©rifier: {e}\n")
        return None

def print_summary(results):
    """Affiche le rÃƒÂ©sumÃƒÂ© des tests"""
    print("\n" + "=" * 60)
    print("Ã°Å¸â€œÅ  RÃƒâ€°SUMÃƒâ€° DES TESTS")
    print("=" * 60 + "\n")
    
    test_names = [
        "DÃƒÂ©pendances Python",
        "TA-Lib",
        "Fichier .env",
        "Fichier config.py",
        "Connexion Binance",
        "Redis (optionnel)",
        "Structure dossiers",
        "Ressources systÃƒÂ¨me"
    ]
    
    passed = 0
    failed = 0
    optional = 0
    
    for i, (name, result) in enumerate(zip(test_names, results), 1):
        if result is True:
            print(f"   Ã¢Å“â€¦ Test {i}: {name}")
            passed += 1
        elif result is None:
            print(f"   Ã¢Å¡Â Ã¯Â¸Â  Test {i}: {name} (optionnel)")
            optional += 1
        else:
            print(f"   Ã¢ÂÅ’ Test {i}: {name}")
            failed += 1
    
    print(f"\n   Total: {passed} rÃƒÂ©ussis, {failed} ÃƒÂ©chouÃƒÂ©s, {optional} optionnels")
    
    print("\n" + "=" * 60)
    
    if failed == 0:
        print("Ã°Å¸Å½â€° TOUS LES TESTS CRITIQUES SONT RÃƒâ€°USSIS!")
        print("=" * 60)
        print("\nÃ¢Å“Â¨ Vous pouvez maintenant lancer The Bot:\n")
        print("   Mode Paper Trading (recommandÃƒÂ©):")
        print("   $ python main.py --mode paper\n")
        print("   Mode Live (argent rÃƒÂ©el):")
        print("   $ python main.py --mode live\n")
        return True
    else:
        print("Ã¢ÂÅ’ CERTAINS TESTS ONT Ãƒâ€°CHOUÃƒâ€°")
        print("=" * 60)
        print("\nÃ¢Å¡Â Ã¯Â¸Â  Corrigez les erreurs avant de lancer le bot.")
        print("Ã°Å¸â€œâ€“ Consultez docs/INSTALLATION.md pour plus d'aide.\n")
        return False

def main():
    """Fonction principale"""
    print_header()
    
    # ExÃƒÂ©cuter tous les tests
    results = [
        test_imports(),
        test_talib(),
        test_env_file(),
        test_config_file(),
        test_binance_connection(),
        test_redis(),
        test_directory_structure(),
        test_system_resources()
    ]
    
    # Afficher le rÃƒÂ©sumÃƒÂ©
    success = print_summary(results)
    
    # Code de sortie
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
