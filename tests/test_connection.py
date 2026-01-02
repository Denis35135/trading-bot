#!/usr/bin/env python3
"""
Script de test de connexion  Binance
Verifie que tout est correctement configure avant de lancer le bot
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def print_header():
    """Affiche l'en-tete"""
    print("\n" + "=" * 60)
    print(" TEST DE CONNEXION - THE BOT")
    print("=" * 60 + "\n")

def test_imports():
    """Test 1: Verifier les imports Python"""
    print(""| Test 1/8: Verification des dependances Python...")
    
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
            print(f"   ""| {pip_name}")
        except ImportError:
            print(f"   ' {pip_name} - MANQUANT")
            missing.append(pip_name)
    
    if missing:
        print(f"\n  Packages manquants: {', '.join(missing)}")
        print(f"' Installez-les avec: pip install {' '.join(missing)}")
        return False
    
    print("   ""| Toutes les dependances sont installees\n")
    return True

def test_talib():
    """Test 2: Verifier TA-Lib"""
    print("" Test 2/8: Verification de TA-Lib...")
    
    try:
        import talib
        print(f"   ""| TA-Lib version {talib.__version__} installe")
        
        # Test rapide
        import numpy as np
        test_data = np.random.random(100)
        sma = talib.SMA(test_data, timeperiod=14)
        print("   ""| TA-Lib fonctionnel\n")
        return True
    except ImportError:
        print("   ' TA-Lib non installe")
        print("   ' Guide d'installation:")
        print("      - Windows: Telecharger le wheel depuis")
        print("        https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("      - macOS: brew install ta-lib && pip install TA-Lib")
        print("      - Linux: Voir docs/INSTALLATION.md\n")
        return False
    except Exception as e:
        print(f"   ' Erreur TA-Lib: {e}\n")
        return False

def test_env_file():
    """Test 3: Verifier le fichier .env"""
    print("" Test 3/8: Verification du fichier .env...")
    
    if not os.path.exists('.env'):
        print("   ' Fichier .env non trouve")
        print("   ' Creez-le avec: cp .env.example .env")
        print("   ' Puis editez-le avec vos cles API Binance\n")
        return False
    
    print("   ""| Fichier .env trouve")
    
    # Charger les variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or api_key == 'your_api_key_here':
            print("     BINANCE_API_KEY non configuree")
            return False
        
        if not secret_key or secret_key == 'your_secret_key_here':
            print("     BINANCE_SECRET_KEY non configuree")
            return False
        
        print(f"   ""| API Key configuree (commence par: {api_key[:8]}...)")
        print(f"   ""| Secret Key configuree\n")
        return True
        
    except Exception as e:
        print(f"   ' Erreur lecture .env: {e}\n")
        return False

def test_config_file():
    """Test 4: Verifier le fichier config.py"""
    print("  Test 4/8: Verification du fichier config.py...")
    
    if not os.path.exists('config.py'):
        print("   ' Fichier config.py non trouve")
        print("   ' Creez-le avec: cp config.example.py config.py\n")
        return False
    
    try:
        from config import Config
        config = Config()
        
        print("   ""| config.py charge avec succes")
        print(f"   ""| Capital initial: ${config.INITIAL_CAPITAL:,.2f}")
        print(f"   ""| Risk per trade: {config.RISK_PER_TRADE:.1%}")
        print(f"   ""| Max drawdown: {config.MAX_DRAWDOWN:.1%}\n")
        return True
    except Exception as e:
        print(f"   ' Erreur config.py: {e}\n")
        return False

def test_binance_connection():
    """Test 5: Tester la connexion  Binance"""
    print("' Test 5/8: Connexion  Binance...")
    
    try:
        from dotenv import load_dotenv
        from binance.client import Client
        
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        testnet = os.getenv('TESTNET', 'True').lower() == 'true'
        
        # Creer le client
        if testnet:
            print("   " Mode: TESTNET")
            client = Client(api_key, secret_key, testnet=True)
        else:
            print("   " Mode: PRODUCTION")
            client = Client(api_key, secret_key)
        
        # Test 1: Ping
        print("   " Test ping...", end=" ")
        client.ping()
        print("""|")
        
        # Test 2: Server time
        print("   " Test server time...", end=" ")
        server_time = client.get_server_time()
        print(f"""| ({datetime.fromtimestamp(server_time['serverTime']/1000).strftime('%H:%M:%S')})")
        
        # Test 3: Account info
        print("   " Test account info...", end=" ")
        account = client.get_account()
        print("""|")
        
        # Test 4: Balances
        print("   ' Test balances...", end=" ")
        balances = {b['asset']: float(b['free']) 
                   for b in account['balances'] 
                   if float(b['free']) > 0}
        print("""|")
        
        if balances:
            print("\n   ' Soldes disponibles:")
            for asset, amount in list(balances.items())[:5]:  # Top 5
                print(f"      " {asset}: {amount:,.4f}")
        else:
            print("     Aucun solde (normal pour testnet)")
        
        # Test 5: Ticker price
        print("\n   " Test ticker price...", end=" ")
        ticker = client.get_symbol_ticker(symbol="BTCUSDC")
        btc_price = float(ticker['price'])
        print(f"""| (BTC = ${btc_price:,.2f})")
        
        # Test 6: Klines
        print("   " Test donnees historiques...", end=" ")
        klines = client.get_klines(symbol="BTCUSDC", interval="5m", limit=10)
        print(f"""| ({len(klines)} bougies recuperees)")
        
        print("\n   ""| Connexion Binance operationnelle!\n")
        return True
        
    except Exception as e:
        print(f"\n   ' Erreur connexion Binance: {e}")
        print("\n   " Verifiez:")
        print("      1. Vos cles API dans .env")
        print("      2. Les permissions de l'API sur Binance")
        print("      3. Votre connexion internet")
        print("      4. Le mode TESTNET/PRODUCTION\n")
        return False

def test_redis():
    """Test 6: Tester Redis (optionnel)"""
    print("" Test 6/8: Connexion Redis (optionnel)...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("   ""| Redis connecte et operationnel")
        
        # Test lecture/ecriture
        r.setex("test_key", 5, "test_value")
        value = r.get("test_key")
        if value == "test_value":
            print("   ""| Redis lecture/ecriture OK\n")
            return True
        
    except Exception as e:
        print("     Redis non disponible (optionnel)")
        print(f"      Raison: {e}")
        print("   ' Le bot peut fonctionner sans Redis,")
        print("      mais les performances seront reduites\n")
        return None  # None = optionnel

def test_directory_structure():
    """Test 7: Verifier la structure des dossiers"""
    print("" Test 7/8: Verification de la structure...")
    
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
            print(f"   ""| {directory}/")
        else:
            print(f"     {directory}/ - MANQUANT")
            missing.append(directory)
    
    if missing:
        print(f"\n   ' Creez les dossiers manquants:")
        for d in missing:
            print(f"      mkdir -p {d}")
    
    print()
    return len(missing) == 0

def test_system_resources():
    """Test 8: Verifier les ressources systeme"""
    print("' Test 8/8: Verification des ressources systeme...")
    
    try:
        import psutil
        
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   " CPU: {cpu_count} cores ({cpu_percent}% utilises)")
        
        if cpu_count < 4:
            print("     Minimum 4 cores recommande")
        else:
            print("   ""| CPU suffisant")
        
        # RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        print(f"   " RAM: {ram_gb:.1f} GB total ({ram_available_gb:.1f} GB disponible)")
        
        if ram_gb < 8:
            print("   ' Minimum 8 GB requis")
            return False
        elif ram_gb < 16:
            print("     16 GB recommande pour performances optimales")
        else:
            print("   ""| RAM suffisante")
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        print(f"   " Disque: {disk_free_gb:.1f} GB disponible")
        
        if disk_free_gb < 10:
            print("     Minimum 10 GB recommande")
        else:
            print("   ""| Espace disque suffisant")
        
        print()
        return True
        
    except Exception as e:
        print(f"     Impossible de verifier: {e}\n")
        return None

def print_summary(results):
    """Affiche le resume des tests"""
    print("\n" + "=" * 60)
    print("" R"SUM" DES TESTS")
    print("=" * 60 + "\n")
    
    test_names = [
        "Dependances Python",
        "TA-Lib",
        "Fichier .env",
        "Fichier config.py",
        "Connexion Binance",
        "Redis (optionnel)",
        "Structure dossiers",
        "Ressources systeme"
    ]
    
    passed = 0
    failed = 0
    optional = 0
    
    for i, (name, result) in enumerate(zip(test_names, results), 1):
        if result is True:
            print(f"   ""| Test {i}: {name}")
            passed += 1
        elif result is None:
            print(f"     Test {i}: {name} (optionnel)")
            optional += 1
        else:
            print(f"   ' Test {i}: {name}")
            failed += 1
    
    print(f"\n   Total: {passed} reussis, {failed} echoues, {optional} optionnels")
    
    print("\n" + "=" * 60)
    
    if failed == 0:
        print("" TOUS LES TESTS CRITIQUES SONT R"USSIS!")
        print("=" * 60)
        print("\n" Vous pouvez maintenant lancer The Bot:\n")
        print("   Mode Paper Trading (recommande):")
        print("   $ python main.py --mode paper\n")
        print("   Mode Live (argent reel):")
        print("   $ python main.py --mode live\n")
        return True
    else:
        print("' CERTAINS TESTS ONT "CHOU"")
        print("=" * 60)
        print("\n  Corrigez les erreurs avant de lancer le bot.")
        print("""" Consultez docs/INSTALLATION.md pour plus d'aide.\n")
        return False

def main():
    """Fonction principale"""
    print_header()
    
    # Executer tous les tests
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
    
    # Afficher le resume
    success = print_summary(results)
    
    # Code de sortie
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
