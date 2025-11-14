#!/usr/bin/env python3
"""
✅ Requirements Checker
Vérifie que toutes les dépendances Python sont installées
"""

import sys
import subprocess
from typing import List, Tuple


# Liste des packages requis (selon Documentation.docx)
REQUIRED_PACKAGES = [
    ('pandas', '1.4.0'),
    ('numpy', '1.22.0'),
    ('scikit-learn', '1.0.2'),
    ('xgboost', '1.5.1'),
    ('lightgbm', '3.3.2'),
    ('python-binance', '1.0.16'),
    ('websocket-client', '1.3.1'),
    ('ta-lib', '0.4.24'),
    ('ccxt', '2.5.0'),
    ('redis', '4.1.0'),
    ('psutil', '5.9.0'),
]

# Packages optionnels (recommandés mais pas obligatoires)
OPTIONAL_PACKAGES = [
    ('numba', '0.55.0'),
    ('matplotlib', '3.5.0'),
    ('seaborn', '0.11.0'),
    ('plotly', '5.6.0'),
]


def check_python_version() -> Tuple[bool, str]:
    """Vérifie la version de Python (3.9+ requis)"""
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 9:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (3.9+ requis)"


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Vérifie si un package est installé
    
    Args:
        package_name: Nom du package
        min_version: Version minimale requise (optionnel)
        
    Returns:
        (is_installed, message)
    """
    try:
        # Import du package
        if package_name == 'python-binance':
            import binance
            package_obj = binance
        elif package_name == 'scikit-learn':
            import sklearn
            package_obj = sklearn
        elif package_name == 'websocket-client':
            import websocket
            package_obj = websocket
        elif package_name == 'ta-lib':
            import talib
            package_obj = talib
        else:
            package_obj = __import__(package_name)
        
        # Récupère la version
        version = 'unknown'
        if hasattr(package_obj, '__version__'):
            version = package_obj.__version__
        elif hasattr(package_obj, 'VERSION'):
            version = package_obj.VERSION
            
        return True, f"✅ {package_name} ({version})"
        
    except ImportError:
        return False, f"❌ {package_name} - NON INSTALLÉ"
    except Exception as e:
        return False, f"⚠️  {package_name} - Erreur: {e}"


def install_package(package_name: str) -> bool:
    """
    Installe un package via pip
    
    Args:
        package_name: Nom du package
        
    Returns:
        True si installation réussie
    """
    try:
        print(f"Installation de {package_name}...")
        subprocess.check_call([
            sys.executable, 
            '-m', 
            'pip', 
            'install', 
            package_name,
            '--break-system-packages'  # Pour éviter erreur sur certains systèmes
        ])
        return True
    except Exception as e:
        print(f"❌ Erreur installation {package_name}: {e}")
        return False


def check_all_requirements(auto_install: bool = False) -> Tuple[bool, List[str]]:
    """
    Vérifie toutes les dépendances
    
    Args:
        auto_install: Si True, installe automatiquement les packages manquants
        
    Returns:
        (all_ok, list_of_issues)
    """
    print("="*60)
    print("🔍 VÉRIFICATION DES DÉPENDANCES")
    print("="*60 + "\n")
    
    issues = []
    missing_packages = []
    
    # Check Python version
    py_ok, py_msg = check_python_version()
    print(py_msg)
    if not py_ok:
        issues.append(py_msg)
        
    print("\n📦 Packages requis:")
    print("-"*60)
    
    # Check packages requis
    for package, min_version in REQUIRED_PACKAGES:
        is_installed, msg = check_package(package, min_version)
        print(msg)
        
        if not is_installed:
            issues.append(f"{package} manquant")
            missing_packages.append(package)
            
    # Auto-install si demandé
    if auto_install and missing_packages:
        print("\n🔧 Installation automatique des packages manquants...")
        for package in missing_packages:
            if install_package(package):
                print(f"✅ {package} installé")
            else:
                print(f"❌ Échec installation {package}")
                
    print("\n📦 Packages optionnels:")
    print("-"*60)
    
    # Check packages optionnels
    for package, min_version in OPTIONAL_PACKAGES:
        is_installed, msg = check_package(package, min_version)
        print(msg)
        
    # Résumé
    print("\n" + "="*60)
    if not issues:
        print("✅ TOUTES LES DÉPENDANCES SONT INSTALLÉES")
        print("="*60 + "\n")
        return True, []
    else:
        print(f"⚠️  {len(issues)} PROBLÈME(S) DÉTECTÉ(S)")
        print("="*60)
        for issue in issues:
            print(f"  • {issue}")
        print()
        
        if missing_packages:
            print("💡 Pour installer les packages manquants:")
            print(f"   pip install {' '.join(missing_packages)} --break-system-packages")
            print()
            
        return False, issues


def generate_requirements_txt():
    """Génère le fichier requirements.txt"""
    print("📝 Génération de requirements.txt...")
    
    requirements = [
        "# Requirements pour AUTOBOT ULTIMATE",
        "# Installation: pip install -r requirements.txt --break-system-packages",
        "",
        "# Packages essentiels",
    ]
    
    for package, version in REQUIRED_PACKAGES:
        requirements.append(f"{package}>={version}")
        
    requirements.extend([
        "",
        "# Packages optionnels (recommandés)",
    ])
    
    for package, version in OPTIONAL_PACKAGES:
        requirements.append(f"# {package}>={version}  # Optionnel")
        
    # Écrit le fichier
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
        
    print("✅ requirements.txt généré")


def check_system_requirements():
    """Vérifie les prérequis système"""
    print("\n🖥️  PRÉREQUIS SYSTÈME")
    print("-"*60)
    
    import platform
    import psutil
    
    # OS
    print(f"OS: {platform.system()} {platform.release()}")
    
    # CPU
    cpu_count = psutil.cpu_count()
    print(f"CPU: {cpu_count} cores")
    
    if cpu_count < 4:
        print("  ⚠️  Recommandé: 4+ cores")
    else:
        print("  ✅ OK")
        
    # RAM
    ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    print(f"RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 8:
        print("  ⚠️  Recommandé: 8+ GB")
    elif ram_gb < 16:
        print("  ⚠️  Idéal: 16+ GB")
    else:
        print("  ✅ OK")
        
    # Disque
    disk = psutil.disk_usage('/')
    disk_free_gb = disk.free / 1024 / 1024 / 1024
    print(f"Disque: {disk_free_gb:.1f} GB libres")
    
    if disk_free_gb < 5:
        print("  ⚠️  Recommandé: 5+ GB libres")
    else:
        print("  ✅ OK")


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vérifie les dépendances du bot')
    parser.add_argument('--install', action='store_true', 
                       help='Installe automatiquement les packages manquants')
    parser.add_argument('--generate', action='store_true',
                       help='Génère requirements.txt')
    parser.add_argument('--system', action='store_true',
                       help='Vérifie les prérequis système')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_requirements_txt()
        return 0
        
    if args.system:
        check_system_requirements()
        
    # Vérifie les dépendances
    all_ok, issues = check_all_requirements(auto_install=args.install)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
