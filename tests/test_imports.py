#!/usr/bin/env python3
"""
Test Imports - Script de validation pour The Bot
VÃ©rifie que tous les modules peuvent Ãªtre importÃ©s sans erreur
"""

import sys
import importlib
from pathlib import Path

# Couleurs pour l'affichage
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def test_module_import(module_name: str) -> tuple:
    """
    Teste l'import d'un module
    
    Returns:
        (success: bool, error_msg: str)
    """
    try:
        # Tenter d'importer le module
        module = importlib.import_module(module_name)
        
        # VÃ©rifier les attributs __all__ si prÃ©sents
        if hasattr(module, '__all__'):
            exports = module.__all__
            return True, f"OK - {len(exports)} exports"
        else:
            return True, "OK - No __all__ defined"
            
    except ImportError as e:
        return False, f"ImportError: {str(e)}"
    except Exception as e:
        return False,
