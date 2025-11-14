"""
Models pour The Bot
Alias vers MLEnsemble pour compatibilité
"""

from .ensemble import MLEnsemble

# Alias pour compatibilité avec d'autres modules
Models = MLEnsemble

__all__ = ['MLEnsemble', 'Models']


# Fonctions helper pour usage rapide
def create_ensemble(config=None):
    """
    Crée un MLEnsemble avec configuration optionnelle
    
    Args:
        config: Dict de configuration ou None pour défaut
        
    Returns:
        MLEnsemble configuré
    """
    return MLEnsemble(config)


def load_ensemble(filepath):
    """
    Charge un ensemble depuis un fichier
    
    Args:
        filepath: Chemin vers les modèles
        
    Returns:
        MLEnsemble chargé
    """
    ensemble = MLEnsemble()
    ensemble.load(filepath)
    return ensemble


def get_default_config():
    """
    Retourne la configuration par défaut
    
    Returns:
        Dict avec config par défaut
    """
    return {
        'n_estimators': 100,
        'max_depth_lgb': 6,
        'max_depth_xgb': 5,
        'max_depth_rf': 8,
        'learning_rate': 0.1,
        'n_jobs': 4,
        'confidence_threshold': 0.65
    }