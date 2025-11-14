"""
Features pour The Bot
Alias vers FeatureEngineer pour compatibilitÃ©
"""

from .feature_engineering import FeatureEngineer, FeatureConfig

# Alias pour compatibilitÃ© avec d'autres modules
Features = FeatureEngineer

__all__ = ['FeatureEngineer', 'FeatureConfig', 'Features']


# Fonctions helper pour usage rapide
def create_feature_engineer(config=None):
    """
    CrÃ©e un FeatureEngineer avec configuration optionnelle
    
    Args:
        config: Dict de configuration ou None pour dÃ©faut
        
    Returns:
        FeatureEngineer configurÃ©
    """
    if config:
        feature_config = FeatureConfig(**config)
        return FeatureEngineer(feature_config)
    return FeatureEngineer()


def get_default_feature_names():
    """
    Retourne les noms des features par dÃ©faut
    
    Returns:
        Liste des noms de features
    """
    engineer = FeatureEngineer()
    return engineer.get_feature_names()


def get_feature_count():
    """
    Retourne le nombre de features par dÃ©faut
    
    Returns:
        Nombre de features (30 par dÃ©faut)
    """
    engineer = FeatureEngineer()
    return engineer.get_feature_count()