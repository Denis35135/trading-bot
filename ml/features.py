"""
Features pour The Bot
Alias vers FeatureEngineer pour compatibilite
"""

from .feature_engineering import FeatureEngineer, FeatureConfig

# Alias pour compatibilite avec d'autres modules
Features = FeatureEngineer

__all__ = ['FeatureEngineer', 'FeatureConfig', 'Features']


# Fonctions helper pour usage rapide
def create_feature_engineer(config=None):
    """
    Cree un FeatureEngineer avec configuration optionnelle
    
    Args:
        config: Dict de configuration ou None pour defaut
        
    Returns:
        FeatureEngineer configure
    """
    if config:
        feature_config = FeatureConfig(**config)
        return FeatureEngineer(feature_config)
    return FeatureEngineer()


def get_default_feature_names():
    """
    Retourne les noms des features par defaut
    
    Returns:
        Liste des noms de features
    """
    engineer = FeatureEngineer()
    return engineer.get_feature_names()


def get_feature_count():
    """
    Retourne le nombre de features par defaut
    
    Returns:
        Nombre de features (30 par defaut)
    """
    engineer = FeatureEngineer()
    return engineer.get_feature_count()
