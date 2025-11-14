"""
Correlation Analyzer
Analyse les corrÃƒÂ©lations entre symboles pour ÃƒÂ©viter le sur-risque
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyseur de corrÃƒÂ©lations entre symboles
    
    FonctionnalitÃƒÂ©s:
    - Calcul de la matrice de corrÃƒÂ©lation
    - DÃƒÂ©tection des paires fortement corrÃƒÂ©lÃƒÂ©es
    - Groupement par cluster de corrÃƒÂ©lation
    - Diversification intelligente du portfolio
    - Alerte sur sur-exposition
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur de corrÃƒÂ©lations
        
        Args:
            config: Configuration
        """
        default_config = {
            'lookback_period': 100,  # Nombre de candles pour corrÃƒÂ©lation
            'high_correlation_threshold': 0.7,  # Seuil corrÃƒÂ©lation forte
            'update_frequency': 3600,  # Mise ÃƒÂ  jour toutes les heures
            'min_data_points': 50  # Minimum de points pour calcul
        }
        
        if config:
            # Gestion objet Config ou dict
if hasattr(config, '__dict__'):
    default_config.update(vars(config))
elif isinstance(config, dict):
    default_config.update(config)
else:
    default_config.update(config if isinstance(config, dict) else {})
        
        self.config = default_config
        self.correlation_matrix = None
        self.last_update = None
        self.price_data = {}  # {symbol: prix_history}
        
        # Statistiques
        self.stats = {
            'total_pairs': 0,
            'high_correlation_pairs': 0,
            'clusters': 0
        }
        
        logger.info("Ã°Å¸â€œÅ  Correlation Analyzer initialisÃƒÂ©")
    
    def add_symbol_data(self, symbol: str, prices: pd.Series):
        """
        Ajoute les donnÃƒÂ©es de prix d'un symbole
        
        Args:
            symbol: Le symbole
            prices: Series de prix
        """
        if len(prices) < self.config['min_data_points']:
            logger.warning(f"Pas assez de donnÃƒÂ©es pour {symbol}: {len(prices)} points")
            return
        
        # Garder seulement les N derniers points
        self.price_data[symbol] = prices.tail(self.config['lookback_period'])
    
    def update_correlations(self):
        """
        Met ÃƒÂ  jour la matrice de corrÃƒÂ©lation
        """
        try:
            if len(self.price_data) < 2:
                logger.warning("Pas assez de symboles pour calculer les corrÃƒÂ©lations")
                return
            
            logger.info(f"Ã°Å¸â€â€ž Calcul des corrÃƒÂ©lations pour {len(self.price_data)} symboles")
            
            # Aligner les longueurs
            min_length = min(len(data) for data in self.price_data.values())
            
            aligned_data = {}
            for symbol, prices in self.price_data.items():
                aligned_data[symbol] = prices.tail(min_length).values
            
            # CrÃƒÂ©er DataFrame et calculer corrÃƒÂ©lations
            df = pd.DataFrame(aligned_data)
            self.correlation_matrix = df.corr()
            
            # Mettre ÃƒÂ  jour les stats
            self._update_stats()
            
            self.last_update = datetime.now()
            
            logger.info(f"Ã¢Å“â€¦ Matrice de corrÃƒÂ©lation mise ÃƒÂ  jour")
            logger.info(f"   Paires fortement corrÃƒÂ©lÃƒÂ©es: {self.stats['high_correlation_pairs']}")
            
        except Exception as e:
            logger.error(f"Erreur calcul corrÃƒÂ©lations: {e}")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Retourne la corrÃƒÂ©lation entre deux symboles
        
        Args:
            symbol1: Premier symbole
            symbol2: DeuxiÃƒÂ¨me symbole
            
        Returns:
            CorrÃƒÂ©lation ou None
        """
        if self.correlation_matrix is None:
            return None
        
        try:
            if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                return self.correlation_matrix.loc[symbol1, symbol2]
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration corrÃƒÂ©lation: {e}")
        
        return None
    
    def get_highly_correlated_pairs(self, threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Retourne les paires fortement corrÃƒÂ©lÃƒÂ©es
        
        Args:
            threshold: Seuil de corrÃƒÂ©lation (utilise config si None)
            
        Returns:
            Liste de tuples (symbol1, symbol2, correlation)
        """
        if self.correlation_matrix is None:
            return []
        
        if threshold is None:
            threshold = self.config['high_correlation_threshold']
        
        pairs = []
        
        # Parcourir la matrice (triangle supÃƒÂ©rieur uniquement)
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                corr = self.correlation_matrix.iloc[i, j]
                
                if abs(corr) > threshold:
                    symbol1 = self.correlation_matrix.index[i]
                    symbol2 = self.correlation_matrix.columns[j]
                    pairs.append((symbol1, symbol2, corr))
        
        # Trier par corrÃƒÂ©lation dÃƒÂ©croissante
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return pairs
    
    def get_symbol_correlations(self, symbol: str) -> Dict[str, float]:
        """
        Retourne toutes les corrÃƒÂ©lations d'un symbole
        
        Args:
            symbol: Le symbole
            
        Returns:
            Dict {symbole: corrÃƒÂ©lation}
        """
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.index:
            return {}
        
        correlations = self.correlation_matrix[symbol].to_dict()
        
        # Enlever l'auto-corrÃƒÂ©lation
        if symbol in correlations:
            del correlations[symbol]
        
        # Trier par corrÃƒÂ©lation absolue dÃƒÂ©croissante
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return sorted_corr
    
    def find_diversified_symbols(self, n: int, existing_symbols: List[str] = None) -> List[str]:
        """
        Trouve N symboles les moins corrÃƒÂ©lÃƒÂ©s entre eux
        
        Args:
            n: Nombre de symboles ÃƒÂ  trouver
            existing_symbols: Symboles dÃƒÂ©jÃƒÂ  en portfolio
            
        Returns:
            Liste de symboles diversifiÃƒÂ©s
        """
        if self.correlation_matrix is None:
            return []
        
        available_symbols = list(self.correlation_matrix.index)
        
        # Enlever les symboles existants
        if existing_symbols:
            available_symbols = [s for s in available_symbols if s not in existing_symbols]
        
        if len(available_symbols) <= n:
            return available_symbols
        
        # Algorithme glouton: ajouter les symboles un par un
        # en minimisant la corrÃƒÂ©lation moyenne avec les dÃƒÂ©jÃƒÂ  sÃƒÂ©lectionnÃƒÂ©s
        selected = []
        
        # Commencer avec le symbole ayant la corrÃƒÂ©lation moyenne la plus faible
        avg_correlations = {}
        for symbol in available_symbols:
            corrs = [abs(self.correlation_matrix.loc[symbol, other]) 
                    for other in available_symbols if other != symbol]
            avg_correlations[symbol] = np.mean(corrs) if corrs else 0
        
        first = min(avg_correlations.items(), key=lambda x: x[1])[0]
        selected.append(first)
        available_symbols.remove(first)
        
        # Ajouter les symboles restants
        while len(selected) < n and available_symbols:
            best_symbol = None
            min_avg_corr = float('inf')
            
            for symbol in available_symbols:
                # Calculer corrÃƒÂ©lation moyenne avec les dÃƒÂ©jÃƒÂ  sÃƒÂ©lectionnÃƒÂ©s
                corrs = [abs(self.correlation_matrix.loc[symbol, sel]) for sel in selected]
                avg_corr = np.mean(corrs)
                
                if avg_corr < min_avg_corr:
                    min_avg_corr = avg_corr
                    best_symbol = symbol
            
            if best_symbol:
                selected.append(best_symbol)
                available_symbols.remove(best_symbol)
        
        logger.info(f"Ã°Å¸Å½Â¯ Symboles diversifiÃƒÂ©s trouvÃƒÂ©s: {', '.join(selected)}")
        logger.info(f"   CorrÃƒÂ©lation moyenne: {self._calculate_avg_correlation(selected):.2f}")
        
        return selected
    
    def cluster_symbols(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Groupe les symboles en clusters selon leurs corrÃƒÂ©lations
        
        Args:
            n_clusters: Nombre de clusters souhaitÃƒÂ©s
            
        Returns:
            Dict {cluster_id: [symboles]}
        """
        if self.correlation_matrix is None:
            return {}
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Utiliser 1-corrÃƒÂ©lation comme distance
            distance_matrix = 1 - np.abs(self.correlation_matrix.values)
            
            # Clustering hiÃƒÂ©rarchique
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # Grouper les symboles par cluster
            clusters = {}
            for i, symbol in enumerate(self.correlation_matrix.index):
                cluster_id = int(labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(symbol)
            
            self.stats['clusters'] = len(clusters)
            
            logger.info(f"Ã°Å¸â€œÅ  {len(clusters)} clusters crÃƒÂ©ÃƒÂ©s")
            for cluster_id, symbols in clusters.items():
                logger.info(f"   Cluster {cluster_id}: {len(symbols)} symboles")
            
            return clusters
            
        except ImportError:
            logger.warning("scikit-learn requis pour le clustering")
            return {}
        except Exception as e:
            logger.error(f"Erreur clustering: {e}")
            return {}
    
    def check_portfolio_correlation(self, symbols: List[str]) -> Dict:
        """
        Analyse la corrÃƒÂ©lation d'un portfolio
        
        Args:
            symbols: Liste des symboles du portfolio
            
        Returns:
            Dict avec analyse
        """
        if self.correlation_matrix is None:
            return {'error': 'Matrice de corrÃƒÂ©lation non disponible'}
        
        try:
            # Filtrer les symboles disponibles
            available = [s for s in symbols if s in self.correlation_matrix.index]
            
            if len(available) < 2:
                return {'error': 'Pas assez de symboles disponibles'}
            
            # Calculer corrÃƒÂ©lation moyenne
            correlations = []
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    corr = self.correlation_matrix.loc[available[i], available[j]]
                    correlations.append(abs(corr))
            
            avg_corr = np.mean(correlations)
            max_corr = max(correlations)
            min_corr = min(correlations)
            
            # Nombre de paires fortement corrÃƒÂ©lÃƒÂ©es
            high_corr_count = sum(1 for c in correlations if c > self.config['high_correlation_threshold'])
            
            # Score de diversification (0-100, 100 = trÃƒÂ¨s diversifiÃƒÂ©)
            diversification_score = max(0, 100 * (1 - avg_corr))
            
            # Avertissement si corrÃƒÂ©lation ÃƒÂ©levÃƒÂ©e
            warning = None
            if avg_corr > 0.6:
                warning = "Portfolio fortement corrÃƒÂ©lÃƒÂ© - risque de concentration ÃƒÂ©levÃƒÂ©"
            elif avg_corr > 0.4:
                warning = "CorrÃƒÂ©lation modÃƒÂ©rÃƒÂ©e - envisager plus de diversification"
            
            return {
                'symbols': available,
                'avg_correlation': avg_corr,
                'max_correlation': max_corr,
                'min_correlation': min_corr,
                'high_correlation_pairs': high_corr_count,
                'diversification_score': diversification_score,
                'warning': warning
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse portfolio: {e}")
            return {'error': str(e)}
    
    def _update_stats(self):
        """Met ÃƒÂ  jour les statistiques"""
        if self.correlation_matrix is None:
            return
        
        n = len(self.correlation_matrix)
        self.stats['total_pairs'] = n * (n - 1) // 2
        
        # Compter les paires fortement corrÃƒÂ©lÃƒÂ©es
        high_corr = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.correlation_matrix.iloc[i, j]) > self.config['high_correlation_threshold']:
                    high_corr += 1
        
        self.stats['high_correlation_pairs'] = high_corr
    
    def _calculate_avg_correlation(self, symbols: List[str]) -> float:
        """
        Calcule la corrÃƒÂ©lation moyenne entre symboles
        
        Args:
            symbols: Liste des symboles
            
        Returns:
            CorrÃƒÂ©lation moyenne
        """
        if self.correlation_matrix is None or len(symbols) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if symbols[i] in self.correlation_matrix.index and symbols[j] in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[symbols[i], symbols[j]]
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques
        
        Returns:
            Dict avec stats
        """
        return {
            'total_symbols': len(self.price_data),
            'total_pairs': self.stats['total_pairs'],
            'high_correlation_pairs': self.stats['high_correlation_pairs'],
            'clusters': self.stats['clusters'],
            'last_update': self.last_update,
            'matrix_available': self.correlation_matrix is not None
        }
    
    def needs_update(self) -> bool:
        """
        VÃƒÂ©rifie si une mise ÃƒÂ  jour est nÃƒÂ©cessaire
        
        Returns:
            True si mise ÃƒÂ  jour nÃƒÂ©cessaire
        """
        if self.correlation_matrix is None:
            return True
        
        if self.last_update is None:
            return True
        
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update > self.config['update_frequency']


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Correlation Analyzer"""
    
    # DonnÃƒÂ©es de test
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    
    # CrÃƒÂ©er des prix corrÃƒÂ©lÃƒÂ©s
    base_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    symbols_data = {
        'BTCUSDC': pd.Series(base_prices, index=dates),
        'ETHUSDC': pd.Series(base_prices + np.random.randn(200) * 2, index=dates),  # CorrÃƒÂ©lÃƒÂ©
        'BNBUSDC': pd.Series(base_prices * 0.5 + np.random.randn(200) * 3, index=dates),  # Moyennement corrÃƒÂ©lÃƒÂ©
        'ADAUSDC': pd.Series(100 + np.cumsum(np.random.randn(200) * 0.3), index=dates),  # IndÃƒÂ©pendant
        'DOGEUSDC': pd.Series(50 + np.cumsum(np.random.randn(200) * 0.2), index=dates)  # IndÃƒÂ©pendant
    }
    
    analyzer = CorrelationAnalyzer()
    
    print("Test Correlation Analyzer")
    print("=" * 50)
    
    # Ajouter les donnÃƒÂ©es
    for symbol, prices in symbols_data.items():
        analyzer.add_symbol_data(symbol, prices)
    
    # Calculer les corrÃƒÂ©lations
    analyzer.update_correlations()
    
    # Tester les fonctions
    print("\n1. Paires fortement corrÃƒÂ©lÃƒÂ©es:")
    pairs = analyzer.get_highly_correlated_pairs(0.5)
    for s1, s2, corr in pairs[:5]:
        print(f"   {s1} <-> {s2}: {corr:.2f}")
    
    print("\n2. CorrÃƒÂ©lations de BTCUSDC:")
    btc_corrs = analyzer.get_symbol_correlations('BTCUSDC')
    for symbol, corr in list(btc_corrs.items())[:3]:
        print(f"   {symbol}: {corr:.2f}")
    
    print("\n3. Portfolio diversifiÃƒÂ© (3 symboles):")
    diversified = analyzer.find_diversified_symbols(3)
    print(f"   {', '.join(diversified)}")
    
    print("\n4. Analyse du portfolio:")
    analysis = analyzer.check_portfolio_correlation(['BTCUSDC', 'ETHUSDC', 'ADAUSDC'])
    print(f"   CorrÃƒÂ©lation moyenne: {analysis['avg_correlation']:.2f}")
    print(f"   Score de diversification: {analysis['diversification_score']:.1f}")
    if analysis['warning']:
        print(f"   Ã¢Å¡Â Ã¯Â¸Â  {analysis['warning']}")
    
    print("\n5. Statistiques:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
