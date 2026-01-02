"""
Correlation Analyzer
Analyse les correlations entre symboles pour eviter le sur-risque
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyseur de correlations entre symboles
    
    Fonctionnalites:
    - Calcul de la matrice de correlation
    - Detection des paires fortement correlees
    - Groupement par cluster de correlation
    - Diversification intelligente du portfolio
    - Alerte sur sur-exposition
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur de correlations
        
        Args:
            config: Configuration
        """
        default_config = {
            'lookback_period': 100,  # Nombre de candles pour correlation
            'high_correlation_threshold': 0.7,  # Seuil correlation forte
            'update_frequency': 3600,  # Mise  jour toutes les heures
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
        
        logger.info("" Correlation Analyzer initialise")
    
    def add_symbol_data(self, symbol: str, prices: pd.Series):
        """
        Ajoute les donnees de prix d'un symbole
        
        Args:
            symbol: Le symbole
            prices: Series de prix
        """
        if len(prices) < self.config['min_data_points']:
            logger.warning(f"Pas assez de donnees pour {symbol}: {len(prices)} points")
            return
        
        # Garder seulement les N derniers points
        self.price_data[symbol] = prices.tail(self.config['lookback_period'])
    
    def update_correlations(self):
        """
        Met  jour la matrice de correlation
        """
        try:
            if len(self.price_data) < 2:
                logger.warning("Pas assez de symboles pour calculer les correlations")
                return
            
            logger.info(f""" Calcul des correlations pour {len(self.price_data)} symboles")
            
            # Aligner les longueurs
            min_length = min(len(data) for data in self.price_data.values())
            
            aligned_data = {}
            for symbol, prices in self.price_data.items():
                aligned_data[symbol] = prices.tail(min_length).values
            
            # Creer DataFrame et calculer correlations
            df = pd.DataFrame(aligned_data)
            self.correlation_matrix = df.corr()
            
            # Mettre  jour les stats
            self._update_stats()
            
            self.last_update = datetime.now()
            
            logger.info(f"""| Matrice de correlation mise  jour")
            logger.info(f"   Paires fortement correlees: {self.stats['high_correlation_pairs']}")
            
        except Exception as e:
            logger.error(f"Erreur calcul correlations: {e}")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Retourne la correlation entre deux symboles
        
        Args:
            symbol1: Premier symbole
            symbol2: Deuxieme symbole
            
        Returns:
            Correlation ou None
        """
        if self.correlation_matrix is None:
            return None
        
        try:
            if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                return self.correlation_matrix.loc[symbol1, symbol2]
        except Exception as e:
            logger.error(f"Erreur recuperation correlation: {e}")
        
        return None
    
    def get_highly_correlated_pairs(self, threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Retourne les paires fortement correlees
        
        Args:
            threshold: Seuil de correlation (utilise config si None)
            
        Returns:
            Liste de tuples (symbol1, symbol2, correlation)
        """
        if self.correlation_matrix is None:
            return []
        
        if threshold is None:
            threshold = self.config['high_correlation_threshold']
        
        pairs = []
        
        # Parcourir la matrice (triangle superieur uniquement)
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                corr = self.correlation_matrix.iloc[i, j]
                
                if abs(corr) > threshold:
                    symbol1 = self.correlation_matrix.index[i]
                    symbol2 = self.correlation_matrix.columns[j]
                    pairs.append((symbol1, symbol2, corr))
        
        # Trier par correlation decroissante
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return pairs
    
    def get_symbol_correlations(self, symbol: str) -> Dict[str, float]:
        """
        Retourne toutes les correlations d'un symbole
        
        Args:
            symbol: Le symbole
            
        Returns:
            Dict {symbole: correlation}
        """
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.index:
            return {}
        
        correlations = self.correlation_matrix[symbol].to_dict()
        
        # Enlever l'auto-correlation
        if symbol in correlations:
            del correlations[symbol]
        
        # Trier par correlation absolue decroissante
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return sorted_corr
    
    def find_diversified_symbols(self, n: int, existing_symbols: List[str] = None) -> List[str]:
        """
        Trouve N symboles les moins correles entre eux
        
        Args:
            n: Nombre de symboles  trouver
            existing_symbols: Symboles dej en portfolio
            
        Returns:
            Liste de symboles diversifies
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
        # en minimisant la correlation moyenne avec les dej selectionnes
        selected = []
        
        # Commencer avec le symbole ayant la correlation moyenne la plus faible
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
                # Calculer correlation moyenne avec les dej selectionnes
                corrs = [abs(self.correlation_matrix.loc[symbol, sel]) for sel in selected]
                avg_corr = np.mean(corrs)
                
                if avg_corr < min_avg_corr:
                    min_avg_corr = avg_corr
                    best_symbol = symbol
            
            if best_symbol:
                selected.append(best_symbol)
                available_symbols.remove(best_symbol)
        
        logger.info(f" Symboles diversifies trouves: {', '.join(selected)}")
        logger.info(f"   Correlation moyenne: {self._calculate_avg_correlation(selected):.2f}")
        
        return selected
    
    def cluster_symbols(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Groupe les symboles en clusters selon leurs correlations
        
        Args:
            n_clusters: Nombre de clusters souhaites
            
        Returns:
            Dict {cluster_id: [symboles]}
        """
        if self.correlation_matrix is None:
            return {}
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Utiliser 1-correlation comme distance
            distance_matrix = 1 - np.abs(self.correlation_matrix.values)
            
            # Clustering hierarchique
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
            
            logger.info(f"" {len(clusters)} clusters crees")
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
        Analyse la correlation d'un portfolio
        
        Args:
            symbols: Liste des symboles du portfolio
            
        Returns:
            Dict avec analyse
        """
        if self.correlation_matrix is None:
            return {'error': 'Matrice de correlation non disponible'}
        
        try:
            # Filtrer les symboles disponibles
            available = [s for s in symbols if s in self.correlation_matrix.index]
            
            if len(available) < 2:
                return {'error': 'Pas assez de symboles disponibles'}
            
            # Calculer correlation moyenne
            correlations = []
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    corr = self.correlation_matrix.loc[available[i], available[j]]
                    correlations.append(abs(corr))
            
            avg_corr = np.mean(correlations)
            max_corr = max(correlations)
            min_corr = min(correlations)
            
            # Nombre de paires fortement correlees
            high_corr_count = sum(1 for c in correlations if c > self.config['high_correlation_threshold'])
            
            # Score de diversification (0-100, 100 = tres diversifie)
            diversification_score = max(0, 100 * (1 - avg_corr))
            
            # Avertissement si correlation elevee
            warning = None
            if avg_corr > 0.6:
                warning = "Portfolio fortement correle - risque de concentration eleve"
            elif avg_corr > 0.4:
                warning = "Correlation moderee - envisager plus de diversification"
            
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
        """Met  jour les statistiques"""
        if self.correlation_matrix is None:
            return
        
        n = len(self.correlation_matrix)
        self.stats['total_pairs'] = n * (n - 1) // 2
        
        # Compter les paires fortement correlees
        high_corr = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.correlation_matrix.iloc[i, j]) > self.config['high_correlation_threshold']:
                    high_corr += 1
        
        self.stats['high_correlation_pairs'] = high_corr
    
    def _calculate_avg_correlation(self, symbols: List[str]) -> float:
        """
        Calcule la correlation moyenne entre symboles
        
        Args:
            symbols: Liste des symboles
            
        Returns:
            Correlation moyenne
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
        Verifie si une mise  jour est necessaire
        
        Returns:
            True si mise  jour necessaire
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
    
    # Donnees de test
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    
    # Creer des prix correles
    base_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    symbols_data = {
        'BTCUSDC': pd.Series(base_prices, index=dates),
        'ETHUSDC': pd.Series(base_prices + np.random.randn(200) * 2, index=dates),  # Correle
        'BNBUSDC': pd.Series(base_prices * 0.5 + np.random.randn(200) * 3, index=dates),  # Moyennement correle
        'ADAUSDC': pd.Series(100 + np.cumsum(np.random.randn(200) * 0.3), index=dates),  # Independant
        'DOGEUSDC': pd.Series(50 + np.cumsum(np.random.randn(200) * 0.2), index=dates)  # Independant
    }
    
    analyzer = CorrelationAnalyzer()
    
    print("Test Correlation Analyzer")
    print("=" * 50)
    
    # Ajouter les donnees
    for symbol, prices in symbols_data.items():
        analyzer.add_symbol_data(symbol, prices)
    
    # Calculer les correlations
    analyzer.update_correlations()
    
    # Tester les fonctions
    print("\n1. Paires fortement correlees:")
    pairs = analyzer.get_highly_correlated_pairs(0.5)
    for s1, s2, corr in pairs[:5]:
        print(f"   {s1} <-> {s2}: {corr:.2f}")
    
    print("\n2. Correlations de BTCUSDC:")
    btc_corrs = analyzer.get_symbol_correlations('BTCUSDC')
    for symbol, corr in list(btc_corrs.items())[:3]:
        print(f"   {symbol}: {corr:.2f}")
    
    print("\n3. Portfolio diversifie (3 symboles):")
    diversified = analyzer.find_diversified_symbols(3)
    print(f"   {', '.join(diversified)}")
    
    print("\n4. Analyse du portfolio:")
    analysis = analyzer.check_portfolio_correlation(['BTCUSDC', 'ETHUSDC', 'ADAUSDC'])
    print(f"   Correlation moyenne: {analysis['avg_correlation']:.2f}")
    print(f"   Score de diversification: {analysis['diversification_score']:.1f}")
    if analysis['warning']:
        print(f"     {analysis['warning']}")
    
    print("\n5. Statistiques:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
