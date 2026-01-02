"""
Performance Optimizer pour The Bot
Optimisations Python pour maximiser les performances sur PC classique
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, Callable, Any, Optional
from functools import wraps
import gc

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimiseur de performance pour PC classique
    
    Optimisations:
    - Numpy et Pandas optimises
    - Numba JIT compilation
    - Vectorisation
    - Cache et memoization
    - Gestion memoire efficace
    """
    
    def __init__(self, config: Dict):
        """
        Initialise l'optimiseur
        
        Args:
            config: Configuration
        """
        self.config = config
        self.enabled = getattr(config, 'ENABLED', True)
        
        # Statistiques
        self.stats = {
            'optimizations_applied': 0,
            'time_saved_ms': 0,
            'functions_optimized': []
        }
        
        # Appliquer les optimisations globales
        if self.enabled:
            self._apply_global_optimizations()
        
        logger.info(f"Performance Optimizer initialise (enabled: {self.enabled})")
    
    def _apply_global_optimizations(self):
        """Applique les optimisations globales"""
        logger.info(" Application des optimisations globales...")
        
        # Optimiser Numpy
        self.optimize_numpy()
        
        # Optimiser Pandas
        self.optimize_pandas()
        
        # Configurer le GC
        self.optimize_gc()
        
        logger.info("""| Optimisations globales appliquees")
    
    def optimize_numpy(self):
        """Optimise Numpy"""
        try:
            # Desactiver les warnings Numpy (performance)
            np.seterr(all='ignore')
            
            # Configuration pour performance
            np.set_printoptions(precision=4, suppress=True)
            
            self.stats['optimizations_applied'] += 1
            logger.info("""| Numpy optimise")
            
        except Exception as e:
            logger.error(f"Erreur optimisation Numpy: {e}")
    
    def optimize_pandas(self):
        """Optimise Pandas"""
        try:
            # Desactiver les warnings de chanage
            pd.options.mode.chained_assignment = None
            
            # Activer numexpr pour calculs vectorises (2-3x plus rapide)
            pd.options.compute.use_numexpr = True
            
            # Activer bottleneck pour reductions (sommes, moyennes, etc.)
            pd.options.compute.use_bottleneck = True
            
            # Optimiser l'affichage
            pd.options.display.precision = 4
            pd.options.display.max_rows = 100
            pd.options.display.max_columns = 20
            
            self.stats['optimizations_applied'] += 1
            logger.info("""| Pandas optimise")
            
        except Exception as e:
            logger.error(f"Erreur optimisation Pandas: {e}")
    
    def optimize_gc(self):
        """Optimise le garbage collector"""
        try:
            # Ajuster les seuils du GC pour moins d'interruptions
            # Valeurs pour PC classique avec 16GB RAM
            gc.set_threshold(700, 10, 10)
            
            # S'assurer que le GC est active
            gc.enable()
            
            self.stats['optimizations_applied'] += 1
            logger.info("""| Garbage Collector optimise")
            
        except Exception as e:
            logger.error(f"Erreur optimisation GC: {e}")
    
    # ========================================================================
    # D"CORATEURS D'OPTIMISATION
    # ========================================================================
    
    @staticmethod
    def vectorize(func: Callable) -> Callable:
        """
        Decorateur pour vectoriser une fonction Numpy
        
        Usage:
            @PerformanceOptimizer.vectorize
            def my_function(x, y):
                return x + y
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convertir en arrays Numpy si necessaire
            args_array = []
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    args_array.append(np.array(arg))
                else:
                    args_array.append(arg)
            
            return func(*args_array, **kwargs)
        
        return wrapper
    
    @staticmethod
    def use_numba(nopython: bool = True, cache: bool = True):
        """
        Decorateur pour compiler avec Numba JIT
        
        Args:
            nopython: Mode nopython (plus rapide mais plus restrictif)
            cache: Cache la compilation
            
        Usage:
            @PerformanceOptimizer.use_numba()
            def fast_calculation(prices):
                result = 0
                for i in range(len(prices)):
                    result += prices[i] * 2
                return result
        """
        def decorator(func: Callable) -> Callable:
            try:
                from numba import jit
                return jit(nopython=nopython, cache=cache)(func)
            except ImportError:
                logger.warning("Numba non disponible, fonction non optimisee")
                return func
        
        return decorator
    
    # ========================================================================
    # FONCTIONS OPTIMIS"ES COURANTES
    # ========================================================================
    
    @staticmethod
    def fast_mean(arr: np.ndarray) -> float:
        """
        Calcul rapide de la moyenne
        
        Args:
            arr: Array numpy
            
        Returns:
            Moyenne
        """
        return np.mean(arr)
    
    @staticmethod
    def fast_std(arr: np.ndarray) -> float:
        """
        Calcul rapide de l'ecart-type
        
        Args:
            arr: Array numpy
            
        Returns:
            "cart-type
        """
        return np.std(arr, ddof=1)
    
    @staticmethod
    def fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """
        Moyenne mobile rapide avec Numpy
        
        Args:
            arr: Array des valeurs
            window: Taille de la fenetre
            
        Returns:
            Array des moyennes mobiles
        """
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        
        # Utiliser convolution pour performance
        weights = np.ones(window) / window
        result = np.convolve(arr, weights, mode='valid')
        
        # Padding avec NaN au debut
        padding = np.full(window - 1, np.nan)
        return np.concatenate([padding, result])
    
    @staticmethod
    def fast_ema(arr: np.ndarray, period: int) -> np.ndarray:
        """
        EMA rapide avec Numpy
        
        Args:
            arr: Array des valeurs
            period: Periode
            
        Returns:
            Array des EMA
        """
        alpha = 2 / (period + 1)
        result = np.empty_like(arr)
        result[0] = arr[0]
        
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Correlation rapide
        
        Args:
            x: Premier array
            y: Deuxieme array
            
        Returns:
            Coefficient de correlation
        """
        return np.corrcoef(x, y)[0, 1]
    
    @staticmethod
    def batch_process_dataframe(df: pd.DataFrame, 
                               func: Callable, 
                               batch_size: int = 1000) -> pd.DataFrame:
        """
        Traite un DataFrame par batch pour economiser la memoire
        
        Args:
            df: DataFrame  traiter
            func: Fonction  appliquer
            batch_size: Taille des batches
            
        Returns:
            DataFrame traite
        """
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            result = func(batch)
            results.append(result)
            
            # GC periodique
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        return pd.concat(results, ignore_index=True)
    
    # ========================================================================
    # PROFILING ET MESURES
    # ========================================================================
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile une fonction
        
        Args:
            func: Fonction  profiler
            *args, **kwargs: Arguments de la fonction
            
        Returns:
            Dict avec resultats et metriques
        """
        import psutil
        process = psutil.Process()
        
        # Mesures avant
        mem_before = process.memory_info().rss / 1024 / 1024
        time_before = time.time()
        
        # Execution
        result = func(*args, **kwargs)
        
        # Mesures apres
        time_after = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        # Metriques
        elapsed_ms = (time_after - time_before) * 1000
        mem_delta = mem_after - mem_before
        
        profile_result = {
            'result': result,
            'elapsed_ms': elapsed_ms,
            'memory_delta_mb': mem_delta,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after
        }
        
        logger.debug(
            f"Profile {func.__name__}: "
            f"{elapsed_ms:.2f}ms, "
            f"mem: {mem_delta:+.1f}MB"
        )
        
        return profile_result
    
    def compare_implementations(self, 
                               implementations: Dict[str, Callable],
                               *args, **kwargs) -> Dict[str, Dict]:
        """
        Compare plusieurs implementations d'une fonction
        
        Args:
            implementations: Dict {nom: fonction}
            *args, **kwargs: Arguments  passer
            
        Returns:
            Dict avec comparaisons
        """
        results = {}
        
        logger.info(f"" Comparaison de {len(implementations)} implementations...")
        
        for name, func in implementations.items():
            profile = self.profile_function(func, *args, **kwargs)
            results[name] = profile
            
            logger.info(
                f"  {name}: {profile['elapsed_ms']:.2f}ms, "
                f"mem: {profile['memory_delta_mb']:+.1f}MB"
            )
        
        # Trouver la plus rapide
        fastest = min(results.items(), key=lambda x: x[1]['elapsed_ms'])
        logger.info(f"""| Plus rapide: {fastest[0]}")
        
        return results
    
    # ========================================================================
    # RECOMMANDATIONS
    # ========================================================================
    
    def analyze_dataframe_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse l'efficacite d'un DataFrame
        
        Args:
            df: DataFrame  analyser
            
        Returns:
            Dict avec recommandations
        """
        analysis = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'null_counts': df.isnull().sum().sum(),
            'recommendations': []
        }
        
        # Recommandations
        
        # 1. Colonnes object
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            analysis['recommendations'].append(
                f"Convertir {len(object_cols)} colonnes 'object' en category pour economiser la memoire"
            )
        
        # 2. Float64 -> Float32
        float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
        if float64_cols:
            analysis['recommendations'].append(
                f"Convertir {len(float64_cols)} colonnes float64 en float32 (economie ~50%)"
            )
        
        # 3. Int64 -> Int32
        int64_cols = df.select_dtypes(include=['int64']).columns.tolist()
        if int64_cols:
            analysis['recommendations'].append(
                f"Convertir {len(int64_cols)} colonnes int64 en int32 si les valeurs le permettent"
            )
        
        # 4. Valeurs nulles
        if analysis['null_counts'] > len(df) * 0.1:
            analysis['recommendations'].append(
                f"{analysis['null_counts']} valeurs nulles detectees, envisager fillna() ou dropna()"
            )
        
        return analysis
    
    def optimize_dataframe(self, df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Optimise un DataFrame automatiquement
        
        Args:
            df: DataFrame  optimiser
            aggressive: Optimisation agressive (peut perdre en precision)
            
        Returns:
            DataFrame optimise
        """
        df_optimized = df.copy()
        
        # Convertir float64 -> float32
        float_cols = df_optimized.select_dtypes(include=['float64']).columns
        df_optimized[float_cols] = df_optimized[float_cols].astype('float32')
        
        # Convertir int64 -> int32 si possible
        if aggressive:
            int_cols = df_optimized.select_dtypes(include=['int64']).columns
            for col in int_cols:
                if df_optimized[col].max() < 2147483647:  # Max int32
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Convertir object -> category si peu de valeurs uniques
        object_cols = df_optimized.select_dtypes(include=['object']).columns
        for col in object_cols:
            num_unique = df_optimized[col].nunique()
            if num_unique / len(df_optimized) < 0.5:  # Moins de 50% de valeurs uniques
                df_optimized[col] = df_optimized[col].astype('category')
        
        # Log economie
        mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024
        mem_after = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
        savings = mem_before - mem_after
        savings_pct = (savings / mem_before) * 100
        
        logger.info(
            f"""| DataFrame optimise: {mem_before:.1f}MB -> {mem_after:.1f}MB "
            f"(economie: {savings:.1f}MB, {savings_pct:.1f}%)"
        )
        
        return df_optimized
    
    # ========================================================================
    # TIPS ET BEST PRACTICES
    # ========================================================================
    
    @staticmethod
    def get_optimization_tips() -> List[str]:
        """Retourne des tips d'optimisation"""
        return [
            """| Utilisez numpy pour calculs vectorises (100x plus rapide que boucles Python)",
            """| Preferez pandas.apply() avec engine='numba' pour DataFrames",
            """| Utilisez @numba.jit pour fonctions avec boucles intensives",
            """| "vitez les boucles Python, preferez la vectorisation",
            """| Utilisez dtype appropries (float32 au lieu de float64 si possible)",
            """| Liberez la memoire avec gc.collect() apres gros calculs",
            """| Utilisez pandas.eval() pour expressions complexes",
            """| Chargez seulement les colonnes necessaires des CSV",
            """| Utilisez category dtype pour colonnes avec peu de valeurs uniques",
            """| "vitez .iterrows(), utilisez .itertuples() ou mieux: vectorisation"
        ]
    
    def print_optimization_report(self):
        """Affiche un rapport d'optimisation"""
        logger.info("\n" + "=" * 60)
        logger.info("RAPPORT D'OPTIMISATION")
        logger.info("=" * 60)
        logger.info(f"Optimisations appliquees: {self.stats['optimizations_applied']}")
        logger.info(f"Temps economise: {self.stats['time_saved_ms']:.0f}ms")
        logger.info(f"Fonctions optimisees: {len(self.stats['functions_optimized'])}")
        
        if self.stats['functions_optimized']:
            logger.info("\nFonctions optimisees:")
            for func_name in self.stats['functions_optimized']:
                logger.info(f"  " {func_name}")
        
        logger.info("\nTips d'optimisation:")
        for tip in self.get_optimization_tips():
            logger.info(f"  {tip}")
        
        logger.info("=" * 60 + "\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return {
            'enabled': self.enabled,
            'optimizations_applied': self.stats['optimizations_applied'],
            'time_saved_ms': self.stats['time_saved_ms'],
            'functions_optimized': self.stats['functions_optimized']
        }


# ============================================================================
# FONCTIONS UTILITAIRES NUMBA (si disponible)
# ============================================================================

try:
    from numba import jit
    
    @jit(nopython=True, cache=True)
    def fast_sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """
        SMA ultra-rapide avec Numba
        
        Args:
            prices: Array des prix
            period: Periode
            
        Returns:
            Array des SMA
        """
        n = len(prices)
        sma = np.empty(n)
        sma[:period-1] = np.nan
        
        for i in range(period - 1, n):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
    
    @jit(nopython=True, cache=True)
    def fast_returns_numba(prices: np.ndarray) -> np.ndarray:
        """
        Calcul rapide des returns avec Numba
        
        Args:
            prices: Array des prix
            
        Returns:
            Array des returns
        """
        n = len(prices)
        returns = np.empty(n)
        returns[0] = 0.0
        
        for i in range(1, n):
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        
        return returns
    
    logger.info("""| Fonctions Numba disponibles")
    
    # except ImportError:
    logger.warning(" Numba non disponible, fonctions standard seront utilisees")
    
    def fast_sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Fallback sans Numba"""
        return PerformanceOptimizer.fast_rolling_mean(prices, period)
    
    def fast_returns_numba(prices: np.ndarray) -> np.ndarray:
        """Fallback sans Numba"""
        return np.diff(prices) / prices[:-1]
