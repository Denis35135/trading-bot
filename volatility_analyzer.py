"""
Volatility Analyzer
Analyse la volatilitÃƒÂ© des symboles pour optimiser le sizing et le timing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Analyseur de volatilitÃƒÂ© avancÃƒÂ©
    
    MÃƒÂ©thodes:
    - VolatilitÃƒÂ© historique (HV)
    - VolatilitÃƒÂ© intraday vs overnight
    - RÃƒÂ©gimes de volatilitÃƒÂ© (high/medium/low)
    - ATR et variations
    - PrÃƒÂ©diction de volatilitÃƒÂ©
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur
        
        Args:
            config: Configuration
        """
        default_config = {
            'hv_period': 20,  # PÃƒÂ©riode pour volatilitÃƒÂ© historique
            'atr_period': 14,  # PÃƒÂ©riode pour ATR
            'regime_lookback': 100,  # PÃƒÂ©riode pour dÃƒÂ©tecter rÃƒÂ©gimes
            'high_vol_threshold': 0.03,  # 3% daily vol
            'low_vol_threshold': 0.01   # 1% daily vol
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
        self.volatility_data = {}  # {symbol: metrics}
        
        logger.info("Ã°Å¸â€œÅ  Volatility Analyzer initialisÃƒÂ©")
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Analyse complÃƒÂ¨te de la volatilitÃƒÂ© d'un symbole
        
        Args:
            symbol: Le symbole
            df: DataFrame OHLCV
            
        Returns:
            Dict avec mÃƒÂ©triques de volatilitÃƒÂ©
        """
        if len(df) < self.config['regime_lookback']:
            logger.warning(f"Pas assez de donnÃƒÂ©es pour {symbol}: {len(df)} candles")
            return {}
        
        try:
            metrics = {}
            
            # 1. VolatilitÃƒÂ© historique (returns)
            returns = df['close'].pct_change().dropna()
            metrics['hv_current'] = returns.tail(self.config['hv_period']).std()
            metrics['hv_mean'] = returns.std()
            
            # 2. ATR (Average True Range)
            atr = self._calculate_atr(df)
            metrics['atr'] = atr.iloc[-1] if len(atr) > 0 else 0
            metrics['atr_pct'] = (metrics['atr'] / df['close'].iloc[-1]) if df['close'].iloc[-1] > 0 else 0
            
            # 3. RÃƒÂ©gime de volatilitÃƒÂ©
            regime = self._detect_volatility_regime(returns)
            metrics['regime'] = regime
            metrics['regime_numeric'] = {'low': 0, 'medium': 1, 'high': 2}[regime]
            
            # 4. VolatilitÃƒÂ© intraday
            intraday_range = (df['high'] - df['low']) / df['close']
            metrics['avg_intraday_range'] = intraday_range.tail(20).mean()
            
            # 5. Tendance de la volatilitÃƒÂ©
            hv_short = returns.tail(10).std()
            hv_long = returns.tail(30).std()
            metrics['vol_trend'] = 'increasing' if hv_short > hv_long * 1.2 else 'decreasing' if hv_short < hv_long * 0.8 else 'stable'
            
            # 6. Coefficient de variation (volatilitÃƒÂ© relative au prix)
            metrics['cv'] = metrics['hv_current'] / abs(returns.mean()) if returns.mean() != 0 else 0
            
            # 7. Score de volatilitÃƒÂ© (0-100)
            metrics['volatility_score'] = self._calculate_volatility_score(metrics)
            
            # 8. Percentile de volatilitÃƒÂ© actuelle
            all_hvs = returns.rolling(self.config['hv_period']).std().dropna()
            if len(all_hvs) > 0:
                metrics['vol_percentile'] = (all_hvs < metrics['hv_current']).sum() / len(all_hvs)
            else:
                metrics['vol_percentile'] = 0.5
            
            self.volatility_data[symbol] = metrics
            
            logger.debug(f"{symbol}: HV={metrics['hv_current']:.2%}, Regime={regime}, Score={metrics['volatility_score']:.0f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur analyse volatilitÃƒÂ© {symbol}: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule l'Average True Range
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            Series avec ATR
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = moyenne mobile du TR
        atr = tr.rolling(self.config['atr_period']).mean()
        
        return atr
    
    def _detect_volatility_regime(self, returns: pd.Series) -> str:
        """
        DÃƒÂ©tecte le rÃƒÂ©gime de volatilitÃƒÂ© actuel
        
        Args:
            returns: Series des returns
            
        Returns:
            RÃƒÂ©gime (low/medium/high)
        """
        # VolatilitÃƒÂ© rÃƒÂ©cente
        recent_vol = returns.tail(20).std()
        
        if recent_vol > self.config['high_vol_threshold']:
            return 'high'
        elif recent_vol < self.config['low_vol_threshold']:
            return 'low'
        else:
            return 'medium'
    
    def _calculate_volatility_score(self, metrics: Dict) -> float:
        """
        Calcule un score de volatilitÃƒÂ© (0-100)
        
        Args:
            metrics: MÃƒÂ©triques calculÃƒÂ©es
            
        Returns:
            Score 0-100
        """
        score = 0.0
        
        # Composante 1: Niveau de volatilitÃƒÂ© (0-40)
        hv = metrics.get('hv_current', 0)
        if 0.01 < hv < 0.03:  # Sweet spot
            score += 40
        elif 0.005 < hv < 0.05:
            score += 30
        else:
            score += 10
        
        # Composante 2: StabilitÃƒÂ© du rÃƒÂ©gime (0-30)
        regime = metrics.get('regime', 'medium')
        if regime == 'medium':
            score += 30
        elif regime == 'high':
            score += 20
        else:
            score += 15
        
        # Composante 3: Tendance de volatilitÃƒÂ© (0-30)
        trend = metrics.get('vol_trend', 'stable')
        if trend == 'stable':
            score += 30
        elif trend == 'decreasing':
            score += 20
        else:  # increasing
            score += 10
        
        return min(score, 100)
    
    def compare_volatilities(self, symbols: List[str]) -> pd.DataFrame:
        """
        Compare les volatilitÃƒÂ©s de plusieurs symboles
        
        Args:
            symbols: Liste des symboles
            
        Returns:
            DataFrame comparatif
        """
        data = []
        
        for symbol in symbols:
            if symbol in self.volatility_data:
                metrics = self.volatility_data[symbol]
                data.append({
                    'symbol': symbol,
                    'hv': metrics.get('hv_current', 0),
                    'atr_pct': metrics.get('atr_pct', 0),
                    'regime': metrics.get('regime', 'unknown'),
                    'score': metrics.get('volatility_score', 0)
                })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            df = df.sort_values('score', ascending=False)
        
        return df
    
    def get_optimal_position_size_multiplier(self, symbol: str) -> float:
        """
        SuggÃƒÂ¨re un multiplicateur de taille de position basÃƒÂ© sur la volatilitÃƒÂ©
        
        Args:
            symbol: Le symbole
            
        Returns:
            Multiplicateur (0.5 ÃƒÂ  1.5)
        """
        if symbol not in self.volatility_data:
            return 1.0
        
        metrics = self.volatility_data[symbol]
        regime = metrics.get('regime', 'medium')
        hv = metrics.get('hv_current', 0.02)
        
        # RÃƒÂ©duire la taille en haute volatilitÃƒÂ©, augmenter en basse
        if regime == 'high':
            multiplier = 0.5
        elif regime == 'low':
            multiplier = 1.3
        else:
            # Ajustement graduel dans le rÃƒÂ©gime medium
            if hv > 0.025:
                multiplier = 0.8
            elif hv < 0.015:
                multiplier = 1.2
            else:
                multiplier = 1.0
        
        return multiplier
    
    def get_optimal_stop_distance(self, symbol: str, default_pct: float = 0.02) -> float:
        """
        Calcule la distance optimale de stop loss basÃƒÂ©e sur la volatilitÃƒÂ©
        
        Args:
            symbol: Le symbole
            default_pct: Distance par dÃƒÂ©faut (2%)
            
        Returns:
            Distance de stop en %
        """
        if symbol not in self.volatility_data:
            return default_pct
        
        metrics = self.volatility_data[symbol]
        atr_pct = metrics.get('atr_pct', default_pct)
        
        # Stop = 2x ATR (minimum 1%, maximum 5%)
        stop_distance = min(max(atr_pct * 2, 0.01), 0.05)
        
        return stop_distance
    
    def predict_volatility_change(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        PrÃƒÂ©dit si la volatilitÃƒÂ© va augmenter ou diminuer (simple)
        
        Args:
            symbol: Le symbole
            df: DataFrame avec prix
            
        Returns:
            Dict avec prÃƒÂ©diction
        """
        try:
            if len(df) < 50:
                return {'prediction': 'unknown', 'confidence': 0}
            
            returns = df['close'].pct_change().dropna()
            
            # VolatilitÃƒÂ© historique sur diffÃƒÂ©rentes fenÃƒÂªtres
            vol_10 = returns.tail(10).std()
            vol_20 = returns.tail(20).std()
            vol_50 = returns.tail(50).std()
            
            # Tendance de la volatilitÃƒÂ©
            if vol_10 > vol_20 * 1.2 and vol_20 > vol_50 * 1.1:
                prediction = 'increasing'
                confidence = 0.7
            elif vol_10 < vol_20 * 0.8 and vol_20 < vol_50 * 0.9:
                prediction = 'decreasing'
                confidence = 0.7
            else:
                prediction = 'stable'
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'vol_10': vol_10,
                'vol_20': vol_20,
                'vol_50': vol_50
            }
            
        except Exception as e:
            logger.error(f"Erreur prÃƒÂ©diction volatilitÃƒÂ©: {e}")
            return {'prediction': 'unknown', 'confidence': 0}
    
    def get_volatility_percentiles(self, symbol: str) -> Dict:
        """
        Retourne les percentiles de volatilitÃƒÂ© historique
        
        Args:
            symbol: Le symbole
            
        Returns:
            Dict avec percentiles
        """
        if symbol not in self.volatility_data:
            return {}
        
        metrics = self.volatility_data[symbol]
        
        return {
            'current_percentile': metrics.get('vol_percentile', 0.5),
            'regime': metrics.get('regime', 'unknown'),
            'is_extreme': metrics.get('vol_percentile', 0.5) > 0.9 or metrics.get('vol_percentile', 0.5) < 0.1
        }
    
    def get_high_volatility_symbols(self, min_score: float = 60) -> List[str]:
        """
        Retourne les symboles ÃƒÂ  haute volatilitÃƒÂ©
        
        Args:
            min_score: Score minimum
            
        Returns:
            Liste de symboles
        """
        high_vol = []
        
        for symbol, metrics in self.volatility_data.items():
            if metrics.get('regime') == 'high' and metrics.get('volatility_score', 0) >= min_score:
                high_vol.append(symbol)
        
        return high_vol
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques globales
        
        Returns:
            Dict avec stats
        """
        if not self.volatility_data:
            return {}
        
        regimes = [m.get('regime', 'unknown') for m in self.volatility_data.values()]
        scores = [m.get('volatility_score', 0) for m in self.volatility_data.values()]
        
        return {
            'total_symbols': len(self.volatility_data),
            'high_vol_count': regimes.count('high'),
            'medium_vol_count': regimes.count('medium'),
            'low_vol_count': regimes.count('low'),
            'avg_score': np.mean(scores) if scores else 0,
            'avg_hv': np.mean([m.get('hv_current', 0) for m in self.volatility_data.values()])
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Volatility Analyzer"""
    
    # DonnÃƒÂ©es de test
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    
    # CrÃƒÂ©er des donnÃƒÂ©es avec diffÃƒÂ©rentes volatilitÃƒÂ©s
    test_data = {
        'HIGH_VOL': pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(200) * 2),
            'high': 102 + np.cumsum(np.random.randn(200) * 2),
            'low': 98 + np.cumsum(np.random.randn(200) * 2),
            'close': 100 + np.cumsum(np.random.randn(200) * 2),
        }, index=dates),
        'LOW_VOL': pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(200) * 0.2),
            'high': 100.5 + np.cumsum(np.random.randn(200) * 0.2),
            'low': 99.5 + np.cumsum(np.random.randn(200) * 0.2),
            'close': 100 + np.cumsum(np.random.randn(200) * 0.2),
        }, index=dates),
        'MED_VOL': pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(200) * 0.8),
            'high': 101 + np.cumsum(np.random.randn(200) * 0.8),
            'low': 99 + np.cumsum(np.random.randn(200) * 0.8),
            'close': 100 + np.cumsum(np.random.randn(200) * 0.8),
        }, index=dates)
    }
    
    analyzer = VolatilityAnalyzer()
    
    print("Test Volatility Analyzer")
    print("=" * 50)
    
    # Analyser chaque symbole
    for symbol, df in test_data.items():
        print(f"\n{symbol}:")
        metrics = analyzer.analyze_symbol(symbol, df)
        
        print(f"  HV: {metrics.get('hv_current', 0):.2%}")
        print(f"  ATR%: {metrics.get('atr_pct', 0):.2%}")
        print(f"  RÃƒÂ©gime: {metrics.get('regime', 'unknown')}")
        print(f"  Score: {metrics.get('volatility_score', 0):.0f}")
        print(f"  Tendance: {metrics.get('vol_trend', 'unknown')}")
        
        # Recommandations
        multiplier = analyzer.get_optimal_position_size_multiplier(symbol)
        stop_dist = analyzer.get_optimal_stop_distance(symbol)
        
        print(f"  Position multiplier: {multiplier:.2f}x")
        print(f"  Stop distance: {stop_dist:.2%}")
    
    # Comparaison
    print("\n" + "=" * 50)
    print("Comparaison:")
    comparison = analyzer.compare_volatilities(list(test_data.keys()))
    print(comparison.to_string(index=False))
    
    # Stats globales
    print("\n" + "=" * 50)
    print("Statistiques:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")