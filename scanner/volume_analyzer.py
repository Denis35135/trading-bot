"""
Volume Analyzer
Analyse le volume des trades pour dÃƒÂ©tecter l'intÃƒÂ©rÃƒÂªt institutionnel et retail
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VolumeAnalyzer:
    """
    Analyseur de volume avancÃƒÂ©
    
    Analyses:
    - Volume Profile (support/resistance basÃƒÂ©s sur volume)
    - DÃƒÂ©tection de volumes anormaux (spikes)
    - Volume Trend (accumulation/distribution)
    - Buy vs Sell pressure
    - Large trades detection
    - Volume Weighted Average Price (VWAP)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur
        
        Args:
            config: Configuration
        """
        default_config = {
            'spike_threshold': 2.0,  # 2x volume moyen = spike
            'large_trade_threshold': 3.0,  # 3x volume moyen
            'volume_ma_period': 20,
            'vwap_period': 20,
            'min_volume_24h': 1_000_000  # 1M$ minimum
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
        self.volume_data = {}  # {symbol: metrics}
        
        logger.info("Ã°Å¸â€œÅ  Volume Analyzer initialisÃƒÂ©")
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Analyse complÃƒÂ¨te du volume d'un symbole
        
        Args:
            symbol: Le symbole
            df: DataFrame OHLCV
            
        Returns:
            Dict avec mÃƒÂ©triques de volume
        """
        if len(df) < self.config['volume_ma_period']:
            logger.warning(f"Pas assez de donnÃƒÂ©es pour {symbol}")
            return {}
        
        try:
            metrics = {}
            
            # 1. Volume moyen et actuel
            volume_ma = df['volume'].rolling(self.config['volume_ma_period']).mean()
            metrics['volume_ma'] = volume_ma.iloc[-1]
            metrics['volume_current'] = df['volume'].iloc[-1]
            metrics['volume_ratio'] = metrics['volume_current'] / metrics['volume_ma'] if metrics['volume_ma'] > 0 else 0
            
            # 2. Tendance du volume
            volume_trend = self._calculate_volume_trend(df)
            metrics['volume_trend'] = volume_trend
            
            # 3. Spikes de volume
            spikes = self._detect_volume_spikes(df, volume_ma)
            metrics['spike_count_recent'] = len([s for s in spikes if s['index'] >= len(df) - 20])
            metrics['has_recent_spike'] = metrics['spike_count_recent'] > 0
            
            # 4. VWAP
            vwap = self._calculate_vwap(df)
            metrics['vwap'] = vwap.iloc[-1] if len(vwap) > 0 else 0
            metrics['price_vs_vwap'] = (df['close'].iloc[-1] - metrics['vwap']) / metrics['vwap'] if metrics['vwap'] > 0 else 0
            
            # 5. Volume Profile (prix avec le plus de volume)
            profile = self._calculate_volume_profile(df)
            metrics['volume_profile'] = profile
            
            # 6. Pression achat/vente (si disponible)
            if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
                metrics['buy_sell_ratio'] = df['buy_volume'].sum() / df['sell_volume'].sum() if df['sell_volume'].sum() > 0 else 1
            else:
                # Estimation basÃƒÂ©e sur le prix
                up_volume = df[df['close'] > df['open']]['volume'].sum()
                down_volume = df[df['close'] <= df['open']]['volume'].sum()
                metrics['buy_sell_ratio'] = up_volume / down_volume if down_volume > 0 else 1
            
            # 7. Large trades
            large_trades = self._detect_large_trades(df, volume_ma)
            metrics['large_trades_count'] = len(large_trades)
            
            # 8. Volume momentum
            vol_momentum = self._calculate_volume_momentum(df)
            metrics['volume_momentum'] = vol_momentum
            
            # 9. Score de volume (0-100)
            metrics['volume_score'] = self._calculate_volume_score(metrics)
            
            # 10. Distribution/Accumulation
            distribution = self._detect_distribution_accumulation(df)
            metrics['distribution_phase'] = distribution
            
            self.volume_data[symbol] = metrics
            
            logger.debug(f"{symbol}: Vol Ratio={metrics['volume_ratio']:.2f}, Trend={volume_trend}, Score={metrics['volume_score']:.0f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur analyse volume {symbol}: {e}")
            return {}
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> str:
        """
        DÃƒÂ©termine la tendance du volume (increasing/decreasing/stable)
        
        Args:
            df: DataFrame
            
        Returns:
            Tendance
        """
        vol_short = df['volume'].tail(10).mean()
        vol_long = df['volume'].tail(30).mean()
        
        if vol_short > vol_long * 1.2:
            return 'increasing'
        elif vol_short < vol_long * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_volume_spikes(self, df: pd.DataFrame, volume_ma: pd.Series) -> List[Dict]:
        """
        DÃƒÂ©tecte les spikes de volume
        
        Args:
            df: DataFrame
            volume_ma: Volume moyen
            
        Returns:
            Liste des spikes
        """
        spikes = []
        
        for i in range(len(df)):
            if volume_ma.iloc[i] > 0:
                ratio = df['volume'].iloc[i] / volume_ma.iloc[i]
                
                if ratio > self.config['spike_threshold']:
                    spikes.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'volume': df['volume'].iloc[i],
                        'ratio': ratio,
                        'price': df['close'].iloc[i]
                    })
        
        return spikes
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule le Volume Weighted Average Price
        
        Args:
            df: DataFrame
            
        Returns:
            Series VWAP
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(self.config['vwap_period']).sum() / \
               df['volume'].rolling(self.config['vwap_period']).sum()
        
        return vwap
    
    def _calculate_volume_profile(self, df: pd.DataFrame, n_bins: int = 10) -> Dict:
        """
        Calcule le volume profile (distribution du volume par niveau de prix)
        
        Args:
            df: DataFrame
            n_bins: Nombre de bins de prix
            
        Returns:
            Dict avec POC (Point of Control) et VAH/VAL
        """
        try:
            # DÃƒÂ©finir les bins de prix
            price_range = df['close'].max() - df['close'].min()
            bin_size = price_range / n_bins
            
            # Calculer le volume par bin
            bins = {}
            for i in range(len(df)):
                price = df['close'].iloc[i]
                bin_idx = int((price - df['close'].min()) / bin_size) if bin_size > 0 else 0
                bin_idx = min(bin_idx, n_bins - 1)  # Cap au dernier bin
                
                if bin_idx not in bins:
                    bins[bin_idx] = 0
                bins[bin_idx] += df['volume'].iloc[i]
            
            # POC = bin avec le plus de volume
            poc_bin = max(bins.items(), key=lambda x: x[1])[0] if bins else 0
            poc_price = df['close'].min() + (poc_bin * bin_size) + (bin_size / 2)
            
            # VAH/VAL = Value Area High/Low (70% du volume)
            total_volume = sum(bins.values())
            target_volume = total_volume * 0.7
            
            # Trier les bins par volume dÃƒÂ©croissant
            sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
            
            cumulative = 0
            value_area_bins = []
            for bin_idx, vol in sorted_bins:
                value_area_bins.append(bin_idx)
                cumulative += vol
                if cumulative >= target_volume:
                    break
            
            vah = df['close'].min() + (max(value_area_bins) * bin_size) if value_area_bins else poc_price
            val = df['close'].min() + (min(value_area_bins) * bin_size) if value_area_bins else poc_price
            
            return {
                'poc': poc_price,  # Point of Control
                'vah': vah,  # Value Area High
                'val': val,  # Value Area Low
                'current_vs_poc': (df['close'].iloc[-1] - poc_price) / poc_price if poc_price > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul volume profile: {e}")
            return {'poc': 0, 'vah': 0, 'val': 0, 'current_vs_poc': 0}
    
    def _detect_large_trades(self, df: pd.DataFrame, volume_ma: pd.Series) -> List[Dict]:
        """
        DÃƒÂ©tecte les large trades (volume institutionnel)
        
        Args:
            df: DataFrame
            volume_ma: Volume moyen
            
        Returns:
            Liste des large trades
        """
        large_trades = []
        
        for i in range(len(df)):
            if volume_ma.iloc[i] > 0:
                ratio = df['volume'].iloc[i] / volume_ma.iloc[i]
                
                if ratio > self.config['large_trade_threshold']:
                    large_trades.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'volume': df['volume'].iloc[i],
                        'ratio': ratio,
                        'direction': 'buy' if df['close'].iloc[i] > df['open'].iloc[i] else 'sell'
                    })
        
        return large_trades
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> float:
        """
        Calcule le momentum du volume
        
        Args:
            df: DataFrame
            
        Returns:
            Momentum (-1 ÃƒÂ  1)
        """
        vol_10 = df['volume'].tail(10).mean()
        vol_20 = df['volume'].tail(20).mean()
        
        if vol_20 == 0:
            return 0
        
        momentum = (vol_10 - vol_20) / vol_20
        
        # Normaliser entre -1 et 1
        return max(-1, min(1, momentum))
    
    def _detect_distribution_accumulation(self, df: pd.DataFrame) -> str:
        """
        DÃƒÂ©tecte si on est en phase de distribution ou accumulation
        
        Args:
            df: DataFrame
            
        Returns:
            Phase (accumulation/distribution/neutral)
        """
        recent_df = df.tail(20)
        
        # Volume moyen sur hausses vs baisses
        up_days = recent_df[recent_df['close'] > recent_df['open']]
        down_days = recent_df[recent_df['close'] <= recent_df['open']]
        
        up_vol = up_days['volume'].mean() if len(up_days) > 0 else 0
        down_vol = down_days['volume'].mean() if len(down_days) > 0 else 0
        
        if up_vol > down_vol * 1.3:
            return 'accumulation'
        elif down_vol > up_vol * 1.3:
            return 'distribution'
        else:
            return 'neutral'
    
    def _calculate_volume_score(self, metrics: Dict) -> float:
        """
        Calcule un score de qualitÃƒÂ© du volume (0-100)
        
        Args:
            metrics: MÃƒÂ©triques calculÃƒÂ©es
            
        Returns:
            Score
        """
        score = 0.0
        
        # 1. Ratio de volume actuel (0-30)
        vol_ratio = metrics.get('volume_ratio', 0)
        if 1.2 < vol_ratio < 3:  # Sweet spot
            score += 30
        elif 0.8 < vol_ratio < 5:
            score += 20
        else:
            score += 10
        
        # 2. Tendance du volume (0-25)
        trend = metrics.get('volume_trend', 'stable')
        if trend == 'increasing':
            score += 25
        elif trend == 'stable':
            score += 15
        else:
            score += 5
        
        # 3. Buy/Sell ratio (0-25)
        bs_ratio = metrics.get('buy_sell_ratio', 1)
        if 0.8 < bs_ratio < 1.2:  # Ãƒâ€°quilibrÃƒÂ©
            score += 15
        elif bs_ratio > 1.2:  # Acheteurs dominants
            score += 25
        else:
            score += 10
        
        # 4. Volume momentum (0-20)
        momentum = metrics.get('volume_momentum', 0)
        if momentum > 0.2:
            score += 20
        elif momentum > 0:
            score += 15
        else:
            score += 5
        
        return min(score, 100)
    
    def get_volume_summary(self, symbol: str) -> Dict:
        """
        Retourne un rÃƒÂ©sumÃƒÂ© du volume pour un symbole
        
        Args:
            symbol: Le symbole
            
        Returns:
            Dict avec rÃƒÂ©sumÃƒÂ©
        """
        if symbol not in self.volume_data:
            return {'error': 'Symbole non analysÃƒÂ©'}
        
        metrics = self.volume_data[symbol]
        
        return {
            'symbol': symbol,
            'volume_ratio': metrics.get('volume_ratio', 0),
            'trend': metrics.get('volume_trend', 'unknown'),
            'score': metrics.get('volume_score', 0),
            'buy_sell_ratio': metrics.get('buy_sell_ratio', 1),
            'has_spike': metrics.get('has_recent_spike', False),
            'phase': metrics.get('distribution_phase', 'unknown'),
            'vwap_position': 'above' if metrics.get('price_vs_vwap', 0) > 0 else 'below'
        }
    
    def compare_volumes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Compare les volumes de plusieurs symboles
        
        Args:
            symbols: Liste des symboles
            
        Returns:
            DataFrame comparatif
        """
        data = []
        
        for symbol in symbols:
            if symbol in self.volume_data:
                metrics = self.volume_data[symbol]
                data.append({
                    'symbol': symbol,
                    'vol_ratio': metrics.get('volume_ratio', 0),
                    'trend': metrics.get('volume_trend', 'unknown'),
                    'score': metrics.get('volume_score', 0),
                    'phase': metrics.get('distribution_phase', 'unknown')
                })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            df = df.sort_values('score', ascending=False)
        
        return df
    
    def get_high_volume_symbols(self, min_ratio: float = 1.5) -> List[str]:
        """
        Retourne les symboles avec volume ÃƒÂ©levÃƒÂ©
        
        Args:
            min_ratio: Ratio minimum
            
        Returns:
            Liste de symboles
        """
        high_vol = []
        
        for symbol, metrics in self.volume_data.items():
            if metrics.get('volume_ratio', 0) >= min_ratio:
                high_vol.append(symbol)
        
        return high_vol
    
    def get_accumulation_symbols(self) -> List[str]:
        """
        Retourne les symboles en phase d'accumulation
        
        Returns:
            Liste de symboles
        """
        accumulation = []
        
        for symbol, metrics in self.volume_data.items():
            if metrics.get('distribution_phase') == 'accumulation':
                accumulation.append(symbol)
        
        return accumulation
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques globales
        
        Returns:
            Dict avec stats
        """
        if not self.volume_data:
            return {}
        
        trends = [m.get('volume_trend', 'unknown') for m in self.volume_data.values()]
        phases = [m.get('distribution_phase', 'unknown') for m in self.volume_data.values()]
        scores = [m.get('volume_score', 0) for m in self.volume_data.values()]
        
        return {
            'total_symbols': len(self.volume_data),
            'increasing_trend': trends.count('increasing'),
            'decreasing_trend': trends.count('decreasing'),
            'stable_trend': trends.count('stable'),
            'accumulation_phase': phases.count('accumulation'),
            'distribution_phase': phases.count('distribution'),
            'avg_score': np.mean(scores) if scores else 0
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Volume Analyzer"""
    
    # DonnÃƒÂ©es de test
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    # CrÃƒÂ©er des donnÃƒÂ©es avec diffÃƒÂ©rents patterns de volume
    test_data = {
        'HIGH_VOL': pd.DataFrame({
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
            'volume': np.random.randint(8000, 15000, 100)  # Volume ÃƒÂ©levÃƒÂ©
        }, index=dates),
        'LOW_VOL': pd.DataFrame({
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
            'volume': np.random.randint(1000, 3000, 100)  # Volume faible
        }, index=dates),
        'SPIKE': pd.DataFrame({
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
            'volume': np.concatenate([
                np.random.randint(3000, 5000, 95),
                np.random.randint(15000, 20000, 5)  # Spike ÃƒÂ  la fin
            ])
        }, index=dates)
    }
    
    analyzer = VolumeAnalyzer()
    
    print("Test Volume Analyzer")
    print("=" * 50)
    
    # Analyser chaque symbole
    for symbol, df in test_data.items():
        print(f"\n{symbol}:")
        metrics = analyzer.analyze_symbol(symbol, df)
        
        print(f"  Volume ratio: {metrics.get('volume_ratio', 0):.2f}x")
        print(f"  Tendance: {metrics.get('volume_trend', 'unknown')}")
        print(f"  Buy/Sell ratio: {metrics.get('buy_sell_ratio', 0):.2f}")
        print(f"  Score: {metrics.get('volume_score', 0):.0f}")
        print(f"  Phase: {metrics.get('distribution_phase', 'unknown')}")
        print(f"  Recent spike: {'Oui' if metrics.get('has_recent_spike') else 'Non'}")
        
        # Volume profile
        profile = metrics.get('volume_profile', {})
        print(f"  POC: {profile.get('poc', 0):.2f}")
    
    # Comparaison
    print("\n" + "=" * 50)
    print("Comparaison:")
    comparison = analyzer.compare_volumes(list(test_data.keys()))
    print(comparison.to_string(index=False))
    
    # Stats
    print("\n" + "=" * 50)
    print("Statistiques:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
