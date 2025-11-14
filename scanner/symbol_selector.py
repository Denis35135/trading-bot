"""
Symbol Selector
SÃƒÂ©lectionne intelligemment les meilleurs symboles ÃƒÂ  trader
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SymbolSelector:
    """
    SÃƒÂ©lecteur intelligent de symboles
    
    CritÃƒÂ¨res de sÃƒÂ©lection:
    - Volume 24h
    - VolatilitÃƒÂ© optimale
    - Spread bid/ask
    - LiquiditÃƒÂ©
    - Tendance rÃƒÂ©cente
    - Performance historique
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le sÃƒÂ©lecteur
        
        Args:
            config: Configuration
        """
        default_config = {
            'min_volume_24h': 5_000_000,  # 5M$ minimum
            'max_spread_pct': 0.002,  # 0.2% max
            'min_volatility': 0.005,  # 0.5% min
            'max_volatility': 0.05,  # 5% max
            'min_trades_24h': 1000,
            'lookback_period': 100,
            'top_n': 20  # Garder les top 20
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
        self.symbol_scores = {}
        self.symbol_data = {}
        
        logger.info("Ã°Å¸Å½Â¯ Symbol Selector initialisÃƒÂ©")
    
    def add_symbol(self, symbol: str, data: Dict):
        """
        Ajoute un symbole avec ses donnÃƒÂ©es
        
        Args:
            symbol: Le symbole
            data: Dict avec donnÃƒÂ©es (df, ticker, etc.)
        """
        self.symbol_data[symbol] = data
    
    def calculate_scores(self) -> Dict[str, float]:
        """
        Calcule les scores pour tous les symboles
        
        Returns:
            Dict {symbol: score}
        """
        logger.info(f"Ã°Å¸â€œÅ  Calcul des scores pour {len(self.symbol_data)} symboles")
        
        scores = {}
        
        for symbol, data in self.symbol_data.items():
            try:
                score = self._calculate_symbol_score(symbol, data)
                if score > 0:
                    scores[symbol] = score
            except Exception as e:
                logger.error(f"Erreur calcul score {symbol}: {e}")
        
        self.symbol_scores = scores
        
        logger.info(f"Ã¢Å“â€¦ {len(scores)} symboles scorÃƒÂ©s")
        
        return scores
    
    def _calculate_symbol_score(self, symbol: str, data: Dict) -> float:
        """
        Calcule le score d'un symbole (0-100)
        
        Args:
            symbol: Le symbole
            data: DonnÃƒÂ©es du symbole
            
        Returns:
            Score (0-100)
        """
        score = 0.0
        
        # 1. Volume Score (0-30 points)
        volume_24h = data.get('volume_24h', 0)
        if volume_24h < self.config['min_volume_24h']:
            return 0.0  # DisqualifiÃƒÂ©
        
        # Score logarithmique du volume
        volume_score = min(30, 10 * np.log10(volume_24h / self.config['min_volume_24h']))
        score += volume_score
        
        # 2. Volatility Score (0-25 points)
        df = data.get('df')
        if df is not None and len(df) > 0:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # VolatilitÃƒÂ© optimale entre min et max
            if volatility < self.config['min_volatility']:
                vol_score = 0
            elif volatility > self.config['max_volatility']:
                vol_score = 10
            else:
                # Score optimal entre min et max
                vol_range = self.config['max_volatility'] - self.config['min_volatility']
                normalized = (volatility - self.config['min_volatility']) / vol_range
                vol_score = 25 * (1 - abs(0.5 - normalized) * 2)  # Max au milieu
            
            score += vol_score
        
        # 3. Spread Score (0-15 points)
        spread_pct = data.get('spread_pct', 0)
        if spread_pct > self.config['max_spread_pct']:
            spread_score = 0
        else:
            spread_score = 15 * (1 - spread_pct / self.config['max_spread_pct'])
        
        score += spread_score
        
        # 4. Liquidity Score (0-15 points)
        trades_24h = data.get('trades_24h', 0)
        if trades_24h < self.config['min_trades_24h']:
            liquidity_score = 0
        else:
            liquidity_score = min(15, 5 * np.log10(trades_24h / self.config['min_trades_24h']))
        
        score += liquidity_score
        
        # 5. Trend Score (0-15 points)
        if df is not None and len(df) >= 20:
            # Tendance rÃƒÂ©cente (20 derniÃƒÂ¨res pÃƒÂ©riodes)
            recent_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            # Bonus pour tendance positive modÃƒÂ©rÃƒÂ©e
            if 0 < recent_change < 0.1:  # Entre 0% et 10%
                trend_score = 15
            elif recent_change > 0:
                trend_score = 10
            else:
                trend_score = 5
            
            score += trend_score
        
        return min(score, 100)
    
    def get_top_symbols(self, n: int = None) -> List[str]:
        """
        Retourne les top N symboles
        
        Args:
            n: Nombre de symboles (utilise config si None)
            
        Returns:
            Liste des symboles triÃƒÂ©s par score
        """
        if not self.symbol_scores:
            logger.warning("Aucun score calculÃƒÂ©, appeler calculate_scores() d'abord")
            return []
        
        if n is None:
            n = self.config['top_n']
        
        # Trier par score dÃƒÂ©croissant
        sorted_symbols = sorted(
            self.symbol_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top = [symbol for symbol, score in sorted_symbols[:n]]
        
        logger.info(f"Ã°Å¸Ââ€  Top {len(top)} symboles sÃƒÂ©lectionnÃƒÂ©s")
        for i, (symbol, score) in enumerate(sorted_symbols[:5], 1):
            logger.info(f"   {i}. {symbol}: {score:.1f} points")
        
        return top
    
    def filter_by_criteria(self, criteria: Dict) -> List[str]:
        """
        Filtre les symboles selon des critÃƒÂ¨res personnalisÃƒÂ©s
        
        Args:
            criteria: Dict avec critÃƒÂ¨res (min_score, min_volume, etc.)
            
        Returns:
            Liste des symboles qualifiÃƒÂ©s
        """
        qualified = []
        
        min_score = criteria.get('min_score', 50)
        min_volume = criteria.get('min_volume', self.config['min_volume_24h'])
        max_spread = criteria.get('max_spread', self.config['max_spread_pct'])
        
        for symbol, score in self.symbol_scores.items():
            data = self.symbol_data.get(symbol, {})
            
            # VÃƒÂ©rifier les critÃƒÂ¨res
            if score < min_score:
                continue
            
            if data.get('volume_24h', 0) < min_volume:
                continue
            
            if data.get('spread_pct', 0) > max_spread:
                continue
            
            qualified.append(symbol)
        
        logger.info(f"Ã¢Å“â€¦ {len(qualified)} symboles qualifiÃƒÂ©s selon critÃƒÂ¨res")
        
        return qualified
    
    def get_symbols_by_category(self) -> Dict[str, List[str]]:
        """
        Groupe les symboles par catÃƒÂ©gorie (haute/moyenne/basse volatilitÃƒÂ©)
        
        Returns:
            Dict {catÃƒÂ©gorie: [symboles]}
        """
        categories = {
            'high_volatility': [],
            'medium_volatility': [],
            'low_volatility': []
        }
        
        for symbol, data in self.symbol_data.items():
            df = data.get('df')
            if df is None or len(df) < 20:
                continue
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            if volatility > 0.03:  # > 3%
                categories['high_volatility'].append(symbol)
            elif volatility > 0.01:  # 1-3%
                categories['medium_volatility'].append(symbol)
            else:  # < 1%
                categories['low_volatility'].append(symbol)
        
        logger.info("Ã°Å¸â€œÅ  Symboles par catÃƒÂ©gorie:")
        for cat, symbols in categories.items():
            logger.info(f"   {cat}: {len(symbols)}")
        
        return categories
    
    def recommend_for_strategy(self, strategy_type: str) -> List[str]:
        """
        Recommande des symboles pour un type de stratÃƒÂ©gie
        
        Args:
            strategy_type: Type (scalping, momentum, mean_reversion)
            
        Returns:
            Liste de symboles recommandÃƒÂ©s
        """
        if not self.symbol_scores:
            return []
        
        recommendations = []
        
        if strategy_type == 'scalping':
            # Scalping: haute liquiditÃƒÂ©, faible spread, volatilitÃƒÂ© modÃƒÂ©rÃƒÂ©e
            for symbol, data in self.symbol_data.items():
                if symbol not in self.symbol_scores:
                    continue
                
                volume = data.get('volume_24h', 0)
                spread = data.get('spread_pct', 1)
                df = data.get('df')
                
                if df is not None and len(df) >= 20:
                    vol = df['close'].pct_change().std()
                    
                    # CritÃƒÂ¨res scalping
                    if (volume > 10_000_000 and 
                        spread < 0.001 and 
                        0.005 < vol < 0.02):
                        recommendations.append(symbol)
        
        elif strategy_type == 'momentum':
            # Momentum: haute volatilitÃƒÂ©, volume ÃƒÂ©levÃƒÂ©, tendance forte
            for symbol, data in self.symbol_data.items():
                if symbol not in self.symbol_scores:
                    continue
                
                df = data.get('df')
                volume = data.get('volume_24h', 0)
                
                if df is not None and len(df) >= 50:
                    vol = df['close'].pct_change().std()
                    trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
                    
                    # CritÃƒÂ¨res momentum
                    if (volume > 5_000_000 and 
                        vol > 0.015 and 
                        abs(trend) > 0.03):
                        recommendations.append(symbol)
        
        elif strategy_type == 'mean_reversion':
            # Mean reversion: volatilitÃƒÂ© moyenne, range-bound
            for symbol, data in self.symbol_data.items():
                if symbol not in self.symbol_scores:
                    continue
                
                df = data.get('df')
                
                if df is not None and len(df) >= 50:
                    vol = df['close'].pct_change().std()
                    
                    # Prix dans un range (pas de forte tendance)
                    high_50 = df['high'].tail(50).max()
                    low_50 = df['low'].tail(50).min()
                    current = df['close'].iloc[-1]
                    position = (current - low_50) / (high_50 - low_50)
                    
                    # CritÃƒÂ¨res mean reversion
                    if (0.008 < vol < 0.025 and 
                        0.2 < position < 0.8):  # Pas aux extrÃƒÂªmes
                        recommendations.append(symbol)
        
        logger.info(f"Ã°Å¸â€™Â¡ {len(recommendations)} symboles recommandÃƒÂ©s pour {strategy_type}")
        
        return recommendations[:15]  # Max 15 recommandations
    
    def get_symbol_details(self, symbol: str) -> Dict:
        """
        Retourne les dÃƒÂ©tails d'un symbole
        
        Args:
            symbol: Le symbole
            
        Returns:
            Dict avec dÃƒÂ©tails
        """
        if symbol not in self.symbol_data:
            return {'error': 'Symbole non trouvÃƒÂ©'}
        
        data = self.symbol_data[symbol]
        score = self.symbol_scores.get(symbol, 0)
        
        details = {
            'symbol': symbol,
            'score': score,
            'volume_24h': data.get('volume_24h', 0),
            'spread_pct': data.get('spread_pct', 0),
            'trades_24h': data.get('trades_24h', 0)
        }
        
        # Ajouter stats du DataFrame si disponible
        df = data.get('df')
        if df is not None and len(df) > 0:
            returns = df['close'].pct_change().dropna()
            details['volatility'] = returns.std()
            details['price_change_24h'] = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            details['current_price'] = df['close'].iloc[-1]
        
        return details
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques
        
        Returns:
            Dict avec stats
        """
        if not self.symbol_scores:
            return {'total_symbols': 0}
        
        scores = list(self.symbol_scores.values())
        
        return {
            'total_symbols': len(self.symbol_scores),
            'avg_score': np.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'symbols_above_50': sum(1 for s in scores if s >= 50),
            'symbols_above_70': sum(1 for s in scores if s >= 70)
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Symbol Selector"""
    
    # DonnÃƒÂ©es de test
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    test_symbols = {
        'BTCUSDC': {
            'volume_24h': 50_000_000,
            'spread_pct': 0.0005,
            'trades_24h': 50000,
            'df': pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
                'high': 101 + np.cumsum(np.random.randn(100) * 0.5),
                'low': 99 + np.cumsum(np.random.randn(100) * 0.5)
            }, index=dates)
        },
        'ETHUSDC': {
            'volume_24h': 30_000_000,
            'spread_pct': 0.0008,
            'trades_24h': 30000,
            'df': pd.DataFrame({
                'close': 50 + np.cumsum(np.random.randn(100) * 0.3),
                'high': 51 + np.cumsum(np.random.randn(100) * 0.3),
                'low': 49 + np.cumsum(np.random.randn(100) * 0.3)
            }, index=dates)
        },
        'LOWVOL': {
            'volume_24h': 2_000_000,  # Trop faible
            'spread_pct': 0.003,
            'trades_24h': 500,
            'df': pd.DataFrame({
                'close': 10 + np.random.randn(100) * 0.01,
                'high': 10.1 + np.random.randn(100) * 0.01,
                'low': 9.9 + np.random.randn(100) * 0.01
            }, index=dates)
        }
    }
    
    selector = SymbolSelector()
    
    print("Test Symbol Selector")
    print("=" * 50)
    
    # Ajouter les symboles
    for symbol, data in test_symbols.items():
        selector.add_symbol(symbol, data)
    
    # Calculer les scores
    scores = selector.calculate_scores()
    
    print("\n1. Scores calculÃƒÂ©s:")
    for symbol, score in scores.items():
        print(f"   {symbol}: {score:.1f}")
    
    print("\n2. Top symboles:")
    top = selector.get_top_symbols(2)
    print(f"   {', '.join(top)}")
    
    print("\n3. Recommandations pour scalping:")
    scalping_symbols = selector.recommend_for_strategy('scalping')
    print(f"   {', '.join(scalping_symbols) if scalping_symbols else 'Aucun'}")
    
    print("\n4. CatÃƒÂ©gories:")
    categories = selector.get_symbols_by_category()
    for cat, symbols in categories.items():
        if symbols:
            print(f"   {cat}: {', '.join(symbols)}")
    
    print("\n5. Statistiques:")
    stats = selector.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
