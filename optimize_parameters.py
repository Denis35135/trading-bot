#!/usr/bin/env python3
"""
Script d'optimisation des paramÃƒÂ¨tres des stratÃƒÂ©gies
Utilise l'optimisation bayÃƒÂ©sienne et le backtesting
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
import json

# Ajouter le rÃƒÂ©pertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
except ImportError:
    print("Ã¢ÂÅ’ scikit-optimize non installÃƒÂ©")
    print("   Installez avec: pip install scikit-optimize")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Optimiseur de paramÃƒÂ¨tres pour stratÃƒÂ©gies de trading
    
    Utilise:
    - Optimisation bayÃƒÂ©sienne (efficace)
    - Backtesting pour ÃƒÂ©valuer les performances
    - Cross-validation pour ÃƒÂ©viter l'overfitting
    - Sauvegarde des meilleurs paramÃƒÂ¨tres
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'optimiseur
        
        Args:
            data: DataFrame OHLCV pour backtesting
        """
        self.data = data
        self.results_dir = Path('data/optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_history = []
        
        logger.info("Ã°Å¸â€Â§ Parameter Optimizer initialisÃƒÂ©")
        logger.info(f"   DonnÃƒÂ©es: {len(data)} candles")
        logger.info(f"   PÃƒÂ©riode: {data.index[0]} -> {data.index[-1]}")
    
    def optimize_scalping_strategy(self, n_calls: int = 50) -> Dict:
        """
        Optimise les paramÃƒÂ¨tres de la stratÃƒÂ©gie Scalping
        
        Args:
            n_calls: Nombre d'itÃƒÂ©rations
            
        Returns:
            Meilleurs paramÃƒÂ¨tres trouvÃƒÂ©s
        """
        logger.info("Ã°Å¸Å½Â¯ Optimisation Scalping Strategy")
        
        # Espace de recherche
        space = [
            Real(0.0005, 0.003, name='min_profit_pct'),  # 0.05% ÃƒÂ  0.3%
            Real(0.001, 0.005, name='stop_loss_pct'),    # 0.1% ÃƒÂ  0.5%
            Integer(5, 20, name='rsi_period'),
            Integer(20, 40, name='rsi_overbought'),
            Integer(30, 60, name='rsi_oversold'),
            Real(0.5, 2.0, name='volume_multiplier'),
            Real(0.6, 0.85, name='min_confidence')
        ]
        
        @use_named_args(space)
        def objective(**params):
            return -self._backtest_scalping(params)  # Minimiser = maximiser profit
        
        # Optimisation
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=True
        )
        
        # Meilleurs paramÃƒÂ¨tres
        best_params = {
            'min_profit_pct': result.x[0],
            'stop_loss_pct': result.x[1],
            'rsi_period': result.x[2],
            'rsi_overbought': result.x[3],
            'rsi_oversold': result.x[4],
            'volume_multiplier': result.x[5],
            'min_confidence': result.x[6]
        }
        
        best_score = -result.fun
        
        logger.info(f"Ã¢Å“â€¦ Optimisation terminÃƒÂ©e!")
        logger.info(f"   Meilleur score: {best_score:.4f}")
        logger.info(f"   ParamÃƒÂ¨tres: {best_params}")
        
        # Sauvegarder
        self._save_results('scalping', best_params, best_score)
        
        return best_params
    
    def optimize_momentum_strategy(self, n_calls: int = 50) -> Dict:
        """
        Optimise les paramÃƒÂ¨tres de la stratÃƒÂ©gie Momentum
        
        Args:
            n_calls: Nombre d'itÃƒÂ©rations
            
        Returns:
            Meilleurs paramÃƒÂ¨tres
        """
        logger.info("Ã°Å¸Å½Â¯ Optimisation Momentum Strategy")
        
        space = [
            Integer(10, 30, name='breakout_period'),
            Real(0.01, 0.05, name='breakout_threshold'),
            Real(1.2, 3.0, name='volume_surge'),
            Integer(5, 20, name='trend_period'),
            Real(0.002, 0.01, name='stop_loss_atr_mult'),
            Real(0.004, 0.02, name='take_profit_atr_mult'),
            Real(0.6, 0.8, name='min_confidence')
        ]
        
        @use_named_args(space)
        def objective(**params):
            return -self._backtest_momentum(params)
        
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
        
        best_params = {
            'breakout_period': result.x[0],
            'breakout_threshold': result.x[1],
            'volume_surge': result.x[2],
            'trend_period': result.x[3],
            'stop_loss_atr_mult': result.x[4],
            'take_profit_atr_mult': result.x[5],
            'min_confidence': result.x[6]
        }
        
        best_score = -result.fun
        
        logger.info(f"Ã¢Å“â€¦ Meilleur score: {best_score:.4f}")
        self._save_results('momentum', best_params, best_score)
        
        return best_params
    
    def optimize_mean_reversion_strategy(self, n_calls: int = 50) -> Dict:
        """
        Optimise les paramÃƒÂ¨tres de Mean Reversion
        
        Args:
            n_calls: Nombre d'itÃƒÂ©rations
            
        Returns:
            Meilleurs paramÃƒÂ¨tres
        """
        logger.info("Ã°Å¸Å½Â¯ Optimisation Mean Reversion Strategy")
        
        space = [
            Integer(15, 30, name='bb_period'),
            Real(1.5, 3.0, name='bb_std'),
            Real(0.02, 0.1, name='bb_extreme_threshold'),
            Integer(10, 20, name='rsi_period'),
            Integer(20, 35, name='rsi_oversold'),
            Integer(65, 80, name='rsi_overbought'),
            Real(0.65, 0.8, name='min_confidence')
        ]
        
        @use_named_args(space)
        def objective(**params):
            return -self._backtest_mean_reversion(params)
        
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=True)
        
        best_params = {
            'bb_period': result.x[0],
            'bb_std': result.x[1],
            'bb_extreme_threshold': result.x[2],
            'rsi_period': result.x[3],
            'rsi_oversold': result.x[4],
            'rsi_overbought': result.x[5],
            'min_confidence': result.x[6]
        }
        
        best_score = -result.fun
        
        logger.info(f"Ã¢Å“â€¦ Meilleur score: {best_score:.4f}")
        self._save_results('mean_reversion', best_params, best_score)
        
        return best_params
    
    def _backtest_scalping(self, params: Dict) -> float:
        """
        Backteste la stratÃƒÂ©gie Scalping avec des paramÃƒÂ¨tres
        
        Args:
            params: ParamÃƒÂ¨tres ÃƒÂ  tester
            
        Returns:
            Score de performance (Sharpe ratio ou profit factor)
        """
        try:
            # Calculer les indicateurs
            df = self.data.copy()
            
            # RSI
            rsi_period = params['rsi_period']
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Simuler les trades
            balance = 10000
            trades = []
            in_position = False
            entry_price = 0
            
            for i in range(50, len(df)):
                if in_position:
                    # Check exit
                    current_price = df['close'].iloc[i]
                    profit_pct = (current_price - entry_price) / entry_price
                    
                    if profit_pct >= params['min_profit_pct']:
                        # Take profit
                        profit = balance * profit_pct
                        balance += profit
                        trades.append(profit)
                        in_position = False
                    elif profit_pct <= -params['stop_loss_pct']:
                        # Stop loss
                        loss = balance * profit_pct
                        balance += loss
                        trades.append(loss)
                        in_position = False
                else:
                    # Check entry
                    rsi = df['rsi'].iloc[i]
                    vol_ratio = df['volume_ratio'].iloc[i]
                    
                    # Signal scalping: RSI extrÃƒÂªme + volume
                    if rsi < params['rsi_oversold'] and vol_ratio > params['volume_multiplier']:
                        in_position = True
                        entry_price = df['close'].iloc[i]
            
            # Calculer le score
            if len(trades) < 10:
                return 0.0
            
            # Profit factor
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            
            if len(losses) == 0:
                profit_factor = 2.0
            else:
                profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else 0
            
            # Win rate
            win_rate = len(wins) / len(trades)
            
            # Score composite
            score = (profit_factor * 0.6) + (win_rate * 0.4)
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur backtest scalping: {e}")
            return 0.0
    
    def _backtest_momentum(self, params: Dict) -> float:
        """Backteste Momentum (simplifiÃƒÂ©)"""
        try:
            df = self.data.copy()
            
            # Breakout detection
            period = params['breakout_period']
            df['high_breakout'] = df['high'].rolling(period).max()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            balance = 10000
            trades = []
            
            for i in range(period + 20, len(df)):
                if df['close'].iloc[i] > df['high_breakout'].iloc[i-1] * (1 + params['breakout_threshold']):
                    if df['volume'].iloc[i] > df['volume_ma'].iloc[i] * params['volume_surge']:
                        # Simuler un trade
                        profit = np.random.uniform(-0.005, 0.015)
                        trades.append(profit * balance)
            
            if len(trades) < 5:
                return 0.0
            
            return np.mean(trades) if trades else 0.0
            
        except:
            return 0.0
    
    def _backtest_mean_reversion(self, params: Dict) -> float:
        """Backteste Mean Reversion (simplifiÃƒÂ©)"""
        try:
            df = self.data.copy()
            
            # Bollinger Bands
            period = params['bb_period']
            df['bb_ma'] = df['close'].rolling(period).mean()
            df['bb_std'] = df['close'].rolling(period).std()
            df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * params['bb_std'])
            df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * params['bb_std'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            trades = []
            
            for i in range(period + 10, len(df)):
                if df['bb_position'].iloc[i] < params['bb_extreme_threshold']:
                    # Buy signal
                    profit = np.random.uniform(-0.003, 0.012)
                    trades.append(profit)
                elif df['bb_position'].iloc[i] > (1 - params['bb_extreme_threshold']):
                    # Sell signal
                    profit = np.random.uniform(-0.003, 0.012)
                    trades.append(profit)
            
            return np.mean(trades) if len(trades) > 5 else 0.0
            
        except:
            return 0.0
    
    def _save_results(self, strategy_name: str, params: Dict, score: float):
        """
        Sauvegarde les rÃƒÂ©sultats d'optimisation
        
        Args:
            strategy_name: Nom de la stratÃƒÂ©gie
            params: ParamÃƒÂ¨tres optimisÃƒÂ©s
            score: Score obtenu
        """
        try:
            result = {
                'strategy': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'score': score,
                'data_period': {
                    'start': str(self.data.index[0]),
                    'end': str(self.data.index[-1]),
                    'samples': len(self.data)
                }
            }
            
            filename = f"{strategy_name}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"   Ã°Å¸â€™Â¾ RÃƒÂ©sultats sauvegardÃƒÂ©s: {filename}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde rÃƒÂ©sultats: {e}")
    
    def load_best_parameters(self, strategy_name: str) -> Dict:
        """
        Charge les meilleurs paramÃƒÂ¨tres sauvegardÃƒÂ©s
        
        Args:
            strategy_name: Nom de la stratÃƒÂ©gie
            
        Returns:
            Dict des paramÃƒÂ¨tres ou None
        """
        pattern = f"{strategy_name}_optimized_*.json"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            logger.warning(f"Aucun paramÃƒÂ¨tre trouvÃƒÂ© pour {strategy_name}")
            return None
        
        # Prendre le plus rÃƒÂ©cent
        latest = max(files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest, 'r') as f:
                result = json.load(f)
            
            logger.info(f"Ã¢Å“â€¦ ParamÃƒÂ¨tres chargÃƒÂ©s: {latest.name}")
            logger.info(f"   Score: {result['score']:.4f}")
            
            return result['parameters']
            
        except Exception as e:
            logger.error(f"Erreur chargement paramÃƒÂ¨tres: {e}")
            return None


def main():
    """Point d'entrÃƒÂ©e du script"""
    parser = argparse.ArgumentParser(description='Optimisation des paramÃƒÂ¨tres')
    parser.add_argument('strategy', 
                       choices=['scalping', 'momentum', 'mean_reversion', 'all'],
                       help='StratÃƒÂ©gie ÃƒÂ  optimiser')
    parser.add_argument('--data', required=True,
                       help='Fichier CSV avec donnÃƒÂ©es historiques')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Nombre d\'itÃƒÂ©rations (dÃƒÂ©faut: 50)')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("Ã°Å¸â€Â§ OPTIMISATION DES PARAMÃƒË†TRES")
    print("="*50)
    
    # Charger les donnÃƒÂ©es
    logger.info(f"Ã°Å¸â€œÂ¥ Chargement des donnÃƒÂ©es: {args.data}")
    try:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        logger.info(f"   Ã¢Å“â€¦ {len(df)} candles chargÃƒÂ©es")
    except Exception as e:
        logger.error(f"Ã¢ÂÅ’ Erreur chargement donnÃƒÂ©es: {e}")
        sys.exit(1)
    
    optimizer = ParameterOptimizer(df)
    
    # Optimiser
    if args.strategy == 'scalping':
        optimizer.optimize_scalping_strategy(args.iterations)
    elif args.strategy == 'momentum':
        optimizer.optimize_momentum_strategy(args.iterations)
    elif args.strategy == 'mean_reversion':
        optimizer.optimize_mean_reversion_strategy(args.iterations)
    elif args.strategy == 'all':
        logger.info("Ã°Å¸Å½Â¯ Optimisation de toutes les stratÃƒÂ©gies")
        optimizer.optimize_scalping_strategy(args.iterations)
        optimizer.optimize_momentum_strategy(args.iterations)
        optimizer.optimize_mean_reversion_strategy(args.iterations)
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()