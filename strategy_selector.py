"""
Strategy Selector - SÃƒÂ©lectionne et coordonne les stratÃƒÂ©gies
GÃƒÂ¨re l'allocation du capital entre stratÃƒÂ©gies
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    SÃƒÂ©lecteur intelligent de stratÃƒÂ©gies
    
    ResponsabilitÃƒÂ©s:
    - GÃƒÂ©rer l'allocation du capital par stratÃƒÂ©gie
    - Activer/dÃƒÂ©sactiver dynamiquement les stratÃƒÂ©gies
    - Prioriser les signaux selon les conditions de marchÃƒÂ©
    - Suivre les performances relatives
    """
    
    def __init__(self, strategies: List, config: Dict = None):
        """
        Initialise le sÃƒÂ©lecteur
        
        Args:
            strategies: Liste des stratÃƒÂ©gies disponibles
            config: Configuration
        """
        self.strategies = {s.name: s for s in strategies}
        
        default_config = {
            'allocations': {
                'Scalping_Strategy': 0.40,
                'Momentum_Strategy': 0.25,
                'Mean_Reversion_Strategy': 0.20,
                'Pattern_Strategy': 0.10,
                'ML_Strategy': 0.05
            },
            'min_win_rate': 0.55,  # Win rate minimum pour garder active
            'performance_window': 100,  # Nombre de trades pour ÃƒÂ©valuation
            'rebalance_frequency': 3600,  # 1h en secondes
            'max_concurrent_strategies': 3  # Max stratÃƒÂ©gies simultanÃƒÂ©es par symbole
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.last_rebalance = datetime.now()
        
        # Stats globales
        self.stats = {
            'total_signals': 0,
            'signals_by_strategy': {},
            'active_strategies': len(strategies)
        }
        
        # Initialiser les allocations
        self._apply_allocations()
        
        logger.info(f"Ã¢Å“â€¦ Strategy Selector initialisÃƒÂ© avec {len(self.strategies)} stratÃƒÂ©gies")
    
    def _apply_allocations(self):
        """Applique les allocations de capital aux stratÃƒÂ©gies"""
        for name, strategy in self.strategies.items():
            if name in self.config['allocations']:
                strategy.config['allocation'] = self.config['allocations'][name]
                logger.info(f"  {name}: {strategy.config['allocation']:.1%} allocation")
    
    def select_best_signal(self, signals: List[Dict], market_data: Dict) -> Optional[Dict]:
        """
        SÃƒÂ©lectionne le meilleur signal parmi plusieurs
        
        Args:
            signals: Liste des signaux gÃƒÂ©nÃƒÂ©rÃƒÂ©s
            market_data: DonnÃƒÂ©es de marchÃƒÂ© actuelles
            
        Returns:
            Meilleur signal ou None
        """
        if not signals:
            return None
        
        try:
            # Filtrer les signaux valides
            valid_signals = []
            
            for signal in signals:
                strategy_name = signal.get('metadata', {}).get('strategy')
                
                # VÃƒÂ©rifier que la stratÃƒÂ©gie est active
                if strategy_name and strategy_name in self.strategies:
                    if self.strategies[strategy_name].is_active:
                        valid_signals.append(signal)
            
            if not valid_signals:
                return None
            
            # Scoring des signaux
            scored_signals = []
            for signal in valid_signals:
                score = self._score_signal(signal, market_data)
                scored_signals.append((score, signal))
            
            # Trier par score dÃƒÂ©croissant
            scored_signals.sort(key=lambda x: x[0], reverse=True)
            
            # Prendre le meilleur
            best_score, best_signal = scored_signals[0]
            
            if best_score > 0:
                self.stats['total_signals'] += 1
                strategy_name = best_signal.get('metadata', {}).get('strategy', 'Unknown')
                self.stats['signals_by_strategy'][strategy_name] = \
                    self.stats['signals_by_strategy'].get(strategy_name, 0) + 1
                
                logger.info(f"Ã¢Å“â€¦ Signal sÃƒÂ©lectionnÃƒÂ©: {strategy_name} (score: {best_score:.2f})")
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur select_best_signal: {e}")
            return None
    
    def _score_signal(self, signal: Dict, market_data: Dict) -> float:
        """
        Score un signal selon plusieurs critÃƒÂ¨res
        
        Args:
            signal: Le signal ÃƒÂ  scorer
            market_data: DonnÃƒÂ©es de marchÃƒÂ©
            
        Returns:
            Score du signal (0-100)
        """
        score = 0.0
        
        try:
            # 1. Confiance du signal (0-40 points)
            confidence = signal.get('confidence', 0)
            score += confidence * 40
            
            # 2. Performance de la stratÃƒÂ©gie (0-30 points)
            strategy_name = signal.get('metadata', {}).get('strategy')
            if strategy_name and strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                win_rate = strategy.performance.get('win_rate', 0)
                score += win_rate * 30
            
            # 3. QualitÃƒÂ© du Risk/Reward (0-20 points)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            entry_price = signal.get('price', 0)
            
            if entry_price > 0 and stop_loss > 0 and take_profit > 0:
                if signal['side'] == 'BUY':
                    risk = entry_price - stop_loss
                    reward = take_profit - entry_price
                else:
                    risk = stop_loss - entry_price
                    reward = entry_price - take_profit
                
                if risk > 0:
                    rr_ratio = reward / risk
                    # Bonus si RR > 2
                    if rr_ratio >= 2:
                        score += 20
                    elif rr_ratio >= 1.5:
                        score += 15
                    elif rr_ratio >= 1:
                        score += 10
            
            # 4. Alignement avec tendance globale (0-10 points)
            if market_data:
                df = market_data.get('df')
                if df is not None and len(df) > 50:
                    # Tendance simple: prix actuel vs MA50
                    current_price = df['close'].iloc[-1]
                    ma50 = df['close'].rolling(50).mean().iloc[-1]
                    
                    if signal['side'] == 'BUY' and current_price > ma50:
                        score += 10
                    elif signal['side'] == 'SELL' and current_price < ma50:
                        score += 10
            
            return min(score, 100)  # Cap ÃƒÂ  100
            
        except Exception as e:
            logger.error(f"Erreur scoring signal: {e}")
            return 0.0
    
    def analyze_all_strategies(self, data: Dict) -> List[Dict]:
        """
        Analyse avec toutes les stratÃƒÂ©gies actives
        
        Args:
            data: DonnÃƒÂ©es de marchÃƒÂ©
            
        Returns:
            Liste de tous les signaux gÃƒÂ©nÃƒÂ©rÃƒÂ©s
        """
        signals = []
        
        for name, strategy in self.strategies.items():
            if not strategy.is_active:
                continue
            
            try:
                signal = strategy.analyze(data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Erreur analyse {name}: {e}")
        
        return signals
    
    def rebalance_strategies(self):
        """
        RÃƒÂ©ÃƒÂ©quilibre les stratÃƒÂ©gies selon leurs performances
        """
        try:
            current_time = datetime.now()
            time_since_rebalance = (current_time - self.last_rebalance).total_seconds()
            
            if time_since_rebalance < self.config['rebalance_frequency']:
                return
            
            logger.info("Ã°Å¸â€â€ž RÃƒÂ©ÃƒÂ©quilibrage des stratÃƒÂ©gies...")
            
            # Ãƒâ€°valuer chaque stratÃƒÂ©gie
            performances = {}
            
            for name, strategy in self.strategies.items():
                perf = strategy.performance
                
                # Calculer un score de performance
                win_rate = perf.get('win_rate', 0)
                profit_factor = perf.get('profit_factor', 0)
                total_trades = perf.get('winning_trades', 0) + perf.get('losing_trades', 0)
                
                # Score composite
                performance_score = 0
                if total_trades >= 10:  # Minimum de trades pour ÃƒÂªtre significatif
                    performance_score = (win_rate * 0.6) + (min(profit_factor / 3, 1) * 0.4)
                
                performances[name] = {
                    'score': performance_score,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': total_trades,
                    'is_active': strategy.is_active
                }
            
            # DÃƒÂ©sactiver les stratÃƒÂ©gies sous-performantes
            for name, perf in performances.items():
                if perf['total_trades'] >= 20:  # Au moins 20 trades
                    if perf['win_rate'] < self.config['min_win_rate']:
                        if self.strategies[name].is_active:
                            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â DÃƒÂ©sactivation {name}: win rate {perf['win_rate']:.2%}")
                            self.strategies[name].disable()
                    else:
                        if not self.strategies[name].is_active:
                            logger.info(f"Ã¢Å“â€¦ RÃƒÂ©activation {name}: win rate {perf['win_rate']:.2%}")
                            self.strategies[name].enable()
            
            # Ajuster les allocations selon les performances
            self._adjust_allocations(performances)
            
            self.last_rebalance = current_time
            self.stats['active_strategies'] = sum(1 for s in self.strategies.values() if s.is_active)
            
            logger.info(f"Ã¢Å“â€¦ RÃƒÂ©ÃƒÂ©quilibrage terminÃƒÂ© - {self.stats['active_strategies']} stratÃƒÂ©gies actives")
            
        except Exception as e:
            logger.error(f"Erreur rebalance_strategies: {e}")
    
    def _adjust_allocations(self, performances: Dict):
        """
        Ajuste dynamiquement les allocations selon performances
        
        Args:
            performances: Dict des performances par stratÃƒÂ©gie
        """
        try:
            # Calculer les scores normalisÃƒÂ©s
            total_score = sum(p['score'] for p in performances.values() if p['score'] > 0)
            
            if total_score == 0:
                return  # Garder les allocations par dÃƒÂ©faut
            
            # Nouvelles allocations proportionnelles aux scores
            new_allocations = {}
            for name, perf in performances.items():
                if perf['score'] > 0 and perf['is_active']:
                    new_allocations[name] = perf['score'] / total_score
                else:
                    new_allocations[name] = 0
            
            # Appliquer avec lissage (70% ancien, 30% nouveau)
            for name, new_alloc in new_allocations.items():
                old_alloc = self.config['allocations'].get(name, 0)
                smoothed_alloc = (old_alloc * 0.7) + (new_alloc * 0.3)
                self.config['allocations'][name] = smoothed_alloc
                
                if name in self.strategies:
                    self.strategies[name].config['allocation'] = smoothed_alloc
            
            logger.info("Allocations ajustÃƒÂ©es:")
            for name, alloc in self.config['allocations'].items():
                if alloc > 0:
                    logger.info(f"  {name}: {alloc:.1%}")
                    
        except Exception as e:
            logger.error(f"Erreur adjust_allocations: {e}")
    
    def get_strategy(self, name: str):
        """Retourne une stratÃƒÂ©gie par son nom"""
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> List:
        """Retourne toutes les stratÃƒÂ©gies"""
        return list(self.strategies.values())
    
    def get_active_strategies(self) -> List:
        """Retourne les stratÃƒÂ©gies actives"""
        return [s for s in self.strategies.values() if s.is_active]
    
    def get_strategy_stats(self) -> Dict:
        """Retourne les stats de toutes les stratÃƒÂ©gies"""
        stats = {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.get_active_strategies()),
            'strategies': {}
        }
        
        for name, strategy in self.strategies.items():
            stats['strategies'][name] = {
                'active': strategy.is_active,
                'allocation': strategy.config.get('allocation', 0),
                'performance': strategy.get_performance_summary()
            }
        
        return stats
    
    def get_selector_stats(self) -> Dict:
        """Retourne les stats du sÃƒÂ©lecteur"""
        return {
            'total_signals': self.stats['total_signals'],
            'signals_by_strategy': self.stats['signals_by_strategy'].copy(),
            'active_strategies': self.stats['active_strategies'],
            'last_rebalance': self.last_rebalance
        }
    
    def force_rebalance(self):
        """Force un rÃƒÂ©ÃƒÂ©quilibrage immÃƒÂ©diat"""
        self.last_rebalance = datetime.now() - timedelta(seconds=self.config['rebalance_frequency'] + 1)
        self.rebalance_strategies()
    
    def enable_all_strategies(self):
        """Active toutes les stratÃƒÂ©gies"""
        for strategy in self.strategies.values():
            strategy.enable()
        logger.info("Ã¢Å“â€¦ Toutes les stratÃƒÂ©gies activÃƒÂ©es")
    
    def disable_all_strategies(self):
        """DÃƒÂ©sactive toutes les stratÃƒÂ©gies"""
        for strategy in self.strategies.values():
            strategy.disable()
        logger.info("Ã¢Å¡Â Ã¯Â¸Â Toutes les stratÃƒÂ©gies dÃƒÂ©sactivÃƒÂ©es")
    
    def reset_all_performances(self):
        """RÃƒÂ©initialise les performances de toutes les stratÃƒÂ©gies"""
        for strategy in self.strategies.values():
            strategy.reset_performance()
        logger.info("Ã°Å¸â€â€ž Performances rÃƒÂ©initialisÃƒÂ©es")


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Strategy Selector"""
    from datetime import timedelta
    
    # Mock strategies pour test
    class MockStrategy:
        def __init__(self, name, allocation):
            self.name = name
            self.is_active = True
            self.config = {'allocation': allocation}
            self.performance = {
                'win_rate': np.random.uniform(0.5, 0.8),
                'profit_factor': np.random.uniform(1.0, 2.5),
                'winning_trades': np.random.randint(10, 50),
                'losing_trades': np.random.randint(5, 20),
                'total_profit': np.random.uniform(100, 500)
            }
        
        def analyze(self, data):
            # Simuler un signal alÃƒÂ©atoire
            if np.random.random() > 0.7:
                return {
                    'type': 'ENTRY',
                    'side': 'BUY',
                    'price': 100,
                    'confidence': np.random.uniform(0.6, 0.9),
                    'stop_loss': 98,
                    'take_profit': 105,
                    'metadata': {'strategy': self.name}
                }
            return None
        
        def enable(self):
            self.is_active = True
        
        def disable(self):
            self.is_active = False
        
        def get_performance_summary(self):
            return self.performance
        
        def reset_performance(self):
            pass
    
    # CrÃƒÂ©er des mock strategies
    strategies = [
        MockStrategy('Scalping_Strategy', 0.40),
        MockStrategy('Momentum_Strategy', 0.25),
        MockStrategy('Mean_Reversion_Strategy', 0.20),
        MockStrategy('Pattern_Strategy', 0.10),
        MockStrategy('ML_Strategy', 0.05)
    ]
    
    selector = StrategySelector(strategies)
    
    print("Test Strategy Selector")
    print("=" * 50)
    print(f"StratÃƒÂ©gies chargÃƒÂ©es: {len(selector.strategies)}")
    print(f"StratÃƒÂ©gies actives: {selector.stats['active_strategies']}")
    
    # Test analyse
    print("\nTest analyse avec toutes les stratÃƒÂ©gies:")
    signals = selector.analyze_all_strategies({'df': None})
    print(f"Signaux gÃƒÂ©nÃƒÂ©rÃƒÂ©s: {len(signals)}")
    
    if signals:
        best = selector.select_best_signal(signals, {'df': None})
        if best:
            print(f"\nÃ¢Å“â€¦ Meilleur signal sÃƒÂ©lectionnÃƒÂ©:")
            print(f"   StratÃƒÂ©gie: {best['metadata']['strategy']}")
            print(f"   Confiance: {best['confidence']:.2%}")
            print(f"   Side: {best['side']}")
    
    # Test rÃƒÂ©ÃƒÂ©quilibrage
    print("\nTest rÃƒÂ©ÃƒÂ©quilibrage:")
    selector.force_rebalance()
    
    # Stats finales
    print("\nStats finales:")
    stats = selector.get_selector_stats()
    print(f"Total signaux: {stats['total_signals']}")
    print(f"Signaux par stratÃƒÂ©gie: {stats['signals_by_strategy']}")
    print(f"StratÃƒÂ©gies actives: {stats['active_strategies']}")