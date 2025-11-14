"""
Position Sizing Module pour The Bot
Calcul optimisÃƒÂ© des tailles de position avec gestion du risque
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Set
from dataclasses import dataclass
import logging

# IMPORTER LES CONSTANTES DEPUIS utils/constants.py
from utils.constants import (
    MAX_CORRELATION_ALLOWED,
    MIN_POSITION_SIZE_PCT,
    MAX_POSITION_SIZE_PCT
)

logger = logging.getLogger(__name__)

# ===== CONSTANTES MANQUANTES Ãƒâ‚¬ AJOUTER =====
# Limites de position et corrÃƒÂ©lation
MAX_CORRELATION_ALLOWED = 0.70  # CorrÃƒÂ©lation max tolÃƒÂ©rÃƒÂ©e entre positions
MIN_POSITION_SIZE_PCT = 0.01    # Taille minimum en % du capital (1%)
MAX_POSITION_SIZE_PCT = 0.25    # Taille maximum en % du capital (25%)

# Autres constantes utiles pour le module
DEFAULT_RISK_PER_TRADE = 0.02   # Risque par dÃƒÂ©faut par trade (2%)
KELLY_FRACTION = 0.25            # Fraction de Kelly pour sizing
MIN_TRADE_SIZE_USDC = 50.0      # Taille minimum Binance


class PositionSizer:
    """
    GÃƒÂ¨re le calcul de la taille des positions avec plusieurs mÃƒÂ©thodes
    
    MÃƒÂ©thodes supportÃƒÂ©es:
    - Fixed Risk (risque fixe par trade)
    - Kelly Criterion (optimal mais risquÃƒÂ©)
    - Volatility-based (ajustÃƒÂ© ÃƒÂ  la volatilitÃƒÂ©)
    - Risk Parity (ÃƒÂ©quilibrage du risque)
    - Dynamic (combinaison adaptative)
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le position sizer
        
        Args:
            config: Configuration incluant:
                - initial_capital: Capital de dÃƒÂ©part
                - risk_per_trade: Risque max par trade (ex: 0.02 pour 2%)
                - max_position_size: Taille max d'une position (ex: 0.25 pour 25%)
                - min_position_size: Taille min en USDC (ex: 50)
                - kelly_fraction: Fraction de Kelly ÃƒÂ  utiliser (ex: 0.25)
                - use_dynamic_sizing: Utiliser sizing dynamique
        """
        self.config = config
        self.capital = config.get('initial_capital', 1000)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.max_position_pct = config.get('max_position_size', 0.25)
        self.min_position_usdc = config.get('min_position_size', 50)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        self.use_dynamic = config.get('use_dynamic_sizing', True)
        
        # Ãƒâ€°tat du portfolio
        self.open_positions = {}
        self.total_exposure = 0
        self.current_drawdown = 0
        self.peak_capital = self.capital
        
        # Historique pour calculs
        self.trade_history = []
        self.win_rate = 0.5  # Initial assumption
        self.avg_win = 0.01
        self.avg_loss = 0.01
        
        logger.info(f"Position Sizer initialisÃƒÂ© - Capital: ${self.capital:,.2f}")
    
    def calculate_position_size(self, 
                               signal: Dict,
                               current_price: float,
                               stop_loss_price: float,
                               market_conditions: Optional[Dict] = None) -> Dict:
        """
        Calcule la taille optimale de la position
        
        Args:
            signal: Signal de trading avec confidence, side, etc.
            current_price: Prix actuel de l'actif
            stop_loss_price: Prix du stop loss
            market_conditions: Conditions de marchÃƒÂ© (volatilitÃƒÂ©, trend, etc.)
            
        Returns:
            Dict avec:
                - position_size_usdc: Taille en USDC
                - position_size_units: Nombre d'unitÃƒÂ©s
                - risk_amount: Montant risquÃƒÂ©
                - method_used: MÃƒÂ©thode utilisÃƒÂ©e
                - confidence_adjusted: Si ajustÃƒÂ© par confidence
        """
        try:
            # Calcul du risque de base
            stop_loss_distance = abs(current_price - stop_loss_price) / current_price
            
            if stop_loss_distance == 0:
                logger.warning("Stop loss distance = 0, utilisation du dÃƒÂ©faut")
                stop_loss_distance = 0.003  # 0.3% dÃƒÂ©faut
            
            # Calculer avec diffÃƒÂ©rentes mÃƒÂ©thodes
            sizes = {}
            
            # 1. Fixed Risk Method (mÃƒÂ©thode de base)
            sizes['fixed_risk'] = self._fixed_risk_sizing(stop_loss_distance)
            
            # 2. Kelly Criterion (si assez d'historique)
            if len(self.trade_history) >= 20:
                sizes['kelly'] = self._kelly_sizing(signal.get('confidence', 0.65))
            
            # 3. Volatility-based (si donnÃƒÂ©es disponibles)
            if market_conditions and 'volatility' in market_conditions:
                sizes['volatility'] = self._volatility_sizing(
                    market_conditions['volatility'],
                    stop_loss_distance
                )
            
            # 4. Risk Parity (si plusieurs positions)
            if len(self.open_positions) > 0:
                sizes['risk_parity'] = self._risk_parity_sizing(signal['symbol'])
            
            # SÃƒÂ©lection de la taille finale
            if self.use_dynamic:
                final_size = self._dynamic_sizing(sizes, signal, market_conditions)
            else:
                final_size = sizes.get('fixed_risk', self.min_position_usdc)
            
            # Ajustements finaux
            final_size = self._apply_adjustments(
                final_size,
                signal,
                market_conditions
            )
            
            # Contraintes absolues
            final_size = self._apply_constraints(final_size)
            
            # Calculer le nombre d'unitÃƒÂ©s
            position_units = final_size / current_price
            
            # Montant risquÃƒÂ©
            risk_amount = final_size * stop_loss_distance
            
            result = {
                'position_size_usdc': round(final_size, 2),
                'position_size_units': position_units,
                'risk_amount': round(risk_amount, 2),
                'risk_percent': risk_amount / self.capital,
                'method_used': 'dynamic' if self.use_dynamic else 'fixed_risk',
                'stop_loss_distance': stop_loss_distance,
                'methods_calculated': sizes
            }
            
            logger.info(f"Position calculÃƒÂ©e: ${result['position_size_usdc']:.2f} "
                       f"(Risk: ${result['risk_amount']:.2f} = {result['risk_percent']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur calcul position: {e}")
            # Retour sÃƒÂ©curitaire
            return {
                'position_size_usdc': self.min_position_usdc,
                'position_size_units': self.min_position_usdc / current_price,
                'risk_amount': self.min_position_usdc * 0.02,
                'risk_percent': 0.02,
                'method_used': 'fallback',
                'error': str(e)
            }
    
    def _fixed_risk_sizing(self, stop_loss_distance: float) -> float:
        """
        MÃƒÂ©thode de risque fixe : risque toujours le mÃƒÂªme % du capital
        
        Args:
            stop_loss_distance: Distance au stop loss en %
            
        Returns:
            Taille de position en USDC
        """
        risk_amount = self.capital * self.risk_per_trade
        position_size = risk_amount / stop_loss_distance
        
        return position_size
    
    def _kelly_sizing(self, win_probability: float) -> float:
        """
        Kelly Criterion pour taille optimale
        f* = (p*b - q) / b
        oÃƒÂ¹:
        - p = probabilitÃƒÂ© de gain
        - q = probabilitÃƒÂ© de perte (1-p)
        - b = ratio gain/perte
        
        Args:
            win_probability: ProbabilitÃƒÂ© de succÃƒÂ¨s estimÃƒÂ©e
            
        Returns:
            Taille de position en USDC
        """
        if self.avg_loss == 0:
            return self.min_position_usdc
        
        # Calcul Kelly
        p = win_probability
        q = 1 - p
        b = self.avg_win / self.avg_loss
        
        kelly_percent = (p * b - q) / b
        
        # Appliquer fraction de Kelly (plus conservateur)
        kelly_percent = kelly_percent * self.kelly_fraction
        
        # Contraindre entre 0 et max position
        kelly_percent = max(0, min(kelly_percent, self.max_position_pct))
        
        position_size = self.capital * kelly_percent
        
        return position_size
    
    def _volatility_sizing(self, volatility: float, stop_loss_distance: float) -> float:
        """
        Sizing basÃƒÂ© sur la volatilitÃƒÂ© : moins de taille si marchÃƒÂ© volatil
        
        Args:
            volatility: VolatilitÃƒÂ© actuelle (ATR/prix par exemple)
            stop_loss_distance: Distance au stop loss
            
        Returns:
            Taille de position en USDC
        """
        # VolatilitÃƒÂ© cible (2% par jour est raisonnable)
        target_volatility = 0.02
        
        # Ajustement
        if volatility > 0:
            volatility_scalar = target_volatility / volatility
            volatility_scalar = max(0.5, min(volatility_scalar, 2.0))  # Limiter entre 0.5x et 2x
        else:
            volatility_scalar = 1.0
        
        # Position de base
        base_position = self.capital * self.risk_per_trade / stop_loss_distance
        
        # Ajuster par volatilitÃƒÂ©
        position_size = base_position * volatility_scalar
        
        return position_size
    
    def _risk_parity_sizing(self, symbol: str) -> float:
        """
        Risk Parity : ÃƒÂ©quilibrer le risque entre toutes les positions
        
        Args:
            symbol: Symbole de la nouvelle position
            
        Returns:
            Taille de position en USDC
        """
        # Nombre total de positions (incluant la nouvelle)
        total_positions = len(self.open_positions) + 1
        
        # Capital disponible
        available_capital = self.capital - self.total_exposure
        
        # Allouer ÃƒÂ©quitablement le risque
        risk_per_position = self.risk_per_trade * self.capital / total_positions
        
        # Convertir en taille de position (assume stop loss ÃƒÂ  2%)
        position_size = risk_per_position / 0.02
        
        # Ne pas dÃƒÂ©passer le capital disponible
        position_size = min(position_size, available_capital * 0.9)
        
        return position_size
    
    def _dynamic_sizing(self, 
                       sizes: Dict[str, float],
                       signal: Dict,
                       market_conditions: Optional[Dict]) -> float:
        """
        Combine intelligemment plusieurs mÃƒÂ©thodes selon les conditions
        
        Args:
            sizes: Dict des tailles calculÃƒÂ©es par diffÃƒÂ©rentes mÃƒÂ©thodes
            signal: Signal de trading
            market_conditions: Conditions de marchÃƒÂ©
            
        Returns:
            Taille finale en USDC
        """
        weights = {}
        
        # Poids de base
        weights['fixed_risk'] = 0.4  # Toujours inclure pour stabilitÃƒÂ©
        
        # Kelly si disponible et win rate bon
        if 'kelly' in sizes and self.win_rate > 0.55:
            weights['kelly'] = 0.3
        
        # Volatility si disponible
        if 'volatility' in sizes:
            weights['volatility'] = 0.2
        
        # Risk parity si plusieurs positions
        if 'risk_parity' in sizes:
            weights['risk_parity'] = 0.1
        
        # Ajuster les poids selon la confiance du signal
        confidence = signal.get('confidence', 0.65)
        if confidence > 0.8:
            # Augmenter le poids de Kelly si trÃƒÂ¨s confiant
            if 'kelly' in weights:
                weights['kelly'] *= 1.5
        elif confidence < 0.6:
            # Plus conservateur si peu confiant
            weights['fixed_risk'] *= 1.5
        
        # Normaliser les poids
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculer la moyenne pondÃƒÂ©rÃƒÂ©e
        final_size = sum(sizes.get(method, 0) * weight 
                        for method, weight in weights.items())
        
        logger.debug(f"Dynamic sizing - Weights: {weights}, Final: ${final_size:.2f}")
        
        return final_size
    
    def _apply_adjustments(self,
                          size: float,
                          signal: Dict,
                          market_conditions: Optional[Dict]) -> float:
        """
        Applique des ajustements contextuels ÃƒÂ  la taille
        
        Args:
            size: Taille de base
            signal: Signal de trading
            market_conditions: Conditions de marchÃƒÂ©
            
        Returns:
            Taille ajustÃƒÂ©e
        """
        adjusted_size = size
        
        # 1. Ajustement par confidence du signal
        confidence = signal.get('confidence', 0.65)
        if confidence > 0.8:
            adjusted_size *= 1.2  # +20% si trÃƒÂ¨s confiant
        elif confidence < 0.6:
            adjusted_size *= 0.7  # -30% si peu confiant
        
        # 2. Ajustement par drawdown
        if self.current_drawdown > 0.05:  # Drawdown > 5%
            # RÃƒÂ©duire progressivement
            reduction = min(self.current_drawdown * 2, 0.5)  # Max 50% rÃƒÂ©duction
            adjusted_size *= (1 - reduction)
            logger.info(f"RÃƒÂ©duction pour drawdown: -{reduction:.1%}")
        
        # 3. Ajustement par conditions de marchÃƒÂ©
        if market_conditions:
            # Trend fort
            if market_conditions.get('trend_strength', 0) > 0.7:
                if signal['side'] == market_conditions.get('trend_direction'):
                    adjusted_size *= 1.1  # +10% dans le sens du trend
                else:
                    adjusted_size *= 0.8  # -20% contre trend
            
            # Haute volatilitÃƒÂ©
            if market_conditions.get('volatility', 0) > 0.05:  # > 5% volatilitÃƒÂ©
                adjusted_size *= 0.8  # RÃƒÂ©duire en pÃƒÂ©riode volatile
        
        # 4. Ajustement par nombre de positions ouvertes
        if len(self.open_positions) > 10:
            # RÃƒÂ©duire si trop de positions
            adjusted_size *= 0.9
        elif len(self.open_positions) < 3:
            # Augmenter si peu de positions
            adjusted_size *= 1.1
        
        # 5. Ajustement par performance rÃƒÂ©cente
        recent_performance = self._calculate_recent_performance(10)
        if recent_performance > 0.1:  # +10% sur 10 derniers trades
            adjusted_size *= 1.15  # Augmenter aprÃƒÂ¨s bonne perf
        elif recent_performance < -0.05:  # -5% sur 10 derniers
            adjusted_size *= 0.85  # RÃƒÂ©duire aprÃƒÂ¨s mauvaise perf
        
        return adjusted_size
    
    def _apply_constraints(self, size: float) -> float:
        """
        Applique les contraintes absolues
        
        Args:
            size: Taille calculÃƒÂ©e
            
        Returns:
            Taille contrainte
        """
        # Minimum absolu
        size = max(size, self.min_position_usdc)
        
        # Maximum par position
        max_position = self.capital * self.max_position_pct
        size = min(size, max_position)
        
        # Capital disponible
        available = self.capital - self.total_exposure
        size = min(size, available * 0.95)  # Garder 5% de marge
        
        # Arrondir ÃƒÂ  2 dÃƒÂ©cimales
        size = round(size, 2)
        
        return size
    
    def _calculate_recent_performance(self, n_trades: int = 10) -> float:
        """
        Calcule la performance sur les n derniers trades
        
        Args:
            n_trades: Nombre de trades ÃƒÂ  considÃƒÂ©rer
            
        Returns:
            Performance en %
        """
        if len(self.trade_history) < n_trades:
            return 0
        
        recent = self.trade_history[-n_trades:]
        total_return = sum(trade.get('profit_pct', 0) for trade in recent)
        
        return total_return / n_trades
    
    def update_capital(self, new_capital: float):
        """
        Met ÃƒÂ  jour le capital disponible
        
        Args:
            new_capital: Nouveau capital
        """
        self.capital = new_capital
        
        # Mettre ÃƒÂ  jour le peak pour drawdown
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
        
        logger.info(f"Capital mis ÃƒÂ  jour: ${new_capital:.2f} (DD: {self.current_drawdown:.2%})")
    
    def register_position(self, 
                         symbol: str,
                         size_usdc: float,
                         entry_price: float,
                         side: str):
        """
        Enregistre une nouvelle position
        
        Args:
            symbol: Symbole
            size_usdc: Taille en USDC
            entry_price: Prix d'entrÃƒÂ©e
            side: BUY ou SELL
        """
        self.open_positions[symbol] = {
            'size_usdc': size_usdc,
            'entry_price': entry_price,
            'side': side,
            'timestamp': np.datetime64('now')
        }
        
        self.total_exposure += size_usdc
        
        logger.info(f"Position enregistrÃƒÂ©e: {symbol} ${size_usdc:.2f} @ {entry_price}")
        logger.info(f"Exposition totale: ${self.total_exposure:.2f} ({self.total_exposure/self.capital:.1%})")
    
    def close_position(self,
                       symbol: str,
                       exit_price: float,
                       profit_pct: float):
        """
        Ferme une position et met ÃƒÂ  jour les stats
        
        Args:
            symbol: Symbole
            exit_price: Prix de sortie
            profit_pct: Profit en %
        """
        if symbol in self.open_positions:
            position = self.open_positions[symbol]
            
            # Mettre ÃƒÂ  jour l'exposition
            self.total_exposure -= position['size_usdc']
            
            # Ajouter ÃƒÂ  l'historique
            self.trade_history.append({
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'size_usdc': position['size_usdc'],
                'side': position['side']
            })
            
            # Mettre ÃƒÂ  jour les stats
            self._update_stats()
            
            # Retirer la position
            del self.open_positions[symbol]
            
            logger.info(f"Position fermÃƒÂ©e: {symbol} @ {exit_price} ({profit_pct:+.2%})")
    
    def _update_stats(self):
        """Met ÃƒÂ  jour les statistiques de trading"""
        if len(self.trade_history) == 0:
            return
        
        # Win rate
        wins = [t for t in self.trade_history if t['profit_pct'] > 0]
        self.win_rate = len(wins) / len(self.trade_history)
        
        # Average win/loss
        if wins:
            self.avg_win = np.mean([t['profit_pct'] for t in wins])
        
        losses = [t for t in self.trade_history if t['profit_pct'] <= 0]
        if losses:
            self.avg_loss = abs(np.mean([t['profit_pct'] for t in losses]))
        
        # Garder seulement les 100 derniers trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques du position sizer
        
        Returns:
            Dict avec toutes les stats
        """
        return {
            'capital': self.capital,
            'total_exposure': self.total_exposure,
            'exposure_pct': self.total_exposure / self.capital if self.capital > 0 else 0,
            'open_positions': len(self.open_positions),
            'current_drawdown': self.current_drawdown,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_trades': len(self.trade_history),
            'risk_per_trade': self.risk_per_trade
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du position sizer"""
    
    # Configuration
    config = {
        'initial_capital': 1000,
        'risk_per_trade': 0.02,  # 2%
        'max_position_size': 0.25,  # 25% max
        'min_position_size': 50,
        'kelly_fraction': 0.25,
        'use_dynamic_sizing': True
    }
    
    # Initialiser
    sizer = PositionSizer(config)
    
    # Test 1: Signal basique
    print("=" * 50)
    print("TEST 1: Signal basique")
    
    signal = {
        'symbol': 'BTCUSDC',
        'side': 'BUY',
        'confidence': 0.75
    }
    
    result = sizer.calculate_position_size(
        signal=signal,
        current_price=50000,
        stop_loss_price=49000  # Stop ÃƒÂ  -2%
    )
    
    print(f"Position size: ${result['position_size_usdc']:.2f}")
    print(f"Risk amount: ${result['risk_amount']:.2f}")
    print(f"Risk percent: {result['risk_percent']:.2%}")
    print(f"Method: {result['method_used']}")
    
    # Test 2: Avec conditions de marchÃƒÂ©
    print("\n" + "=" * 50)
    print("TEST 2: Avec conditions de marchÃƒÂ©")
    
    market_conditions = {
        'volatility': 0.03,  # 3% volatilitÃƒÂ©
        'trend_strength': 0.8,
        'trend_direction': 'BUY'
    }
    
    result = sizer.calculate_position_size(
        signal=signal,
        current_price=50000,
        stop_loss_price=49500,  # Stop plus serrÃƒÂ©
        market_conditions=market_conditions
    )
    
    print(f"Position size: ${result['position_size_usdc']:.2f}")
    print(f"MÃƒÂ©thodes calculÃƒÂ©es: {result.get('methods_calculated', {})}")
    
    # Test 3: AprÃƒÂ¨s plusieurs trades
    print("\n" + "=" * 50)
    print("TEST 3: AprÃƒÂ¨s historique de trades")
    
    # Simuler des trades
    for i in range(10):
        profit = np.random.randn() * 0.02  # Ã‚Â±2%
        sizer.trade_history.append({
            'profit_pct': profit
        })
    
    sizer._update_stats()
    
    result = sizer.calculate_position_size(
        signal=signal,
        current_price=50000,
        stop_loss_price=49000
    )
    
    stats = sizer.get_stats()
    print(f"Win rate: {stats['win_rate']:.1%}")
    print(f"Position finale: ${result['position_size_usdc']:.2f}")
