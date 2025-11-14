"""
Risk Monitor pour The Bot
Surveillance en temps rÃƒÂ©el et protection du capital
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Niveaux de risque"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class RiskMonitor:
    """
    Moniteur de risque en temps rÃƒÂ©el
    
    Surveille:
    - Drawdown (perte depuis le peak)
    - Exposition totale
    - CorrÃƒÂ©lations entre positions
    - VolatilitÃƒÂ© du portfolio
    - Circuit breakers multi-niveaux
    - Risk metrics (VaR, Sharpe, etc.)
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le risk monitor
        
        Args:
            config: Configuration avec:
                - max_drawdown: Drawdown maximum autorisÃƒÂ©
                - max_daily_loss: Perte max par jour
                - max_exposure: Exposition max du portfolio
                - circuit_breaker_levels: Niveaux de circuit breakers
                - correlation_threshold: Seuil de corrÃƒÂ©lation
        """
        self.config = config
        
        # Limites de risque
        self.max_drawdown = getattr(config, 'MAX_DRAWDOWN', 0.08)  # 8%
        self.max_daily_loss = getattr(config, 'MAX_DAILY_LOSS', 0.05)  # 5%
        self.max_exposure = getattr(config, 'MAX_EXPOSURE', 0.8)  # 80% du capital
        self.correlation_threshold = getattr(config, 'CORRELATION_THRESHOLD', 0.7)
        
        # Circuit breakers
        self.circuit_breaker_levels = getattr(config, 'CIRCUIT_BREAKER_LEVELS', {
            'WARNING': 0.03,    # 3% drawdown
            'REDUCE': 0.05,     # 5% drawdown
            'PAUSE': 0.07,      # 7% drawdown
            'EMERGENCY': 0.08   # 8% drawdown
        })
        
        # Ãƒâ€°tat du portfolio
        self.capital = getattr(config, 'INITIAL_CAPITAL', 1000)
        self.peak_capital = self.capital
        self.daily_starting_capital = self.capital
        self.positions = {}
        self.closed_trades_today = []
        
        # MÃƒÂ©triques
        self.current_drawdown = 0
        self.daily_pnl = 0
        self.total_pnl = 0
        self.current_risk_level = RiskLevel.NORMAL
        
        # Historique pour calculs
        self.pnl_history = []
        self.drawdown_history = []
        self.exposure_history = []
        
        # Ãƒâ€°tat des circuit breakers
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.position_reduction_factor = 1.0
        
        # Alertes
        self.alerts = []
        self.last_alert_time = {}
        
        logger.info("Risk Monitor initialisÃƒÂ©")
        logger.info(f"Max Drawdown: {self.max_drawdown:.1%}, Max Daily Loss: {self.max_daily_loss:.1%}")
    
    def update(self, current_capital: float, positions: Dict) -> Dict:
        """
        Mise ÃƒÂ  jour principale du risk monitor
        
        Args:
            current_capital: Capital actuel
            positions: Dict des positions ouvertes
            
        Returns:
            Dict avec l'ÃƒÂ©tat du risque et actions recommandÃƒÂ©es
        """
        try:
            # Mettre ÃƒÂ  jour le capital
            self.capital = current_capital
            self.positions = positions
            
            # Calculer toutes les mÃƒÂ©triques
            self._update_drawdown()
            self._update_daily_pnl()
            self._update_exposure()
            
            # VÃƒÂ©rifier les limites de risque
            risk_checks = self._check_all_risk_limits()
            
            # DÃƒÂ©terminer le niveau de risque
            self.current_risk_level = self._determine_risk_level(risk_checks)
            
            # Appliquer les circuit breakers si nÃƒÂ©cessaire
            actions = self._apply_circuit_breakers()
            
            # Calculer les mÃƒÂ©triques avancÃƒÂ©es
            advanced_metrics = self._calculate_advanced_metrics()
            
            # PrÃƒÂ©parer le rapport
            report = {
                'timestamp': datetime.now(),
                'risk_level': self.current_risk_level.value,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': self.daily_pnl / self.daily_starting_capital if self.daily_starting_capital > 0 else 0,
                'total_exposure': self._calculate_total_exposure(),
                'position_count': len(positions),
                'risk_checks': risk_checks,
                'required_actions': actions,
                'advanced_metrics': advanced_metrics,
                'alerts': self.alerts[-10:]  # 10 derniÃƒÂ¨res alertes
            }
            
            # Log si niveau ÃƒÂ©levÃƒÂ©
            if self.current_risk_level != RiskLevel.NORMAL:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Risk Level: {self.current_risk_level.value}")
                logger.warning(f"   Drawdown: {self.current_drawdown:.2%}")
                logger.warning(f"   Actions: {actions}")
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur mise ÃƒÂ  jour risk monitor: {e}")
            return {'error': str(e), 'risk_level': RiskLevel.EMERGENCY.value}
    
    def _update_drawdown(self):
        """Calcule le drawdown actuel"""
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        
        # Historique
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'drawdown': self.current_drawdown,
            'capital': self.capital,
            'peak': self.peak_capital
        })
        
        # Garder 1000 derniers points
        if len(self.drawdown_history) > 1000:
            self.drawdown_history.pop(0)
    
    def _update_daily_pnl(self):
        """Calcule le P&L du jour"""
        # Reset ÃƒÂ  minuit
        now = datetime.now()
        if hasattr(self, 'last_pnl_update'):
            if now.date() > self.last_pnl_update.date():
                self.daily_starting_capital = self.capital
                self.daily_pnl = 0
                self.closed_trades_today = []
        
        self.last_pnl_update = now
        
        # Calculer P&L
        self.daily_pnl = self.capital - self.daily_starting_capital
        
        # Ajouter ÃƒÂ  l'historique
        self.pnl_history.append({
            'timestamp': now,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl
        })
    
    def _update_exposure(self):
        """Calcule l'exposition totale"""
        total_exposure = self._calculate_total_exposure()
        
        self.exposure_history.append({
            'timestamp': datetime.now(),
            'exposure': total_exposure,
            'exposure_pct': total_exposure / self.capital if self.capital > 0 else 0
        })
        
        # Garder 1000 derniers points
        if len(self.exposure_history) > 1000:
            self.exposure_history.pop(0)
    
    def _calculate_total_exposure(self) -> float:
        """Calcule l'exposition totale en USDC"""
        if not self.positions:
            return 0
        
        total = sum(pos.get('size_usdc', 0) for pos in self.positions.values())
        return total
    
    def _check_all_risk_limits(self) -> Dict:
        """
        VÃƒÂ©rifie toutes les limites de risque
        
        Returns:
            Dict avec statut de chaque vÃƒÂ©rification
        """
        checks = {}
        
        # 1. Drawdown
        checks['drawdown'] = {
            'value': self.current_drawdown,
            'limit': self.max_drawdown,
            'exceeded': self.current_drawdown >= self.max_drawdown,
            'severity': self._get_severity(self.current_drawdown, self.max_drawdown)
        }
        
        # 2. Daily Loss
        daily_loss_pct = abs(self.daily_pnl / self.daily_starting_capital) if self.daily_starting_capital > 0 else 0
        checks['daily_loss'] = {
            'value': daily_loss_pct if self.daily_pnl < 0 else 0,
            'limit': self.max_daily_loss,
            'exceeded': self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss,
            'severity': self._get_severity(daily_loss_pct, self.max_daily_loss) if self.daily_pnl < 0 else 'LOW'
        }
        
        # 3. Exposure
        exposure = self._calculate_total_exposure()
        exposure_pct = exposure / self.capital if self.capital > 0 else 0
        checks['exposure'] = {
            'value': exposure_pct,
            'limit': self.max_exposure,
            'exceeded': exposure_pct >= self.max_exposure,
            'severity': self._get_severity(exposure_pct, self.max_exposure)
        }
        
        # 4. Position Correlation
        max_correlation = self._calculate_max_correlation()
        checks['correlation'] = {
            'value': max_correlation,
            'limit': self.correlation_threshold,
            'exceeded': max_correlation >= self.correlation_threshold,
            'severity': 'HIGH' if max_correlation >= 0.8 else 'MEDIUM' if max_correlation >= 0.6 else 'LOW'
        }
        
        # 5. Position Concentration
        concentration = self._calculate_position_concentration()
        checks['concentration'] = {
            'value': concentration,
            'limit': 0.30,  # 30% max dans une position
            'exceeded': concentration >= 0.30,
            'severity': 'HIGH' if concentration >= 0.40 else 'MEDIUM' if concentration >= 0.30 else 'LOW'
        }
        
        # 6. Volatility Check
        portfolio_vol = self._estimate_portfolio_volatility()
        checks['volatility'] = {
            'value': portfolio_vol,
            'limit': 0.05,  # 5% volatilitÃƒÂ© max
            'exceeded': portfolio_vol >= 0.05,
            'severity': 'HIGH' if portfolio_vol >= 0.07 else 'MEDIUM' if portfolio_vol >= 0.05 else 'LOW'
        }
        
        return checks
    
    def _get_severity(self, value: float, limit: float) -> str:
        """DÃƒÂ©termine la sÃƒÂ©vÃƒÂ©ritÃƒÂ© d'un dÃƒÂ©passement"""
        ratio = value / limit if limit > 0 else 0
        
        if ratio >= 1.0:
            return 'CRITICAL'
        elif ratio >= 0.9:
            return 'HIGH'
        elif ratio >= 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _determine_risk_level(self, risk_checks: Dict) -> RiskLevel:
        """
        DÃƒÂ©termine le niveau de risque global
        
        Args:
            risk_checks: RÃƒÂ©sultats des vÃƒÂ©rifications
            
        Returns:
            RiskLevel
        """
        # Compter les violations
        critical_count = sum(1 for check in risk_checks.values() 
                           if check.get('severity') == 'CRITICAL')
        high_count = sum(1 for check in risk_checks.values() 
                        if check.get('severity') == 'HIGH')
        exceeded_count = sum(1 for check in risk_checks.values() 
                           if check.get('exceeded', False))
        
        # DÃƒÂ©terminer le niveau
        if critical_count >= 2 or self.current_drawdown >= self.max_drawdown:
            return RiskLevel.EMERGENCY
        elif critical_count >= 1 or high_count >= 2:
            return RiskLevel.CRITICAL
        elif high_count >= 1 or exceeded_count >= 2:
            return RiskLevel.HIGH
        elif exceeded_count >= 1 or self.current_drawdown >= self.max_drawdown * 0.5:
            return RiskLevel.WARNING
        else:
            return RiskLevel.NORMAL
    
    def _apply_circuit_breakers(self) -> List[str]:
        """
        Applique les circuit breakers selon le niveau de risque
        
        Returns:
            Liste des actions ÃƒÂ  prendre
        """
        actions = []
        
        # VÃƒÂ©rifier si circuit breaker dÃƒÂ©jÃƒÂ  actif
        if self.circuit_breaker_active:
            if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
                actions.append(f"CIRCUIT_BREAKER_ACTIVE until {self.circuit_breaker_until}")
                return actions
            else:
                # DÃƒÂ©sactiver le circuit breaker
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None
                self.position_reduction_factor = 1.0
                logger.info("Circuit breaker dÃƒÂ©sactivÃƒÂ©")
        
        # Appliquer selon le niveau
        if self.current_risk_level == RiskLevel.EMERGENCY:
            actions.extend([
                "CLOSE_ALL_POSITIONS",
                "HALT_TRADING_24H",
                "SEND_EMERGENCY_ALERT"
            ])
            self.circuit_breaker_active = True
            self.circuit_breaker_until = datetime.now() + timedelta(hours=24)
            self._send_alert("EMERGENCY", "ArrÃƒÂªt d'urgence! Toutes positions fermÃƒÂ©es.")
            
        elif self.current_risk_level == RiskLevel.CRITICAL:
            actions.extend([
                "CLOSE_LOSING_POSITIONS",
                "REDUCE_ALL_POSITIONS_50%",
                "HALT_NEW_TRADES_2H"
            ])
            self.position_reduction_factor = 0.5
            self.circuit_breaker_active = True
            self.circuit_breaker_until = datetime.now() + timedelta(hours=2)
            self._send_alert("CRITICAL", f"Niveau critique! DD: {self.current_drawdown:.2%}")
            
        elif self.current_risk_level == RiskLevel.HIGH:
            actions.extend([
                "REDUCE_POSITION_SIZES_30%",
                "CLOSE_WORST_POSITION",
                "PAUSE_NEW_TRADES_30MIN"
            ])
            self.position_reduction_factor = 0.7
            self.circuit_breaker_active = True
            self.circuit_breaker_until = datetime.now() + timedelta(minutes=30)
            
        elif self.current_risk_level == RiskLevel.WARNING:
            actions.extend([
                "REDUCE_NEW_POSITION_SIZES_20%",
                "TIGHTEN_STOP_LOSSES",
                "INCREASE_MONITORING"
            ])
            self.position_reduction_factor = 0.8
            
        return actions
    
    def _calculate_max_correlation(self) -> float:
        """
        Calcule la corrÃƒÂ©lation maximale entre positions
        
        Returns:
            CorrÃƒÂ©lation max (0 ÃƒÂ  1)
        """
        if len(self.positions) < 2:
            return 0
        
        # Simplified: utiliser les symboles pour estimer corrÃƒÂ©lation
        # En pratique, utiliser les prix historiques
        symbols = list(self.positions.keys())
        
        # Pour l'instant, estimation basique
        # MÃƒÂªme base = haute corrÃƒÂ©lation
        max_corr = 0
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                # Si mÃƒÂªme base asset (ex: BTC dans BTCUSDC et BTCETH)
                if symbols[i][:3] == symbols[j][:3]:
                    max_corr = max(max_corr, 0.8)
                # Si stablecoins
                elif 'USDC' in symbols[i] and 'USDC' in symbols[j]:
                    max_corr = max(max_corr, 0.3)
                else:
                    max_corr = max(max_corr, 0.5)  # DÃƒÂ©faut crypto
        
        return max_corr
    
    def _calculate_position_concentration(self) -> float:
        """
        Calcule la concentration de la plus grosse position
        
        Returns:
            % du capital dans la plus grosse position
        """
        if not self.positions:
            return 0
        
        max_position = max(pos.get('size_usdc', 0) for pos in self.positions.values())
        
        if self.capital <= 0:
            return 0
        
        return max_position / self.capital
    
    def _estimate_portfolio_volatility(self) -> float:
        """
        Estime la volatilitÃƒÂ© du portfolio
        
        Returns:
            VolatilitÃƒÂ© estimÃƒÂ©e (0 ÃƒÂ  1)
        """
        if len(self.pnl_history) < 20:
            return 0.02  # DÃƒÂ©faut 2%
        
        # Calculer la volatilitÃƒÂ© des returns rÃƒÂ©cents
        recent_pnls = [h['daily_pnl'] for h in self.pnl_history[-20:]]
        
        if self.capital <= 0:
            return 0.02
        
        returns = [pnl / self.capital for pnl in recent_pnls]
        volatility = np.std(returns) if len(returns) > 1 else 0.02
        
        return volatility
    
    def _calculate_advanced_metrics(self) -> Dict:
        """
        Calcule les mÃƒÂ©triques avancÃƒÂ©es (VaR, Sharpe, etc.)
        
        Returns:
            Dict des mÃƒÂ©triques avancÃƒÂ©es
        """
        metrics = {}
        
        # Value at Risk (95%)
        if len(self.pnl_history) >= 20:
            pnls = [h['daily_pnl'] for h in self.pnl_history[-100:]]
            metrics['var_95'] = np.percentile(pnls, 5) if pnls else 0
        else:
            metrics['var_95'] = -self.capital * 0.02  # DÃƒÂ©faut -2%
        
        # Sharpe Ratio (simplifiÃƒÂ©)
        if len(self.pnl_history) >= 30:
            returns = []
            for i in range(1, min(30, len(self.pnl_history))):
                if self.pnl_history[i-1]['total_pnl'] != 0:
                    ret = (self.pnl_history[i]['total_pnl'] - self.pnl_history[i-1]['total_pnl']) / abs(self.pnl_history[i-1]['total_pnl'])
                    returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                metrics['sharpe_ratio'] = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                metrics['sharpe_ratio'] = 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # Profit Factor
        if self.closed_trades_today:
            wins = sum(t['profit'] for t in self.closed_trades_today if t['profit'] > 0)
            losses = abs(sum(t['profit'] for t in self.closed_trades_today if t['profit'] < 0))
            metrics['profit_factor'] = wins / losses if losses > 0 else float('inf') if wins > 0 else 0
        else:
            metrics['profit_factor'] = 0
        
        # Recovery Factor (profit / max drawdown)
        if self.total_pnl > 0 and max([h['drawdown'] for h in self.drawdown_history] + [0.01]) > 0:
            max_dd = max(h['drawdown'] for h in self.drawdown_history)
            metrics['recovery_factor'] = self.total_pnl / (max_dd * self.peak_capital) if max_dd > 0 else 0
        else:
            metrics['recovery_factor'] = 0
        
        return metrics
    
    def approve_new_trade(self, signal: Dict, proposed_size: float) -> Tuple[bool, float, str]:
        """
        Approuve ou rejette un nouveau trade
        
        Args:
            signal: Signal de trading
            proposed_size: Taille proposÃƒÂ©e en USDC
            
        Returns:
            Tuple (approved, adjusted_size, reason)
        """
        # VÃƒÂ©rifier circuit breaker
        if self.circuit_breaker_active:
            return False, 0, "Circuit breaker actif"
        
        # VÃƒÂ©rifier niveau de risque
        if self.current_risk_level == RiskLevel.EMERGENCY:
            return False, 0, "Niveau d'urgence - trading arrÃƒÂªtÃƒÂ©"
        
        if self.current_risk_level == RiskLevel.CRITICAL:
            return False, 0, "Niveau critique - nouvelles positions interdites"
        
        # VÃƒÂ©rifier limites
        current_exposure = self._calculate_total_exposure()
        new_exposure = current_exposure + proposed_size
        exposure_pct = new_exposure / self.capital if self.capital > 0 else 1
        
        if exposure_pct > self.max_exposure:
            # Calculer taille max possible
            max_possible = max(0, (self.max_exposure * self.capital) - current_exposure)
            if max_possible < 50:  # Minimum trade size
                return False, 0, f"Exposition max atteinte ({exposure_pct:.1%})"
            else:
                return True, max_possible, f"Taille rÃƒÂ©duite pour exposition (max: ${max_possible:.2f})"
        
        # Appliquer rÃƒÂ©duction si nÃƒÂ©cessaire
        adjusted_size = proposed_size * self.position_reduction_factor
        
        if self.position_reduction_factor < 1.0:
            reason = f"Taille rÃƒÂ©duite de {(1-self.position_reduction_factor):.0%} (risk management)"
        else:
            reason = "ApprouvÃƒÂ©"
        
        return True, adjusted_size, reason
    
    def register_trade_close(self, symbol: str, entry_price: float, 
                           exit_price: float, size_usdc: float, side: str):
        """
        Enregistre la fermeture d'un trade
        
        Args:
            symbol: Symbole
            entry_price: Prix d'entrÃƒÂ©e
            exit_price: Prix de sortie
            size_usdc: Taille en USDC
            side: BUY ou SELL
        """
        # Calculer profit
        if side == 'BUY':
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price
        
        profit_usdc = size_usdc * profit_pct
        
        # Ajouter ÃƒÂ  l'historique du jour
        self.closed_trades_today.append({
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit_usdc,
            'profit_pct': profit_pct,
            'size': size_usdc,
            'timestamp': datetime.now()
        })
        
        # Mettre ÃƒÂ  jour P&L
        self.total_pnl += profit_usdc
        self.daily_pnl += profit_usdc
        
        logger.info(f"Trade fermÃƒÂ©: {symbol} P&L: ${profit_usdc:.2f} ({profit_pct:.2%})")
    
    def _send_alert(self, level: str, message: str):
        """Envoie une alerte"""
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        }
        
        self.alerts.append(alert)
        
        # Limiter ÃƒÂ  100 alertes
        if len(self.alerts) > 100:
            self.alerts.pop(0)
        
        # Log selon niveau
        if level == 'EMERGENCY':
            logger.critical(f"Ã°Å¸Å¡Â¨ {message}")
        elif level == 'CRITICAL':
            logger.error(f"Ã¢ÂÅ’ {message}")
        else:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â {message}")
    
    def get_risk_summary(self) -> str:
        """
        GÃƒÂ©nÃƒÂ¨re un rÃƒÂ©sumÃƒÂ© textuel du risque
        
        Returns:
            RÃƒÂ©sumÃƒÂ© formatÃƒÂ©
        """
        summary = f"""
Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
Ã¢â€¢â€˜                 RISK MONITOR SUMMARY                 Ã¢â€¢â€˜
Ã¢â€¢Â Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â£
Ã¢â€¢â€˜ Risk Level:     {self.current_risk_level.value:>37} Ã¢â€¢â€˜
Ã¢â€¢â€˜ Drawdown:       {self.current_drawdown:>36.2%} Ã¢â€¢â€˜
Ã¢â€¢â€˜ Daily P&L:      ${self.daily_pnl:>35,.2f} Ã¢â€¢â€˜
Ã¢â€¢â€˜ Exposure:       {self._calculate_total_exposure()/self.capital if self.capital > 0 else 0:>36.1%} Ã¢â€¢â€˜
Ã¢â€¢â€˜ Positions:      {len(self.positions):>37} Ã¢â€¢â€˜
Ã¢â€¢â€˜ Circuit Breaker: {'ACTIVE' if self.circuit_breaker_active else 'OFF':>36} Ã¢â€¢â€˜
Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
        """
        return summary


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du risk monitor"""
    
    # Configuration
    config = {
        'initial_capital': 1000,
        'max_drawdown': 0.08,
        'max_daily_loss': 0.05,
        'max_exposure': 0.8
    }
    
    # Initialiser
    monitor = RiskMonitor(config)
    
    # Simuler des positions
    positions = {
        'BTCUSDC': {'size_usdc': 200, 'entry_price': 50000},
        'ETHUSDC': {'size_usdc': 150, 'entry_price': 3000},
        'BNBUSDC': {'size_usdc': 100, 'entry_price': 400}
    }
    
    # Test 1: Ãƒâ€°tat normal
    print("=" * 60)
    print("TEST 1: Ãƒâ€°tat Normal")
    report = monitor.update(1000, positions)
    print(f"Risk Level: {report['risk_level']}")
    print(f"Drawdown: {report['current_drawdown']:.2%}")
    print(f"Exposure: {report['total_exposure']:.2f}")
    
    # Test 2: Drawdown warning
    print("\n" + "=" * 60)
    print("TEST 2: Drawdown Warning")
    report = monitor.update(970, positions)  # -3% loss
    print(f"Risk Level: {report['risk_level']}")
    print(f"Actions: {report['required_actions']}")
    
    # Test 3: Drawdown critique
    print("\n" + "=" * 60)
    print("TEST 3: Drawdown Critique")
    report = monitor.update(930, positions)  # -7% loss
    print(f"Risk Level: {report['risk_level']}")
    print(f"Actions: {report['required_actions']}")
    
    # Test 4: Approbation trade
    print("\n" + "=" * 60)
    print("TEST 4: Approbation nouveau trade")
    signal = {'symbol': 'ADAUSDC', 'confidence': 0.75}
    approved, size, reason = monitor.approve_new_trade(signal, 200)
    print(f"ApprouvÃƒÂ©: {approved}")
    print(f"Taille ajustÃƒÂ©e: ${size:.2f}")
    print(f"Raison: {reason}")
    
    # Afficher rÃƒÂ©sumÃƒÂ©
    print("\n" + monitor.get_risk_summary())