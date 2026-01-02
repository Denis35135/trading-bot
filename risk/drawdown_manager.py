"""
Drawdown Manager
Gere et surveille le drawdown du portfolio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DrawdownManager:
    """
    Gestionnaire de drawdown
    
    Fonctionnalites:
    - Calcul du drawdown actuel
    - Historique des drawdowns
    - Detection des periodes de recuperation
    - Alertes sur drawdown excessif
    - Ajustement automatique du risque
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le drawdown manager
        
        Args:
            config: Configuration
        """
        default_config = {
            'max_drawdown_pct': 0.08,  # 8% maximum
            'warning_drawdown_pct': 0.05,  # 5% warning
            'recovery_target_pct': 0.02,  # Objectif: revenir sous 2%
            'max_recovery_days': 7,  # Max 7 jours pour recuperer
            'risk_reduction_threshold': 0.06  # Reduire risque  6% DD
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # Historique
        self.equity_curve = []
        self.drawdown_history = []
        self.peak_equity = 0
        self.current_drawdown = 0
        
        # Periodes de drawdown
        self.in_drawdown = False
        self.drawdown_start_time = None
        self.drawdown_start_equity = 0
        
        # Stats
        self.stats = {
            'max_drawdown_ever': 0,
            'total_drawdown_periods': 0,
            'avg_recovery_time': 0,
            'longest_drawdown_days': 0,
            'current_drawdown_days': 0
        }
        
        logger.info(""" Drawdown Manager initialise")
        logger.info(f"   Max drawdown: {self.config['max_drawdown_pct']:.1%}")
    
    def update(self, current_equity: float, timestamp: datetime = None) -> Dict:
        """
        Met  jour le drawdown avec la nouvelle equity
        
        Args:
            current_equity: Equity actuelle
            timestamp: Timestamp (maintenant si None)
            
        Returns:
            Dict avec metriques de drawdown
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ajouter  l'historique
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity
        })
        
        # Mettre  jour le peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
            # Si on etait en drawdown et qu'on revient au peak, fin du drawdown
            if self.in_drawdown:
                self._end_drawdown_period(timestamp)
        
        # Calculer le drawdown actuel
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0
        
        # Mettre  jour les stats
        if self.current_drawdown > self.stats['max_drawdown_ever']:
            self.stats['max_drawdown_ever'] = self.current_drawdown
        
        # Detecter debut de drawdown
        if not self.in_drawdown and self.current_drawdown > 0.01:  # >1%
            self._start_drawdown_period(current_equity, timestamp)
        
        # Si en drawdown, mettre  jour la duree
        if self.in_drawdown:
            days = (timestamp - self.drawdown_start_time).days
            self.stats['current_drawdown_days'] = days
            
            if days > self.stats['longest_drawdown_days']:
                self.stats['longest_drawdown_days'] = days
        
        # Enregistrer dans l'historique
        self.drawdown_history.append({
            'timestamp': timestamp,
            'drawdown': self.current_drawdown,
            'equity': current_equity,
            'peak': self.peak_equity
        })
        
        # Analyse
        analysis = self._analyze_drawdown()
        
        return analysis
    
    def _start_drawdown_period(self, equity: float, timestamp: datetime):
        """Demarre une nouvelle periode de drawdown"""
        self.in_drawdown = True
        self.drawdown_start_time = timestamp
        self.drawdown_start_equity = equity
        
        logger.info(f""" Debut de drawdown  {equity:.2f}")
    
    def _end_drawdown_period(self, timestamp: datetime):
        """Termine une periode de drawdown"""
        if not self.in_drawdown:
            return
        
        duration = (timestamp - self.drawdown_start_time).days
        recovery_pct = (self.peak_equity - self.drawdown_start_equity) / self.drawdown_start_equity
        
        logger.info(f"""| Fin de drawdown apres {duration} jours")
        logger.info(f"   Recuperation: {recovery_pct:.2%}")
        
        # Mettre  jour les stats
        self.stats['total_drawdown_periods'] += 1
        
        # Calculer la moyenne du temps de recuperation
        if self.stats['total_drawdown_periods'] > 0:
            current_avg = self.stats['avg_recovery_time']
            self.stats['avg_recovery_time'] = (
                (current_avg * (self.stats['total_drawdown_periods'] - 1) + duration) /
                self.stats['total_drawdown_periods']
            )
        
        self.in_drawdown = False
        self.drawdown_start_time = None
        self.stats['current_drawdown_days'] = 0
    
    def _analyze_drawdown(self) -> Dict:
        """
        Analyse le drawdown actuel et retourne des recommandations
        
        Returns:
            Dict avec analyse et actions
        """
        analysis = {
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'in_drawdown': self.in_drawdown,
            'level': 'ok'
        }
        
        # Determiner le niveau de gravite
        if self.current_drawdown >= self.config['max_drawdown_pct']:
            analysis['level'] = 'critical'
            analysis['action'] = 'halt_trading'
            analysis['message'] = f'CRITIQUE: Drawdown {self.current_drawdown:.2%} >= limite {self.config["max_drawdown_pct"]:.2%}'
            
        elif self.current_drawdown >= self.config['risk_reduction_threshold']:
            analysis['level'] = 'high'
            analysis['action'] = 'reduce_risk'
            analysis['risk_multiplier'] = 0.5  # Reduire de 50%
            analysis['message'] = f'"LEV": Drawdown {self.current_drawdown:.2%} - Reduire le risque'
            
        elif self.current_drawdown >= self.config['warning_drawdown_pct']:
            analysis['level'] = 'warning'
            analysis['action'] = 'monitor_closely'
            analysis['risk_multiplier'] = 0.7  # Reduire de 30%
            analysis['message'] = f'ATTENTION: Drawdown {self.current_drawdown:.2%}'
            
        else:
            analysis['level'] = 'ok'
            analysis['action'] = 'continue'
            analysis['risk_multiplier'] = 1.0
            analysis['message'] = 'Drawdown dans les limites normales'
        
        # Verifier la duree du drawdown
        if self.in_drawdown:
            days = self.stats['current_drawdown_days']
            analysis['drawdown_days'] = days
            
            if days > self.config['max_recovery_days']:
                analysis['prolonged_drawdown'] = True
                analysis['message'] += f' | Drawdown prolonge: {days} jours'
        
        return analysis
    
    def get_risk_adjustment_multiplier(self) -> float:
        """
        Retourne le multiplicateur d'ajustement du risque base sur le drawdown
        
        Returns:
            Multiplicateur (0.5  1.0)
        """
        if self.current_drawdown >= self.config['max_drawdown_pct']:
            return 0.0  # Arret complet
        elif self.current_drawdown >= self.config['risk_reduction_threshold']:
            return 0.5  # Reduction de 50%
        elif self.current_drawdown >= self.config['warning_drawdown_pct']:
            return 0.7  # Reduction de 30%
        else:
            return 1.0  # Risque normal
    
    def get_drawdown_curve(self, period_days: int = 30) -> pd.DataFrame:
        """
        Retourne la courbe de drawdown sur une periode
        
        Args:
            period_days: Nombre de jours
            
        Returns:
            DataFrame avec historique
        """
        if not self.drawdown_history:
            return pd.DataFrame()
        
        cutoff = datetime.now() - timedelta(days=period_days)
        
        recent = [
            d for d in self.drawdown_history
            if d['timestamp'] >= cutoff
        ]
        
        df = pd.DataFrame(recent)
        if len(df) > 0:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def calculate_max_drawdown(self, equity_curve: List[float] = None) -> float:
        """
        Calcule le drawdown maximum
        
        Args:
            equity_curve: Courbe d'equity custom (utilise l'historique si None)
            
        Returns:
            Max drawdown
        """
        if equity_curve is None:
            equity_curve = [d['equity'] for d in self.equity_curve]
        
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_underwater_time(self) -> Tuple[int, float]:
        """
        Calcule le temps passe en drawdown
        
        Returns:
            Tuple (jours totaux, pourcentage du temps)
        """
        if not self.equity_curve:
            return 0, 0.0
        
        underwater_days = 0
        total_days = (self.equity_curve[-1]['timestamp'] - self.equity_curve[0]['timestamp']).days
        
        current_peak = 0
        for point in self.equity_curve:
            if point['equity'] > current_peak:
                current_peak = point['equity']
            elif point['equity'] < current_peak:
                underwater_days += 1
        
        underwater_pct = underwater_days / total_days if total_days > 0 else 0
        
        return underwater_days, underwater_pct
    
    def get_recovery_metrics(self) -> Dict:
        """
        Retourne les metriques de recuperation
        
        Returns:
            Dict avec metriques
        """
        if not self.in_drawdown:
            return {
                'in_recovery': False,
                'message': 'Pas en drawdown'
            }
        
        days_in_drawdown = self.stats['current_drawdown_days']
        recovery_needed = self.current_drawdown * self.peak_equity
        
        # Estimer le temps de recuperation base sur la moyenne historique
        estimated_days = self.stats['avg_recovery_time'] if self.stats['avg_recovery_time'] > 0 else 7
        
        return {
            'in_recovery': True,
            'days_in_drawdown': days_in_drawdown,
            'current_drawdown_pct': self.current_drawdown,
            'recovery_needed_dollars': recovery_needed,
            'estimated_recovery_days': estimated_days,
            'is_prolonged': days_in_drawdown > self.config['max_recovery_days']
        }
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques completes"""
        underwater_days, underwater_pct = self.calculate_underwater_time()
        
        return {
            **self.stats,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'in_drawdown': self.in_drawdown,
            'underwater_days': underwater_days,
            'underwater_pct': underwater_pct,
            'equity_points': len(self.equity_curve)
        }
    
    def reset(self):
        """Reinitialise le drawdown manager"""
        self.equity_curve = []
        self.drawdown_history = []
        self.peak_equity = 0
        self.current_drawdown = 0
        self.in_drawdown = False
        
        logger.info(""" Drawdown Manager reinitialise")


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Drawdown Manager"""
    
    manager = DrawdownManager()
    
    print("Test Drawdown Manager")
    print("=" * 50)
    
    # Simuler une courbe d'equity avec drawdown
    equity_curve = [
        10000, 10200, 10500, 10300,  # Hausse puis legere baisse
        10100, 9800, 9500, 9300,     # Drawdown
        9600, 9900, 10200, 10500,    # Recuperation
        10800, 11000                  # Nouveau peak
    ]
    
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(equity_curve))]
    
    print("\n1. Simulation d'equity curve:")
    for i, (equity, ts) in enumerate(zip(equity_curve, timestamps)):
        analysis = manager.update(equity, ts)
        
        if i % 4 == 0:  # Afficher tous les 4 points
            print(f"\n   Point {i}:")
            print(f"   Equity: ${equity:,.0f}")
            print(f"   Drawdown: {analysis['current_drawdown']:.2%}")
            print(f"   Level: {analysis['level']}")
            print(f"   Action: {analysis['action']}")
    
    print("\n2. Metriques de recuperation:")
    recovery = manager.get_recovery_metrics()
    for key, value in recovery.items():
        print(f"   {key}: {value}")
    
    print("\n3. Statistiques:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n4. Ajustement du risque:")
    multiplier = manager.get_risk_adjustment_multiplier()
    print(f"   Multiplicateur de risque: {multiplier:.1%}")
