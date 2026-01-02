"""
Metrics Collector pour The Bot
Collecte et stocke toutes les metriques de performance en temps reel
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de metriques collectees"""
    CAPITAL = "capital"
    PNL = "pnl"
    POSITION = "position"
    TRADE = "trade"
    RISK = "risk"
    EXECUTION = "execution"
    STRATEGY = "strategy"
    SYSTEM = "system"


@dataclass
class Metric:
    """Represente une metrique collectee"""
    timestamp: datetime
    type: MetricType
    name: str
    value: float
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['type'] = self.type.value
        return data


class MetricsCollector:
    """
    Collecteur de metriques central
    
    Responsabilites:
    - Collecter toutes les metriques du bot
    - Stocker l'historique (buffer circulaire)
    - Calculer des metriques derivees
    - Fournir des snapshots pour reporting
    - Detecter les anomalies
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le collecteur de metriques
        
        Args:
            config: Configuration du collecteur
        """
        self.config = config
        
        # Buffers circulaires par type de metrique (optimise memoire)
        self.buffer_size = getattr(config, 'BUFFER_SIZE', 10000)
        self.metrics_buffer = {
            metric_type: deque(maxlen=self.buffer_size)
            for metric_type in MetricType
        }
        
        # Agregations rapides (derniere minute/heure/jour)
        self.aggregations = {
            '1m': defaultdict(list),   # 1 minute
            '5m': defaultdict(list),   # 5 minutes
            '1h': defaultdict(list),   # 1 heure
            '1d': defaultdict(list),   # 1 jour
        }
        
        # Timestamps de derniere agregation
        self.last_aggregation = {
            '1m': datetime.now(),
            '5m': datetime.now(),
            '1h': datetime.now(),
            '1d': datetime.now(),
        }
        
        # Statistiques en temps reel
        self.stats = {
            'total_metrics_collected': 0,
            'metrics_per_second': 0,
            'last_collection_time': None,
            'anomalies_detected': 0
        }
        
        # Seuils d'anomalie
        self.anomaly_thresholds = getattr(config, 'ANOMALY_THRESHOLDS', {
            'latency_ms': 1000,
            'pnl_spike_percent': 0.05,
            'drawdown_increase_percent': 0.02
        })
        
        logger.info(f"""| Metrics Collector initialise (buffer: {self.buffer_size})")
    
    def record(self, 
    """
    metric_type: MetricType,
    """
               name: str, 
               value: float,
               metadata: Optional[Dict] = None):
        """
        Enregistre une metrique
        
        Args:
            metric_type: Type de metrique
            name: Nom de la metrique
            value: Valeur
            metadata: Metadonnees optionnelles
        """
        try:
            metric = Metric(
                timestamp=datetime.now(),
                type=metric_type,
                name=name,
                value=value,
                metadata=metadata or {}
            )
            
            # Ajouter au buffer
            self.metrics_buffer[metric_type].append(metric)
            
            # Ajouter aux agregations
            for interval in self.aggregations:
                self.aggregations[interval][name].append(value)
            
            # Statistiques
            self.stats['total_metrics_collected'] += 1
            self.stats['last_collection_time'] = datetime.now()
            
            # Detection d'anomalie
            self._check_anomaly(metric)
            
        except Exception as e:
            logger.error(f"Erreur enregistrement metrique {name}: {e}")
    
    def record_capital(self, capital: float, metadata: Optional[Dict] = None):
        """Enregistre le capital actuel"""
        self.record(MetricType.CAPITAL, 'total_capital', capital, metadata)
    
    def record_pnl(self, pnl: float, pnl_type: str = 'total', metadata: Optional[Dict] = None):
        """Enregistre le P&L"""
        meta = metadata or {}
        meta['pnl_type'] = pnl_type
        self.record(MetricType.PNL, f'pnl_{pnl_type}', pnl, meta)
    
    def record_position(self, 
                       symbol: str, 
                       side: str, 
                       size_usdc: float,
                       metadata: Optional[Dict] = None):
        """Enregistre une position"""
        meta = metadata or {}
        meta.update({'symbol': symbol, 'side': side})
        self.record(MetricType.POSITION, f'position_{symbol}', size_usdc, meta)
    
    def record_trade(self,
                    symbol: str,
                    side: str,
                    quantity: float,
                    price: float,
                    profit: Optional[float] = None,
                    metadata: Optional[Dict] = None):
        """Enregistre un trade"""
        meta = metadata or {}
        meta.update({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'profit': profit
        })
        self.record(MetricType.TRADE, 'trade_executed', quantity * price, meta)
    
    def record_risk_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        """Enregistre une metrique de risque"""
        self.record(MetricType.RISK, name, value, metadata)
    
    def record_execution_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        """Enregistre une metrique d'execution"""
        self.record(MetricType.EXECUTION, name, value, metadata)
    
    def record_strategy_metric(self, 
                               strategy: str, 
                               name: str, 
                               value: float,
                               metadata: Optional[Dict] = None):
        """Enregistre une metrique de strategie"""
        meta = metadata or {}
        meta['strategy'] = strategy
        self.record(MetricType.STRATEGY, f'{strategy}_{name}', value, meta)
    
    def record_system_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        """Enregistre une metrique systeme"""
        self.record(MetricType.SYSTEM, name, value, metadata)
    
    def get_latest(self, metric_type: MetricType, name: str) -> Optional[Metric]:
        """
        Recupere la derniere metrique d'un type donne
        
        Args:
            metric_type: Type de metrique
            name: Nom de la metrique
            
        Returns:
            Derniere metrique ou None
        """
        buffer = self.metrics_buffer[metric_type]
        for metric in reversed(buffer):
            if metric.name == name:
                return metric
        return None
    
    def get_range(self,
                 metric_type: MetricType,
                 name: str,
                 start_time: datetime,
                 end_time: Optional[datetime] = None) -> List[Metric]:
        """
        Recupere les metriques dans une plage de temps
        
        Args:
            metric_type: Type de metrique
            name: Nom de la metrique
            start_time: Debut de la plage
            end_time: Fin de la plage (defaut: maintenant)
            
        Returns:
            Liste des metriques dans la plage
        """
        end_time = end_time or datetime.now()
        buffer = self.metrics_buffer[metric_type]
        
        return [
            metric for metric in buffer
            if metric.name == name and start_time <= metric.timestamp <= end_time
        ]
    
    def calculate_statistics(self,
                            metric_type: MetricType,
                            name: str,
                            window_minutes: int = 60) -> Dict:
        """
        Calcule des statistiques sur une metrique
        
        Args:
            metric_type: Type de metrique
            name: Nom de la metrique
            window_minutes: Fenetre de temps en minutes
            
        Returns:
            Dict avec les statistiques
        """
        start_time = datetime.now() - timedelta(minutes=window_minutes)
        metrics = self.get_range(metric_type, name, start_time)
        
        if not metrics:
            return {
                'count': 0,
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'last': 0
            }
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1] if values else 0,
            'median': np.median(values),
            'p95': np.percentile(values, 95) if len(values) > 1 else values[0]
        }
    
    def get_snapshot(self) -> Dict:
        """
        Retourne un snapshot complet des metriques actuelles
        
        Returns:
            Dict avec toutes les metriques importantes
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats.copy(),
            'metrics': {}
        }
        
        # Dernieres valeurs par type
        for metric_type in MetricType:
            buffer = self.metrics_buffer[metric_type]
            if buffer:
                last_metric = buffer[-1]
                snapshot['metrics'][metric_type.value] = {
                    'last_value': last_metric.value,
                    'last_timestamp': last_metric.timestamp.isoformat(),
                    'count_1h': len([m for m in buffer 
                                    if m.timestamp > datetime.now() - timedelta(hours=1)])
                }
        
        # Agregations recentes
        snapshot['aggregations'] = {}
        for interval, data in self.aggregations.items():
            snapshot['aggregations'][interval] = {
                name: {
                    'count': len(values),
                    'mean': np.mean(values) if values else 0,
                    'last': values[-1] if values else 0
                }
                for name, values in data.items()
            }
        
        return snapshot
    
    def get_performance_summary(self) -> Dict:
        """
        Retourne un resume des performances
        
        Returns:
            Dict avec les metriques de performance cles
        """
        # Capital
        capital_stats = self.calculate_statistics(MetricType.CAPITAL, 'total_capital', 1440)  # 24h
        
        # P&L
        pnl_stats = self.calculate_statistics(MetricType.PNL, 'pnl_total', 1440)
        
        # Trades
        trades_1h = len(self.get_range(
            MetricType.TRADE, 
            'trade_executed',
            datetime.now() - timedelta(hours=1)
        ))
        
        trades_24h = len(self.get_range(
            MetricType.TRADE,
            'trade_executed',
            datetime.now() - timedelta(hours=24)
        ))
        
        # Risk
        drawdown_latest = self.get_latest(MetricType.RISK, 'current_drawdown')
        
        return {
            'capital': {
                'current': capital_stats['last'],
                'peak_24h': capital_stats['max'],
                'low_24h': capital_stats['min'],
                'change_24h': capital_stats['last'] - capital_stats['mean']
            },
            'pnl': {
                'total': pnl_stats['last'],
                'mean_24h': pnl_stats['mean'],
                'max_24h': pnl_stats['max'],
                'min_24h': pnl_stats['min']
            },
            'trading': {
                'trades_1h': trades_1h,
                'trades_24h': trades_24h,
                'avg_per_hour': trades_24h / 24 if trades_24h > 0 else 0
            },
            'risk': {
                'current_drawdown': drawdown_latest.value if drawdown_latest else 0
            }
        }
    
    def _check_anomaly(self, metric: Metric):
        """
        Verifie si une metrique est anormale
        
        Args:
            metric: Metrique  verifier
        """
        try:
            # Verifier selon le type de metrique
            if metric.type == MetricType.EXECUTION:
                if metric.name == 'order_latency_ms' and metric.value > self.anomaly_thresholds['latency_ms']:
                    logger.warning(f" Anomalie detectee: latence elevee {metric.value:.0f}ms")
                    self.stats['anomalies_detected'] += 1
            
            elif metric.type == MetricType.PNL:
                # Verifier les variations brusques
                recent = self.get_range(
                    metric.type,
                    metric.name,
                    datetime.now() - timedelta(minutes=5)
                )
                if len(recent) > 1:
                    prev_value = recent[-2].value
                    if prev_value != 0:
                        change_pct = abs((metric.value - prev_value) / prev_value)
                        if change_pct > self.anomaly_thresholds['pnl_spike_percent']:
                            logger.warning(f" Anomalie detectee: spike P&L {change_pct:.2%}")
                            self.stats['anomalies_detected'] += 1
            
            elif metric.type == MetricType.RISK:
                if metric.name == 'current_drawdown':
                    recent = self.get_range(
                        metric.type,
                        metric.name,
                        datetime.now() - timedelta(minutes=5)
                    )
                    if len(recent) > 1:
                        prev_value = recent[-2].value
                        change = metric.value - prev_value
                        if change > self.anomaly_thresholds['drawdown_increase_percent']:
                            logger.warning(f" Anomalie detectee: augmentation drawdown {change:.2%}")
                            self.stats['anomalies_detected'] += 1
        
        except Exception as e:
            logger.error(f"Erreur verification anomalie: {e}")
    
    def cleanup_old_data(self, days: int = 7):
        """
        Nettoie les donnees anciennes
        
        Args:
            days: Nombre de jours  conserver
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        for metric_type in MetricType:
            buffer = self.metrics_buffer[metric_type]
            original_size = len(buffer)
            
            # Filtrer les metriques anciennes
            # Note: deque ne supporte pas de filtrage direct, on recree
            new_buffer = deque(
                (m for m in buffer if m.timestamp > cutoff_time),
                maxlen=self.buffer_size
            )
            
            self.metrics_buffer[metric_type] = new_buffer
            cleaned += original_size - len(new_buffer)
        
        if cleaned > 0:
            logger.info(f" Nettoyage: {cleaned} metriques anciennes supprimees")
    
    def export_to_dict(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Dict:
        """
        Exporte les metriques en dictionnaire
        
        Args:
            start_time: Debut de la periode (defaut: tout)
            end_time: Fin de la periode (defaut: maintenant)
            
        Returns:
            Dict avec toutes les metriques
        """
        end_time = end_time or datetime.now()
        start_time = start_time or datetime.now() - timedelta(days=1)
        
        export = {
            'export_time': datetime.now().isoformat(),
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'metrics': {}
        }
        
        for metric_type in MetricType:
            buffer = self.metrics_buffer[metric_type]
            filtered = [
                m for m in buffer
                if start_time <= m.timestamp <= end_time
            ]
            
            export['metrics'][metric_type.value] = [
                m.to_dict() for m in filtered
            ]
        
        return export


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du metrics collector"""
    
    # Configuration de test
    config = {
        'buffer_size': 1000,
        'anomaly_thresholds': {
            'latency_ms': 500,
            'pnl_spike_percent': 0.03,
            'drawdown_increase_percent': 0.01
        }
    }
    
    collector = MetricsCollector(config)
    
    print("\n=== Test Metrics Collector ===\n")
    
    # Enregistrer quelques metriques
    collector.record_capital(1000.0)
    time.sleep(0.1)
    collector.record_pnl(50.0, 'daily')
    time.sleep(0.1)
    collector.record_trade('BTCUSDT', 'BUY', 0.01, 50000.0, profit=10.0)
    time.sleep(0.1)
    collector.record_risk_metric('current_drawdown', 0.02)
    
    # Snapshot
    snapshot = collector.get_snapshot()
    print(f"Snapshot: {snapshot['stats']}")
    
    # Performance summary
    summary = collector.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Capital: ${summary['capital']['current']:,.2f}")
    print(f"P&L: ${summary['pnl']['total']:+,.2f}")
    print(f"Trades 1h: {summary['trading']['trades_1h']}")
    
    print("\n""| Tests termines")

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Retourne les metriques actuelles
        
        Returns:
            Dict avec les metriques principales
        """
        try:
            return {
                'capital': getattr(self, 'current_capital', 0),
                'total_pnl': getattr(self, 'total_pnl', 0),
                'daily_pnl': getattr(self, 'daily_pnl', 0),
                'positions': getattr(self, 'active_positions', 0),
                'trades_today': getattr(self, 'trades_today', 0),
                'win_rate': getattr(self, 'win_rate', 0),
                'total_metrics': self.stats.get('total_metrics_collected', 0),
                'metrics_per_second': self.stats.get('metrics_per_second', 0),
            }
        except Exception as e:
            # Retour par defaut en cas d'erreur
            return {
                'capital': 0,
                'total_pnl': 0,
                'daily_pnl': 0,
                'positions': 0,
                'trades_today': 0,
                'win_rate': 0,
                'total_metrics': 0,
                'metrics_per_second': 0,
            }
