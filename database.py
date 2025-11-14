"""
Database Manager pour The Bot
Gestion de la persistance des donnÃƒÂ©es avec SQLAlchemy
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from sqlalchemy import func, desc

logger = logging.getLogger(__name__)

# Base pour les modÃƒÂ¨les
Base = declarative_base()


# ============================================================================
# MODÃƒË†LES DE BASE DE DONNÃƒâ€°ES
# ============================================================================

class Trade(Base):
    """Table des trades exÃƒÂ©cutÃƒÂ©s"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Informations trade
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    
    # Prix et quantitÃƒÂ©
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    
    # RÃƒÂ©sultats
    profit_usdc = Column(Float, default=0)
    profit_pct = Column(Float, default=0)
    fees = Column(Float, default=0)
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Contexte
    entry_reason = Column(String(100))
    exit_reason = Column(String(100))
    
    # Stop loss et take profit
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Statut
    status = Column(String(20), default='open')  # open/closed/cancelled
    
    # MÃƒÂ©tadonnÃƒÂ©es
    created_at = Column(DateTime, default=datetime.now)
    
    # Index composites pour performance
    __table_args__ = (
        Index('idx_symbol_entry_time', 'symbol', 'entry_time'),
        Index('idx_strategy_status', 'strategy', 'status'),
    )


class Position(Base):
    """Table des positions ouvertes"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identification
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    strategy = Column(String(50), nullable=False)
    
    # Position
    side = Column(String(10), nullable=False)  # LONG/SHORT
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trailing_stop = Column(Float)
    
    # Performance actuelle
    current_price = Column(Float)
    unrealized_pnl = Column(Float, default=0)
    unrealized_pnl_pct = Column(Float, default=0)
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    last_update = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # MÃƒÂ©tadonnÃƒÂ©es
    entry_reason = Column(String(200))
    
    created_at = Column(DateTime, default=datetime.now)


class PerformanceSnapshot(Base):
    """Snapshots de performance (toutes les 5 minutes)"""
    __tablename__ = 'performance_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Capital
    total_capital = Column(Float, nullable=False)
    available_capital = Column(Float, nullable=False)
    total_exposure = Column(Float, default=0)
    
    # Performance
    daily_pnl = Column(Float, default=0)
    daily_pnl_pct = Column(Float, default=0)
    total_pnl = Column(Float, default=0)
    total_pnl_pct = Column(Float, default=0)
    
    # Trades
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    
    # Positions
    open_positions = Column(Integer, default=0)
    
    # Risk metrics
    current_drawdown = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)


class StrategyPerformance(Base):
    """Performance par stratÃƒÂ©gie"""
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # StratÃƒÂ©gie
    strategy_name = Column(String(50), nullable=False, index=True)
    
    # Performance
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    
    total_profit = Column(Float, default=0)
    average_win = Column(Float, default=0)
    average_loss = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    
    sharpe_ratio = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    
    # Best/Worst
    best_trade_pct = Column(Float, default=0)
    worst_trade_pct = Column(Float, default=0)
    
    # Streaks
    current_streak = Column(Integer, default=0)
    best_streak = Column(Integer, default=0)
    worst_streak = Column(Integer, default=0)
    
    # Allocation
    allocation_pct = Column(Float, default=0)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    last_trade_time = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    created_at = Column(DateTime, default=datetime.now)


class SystemLog(Base):
    """Logs systÃƒÂ¨me importants"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    level = Column(String(20), nullable=False, index=True)  # INFO/WARNING/ERROR
    category = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    
    # Contexte
    symbol = Column(String(20))
    strategy = Column(String(50))
    
    # DonnÃƒÂ©es supplÃƒÂ©mentaires (JSON)
    extra_data = Column(Text)  # JSON serialized
    
    timestamp = Column(DateTime, default=datetime.now, index=True)


# ============================================================================
# GESTIONNAIRE DE BASE DE DONNÃƒâ€°ES
# ============================================================================

class DatabaseManager:
    """
    Gestionnaire central de la base de donnÃƒÂ©es
    
    ResponsabilitÃƒÂ©s:
    - Connexion et initialisation
    - CRUD operations
    - RequÃƒÂªtes optimisÃƒÂ©es
    - Backup et restore
    - Maintenance
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le database manager
        
        Args:
            config: Configuration de la base de donnÃƒÂ©es
        """
        self.config = config
        self.db_path = getattr(config, 'DB_PATH', 'data/bot.db')
        self.db_type = getattr(config, 'DB_TYPE', 'sqlite')  # sqlite ou postgresql
        
        # CrÃƒÂ©er le rÃƒÂ©pertoire si nÃƒÂ©cessaire
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # CrÃƒÂ©er l'engine
        self.engine = self._create_engine()
        
        # CrÃƒÂ©er les tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)
        
        logger.info(f"Ã¢Å“â€¦ Database initialisÃƒÂ©e: {self.db_path}")
    
    def _create_engine(self):
        """CrÃƒÂ©e l'engine SQLAlchemy"""
        if self.db_type == 'sqlite':
            # SQLite pour PC classique
            engine = create_engine(
                f'sqlite:///{self.db_path}',
                connect_args={'check_same_thread': False},
                poolclass=StaticPool,
                echo=False
            )
        elif self.db_type == 'postgresql':
            # PostgreSQL pour production
            db_url = self.config.get('db_url')
            engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
        else:
            raise ValueError(f"Type de DB non supportÃƒÂ©: {self.db_type}")
        
        return engine
    
    # ========================================================================
    # TRADES
    # ========================================================================
    
    def save_trade(self, trade_data: Dict) -> Trade:
        """
        Sauvegarde un trade
        
        Args:
            trade_data: DonnÃƒÂ©es du trade
            
        Returns:
            Trade sauvegardÃƒÂ©
        """
        session = self.Session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur sauvegarde trade: {e}")
            raise
        finally:
            session.close()
    
    def update_trade(self, trade_id: int, updates: Dict):
        """
        Met ÃƒÂ  jour un trade
        
        Args:
            trade_id: ID du trade
            updates: Mises ÃƒÂ  jour
        """
        session = self.Session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur mise ÃƒÂ  jour trade: {e}")
        finally:
            session.close()
    
    def get_recent_trades(self, limit: int = 100, strategy: str = None) -> List[Trade]:
        """
        RÃƒÂ©cupÃƒÂ¨re les trades rÃƒÂ©cents
        
        Args:
            limit: Nombre max de trades
            strategy: Filtrer par stratÃƒÂ©gie (optionnel)
            
        Returns:
            Liste des trades
        """
        session = self.Session()
        try:
            query = session.query(Trade).order_by(desc(Trade.entry_time))
            
            if strategy:
                query = query.filter(Trade.strategy == strategy)
            
            trades = query.limit(limit).all()
            return trades
        finally:
            session.close()
    
    def get_trades_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Trade]:
        """RÃƒÂ©cupÃƒÂ¨re les trades dans une pÃƒÂ©riode"""
        session = self.Session()
        try:
            trades = session.query(Trade).filter(
                Trade.entry_time >= start_date,
                Trade.entry_time <= end_date
            ).all()
            return trades
        finally:
            session.close()
    
    # ========================================================================
    # POSITIONS
    # ========================================================================
    
    def save_position(self, position_data: Dict) -> Position:
        """Sauvegarde une position"""
        session = self.Session()
        try:
            position = Position(**position_data)
            session.add(position)
            session.commit()
            session.refresh(position)
            return position
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur sauvegarde position: {e}")
            raise
        finally:
            session.close()
    
    def update_position(self, symbol: str, updates: Dict):
        """Met ÃƒÂ  jour une position"""
        session = self.Session()
        try:
            position = session.query(Position).filter(Position.symbol == symbol).first()
            if position:
                for key, value in updates.items():
                    setattr(position, key, value)
                position.last_update = datetime.now()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur mise ÃƒÂ  jour position: {e}")
        finally:
            session.close()
    
    def get_open_positions(self) -> List[Position]:
        """RÃƒÂ©cupÃƒÂ¨re toutes les positions ouvertes"""
        session = self.Session()
        try:
            positions = session.query(Position).all()
            return positions
        finally:
            session.close()
    
    def delete_position(self, symbol: str):
        """Supprime une position"""
        session = self.Session()
        try:
            position = session.query(Position).filter(Position.symbol == symbol).first()
            if position:
                session.delete(position)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur suppression position: {e}")
        finally:
            session.close()
    
    # ========================================================================
    # PERFORMANCE
    # ========================================================================
    
    def save_performance_snapshot(self, snapshot_data: Dict):
        """Sauvegarde un snapshot de performance"""
        session = self.Session()
        try:
            snapshot = PerformanceSnapshot(**snapshot_data)
            session.add(snapshot)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur sauvegarde snapshot: {e}")
        finally:
            session.close()
    
    def get_performance_history(self, hours: int = 24) -> List[PerformanceSnapshot]:
        """RÃƒÂ©cupÃƒÂ¨re l'historique de performance"""
        session = self.Session()
        try:
            since = datetime.now() - timedelta(hours=hours)
            snapshots = session.query(PerformanceSnapshot).filter(
                PerformanceSnapshot.timestamp >= since
            ).order_by(PerformanceSnapshot.timestamp).all()
            return snapshots
        finally:
            session.close()
    
    def update_strategy_performance(self, strategy_name: str, performance_data: Dict):
        """Met ÃƒÂ  jour la performance d'une stratÃƒÂ©gie"""
        session = self.Session()
        try:
            perf = session.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_name == strategy_name
            ).first()
            
            if not perf:
                # CrÃƒÂ©er si n'existe pas
                perf = StrategyPerformance(strategy_name=strategy_name)
                session.add(perf)
            
            # Mettre ÃƒÂ  jour
            for key, value in performance_data.items():
                setattr(perf, key, value)
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur update strategy performance: {e}")
        finally:
            session.close()
    
    def get_all_strategy_performance(self) -> List[StrategyPerformance]:
        """RÃƒÂ©cupÃƒÂ¨re les performances de toutes les stratÃƒÂ©gies"""
        session = self.Session()
        try:
            perfs = session.query(StrategyPerformance).all()
            return perfs
        finally:
            session.close()
    
    # ========================================================================
    # LOGS
    # ========================================================================
    
    def log_event(self, level: str, category: str, message: str, 
                   symbol: str = None, strategy: str = None, extra_data: Dict = None):
        """
        Enregistre un ÃƒÂ©vÃƒÂ©nement systÃƒÂ¨me
        
        Args:
            level: Niveau (INFO/WARNING/ERROR)
            category: CatÃƒÂ©gorie
            message: Message
            symbol: Symbole (optionnel)
            strategy: StratÃƒÂ©gie (optionnel)
            extra_data: DonnÃƒÂ©es supplÃƒÂ©mentaires (optionnel)
        """
        session = self.Session()
        try:
            log = SystemLog(
                level=level,
                category=category,
                message=message,
                symbol=symbol,
                strategy=strategy,
                extra_data=json.dumps(extra_data) if extra_data else None
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur log event: {e}")
        finally:
            session.close()
    
    def get_recent_logs(self, limit: int = 100, level: str = None) -> List[SystemLog]:
        """RÃƒÂ©cupÃƒÂ¨re les logs rÃƒÂ©cents"""
        session = self.Session()
        try:
            query = session.query(SystemLog).order_by(desc(SystemLog.timestamp))
            
            if level:
                query = query.filter(SystemLog.level == level)
            
            logs = query.limit(limit).all()
            return logs
        finally:
            session.close()
    
    # ========================================================================
    # STATISTIQUES
    # ========================================================================
    
    def get_trading_stats(self, days: int = 30) -> Dict:
        """
        Calcule les statistiques de trading
        
        Args:
            days: Nombre de jours
            
        Returns:
            Dict avec les stats
        """
        session = self.Session()
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Tous les trades
            trades = session.query(Trade).filter(
                Trade.entry_time >= since,
                Trade.status == 'closed'
            ).all()
            
            if not trades:
                return {}
            
            # Calculer les stats
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.profit_usdc > 0])
            losing_trades = len([t for t in trades if t.profit_usdc < 0])
            
            total_profit = sum(t.profit_usdc for t in trades)
            wins = [t.profit_usdc for t in trades if t.profit_usdc > 0]
            losses = [abs(t.profit_usdc) for t in trades if t.profit_usdc < 0]
            
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_profit': total_profit,
                'average_win': sum(wins) / len(wins) if wins else 0,
                'average_loss': sum(losses) / len(losses) if losses else 0,
                'profit_factor': sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0,
                'best_trade': max([t.profit_usdc for t in trades]) if trades else 0,
                'worst_trade': min([t.profit_usdc for t in trades]) if trades else 0,
            }
            
            return stats
            
        finally:
            session.close()
    
    # ========================================================================
    # MAINTENANCE
    # ========================================================================
    
    def cleanup_old_data(self, days: int = 30):
        """
        Nettoie les anciennes donnÃƒÂ©es
        
        Args:
            days: Garder les donnÃƒÂ©es des X derniers jours
        """
        session = self.Session()
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Supprimer vieux logs
            session.query(SystemLog).filter(SystemLog.timestamp < cutoff).delete()
            
            # Supprimer vieux snapshots (garder 1 par jour pour historique)
            old_snapshots = session.query(PerformanceSnapshot).filter(
                PerformanceSnapshot.timestamp < cutoff
            ).all()
            
            # Garder 1 snapshot par jour
            snapshots_to_keep = {}
            for snapshot in old_snapshots:
                day_key = snapshot.timestamp.date()
                if day_key not in snapshots_to_keep:
                    snapshots_to_keep[day_key] = snapshot
                else:
                    session.delete(snapshot)
            
            session.commit()
            logger.info(f"Ã°Å¸Â§Â¹ DonnÃƒÂ©es > {days} jours nettoyÃƒÂ©es")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur cleanup: {e}")
        finally:
            session.close()
    
    def optimize_database(self):
        """Optimise la base de donnÃƒÂ©es"""
        try:
            if self.db_type == 'sqlite':
                session = self.Session()
                session.execute('VACUUM')
                session.execute('ANALYZE')
                session.commit()
                session.close()
                logger.info("Ã¢Å“â€¦ Database optimisÃƒÂ©e")
        except Exception as e:
            logger.error(f"Erreur optimisation DB: {e}")
    
    def backup_database(self, backup_dir: str = "data/backups"):
        """
        Sauvegarde la base de donnÃƒÂ©es
        
        Args:
            backup_dir: RÃƒÂ©pertoire de backup
        """
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"bot_backup_{timestamp}.db")
            
            if self.db_type == 'sqlite':
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Ã°Å¸â€™Â¾ Backup crÃƒÂ©ÃƒÂ©: {backup_path}")
                
                # Garder seulement les 7 derniers backups
                self._cleanup_old_backups(backup_dir, keep=7)
            
        except Exception as e:
            logger.error(f"Erreur backup: {e}")
    
    def _cleanup_old_backups(self, backup_dir: str, keep: int = 7):
        """Nettoie les vieux backups"""
        try:
            backups = sorted(Path(backup_dir).glob("bot_backup_*.db"))
            if len(backups) > keep:
                for backup in backups[:-keep]:
                    backup.unlink()
                    logger.info(f"Ã°Å¸Â§Â¹ Ancien backup supprimÃƒÂ©: {backup.name}")
        except Exception as e:
            logger.error(f"Erreur cleanup backups: {e}")
    
    def get_database_stats(self) -> Dict:
        """Retourne les stats de la DB"""
        session = self.Session()
        try:
            stats = {
                'total_trades': session.query(Trade).count(),
                'open_positions': session.query(Position).count(),
                'performance_snapshots': session.query(PerformanceSnapshot).count(),
                'system_logs': session.query(SystemLog).count(),
                'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }
            return stats
        finally:
            session.close()
    
    def close(self):
        """Ferme la connexion"""
        try:
            self.Session.remove()
            self.engine.dispose()
            logger.info("Database fermÃƒÂ©e")
        except Exception as e:
            logger.error(f"Erreur fermeture DB: {e}")
# Alias pour compatibilitÃ©
Database = DatabaseManager

