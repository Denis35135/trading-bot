"""
test_risk.py - Tests pour la gestion des risques
"""

import pytest
import numpy as np
from datetime import datetime, timedelta


class TestRiskMonitor:
    """Tests du Risk Monitor"""
    
    def test_init(self, sample_config):
        """Test initialisation"""
        from risk.risk_monitor import RiskMonitor
        
        config = {
            'initial_capital': 1000,
            'max_drawdown': 0.08,
            'max_daily_loss': 0.05,
            'max_exposure': 0.8
        }
        
        monitor = RiskMonitor(config)
        assert monitor.capital == 1000
        assert monitor.max_drawdown == 0.08
    
    def test_drawdown_calculation(self):
        """Test calcul drawdown"""
        from risk.risk_monitor import RiskMonitor
        
        config = {'initial_capital': 1000, 'max_drawdown': 0.08}
        monitor = RiskMonitor(config)
        
        # Simuler perte
        monitor.update(900, {})  # -10%
        
        assert monitor.current_drawdown == 0.1
        assert monitor.current_risk_level.value in ['WARNING', 'HIGH']
    
    def test_circuit_breaker_emergency(self):
        """Test circuit breaker niveau urgence"""
        from risk.risk_monitor import RiskMonitor
        
        config = {'initial_capital': 1000, 'max_drawdown': 0.08}
        monitor = RiskMonitor(config)
        
        # Perte critique
        report = monitor.update(920, {})  # -8%
        
        assert report['risk_level'] in ['CRITICAL', 'EMERGENCY']
        assert 'CLOSE' in str(report.get('actions', []))
    
    def test_position_size_approval(self):
        """Test approbation taille position"""
        from risk.risk_monitor import RiskMonitor
        
        config = {
            'initial_capital': 1000,
            'max_drawdown': 0.08,
            'max_exposure': 0.8
        }
        monitor = RiskMonitor(config)
        
        signal = {'symbol': 'BTCUSDT', 'side': 'BUY', 'confidence': 0.8}
        approved, size, reason = monitor.approve_new_trade(signal, 250)
        
        assert approved == True
        assert size <= 250


class TestPositionSizing:
    """Tests du Position Sizing"""
    
    def test_calculate_size(self):
        """Test calcul taille position"""
        from risk.position_sizing import PositionSizer
        
        config = {
            'capital': 1000,
            'risk_per_trade': 0.02,
            'max_position_size': 0.25
        }
        sizer = PositionSizer(config)
        
        size = sizer.calculate(
            symbol='BTCUSDT',
            side='BUY',
            entry_price=50000,
            stop_loss=49000
        )
        
        assert size > 0
        assert size <= 250  # 25% de 1000
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion"""
        from risk.position_sizing import PositionSizer
        from tests import generate_random_trades
        
        config = {'capital': 1000, 'max_position_size': 0.25}
        sizer = PositionSizer(config)
        
        trades = generate_random_trades(100)
        kelly = sizer._kelly_criterion(trades)
        
        assert kelly >= 0
        assert kelly <= 1