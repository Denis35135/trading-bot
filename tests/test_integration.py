# =============================================================================
# test_integration.py - Tests d'intÃƒÂ©gration
# =============================================================================

class TestEndToEndFlow:
    """Tests du flux complet de trading"""
    
    def test_full_trading_cycle(self, mock_exchange, sample_config):
        """Test cycle complet de trading"""
        # 1. Scanner trouve des symboles
        from scanner.market_scanner import MarketScanner
        
        scanner = MarketScanner(mock_exchange, sample_config)
        # Mock symboles
        scanner.top_symbols = ['BTCUSDT', 'ETHUSDT']
        
        # 2. RÃƒÂ©cupÃƒÂ©rer donnÃƒÂ©es de marchÃƒÂ©
        symbols = scanner.get_top_symbols()
        assert len(symbols) > 0
        
        # 3. Analyser avec stratÃƒÂ©gie
        # (simulÃƒÂ© car nÃƒÂ©cessite donnÃƒÂ©es rÃƒÂ©elles)
        
        # 4. Valider avec risk monitor
        from risk.risk_monitor import RiskMonitor
        
        risk_config = {
            'initial_capital': 1000,
            'max_drawdown': 0.08
        }
        risk_monitor = RiskMonitor(risk_config)
        
        signal = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'price': 50000,
            'confidence': 0.8
        }
        
        approved, size, reason = risk_monitor.approve_new_trade(signal, 200)
        assert approved == True
    
    def test_data_pipeline(self, sample_ohlcv_data):
        """Test pipeline de donnÃƒÂ©es"""
        # 1. DonnÃƒÂ©es brutes
        assert not sample_ohlcv_data.empty
        
        # 2. Calcul indicateurs
        from utils.indicators import TechnicalIndicators
        df = TechnicalIndicators.calculate_all(sample_ohlcv_data)
        assert 'rsi' in df.columns
        
        # 3. Features ML
        from ml.features import FeatureEngineer
        engineer = FeatureEngineer()
        features = engineer.create_features(df)
        assert not features.empty
    
    def test_signal_to_execution(self, mock_exchange):
        """Test du signal ÃƒÂ  l'exÃƒÂ©cution"""
        signal = {
            'type': 'ENTRY',
            'side': 'BUY',
            'symbol': 'BTCUSDT',
            'price': 50000,
            'confidence': 0.8,
            'stop_loss': 49000,
            'take_profit': 51000
        }
        
        # 1. Valider signal
        from tests import assert_signal_valid
        assert_signal_valid(signal)
        
        # 2. Calculer taille position
        from risk.position_sizing import PositionSizer
        config = {'capital': 1000, 'risk_per_trade': 0.02}
        sizer = PositionSizer(config)
        
        size = sizer.calculate(
            symbol=signal['symbol'],
            side=signal['side'],
            entry_price=signal['price'],
            stop_loss=signal['stop_loss']
        )
        assert size > 0
        
        # 3. ExÃƒÂ©cuter (mock)
        mock_exchange.create_order.return_value = {
            'orderId': '123',
            'status': 'FILLED'
        }
        
        order = mock_exchange.create_order(
            symbol=signal['symbol'],
            side=signal['side'],
            type='MARKET',
            quantity=size
        )
        
        assert order['status'] == 'FILLED'


class TestComponentIntegration:
    """Tests d'intÃƒÂ©gration des composants"""
    
    def test_strategy_manager_integration(self, mock_exchange, sample_config):
        """Test intÃƒÂ©gration Strategy Manager"""
        from strategies.strategy_manager import StrategyManager
        from risk.position_sizing import PositionSizer
        from risk.risk_monitor import RiskMonitor
        
        # Initialiser composants
        risk_monitor = RiskMonitor({'initial_capital': 1000, 'max_drawdown': 0.08})
        position_sizer = PositionSizer({'capital': 1000, 'risk_per_trade': 0.02})
        
        config = {
            'strategies': [
                {'name': 'scalping', 'allocation': 1.0, 'enabled': True}
            ]
        }
        
        # CrÃƒÂ©er strategy manager (peut ÃƒÂ©chouer si dÃƒÂ©pendances manquantes)
        try:
            manager = StrategyManager(
                config=config,
                exchange_client=mock_exchange,
                order_manager=None,
                position_sizer=position_sizer,
                risk_monitor=risk_monitor
            )
            assert manager is not None
        except ImportError:
            pytest.skip("StratÃƒÂ©gies non disponibles")
    
    def test_thread_coordination(self):
        """Test coordination des threads"""
        from threads.market_data_thread import MarketDataThread
        from threads.execution_thread import ExecutionThread
        from unittest.mock import MagicMock
        
        bot = MagicMock()
        bot.running = True
        bot.capital = 1000
        
        # CrÃƒÂ©er threads
        md_config = {'update_interval': 1, 'buffer_size': 100}
        md_thread = MarketDataThread(bot, md_config)
        exec_thread = ExecutionThread(bot)
        
        # DÃƒÂ©marrer (briÃƒÂ¨vement)
        md_thread.start()
        exec_thread.start()
        
        # VÃƒÂ©rifier qu'ils tournent
        assert md_thread.is_running
        assert exec_thread.is_running
        
        # ArrÃƒÂªter
        md_thread.stop()
        exec_thread.stop()
        
        assert not md_thread.is_running
        assert not exec_thread.is_running


class TestPerformanceMetrics:
    """Tests des mÃƒÂ©triques de performance"""
    
    def test_calculate_returns(self):
        """Test calcul des returns"""
        from utils.helpers import calculate_profit
        
        profit_data = calculate_profit(
            entry_price=50000,
            exit_price=51000,
            quantity=0.1,
            side='BUY',
            fees=0.001
        )
        
        assert profit_data['profit_usdc'] > 0
        assert profit_data['profit_pct'] > 0
    
    def test_sharpe_ratio(self):
        """Test calcul Sharpe ratio"""
        from utils.helpers import calculate_sharpe_ratio
        
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02]
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe is not None
        assert isinstance(sharpe, float)
    
    def test_max_drawdown(self):
        """Test calcul max drawdown"""
        from utils.helpers import calculate_max_drawdown
        
        capital_history = [1000, 1100, 1050, 900, 950, 1200]
        dd_info = calculate_max_drawdown(capital_history)
        
        assert 'max_dd' in dd_info
        assert 'max_dd_pct' in dd_info
        assert dd_info['max_dd'] < 0


class TestErrorRecovery:
    """Tests de rÃƒÂ©cupÃƒÂ©ration d'erreurs"""
    
    def test_connection_retry(self, mock_exchange):
        """Test retry en cas d'erreur de connexion"""
        from utils.decorators import retry
        
        call_count = {'count': 0}
        
        @retry(max_attempts=3, delay=0.1)
        def failing_function():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise ConnectionError("Failed")
            return "Success"
        
        result = failing_function()
        assert result == "Success"
        assert call_count['count'] == 3
    
    def test_circuit_breaker_recovery(self):
        """Test rÃƒÂ©cupÃƒÂ©ration aprÃƒÂ¨s circuit breaker"""
        from risk.risk_monitor import RiskMonitor
        
        config = {'initial_capital': 1000, 'max_drawdown': 0.08}
        monitor = RiskMonitor(config)
        
        # DÃƒÂ©clencher circuit breaker
        monitor.update(920, {})  # -8%
        assert monitor.circuit_breaker_active
        
        # RÃƒÂ©cupÃƒÂ©ration
        monitor.update(1000, {})  # Retour au capital initial
        # Circuit breaker devrait se dÃƒÂ©sactiver aprÃƒÂ¨s un certain temps
        # (logique ÃƒÂ  implÃƒÂ©menter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
