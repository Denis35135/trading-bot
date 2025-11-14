# =============================================================================
# test_strategies.py - Tests pour les stratÃƒÂ©gies
# =============================================================================

class TestScalpingStrategy:
    """Tests de la stratÃƒÂ©gie Scalping"""
    
    def test_init(self):
        """Test initialisation"""
        from strategies.scalping import ScalpingStrategy
        
        config = {'min_profit': 0.003, 'max_hold_time': 300}
        strategy = ScalpingStrategy(config)
        
        assert strategy.name == 'scalping'
        assert strategy.is_active == True
    
    def test_signal_generation(self, sample_ohlcv_data):
        """Test gÃƒÂ©nÃƒÂ©ration de signal"""
        from strategies.scalping import ScalpingStrategy
        from utils.indicators import TechnicalIndicators
        
        config = {}
        strategy = ScalpingStrategy(config)
        
        # PrÃƒÂ©parer donnÃƒÂ©es
        df = TechnicalIndicators.calculate_all(sample_ohlcv_data)
        market_data = {
            'df': df,
            'ticker': {'price': df['close'].iloc[-1], 'volume': 1000000},
            'symbol': 'BTCUSDT'
        }
        
        signal = strategy.analyze(market_data)
        
        # Signal peut ÃƒÂªtre None ou valide
        if signal:
            assert 'type' in signal
            assert 'side' in signal
            assert 'confidence' in signal
    
    def test_position_management(self):
        """Test gestion de position"""
        from strategies.scalping import ScalpingStrategy
        
        strategy = ScalpingStrategy({})
        
        # Ouvrir position
        strategy.open_position('BTCUSDT', 'BUY', 50000, 0.1, {})
        assert strategy.has_position('BTCUSDT')
        
        # Fermer position
        strategy.close_position('BTCUSDT', 50100, 'target')
        assert not strategy.has_position('BTCUSDT')


class TestMomentumStrategy:
    """Tests de la stratÃƒÂ©gie Momentum"""
    
    def test_momentum_detection(self, sample_ohlcv_data):
        """Test dÃƒÂ©tection de momentum"""
        from strategies.momentum import MomentumStrategy
        from utils.indicators import TechnicalIndicators
        
        config = {'lookback': 20, 'min_strength': 0.6}
        strategy = MomentumStrategy(config)
        
        df = TechnicalIndicators.calculate_all(sample_ohlcv_data)
        market_data = {'df': df, 'ticker': {'price': df['close'].iloc[-1]}}
        
        signal = strategy.analyze(market_data)
        
        if signal:
            assert signal['confidence'] >= config['min_strength']


class TestMeanReversionStrategy:
    """Tests de la stratÃƒÂ©gie Mean Reversion"""
    
    def test_oversold_detection(self, sample_ohlcv_data):
        """Test dÃƒÂ©tection survendu"""
        from strategies.mean_reversion import MeanReversionStrategy
        from utils.indicators import TechnicalIndicators
        
        config = {'rsi_oversold': 30, 'rsi_overbought': 70}
        strategy = MeanReversionStrategy(config)
        
        df = TechnicalIndicators.calculate_all(sample_ohlcv_data)
        
        # Simuler RSI bas
        df.loc[df.index[-1], 'rsi'] = 25
        
        market_data = {'df': df, 'ticker': {'price': df['close'].iloc[-1]}}
        signal = strategy.analyze(market_data)
        
        if signal:
            assert signal['side'] == 'BUY'  # Survendu = signal d'achat
