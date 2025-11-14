# =============================================================================
# test_scanner.py - Tests pour le scanner de marchÃƒÂ©
# =============================================================================

class TestMarketScanner:
    """Tests du Market Scanner"""
    
    def test_init(self, mock_exchange):
        """Test initialisation"""
        from scanner.market_scanner import MarketScanner
        
        config = {
            'min_volume_24h': 10000000,
            'max_spread_percent': 0.002,
            'symbols_to_scan': 100,
            'symbols_to_trade': 20
        }
        
        scanner = MarketScanner(mock_exchange, config)
        assert scanner.config == config
    
    def test_scan_symbols(self, mock_exchange):
        """Test scan des symboles"""
        from scanner.market_scanner import MarketScanner
        
        # Mock retour de symboles
        mock_exchange.get_all_tickers.return_value = [
            {'symbol': 'BTCUSDT', 'volume': 50000000, 'price': 50000},
            {'symbol': 'ETHUSDT', 'volume': 30000000, 'price': 3000},
            {'symbol': 'BNBUSDT', 'volume': 20000000, 'price': 400},
        ]
        
        config = {
            'min_volume_24h': 10000000,
            'symbols_to_scan': 100,
            'symbols_to_trade': 20
        }
        scanner = MarketScanner(mock_exchange, config)
        scanner.perform_scan()
        
        top = scanner.get_top_symbols()
        assert len(top) > 0
        assert 'BTCUSDT' in top
    
    def test_symbol_scoring(self):
        """Test scoring des symboles"""
        from scanner.market_scanner import MarketScanner
        
        config = {}
        scanner = MarketScanner(None, config)
        
        symbol_data = {
            'volume': 50000000,
            'volatility': 0.03,
            'spread': 0.001
        }
        
        score = scanner._calculate_symbol_score(symbol_data)
        assert score >= 0
        assert score <= 100
    
    def test_blacklist(self, mock_exchange):
        """Test blacklist"""
        from scanner.market_scanner import MarketScanner
        
        config = {}
        scanner = MarketScanner(mock_exchange, config)
        
        scanner.add_to_blacklist('BTCUSDT')
        assert 'BTCUSDT' in scanner.blacklisted_symbols
        
        scanner.remove_from_blacklist('BTCUSDT')
        assert 'BTCUSDT' not in scanner.blacklisted_symbols
    
    def test_volatility_filter(self):
        """Test filtre de volatilitÃƒÂ©"""
        from scanner.market_scanner import MarketScanner
        
        config = {
            'volatility_range': [0.01, 0.10]
        }
        scanner = MarketScanner(None, config)
        
        # VolatilitÃƒÂ© trop faible
        assert not scanner._check_volatility({'volatility': 0.005})
        
        # VolatilitÃƒÂ© OK
        assert scanner._check_volatility({'volatility': 0.03})
        
        # VolatilitÃƒÂ© trop ÃƒÂ©levÃƒÂ©e
        assert not scanner._check_volatility({'volatility': 0.15})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])