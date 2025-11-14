"""
Tests pour le client d'exchange Binance
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestBinanceClient:
    """Tests pour BinanceClient"""
    
    def test_connection(self, mock_exchange):
        """Test de connexion ÃƒÂ  l'exchange"""
        assert mock_exchange is not None
        assert hasattr(mock_exchange, 'get_symbol_ticker')
    
    def test_get_symbol_ticker(self, mock_exchange):
        """Test rÃƒÂ©cupÃƒÂ©ration ticker"""
        ticker = mock_exchange.get_symbol_ticker('BTCUSDT')
        
        assert ticker is not None
        assert 'symbol' in ticker
        assert 'price' in ticker
        assert ticker['price'] > 0
    
    def test_get_orderbook(self, mock_exchange, sample_orderbook):
        """Test rÃƒÂ©cupÃƒÂ©ration orderbook"""
        mock_exchange.get_orderbook.return_value = sample_orderbook
        
        orderbook = mock_exchange.get_orderbook('BTCUSDT')
        
        assert orderbook is not None
        assert 'bids' in orderbook
        assert 'asks' in orderbook
        assert len(orderbook['bids']) > 0
        assert len(orderbook['asks']) > 0
    
    def test_get_klines(self, mock_exchange, sample_ohlcv_data):
        """Test rÃƒÂ©cupÃƒÂ©ration klines"""
        mock_exchange.get_klines.return_value = sample_ohlcv_data
        
        df = mock_exchange.get_klines('BTCUSDT', '5m', limit=100)
        
        assert df is not None
        assert not df.empty
        assert len(df) == 100
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_create_order(self, mock_exchange):
        """Test crÃƒÂ©ation d'ordre"""
        mock_exchange.create_order.return_value = {
            'orderId': '123456',
            'status': 'FILLED',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'price': 50000.0,
            'executedQty': 0.1
        }
        
        order = mock_exchange.create_order(
            symbol='BTCUSDT',
            side='BUY',
            type='MARKET',
            quantity=0.1
        )
        
        assert order is not None
        assert order['status'] == 'FILLED'
        assert order['executedQty'] == 0.1
    
    def test_get_account_balance(self, mock_exchange):
        """Test rÃƒÂ©cupÃƒÂ©ration solde"""
        mock_exchange.get_account_balance.return_value = 1000.0
        
        balance = mock_exchange.get_account_balance('USDT')
        
        assert balance > 0
        assert balance == 1000.0
    
    def test_cancel_order(self, mock_exchange):
        """Test annulation d'ordre"""
        mock_exchange.cancel_order.return_value = {
            'orderId': '123456',
            'status': 'CANCELED'
        }
        
        result = mock_exchange.cancel_order('BTCUSDT', '123456')
        
        assert result is not None
        assert result['status'] == 'CANCELED'
    
    def test_get_open_orders(self, mock_exchange):
        """Test rÃƒÂ©cupÃƒÂ©ration ordres ouverts"""
        mock_exchange.get_open_orders.return_value = [
            {'orderId': '123', 'symbol': 'BTCUSDT'},
            {'orderId': '456', 'symbol': 'ETHUSDT'}
        ]
        
        orders = mock_exchange.get_open_orders()
        
        assert orders is not None
        assert len(orders) == 2
    
    def test_websocket_connection(self, mock_exchange):
        """Test connexion WebSocket"""
        mock_exchange.start_websocket.return_value = True
        
        result = mock_exchange.start_websocket()
        
        assert result == True
    
    def test_subscribe_ticker(self, mock_exchange):
        """Test souscription ticker WebSocket"""
        callback = MagicMock()
        mock_exchange.subscribe_ticker.return_value = True
        
        result = mock_exchange.subscribe_ticker('BTCUSDT', callback)
        
        assert result == True


class TestExchangeValidation:
    """Tests de validation des donnÃƒÂ©es exchange"""
    
    def test_ticker_validation(self, sample_ticker):
        """Test validation ticker"""
        assert sample_ticker['price'] > 0
        assert sample_ticker['bid'] < sample_ticker['ask']
        assert sample_ticker['volume'] > 0
    
    def test_orderbook_validation(self, sample_orderbook):
        """Test validation orderbook"""
        # VÃƒÂ©rifier structure
        assert 'bids' in sample_orderbook
        assert 'asks' in sample_orderbook
        
        # VÃƒÂ©rifier prix
        best_bid = sample_orderbook['bids'][0][0]
        best_ask = sample_orderbook['asks'][0][0]
        assert best_bid < best_ask
        
        # VÃƒÂ©rifier ordres triÃƒÂ©s
        bid_prices = [bid[0] for bid in sample_orderbook['bids']]
        assert bid_prices == sorted(bid_prices, reverse=True)
        
        ask_prices = [ask[0] for ask in sample_orderbook['asks']]
        assert ask_prices == sorted(ask_prices)
    
    def test_klines_validation(self, sample_ohlcv_data):
        """Test validation klines"""
        df = sample_ohlcv_data
        
        # VÃƒÂ©rifier colonnes
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # VÃƒÂ©rifier relations OHLC
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        
        # VÃƒÂ©rifier volume positif
        assert (df['volume'] > 0).all()


class TestExchangeErrors:
    """Tests de gestion d'erreurs"""
    
    def test_connection_error(self, mock_exchange):
        """Test erreur de connexion"""
        mock_exchange.get_symbol_ticker.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            mock_exchange.get_symbol_ticker('BTCUSDT')
    
    def test_invalid_symbol(self, mock_exchange):
        """Test symbole invalide"""
        mock_exchange.get_symbol_ticker.side_effect = ValueError("Invalid symbol")
        
        with pytest.raises(ValueError):
            mock_exchange.get_symbol_ticker('INVALID')
    
    def test_insufficient_balance(self, mock_exchange):
        """Test solde insuffisant"""
        mock_exchange.create_order.side_effect = Exception("Insufficient balance")
        
        with pytest.raises(Exception):
            mock_exchange.create_order(
                symbol='BTCUSDT',
                side='BUY',
                type='MARKET',
                quantity=1000  # Trop grand
            )
    
    def test_rate_limit_exceeded(self, mock_exchange):
        """Test dÃƒÂ©passement rate limit"""
        mock_exchange.get_symbol_ticker.side_effect = Exception("Rate limit exceeded")
        
        with pytest.raises(Exception):
            mock_exchange.get_symbol_ticker('BTCUSDT')


class TestExchangeHelpers:
    """Tests des fonctions helper"""
    
    def test_symbol_validation(self):
        """Test validation de symbole"""
        from utils.helpers import is_valid_symbol
        
        assert is_valid_symbol('BTCUSDT') == True
        assert is_valid_symbol('ETHUSDC') == True
        assert is_valid_symbol('invalid') == False
        assert is_valid_symbol('BTC') == False
    
    def test_price_rounding(self):
        """Test arrondi de prix"""
        from utils.helpers import round_price
        
        price = 50000.123456
        rounded = round_price(price, 0.01)
        
        assert rounded == 50000.12
    
    def test_quantity_rounding(self):
        """Test arrondi de quantitÃƒÂ©"""
        from utils.helpers import round_step_size
        
        qty = 0.123456
        rounded = round_step_size(qty, 0.001)
        
        assert rounded == 0.123


if __name__ == '__main__':
    pytest.main([__file__, '-v'])