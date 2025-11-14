"""
Tests pour les indicateurs techniques
"""

import pytest
import numpy as np
import pandas as pd
from utils.indicators import TechnicalIndicators


class TestMovingAverages:
    """Tests des moyennes mobiles"""
    
    def test_sma(self, sample_ohlcv_data):
        """Test SMA"""
        close = sample_ohlcv_data['close'].values
        sma = TechnicalIndicators.sma(close, 20)
        
        assert sma is not None
        assert len(sma) == len(close)
        assert not np.isnan(sma[-1])
        assert np.isnan(sma[0])  # Premiers valeurs = NaN
    
    def test_ema(self, sample_ohlcv_data):
        """Test EMA"""
        close = sample_ohlcv_data['close'].values
        ema = TechnicalIndicators.ema(close, 20)
        
        assert ema is not None
        assert len(ema) == len(close)
        assert not np.isnan(ema[-1])
    
    def test_sma_vs_ema(self, sample_ohlcv_data):
        """Test SMA vs EMA (EMA plus rÃƒÂ©active)"""
        close = sample_ohlcv_data['close'].values
        sma = TechnicalIndicators.sma(close, 20)
        ema = TechnicalIndicators.ema(close, 20)
        
        # EMA doit ÃƒÂªtre diffÃƒÂ©rente de SMA
        assert not np.array_equal(sma, ema)


class TestMomentumIndicators:
    """Tests des indicateurs de momentum"""
    
    def test_rsi(self, sample_ohlcv_data):
        """Test RSI"""
        close = sample_ohlcv_data['close'].values
        rsi = TechnicalIndicators.rsi(close, 14)
        
        assert rsi is not None
        assert len(rsi) == len(close)
        
        # RSI doit ÃƒÂªtre entre 0 et 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_ohlcv_data):
        """Test MACD"""
        close = sample_ohlcv_data['close'].values
        macd, signal, hist = TechnicalIndicators.macd(close)
        
        assert macd is not None
        assert signal is not None
        assert hist is not None
        assert len(macd) == len(close)
        
        # Histogram = MACD - Signal
        np.testing.assert_array_almost_equal(
            hist[~np.isnan(hist)],
            (macd - signal)[~np.isnan(hist)],
            decimal=5
        )
    
    def test_stochastic(self, sample_ohlcv_data):
        """Test Stochastic"""
        high = sample_ohlcv_data['high'].values
        low = sample_ohlcv_data['low'].values
        close = sample_ohlcv_data['close'].values
        
        k, d = TechnicalIndicators.stochastic(high, low, close)
        
        assert k is not None
        assert d is not None
        
        # K et D doivent ÃƒÂªtre entre 0 et 100
        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()


class TestVolatilityIndicators:
    """Tests des indicateurs de volatilitÃƒÂ©"""
    
    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands"""
        close = sample_ohlcv_data['close'].values
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close, 20, 2)
        
        assert upper is not None
        assert middle is not None
        assert lower is not None
        
        # Upper > Middle > Lower
        valid_idx = ~np.isnan(upper)
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()
    
    def test_atr(self, sample_ohlcv_data):
        """Test ATR"""
        high = sample_ohlcv_data['high'].values
        low = sample_ohlcv_data['low'].values
        close = sample_ohlcv_data['close'].values
        
        atr = TechnicalIndicators.atr(high, low, close, 14)
        
        assert atr is not None
        assert len(atr) == len(close)
        
        # ATR doit ÃƒÂªtre positif
        valid_atr = atr[~np.isnan(atr)]
        assert (valid_atr >= 0).all()


class TestTrendIndicators:
    """Tests des indicateurs de tendance"""
    
    def test_adx(self, sample_ohlcv_data):
        """Test ADX"""
        high = sample_ohlcv_data['high'].values
        low = sample_ohlcv_data['low'].values
        close = sample_ohlcv_data['close'].values
        
        adx = TechnicalIndicators.adx(high, low, close, 14)
        
        assert adx is not None
        
        # ADX entre 0 et 100
        valid_adx = adx[~np.isnan(adx)]
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()


class TestVolumeIndicators:
    """Tests des indicateurs de volume"""
    
    def test_obv(self, sample_ohlcv_data):
        """Test OBV"""
        close = sample_ohlcv_data['close'].values
        volume = sample_ohlcv_data['volume'].values
        
        obv = TechnicalIndicators.obv(close, volume)
        
        assert obv is not None
        assert len(obv) == len(close)
    
    def test_vwap(self, sample_ohlcv_data):
        """Test VWAP"""
        high = sample_ohlcv_data['high'].values
        low = sample_ohlcv_data['low'].values
        close = sample_ohlcv_data['close'].values
        volume = sample_ohlcv_data['volume'].values
        
        vwap = TechnicalIndicators.vwap(high, low, close, volume)
        
        assert vwap is not None
        assert len(vwap) == len(close)
        
        # VWAP doit ÃƒÂªtre positif
        assert (vwap > 0).all()
    
    def test_mfi(self, sample_ohlcv_data):
        """Test MFI"""
        high = sample_ohlcv_data['high'].values
        low = sample_ohlcv_data['low'].values
        close = sample_ohlcv_data['close'].values
        volume = sample_ohlcv_data['volume'].values
        
        mfi = TechnicalIndicators.mfi(high, low, close, volume, 14)
        
        assert mfi is not None
        
        # MFI entre 0 et 100
        valid_mfi = mfi[~np.isnan(mfi)]
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()


class TestCalculateAll:
    """Tests du calcul de tous les indicateurs"""
    
    def test_calculate_all(self, sample_ohlcv_data):
        """Test calcul de tous les indicateurs"""
        df = TechnicalIndicators.calculate_all(sample_ohlcv_data)
        
        assert df is not None
        assert not df.empty
        
        # VÃƒÂ©rifier que les indicateurs sont prÃƒÂ©sents
        expected_columns = [
            'sma_20', 'ema_9', 'ema_21',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'stoch_k', 'stoch_d', 'adx',
            'obv', 'vwap', 'mfi'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Colonne manquante: {col}"
    
    def test_calculate_all_with_config(self, sample_ohlcv_data):
        """Test avec configuration personnalisÃƒÂ©e"""
        config = {
            'rsi_period': 10,
            'ema_fast': 5,
            'ema_slow': 15,
            'bb_period': 15,
            'bb_std': 2.5,
            'atr_period': 10,
            'adx_period': 10
        }
        
        df = TechnicalIndicators.calculate_all(sample_ohlcv_data, config)
        
        assert df is not None
        assert not df.empty


class TestIndicatorEdgeCases:
    """Tests des cas limites"""
    
    def test_empty_data(self):
        """Test avec donnÃƒÂ©es vides"""
        empty_array = np.array([])
        
        sma = TechnicalIndicators.sma(empty_array, 20)
        assert len(sma) == 0
    
    def test_insufficient_data(self):
        """Test avec donnÃƒÂ©es insuffisantes"""
        short_data = np.array([1, 2, 3, 4, 5])
        
        sma = TechnicalIndicators.sma(short_data, 20)
        assert len(sma) == 5
        assert np.isnan(sma).all()
    
    def test_nan_handling(self):
        """Test gestion des NaN"""
        data_with_nan = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        
        # Les indicateurs doivent gÃƒÂ©rer les NaN
        sma = TechnicalIndicators.sma(data_with_nan, 3)
        assert sma is not None


class TestIndicatorAccuracy:
    """Tests de prÃƒÂ©cision des indicateurs"""
    
    def test_sma_accuracy(self):
        """Test prÃƒÂ©cision SMA"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma = TechnicalIndicators.sma(data, 3)
        
        # SMA(3) pour les 3 derniers: (8+9+10)/3 = 9
        assert abs(sma[-1] - 9.0) < 0.001
    
    def test_ema_accuracy(self):
        """Test prÃƒÂ©cision EMA"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ema = TechnicalIndicators.ema(data, 3)
        
        # EMA doit ÃƒÂªtre diffÃƒÂ©rente de SMA
        sma = TechnicalIndicators.sma(data, 3)
        assert not np.array_equal(ema, sma)
    
    def test_rsi_extremes(self):
        """Test RSI aux extrÃƒÂªmes"""
        # Prix qui monte constamment
        up_data = np.arange(1, 101)
        rsi_up = TechnicalIndicators.rsi(up_data, 14)
        
        # RSI devrait ÃƒÂªtre proche de 100
        assert rsi_up[-1] > 90
        
        # Prix qui descend constamment
        down_data = np.arange(100, 0, -1)
        rsi_down = TechnicalIndicators.rsi(down_data, 14)
        
        # RSI devrait ÃƒÂªtre proche de 0
        assert rsi_down[-1] < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])