"""
ML Strategy - StratÃƒÂ©gie basÃƒÂ©e sur Machine Learning
Utilise un ensemble de 3 modÃƒÂ¨les optimisÃƒÂ©s pour CPU
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging
from datetime import datetime
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    StratÃƒÂ©gie ML lÃƒÂ©gÃƒÂ¨re mais efficace (5% du capital)
    
    CaractÃƒÂ©ristiques:
    - 3 modÃƒÂ¨les lÃƒÂ©gers (RF, XGB, LogReg)
    - 30 features seulement
    - Vote majoritaire avec seuil de confiance
    - RÃƒÂ©entraÃƒÂ®nement automatique quotidien
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise la stratÃƒÂ©gie ML
        
        Args:
            config: Configuration de la stratÃƒÂ©gie
        """
        default_config = {
            'name': 'ML_Strategy',
            'allocation': 0.05,  # 5% du capital
            'min_confidence': 0.70,  # Confiance minimum 70%
            'feature_count': 30,
            'retrain_frequency': 86400,  # 24h en secondes
            'models_path': 'data/models/',
            'use_cache': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # ModÃƒÂ¨les ML
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                n_jobs=4,
                random_state=42
            ),
            'xgb': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist',
                n_jobs=4,
                random_state=42
            ),
            'logreg': LogisticRegression(
                max_iter=1000,
                n_jobs=4,
                random_state=42
            )
        }
        
        self.models_trained = False
        self.last_retrain = None
        self.feature_names = []
        
        # Charger les modÃƒÂ¨les existants
        self._load_models()
    
    def analyze(self, data: Dict) -> Optional[Dict]:
        """
        Analyse avec ML et gÃƒÂ©nÃƒÂ¨re un signal
        
        Args:
            data: Dict avec df, orderbook, trades, symbol
            
        Returns:
            Signal de trading ou None
        """
        try:
            if not self.is_active:
                return None
            
            df = data.get('df')
            symbol = data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < 100:
                return None
            
            # Calculer les indicateurs
            df = self.calculate_indicators(df)
            
            # Extraire les features
            features = self._extract_features(df)
            
            if features is None:
                return None
            
            # VÃƒÂ©rifier si les modÃƒÂ¨les sont entraÃƒÂ®nÃƒÂ©s
            if not self.models_trained:
                logger.warning("ModÃƒÂ¨les ML non entraÃƒÂ®nÃƒÂ©s, signal ignorÃƒÂ©")
                return None
            
            # PrÃƒÂ©diction avec ensemble voting
            signal = self._predict_with_ensemble(features, df, symbol)
            
            if signal and self.validate_signal(signal):
                self.performance['total_signals'] += 1
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur ML analyze: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les 30 features pour le ML
        
        Args:
            df: DataFrame avec OHLCV
            
        Returns:
            DataFrame avec features
        """
        try:
            df = df.copy()
            
            # Prix (5 features)
            df['price_change_5m'] = df['close'].pct_change(5)
            df['price_change_15m'] = df['close'].pct_change(15)
            df['price_change_60m'] = df['close'].pct_change(60)
            
            # Position dans le range 24h
            high_24h = df['high'].rolling(288).max()  # 288 * 5min = 24h
            low_24h = df['low'].rolling(288).min()
            df['price_position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
            
            # Distance from VWAP
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['distance_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']
            
            # Volume (5 features)
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
            df['volume_trend'] = df['volume'].rolling(10).apply(
                lambda x: 1 if x[-1] > x[0] else -1, raw=True
            )
            
            # Indicateurs techniques (10 features)
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['rsi_divergence'] = self._calculate_rsi_divergence(df)
            
            # MACD
            ema_fast = df['close'].ewm(span=12).mean()
            ema_slow = df['close'].ewm(span=26).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_ma = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = bb_ma + (bb_std * 2)
            df['bb_lower'] = bb_ma - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_normalized'] = df['atr'] / df['close']
            
            # EMA trend
            df['ema_fast'] = df['close'].ewm(span=9).mean()
            df['ema_slow'] = df['close'].ewm(span=21).mean()
            df['ema_trend'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
            
            # Stochastic
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
            
            # ADX (simplifiÃƒÂ©)
            df['adx'] = self._calculate_adx_simple(df)
            
            # Market Structure (5 features)
            df['support_distance'] = self._calculate_support_distance(df)
            df['resistance_distance'] = self._calculate_resistance_distance(df)
            
            # Spread (si disponible dans orderbook)
            df['spread_ratio'] = 0.001  # Valeur par dÃƒÂ©faut
            
            # Momentum (5 features)
            df['momentum'] = df['close'].pct_change(20)
            df['trend_strength'] = df['close'].rolling(20).apply(
                lambda x: (x[-1] - x[0]) / x.std() if x.std() > 0 else 0, raw=True
            )
            
            # Volatility regime
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volatility_ma'] = df['volatility'].rolling(50).mean()
            df['volatility_regime'] = df['volatility'] / (df['volatility_ma'] + 1e-10)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs ML: {e}")
            return df
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extrait les 30 features pour la prÃƒÂ©diction
        
        Args:
            df: DataFrame avec indicateurs
            
        Returns:
            Array numpy avec les features ou None
        """
        try:
            # Liste des 30 features
            feature_cols = [
                'price_change_5m', 'price_change_15m', 'price_change_60m',
                'price_position_24h', 'distance_from_vwap',
                'volume_ratio', 'volume_trend',
                'rsi', 'rsi_divergence', 'macd_hist', 'bb_position',
                'atr_normalized', 'ema_trend', 'stoch_k', 'adx',
                'support_distance', 'resistance_distance', 'spread_ratio',
                'momentum', 'trend_strength', 'volatility_regime'
            ]
            
            # Prendre la derniÃƒÂ¨re ligne
            last_row = df.iloc[-1]
            
            # Extraire les features
            features = []
            for col in feature_cols:
                if col in df.columns:
                    val = last_row[col]
                    # Remplacer NaN par 0
                    features.append(0 if pd.isna(val) else val)
                else:
                    features.append(0)
            
            # Stocker les noms de features
            self.feature_names = feature_cols[:len(features)]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erreur extraction features: {e}")
            return None
    
    def _predict_with_ensemble(self, features: np.ndarray, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        PrÃƒÂ©diction avec vote majoritaire des 3 modÃƒÂ¨les
        
        Args:
            features: Features extraites
            df: DataFrame pour contexte
            symbol: Le symbole
            
        Returns:
            Signal ou None
        """
        try:
            predictions = []
            probabilities = []
            
            # PrÃƒÂ©diction de chaque modÃƒÂ¨le
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(features)[0]
                    pred_class = np.argmax(proba)
                    predictions.append(pred_class)
                    probabilities.append(proba[1])  # Proba de classe 1 (BUY)
                except Exception as e:
                    logger.warning(f"Erreur prÃƒÂ©diction modÃƒÂ¨le {name}: {e}")
                    continue
            
            if not predictions:
                return None
            
            # Moyenne des probabilitÃƒÂ©s
            avg_confidence = np.mean(probabilities)
            
            # DÃƒÂ©terminer le signal
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
            
            # Signal BUY si confiance > 70%
            if avg_confidence > self.config['min_confidence']:
                return {
                    'type': 'ENTRY',
                    'side': 'BUY',
                    'price': current_price,
                    'confidence': avg_confidence,
                    'stop_loss': current_price - (atr * 2),
                    'take_profit': current_price + (atr * 3),
                    'reasons': [
                        f'ML Ensemble confiance: {avg_confidence:.2%}',
                        f'Consensus de {len(predictions)} modÃƒÂ¨les',
                        f'Features favorables dÃƒÂ©tectÃƒÂ©es'
                    ],
                    'metadata': {
                        'strategy': self.name,
                        'model_predictions': predictions,
                        'probabilities': probabilities,
                        'feature_count': len(features[0])
                    }
                }
            
            # Signal SELL si confiance < 30%
            elif avg_confidence < (1 - self.config['min_confidence']):
                return {
                    'type': 'ENTRY',
                    'side': 'SELL',
                    'price': current_price,
                    'confidence': 1 - avg_confidence,
                    'stop_loss': current_price + (atr * 2),
                    'take_profit': current_price - (atr * 3),
                    'reasons': [
                        f'ML Ensemble confiance SHORT: {1-avg_confidence:.2%}',
                        f'Signal baissier dÃƒÂ©tectÃƒÂ©',
                        f'Consensus nÃƒÂ©gatif'
                    ],
                    'metadata': {
                        'strategy': self.name,
                        'model_predictions': predictions,
                        'probabilities': probabilities
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur ensemble prediction: {e}")
            return None
    
    def train_models(self, training_data: pd.DataFrame, labels: np.ndarray):
        """
        EntraÃƒÂ®ne les 3 modÃƒÂ¨les ML
        
        Args:
            training_data: Features d'entraÃƒÂ®nement
            labels: Labels (0=SELL, 1=BUY, 2=HOLD)
        """
        try:
            logger.info("Ã°Å¸Â¤â€“ DÃƒÂ©but entraÃƒÂ®nement modÃƒÂ¨les ML...")
            
            # Split train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                training_data, labels, test_size=0.2, random_state=42
            )
            
            # EntraÃƒÂ®ner chaque modÃƒÂ¨le
            for name, model in self.models.items():
                try:
                    logger.info(f"EntraÃƒÂ®nement {name}...")
                    model.fit(X_train, y_train)
                    
                    # Validation
                    score = model.score(X_val, y_val)
                    logger.info(f"  {name} accuracy: {score:.2%}")
                    
                except Exception as e:
                    logger.error(f"Erreur entraÃƒÂ®nement {name}: {e}")
            
            self.models_trained = True
            self.last_retrain = datetime.now()
            
            # Sauvegarder les modÃƒÂ¨les
            self._save_models()
            
            logger.info("Ã¢Å“â€¦ EntraÃƒÂ®nement terminÃƒÂ©!")
            
        except Exception as e:
            logger.error(f"Erreur train_models: {e}")
    
    def _save_models(self):
        """Sauvegarde les modÃƒÂ¨les entraÃƒÂ®nÃƒÂ©s"""
        try:
            models_dir = Path(self.config['models_path'])
            models_dir.mkdir(parents=True, exist_ok=True)
            
            for name, model in self.models.items():
                filepath = models_dir / f'{name}_model.joblib'
                joblib.dump(model, filepath)
                logger.debug(f"ModÃƒÂ¨le {name} sauvegardÃƒÂ©: {filepath}")
            
            # Sauvegarder les feature names
            features_file = models_dir / 'feature_names.joblib'
            joblib.dump(self.feature_names, features_file)
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modÃƒÂ¨les: {e}")
    
    def _load_models(self):
        """Charge les modÃƒÂ¨les entraÃƒÂ®nÃƒÂ©s"""
        try:
            models_dir = Path(self.config['models_path'])
            
            if not models_dir.exists():
                logger.info("Aucun modÃƒÂ¨le sauvegardÃƒÂ© trouvÃƒÂ©")
                return
            
            loaded_count = 0
            for name in self.models.keys():
                filepath = models_dir / f'{name}_model.joblib'
                if filepath.exists():
                    self.models[name] = joblib.load(filepath)
                    loaded_count += 1
                    logger.debug(f"ModÃƒÂ¨le {name} chargÃƒÂ©")
            
            # Charger les feature names
            features_file = models_dir / 'feature_names.joblib'
            if features_file.exists():
                self.feature_names = joblib.load(features_file)
            
            if loaded_count == len(self.models):
                self.models_trained = True
                logger.info(f"Ã¢Å“â€¦ {loaded_count} modÃƒÂ¨les ML chargÃƒÂ©s")
            else:
                logger.warning(f"Seulement {loaded_count}/{len(self.models)} modÃƒÂ¨les chargÃƒÂ©s")
                
        except Exception as e:
            logger.error(f"Erreur chargement modÃƒÂ¨les: {e}")
    
    def should_retrain(self) -> bool:
        """
        VÃƒÂ©rifie si un rÃƒÂ©entraÃƒÂ®nement est nÃƒÂ©cessaire
        
        Returns:
            True si rÃƒÂ©entraÃƒÂ®nement nÃƒÂ©cessaire
        """
        if not self.models_trained:
            return True
        
        if self.last_retrain is None:
            return True
        
        # VÃƒÂ©rifier la frÃƒÂ©quence
        time_since_retrain = (datetime.now() - self.last_retrain).total_seconds()
        if time_since_retrain > self.config['retrain_frequency']:
            return True
        
        # VÃƒÂ©rifier la performance
        if self.performance['win_rate'] < 0.55:  # Moins de 55% win rate
            logger.info("Performance dÃƒÂ©gradÃƒÂ©e, rÃƒÂ©entraÃƒÂ®nement nÃƒÂ©cessaire")
            return True
        
        return False
    
    # MÃƒÂ©thodes helper pour calculs
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """DÃƒÂ©tecte les divergences RSI (simplifiÃƒÂ©)"""
        if 'rsi' not in df.columns:
            return pd.Series(0, index=df.index)
        
        rsi_slope = df['rsi'].diff(5)
        price_slope = df['close'].pct_change(5)
        
        # Divergence = RSI et prix vont dans des directions opposÃƒÂ©es
        divergence = np.where(
            (rsi_slope * price_slope) < 0, 1, 0
        )
        return pd.Series(divergence, index=df.index)
    
    def _calculate_adx_simple(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule un ADX simplifiÃƒÂ©"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        # ADX simplifiÃƒÂ© basÃƒÂ© sur l'ATR
        adx_simple = (atr / df['close']) * 100
        return adx_simple
    
    def _calculate_support_distance(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Calcule la distance au support le plus proche"""
        rolling_low = df['low'].rolling(window).min()
        distance = (df['close'] - rolling_low) / df['close']
        return distance
    
    def _calculate_resistance_distance(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Calcule la distance ÃƒÂ  la rÃƒÂ©sistance la plus proche"""
        rolling_high = df['high'].rolling(window).max()
        distance = (rolling_high - df['close']) / df['close']
        return distance


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test de la stratÃƒÂ©gie ML"""
    
    # Configuration de test
    config = {
        'min_confidence': 0.70,
        'models_path': 'data/models/'
    }
    
    strategy = MLStrategy(config)
    
    # DonnÃƒÂ©es de test
    dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
    test_df = pd.DataFrame({
        'open': 100 + np.random.randn(500).cumsum(),
        'high': 101 + np.random.randn(500).cumsum(),
        'low': 99 + np.random.randn(500).cumsum(),
        'close': 100 + np.random.randn(500).cumsum(),
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    data = {
        'df': test_df,
        'symbol': 'BTCUSDC'
    }
    
    print("Test ML Strategy")
    print("=" * 50)
    print(f"StratÃƒÂ©gie: {strategy.name}")
    print(f"Active: {strategy.is_active}")
    print(f"ModÃƒÂ¨les entraÃƒÂ®nÃƒÂ©s: {strategy.models_trained}")
    
    # Test d'analyse (sans modÃƒÂ¨les entraÃƒÂ®nÃƒÂ©s)
    signal = strategy.analyze(data)
    print(f"\nSignal gÃƒÂ©nÃƒÂ©rÃƒÂ©: {signal}")