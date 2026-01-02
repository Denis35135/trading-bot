#!/usr/bin/env python3
"""
Script d'entranement des modeles ML
Entrane les 3 modeles (RandomForest, XGBoost, LogisticRegression)
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List
import argparse
from datetime import datetime
import joblib

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from xgboost import XGBClassifier
except ImportError as e:
    print(f"' Erreur import: {e}")
    print("   Installez: pip install scikit-learn xgboost")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """
    Entraneur de modeles ML pour le trading
    
    Entrane 3 modeles:
    - RandomForest: bon equilibre performance/vitesse
    - XGBoost: meilleure precision
    - LogisticRegression: rapide et interpretable
    """
    
    def __init__(self):
        """Initialise le trainer"""
        self.models_dir = Path('data/models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Definir les modeles
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                n_jobs=4,
                random_state=42,
                verbose=1
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
                random_state=42,
                verbose=1
            )
        }
        
        logger.info(""" ML Model Trainer initialise")
        logger.info(f"   Modeles: {', '.join(self.models.keys())}")
        logger.info(f"   Dossier: {self.models_dir}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare les 30 features pour le ML
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec features
        """
        logger.info("" Preparation des features...")
        
        df = df.copy()
        
        # Prix (5 features)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_15'] = df['close'].pct_change(15)
        df['price_change_60'] = df['close'].pct_change(60)
        
        # Position dans le range 24h
        high_24h = df['high'].rolling(288).max()
        low_24h = df['low'].rolling(288).min()
        df['price_position_24h'] = (df['close'] - low_24h) / (high_24h - low_24h + 1e-10)
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['distance_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume (5 features)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        df['volume_trend'] = df['volume'].pct_change(10)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
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
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=9).mean()
        df['ema_slow'] = df['close'].ewm(span=21).mean()
        df['ema_trend'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        
        # Support/Resistance
        df['support_dist'] = (df['close'] - df['low'].rolling(50).min()) / df['close']
        df['resistance_dist'] = (df['high'].rolling(50).max() - df['close']) / df['close']
        
        # Momentum
        df['momentum'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['volatility_ma'] = df['volatility'].rolling(50).mean()
        df['volatility_regime'] = df['volatility'] / (df['volatility_ma'] + 1e-10)
        
        logger.info(f"   ""| {len(df.columns)} colonnes creees")
        
        return df
    
    def create_labels(self, df: pd.DataFrame, forward_window: int = 10, threshold: float = 0.005) -> pd.Series:
        """
        Cree les labels pour l'entranement
        
        Args:
            df: DataFrame avec prix
            forward_window: Fenetre de prediction (candles)
            threshold: Seuil de mouvement (0.5% par defaut)
            
        Returns:
            Series de labels (0=SELL, 1=BUY, 2=HOLD)
        """
        logger.info(f"  Creation des labels (forward={forward_window}, threshold={threshold})")
        
        # Calculer le rendement futur
        future_returns = df['close'].shift(-forward_window) / df['close'] - 1
        
        # Creer les labels
        labels = pd.Series(2, index=df.index)  # 2 = HOLD par defaut
        labels[future_returns > threshold] = 1  # BUY
        labels[future_returns < -threshold] = 0  # SELL
        
        # Stats
        buy_count = (labels == 1).sum()
        sell_count = (labels == 0).sum()
        hold_count = (labels == 2).sum()
        
        logger.info(f"   BUY: {buy_count} ({buy_count/len(labels):.1%})")
        logger.info(f"   SELL: {sell_count} ({sell_count/len(labels):.1%})")
        logger.info(f"   HOLD: {hold_count} ({hold_count/len(labels):.1%})")
        
        return labels
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """
        Entrane tous les modeles
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion de test
            
        Returns:
            Dict avec resultats
        """
        logger.info(" Debut de l'entranement...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\n" Entranement {name.upper()}...")
            
            try:
                # Entranement
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metriques
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                logger.info(f"   Train Accuracy: {train_acc:.2%}")
                logger.info(f"   Test Accuracy: {test_acc:.2%}")
                logger.info(f"   CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
                
                # Classification report
                logger.info("\n" + classification_report(y_test, y_pred_test, 
                                                        target_names=['SELL', 'BUY', 'HOLD']))
                
                # Sauvegarder le modele
                self._save_model(model, name)
                
                results[name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model
                }
                
                logger.info(f"   ""| {name.upper()} entrane et sauvegarde")
                
            except Exception as e:
                logger.error(f"   ' Erreur entranement {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _save_model(self, model, name: str):
        """
        Sauvegarde un modele
        
        Args:
            model: Le modele entrane
            name: Nom du modele
        """
        filepath = self.models_dir / f"{name}_model.joblib"
        joblib.dump(model, filepath)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"   ' Sauvegarde: {filepath.name} ({size_mb:.2f} MB)")
    
    def load_model(self, name: str):
        """
        Charge un modele sauvegarde
        
        Args:
            name: Nom du modele
            
        Returns:
            Modele charge ou None
        """
        filepath = self.models_dir / f"{name}_model.joblib"
        
        if not filepath.exists():
            logger.warning(f"Modele {name} non trouve: {filepath}")
            return None
        
        try:
            model = joblib.load(filepath)
            logger.info(f"""| Modele {name} charge")
            return model
        except Exception as e:
            logger.error(f"Erreur chargement {name}: {e}")
            return None
    
    def list_saved_models(self) -> List[dict]:
        """
        Liste les modeles sauvegardes
        
        Returns:
            Liste des modeles
        """
        models = []
        
        for file in self.models_dir.glob('*_model.joblib'):
            stat = file.stat()
            models.append({
                'name': file.stem.replace('_model', ''),
                'path': str(file),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        
        return models


def main():
    """Point d'entree du script"""
    parser = argparse.ArgumentParser(description='Entranement des modeles ML')
    parser.add_argument('--data', required=True,
                       help='Fichier CSV avec donnees historiques')
    parser.add_argument('--forward-window', type=int, default=10,
                       help='Fenetre de prediction en candles (defaut: 10)')
    parser.add_argument('--threshold', type=float, default=0.005,
                       help='Seuil de mouvement pour labels (defaut: 0.005)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion de test (defaut: 0.2)')
    parser.add_argument('--list', action='store_true',
                       help='Liste les modeles sauvegardes')
    
    args = parser.parse_args()
    
    trainer = MLModelTrainer()
    
    print("\n" + "="*50)
    print(""" ENTRANEMENT DES MODLES ML")
    print("="*50)
    
    if args.list:
        models = trainer.list_saved_models()
        print(f"\n"" Modeles sauvegardes: {len(models)}\n")
        
        for model in models:
            print(f"" {model['name'].upper()}")
            print(f"  Taille: {model['size_mb']:.2f} MB")
            print(f"  Modifie: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        print("="*50 + "\n")
        sys.exit(0)
    
    # Charger les donnees
    logger.info(f"" Chargement des donnees: {args.data}"")
    try:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        logger.info(f"   ""| {len(df)} candles chargees")
        logger.info(f"   Periode: {df.index[0]} -> {df.index[-1]}")
    except Exception as e:
        logger.error(f"' Erreur chargement: {e}")
        sys.exit(1)
    
    # Preparer les features
    df = trainer.prepare_features(df)
    
    # Creer les labels
    labels = trainer.create_labels(df, args.forward_window, args.threshold)
    
    # Selectionner les features
    feature_cols = [
        'price_change_5', 'price_change_15', 'price_change_60',
        'price_position_24h', 'distance_from_vwap',
        'volume_ratio', 'volume_trend',
        'rsi', 'macd_hist', 'bb_position',
        'atr_normalized', 'ema_trend', 'stoch_k',
        'support_dist', 'resistance_dist',
        'momentum', 'volatility_regime'
    ]
    
    # Garder seulement les colonnes disponibles
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"\n" Features utilisees: {len(available_features)}"")
    
    X = df[available_features].dropna()
    y = labels.loc[X.index]
    
    logger.info(f"   Samples finaux: {len(X)}")
    
    # Entraner
    results = trainer.train_models(X, y, args.test_size)
    
    # Resume
    print("\n" + "="*50)
    print("" R"SUM" DE L'ENTRANEMENT")
    print("="*50)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"\n' {name.upper()}: "CHEC")
            print(f"   Erreur: {result['error']}")
        else:
            print(f"\n""| {name.upper()}")
            print(f"   Test Accuracy: {result['test_accuracy']:.2%}")
            print(f"   CV Score: {result['cv_mean']:.2%}  {result['cv_std']:.2%}")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
