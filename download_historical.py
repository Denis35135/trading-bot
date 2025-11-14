#!/usr/bin/env python3
"""
Script de tÃƒÂ©lÃƒÂ©chargement de donnÃƒÂ©es historiques
TÃƒÂ©lÃƒÂ©charge les donnÃƒÂ©es OHLCV de Binance pour backtesting et training ML
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import time
import argparse
from typing import List

# Ajouter le rÃƒÂ©pertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("Ã¢ÂÅ’ Erreur: python-binance non installÃƒÂ©")
    print("   Installez avec: pip install python-binance")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataDownloader:
    """
    TÃƒÂ©lÃƒÂ©chargeur de donnÃƒÂ©es historiques Binance
    
    FonctionnalitÃƒÂ©s:
    - TÃƒÂ©lÃƒÂ©chargement multi-symboles
    - Plusieurs timeframes
    - Sauvegarde en CSV et Parquet
    - Gestion de la limite de rate API
    - Reprise en cas d'interruption
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialise le downloader
        
        Args:
            api_key: ClÃƒÂ© API Binance (optionnel pour donnÃƒÂ©es publiques)
            api_secret: Secret API Binance
        """
        self.client = Client(api_key, api_secret)
        self.data_dir = Path('data/historical')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Limites API Binance
        self.max_klines_per_request = 1000
        self.request_delay = 0.5  # 500ms entre requÃƒÂªtes
        
        logger.info("Ã°Å¸â€œÂ¥ Historical Data Downloader initialisÃƒÂ©")
        logger.info(f"   Dossier donnÃƒÂ©es: {self.data_dir}")
    
    def download_symbol_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str = None,
        save_format: str = 'csv'
    ) -> pd.DataFrame:
        """
        TÃƒÂ©lÃƒÂ©charge les donnÃƒÂ©es historiques pour un symbole
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            interval: Intervalle (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Date de dÃƒÂ©but (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD), dÃƒÂ©faut: aujourd'hui
            save_format: Format de sauvegarde (csv ou parquet)
            
        Returns:
            DataFrame avec les donnÃƒÂ©es
        """
        try:
            logger.info(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement {symbol} - {interval}")
            logger.info(f"   PÃƒÂ©riode: {start_date} -> {end_date or 'aujourd\'hui'}")
            
            # Convertir les dates
            start_ts = self._date_to_timestamp(start_date)
            end_ts = self._date_to_timestamp(end_date) if end_date else int(time.time() * 1000)
            
            # TÃƒÂ©lÃƒÂ©charger les donnÃƒÂ©es par chunks
            all_klines = []
            current_ts = start_ts
            
            while current_ts < end_ts:
                try:
                    # RequÃƒÂªte API
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=current_ts,
                        endTime=end_ts,
                        limit=self.max_klines_per_request
                    )
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    
                    # Mettre ÃƒÂ  jour le timestamp
                    current_ts = klines[-1][0] + 1
                    
                    # Afficher la progression
                    current_date = datetime.fromtimestamp(current_ts / 1000)
                    logger.info(f"   Progress: {current_date.strftime('%Y-%m-%d')} "
                              f"({len(all_klines)} candles)")
                    
                    # DÃƒÂ©lai pour respecter les limites API
                    time.sleep(self.request_delay)
                    
                except BinanceAPIException as e:
                    logger.error(f"   Erreur API: {e}")
                    if e.code == -1121:  # Invalid symbol
                        logger.error(f"   Symbole invalide: {symbol}")
                        return None
                    time.sleep(5)  # Attendre plus longtemps en cas d'erreur
                    continue
            
            if not all_klines:
                logger.warning(f"   Aucune donnÃƒÂ©e trouvÃƒÂ©e pour {symbol}")
                return None
            
            # Convertir en DataFrame
            df = self._klines_to_dataframe(all_klines)
            
            # Sauvegarder
            self._save_data(df, symbol, interval, save_format)
            
            logger.info(f"   Ã¢Å“â€¦ TÃƒÂ©lÃƒÂ©chargÃƒÂ©: {len(df)} candles")
            logger.info(f"   PÃƒÂ©riode: {df.index[0]} -> {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur tÃƒÂ©lÃƒÂ©chargement {symbol}: {e}")
            return None
    
    def download_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        start_date: str,
        end_date: str = None,
        save_format: str = 'csv'
    ) -> dict:
        """
        TÃƒÂ©lÃƒÂ©charge les donnÃƒÂ©es pour plusieurs symboles
        
        Args:
            symbols: Liste des symboles
            interval: Intervalle
            start_date: Date de dÃƒÂ©but
            end_date: Date de fin
            save_format: Format de sauvegarde
            
        Returns:
            Dict {symbol: DataFrame}
        """
        logger.info(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement de {len(symbols)} symboles")
        
        results = {}
        failed = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Traitement de {symbol}")
            
            df = self.download_symbol_data(
                symbol, interval, start_date, end_date, save_format
            )
            
            if df is not None:
                results[symbol] = df
            else:
                failed.append(symbol)
            
            # Pause entre symboles
            time.sleep(1)
        
        logger.info(f"\nÃ¢Å“â€¦ TerminÃƒÂ©!")
        logger.info(f"   RÃƒÂ©ussis: {len(results)}/{len(symbols)}")
        
        if failed:
            logger.warning(f"   Ãƒâ€°checs: {', '.join(failed)}")
        
        return results
    
    def download_top_symbols(
        self,
        n: int,
        interval: str,
        start_date: str,
        end_date: str = None,
        save_format: str = 'csv',
        quote_asset: str = 'USDC'
    ) -> dict:
        """
        TÃƒÂ©lÃƒÂ©charge les N meilleurs symboles par volume
        
        Args:
            n: Nombre de symboles
            interval: Intervalle
            start_date: Date de dÃƒÂ©but
            end_date: Date de fin
            save_format: Format
            quote_asset: Asset de quote (USDC, USDT, BTC)
            
        Returns:
            Dict des donnÃƒÂ©es tÃƒÂ©lÃƒÂ©chargÃƒÂ©es
        """
        logger.info(f"Ã°Å¸â€Â Recherche des top {n} symboles {quote_asset}")
        
        try:
            # RÃƒÂ©cupÃƒÂ©rer les stats 24h
            tickers = self.client.get_ticker()
            
            # Filtrer par quote asset
            filtered = [
                t for t in tickers 
                if t['symbol'].endswith(quote_asset)
            ]
            
            # Trier par volume
            sorted_symbols = sorted(
                filtered,
                key=lambda x: float(x['quoteVolume']),
                reverse=True
            )
            
            # Prendre les top N
            top_symbols = [t['symbol'] for t in sorted_symbols[:n]]
            
            logger.info(f"   Top {n}: {', '.join(top_symbols[:5])}...")
            
            # TÃƒÂ©lÃƒÂ©charger
            return self.download_multiple_symbols(
                top_symbols, interval, start_date, end_date, save_format
            )
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur rÃƒÂ©cupÃƒÂ©ration top symboles: {e}")
            return {}
    
    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """
        Convertit les klines Binance en DataFrame
        
        Args:
            klines: Liste des klines de l'API
            
        Returns:
            DataFrame OHLCV
        """
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convertir les types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        # Garder les colonnes importantes
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _save_data(self, df: pd.DataFrame, symbol: str, interval: str, format: str):
        """
        Sauvegarde les donnÃƒÂ©es
        
        Args:
            df: DataFrame ÃƒÂ  sauvegarder
            symbol: Symbole
            interval: Intervalle
            format: Format (csv ou parquet)
        """
        try:
            filename = f"{symbol}_{interval}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"
            
            if format == 'csv':
                filepath = self.data_dir / f"{filename}.csv"
                df.to_csv(filepath)
            elif format == 'parquet':
                filepath = self.data_dir / f"{filename}.parquet"
                df.to_parquet(filepath)
            else:
                logger.warning(f"   Format inconnu: {format}, utilisation de CSV")
                filepath = self.data_dir / f"{filename}.csv"
                df.to_csv(filepath)
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"   Ã°Å¸â€™Â¾ SauvegardÃƒÂ©: {filepath.name} ({size_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"   Ã¢Å“â€” Erreur sauvegarde: {e}")
    
    def _date_to_timestamp(self, date_str: str) -> int:
        """
        Convertit une date en timestamp ms
        
        Args:
            date_str: Date au format YYYY-MM-DD
            
        Returns:
            Timestamp en millisecondes
        """
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.timestamp() * 1000)
    
    def list_downloaded_data(self) -> List[dict]:
        """
        Liste les donnÃƒÂ©es tÃƒÂ©lÃƒÂ©chargÃƒÂ©es
        
        Returns:
            Liste des fichiers disponibles
        """
        files = []
        
        for file in self.data_dir.iterdir():
            if file.suffix in ['.csv', '.parquet']:
                stat = file.stat()
                files.append({
                    'name': file.name,
                    'path': str(file),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
        
        files.sort(key=lambda x: x['modified'], reverse=True)
        return files
    
    def load_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Charge des donnÃƒÂ©es sauvegardÃƒÂ©es
        
        Args:
            symbol: Symbole
            interval: Intervalle
            
        Returns:
            DataFrame ou None
        """
        # Chercher les fichiers correspondants
        pattern = f"{symbol}_{interval}_*"
        
        for file in self.data_dir.glob(pattern):
            try:
                if file.suffix == '.csv':
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                elif file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    continue
                
                logger.info(f"Ã¢Å“â€¦ ChargÃƒÂ©: {file.name}")
                return df
                
            except Exception as e:
                logger.error(f"Erreur chargement {file.name}: {e}")
        
        logger.warning(f"Aucune donnÃƒÂ©e trouvÃƒÂ©e pour {symbol} {interval}")
        return None


def main():
    """Point d'entrÃƒÂ©e du script"""
    parser = argparse.ArgumentParser(description='TÃƒÂ©lÃƒÂ©chargement donnÃƒÂ©es historiques')
    parser.add_argument('--symbols', nargs='+', help='Liste des symboles')
    parser.add_argument('--top', type=int, help='TÃƒÂ©lÃƒÂ©charger les top N symboles')
    parser.add_argument('--interval', default='5m', 
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
                       help='Intervalle (dÃƒÂ©faut: 5m)')
    parser.add_argument('--start', required=True, help='Date de dÃƒÂ©but (YYYY-MM-DD)')
    parser.add_argument('--end', help='Date de fin (YYYY-MM-DD)')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv',
                       help='Format de sauvegarde')
    parser.add_argument('--list', action='store_true', help='Liste les donnÃƒÂ©es tÃƒÂ©lÃƒÂ©chargÃƒÂ©es')
    
    args = parser.parse_args()
    
    # Load API keys (optionnel)
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    downloader = HistoricalDataDownloader(api_key, api_secret)
    
    print("\n" + "="*50)
    print("Ã°Å¸â€œÂ¥ TÃƒâ€°LÃƒâ€°CHARGEMENT DONNÃƒâ€°ES HISTORIQUES")
    print("="*50)
    
    if args.list:
        files = downloader.list_downloaded_data()
        print(f"\nÃ°Å¸â€œâ€¹ DonnÃƒÂ©es disponibles: {len(files)} fichier(s)\n")
        
        for i, file in enumerate(files, 1):
            print(f"{i}. {file['name']}")
            print(f"   Taille: {file['size_mb']:.2f} MB")
            print(f"   ModifiÃƒÂ©: {file['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
    elif args.symbols:
        downloader.download_multiple_symbols(
            args.symbols,
            args.interval,
            args.start,
            args.end,
            args.format
        )
        
    elif args.top:
        downloader.download_top_symbols(
            args.top,
            args.interval,
            args.start,
            args.end,
            args.format
        )
        
    else:
        print("Ã¢ÂÅ’ Erreur: SpÃƒÂ©cifiez --symbols, --top ou --list")
        sys.exit(1)
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()