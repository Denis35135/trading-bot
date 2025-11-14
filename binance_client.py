"""
Client Binance complet pour The Bot
GÃƒÂ¨re REST API + WebSocket + Ordres + Reconnexion automatique
"""

import time
import json
import logging
import threading
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
# WebSocket gÃ©rÃ© via binance.streams (ancienne API websockets supprimÃ©e)
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Client Binance complet avec toutes les fonctionnalitÃƒÂ©s nÃƒÂ©cessaires
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        """
        Initialise le client Binance
        
        Args:
            api_key: ClÃƒÂ© API Binance
            secret_key: ClÃƒÂ© secrÃƒÂ¨te Binance
            testnet: Utiliser le testnet (pour tests)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        # Client REST API
        self.client = None
        
        # WebSocket Manager
        self.bm = None
        self.ws_connections = {}
        
        # Cache des donnÃƒÂ©es
        self.symbols_info = {}
        self.account_info = {}
        self.open_orders = {}
        self.positions = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.request_weight = 0
        self.max_request_weight = 1200  # Limite Binance par minute
        
        # Callbacks WebSocket
        self.callbacks = {
            'price': [],
            'orderbook': [],
            'trade': []
        }
        
        # Ãƒâ€°tat
        self.connected = False
        self.ws_running = False
        
        # Initialisation
        self._initialize_client()
        self._load_exchange_info()
        
    def _initialize_client(self):
        """Initialise le client REST API"""
        try:
            if self.testnet:
                # Configuration testnet
                self.client = Client(
                    self.api_key,
                    self.secret_key,
                    testnet=True
                )
                logger.info("Client initialisÃƒÂ© en mode TESTNET")
            else:
                # Production
                self.client = Client(self.api_key, self.secret_key)
                logger.info("Client initialisÃƒÂ© en mode PRODUCTION")
                
            # Test de connexion
            self.client.ping()
            self.connected = True
            logger.info("Ã¢Å“â€¦ Connexion Binance ÃƒÂ©tablie")
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur initialisation: {e}")
            raise
    
    def _load_exchange_info(self):
        """Charge les informations de l'exchange (symboles, limites, etc.)"""
        try:
            info = self.client.get_exchange_info()
            
            for symbol in info['symbols']:
                if symbol['status'] == 'TRADING':
                    self.symbols_info[symbol['symbol']] = {
                        'base': symbol['baseAsset'],
                        'quote': symbol['quoteAsset'],
                        'min_qty': 0,
                        'max_qty': 0,
                        'step_size': 0,
                        'min_notional': 0,
                        'tick_size': 0
                    }
                    
                    # Parse les filtres
                    for filter in symbol['filters']:
                        if filter['filterType'] == 'LOT_SIZE':
                            self.symbols_info[symbol['symbol']]['min_qty'] = float(filter['minQty'])
                            self.symbols_info[symbol['symbol']]['max_qty'] = float(filter['maxQty'])
                            self.symbols_info[symbol['symbol']]['step_size'] = float(filter['stepSize'])
                        elif filter['filterType'] == 'MIN_NOTIONAL':
                            self.symbols_info[symbol['symbol']]['min_notional'] = float(filter['minNotional'])
                        elif filter['filterType'] == 'PRICE_FILTER':
                            self.symbols_info[symbol['symbol']]['tick_size'] = float(filter['tickSize'])
            
            logger.info(f"Ã¢Å“â€¦ {len(self.symbols_info)} symboles chargÃƒÂ©s")
            
        except Exception as e:
            logger.error(f"Erreur chargement exchange info: {e}")
    
    # =============================================================
    # MÃƒâ€°THODES REST API
    # =============================================================
    
    def get_account_balance(self, asset: str = 'USDC') -> float:
        """
        RÃƒÂ©cupÃƒÂ¨re la balance d'un actif
        
        Args:
            asset: L'actif (dÃƒÂ©faut: USDC)
            
        Returns:
            Balance disponible
        """
        try:
            account = self.client.get_account()
            
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration balance: {e}")
            return 0.0
    
    def get_symbol_ticker(self, symbol: str) -> Dict:
        """
        RÃƒÂ©cupÃƒÂ¨re le ticker d'un symbole
        
        Args:
            symbol: Le symbole (ex: BTCUSDC)
            
        Returns:
            Dict avec prix, volume, etc.
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            
            return {
                'symbol': ticker['symbol'],
                'price': float(ticker['lastPrice']),
                'bid': float(ticker['bidPrice']),
                'ask': float(ticker['askPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'change_24h': float(ticker['priceChangePercent'])
            }
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration ticker {symbol}: {e}")
            return None
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        RÃƒÂ©cupÃƒÂ¨re l'orderbook d'un symbole
        
        Args:
            symbol: Le symbole
            limit: Profondeur (5, 10, 20, 50, 100)
            
        Returns:
            Dict avec bids et asks
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            
            return {
                'bids': [[float(price), float(qty)] for price, qty in depth['bids']],
                'asks': [[float(price), float(qty)] for price, qty in depth['asks']],
                'timestamp': depth.get('lastUpdateId', 0)
            }
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration orderbook {symbol}: {e}")
            return {'bids': [], 'asks': [], 'timestamp': 0}
    
    def get_klines(self, symbol: str, interval: str = '5m', limit: int = 100) -> pd.DataFrame:
        """
        RÃƒÂ©cupÃƒÂ¨re les chandeliers (klines)
        
        Args:
            symbol: Le symbole
            interval: Intervalle (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Nombre de chandeliers
            
        Returns:
            DataFrame avec OHLCV
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convertir en DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convertir les types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration klines {symbol}: {e}")
            return pd.DataFrame()
    
    def get_24h_stats(self, symbol: str = None) -> List[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re les statistiques 24h
        
        Args:
            symbol: Symbole spÃƒÂ©cifique ou None pour tous
            
        Returns:
            Liste des stats 24h
        """
        try:
            if symbol:
                stats = [self.client.get_ticker(symbol=symbol)]
            else:
                stats = self.client.get_ticker()
            
            result = []
            for s in stats:
                result.append({
                    'symbol': s['symbol'],
                    'volume': float(s['volume']),
                    'quote_volume': float(s['quoteVolume']),
                    'price_change': float(s['priceChangePercent']),
                    'high': float(s['highPrice']),
                    'low': float(s['lowPrice'])
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration stats 24h: {e}")
            return []
    
    # =============================================================
    # GESTION DES ORDRES
    # =============================================================
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Arrondit la quantitÃƒÂ© selon les rÃƒÂ¨gles du symbole"""
        if symbol not in self.symbols_info:
            return quantity
            
        step_size = self.symbols_info[symbol]['step_size']
        if step_size == 0:
            return quantity
            
        precision = int(round(-1 * (Decimal(str(step_size)).log10())))
        quantity = round(quantity, precision)
        
        return quantity
    
    def _round_price(self, symbol: str, price: float) -> float:
        """Arrondit le prix selon les rÃƒÂ¨gles du symbole"""
        if symbol not in self.symbols_info:
            return price
            
        tick_size = self.symbols_info[symbol]['tick_size']
        if tick_size == 0:
            return price
            
        precision = int(round(-1 * (Decimal(str(tick_size)).log10())))
        price = round(price, precision)
        
        return price
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Place un ordre au marchÃƒÂ©
        
        Args:
            symbol: Le symbole
            side: 'BUY' ou 'SELL'
            quantity: QuantitÃƒÂ© (sera arrondie automatiquement)
            
        Returns:
            DÃƒÂ©tails de l'ordre ou None si ÃƒÂ©chec
        """
        try:
            # Arrondir la quantitÃƒÂ©
            quantity = self._round_quantity(symbol, quantity)
            
            # VÃƒÂ©rifier les limites
            if symbol in self.symbols_info:
                min_qty = self.symbols_info[symbol]['min_qty']
                max_qty = self.symbols_info[symbol]['max_qty']
                
                if quantity < min_qty:
                    logger.error(f"QuantitÃƒÂ© {quantity} < minimum {min_qty}")
                    return None
                if quantity > max_qty:
                    quantity = max_qty
            
            # Placer l'ordre
            order = self.client.order_market(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            logger.info(f"Ã¢Å“â€¦ Ordre MARKET {side} placÃƒÂ©: {quantity} {symbol}")
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'quantity': float(order['origQty']),
                'status': order['status'],
                'fills': order.get('fills', [])
            }
            
        except BinanceOrderException as e:
            logger.error(f"Erreur ordre: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue ordre: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """
        Place un ordre limite
        
        Args:
            symbol: Le symbole
            side: 'BUY' ou 'SELL'
            quantity: QuantitÃƒÂ©
            price: Prix limite
            
        Returns:
            DÃƒÂ©tails de l'ordre ou None si ÃƒÂ©chec
        """
        try:
            # Arrondir quantitÃƒÂ© et prix
            quantity = self._round_quantity(symbol, quantity)
            price = self._round_price(symbol, price)
            
            # VÃƒÂ©rifier notional minimum
            if symbol in self.symbols_info:
                min_notional = self.symbols_info[symbol]['min_notional']
                if quantity * price < min_notional:
                    logger.error(f"Notional {quantity * price} < minimum {min_notional}")
                    return None
            
            # Placer l'ordre
            order = self.client.order_limit(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price
            )
            
            logger.info(f"Ã¢Å“â€¦ Ordre LIMIT {side} placÃƒÂ©: {quantity} {symbol} @ {price}")
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'quantity': float(order['origQty']),
                'price': float(order['price']),
                'status': order['status']
            }
            
        except BinanceOrderException as e:
            logger.error(f"Erreur ordre limite: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue ordre limite: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """
        Annule un ordre
        
        Args:
            symbol: Le symbole
            order_id: L'ID de l'ordre
            
        Returns:
            True si succÃƒÂ¨s, False sinon
        """
        try:
            result = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            logger.info(f"Ã¢Å“â€¦ Ordre {order_id} annulÃƒÂ©")
            return True
            
        except Exception as e:
            logger.error(f"Erreur annulation ordre {order_id}: {e}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        RÃƒÂ©cupÃƒÂ¨re les ordres ouverts
        
        Args:
            symbol: Symbole spÃƒÂ©cifique ou None pour tous
            
        Returns:
            Liste des ordres ouverts
        """
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            
            result = []
            for order in orders:
                result.append({
                    'order_id': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'quantity': float(order['origQty']),
                    'price': float(order['price']) if order['price'] != '0.00000000' else 0,
                    'status': order['status'],
                    'time': order['time']
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration ordres ouverts: {e}")
            return []
    
    def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """
        RÃƒÂ©cupÃƒÂ¨re le statut d'un ordre
        
        Args:
            symbol: Le symbole
            order_id: L'ID de l'ordre
            
        Returns:
            Statut de l'ordre
        """
        try:
            order = self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )
            
            return {
                'order_id': order['orderId'],
                'status': order['status'],
                'filled_qty': float(order['executedQty']),
                'remaining_qty': float(order['origQty']) - float(order['executedQty'])
            }
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration statut ordre {order_id}: {e}")
            return None
    
    # =============================================================
    # WEBSOCKET STREAMING
    # =============================================================
    
    def start_websocket(self):
        """DÃƒÂ©marre le WebSocket manager"""
        try:
            self.bm = BinanceSocketManager(self.client)
            self.bm.start()
            self.ws_running = True
            logger.info("Ã¢Å“â€¦ WebSocket manager dÃƒÂ©marrÃƒÂ©")
            
        except Exception as e:
            logger.error(f"Erreur dÃƒÂ©marrage WebSocket: {e}")
    
    def stop_websocket(self):
        """ArrÃƒÂªte le WebSocket manager"""
        try:
            if self.bm:
                self.bm.stop()
                self.ws_running = False
                logger.info("WebSocket manager arrÃƒÂªtÃƒÂ©")
                
        except Exception as e:
            logger.error(f"Erreur arrÃƒÂªt WebSocket: {e}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        Souscrit au ticker d'un symbole
        
        Args:
            symbol: Le symbole
            callback: Fonction ÃƒÂ  appeler ÃƒÂ  chaque update
        """
        try:
            def process_message(msg):
                if msg['e'] == 'error':
                    logger.error(f"Erreur WebSocket ticker: {msg['m']}")
                    return
                    
                data = {
                    'symbol': msg['s'],
                    'price': float(msg['c']),
                    'bid': float(msg['b']),
                    'ask': float(msg['a']),
                    'volume': float(msg['v']),
                    'timestamp': msg['E']
                }
                callback(data)
            
            conn_key = self.bm.start_symbol_ticker_socket(symbol, process_message)
            self.ws_connections[f'ticker_{symbol}'] = conn_key
            logger.info(f"Ã¢Å“â€¦ Souscription ticker {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur souscription ticker {symbol}: {e}")
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """
        Souscrit aux chandeliers d'un symbole
        
        Args:
            symbol: Le symbole
            interval: Intervalle (1m, 5m, 15m, etc.)
            callback: Fonction callback
        """
        try:
            def process_message(msg):
                if msg['e'] == 'error':
                    logger.error(f"Erreur WebSocket kline: {msg['m']}")
                    return
                    
                k = msg['k']
                data = {
                    'symbol': k['s'],
                    'interval': k['i'],
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v']),
                    'is_closed': k['x'],
                    'timestamp': k['t']
                }
                callback(data)
            
            conn_key = self.bm.start_kline_socket(symbol, process_message, interval=interval)
            self.ws_connections[f'kline_{symbol}_{interval}'] = conn_key
            logger.info(f"Ã¢Å“â€¦ Souscription kline {symbol} {interval}")
            
        except Exception as e:
            logger.error(f"Erreur souscription kline {symbol}: {e}")
    
    def subscribe_orderbook(self, symbol: str, callback: Callable, depth: int = 20):
        """
        Souscrit ÃƒÂ  l'orderbook d'un symbole
        
        Args:
            symbol: Le symbole
            callback: Fonction callback
            depth: Profondeur (5, 10, 20)
        """
        try:
            def process_message(msg):
                if 'e' in msg and msg['e'] == 'error':
                    logger.error(f"Erreur WebSocket orderbook: {msg['m']}")
                    return
                    
                data = {
                    'symbol': symbol,
                    'bids': [[float(p), float(q)] for p, q in msg.get('bids', [])],
                    'asks': [[float(p), float(q)] for p, q in msg.get('asks', [])],
                    'timestamp': msg.get('lastUpdateId', 0)
                }
                callback(data)
            
            if depth == 5:
                conn_key = self.bm.start_depth_socket(symbol, process_message, depth=5)
            elif depth == 10:
                conn_key = self.bm.start_depth_socket(symbol, process_message, depth=10)
            else:
                conn_key = self.bm.start_depth_socket(symbol, process_message)
                
            self.ws_connections[f'depth_{symbol}'] = conn_key
            logger.info(f"Ã¢Å“â€¦ Souscription orderbook {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur souscription orderbook {symbol}: {e}")
    
    def subscribe_trades(self, symbol: str, callback: Callable):
        """
        Souscrit aux trades d'un symbole
        
        Args:
            symbol: Le symbole
            callback: Fonction callback
        """
        try:
            def process_message(msg):
                if msg['e'] == 'error':
                    logger.error(f"Erreur WebSocket trades: {msg['m']}")
                    return
                    
                data = {
                    'symbol': msg['s'],
                    'price': float(msg['p']),
                    'quantity': float(msg['q']),
                    'is_buyer_maker': msg['m'],
                    'timestamp': msg['T']
                }
                callback(data)
            
            conn_key = self.bm.start_trade_socket(symbol, process_message)
            self.ws_connections[f'trades_{symbol}'] = conn_key
            logger.info(f"Ã¢Å“â€¦ Souscription trades {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur souscription trades {symbol}: {e}")
    
    def unsubscribe(self, connection_name: str):
        """
        DÃƒÂ©souscrit d'un stream WebSocket
        
        Args:
            connection_name: Nom de la connexion
        """
        try:
            if connection_name in self.ws_connections:
                self.bm.stop_socket(self.ws_connections[connection_name])
                del self.ws_connections[connection_name]
                logger.info(f"DÃƒÂ©souscription {connection_name}")
                
        except Exception as e:
            logger.error(f"Erreur dÃƒÂ©souscription {connection_name}: {e}")
    
    # =============================================================
    # MÃƒâ€°THODES UTILITAIRES
    # =============================================================
    
    def test_connection(self) -> bool:
        """Test la connexion ÃƒÂ  Binance"""
        try:
            self.client.ping()
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time['serverTime'] - local_time)
            
            if time_diff > 5000:  # 5 secondes de diffÃƒÂ©rence max
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â DÃƒÂ©calage horaire important: {time_diff}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Test connexion ÃƒÂ©chouÃƒÂ©: {e}")
            return False
    
    def get_exchange_status(self) -> Dict:
        """RÃƒÂ©cupÃƒÂ¨re le statut de l'exchange"""
        try:
            status = self.client.get_system_status()
            
            return {
                'status': status['status'] == 0,  # 0 = normal, 1 = maintenance
                'msg': status.get('msg', 'Normal')
            }
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration statut: {e}")
            return {'status': False, 'msg': str(e)}
    
    def get_trading_fees(self, symbol: str = None) -> Dict:
        """
        RÃƒÂ©cupÃƒÂ¨re les frais de trading
        
        Args:
            symbol: Symbole spÃƒÂ©cifique ou None pour les frais gÃƒÂ©nÃƒÂ©raux
            
        Returns:
            Frais maker et taker
        """
        try:
            if symbol:
                fees = self.client.get_trade_fee(symbol=symbol)
                return {
                    'maker': float(fees[0]['makerCommission']),
                    'taker': float(fees[0]['takerCommission'])
                }
            else:
                account = self.client.get_account()
                return {
                    'maker': float(account['makerCommission']) / 10000,
                    'taker': float(account['takerCommission']) / 10000
                }
                
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration frais: {e}")
            # Frais par dÃƒÂ©faut Binance
            return {'maker': 0.001, 'taker': 0.001}
    
    def calculate_quantity_from_usdc(self, symbol: str, usdc_amount: float) -> float:
        """
        Calcule la quantitÃƒÂ© ÃƒÂ  acheter avec un montant USDC
        
        Args:
            symbol: Le symbole
            usdc_amount: Montant en USDC
            
        Returns:
            QuantitÃƒÂ© arrondie selon les rÃƒÂ¨gles du symbole
        """
        try:
            # RÃƒÂ©cupÃƒÂ¨re le prix actuel
            ticker = self.get_symbol_ticker(symbol)
            if not ticker:
                return 0
            
            price = ticker['price']
            
            # Calcule la quantitÃƒÂ©
            quantity = usdc_amount / price
            
            # Arrondit selon les rÃƒÂ¨gles
            quantity = self._round_quantity(symbol, quantity)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur calcul quantitÃƒÂ©: {e}")
            return 0
    
    def close(self):
        """Ferme proprement le client"""
        try:
            # ArrÃƒÂªte WebSocket
            if self.ws_running:
                self.stop_websocket()
            
            # Clear cache
            self.symbols_info.clear()
            self.account_info.clear()
            self.open_orders.clear()
            self.positions.clear()
            
            self.connected = False
            logger.info("Client Binance fermÃƒÂ©")
            
        except Exception as e:
            logger.error(f"Erreur fermeture client: {e}")


# =============================================================
# EXEMPLE D'UTILISATION
# =============================================================

if __name__ == "__main__":
    """Test du client Binance"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Remplace par tes clÃƒÂ©s API
    API_KEY = "your_api_key"
    SECRET_KEY = "your_secret_key"
    
    # Initialise le client
    client = BinanceClient(API_KEY, SECRET_KEY, testnet=True)
    
    # Test balance
    balance = client.get_account_balance('USDC')
    print(f"Balance USDC: {balance}")
    
    # Test ticker
    ticker = client.get_symbol_ticker('BTCUSDC')
    if ticker:
        print(f"BTC Prix: ${ticker['price']:,.2f}")
    
    # Test klines
    df = client.get_klines('BTCUSDC', '5m', limit=20)
    if not df.empty:
        print(f"\nDerniÃƒÂ¨res bougies:")
        print(df.tail())
    
    # Test WebSocket
    def on_price_update(data):
        print(f"Prix update: {data['symbol']} = ${data['price']:,.2f}")
    
    client.start_websocket()
    client.subscribe_ticker('BTCUSDC', on_price_update)
    
    # Attendre quelques secondes
    time.sleep(10)
    
    # Cleanup
    client.close()