"""
WebSocket Handler pour The Bot
Gestion des connexions WebSocket pour les donnÃƒÂ©es en temps rÃƒÂ©el
"""

import json
import threading
import time
from typing import Dict, Optional, Callable, List, Set
from datetime import datetime
import logging
from websocket import WebSocketApp

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    Gestionnaire de WebSocket pour donnÃƒÂ©es temps rÃƒÂ©el
    
    ResponsabilitÃƒÂ©s:
    - Maintenir une connexion WebSocket avec Binance
    - S'abonner aux streams (klines, trades, orderbook)
    - Distribuer les donnÃƒÂ©es aux callbacks
    - Reconnexion automatique
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le WebSocket handler
        
        Args:
            config: Configuration du WebSocket
        """
        self.config = config or {}
        
        # Configuration
        self.base_url = self.config.get('base_url', 'wss://stream.binance.com:9443/ws')
        self.ping_interval = self.config.get('ping_interval', 20)
        self.reconnect_delay = self.config.get('reconnect_delay', 5)
        
        # Ãƒâ€°tat
        self.ws = None
        self.ws_thread = None
        self.is_running = False
        self.is_connected = False
        
        # Subscriptions
        self.subscribed_streams: Set[str] = set()
        self.symbols: Set[str] = set()
        
        # Callbacks par type de stream
        self.callbacks = {
            'kline': [],
            'trade': [],
            'ticker': [],
            'depth': [],
            'aggTrade': []
        }
        
        # Statistiques
        self.stats = {
            'total_messages': 0,
            'messages_per_second': 0,
            'last_message_time': None,
            'reconnections': 0,
            'errors': 0
        }
        
        # Derniers messages par type
        self.last_messages = {}
        
        logger.info("Ã¢Å“â€¦ WebSocket Handler initialisÃƒÂ©")
    
    def start(self):
        """DÃƒÂ©marre le WebSocket"""
        if self.is_running:
            logger.warning("WebSocket dÃƒÂ©jÃƒÂ  en cours d'exÃƒÂ©cution")
            return
        
        self.is_running = True
        
        # DÃƒÂ©marrer le thread WebSocket
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
        
        logger.info("Ã°Å¸Å¡â‚¬ WebSocket dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte le WebSocket"""
        logger.info("Ã°Å¸â€ºâ€˜ ArrÃƒÂªt du WebSocket...")
        
        self.is_running = False
        
        if self.ws:
            self.ws.close()
        
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        
        logger.info("Ã¢Å“â€¦ WebSocket arrÃƒÂªtÃƒÂ©")
    
    def subscribe_kline(self, symbol: str, interval: str = '5m'):
        """
        S'abonne aux klines (bougies)
        
        Args:
            symbol: Symbole (ex: BTCUSDT)
            interval: Intervalle (1m, 5m, 15m, 1h, etc.)
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        self._subscribe_stream(stream)
        self.symbols.add(symbol)
        logger.info(f"Ã°Å¸â€œÅ  AbonnÃƒÂ© aux klines: {symbol} {interval}")
    
    def subscribe_trades(self, symbol: str):
        """
        S'abonne aux trades
        
        Args:
            symbol: Symbole
        """
        stream = f"{symbol.lower()}@trade"
        self._subscribe_stream(stream)
        self.symbols.add(symbol)
        logger.info(f"Ã°Å¸â€™Â¹ AbonnÃƒÂ© aux trades: {symbol}")
    
    def subscribe_ticker(self, symbol: str):
        """
        S'abonne au ticker 24h
        
        Args:
            symbol: Symbole
        """
        stream = f"{symbol.lower()}@ticker"
        self._subscribe_stream(stream)
        self.symbols.add(symbol)
        logger.info(f"Ã°Å¸â€œË† AbonnÃƒÂ© au ticker: {symbol}")
    
    def subscribe_depth(self, symbol: str, levels: int = 10):
        """
        S'abonne au depth (orderbook)
        
        Args:
            symbol: Symbole
            levels: Nombre de niveaux (5, 10, 20)
        """
        stream = f"{symbol.lower()}@depth{levels}"
        self._subscribe_stream(stream)
        self.symbols.add(symbol)
        logger.info(f"Ã°Å¸â€œÅ¡ AbonnÃƒÂ© au depth: {symbol} (levels: {levels})")
    
    def subscribe_agg_trades(self, symbol: str):
        """
        S'abonne aux trades agrÃƒÂ©gÃƒÂ©s
        
        Args:
            symbol: Symbole
        """
        stream = f"{symbol.lower()}@aggTrade"
        self._subscribe_stream(stream)
        self.symbols.add(symbol)
        logger.info(f"Ã°Å¸â€â‚¬ AbonnÃƒÂ© aux aggTrades: {symbol}")
    
    def _subscribe_stream(self, stream: str):
        """Ajoute un stream aux subscriptions"""
        self.subscribed_streams.add(stream)
    
    def unsubscribe(self, symbol: str):
        """
        Se dÃƒÂ©sabonne de tous les streams d'un symbole
        
        Args:
            symbol: Symbole
        """
        streams_to_remove = [s for s in self.subscribed_streams if symbol.lower() in s]
        for stream in streams_to_remove:
            self.subscribed_streams.remove(stream)
        
        if symbol in self.symbols:
            self.symbols.remove(symbol)
        
        logger.info(f"Ã°Å¸Å¡Â« DÃƒÂ©sabonnÃƒÂ© de {symbol}")
    
    def register_callback(self, stream_type: str, callback: Callable):
        """
        Enregistre un callback pour un type de stream
        
        Args:
            stream_type: Type (kline, trade, ticker, depth, aggTrade)
            callback: Fonction ÃƒÂ  appeler lors de la rÃƒÂ©ception de donnÃƒÂ©es
        """
        if stream_type in self.callbacks:
            self.callbacks[stream_type].append(callback)
            logger.info(f"Ã¢Å“â€¦ Callback enregistrÃƒÂ© pour {stream_type}")
        else:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Type de stream inconnu: {stream_type}")
    
    def _run_websocket(self):
        """Thread principal du WebSocket"""
        while self.is_running:
            try:
                # Construire l'URL avec les streams
                if not self.subscribed_streams:
                    logger.warning("Aucun stream abonnÃƒÂ©, attente...")
                    time.sleep(5)
                    continue
                
                streams = '/'.join(self.subscribed_streams)
                url = f"{self.base_url}/{streams}"
                
                logger.info(f"Ã°Å¸â€Å’ Connexion WebSocket: {len(self.subscribed_streams)} streams")
                
                # CrÃƒÂ©er le WebSocket
                self.ws = WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                # DÃƒÂ©marrer (bloquant)
                self.ws.run_forever(ping_interval=self.ping_interval)
                
                # Si on arrive ici, la connexion s'est fermÃƒÂ©e
                if self.is_running:
                    logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â WebSocket fermÃƒÂ©, reconnexion dans {self.reconnect_delay}s...")
                    self.stats['reconnections'] += 1
                    time.sleep(self.reconnect_delay)
                
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur WebSocket: {e}")
                self.stats['errors'] += 1
                
                if self.is_running:
                    time.sleep(self.reconnect_delay)
    
    def _on_message(self, ws, message):
        """Callback appelÃƒÂ© lors de la rÃƒÂ©ception d'un message"""
        try:
            data = json.loads(message)
            
            # Mettre ÃƒÂ  jour les stats
            self.stats['total_messages'] += 1
            self.stats['last_message_time'] = datetime.now()
            
            # Identifier le type de message
            if 'e' in data:
                event_type = data['e']
                
                # Dispatcher vers les callbacks appropriÃƒÂ©s
                if event_type == 'kline':
                    self._handle_kline(data)
                elif event_type == 'trade':
                    self._handle_trade(data)
                elif event_type == '24hrTicker':
                    self._handle_ticker(data)
                elif event_type == 'depthUpdate':
                    self._handle_depth(data)
                elif event_type == 'aggTrade':
                    self._handle_agg_trade(data)
            
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}")
    
    def _on_error(self, ws, error):
        """Callback appelÃƒÂ© en cas d'erreur"""
        logger.error(f"Ã¢ÂÅ’ WebSocket erreur: {error}")
        self.stats['errors'] += 1
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback appelÃƒÂ© ÃƒÂ  la fermeture"""
        self.is_connected = False
        logger.warning(f"Ã°Å¸â€Å’ WebSocket fermÃƒÂ©: {close_status_code} - {close_msg}")
    
    def _on_open(self, ws):
        """Callback appelÃƒÂ© ÃƒÂ  l'ouverture"""
        self.is_connected = True
        logger.info("Ã¢Å“â€¦ WebSocket connectÃƒÂ©")
    
    def _handle_kline(self, data: Dict):
        """Traite les donnÃƒÂ©es kline"""
        kline = data['k']
        
        formatted = {
            'symbol': data['s'],
            'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': kline['x']
        }
        
        self.last_messages['kline'] = formatted
        
        # Appeler les callbacks
        for callback in self.callbacks['kline']:
            try:
                callback(formatted)
            except Exception as e:
                logger.error(f"Erreur callback kline: {e}")
    
    def _handle_trade(self, data: Dict):
        """Traite les donnÃƒÂ©es trade"""
        formatted = {
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'timestamp': datetime.fromtimestamp(data['T'] / 1000),
            'is_buyer_maker': data['m']
        }
        
        self.last_messages['trade'] = formatted
        
        for callback in self.callbacks['trade']:
            try:
                callback(formatted)
            except Exception as e:
                logger.error(f"Erreur callback trade: {e}")
    
    def _handle_ticker(self, data: Dict):
        """Traite les donnÃƒÂ©es ticker"""
        formatted = {
            'symbol': data['s'],
            'price_change': float(data['p']),
            'price_change_percent': float(data['P']),
            'last_price': float(data['c']),
            'volume': float(data['v']),
            'high': float(data['h']),
            'low': float(data['l'])
        }
        
        self.last_messages['ticker'] = formatted
        
        for callback in self.callbacks['ticker']:
            try:
                callback(formatted)
            except Exception as e:
                logger.error(f"Erreur callback ticker: {e}")
    
    def _handle_depth(self, data: Dict):
        """Traite les donnÃƒÂ©es depth"""
        formatted = {
            'symbol': data['s'],
            'bids': [[float(p), float(q)] for p, q in data['b']],
            'asks': [[float(p), float(q)] for p, q in data['a']]
        }
        
        self.last_messages['depth'] = formatted
        
        for callback in self.callbacks['depth']:
            try:
                callback(formatted)
            except Exception as e:
                logger.error(f"Erreur callback depth: {e}")
    
    def _handle_agg_trade(self, data: Dict):
        """Traite les donnÃƒÂ©es aggTrade"""
        formatted = {
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'timestamp': datetime.fromtimestamp(data['T'] / 1000),
            'is_buyer_maker': data['m']
        }
        
        self.last_messages['aggTrade'] = formatted
        
        for callback in self.callbacks['aggTrade']:
            try:
                callback(formatted)
            except Exception as e:
                logger.error(f"Erreur callback aggTrade: {e}")
    
    def get_status(self) -> Dict:
        """Retourne le statut du WebSocket"""
        return {
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'subscribed_streams': len(self.subscribed_streams),
            'symbols': list(self.symbols),
            'stats': self.stats.copy()
        }
    
    def get_last_message(self, stream_type: str) -> Optional[Dict]:
        """
        Retourne le dernier message reÃƒÂ§u d'un type
        
        Args:
            stream_type: Type de stream
            
        Returns:
            Dernier message ou None
        """
        return self.last_messages.get(stream_type)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        # Calculer messages par seconde
        if self.stats['last_message_time']:
            elapsed = (datetime.now() - self.stats['last_message_time']).total_seconds()
            if elapsed > 0:
                self.stats['messages_per_second'] = self.stats['total_messages'] / elapsed
        
        return self.stats.copy()


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du WebSocket Handler"""
    
    print("\n=== Test WebSocket Handler ===\n")
    
    # CrÃƒÂ©er le handler
    handler = WebSocketHandler()
    
    # DÃƒÂ©finir des callbacks de test
    def on_kline(data):
        print(f"Ã°Å¸â€œÅ  Kline: {data['symbol']} - Close: ${data['close']:,.2f}")
    
    def on_trade(data):
        print(f"Ã°Å¸â€™Â¹ Trade: {data['symbol']} - Price: ${data['price']:,.2f}")
    
    # Enregistrer les callbacks
    handler.register_callback('kline', on_kline)
    handler.register_callback('trade', on_trade)
    
    # S'abonner ÃƒÂ  des streams
    print("Ã°Å¸â€œÂ¡ Abonnement aux streams...")
    handler.subscribe_kline('BTCUSDT', '1m')
    handler.subscribe_trades('BTCUSDT')
    
    # DÃƒÂ©marrer
    print("Ã°Å¸Å¡â‚¬ DÃƒÂ©marrage du WebSocket...\n")
    handler.start()
    
    # Attendre quelques secondes pour recevoir des messages
    print("Ã¢ÂÂ³ Ãƒâ€°coute pendant 10 secondes...\n")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nÃ¢Å¡Â Ã¯Â¸Â Interruption manuelle")
    
    # ArrÃƒÂªter
    print("\nÃ°Å¸â€ºâ€˜ ArrÃƒÂªt du WebSocket...")
    handler.stop()
    
    # Afficher les stats
    print("\nÃ°Å¸â€œÅ  Statistiques finales:")
    stats = handler.get_stats()
    for key, value in stats.items():
        if key == 'last_message_time' and value:
            print(f"   {key}: {value.strftime('%H:%M:%S')}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Dernier message
    last_kline = handler.get_last_message('kline')
    if last_kline:
        print(f"\nÃ°Å¸â€œÅ  DerniÃƒÂ¨re kline:")
        print(f"   Symbol: {last_kline['symbol']}")
        print(f"   Close: ${last_kline['close']:,.2f}")
        print(f"   Volume: {last_kline['volume']:.2f}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")
