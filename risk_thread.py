"""
Risk Thread pour The Bot
Thread de surveillance continue des risques et protection du capital
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class RiskThread:
    """
    Thread de surveillance des risques
    
    ResponsabilitÃƒÂ©s:
    - Monitor continu du drawdown
    - Surveillance de l'exposition
    - VÃƒÂ©rification des corrÃƒÂ©lations
    - DÃƒÂ©clenchement des circuit breakers
    - Alertes en temps rÃƒÂ©el
    - Actions correctives automatiques
    """
    
    def __init__(self, bot_instance, config: Dict):
        """
        Initialise le thread de risque
        
        Args:
            bot_instance: Instance du bot principal
            config: Configuration
        """
        self.bot = bot_instance
        self.config = config
        self.is_running = False
        self.thread = None
        
        # Configuration
        self.check_interval = getattr(config, 'CHECK_INTERVAL', 5)  # 5 secondes
        self.alert_cooldown = getattr(config, 'ALERT_COOLDOWN', 60)  # 60s entre alertes
        
        # Ãƒâ€°tat
        self.last_check = None
        self.last_alert = {}
        self.actions_taken = []
        
        # Statistiques
        self.stats = {
            'checks_performed': 0,
            'alerts_sent': 0,
            'circuit_breakers_triggered': 0,
            'positions_closed': 0,
            'last_risk_level': 'NORMAL',
            'emergency_stops': 0
        }
        
        logger.info("Risk Thread initialisÃƒÂ©")
    
    def start(self):
        """DÃƒÂ©marre le thread"""
        if self.is_running:
            logger.warning("Risk Thread dÃƒÂ©jÃƒÂ  en cours")
            return
        
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="RiskThread"
        )
        self.thread.start()
        
        logger.info("Ã¢Å“â€¦ Risk Thread dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte le thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Risk Thread arrÃƒÂªtÃƒÂ©")
    
    def _run(self):
        """Boucle principale du thread"""
        logger.info("Ã°Å¸â€â€ž Risk Thread running...")
        
        while self.is_running:
            try:
                # VÃƒÂ©rifier les risques
                self._perform_risk_check()
                
                # Pause entre checks
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans risk thread: {e}", exc_info=True)
                time.sleep(10)
        
        logger.info("Risk Thread terminÃƒÂ©")
    
    def _perform_risk_check(self):
        """Effectue une vÃƒÂ©rification complÃƒÂ¨te des risques"""
        try:
            self.stats['checks_performed'] += 1
            self.last_check = datetime.now()
            
            # VÃƒÂ©rifier que le risk monitor existe
            if not hasattr(self.bot, 'risk_monitor'):
                logger.warning("Risk monitor non disponible")
                return
            
            # RÃƒÂ©cupÃƒÂ©rer l'ÃƒÂ©tat actuel
            current_capital = self.bot.capital
            positions = self._get_current_positions()
            
            # Mise ÃƒÂ  jour du risk monitor
            risk_report = self.bot.risk_monitor.update(current_capital, positions)
            
            # Mettre ÃƒÂ  jour les stats
            risk_level = risk_report['risk_level']
            self.stats['last_risk_level'] = risk_level
            
            # Log pÃƒÂ©riodique (toutes les minutes)
            if self.stats['checks_performed'] % 12 == 0:  # 60s / 5s = 12
                self._log_risk_status(risk_report)
            
            # RÃƒÂ©agir selon le niveau de risque
            if risk_level == 'EMERGENCY':
                self._handle_emergency(risk_report)
            elif risk_level == 'CRITICAL':
                self._handle_critical(risk_report)
            elif risk_level == 'HIGH':
                self._handle_high_risk(risk_report)
            elif risk_level == 'WARNING':
                self._handle_warning(risk_report)
            
            # Traiter les actions recommandÃƒÂ©es
            if risk_report.get('actions'):
                self._process_actions(risk_report['actions'])
        
        except Exception as e:
            logger.error(f"Erreur check risque: {e}", exc_info=True)
    
    def _get_current_positions(self) -> Dict:
        """
        RÃƒÂ©cupÃƒÂ¨re les positions actuelles
        
        Returns:
            Dict des positions
        """
        try:
            if hasattr(self.bot, 'strategy_manager'):
                return self.bot.strategy_manager.positions
            return {}
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration positions: {e}")
            return {}
    
    def _handle_emergency(self, risk_report: Dict):
        """
        GÃƒÂ¨re un niveau d'urgence
        
        Args:
            risk_report: Rapport de risque
        """
        logger.critical("Ã°Å¸Å¡Â¨ NIVEAU D'URGENCE - Actions immÃƒÂ©diates!")
        
        self.stats['emergency_stops'] += 1
        
        # Fermer TOUTES les positions
        if self._should_send_alert('emergency'):
            logger.critical(
                f"Ã¢Å¡Â Ã¯Â¸Â ARRÃƒÅ T D'URGENCE Ã¢Å¡Â Ã¯Â¸Â\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}\n"
                f"Capital: ${risk_report['capital']:,.2f}\n"
                f"Fermeture de toutes les positions!"
            )
            
            # Fermer les positions
            self._close_all_positions('emergency')
            
            # DÃƒÂ©sactiver le trading
            if hasattr(self.bot, 'strategy_manager'):
                self.bot.strategy_manager.disable_trading()
            
            # Envoyer notification
            self._send_notification('EMERGENCY', risk_report)
            
            self.last_alert['emergency'] = time.time()
    
    def _handle_critical(self, risk_report: Dict):
        """
        GÃƒÂ¨re un niveau critique
        
        Args:
            risk_report: Rapport de risque
        """
        logger.error("Ã¢ÂÅ’ NIVEAU CRITIQUE - Actions correctives")
        
        self.stats['circuit_breakers_triggered'] += 1
        
        if self._should_send_alert('critical'):
            logger.error(
                f"Niveau critique atteint!\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}\n"
                f"Exposition: {risk_report['total_exposure_pct']:.1%}"
            )
            
            # Fermer les positions perdantes
            self._close_losing_positions()
            
            # RÃƒÂ©duire les autres positions
            self._reduce_all_positions(0.5)
            
            # Envoyer notification
            self._send_notification('CRITICAL', risk_report)
            
            self.last_alert['critical'] = time.time()
    
    def _handle_high_risk(self, risk_report: Dict):
        """
        GÃƒÂ¨re un niveau de risque ÃƒÂ©levÃƒÂ©
        
        Args:
            risk_report: Rapport de risque
        """
        logger.warning("Ã¢Å¡Â Ã¯Â¸Â RISQUE Ãƒâ€°LEVÃƒâ€° - RÃƒÂ©duction des positions")
        
        if self._should_send_alert('high'):
            logger.warning(
                f"Risque ÃƒÂ©levÃƒÂ© dÃƒÂ©tectÃƒÂ©\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}"
            )
            
            # Fermer la pire position
            self._close_worst_position()
            
            # RÃƒÂ©duire les nouvelles positions
            if hasattr(self.bot, 'position_sizer'):
                self.bot.position_sizer.apply_reduction_factor(0.7)
            
            self.last_alert['high'] = time.time()
    
    def _handle_warning(self, risk_report: Dict):
        """
        GÃƒÂ¨re un avertissement
        
        Args:
            risk_report: Rapport de risque
        """
        if self._should_send_alert('warning'):
            logger.warning(
                f"Ã¢Å¡Â Ã¯Â¸Â Avertissement risque\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}"
            )
            
            # Resserrer les stop loss
            self._tighten_stop_losses()
            
            self.last_alert['warning'] = time.time()
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """
        VÃƒÂ©rifie si une alerte doit ÃƒÂªtre envoyÃƒÂ©e (cooldown)
        
        Args:
            alert_type: Type d'alerte
            
        Returns:
            True si alerte autorisÃƒÂ©e
        """
        if alert_type not in self.last_alert:
            return True
        
        elapsed = time.time() - self.last_alert[alert_type]
        return elapsed > self.alert_cooldown
    
    def _close_all_positions(self, reason: str = 'risk'):
        """
        Ferme toutes les positions
        
        Args:
            reason: Raison de la fermeture
        """
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                return
            
            positions = self.bot.strategy_manager.positions.copy()
            
            logger.critical(f"Ã°Å¸Å¡Â¨ Fermeture de {len(positions)} positions ({reason})")
            
            for symbol in positions:
                try:
                    self.bot.strategy_manager.close_position(symbol, reason)
                    self.stats['positions_closed'] += 1
                except Exception as e:
                    logger.error(f"Erreur fermeture {symbol}: {e}")
            
            self.actions_taken.append({
                'action': 'CLOSE_ALL_POSITIONS',
                'reason': reason,
                'positions_count': len(positions),
                'timestamp': datetime.now()
            })
        
        except Exception as e:
            logger.error(f"Erreur close_all_positions: {e}")
    
    def _close_losing_positions(self):
        """Ferme les positions perdantes"""
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                return
            
            positions = self.bot.strategy_manager.positions.copy()
            
            for symbol, position in positions.items():
                # Calculer P&L
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                entry_price = position.get('entry_price')
                side = position.get('side')
                
                if side == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Fermer si nÃƒÂ©gatif
                if pnl_pct < 0:
                    logger.info(f"Fermeture position perdante: {symbol} ({pnl_pct:.2%})")
                    self.bot.strategy_manager.close_position(symbol, 'losing_position')
                    self.stats['positions_closed'] += 1
        
        except Exception as e:
            logger.error(f"Erreur close_losing_positions: {e}")
    
    def _reduce_all_positions(self, factor: float):
        """
        RÃƒÂ©duit toutes les positions d'un facteur
        
        Args:
            factor: Facteur de rÃƒÂ©duction (0.5 = rÃƒÂ©duire de 50%)
        """
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                return
            
            positions = self.bot.strategy_manager.positions.copy()
            
            logger.warning(f"RÃƒÂ©duction de {len(positions)} positions ÃƒÂ  {factor:.0%}")
            
            for symbol, position in positions.items():
                # Calculer nouvelle quantitÃƒÂ©
                current_qty = position.get('quantity', 0)
                new_qty = current_qty * factor
                
                # Fermer la diffÃƒÂ©rence
                qty_to_close = current_qty - new_qty
                
                if qty_to_close > 0:
                    # TODO: ImplÃƒÂ©menter rÃƒÂ©duction partielle
                    logger.debug(f"RÃƒÂ©duction {symbol}: {qty_to_close:.6f}")
        
        except Exception as e:
            logger.error(f"Erreur reduce_all_positions: {e}")
    
    def _close_worst_position(self):
        """Ferme la position avec la pire performance"""
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                return
            
            positions = self.bot.strategy_manager.positions.copy()
            if not positions:
                return
            
            worst_symbol = None
            worst_pnl = float('inf')
            
            for symbol, position in positions.items():
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                entry_price = position.get('entry_price')
                side = position.get('side')
                
                if side == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                if pnl_pct < worst_pnl:
                    worst_pnl = pnl_pct
                    worst_symbol = symbol
            
            if worst_symbol:
                logger.info(f"Fermeture pire position: {worst_symbol} ({worst_pnl:.2%})")
                self.bot.strategy_manager.close_position(worst_symbol, 'worst_performer')
                self.stats['positions_closed'] += 1
        
        except Exception as e:
            logger.error(f"Erreur close_worst_position: {e}")
    
    def _tighten_stop_losses(self):
        """Resserre les stop loss de toutes les positions"""
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                return
            
            positions = self.bot.strategy_manager.positions.copy()
            
            for symbol, position in positions.items():
                # Resserrer le SL de 20%
                current_sl = position.get('stop_loss')
                if current_sl:
                    entry_price = position.get('entry_price')
                    side = position.get('side')
                    
                    if side == 'BUY':
                        new_sl = entry_price - (entry_price - current_sl) * 0.8
                    else:
                        new_sl = entry_price + (current_sl - entry_price) * 0.8
                    
                    # TODO: Mettre ÃƒÂ  jour le stop loss
                    logger.debug(f"SL resserrÃƒÂ© {symbol}: {current_sl:.2f} Ã¢â€ â€™ {new_sl:.2f}")
        
        except Exception as e:
            logger.error(f"Erreur tighten_stop_losses: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """RÃƒÂ©cupÃƒÂ¨re le prix actuel d'un symbole"""
        try:
            if hasattr(self.bot, 'exchange'):
                ticker = self.bot.exchange.get_symbol_ticker(symbol)
                if ticker:
                    return ticker.get('price')
            return None
        except:
            return None
    
    def _process_actions(self, actions: List[str]):
        """
        Traite une liste d'actions recommandÃƒÂ©es
        
        Args:
            actions: Liste des actions
        """
        for action in actions:
            logger.info(f"Action recommandÃƒÂ©e: {action}")
            # Les actions sont dÃƒÂ©jÃƒÂ  traitÃƒÂ©es dans les handlers
    
    def _send_notification(self, level: str, risk_report: Dict):
        """
        Envoie une notification
        
        Args:
            level: Niveau de l'alerte
            risk_report: Rapport de risque
        """
        try:
            if hasattr(self.bot, 'notification_manager'):
                self.bot.notification_manager.notify_critical(
                    message=f"Alerte risque {level}",
                    data=risk_report
                )
                self.stats['alerts_sent'] += 1
        except Exception as e:
            logger.error(f"Erreur notification: {e}")
    
    def _log_risk_status(self, risk_report: Dict):
        """
        Log le statut du risque
        
        Args:
            risk_report: Rapport de risque
        """
        logger.info(
            f"Ã°Å¸â€œÅ  Risk Status: {risk_report['risk_level']} | "
            f"DD: {risk_report['current_drawdown']:.2%} | "
            f"Expo: {risk_report['total_exposure_pct']:.1%} | "
            f"Pos: {len(risk_report.get('positions', {}))}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques
        
        Returns:
            Dict avec stats
        """
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['last_check'] = self.last_check
        stats['actions_count'] = len(self.actions_taken)
        
        return stats
    
    def get_recent_actions(self, limit: int = 10) -> List[Dict]:
        """
        Retourne les actions rÃƒÂ©centes
        
        Args:
            limit: Nombre max d'actions
            
        Returns:
            Liste des actions
        """
        return self.actions_taken[-limit:]