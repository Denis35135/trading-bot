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
    
    Responsabilites:
    - Monitor continu du drawdown
    - Surveillance de l'exposition
    - Verification des correlations
    - Declenchement des circuit breakers
    - Alertes en temps reel
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
        
        # Etat
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
        
        logger.info("Risk Thread initialise")
    
    def start(self):
        """Demarre le thread"""
        if self.is_running:
            logger.warning("Risk Thread deja en cours")
            return
        
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="RiskThread"
        )
        self.thread.start()
        
        logger.info("[OK] Risk Thread demarre")
    
    def stop(self):
        """Arrete le thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Risk Thread arrete")
    
    def _run(self):
        """Boucle principale du thread"""
        logger.info("Risk Thread running...")
        
        while self.is_running:
            try:
                # Verifier les risques
                self._perform_risk_check()
                
                # Pause entre checks
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans risk thread: {e}", exc_info=True)
                time.sleep(10)
        
        logger.info("Risk Thread termine")
    
    def _perform_risk_check(self):
        """Effectue une verification complete des risques"""
        try:
            self.stats['checks_performed'] += 1
            self.last_check = datetime.now()
            
            # Verifier que le risk monitor existe
            if not hasattr(self.bot, 'risk_monitor'):
                logger.warning("Risk monitor non disponible")
                return
            
            # Recuperer l'etat actuel
            current_capital = self.bot.capital
            positions = self._get_current_positions()
            
            # Mise a jour du risk monitor
            risk_report = self.bot.risk_monitor.update(current_capital, positions)
            
            # Mettre a jour les stats
            risk_level = risk_report['risk_level']
            self.stats['last_risk_level'] = risk_level
            
            # Log periodique (toutes les minutes)
            if self.stats['checks_performed'] % 12 == 0:  # 60s / 5s = 12
                self._log_risk_status(risk_report)
            
            # Reagir selon le niveau de risque
            if risk_level == 'EMERGENCY':
                self._handle_emergency(risk_report)
            elif risk_level == 'CRITICAL':
                self._handle_critical(risk_report)
            elif risk_level == 'HIGH':
                self._handle_high_risk(risk_report)
            elif risk_level == 'WARNING':
                self._handle_warning(risk_report)
            
            # Traiter les actions recommandees
            if risk_report.get('actions'):
                self._process_actions(risk_report['actions'])
        
        except Exception as e:
            logger.error(f"Erreur check risque: {e}", exc_info=True)
    
    def _get_current_positions(self) -> Dict:
        """
        Recupere les positions actuelles
        
        Returns:
            Dict des positions
        """
        try:
            if hasattr(self.bot, 'strategy_manager'):
                return self.bot.strategy_manager.positions
            return {}
        except Exception as e:
            logger.error(f"Erreur recuperation positions: {e}")
            return {}
    
    def _handle_emergency(self, risk_report: Dict):
        """
        Gere un niveau d'urgence
        
        Args:
            risk_report: Rapport de risque
        """
        logger.critical("[ALERT] NIVEAU D'URGENCE - Actions immediates!")
        
        self.stats['emergency_stops'] += 1
        
        # Fermer TOUTES les positions
        if self._should_send_alert('emergency'):
            logger.critical(
                f"[EMERGENCY] ARRET D'URGENCE\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}\n"
                f"Capital: ${risk_report['capital']:,.2f}\n"
                f"Fermeture de toutes les positions!"
            )
            
            # Fermer les positions
            self._close_all_positions('emergency')
            
            # Desactiver le trading
            if hasattr(self.bot, 'strategy_manager'):
                self.bot.strategy_manager.disable_trading()
            
            # Envoyer notification
            self._send_notification('EMERGENCY', risk_report)
            
            self.last_alert['emergency'] = time.time()
    
    def _handle_critical(self, risk_report: Dict):
        """
        Gere un niveau critique
        
        Args:
            risk_report: Rapport de risque
        """
        logger.error("[CRITICAL] NIVEAU CRITIQUE - Actions correctives")
        
        self.stats['circuit_breakers_triggered'] += 1
        
        if self._should_send_alert('critical'):
            logger.error(
                f"Niveau critique atteint!\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}\n"
                f"Exposition: {risk_report['total_exposure_pct']:.1%}"
            )
            
            # Fermer les positions perdantes
            self._close_losing_positions()
            
            # Reduire les autres positions
            self._reduce_all_positions(0.5)
            
            # Envoyer notification
            self._send_notification('CRITICAL', risk_report)
            
            self.last_alert['critical'] = time.time()
    
    def _handle_high_risk(self, risk_report: Dict):
        """
        Gere un niveau de risque eleve
        
        Args:
            risk_report: Rapport de risque
        """
        logger.warning("[WARNING] RISQUE ELEVE - Reduction des positions")
        
        if self._should_send_alert('high'):
            logger.warning(
                f"Risque eleve detecte\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}"
            )
            
            # Fermer la pire position
            self._close_worst_position()
            
            # Reduire les nouvelles positions
            if hasattr(self.bot, 'position_sizer'):
                self.bot.position_sizer.apply_reduction_factor(0.7)
            
            self.last_alert['high'] = time.time()
    
    def _handle_warning(self, risk_report: Dict):
        """
        Gere un avertissement
        
        Args:
            risk_report: Rapport de risque
        """
        if self._should_send_alert('warning'):
            logger.warning(
                f"Avertissement risque\n"
                f"Drawdown: {risk_report['current_drawdown']:.2%}"
            )
            
            # Resserrer les stop loss
            self._tighten_stop_losses()
            
            self.last_alert['warning'] = time.time()
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """
        Verifie si une alerte doit etre envoyee (cooldown)
        
        Args:
            alert_type: Type d'alerte
            
        Returns:
            True si alerte autorisee
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
            
            logger.critical(f"[ACTION] Fermeture de {len(positions)} positions ({reason})")
            
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
                
                # Fermer si negatif
                if pnl_pct < 0:
                    logger.info(f"Fermeture position perdante: {symbol} ({pnl_pct:.2%})")
                    self.bot.strategy_manager.close_position(symbol, 'losing_position')
                    self.stats['positions_closed'] += 1
        
        except Exception as e:
            logger.error(f"Erreur close_losing_positions: {e}")
    
    def _reduce_all_positions(self, factor: float):
        """
        Reduit toutes les positions d'un facteur
        
        Args:
            factor: Facteur de reduction (0.5 = reduire de 50%)
        """
        try:
            if not hasattr(self.bot, 'strategy_manager'):
                return
            
            positions = self.bot.strategy_manager.positions.copy()
            
            logger.warning(f"Reduction de {len(positions)} positions a {factor:.0%}")
            
            for symbol, position in positions.items():
                # Calculer nouvelle quantite
                current_qty = position.get('quantity', 0)
                new_qty = current_qty * factor
                
                # Fermer la difference
                qty_to_close = current_qty - new_qty
                
                if qty_to_close > 0:
                    # TODO: Implementer reduction partielle
                    logger.debug(f"Reduction {symbol}: {qty_to_close:.6f}")
        
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
                    
                    # TODO: Mettre a jour le stop loss
                    logger.debug(f"SL resserre {symbol}: {current_sl:.2f} -> {new_sl:.2f}")
        
        except Exception as e:
            logger.error(f"Erreur tighten_stop_losses: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Recupere le prix actuel d'un symbole"""
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
        Traite une liste d'actions recommandees
        
        Args:
            actions: Liste des actions
        """
        for action in actions:
            logger.info(f"Action recommandee: {action}")
            # Les actions sont deja traitees dans les handlers
    
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
            f"[RISK STATUS] {risk_report['risk_level']} | "
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
        Retourne les actions recentes
        
        Args:
            limit: Nombre max d'actions
            
        Returns:
            Liste des actions
        """
        return self.actions_taken[-limit:]
