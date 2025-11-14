#!/usr/bin/env python3
"""
🔍 System Monitor & Health Checker
Surveillance système et vérifications de santé
"""

import psutil
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor système pour éviter surcharge ressources"""
    
    def __init__(self, max_memory_mb: int = 2000, max_cpu_percent: float = 80.0):
        """
        Args:
            max_memory_mb: Mémoire max en MB (défaut 2GB)
            max_cpu_percent: CPU max en % (défaut 80%)
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.process = psutil.Process()
        
        # Métriques
        self.metrics_history = []
        self.alerts = []
        
    def get_system_metrics(self) -> Dict:
        """Récupère les métriques système actuelles"""
        try:
            # Mémoire du process
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU du process
            cpu_percent = self.process.cpu_percent(interval=1)
            
            # Mémoire système globale
            system_memory = psutil.virtual_memory()
            
            # CPU système global
            system_cpu = psutil.cpu_percent(interval=1)
            
            # Disque
            disk = psutil.disk_usage('/')
            
            # Threads
            num_threads = self.process.num_threads()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'process': {
                    'memory_mb': round(memory_mb, 2),
                    'memory_percent': round(memory_info.rss / system_memory.total * 100, 2),
                    'cpu_percent': round(cpu_percent, 2),
                    'num_threads': num_threads
                },
                'system': {
                    'memory_total_gb': round(system_memory.total / 1024 / 1024 / 1024, 2),
                    'memory_available_gb': round(system_memory.available / 1024 / 1024 / 1024, 2),
                    'memory_percent': system_memory.percent,
                    'cpu_percent': system_cpu,
                    'cpu_count': psutil.cpu_count(),
                    'disk_percent': disk.percent
                },
                'health': 'healthy'
            }
            
            # Vérifie les seuils
            if memory_mb > self.max_memory_mb:
                metrics['health'] = 'warning'
                self.alerts.append(f"Mémoire élevée: {memory_mb:.0f}MB > {self.max_memory_mb}MB")
                
            if cpu_percent > self.max_cpu_percent:
                metrics['health'] = 'warning'
                self.alerts.append(f"CPU élevé: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                
            # Conserve historique (dernières 60 mesures = 1h à 1min d'intervalle)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 60:
                self.metrics_history.pop(0)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur récupération métriques: {e}")
            return {'health': 'error', 'error': str(e)}
            
    def check_health(self) -> Tuple[bool, List[str]]:
        """
        Vérifie la santé du système
        
        Returns:
            (is_healthy, list_of_issues)
        """
        issues = []
        
        try:
            metrics = self.get_system_metrics()
            
            # Check mémoire process
            if metrics['process']['memory_mb'] > self.max_memory_mb:
                issues.append(f"⚠️  Mémoire process élevée: {metrics['process']['memory_mb']:.0f}MB")
                
            # Check CPU process
            if metrics['process']['cpu_percent'] > self.max_cpu_percent:
                issues.append(f"⚠️  CPU process élevé: {metrics['process']['cpu_percent']:.1f}%")
                
            # Check mémoire système
            if metrics['system']['memory_percent'] > 90:
                issues.append(f"⚠️  Mémoire système critique: {metrics['system']['memory_percent']:.1f}%")
                
            # Check CPU système
            if metrics['system']['cpu_percent'] > 95:
                issues.append(f"⚠️  CPU système critique: {metrics['system']['cpu_percent']:.1f}%")
                
            # Check disque
            if metrics['system']['disk_percent'] > 90:
                issues.append(f"⚠️  Disque presque plein: {metrics['system']['disk_percent']:.1f}%")
                
            # Check threads (pas plus de 50)
            if metrics['process']['num_threads'] > 50:
                issues.append(f"⚠️  Trop de threads: {metrics['process']['num_threads']}")
                
            is_healthy = len(issues) == 0
            
            return is_healthy, issues
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return False, [f"❌ Erreur health check: {e}"]
            
    def cleanup_if_needed(self) -> bool:
        """
        Nettoie la mémoire si nécessaire
        
        Returns:
            True si nettoyage effectué
        """
        try:
            metrics = self.get_system_metrics()
            
            if metrics['process']['memory_mb'] > self.max_memory_mb * 0.8:  # 80% du max
                logger.warning(f"Nettoyage mémoire nécessaire: {metrics['process']['memory_mb']:.0f}MB")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                time.sleep(0.5)
                
                # Re-check
                new_metrics = self.get_system_metrics()
                memory_freed = metrics['process']['memory_mb'] - new_metrics['process']['memory_mb']
                
                if memory_freed > 0:
                    logger.info(f"✅ Mémoire libérée: {memory_freed:.0f}MB")
                    return True
                else:
                    logger.warning("⚠️  Nettoyage mémoire inefficace")
                    return False
                    
            return False
            
        except Exception as e:
            logger.error(f"Erreur cleanup: {e}")
            return False
            
    def print_metrics(self):
        """Affiche les métriques système"""
        try:
            metrics = self.get_system_metrics()
            
            print(f"""
╔══════════════════════════════════════╗
║       SYSTEM METRICS                 ║
╠══════════════════════════════════════╣
║ Process:                             ║
║  • Memory: {metrics['process']['memory_mb']:.0f} MB ({metrics['process']['memory_percent']:.1f}%)     ║
║  • CPU: {metrics['process']['cpu_percent']:.1f}%                       ║
║  • Threads: {metrics['process']['num_threads']}                        ║
╠══════════════════════════════════════╣
║ System:                              ║
║  • Memory: {metrics['system']['memory_percent']:.1f}% used              ║
║  • CPU: {metrics['system']['cpu_percent']:.1f}%                       ║
║  • Disk: {metrics['system']['disk_percent']:.1f}%                      ║
╠══════════════════════════════════════╣
║ Health: {metrics['health'].upper():28s}    ║
╚══════════════════════════════════════╝
""")
            
        except Exception as e:
            logger.error(f"Erreur affichage métriques: {e}")
            
    def get_alerts(self) -> List[str]:
        """Retourne et vide la liste des alertes"""
        alerts = self.alerts.copy()
        self.alerts.clear()
        return alerts
        
    def get_average_metrics(self, minutes: int = 5) -> Dict:
        """
        Calcule la moyenne des métriques sur X minutes
        
        Args:
            minutes: Nombre de minutes (1-60)
            
        Returns:
            Dict avec moyennes ou None
        """
        if not self.metrics_history:
            return None
            
        # Prend les N dernières mesures
        recent = self.metrics_history[-minutes:]
        
        if not recent:
            return None
            
        # Calcule moyennes
        avg = {
            'memory_mb': sum(m['process']['memory_mb'] for m in recent) / len(recent),
            'cpu_percent': sum(m['process']['cpu_percent'] for m in recent) / len(recent),
            'num_threads': sum(m['process']['num_threads'] for m in recent) / len(recent)
        }
        
        return avg


class HealthChecker:
    """Vérifications de santé du bot"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.last_check = None
        self.health_history = []
        
    def check_all(self, bot_state: Dict = None) -> Tuple[bool, List[str]]:
        """
        Vérifie tous les aspects de santé du bot
        
        Args:
            bot_state: État actuel du bot (optionnel)
            
        Returns:
            (is_healthy, list_of_issues)
        """
        issues = []
        
        # 1. Check système
        system_healthy, system_issues = self.system_monitor.check_health()
        issues.extend(system_issues)
        
        # 2. Check bot state si fourni
        if bot_state:
            # Check capital
            if bot_state.get('capital', 0) <= 0:
                issues.append("❌ Capital épuisé")
                
            # Check drawdown
            drawdown = bot_state.get('current_drawdown', 0)
            if drawdown > 0.08:  # 8% max
                issues.append(f"❌ Drawdown critique: {drawdown:.1%}")
                
            # Check positions
            num_positions = len(bot_state.get('positions', []))
            if num_positions > 20:
                issues.append(f"⚠️  Trop de positions: {num_positions}")
                
        is_healthy = len(issues) == 0
        
        self.last_check = datetime.now()
        self.health_history.append({
            'timestamp': self.last_check.isoformat(),
            'healthy': is_healthy,
            'issues': issues
        })
        
        # Garde seulement les 100 derniers checks
        if len(self.health_history) > 100:
            self.health_history.pop(0)
            
        return is_healthy, issues
        
    def print_health_report(self, bot_state: Dict = None):
        """Affiche un rapport de santé complet"""
        is_healthy, issues = self.check_all(bot_state)
        
        print("\n" + "="*50)
        print("🏥 HEALTH CHECK REPORT")
        print("="*50)
        
        if is_healthy:
            print("✅ Tout va bien!")
        else:
            print(f"⚠️  {len(issues)} problème(s) détecté(s):")
            for issue in issues:
                print(f"  {issue}")
                
        # Métriques système
        print("\n📊 Métriques système:")
        self.system_monitor.print_metrics()
        
        print("="*50 + "\n")


# Test du module
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Test du System Monitor...\n")
    
    monitor = SystemMonitor()
    
    # Test métriques
    metrics = monitor.get_system_metrics()
    print(f"Mémoire: {metrics['process']['memory_mb']:.0f} MB")
    print(f"CPU: {metrics['process']['cpu_percent']:.1f}%")
    
    # Test health check
    healthy, issues = monitor.check_health()
    print(f"\nSanté système: {'✅' if healthy else '⚠️'}")
    if issues:
        for issue in issues:
            print(f"  {issue}")
            
    # Test affichage
    monitor.print_metrics()
    
    # Test HealthChecker
    print("\n🏥 Test du Health Checker...\n")
    checker = HealthChecker()
    checker.print_health_report()
