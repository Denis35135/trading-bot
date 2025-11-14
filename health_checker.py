"""
Health Checker
VÃƒÂ©rifie la santÃƒÂ© globale du systÃƒÂ¨me
"""

import psutil
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Statuts de santÃƒÂ©"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthChecker:
    """
    VÃƒÂ©rificateur de santÃƒÂ© du systÃƒÂ¨me
    
    VÃƒÂ©rifie:
    - Ressources systÃƒÂ¨me (CPU, RAM, Disk)
    - Connexion API
    - Ãƒâ€°tat des stratÃƒÂ©gies
    - Performance globale
    - Erreurs rÃƒÂ©centes
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le health checker
        
        Args:
            config: Configuration
        """
        default_config = {
            'cpu_warning_threshold': 80,  # %
            'cpu_critical_threshold': 95,
            'memory_warning_threshold': 80,
            'memory_critical_threshold': 95,
            'disk_warning_threshold': 85,
            'disk_critical_threshold': 95,
            'max_error_rate': 0.1,  # 10% d'erreurs max
            'check_interval': 60  # secondes
        }
        
        if config:
            default_config.update(config if isinstance(config, dict) else vars(config))
        
        self.config = default_config
        self.last_check = None
        self.health_history = []
        
        # Compteurs
        self.checks_count = 0
        self.issues_count = 0
        
        logger.info("Ã°Å¸ÂÂ¥ Health Checker initialisÃƒÂ©")
    
    def check_health(
        self,
        api_client=None,
        strategies: List = None,
        recent_errors: List = None
    ) -> Dict:
        """
        Effectue un check de santÃƒÂ© complet
        
        Args:
            api_client: Client API (optionnel)
            strategies: Liste des stratÃƒÂ©gies (optionnel)
            recent_errors: Erreurs rÃƒÂ©centes (optionnel)
            
        Returns:
            Dict avec rÃƒÂ©sultats
        """
        self.checks_count += 1
        
        results = {
            'timestamp': datetime.now(),
            'checks': {},
            'overall': HealthStatus.HEALTHY,
            'issues': []
        }
        
        # 1. Ressources systÃƒÂ¨me
        system_health = self._check_system_resources()
        results['checks']['system'] = system_health
        
        if system_health['status'] != HealthStatus.HEALTHY:
            results['overall'] = max(results['overall'], system_health['status'], key=lambda x: x.value)
            results['issues'].extend(system_health['issues'])
        
        # 2. API
        if api_client:
            api_health = self._check_api(api_client)
            results['checks']['api'] = api_health
            
            if api_health['status'] != HealthStatus.HEALTHY:
                results['overall'] = max(results['overall'], api_health['status'], key=lambda x: x.value)
                results['issues'].extend(api_health['issues'])
        
        # 3. StratÃƒÂ©gies
        if strategies:
            strategies_health = self._check_strategies(strategies)
            results['checks']['strategies'] = strategies_health
            
            if strategies_health['status'] != HealthStatus.HEALTHY:
                results['overall'] = max(results['overall'], strategies_health['status'], key=lambda x: x.value)
                results['issues'].extend(strategies_health['issues'])
        
        # 4. Erreurs rÃƒÂ©centes
        if recent_errors:
            errors_health = self._check_error_rate(recent_errors)
            results['checks']['errors'] = errors_health
            
            if errors_health['status'] != HealthStatus.HEALTHY:
                results['overall'] = max(results['overall'], errors_health['status'], key=lambda x: x.value)
                results['issues'].extend(errors_health['issues'])
        
        # Mettre ÃƒÂ  jour l'historique
        self.health_history.append(results)
        if len(self.health_history) > 100:
            self.health_history.pop(0)
        
        self.last_check = datetime.now()
        
        if results['issues']:
            self.issues_count += 1
        
        # Log
        if results['overall'] != HealthStatus.HEALTHY:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Health check: {results['overall'].value}")
            for issue in results['issues']:
                logger.warning(f"   - {issue}")
        else:
            logger.debug("Ã¢Å“â€¦ Health check: all systems healthy")
        
        return results
    
    def _check_system_resources(self) -> Dict:
        """VÃƒÂ©rifie les ressources systÃƒÂ¨me"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            issues = []
            
            # CPU
            if cpu_percent >= self.config['cpu_critical_threshold']:
                status = HealthStatus.UNHEALTHY
                issues.append(f"CPU critique: {cpu_percent:.1f}%")
            elif cpu_percent >= self.config['cpu_warning_threshold']:
                status = HealthStatus.DEGRADED
                issues.append(f"CPU ÃƒÂ©levÃƒÂ©: {cpu_percent:.1f}%")
            
            # Memory
            if memory.percent >= self.config['memory_critical_threshold']:
                status = max(status, HealthStatus.UNHEALTHY, key=lambda x: x.value)
                issues.append(f"MÃƒÂ©moire critique: {memory.percent:.1f}%")
            elif memory.percent >= self.config['memory_warning_threshold']:
                status = max(status, HealthStatus.DEGRADED, key=lambda x: x.value)
                issues.append(f"MÃƒÂ©moire ÃƒÂ©levÃƒÂ©e: {memory.percent:.1f}%")
            
            # Disk
            if disk.percent >= self.config['disk_critical_threshold']:
                status = max(status, HealthStatus.UNHEALTHY, key=lambda x: x.value)
                issues.append(f"Disque critique: {disk.percent:.1f}%")
            elif disk.percent >= self.config['disk_warning_threshold']:
                status = max(status, HealthStatus.DEGRADED, key=lambda x: x.value)
                issues.append(f"Disque ÃƒÂ©levÃƒÂ©: {disk.percent:.1f}%")
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Erreur check systÃƒÂ¨me: {e}")
            return {
                'status': HealthStatus.UNKNOWN,
                'issues': [f"Erreur check systÃƒÂ¨me: {e}"]
            }
    
    def _check_api(self, api_client) -> Dict:
        """VÃƒÂ©rifie la connexion API"""
        try:
            # Test de ping API
            start_time = datetime.now()
            
            # Essayer de rÃƒÂ©cupÃƒÂ©rer le server time (rapide)
            if hasattr(api_client, 'get_server_time'):
                api_client.get_server_time()
            elif hasattr(api_client, 'ping'):
                api_client.ping()
            else:
                # Pas de mÃƒÂ©thode de ping disponible
                return {
                    'status': HealthStatus.UNKNOWN,
                    'issues': ['Impossible de vÃƒÂ©rifier l\'API']
                }
            
            latency = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            status = HealthStatus.HEALTHY
            issues = []
            
            if latency > 1000:  # > 1 seconde
                status = HealthStatus.DEGRADED
                issues.append(f"Latence API ÃƒÂ©levÃƒÂ©e: {latency:.0f}ms")
            
            return {
                'status': status,
                'latency_ms': latency,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Erreur check API: {e}")
            return {
                'status': HealthStatus.UNHEALTHY,
                'issues': [f"API inaccessible: {e}"]
            }
    
    def _check_strategies(self, strategies: List) -> Dict:
        """VÃƒÂ©rifie l'ÃƒÂ©tat des stratÃƒÂ©gies"""
        try:
            total = len(strategies)
            active = sum(1 for s in strategies if hasattr(s, 'is_active') and s.is_active)
            
            status = HealthStatus.HEALTHY
            issues = []
            
            if active == 0:
                status = HealthStatus.UNHEALTHY
                issues.append("Aucune stratÃƒÂ©gie active")
            elif active < total * 0.5:
                status = HealthStatus.DEGRADED
                issues.append(f"Peu de stratÃƒÂ©gies actives: {active}/{total}")
            
            # VÃƒÂ©rifier les performances des stratÃƒÂ©gies
            low_performers = []
            for strategy in strategies:
                if hasattr(strategy, 'performance'):
                    perf = strategy.performance
                    win_rate = perf.get('win_rate', 0)
                    
                    if win_rate < 0.4 and perf.get('total_signals', 0) > 10:
                        low_performers.append(strategy.name)
            
            if low_performers:
                status = max(status, HealthStatus.DEGRADED, key=lambda x: x.value)
                issues.append(f"StratÃƒÂ©gies sous-performantes: {', '.join(low_performers)}")
            
            return {
                'status': status,
                'total_strategies': total,
                'active_strategies': active,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Erreur check stratÃƒÂ©gies: {e}")
            return {
                'status': HealthStatus.UNKNOWN,
                'issues': [f"Erreur check stratÃƒÂ©gies: {e}"]
            }
    
    def _check_error_rate(self, recent_errors: List) -> Dict:
        """VÃƒÂ©rifie le taux d'erreurs"""
        try:
            # Erreurs dans la derniÃƒÂ¨re heure
            cutoff = datetime.now() - timedelta(hours=1)
            recent = [e for e in recent_errors if e.get('timestamp', datetime.min) > cutoff]
            
            error_count = len(recent)
            
            status = HealthStatus.HEALTHY
            issues = []
            
            if error_count > 50:  # Plus de 50 erreurs/heure
                status = HealthStatus.UNHEALTHY
                issues.append(f"Taux d'erreurs critique: {error_count}/h")
            elif error_count > 20:
                status = HealthStatus.DEGRADED
                issues.append(f"Taux d'erreurs ÃƒÂ©levÃƒÂ©: {error_count}/h")
            
            return {
                'status': status,
                'error_count': error_count,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Erreur check erreurs: {e}")
            return {
                'status': HealthStatus.UNKNOWN,
                'issues': [f"Erreur check erreurs: {e}"]
            }
    
    def get_health_summary(self) -> Dict:
        """Retourne un rÃƒÂ©sumÃƒÂ© de santÃƒÂ©"""
        if not self.health_history:
            return {'status': 'no_data'}
        
        latest = self.health_history[-1]
        
        # Compter les checks par statut (derniÃƒÂ¨res 24h)
        cutoff = datetime.now() - timedelta(hours=24)
        recent_checks = [h for h in self.health_history if h['timestamp'] > cutoff]
        
        status_counts = {
            'healthy': sum(1 for h in recent_checks if h['overall'] == HealthStatus.HEALTHY),
            'degraded': sum(1 for h in recent_checks if h['overall'] == HealthStatus.DEGRADED),
            'unhealthy': sum(1 for h in recent_checks if h['overall'] == HealthStatus.UNHEALTHY)
        }
        
        return {
            'current_status': latest['overall'].value,
            'last_check': latest['timestamp'],
            'total_checks': self.checks_count,
            'issues_count': self.issues_count,
            'status_distribution_24h': status_counts,
            'current_issues': latest['issues']
        }
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return {
            'total_checks': self.checks_count,
            'issues_count': self.issues_count,
            'last_check': self.last_check,
            'history_size': len(self.health_history)
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Health Checker"""
    
    checker = HealthChecker()
    
    print("Test Health Checker")
    print("=" * 50)
    
    # Test 1: Check systÃƒÂ¨me de base
    print("\n1. Check santÃƒÂ© du systÃƒÂ¨me:")
    result = checker.check_health()
    
    print(f"   Status global: {result['overall'].value}")
    print(f"   Checks effectuÃƒÂ©s: {len(result['checks'])}")
    
    if result['issues']:
        print("   Issues:")
        for issue in result['issues']:
            print(f"     - {issue}")
    else:
        print("   Ã¢Å“â€¦ Aucun problÃƒÂ¨me dÃƒÂ©tectÃƒÂ©")
    
    # Test 2: DÃƒÂ©tails systÃƒÂ¨me
    if 'system' in result['checks']:
        system = result['checks']['system']
        print(f"\n2. Ressources systÃƒÂ¨me:")
        print(f"   CPU: {system.get('cpu_percent', 0):.1f}%")
        print(f"   Memory: {system.get('memory_percent', 0):.1f}%")
        print(f"   Disk: {system.get('disk_percent', 0):.1f}%")
    
    # Test 3: RÃƒÂ©sumÃƒÂ©
    print("\n3. RÃƒÂ©sumÃƒÂ© de santÃƒÂ©:")
    summary = checker.get_health_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")