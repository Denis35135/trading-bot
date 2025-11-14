"""
Memory Manager pour The Bot
Gestion optimisÃƒÂ©e de la mÃƒÂ©moire pour ÃƒÂ©viter les memory leaks
"""

import gc
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Gestionnaire de mÃƒÂ©moire pour ÃƒÂ©viter les memory leaks
    
    ResponsabilitÃƒÂ©s:
    - Monitor l'utilisation mÃƒÂ©moire
    - DÃƒÂ©clencher le garbage collection
    - Nettoyer les anciens buffers
    - Alerter en cas de fuite mÃƒÂ©moire
    - Optimiser l'utilisation RAM
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le memory manager
        
        Args:
            config: Configuration
        """
        self.config = config
        self.max_memory_mb = getattr(config, 'MAX_MEMORY_MB', 2000)  # 2GB par dÃƒÂ©faut
        self.warning_threshold = getattr(config, 'WARNING_THRESHOLD', 0.8)  # 80%
        self.critical_threshold = getattr(config, 'CRITICAL_THRESHOLD', 0.95)  # 95%
        self.cleanup_interval = getattr(config, 'CLEANUP_INTERVAL', 300)  # 5 min
        
        # Ãƒâ€°tat
        self.is_running = False
        self.monitor_thread = None
        self.last_cleanup = time.time()
        self.cleanup_count = 0
        
        # Buffers gÃƒÂ©rÃƒÂ©s
        self.managed_buffers = {}
        self.buffer_limits = {}
        
        # Statistiques
        self.stats = {
            'peak_memory_mb': 0,
            'avg_memory_mb': 0,
            'cleanup_triggered': 0,
            'gc_collections': 0,
            'memory_warnings': 0,
            'memory_samples': []
        }
        
        # Callbacks
        self.on_warning_callbacks = []
        self.on_critical_callbacks = []
        self.on_cleanup_callbacks = []
        
        # Process actuel
        self.process = psutil.Process()
        
        logger.info(f"Memory Manager initialisÃƒÂ© (max: {self.max_memory_mb}MB)")
    
    def start(self):
        """DÃƒÂ©marre le monitoring mÃƒÂ©moire"""
        if self.is_running:
            logger.warning("Memory Manager dÃƒÂ©jÃƒÂ  en cours")
            return
        
        self.is_running = True
        
        # Thread de monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitor_thread.start()
        
        logger.info("Ã¢Å“â€¦ Memory Manager dÃƒÂ©marrÃƒÂ©")
    
    def stop(self):
        """ArrÃƒÂªte le monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Memory Manager arrÃƒÂªtÃƒÂ©")
    
    def _monitor_loop(self):
        """Boucle principale de monitoring"""
        logger.info("Thread Memory Monitor dÃƒÂ©marrÃƒÂ©")
        
        while self.is_running:
            try:
                # VÃƒÂ©rifier la mÃƒÂ©moire
                memory_info = self.get_memory_info()
                memory_mb = memory_info['rss_mb']
                usage_pct = memory_mb / self.max_memory_mb
                
                # Enregistrer l'ÃƒÂ©chantillon
                self.stats['memory_samples'].append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'usage_pct': usage_pct
                })
                
                # Garder max 1000 ÃƒÂ©chantillons
                if len(self.stats['memory_samples']) > 1000:
                    self.stats['memory_samples'] = self.stats['memory_samples'][-1000:]
                
                # Mettre ÃƒÂ  jour peak
                if memory_mb > self.stats['peak_memory_mb']:
                    self.stats['peak_memory_mb'] = memory_mb
                
                # VÃƒÂ©rifier les seuils
                if usage_pct >= self.critical_threshold:
                    logger.critical(
                        f"Ã°Å¸â€Â´ CRITIQUE: MÃƒÂ©moire ÃƒÂ  {usage_pct:.1%} ({memory_mb:.0f}MB/{self.max_memory_mb}MB)"
                    )
                    self.stats['memory_warnings'] += 1
                    self._trigger_critical_cleanup()
                    self._trigger_callbacks(self.on_critical_callbacks)
                    
                elif usage_pct >= self.warning_threshold:
                    logger.warning(
                        f"Ã¢Å¡Â Ã¯Â¸Â ATTENTION: MÃƒÂ©moire ÃƒÂ  {usage_pct:.1%} ({memory_mb:.0f}MB/{self.max_memory_mb}MB)"
                    )
                    self.stats['memory_warnings'] += 1
                    self._trigger_cleanup()
                    self._trigger_callbacks(self.on_warning_callbacks)
                
                # Cleanup pÃƒÂ©riodique
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    self._scheduled_cleanup()
                
                # Pause
                time.sleep(10)  # Check toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Erreur monitor loop: {e}")
                time.sleep(30)
        
        logger.info("Thread Memory Monitor arrÃƒÂªtÃƒÂ©")
    
    def _trigger_cleanup(self):
        """DÃƒÂ©clenche un nettoyage standard"""
        logger.info("Ã°Å¸Â§Â¹ Nettoyage mÃƒÂ©moire standard...")
        
        start_mem = self.get_memory_info()['rss_mb']
        
        # Nettoyer les buffers gÃƒÂ©rÃƒÂ©s
        self._cleanup_managed_buffers()
        
        # Garbage collection
        collected = gc.collect()
        self.stats['gc_collections'] += 1
        
        end_mem = self.get_memory_info()['rss_mb']
        freed_mb = start_mem - end_mem
        
        self.cleanup_count += 1
        self.last_cleanup = time.time()
        self.stats['cleanup_triggered'] += 1
        
        logger.info(
            f"Ã¢Å“â€¦ Nettoyage terminÃƒÂ©: {freed_mb:.1f}MB libÃƒÂ©rÃƒÂ©s, "
            f"{collected} objets collectÃƒÂ©s, mÃƒÂ©moire: {end_mem:.0f}MB"
        )
        
        # Callbacks
        self._trigger_callbacks(self.on_cleanup_callbacks)
    
    def _trigger_critical_cleanup(self):
        """DÃƒÂ©clenche un nettoyage agressif en cas de critique"""
        logger.critical("Ã°Å¸Å¡Â¨ Nettoyage CRITIQUE en cours...")
        
        start_mem = self.get_memory_info()['rss_mb']
        
        # Nettoyer agressivement les buffers
        for buffer_name in list(self.managed_buffers.keys()):
            buffer = self.managed_buffers[buffer_name]
            if isinstance(buffer, (list, deque)):
                # Garder seulement 10%
                keep = max(10, len(buffer) // 10)
                if isinstance(buffer, deque):
                    while len(buffer) > keep:
                        buffer.popleft()
                else:
                    buffer[:] = buffer[-keep:]
                logger.info(f"Buffer '{buffer_name}' rÃƒÂ©duit ÃƒÂ  {keep} items")
        
        # Plusieurs passes de GC
        for i in range(3):
            collected = gc.collect(generation=2)
            logger.info(f"GC pass {i+1}: {collected} objets collectÃƒÂ©s")
        
        end_mem = self.get_memory_info()['rss_mb']
        freed_mb = start_mem - end_mem
        
        logger.critical(f"Ã¢Å“â€¦ Nettoyage critique terminÃƒÂ©: {freed_mb:.1f}MB libÃƒÂ©rÃƒÂ©s")
        
        # Si toujours critique, log dÃƒÂ©taillÃƒÂ©
        if end_mem / self.max_memory_mb > self.critical_threshold:
            self._log_memory_details()
    
    def _scheduled_cleanup(self):
        """Nettoyage pÃƒÂ©riodique planifiÃƒÂ©"""
        logger.debug("Ã°Å¸Â§Â¹ Nettoyage pÃƒÂ©riodique...")
        
        # Nettoyer les buffers selon leurs limites
        self._cleanup_managed_buffers()
        
        # GC lÃƒÂ©ger
        gc.collect(generation=0)
        
        self.last_cleanup = time.time()
    
    def _cleanup_managed_buffers(self):
        """Nettoie les buffers selon leurs limites"""
        for buffer_name, buffer in self.managed_buffers.items():
            if buffer_name not in self.buffer_limits:
                continue
            
            limit = self.buffer_limits[buffer_name]
            
            if isinstance(buffer, list):
                if len(buffer) > limit:
                    buffer[:] = buffer[-limit:]
                    logger.debug(f"Buffer '{buffer_name}' nettoyÃƒÂ©: {len(buffer)}/{limit}")
                    
            elif isinstance(buffer, deque):
                while len(buffer) > limit:
                    buffer.popleft()
                logger.debug(f"Buffer '{buffer_name}' nettoyÃƒÂ©: {len(buffer)}/{limit}")
                
            elif isinstance(buffer, dict):
                if len(buffer) > limit:
                    # Garder les plus rÃƒÂ©cents
                    items = sorted(buffer.items(), key=lambda x: x[0], reverse=True)
                    buffer.clear()
                    buffer.update(dict(items[:limit]))
                    logger.debug(f"Dict '{buffer_name}' nettoyÃƒÂ©: {len(buffer)}/{limit}")
    
    def register_buffer(self, name: str, buffer: Any, limit: int):
        """
        Enregistre un buffer ÃƒÂ  gÃƒÂ©rer
        
        Args:
            name: Nom du buffer
            buffer: RÃƒÂ©fÃƒÂ©rence au buffer
            limit: Taille maximum
        """
        self.managed_buffers[name] = buffer
        self.buffer_limits[name] = limit
        logger.info(f"Buffer '{name}' enregistrÃƒÂ© (limit: {limit})")
    
    def unregister_buffer(self, name: str):
        """
        DÃƒÂ©senregistre un buffer
        
        Args:
            name: Nom du buffer
        """
        if name in self.managed_buffers:
            del self.managed_buffers[name]
            del self.buffer_limits[name]
            logger.info(f"Buffer '{name}' dÃƒÂ©senregistrÃƒÂ©")
    
    def register_callback(self, event: str, callback: Callable):
        """
        Enregistre un callback
        
        Args:
            event: 'warning', 'critical' ou 'cleanup'
            callback: Fonction ÃƒÂ  appeler
        """
        if event == 'warning':
            self.on_warning_callbacks.append(callback)
        elif event == 'critical':
            self.on_critical_callbacks.append(callback)
        elif event == 'cleanup':
            self.on_cleanup_callbacks.append(callback)
        else:
            logger.warning(f"Event inconnu: {event}")
    
    def _trigger_callbacks(self, callbacks: List[Callable]):
        """DÃƒÂ©clenche une liste de callbacks"""
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Erreur callback: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Retourne les infos mÃƒÂ©moire
        
        Returns:
            Dict avec infos mÃƒÂ©moire
        """
        try:
            mem_info = self.process.memory_info()
            vm = psutil.virtual_memory()
            
            return {
                'rss_mb': mem_info.rss / 1024 / 1024,
                'vms_mb': mem_info.vms / 1024 / 1024,
                'percent': self.process.memory_percent(),
                'available_mb': vm.available / 1024 / 1024,
                'system_total_mb': vm.total / 1024 / 1024,
                'system_used_mb': vm.used / 1024 / 1024,
                'system_percent': vm.percent
            }
        except Exception as e:
            logger.error(f"Erreur get_memory_info: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques
        
        Returns:
            Dict avec statistiques
        """
        current_mem = self.get_memory_info()['rss_mb']
        
        # Moyenne mÃƒÂ©moire sur derniÃƒÂ¨res 100 samples
        recent_samples = self.stats['memory_samples'][-100:]
        if recent_samples:
            avg_mem = sum(s['memory_mb'] for s in recent_samples) / len(recent_samples)
        else:
            avg_mem = current_mem
        
        return {
            'current_memory_mb': current_mem,
            'peak_memory_mb': self.stats['peak_memory_mb'],
            'avg_memory_mb': avg_mem,
            'max_memory_mb': self.max_memory_mb,
            'usage_percent': (current_mem / self.max_memory_mb) * 100,
            'cleanup_count': self.cleanup_count,
            'gc_collections': self.stats['gc_collections'],
            'memory_warnings': self.stats['memory_warnings'],
            'managed_buffers': len(self.managed_buffers),
            'uptime_seconds': time.time() - self.stats['memory_samples'][0]['timestamp'] if self.stats['memory_samples'] else 0
        }
    
    def force_cleanup(self):
        """Force un nettoyage immÃƒÂ©diat"""
        logger.info("Ã°Å¸Â§Â¹ Nettoyage forcÃƒÂ©...")
        self._trigger_cleanup()
    
    def force_gc(self):
        """Force un garbage collection complet"""
        logger.info("Ã°Å¸â€”â€˜Ã¯Â¸Â Garbage collection forcÃƒÂ©...")
        
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        self.stats['gc_collections'] += 1
        logger.info(f"Ã¢Å“â€¦ GC terminÃƒÂ©: {collected} objets collectÃƒÂ©s")
        
        return collected
    
    def _log_memory_details(self):
        """Log des dÃƒÂ©tails sur l'utilisation mÃƒÂ©moire"""
        import sys
        
        mem_info = self.get_memory_info()
        
        logger.critical("\n" + "=" * 60)
        logger.critical("DÃƒâ€°TAILS MÃƒâ€°MOIRE")
        logger.critical("=" * 60)
        logger.critical(f"RSS: {mem_info['rss_mb']:.1f}MB")
        logger.critical(f"VMS: {mem_info['vms_mb']:.1f}MB")
        logger.critical(f"Percent: {mem_info['percent']:.1f}%")
        logger.critical(f"System available: {mem_info['available_mb']:.1f}MB")
        logger.critical(f"System used: {mem_info['system_used_mb']:.1f}MB ({mem_info['system_percent']:.1f}%)")
        
        # Buffers gÃƒÂ©rÃƒÂ©s
        logger.critical("\nBuffers gÃƒÂ©rÃƒÂ©s:")
        for name, buffer in self.managed_buffers.items():
            if isinstance(buffer, (list, deque)):
                size = len(buffer)
                limit = self.buffer_limits.get(name, 'N/A')
                logger.critical(f"  {name}: {size}/{limit} items")
            elif isinstance(buffer, dict):
                size = len(buffer)
                limit = self.buffer_limits.get(name, 'N/A')
                logger.critical(f"  {name}: {size}/{limit} keys")
        
        # GC stats
        logger.critical("\nGarbage Collector:")
        for i, count in enumerate(gc.get_count()):
            logger.critical(f"  Generation {i}: {count} objects")
        
        logger.critical("=" * 60 + "\n")
    
    def get_buffer_info(self) -> Dict[str, Dict]:
        """
        Retourne les infos sur les buffers gÃƒÂ©rÃƒÂ©s
        
        Returns:
            Dict avec infos buffers
        """
        info = {}
        
        for name, buffer in self.managed_buffers.items():
            if isinstance(buffer, (list, deque)):
                size = len(buffer)
            elif isinstance(buffer, dict):
                size = len(buffer)
            else:
                size = 'N/A'
            
            limit = self.buffer_limits.get(name, 'N/A')
            usage_pct = (size / limit * 100) if isinstance(size, int) and isinstance(limit, int) else None
            
            info[name] = {
                'size': size,
                'limit': limit,
                'usage_percent': usage_pct,
                'type': type(buffer).__name__
            }
        
        return info
    
    def optimize_memory(self):
        """
        Optimise l'utilisation mÃƒÂ©moire
        
        Tips d'optimisation appliquÃƒÂ©s automatiquement
        """
        logger.info("Ã°Å¸â€Â§ Optimisation mÃƒÂ©moire...")
        
        # Activer le GC automatique
        gc.enable()
        
        # Ajuster les seuils du GC
        gc.set_threshold(700, 10, 10)
        
        # Force un nettoyage complet
        self.force_gc()
        
        logger.info("Ã¢Å“â€¦ Optimisation terminÃƒÂ©e")
