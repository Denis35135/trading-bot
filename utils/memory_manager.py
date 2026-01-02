"""
Memory Manager pour The Bot
Gestion optimisee de la memoire pour eviter les memory leaks
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
    Gestionnaire de memoire pour eviter les memory leaks
    
    Responsabilites:
    - Monitor l'utilisation memoire
    - Declencher le garbage collection
    - Nettoyer les anciens buffers
    - Alerter en cas de fuite memoire
    - Optimiser l'utilisation RAM
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le memory manager
        
        Args:
            config: Configuration
        """
        self.config = config
        self.max_memory_mb = getattr(config, 'MAX_MEMORY_MB', 2000)  # 2GB par defaut
        self.warning_threshold = getattr(config, 'WARNING_THRESHOLD', 0.8)  # 80%
        self.critical_threshold = getattr(config, 'CRITICAL_THRESHOLD', 0.95)  # 95%
        self.cleanup_interval = getattr(config, 'CLEANUP_INTERVAL', 300)  # 5 min
        
        # "tat
        self.is_running = False
        self.monitor_thread = None
        self.last_cleanup = time.time()
        self.cleanup_count = 0
        
        # Buffers geres
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
        
        logger.info(f"Memory Manager initialise (max: {self.max_memory_mb}MB)")
    
    def start(self):
        """Demarre le monitoring memoire"""
        if self.is_running:
            logger.warning("Memory Manager dej en cours")
            return
        
        self.is_running = True
        
        # Thread de monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitor_thread.start()
        
        logger.info("""| Memory Manager demarre")
    
    def stop(self):
        """Arrete le monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Memory Manager arrete")
    
    def _monitor_loop(self):
        """Boucle principale de monitoring"""
        logger.info("Thread Memory Monitor demarre")
        
        while self.is_running:
            try:
                # Verifier la memoire
                memory_info = self.get_memory_info()
                memory_mb = memory_info['rss_mb']
                usage_pct = memory_mb / self.max_memory_mb
                
                # Enregistrer l'echantillon
                self.stats['memory_samples'].append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'usage_pct': usage_pct
                })
                
                # Garder max 1000 echantillons
                if len(self.stats['memory_samples']) > 1000:
                    self.stats['memory_samples'] = self.stats['memory_samples'][-1000:]
                
                # Mettre  jour peak
                if memory_mb > self.stats['peak_memory_mb']:
                    self.stats['peak_memory_mb'] = memory_mb
                
                # Verifier les seuils
                if usage_pct >= self.critical_threshold:
                    logger.critical(
                        f"" CRITIQUE: Memoire  {usage_pct:.1%} ({memory_mb:.0f}MB/{self.max_memory_mb}MB)"
                    )
                    self.stats['memory_warnings'] += 1
                    self._trigger_critical_cleanup()
                    self._trigger_callbacks(self.on_critical_callbacks)
                    
                elif usage_pct >= self.warning_threshold:
                    logger.warning(
                        f" ATTENTION: Memoire  {usage_pct:.1%} ({memory_mb:.0f}MB/{self.max_memory_mb}MB)"
                    )
                    self.stats['memory_warnings'] += 1
                    self._trigger_cleanup()
                    self._trigger_callbacks(self.on_warning_callbacks)
                
                # Cleanup periodique
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    self._scheduled_cleanup()
                
                # Pause
                time.sleep(10)  # Check toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Erreur monitor loop: {e}")
                time.sleep(30)
        
        logger.info("Thread Memory Monitor arrete")
    
    def _trigger_cleanup(self):
        """Declenche un nettoyage standard"""
        logger.info(" Nettoyage memoire standard...")
        
        start_mem = self.get_memory_info()['rss_mb']
        
        # Nettoyer les buffers geres
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
            f"""| Nettoyage termine: {freed_mb:.1f}MB liberes, "
            f"{collected} objets collectes, memoire: {end_mem:.0f}MB"
        )
        
        # Callbacks
        self._trigger_callbacks(self.on_cleanup_callbacks)
    
    def _trigger_critical_cleanup(self):
        """Declenche un nettoyage agressif en cas de critique"""
        logger.critical(" Nettoyage CRITIQUE en cours...")
        
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
                logger.info(f"Buffer '{buffer_name}' reduit  {keep} items")
        
        # Plusieurs passes de GC
        for i in range(3):
            collected = gc.collect(generation=2)
            logger.info(f"GC pass {i+1}: {collected} objets collectes")
        
        end_mem = self.get_memory_info()['rss_mb']
        freed_mb = start_mem - end_mem
        
        logger.critical(f"""| Nettoyage critique termine: {freed_mb:.1f}MB liberes")
        
        # Si toujours critique, log detaille
        if end_mem / self.max_memory_mb > self.critical_threshold:
            self._log_memory_details()
    
    def _scheduled_cleanup(self):
        """Nettoyage periodique planifie"""
        logger.debug(" Nettoyage periodique...")
        
        # Nettoyer les buffers selon leurs limites
        self._cleanup_managed_buffers()
        
        # GC leger
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
                    logger.debug(f"Buffer '{buffer_name}' nettoye: {len(buffer)}/{limit}")
                    
            elif isinstance(buffer, deque):
                while len(buffer) > limit:
                    buffer.popleft()
                logger.debug(f"Buffer '{buffer_name}' nettoye: {len(buffer)}/{limit}")
                
            elif isinstance(buffer, dict):
                if len(buffer) > limit:
                    # Garder les plus recents
                    items = sorted(buffer.items(), key=lambda x: x[0], reverse=True)
                    buffer.clear()
                    buffer.update(dict(items[:limit]))
                    logger.debug(f"Dict '{buffer_name}' nettoye: {len(buffer)}/{limit}")
    
    def register_buffer(self, name: str, buffer: Any, limit: int):
        """
        Enregistre un buffer  gerer
        
        Args:
            name: Nom du buffer
            buffer: Reference au buffer
            limit: Taille maximum
        """
        self.managed_buffers[name] = buffer
        self.buffer_limits[name] = limit
        logger.info(f"Buffer '{name}' enregistre (limit: {limit})")
    
    def unregister_buffer(self, name: str):
        """
        Desenregistre un buffer
        
        Args:
            name: Nom du buffer
        """
        if name in self.managed_buffers:
            del self.managed_buffers[name]
            del self.buffer_limits[name]
            logger.info(f"Buffer '{name}' desenregistre")
    
    def register_callback(self, event: str, callback: Callable):
        """
        Enregistre un callback
        
        Args:
            event: 'warning', 'critical' ou 'cleanup'
            callback: Fonction  appeler
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
        """Declenche une liste de callbacks"""
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Erreur callback: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Retourne les infos memoire
        
        Returns:
            Dict avec infos memoire
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
        
        # Moyenne memoire sur dernieres 100 samples
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
        """Force un nettoyage immediat"""
        logger.info(" Nettoyage force...")
        self._trigger_cleanup()
    
    def force_gc(self):
        """Force un garbage collection complet"""
        logger.info("""" Garbage collection force...")
        
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        self.stats['gc_collections'] += 1
        logger.info(f"""| GC termine: {collected} objets collectes")
        
        return collected
    
    def _log_memory_details(self):
        """Log des details sur l'utilisation memoire"""
        import sys
        
        mem_info = self.get_memory_info()
        
        logger.critical("\n" + "=" * 60)
        logger.critical("D"TAILS M"MOIRE")
        logger.critical("=" * 60)
        logger.critical(f"RSS: {mem_info['rss_mb']:.1f}MB")
        logger.critical(f"VMS: {mem_info['vms_mb']:.1f}MB")
        logger.critical(f"Percent: {mem_info['percent']:.1f}%")
        logger.critical(f"System available: {mem_info['available_mb']:.1f}MB")
        logger.critical(f"System used: {mem_info['system_used_mb']:.1f}MB ({mem_info['system_percent']:.1f}%)")
        
        # Buffers geres
        logger.critical("\nBuffers geres:")
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
        Retourne les infos sur les buffers geres
        
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
        Optimise l'utilisation memoire
        
        Tips d'optimisation appliques automatiquement
        """
        logger.info("" Optimisation memoire...")
        
        # Activer le GC automatique
        gc.enable()
        
        # Ajuster les seuils du GC
        gc.set_threshold(700, 10, 10)
        
        # Force un nettoyage complet
        self.force_gc()
        
        logger.info("""| Optimisation terminee")
