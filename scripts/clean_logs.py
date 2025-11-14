#!/usr/bin/env python3
"""
Script de nettoyage automatique des logs
Supprime les vieux logs et archive les importants
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import shutil
import argparse

# Ajouter le rÃƒÂ©pertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogCleaner:
    """
    Nettoyeur de logs automatique
    
    Fonctions:
    - Supprime les logs plus vieux que X jours
    - Compresse les logs rÃƒÂ©cents pour ÃƒÂ©conomiser de l'espace
    - Archive les logs critiques
    - GÃƒÂ©nÃƒÂ¨re des rapports de nettoyage
    """
    
    def __init__(self, config: dict = None):
        """
        Initialise le log cleaner
        
        Args:
            config: Configuration du nettoyage
        """
        self.config = config or {
            'logs_dir': 'data/logs',
            'archive_dir': 'data/logs/archive',
            'retention_days': 7,  # Garder 7 jours
            'compress_after_days': 1,  # Compresser aprÃƒÂ¨s 1 jour
            'critical_patterns': ['ERROR', 'CRITICAL', 'FATAL'],
            'keep_critical': True
        }
        
        self.logs_dir = Path(self.config['logs_dir'])
        self.archive_dir = Path(self.config['archive_dir'])
        
        # CrÃƒÂ©er les dossiers si nÃƒÂ©cessaire
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ã°Å¸Â§Â¹ Log Cleaner initialisÃƒÂ©")
        logger.info(f"   Dossier logs: {self.logs_dir}")
        logger.info(f"   RÃƒÂ©tention: {self.config['retention_days']} jours")
    
    def clean_logs(self) -> dict:
        """
        Nettoie les logs selon la configuration
        
        Returns:
            Statistiques du nettoyage
        """
        try:
            logger.info("Ã°Å¸â€â€ž DÃƒÂ©but du nettoyage des logs...")
            
            stats = {
                'total_files': 0,
                'deleted_files': 0,
                'compressed_files': 0,
                'archived_files': 0,
                'space_freed_mb': 0,
                'errors': []
            }
            
            cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
            compress_date = datetime.now() - timedelta(days=self.config['compress_after_days'])
            
            # Scanner tous les fichiers de log
            for log_file in self.logs_dir.rglob('*.log'):
                stats['total_files'] += 1
                
                try:
                    file_stat = log_file.stat()
                    file_age = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    # 1. Archiver les logs critiques
                    if self.config['keep_critical'] and self._is_critical_log(log_file):
                        if self._archive_critical_log(log_file):
                            stats['archived_files'] += 1
                    
                    # 2. Supprimer les vieux logs
                    if file_age < cutoff_date:
                        size_mb = file_stat.st_size / (1024 * 1024)
                        log_file.unlink()
                        stats['deleted_files'] += 1
                        stats['space_freed_mb'] += size_mb
                        logger.info(f"   Ã¢Å“â€œ SupprimÃƒÂ©: {log_file.name} ({size_mb:.2f} MB)")
                    
                    # 3. Compresser les logs rÃƒÂ©cents non compressÃƒÂ©s
                    elif file_age < compress_date and not log_file.suffix == '.gz':
                        if self._compress_log(log_file):
                            stats['compressed_files'] += 1
                            size_mb = file_stat.st_size / (1024 * 1024)
                            stats['space_freed_mb'] += size_mb * 0.7  # ~70% compression
                
                except Exception as e:
                    error_msg = f"Erreur traitement {log_file.name}: {e}"
                    logger.error(f"   Ã¢Å“â€” {error_msg}")
                    stats['errors'].append(error_msg)
            
            # 4. Nettoyer les logs compressÃƒÂ©s trop vieux
            self._clean_compressed_logs(cutoff_date, stats)
            
            # 5. GÃƒÂ©nÃƒÂ©rer un rapport
            self._generate_cleaning_report(stats)
            
            logger.info(f"Ã¢Å“â€¦ Nettoyage terminÃƒÂ©!")
            logger.info(f"   Fichiers traitÃƒÂ©s: {stats['total_files']}")
            logger.info(f"   Fichiers supprimÃƒÂ©s: {stats['deleted_files']}")
            logger.info(f"   Fichiers compressÃƒÂ©s: {stats['compressed_files']}")
            logger.info(f"   Espace libÃƒÂ©rÃƒÂ©: {stats['space_freed_mb']:.2f} MB")
            
            return stats
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur nettoyage logs: {e}")
            return None
    
    def _is_critical_log(self, log_file: Path) -> bool:
        """
        VÃƒÂ©rifie si un log contient des messages critiques
        
        Args:
            log_file: Fichier de log
            
        Returns:
            True si critique
        """
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return any(pattern in content for pattern in self.config['critical_patterns'])
        except Exception as e:
            logger.warning(f"Impossible de lire {log_file.name}: {e}")
            return False
    
    def _archive_critical_log(self, log_file: Path) -> bool:
        """
        Archive un log critique
        
        Args:
            log_file: Fichier de log
            
        Returns:
            True si succÃƒÂ¨s
        """
        try:
            # CrÃƒÂ©er un sous-dossier par date
            date_folder = self.archive_dir / datetime.now().strftime('%Y-%m')
            date_folder.mkdir(parents=True, exist_ok=True)
            
            # Copier avec timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"{log_file.stem}_{timestamp}{log_file.suffix}"
            archive_path = date_folder / archive_name
            
            shutil.copy2(log_file, archive_path)
            
            # Compresser l'archive
            self._compress_log(archive_path)
            
            logger.info(f"   Ã¢Å“â€œ ArchivÃƒÂ© (critique): {log_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"   Ã¢Å“â€” Erreur archivage {log_file.name}: {e}")
            return False
    
    def _compress_log(self, log_file: Path) -> bool:
        """
        Compresse un fichier de log en .gz
        
        Args:
            log_file: Fichier de log
            
        Returns:
            True si succÃƒÂ¨s
        """
        try:
            gz_file = log_file.with_suffix(log_file.suffix + '.gz')
            
            with open(log_file, 'rb') as f_in:
                with gzip.open(gz_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Supprimer l'original
            log_file.unlink()
            
            logger.info(f"   Ã¢Å“â€œ CompressÃƒÂ©: {log_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"   Ã¢Å“â€” Erreur compression {log_file.name}: {e}")
            return False
    
    def _clean_compressed_logs(self, cutoff_date: datetime, stats: dict):
        """
        Nettoie les logs compressÃƒÂ©s trop vieux
        
        Args:
            cutoff_date: Date limite
            stats: Statistiques ÃƒÂ  mettre ÃƒÂ  jour
        """
        for gz_file in self.logs_dir.rglob('*.gz'):
            try:
                file_stat = gz_file.stat()
                file_age = datetime.fromtimestamp(file_stat.st_mtime)
                
                if file_age < cutoff_date:
                    size_mb = file_stat.st_size / (1024 * 1024)
                    gz_file.unlink()
                    stats['deleted_files'] += 1
                    stats['space_freed_mb'] += size_mb
                    logger.info(f"   Ã¢Å“â€œ SupprimÃƒÂ© (gz): {gz_file.name}")
                    
            except Exception as e:
                logger.error(f"   Ã¢Å“â€” Erreur suppression {gz_file.name}: {e}")
    
    def _generate_cleaning_report(self, stats: dict):
        """
        GÃƒÂ©nÃƒÂ¨re un rapport de nettoyage
        
        Args:
            stats: Statistiques du nettoyage
        """
        try:
            report_file = self.logs_dir / 'cleaning_report.txt'
            
            with open(report_file, 'w') as f:
                f.write("="*50 + "\n")
                f.write("RAPPORT DE NETTOYAGE DES LOGS\n")
                f.write("="*50 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Statistiques:\n")
                f.write(f"  - Fichiers traitÃƒÂ©s: {stats['total_files']}\n")
                f.write(f"  - Fichiers supprimÃƒÂ©s: {stats['deleted_files']}\n")
                f.write(f"  - Fichiers compressÃƒÂ©s: {stats['compressed_files']}\n")
                f.write(f"  - Fichiers archivÃƒÂ©s: {stats['archived_files']}\n")
                f.write(f"  - Espace libÃƒÂ©rÃƒÂ©: {stats['space_freed_mb']:.2f} MB\n\n")
                
                if stats['errors']:
                    f.write("Erreurs rencontrÃƒÂ©es:\n")
                    for error in stats['errors']:
                        f.write(f"  - {error}\n")
            
            logger.info(f"   Ã¢Å“â€œ Rapport gÃƒÂ©nÃƒÂ©rÃƒÂ©: {report_file.name}")
            
        except Exception as e:
            logger.error(f"   Ã¢Å“â€” Erreur gÃƒÂ©nÃƒÂ©ration rapport: {e}")
    
    def get_logs_stats(self) -> dict:
        """
        Retourne les statistiques des logs
        
        Returns:
            Dict avec stats
        """
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'compressed_files': 0,
            'oldest_log': None,
            'newest_log': None
        }
        
        oldest_time = None
        newest_time = None
        
        for log_file in self.logs_dir.rglob('*.log*'):
            stats['total_files'] += 1
            stats['total_size_mb'] += log_file.stat().st_size / (1024 * 1024)
            
            if log_file.suffix == '.gz':
                stats['compressed_files'] += 1
            
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if oldest_time is None or mtime < oldest_time:
                oldest_time = mtime
                stats['oldest_log'] = log_file.name
            
            if newest_time is None or mtime > newest_time:
                newest_time = mtime
                stats['newest_log'] = log_file.name
        
        return stats
    
    def list_logs(self, pattern: str = None) -> list:
        """
        Liste les logs disponibles
        
        Args:
            pattern: Pattern de recherche (optionnel)
            
        Returns:
            Liste des logs
        """
        logs = []
        
        search_pattern = f'*{pattern}*' if pattern else '*'
        
        for log_file in self.logs_dir.rglob(search_pattern):
            if log_file.suffix in ['.log', '.gz']:
                stat = log_file.stat()
                logs.append({
                    'name': log_file.name,
                    'path': str(log_file),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'compressed': log_file.suffix == '.gz'
                })
        
        logs.sort(key=lambda x: x['modified'], reverse=True)
        return logs


def main():
    """Point d'entrÃƒÂ©e du script"""
    parser = argparse.ArgumentParser(description='Nettoyage des logs')
    parser.add_argument('action', choices=['clean', 'list', 'stats', 'compress'],
                       help='Action ÃƒÂ  effectuer')
    parser.add_argument('--retention', type=int, default=7,
                       help='Jours de rÃƒÂ©tention (dÃƒÂ©faut: 7)')
    parser.add_argument('--pattern', help='Pattern de recherche (pour list)')
    parser.add_argument('--no-critical', action='store_true',
                       help='Ne pas archiver les logs critiques')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'logs_dir': 'data/logs',
        'archive_dir': 'data/logs/archive',
        'retention_days': args.retention,
        'compress_after_days': 1,
        'critical_patterns': ['ERROR', 'CRITICAL', 'FATAL'],
        'keep_critical': not args.no_critical
    }
    
    cleaner = LogCleaner(config)
    
    # ExÃƒÂ©cuter l'action
    if args.action == 'clean':
        print("\n" + "="*50)
        print("Ã°Å¸Â§Â¹ NETTOYAGE DES LOGS")
        print("="*50)
        stats = cleaner.clean_logs()
        
        if stats:
            print(f"\nÃ¢Å“â€¦ Nettoyage terminÃƒÂ©!")
            print(f"Ã°Å¸â€œÅ  RÃƒÂ©sultats:")
            print(f"   - Fichiers traitÃƒÂ©s: {stats['total_files']}")
            print(f"   - Fichiers supprimÃƒÂ©s: {stats['deleted_files']}")
            print(f"   - Fichiers compressÃƒÂ©s: {stats['compressed_files']}")
            print(f"   - Fichiers archivÃƒÂ©s: {stats['archived_files']}")
            print(f"   - Espace libÃƒÂ©rÃƒÂ©: {stats['space_freed_mb']:.2f} MB")
            
            if stats['errors']:
                print(f"\nÃ¢Å¡Â Ã¯Â¸Â Erreurs: {len(stats['errors'])}")
        else:
            print("\nÃ¢ÂÅ’ Ãƒâ€°chec du nettoyage")
    
    elif args.action == 'list':
        print("\n" + "="*50)
        print("Ã°Å¸â€œâ€¹ LISTE DES LOGS")
        print("="*50)
        logs = cleaner.list_logs(args.pattern)
        
        if not logs:
            print("\nÃ¢â€žÂ¹Ã¯Â¸Â  Aucun log trouvÃƒÂ©")
        else:
            print(f"\nTotal: {len(logs)} log(s)\n")
            for i, log in enumerate(logs, 1):
                icon = "Ã°Å¸â€œÂ¦" if log['compressed'] else "Ã°Å¸â€œâ€ž"
                print(f"{i}. {icon} {log['name']}")
                print(f"   Taille: {log['size_mb']:.2f} MB")
                print(f"   ModifiÃƒÂ©: {log['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                print()
    
    elif args.action == 'stats':
        print("\n" + "="*50)
        print("Ã°Å¸â€œÅ  STATISTIQUES DES LOGS")
        print("="*50)
        stats = cleaner.get_logs_stats()
        
        print(f"\nTotal fichiers: {stats['total_files']}")
        print(f"Taille totale: {stats['total_size_mb']:.2f} MB")
        print(f"Fichiers compressÃƒÂ©s: {stats['compressed_files']}")
        print(f"Plus ancien: {stats['oldest_log']}")
        print(f"Plus rÃƒÂ©cent: {stats['newest_log']}")
    
    elif args.action == 'compress':
        print("\n" + "="*50)
        print("Ã°Å¸â€œÂ¦ COMPRESSION DES LOGS")
        print("="*50)
        
        compressed = 0
        for log_file in Path('data/logs').rglob('*.log'):
            if cleaner._compress_log(log_file):
                compressed += 1
        
        print(f"\nÃ¢Å“â€¦ {compressed} fichier(s) compressÃƒÂ©(s)")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
