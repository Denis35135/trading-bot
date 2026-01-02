#!/usr/bin/env python3
"""
Script de sauvegarde automatique des donnees
Backup des modeles ML, logs, et donnees de trading
"""

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path
import tarfile
import argparse

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackupManager:
    """
    Gestionnaire de sauvegardes automatiques
    
    Sauvegarde:
    - Modeles ML entranes
    - Logs importants
    - Historique des trades
    - Configuration
    - Base de donnees
    """
    
    def __init__(self, config: dict = None):
        """
        Initialise le backup manager
        
        Args:
            config: Configuration du backup
        """
        self.config = config or {
            'backup_dir': 'backups',
            'data_dirs': ['data/models', 'data/logs', 'data/cache'],
            'config_files': ['config.py', '.env'],
            'retention_days': 7,  # Garder 7 jours de backups
            'compression': True
        }
        
        self.backup_root = Path(self.config['backup_dir'])
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f""| Backup Manager initialise")
        logger.info(f"   Dossier backups: {self.backup_root}")
    
    def create_backup(self, backup_name: str = None) -> str:
        """
        Cree une sauvegarde complete
        
        Args:
            backup_name: Nom du backup (auto si None)
            
        Returns:
            Chemin du backup cree
        """
        try:
            # Nom du backup
            if not backup_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"backup_{timestamp}"
            
            backup_path = self.backup_root / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f""" Creation du backup: {backup_name}")
            
            # 1. Sauvegarder les dossiers de donnees
            for data_dir in self.config['data_dirs']:
                self._backup_directory(data_dir, backup_path)
            
            # 2. Sauvegarder les fichiers de config
            self._backup_config_files(backup_path)
            
            # 3. Creer un fichier info
            self._create_backup_info(backup_path)
            
            # 4. Compression si activee
            if self.config['compression']:
                compressed_path = self._compress_backup(backup_path)
                logger.info(f"""| Backup compresse: {compressed_path}")
                return str(compressed_path)
            
            logger.info(f"""| Backup cree: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"' Erreur creation backup: {e}")
            return None
    
    def _backup_directory(self, source_dir: str, backup_path: Path):
        """
        Sauvegarde un dossier
        
        Args:
            source_dir: Dossier source
            backup_path: Chemin du backup
        """
        source = Path(source_dir)
        
        if not source.exists():
            logger.warning(f" Dossier non trouve: {source_dir}")
            return
        
        dest = backup_path / source.name
        
        try:
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
                file_count = sum(1 for _ in dest.rglob('*') if _.is_file())
                logger.info(f"   "" {source_dir}: {file_count} fichiers sauvegardes")
            else:
                shutil.copy2(source, dest)
                logger.info(f"   "" {source_dir}: fichier sauvegarde")
                
        except Exception as e:
            logger.error(f"   """ Erreur backup {source_dir}: {e}")
    
    def _backup_config_files(self, backup_path: Path):
        """
        Sauvegarde les fichiers de configuration
        
        Args:
            backup_path: Chemin du backup
        """
        config_dir = backup_path / 'config'
        config_dir.mkdir(exist_ok=True)
        
        for config_file in self.config['config_files']:
            source = Path(config_file)
            
            if source.exists():
                try:
                    dest = config_dir / source.name
                    shutil.copy2(source, dest)
                    logger.info(f"   "" Config: {config_file}")
                except Exception as e:
                    logger.error(f"   """ Erreur config {config_file}: {e}")
    
    def _create_backup_info(self, backup_path: Path):
        """
        Cree un fichier d'info sur le backup
        
        Args:
            backup_path: Chemin du backup
        """
        info_file = backup_path / 'backup_info.txt'
        
        try:
            with open(info_file, 'w') as f:
                f.write(f"Backup cree le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nom du backup: {backup_path.name}\n")
                f.write(f"\nContenu:\n")
                
                for item in backup_path.iterdir():
                    if item.is_dir():
                        file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                        f.write(f"  - {item.name}/  ({file_count} fichiers)\n")
                    else:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        f.write(f"  - {item.name}  ({size_mb:.2f} MB)\n")
            
            logger.info(f"   "" Info backup creee")
            
        except Exception as e:
            logger.error(f"   """ Erreur creation info: {e}")
    
    def _compress_backup(self, backup_path: Path) -> Path:
        """
        Compresse un backup en tar.gz
        
        Args:
            backup_path: Chemin du backup
            
        Returns:
            Chemin du fichier compresse
        """
        try:
            archive_name = f"{backup_path.name}.tar.gz"
            archive_path = self.backup_root / archive_name
            
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_path.name)
            
            # Supprimer le dossier non compresse
            shutil.rmtree(backup_path)
            
            size_mb = archive_path.stat().st_size / (1024 * 1024)
            logger.info(f"   "" Archive creee: {archive_name} ({size_mb:.2f} MB)")
            
            return archive_path
            
        except Exception as e:
            logger.error(f"   """ Erreur compression: {e}")
            return backup_path
    
    def list_backups(self) -> list:
        """
        Liste tous les backups disponibles
        
        Returns:
            Liste des backups
        """
        backups = []
        
        for item in self.backup_root.iterdir():
            if item.is_dir() or item.suffix == '.gz':
                stat = item.stat()
                backups.append({
                    'name': item.name,
                    'path': str(item),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'is_compressed': item.suffix == '.gz'
                })
        
        # Trier par date de creation (plus recent en premier)
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        return backups
    
    def cleanup_old_backups(self):
        """
        Nettoie les vieux backups selon la politique de retention
        """
        try:
            logger.info(f" Nettoyage des backups (retention: {self.config['retention_days']} jours)")
            
            cutoff_date = datetime.now().timestamp() - (self.config['retention_days'] * 86400)
            deleted_count = 0
            
            for item in self.backup_root.iterdir():
                if item.stat().st_ctime < cutoff_date:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        
                        logger.info(f"   "" Supprime: {item.name}")
                        deleted_count += 1
                        
                    except Exception as e:
                        logger.error(f"   """ Erreur suppression {item.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"""| {deleted_count} ancien(s) backup(s) supprime(s)")
            else:
                logger.info(f"""| Aucun backup  supprimer")
                
        except Exception as e:
            logger.error(f"' Erreur cleanup: {e}")
    
    def restore_backup(self, backup_name: str, target_dir: str = '.'):
        """
        Restaure un backup
        
        Args:
            backup_name: Nom du backup  restaurer
            target_dir: Dossier de destination
        """
        try:
            logger.info(f""" Restauration du backup: {backup_name}")
            
            backup_path = self.backup_root / backup_name
            
            if not backup_path.exists():
                logger.error(f"' Backup non trouve: {backup_name}")
                return False
            
            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            
            # Si compresse, decompresser d'abord
            if backup_path.suffix == '.gz':
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(target)
                logger.info(f"   "" Archive decompressee")
            else:
                shutil.copytree(backup_path, target, dirs_exist_ok=True)
            
            logger.info(f"""| Backup restaure dans: {target}")
            return True
            
        except Exception as e:
            logger.error(f"' Erreur restauration: {e}")
            return False
    
    def get_backup_stats(self) -> dict:
        """
        Retourne les statistiques des backups
        
        Returns:
            Dict avec les stats
        """
        backups = self.list_backups()
        
        total_size = sum(b['size_mb'] for b in backups)
        
        return {
            'total_backups': len(backups),
            'total_size_mb': total_size,
            'oldest_backup': backups[-1]['created'] if backups else None,
            'newest_backup': backups[0]['created'] if backups else None,
            'compressed_count': sum(1 for b in backups if b['is_compressed'])
        }


def main():
    """Point d'entree du script"""
    parser = argparse.ArgumentParser(description='Gestion des backups')
    parser.add_argument('action', choices=['create', 'list', 'cleanup', 'restore', 'stats'],
                       help='Action  effectuer')
    parser.add_argument('--name', help='Nom du backup (pour restore)')
    parser.add_argument('--retention', type=int, default=7,
                       help='Nombre de jours de retention (defaut: 7)')
    parser.add_argument('--no-compress', action='store_true',
                       help='Desactiver la compression')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'backup_dir': 'backups',
        'data_dirs': ['data/models', 'data/logs', 'data/cache'],
        'config_files': ['config.py', '.env'],
        'retention_days': args.retention,
        'compression': not args.no_compress
    }
    
    manager = BackupManager(config)
    
    # Executer l'action
    if args.action == 'create':
        print("\n" + "="*50)
        print(""" CR"ATION D'UN NOUVEAU BACKUP")
        print("="*50)
        backup_path = manager.create_backup()
        if backup_path:
            print(f"\n""| Backup cree avec succes!")
            print(f"" Emplacement: {backup_path}")
        else:
            print("\n' "chec de la creation du backup")
    
    elif args.action == 'list':
        print("\n" + "="*50)
        print(""" LISTE DES BACKUPS DISPONIBLES")
        print("="*50)
        backups = manager.list_backups()
        
        if not backups:
            print("\n"  Aucun backup trouve")
        else:
            print(f"\nTotal: {len(backups)} backup(s)\n")
            for i, backup in enumerate(backups, 1):
                compressed = ""|" if backup['is_compressed'] else """
                print(f"{i}. {compressed} {backup['name']}")
                print(f"   Taille: {backup['size_mb']:.2f} MB")
                print(f"   Cree le: {backup['created'].strftime('%Y-%m-%d %H:%M:%S')}")
                print()
    
    elif args.action == 'cleanup':
        print("\n" + "="*50)
        print(" NETTOYAGE DES VIEUX BACKUPS")
        print("="*50)
        manager.cleanup_old_backups()
    
    elif args.action == 'restore':
        if not args.name:
            print("' Erreur: --name requis pour restore")
            sys.exit(1)
        
        print("\n" + "="*50)
        print(""" RESTAURATION D'UN BACKUP")
        print("="*50)
        success = manager.restore_backup(args.name)
        if success:
            print("\n""| Backup restaure avec succes!")
        else:
            print("\n' "chec de la restauration")
    
    elif args.action == 'stats':
        print("\n" + "="*50)
        print("" STATISTIQUES DES BACKUPS")
        print("="*50)
        stats = manager.get_backup_stats()
        
        print(f"\nTotal backups: {stats['total_backups']}")
        print(f"Taille totale: {stats['total_size_mb']:.2f} MB")
        print(f"Backups compresses: {stats['compressed_count']}")
        
        if stats['newest_backup']:
            print(f"Plus recent: {stats['newest_backup'].strftime('%Y-%m-%d %H:%M:%S')}")
        if stats['oldest_backup']:
            print(f"Plus ancien: {stats['oldest_backup'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
