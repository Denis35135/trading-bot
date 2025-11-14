"""
Model Selector pour The Bot
SÃƒÂ©lection automatique du meilleur modÃƒÂ¨le
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .ensemble import MLEnsemble
from .model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """ReprÃƒÂ©sente un modÃƒÂ¨le candidat"""
    name: str
    ensemble: MLEnsemble
    config: Dict
    score: float = 0.0
    metrics: Optional[Dict] = None


class ModelSelector:
    """
    SÃƒÂ©lectionneur de modÃƒÂ¨le automatique
    
    ResponsabilitÃƒÂ©s:
    - Tester plusieurs configurations de modÃƒÂ¨les
    - Comparer les performances
    - SÃƒÂ©lectionner le meilleur modÃƒÂ¨le selon une mÃƒÂ©trique
    - GÃƒÂ©nÃƒÂ©rer des rapports de comparaison
    """
    
    def __init__(self, selection_metric: str = 'accuracy'):
        """
        Initialise le sÃƒÂ©lecteur
        
        Args:
            selection_metric: MÃƒÂ©trique pour sÃƒÂ©lectionner le meilleur
                            ('accuracy', 'f1_score', 'precision', 'recall')
        """
        self.selection_metric = selection_metric
        self.evaluator = ModelEvaluator()
        self.candidates = []
        
        logger.info(f"Ã¢Å“â€¦ Model Selector initialisÃƒÂ© (mÃƒÂ©trique: {selection_metric})")
    
    def add_candidate(self, 
                     name: str, 
                     ensemble: MLEnsemble, 
                     config: Dict):
        """
        Ajoute un modÃƒÂ¨le candidat ÃƒÂ  la compÃƒÂ©tition
        
        Args:
            name: Nom du candidat
            ensemble: Ensemble de modÃƒÂ¨les
            config: Configuration utilisÃƒÂ©e
        """
        candidate = ModelCandidate(
            name=name,
            ensemble=ensemble,
            config=config
        )
        self.candidates.append(candidate)
        logger.info(f"Ã¢Å“â€¦ Candidat ajoutÃƒÂ©: {name}")
    
    def evaluate_candidates(self, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray) -> List[ModelCandidate]:
        """
        Ãƒâ€°value tous les candidats
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            
        Returns:
            Liste des candidats triÃƒÂ©s par score (meilleur en premier)
        """
        logger.info(f"Ã°Å¸â€Â Ãƒâ€°valuation de {len(self.candidates)} candidats...")
        
        for candidate in self.candidates:
            try:
                # Ãƒâ€°valuer le modÃƒÂ¨le
                metrics = self.evaluator.evaluate_detailed(
                    candidate.ensemble, 
                    X_test, 
                    y_test
                )
                
                # Extraire le score selon la mÃƒÂ©trique choisie
                score = metrics.get(self.selection_metric, 0.0)
                
                candidate.metrics = metrics
                candidate.score = score
                
                logger.info(f"  {candidate.name}: {self.selection_metric}={score:.2%}")
                
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur ÃƒÂ©valuation {candidate.name}: {e}")
                candidate.score = 0.0
                candidate.metrics = {}
        
        # Trier par score dÃƒÂ©croissant
        self.candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Ã¢Å“â€¦ Ãƒâ€°valuation terminÃƒÂ©e. Meilleur: {self.candidates[0].name} ({self.candidates[0].score:.2%})")
        
        return self.candidates
    
    def select_best(self, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray,
                   min_score: float = 0.6) -> Optional[ModelCandidate]:
        """
        SÃƒÂ©lectionne le meilleur modÃƒÂ¨le
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            min_score: Score minimum acceptable
            
        Returns:
            Meilleur candidat ou None si aucun ne passe le seuil
        """
        if not self.candidates:
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â  Aucun candidat ÃƒÂ  ÃƒÂ©valuer")
            return None
        
        # Ãƒâ€°valuer tous les candidats
        self.evaluate_candidates(X_test, y_test)
        
        # Prendre le meilleur
        best = self.candidates[0]
        
        # VÃƒÂ©rifier le score minimum
        if best.score < min_score:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â  Meilleur score {best.score:.2%} < minimum requis {min_score:.2%}")
            logger.warning("Aucun modÃƒÂ¨le ne satisfait le critÃƒÂ¨re minimum")
            return None
        
        logger.info(f"Ã°Å¸Ââ€  Meilleur modÃƒÂ¨le sÃƒÂ©lectionnÃƒÂ©: {best.name} ({self.selection_metric}={best.score:.2%})")
        
        return best
    
    def auto_select(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   min_score: float = 0.6) -> Optional[MLEnsemble]:
        """
        SÃƒÂ©lection automatique avec plusieurs configurations prÃƒÂ©dÃƒÂ©finies
        
        Args:
            X_train: Features d'entraÃƒÂ®nement
            y_train: Labels d'entraÃƒÂ®nement
            X_test: Features de test
            y_test: Labels de test
            min_score: Score minimum acceptable
            
        Returns:
            Meilleur ensemble ou None
        """
        logger.info("Ã°Å¸Â¤â€“ SÃƒÂ©lection automatique de modÃƒÂ¨le...")
        
        # Configurations ÃƒÂ  tester (du plus rapide au plus prÃƒÂ©cis)
        configs = [
            {
                'name': 'fast',
                'n_estimators': 50,
                'max_depth_lgb': 4,
                'max_depth_xgb': 4,
                'max_depth_rf': 6,
                'learning_rate': 0.1
            },
            {
                'name': 'balanced',
                'n_estimators': 100,
                'max_depth_lgb': 6,
                'max_depth_xgb': 5,
                'max_depth_rf': 8,
                'learning_rate': 0.1
            },
            {
                'name': 'accurate',
                'n_estimators': 150,
                'max_depth_lgb': 8,
                'max_depth_xgb': 6,
                'max_depth_rf': 10,
                'learning_rate': 0.05
            }
        ]
        
        # Tester chaque configuration
        for config in configs:
            name = config.pop('name')
            
            try:
                logger.info(f"  Test configuration: {name}")
                
                # CrÃƒÂ©er et entraÃƒÂ®ner l'ensemble
                ensemble = MLEnsemble(config)
                ensemble.train(X_train, y_train, X_test, y_test)
                
                # Ajouter comme candidat
                self.add_candidate(f"config_{name}", ensemble, config)
                
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Erreur config {name}: {e}")
        
        # SÃƒÂ©lectionner le meilleur
        best = self.select_best(X_test, y_test, min_score=min_score)
        
        if best:
            return best.ensemble
        
        return None
    
    def get_comparison_report(self) -> str:
        """
        GÃƒÂ©nÃƒÂ¨re un rapport de comparaison des modÃƒÂ¨les
        
        Returns:
            Rapport formatÃƒÂ© pour affichage console
        """
        if not self.candidates:
            return "Aucun candidat ÃƒÂ  comparer"
        
        report = "\n" + "="*80 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "="*80 + "\n\n"
        
        report += f"Selection Metric: {self.selection_metric}\n"
        report += f"Total Candidates: {len(self.candidates)}\n\n"
        
        # Tableau comparatif
        report += f"{'Rank':<6} {'Name':<20} {'Score':<12} {'Accuracy':<12} {'F1':<12}\n"
        report += "-"*80 + "\n"
        
        for rank, candidate in enumerate(self.candidates, 1):
            metrics = candidate.metrics or {}
            report += f"{rank:<6} "
            report += f"{candidate.name:<20} "
            report += f"{candidate.score:<11.2%} "
            report += f"{metrics.get('accuracy', 0):<11.2%} "
            report += f"{metrics.get('f1_score', 0):<11.2%}\n"
        
        report += "\n"
        
        # DÃƒÂ©tails du meilleur modÃƒÂ¨le
        if self.candidates and self.candidates[0].score > 0:
            best = self.candidates[0]
            report += "Ã°Å¸Ââ€  BEST MODEL DETAILS\n"
            report += "-"*80 + "\n"
            report += f"Name: {best.name}\n\n"
            
            if best.metrics:
                report += "Performance Metrics:\n"
                report += f"  Accuracy:       {best.metrics.get('accuracy', 0):.2%}\n"
                report += f"  Precision:      {best.metrics.get('precision', 0):.2%}\n"
                report += f"  Recall:         {best.metrics.get('recall', 0):.2%}\n"
                report += f"  F1 Score:       {best.metrics.get('f1_score', 0):.2%}\n"
                report += f"  Specificity:    {best.metrics.get('specificity', 0):.2%}\n"
                report += "\n"
            
            report += "Configuration:\n"
            for key, value in best.config.items():
                report += f"  {key}: {value}\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def clear_candidates(self):
        """Vide la liste des candidats"""
        self.candidates.clear()
        logger.info("Ã°Å¸â€”â€˜Ã¯Â¸Â  Candidats effacÃƒÂ©s")
    
    def get_all_scores(self) -> Dict[str, float]:
        """
        Retourne les scores de tous les candidats
        
        Returns:
            Dict {name: score}
        """
        return {c.name: c.score for c in self.candidates}


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Model Selector"""
    
    print("\n=== Test Model Selector ===\n")
    
    # DonnÃƒÂ©es synthÃƒÂ©tiques
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    # Split
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # CrÃƒÂ©er le sÃƒÂ©lecteur
    selector = ModelSelector(selection_metric='accuracy')
    
    # Test 1: Ajouter manuellement des candidats
    print("\nÃ°Å¸â€œÅ  Test avec candidats manuels:")
    
    for i, n_est in enumerate([50, 100, 150]):
        config = {
            'n_estimators': n_est,
            'max_depth_lgb': 5 + i,
            'confidence_threshold': 0.65
        }
        
        ensemble = MLEnsemble(config)
        ensemble.train(X_train, y_train, X_test, y_test)
        
        selector.add_candidate(f'model_{n_est}', ensemble, config)
    
    # SÃƒÂ©lectionner le meilleur
    best = selector.select_best(X_test, y_test, min_score=0.5)
    
    if best:
        print(f"\nÃ¢Å“â€¦ Meilleur modÃƒÂ¨le: {best.name}")
        print(f"   Score: {best.score:.2%}")
    
    # Rapport de comparaison
    print(selector.get_comparison_report())
    
    # Test 2: SÃƒÂ©lection automatique
    print("\nÃ°Å¸Â¤â€“ Test sÃƒÂ©lection automatique:")
    selector.clear_candidates()
    
    best_auto = selector.auto_select(X_train, y_train, X_test, y_test, min_score=0.5)
    
    if best_auto:
        print("Ã¢Å“â€¦ SÃƒÂ©lection automatique rÃƒÂ©ussie")
        print(selector.get_comparison_report())
    
    # Scores de tous les candidats
    print("\nÃ°Å¸â€œÅ  Tous les scores:")
    scores = selector.get_all_scores()
    for name, score in scores.items():
        print(f"  {name}: {score:.2%}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")
