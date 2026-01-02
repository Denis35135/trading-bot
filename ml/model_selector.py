"""
Model Selector pour The Bot
Selection automatique du meilleur modele
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
    """Represente un modele candidat"""
    name: str
    ensemble: MLEnsemble
    config: Dict
    score: float = 0.0
    metrics: Optional[Dict] = None


class ModelSelector:
    """
    Selectionneur de modele automatique
    
    Responsabilites:
    - Tester plusieurs configurations de modeles
    - Comparer les performances
    - Selectionner le meilleur modele selon une metrique
    - Generer des rapports de comparaison
    """
    
    def __init__(self, selection_metric: str = 'accuracy'):
        """
        Initialise le selecteur
        
        Args:
            selection_metric: Metrique pour selectionner le meilleur
                            ('accuracy', 'f1_score', 'precision', 'recall')
        """
        self.selection_metric = selection_metric
        self.evaluator = ModelEvaluator()
        self.candidates = []
        
        logger.info(f"""| Model Selector initialise (metrique: {selection_metric})")
    
    def add_candidate(self, 
    """
    name: str,
    """
                     ensemble: MLEnsemble, 
                     config: Dict):
        """
        Ajoute un modele candidat  la competition
        
        Args:
            name: Nom du candidat
            ensemble: Ensemble de modeles
            config: Configuration utilisee
        """
        candidate = ModelCandidate(
            name=name,
            ensemble=ensemble,
            config=config
        )
        self.candidates.append(candidate)
        logger.info(f"""| Candidat ajoute: {name}")
    
    def evaluate_candidates(self, 
    """
    X_test: np.ndarray,
    """
                           y_test: np.ndarray) -> List[ModelCandidate]:
        """
        "value tous les candidats
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            
        Returns:
            Liste des candidats tries par score (meilleur en premier)
        """
        logger.info(f"" "valuation de {len(self.candidates)} candidats...")
        
        for candidate in self.candidates:
            try:
                # "valuer le modele
                metrics = self.evaluator.evaluate_detailed(
                    candidate.ensemble, 
                    X_test, 
                    y_test
                )
                
                # Extraire le score selon la metrique choisie
                score = metrics.get(self.selection_metric, 0.0)
                
                candidate.metrics = metrics
                candidate.score = score
                
                logger.info(f"  {candidate.name}: {self.selection_metric}={score:.2%}")
                
            except Exception as e:
                logger.error(f"' Erreur evaluation {candidate.name}: {e}")
                candidate.score = 0.0
                candidate.metrics = {}
        
        # Trier par score decroissant
        self.candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"""| "valuation terminee. Meilleur: {self.candidates[0].name} ({self.candidates[0].score:.2%})")
        
        return self.candidates
    
    def select_best(self, 
    """
    X_test: np.ndarray,
    """
                   y_test: np.ndarray,
                   min_score: float = 0.6) -> Optional[ModelCandidate]:
        """
        Selectionne le meilleur modele
        
        Args:
            X_test: Features de test
            y_test: Labels de test
            min_score: Score minimum acceptable
            
        Returns:
            Meilleur candidat ou None si aucun ne passe le seuil
        """
        if not self.candidates:
            logger.warning("  Aucun candidat  evaluer")
            return None
        
        # "valuer tous les candidats
        self.evaluate_candidates(X_test, y_test)
        
        # Prendre le meilleur
        best = self.candidates[0]
        
        # Verifier le score minimum
        if best.score < min_score:
            logger.warning(f"  Meilleur score {best.score:.2%} < minimum requis {min_score:.2%}")
            logger.warning("Aucun modele ne satisfait le critere minimum")
            return None
        
        logger.info(f"" Meilleur modele selectionne: {best.name} ({self.selection_metric}={best.score:.2%})")
        
        return best
    
    def auto_select(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   min_score: float = 0.6) -> Optional[MLEnsemble]:
        """
        Selection automatique avec plusieurs configurations predefinies
        
        Args:
            X_train: Features d'entranement
            y_train: Labels d'entranement
            X_test: Features de test
            y_test: Labels de test
            min_score: Score minimum acceptable
            
        Returns:
            Meilleur ensemble ou None
        """
        logger.info(""" Selection automatique de modele...")
        
        # Configurations  tester (du plus rapide au plus precis)
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
                
                # Creer et entraner l'ensemble
                ensemble = MLEnsemble(config)
                ensemble.train(X_train, y_train, X_test, y_test)
                
                # Ajouter comme candidat
                self.add_candidate(f"config_{name}", ensemble, config)
                
            except Exception as e:
                logger.error(f"' Erreur config {name}: {e}")
        
        # Selectionner le meilleur
        best = self.select_best(X_test, y_test, min_score=min_score)
        
        if best:
            return best.ensemble
        
        return None
    
    def get_comparison_report(self) -> str:
        """
        Genere un rapport de comparaison des modeles
        
        Returns:
            Rapport formate pour affichage console
        """
        if not self.candidates:
            return "Aucun candidat  comparer"
        
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
        
        # Details du meilleur modele
        if self.candidates and self.candidates[0].score > 0:
            best = self.candidates[0]
            report += "" BEST MODEL DETAILS\n"
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
        logger.info(""""  Candidats effaces")
    
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
    
    # Donnees synthetiques
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
    
    # Creer le selecteur
    selector = ModelSelector(selection_metric='accuracy')
    
    # Test 1: Ajouter manuellement des candidats
    print("\n" Test avec candidats manuels:")
    
    for i, n_est in enumerate([50, 100, 150]):
        config = {
            'n_estimators': n_est,
            'max_depth_lgb': 5 + i,
            'confidence_threshold': 0.65
        }
        
        ensemble = MLEnsemble(config)
        ensemble.train(X_train, y_train, X_test, y_test)
        
        selector.add_candidate(f'model_{n_est}', ensemble, config)
    
    # Selectionner le meilleur
    best = selector.select_best(X_test, y_test, min_score=0.5)
    
    if best:
        print(f"\n""| Meilleur modele: {best.name}")
        print(f"   Score: {best.score:.2%}")
    
    # Rapport de comparaison
    print(selector.get_comparison_report())
    
    # Test 2: Selection automatique
    print("\n"" Test selection automatique:")
    selector.clear_candidates()
    
    best_auto = selector.auto_select(X_train, y_train, X_test, y_test, min_score=0.5)
    
    if best_auto:
        print("""| Selection automatique reussie")
        print(selector.get_comparison_report())
    
    # Scores de tous les candidats
    print("\n" Tous les scores:")
    scores = selector.get_all_scores()
    for name, score in scores.items():
        print(f"  {name}: {score:.2%}")
    
    print("\n""| Tests termines")
