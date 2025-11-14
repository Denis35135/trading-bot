"""
Model Evaluation pour The Bot
Ãƒâ€°valuation dÃƒÂ©taillÃƒÂ©e des modÃƒÂ¨les ML
"""

import numpy as np
from typing import Dict
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from .ensemble import MLEnsemble

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Ãƒâ€°valuateur de modÃƒÂ¨les ML
    
    ResponsabilitÃƒÂ©s:
    - Ãƒâ€°valuer les performances avec mÃƒÂ©triques dÃƒÂ©taillÃƒÂ©es
    - Comparer plusieurs modÃƒÂ¨les
    - GÃƒÂ©nÃƒÂ©rer des rapports formatÃƒÂ©s
    - Analyser la matrice de confusion
    """
    
    def __init__(self):
        """Initialise l'ÃƒÂ©valuateur"""
        logger.info("Ã¢Å“â€¦ Model Evaluator initialisÃƒÂ©")
    
    def evaluate_detailed(self, 
                         ensemble: MLEnsemble,
                         X_test: np.ndarray,
                         y_test: np.ndarray) -> Dict:
        """
        Ãƒâ€°valuation dÃƒÂ©taillÃƒÂ©e d'un ensemble
        
        Args:
            ensemble: Ensemble ÃƒÂ  ÃƒÂ©valuer
            X_test: Features de test
            y_test: Labels de test (0 ou 1)
            
        Returns:
            Dict avec mÃƒÂ©triques dÃƒÂ©taillÃƒÂ©es
        """
        logger.info(f"Ã°Å¸â€œÅ  Ãƒâ€°valuation dÃƒÂ©taillÃƒÂ©e sur {len(X_test)} samples")
        
        # PrÃƒÂ©dictions
        y_pred = ensemble.predict_batch(X_test)
        y_pred_binary = (y_pred > 0).astype(int)
        
        # MÃƒÂ©triques de base
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # MÃƒÂ©triques additionnelles
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Distribution des prÃƒÂ©dictions
        buy_signals = np.sum(y_pred == 1)
        sell_signals = np.sum(y_pred == -1)
        hold_signals = np.sum(y_pred == 0)
        
        results = {
            # MÃƒÂ©triques principales
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            
            # MÃƒÂ©triques additionnelles
            'specificity': float(specificity),
            'npv': float(npv),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            
            # Matrice de confusion
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            
            # Distribution des signaux
            'signal_distribution': {
                'buy': int(buy_signals),
                'sell': int(sell_signals),
                'hold': int(hold_signals),
                'total': len(y_pred)
            },
            
            # Pourcentages
            'percentages': {
                'buy_pct': float(buy_signals / len(y_pred)),
                'sell_pct': float(sell_signals / len(y_pred)),
                'hold_pct': float(hold_signals / len(y_pred))
            }
        }
        
        logger.info(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")
        
        return results
    
    def compare_models(self, 
                      models_dict: Dict[str, MLEnsemble],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict:
        """
        Compare plusieurs modÃƒÂ¨les
        
        Args:
            models_dict: Dict {name: ensemble}
            X_test: Features de test
            y_test: Labels de test
            
        Returns:
            Dict avec comparaison dÃƒÂ©taillÃƒÂ©e
        """
        logger.info(f"Ã°Å¸â€Â Comparaison de {len(models_dict)} modÃƒÂ¨les")
        
        comparison = {}
        
        for name, ensemble in models_dict.items():
            logger.info(f"  Ãƒâ€°valuation {name}...")
            results = self.evaluate_detailed(ensemble, X_test, y_test)
            comparison[name] = results
        
        # Trouver le meilleur modÃƒÂ¨le selon diffÃƒÂ©rentes mÃƒÂ©triques
        best_accuracy = max(comparison.items(), key=lambda x: x[1]['accuracy'])
        best_precision = max(comparison.items(), key=lambda x: x[1]['precision'])
        best_recall = max(comparison.items(), key=lambda x: x[1]['recall'])
        best_f1 = max(comparison.items(), key=lambda x: x[1]['f1_score'])
        
        return {
            'models': comparison,
            'best_by_metric': {
                'accuracy': {
                    'name': best_accuracy[0],
                    'score': best_accuracy[1]['accuracy']
                },
                'precision': {
                    'name': best_precision[0],
                    'score': best_precision[1]['precision']
                },
                'recall': {
                    'name': best_recall[0],
                    'score': best_recall[1]['recall']
                },
                'f1_score': {
                    'name': best_f1[0],
                    'score': best_f1[1]['f1_score']
                }
            }
        }
    
    def generate_report(self, evaluation: Dict) -> str:
        """
        GÃƒÂ©nÃƒÂ¨re un rapport textuel d'ÃƒÂ©valuation
        
        Args:
            evaluation: RÃƒÂ©sultats de evaluate_detailed()
            
        Returns:
            Rapport formatÃƒÂ© pour affichage console
        """
        report = "\n" + "="*70 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        # MÃƒÂ©triques principales
        report += "Ã°Å¸â€œÅ  MAIN METRICS\n"
        report += "-"*70 + "\n"
        report += f"Accuracy:         {evaluation['accuracy']:>8.2%}\n"
        report += f"Precision:        {evaluation['precision']:>8.2%}\n"
        report += f"Recall:           {evaluation['recall']:>8.2%}\n"
        report += f"F1 Score:         {evaluation['f1_score']:>8.2%}\n"
        report += f"Specificity:      {evaluation['specificity']:>8.2%}\n"
        report += f"NPV:              {evaluation['npv']:>8.2%}\n"
        report += "\n"
        
        # Taux d'erreur
        report += "Ã¢Å¡Â Ã¯Â¸Â  ERROR RATES\n"
        report += "-"*70 + "\n"
        report += f"False Positive:   {evaluation['false_positive_rate']:>8.2%}\n"
        report += f"False Negative:   {evaluation['false_negative_rate']:>8.2%}\n"
        report += "\n"
        
        # Matrice de confusion
        cm = evaluation['confusion_matrix']
        report += "Ã°Å¸Å½Â¯ CONFUSION MATRIX\n"
        report += "-"*70 + "\n"
        report += f"                  Predicted Negative    Predicted Positive\n"
        report += f"Actual Negative   {cm['true_negative']:>18}    {cm['false_positive']:>18}\n"
        report += f"Actual Positive   {cm['false_negative']:>18}    {cm['true_positive']:>18}\n"
        report += "\n"
        
        # Distribution des signaux
        dist = evaluation['signal_distribution']
        pct = evaluation['percentages']
        report += "Ã°Å¸â€œË† SIGNAL DISTRIBUTION\n"
        report += "-"*70 + "\n"
        report += f"Buy Signals:      {dist['buy']:>6}   ({pct['buy_pct']:>6.1%})\n"
        report += f"Sell Signals:     {dist['sell']:>6}   ({pct['sell_pct']:>6.1%})\n"
        report += f"Hold Signals:     {dist['hold']:>6}   ({pct['hold_pct']:>6.1%})\n"
        report += f"Total:            {dist['total']:>6}   (100.0%)\n"
        report += "\n"
        
        # InterprÃƒÂ©tation
        report += "Ã°Å¸â€™Â¡ INTERPRETATION\n"
        report += "-"*70 + "\n"
        
        if evaluation['accuracy'] >= 0.70:
            report += "Ã¢Å“â€¦ Excellent: Accuracy >= 70%\n"
        elif evaluation['accuracy'] >= 0.60:
            report += "Ã¢Å“â€œ  Good: Accuracy >= 60%\n"
        else:
            report += "Ã¢Å¡Â Ã¯Â¸Â  Needs Improvement: Accuracy < 60%\n"
        
        if evaluation['precision'] >= 0.70:
            report += "Ã¢Å“â€¦ Low False Positives: Precision >= 70%\n"
        else:
            report += "Ã¢Å¡Â Ã¯Â¸Â  High False Positives: Precision < 70%\n"
        
        if evaluation['recall'] >= 0.70:
            report += "Ã¢Å“â€¦ Low False Negatives: Recall >= 70%\n"
        else:
            report += "Ã¢Å¡Â Ã¯Â¸Â  High False Negatives: Recall < 70%\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """
        GÃƒÂ©nÃƒÂ¨re un rapport de comparaison de modÃƒÂ¨les
        
        Args:
            comparison: RÃƒÂ©sultat de compare_models()
            
        Returns:
            Rapport formatÃƒÂ©
        """
        report = "\n" + "="*80 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "="*80 + "\n\n"
        
        # Tableau comparatif
        report += f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n"
        report += "-"*80 + "\n"
        
        for name, results in comparison['models'].items():
            report += f"{name:<20} "
            report += f"{results['accuracy']:<11.2%} "
            report += f"{results['precision']:<11.2%} "
            report += f"{results['recall']:<11.2%} "
            report += f"{results['f1_score']:<11.2%}\n"
        
        report += "\n"
        
        # Meilleurs modÃƒÂ¨les par mÃƒÂ©trique
        report += "Ã°Å¸Ââ€  BEST MODELS BY METRIC\n"
        report += "-"*80 + "\n"
        
        best = comparison['best_by_metric']
        report += f"Best Accuracy:    {best['accuracy']['name']:<20} ({best['accuracy']['score']:.2%})\n"
        report += f"Best Precision:   {best['precision']['name']:<20} ({best['precision']['score']:.2%})\n"
        report += f"Best Recall:      {best['recall']['name']:<20} ({best['recall']['score']:.2%})\n"
        report += f"Best F1 Score:    {best['f1_score']['name']:<20} ({best['f1_score']['score']:.2%})\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def calculate_trading_metrics(self, 
                                 y_true: np.ndarray,
                                 y_pred_signals: np.ndarray,
                                 returns: np.ndarray) -> Dict:
        """
        Calcule des mÃƒÂ©triques spÃƒÂ©cifiques au trading
        
        Args:
            y_true: Labels rÃƒÂ©els (0 ou 1)
            y_pred_signals: Signaux prÃƒÂ©dits (-1, 0, 1)
            returns: Returns rÃƒÂ©alisÃƒÂ©s pour chaque trade
            
        Returns:
            Dict avec mÃƒÂ©triques de trading
        """
        # Convertir signaux en positions
        y_pred_binary = (y_pred_signals > 0).astype(int)
        
        # Identifier les vrais positifs et nÃƒÂ©gatifs
        correct_predictions = (y_true == y_pred_binary)
        
        # Profits sur prÃƒÂ©dictions correctes vs incorrectes
        correct_returns = returns[correct_predictions]
        incorrect_returns = returns[~correct_predictions]
        
        return {
            'avg_return_correct': float(np.mean(correct_returns)) if len(correct_returns) > 0 else 0,
            'avg_return_incorrect': float(np.mean(incorrect_returns)) if len(incorrect_returns) > 0 else 0,
            'total_return': float(np.sum(returns)),
            'sharpe_approx': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
            'win_rate': float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0,
            'profit_factor': float(np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))) if np.sum(returns < 0) != 0 else 0
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du Model Evaluator"""
    
    print("\n=== Test Model Evaluator ===\n")
    
    from .ensemble import MLEnsemble
    
    # DonnÃƒÂ©es synthÃƒÂ©tiques
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    
    X_test = np.random.randn(n_samples, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] - X_test[:, 2] > 0).astype(int)
    
    print(f"DonnÃƒÂ©es test: {n_samples} samples")
    print(f"Distribution: {np.sum(y_test)} positifs ({np.sum(y_test)/len(y_test):.1%})")
    
    # CrÃƒÂ©er et entraÃƒÂ®ner un ensemble de test
    ensemble = MLEnsemble({'n_estimators': 50})
    X_train = np.random.randn(1000, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] - X_train[:, 2] > 0).astype(int)
    ensemble.train(X_train, y_train)
    
    # CrÃƒÂ©er l'ÃƒÂ©valuateur
    evaluator = ModelEvaluator()
    
    # Ãƒâ€°valuation dÃƒÂ©taillÃƒÂ©e
    print("\nÃ°Å¸â€œÅ  Ãƒâ€°valuation dÃƒÂ©taillÃƒÂ©e:")
    evaluation = evaluator.evaluate_detailed(ensemble, X_test, y_test)
    
    # GÃƒÂ©nÃƒÂ©rer et afficher le rapport
    report = evaluator.generate_report(evaluation)
    print(report)
    
    # Test mÃƒÂ©triques de trading
    print("\nÃ°Å¸â€™Â° MÃƒÂ©triques de trading:")
    y_pred = ensemble.predict_batch(X_test)
    returns = np.random.randn(n_samples) * 0.02  # Returns simulÃƒÂ©s
    
    trading_metrics = evaluator.calculate_trading_metrics(y_test, y_pred, returns)
    for key, value in trading_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    print("\nÃ¢Å“â€¦ Tests terminÃƒÂ©s")
