#!/usr/bin/env python3
"""
Script de backtesting
Teste les stratÃ©gies sur donnÃ©es historiques
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from config import config
from utils.backtester import Backtester
from utils.database import Database
from strategies.scalping import ScalpingStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pattern import PatternRecognitionStrategy
from strategies.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure le logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Backtest des stratÃ©gies de trading')
    
    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Date de dÃ©but (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2024-01-01',
        help='Date de fin (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='Capital initial en USDT'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'scalping', 'momentum', 'mean_reversion', 'pattern', 'ml'],
        help='StratÃ©gies Ã  tester'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTCUSDT', 'ETHUSDT'],
        help='Symboles Ã  tester'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        default=0.0007,
        help='Commission Binance (0.07% par dÃ©faut)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/backtest',
        help='Dossier de sortie pour les rÃ©sultats'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Mode verbeux'
    )
    
    return parser.parse_args()


def initialize_strategies(strategy_names: list) -> dict:
    """
    Initialise les stratÃ©gies demandÃ©es
    
    Args:
        strategy_names: Liste des noms de stratÃ©gies
        
    Returns:
        Dict des stratÃ©gies initialisÃ©es
    """
    all_strategies = {
        'scalping': ScalpingStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'pattern': PatternRecognitionStrategy,
        'ml': MLStrategy
    }
    
    if 'all' in strategy_names:
        strategy_names = list(all_strategies.keys())
    
    strategies = {}
    for name in strategy_names:
        if name in all_strategies:
            strategies[name] = all_strategies[name]()
            logger.info(f"âœ… StratÃ©gie '{name}' initialisÃ©e")
        else:
            logger.warning(f"âš ï¸ StratÃ©gie '{name}' inconnue, ignorÃ©e")
    
    return strategies


def run_backtest(args):
    """
    Lance le backtest
    
    Args:
        args: Arguments parsÃ©s
    """
    print("="*80)
    print("ðŸ”¬ BACKTESTING - THE BOT")
    print("="*80)
    print(f"ðŸ“… PÃ©riode: {args.start} â†’ {args.end}")
    print(f"ðŸ’° Capital initial: ${args.capital:,.2f}")
    print(f"ðŸ“Š Symboles: {', '.join(args.symbols)}")
    print(f"ðŸŽ¯ StratÃ©gies: {', '.join(args.strategies)}")
    print(f"ðŸ’¸ Commission: {args.commission:.2%}")
    print("="*80)
    print()
    
    try:
        # Initialiser les stratÃ©gies
        logger.info("Initialisation des stratÃ©gies...")
        strategies = initialize_strategies(args.strategies)
        
        if not strategies:
            logger.error("âŒ Aucune stratÃ©gie valide spÃ©cifiÃ©e")
            return False
        
        # Initialiser le backtester
        logger.info("Initialisation du backtester...")
        backtester = Backtester(
            initial_capital=args.capital,
            commission=args.commission
        )
        
        # Charger les donnÃ©es historiques
        logger.info("Chargement des donnÃ©es historiques...")
        for symbol in args.symbols:
            logger.info(f"  ðŸ“¥ Chargement {symbol}...")
            backtester.load_data(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end
            )
        
        # Lancer le backtest pour chaque stratÃ©gie
        results = {}
        for name, strategy in strategies.items():
            logger.info(f"\nðŸš€ Backtest de la stratÃ©gie '{name}'...")
            
            result = backtester.run(
                strategy=strategy,
                symbols=args.symbols
            )
            
            results[name] = result
            
            # Afficher les rÃ©sultats
            print(f"\nðŸ“Š RÃ‰SULTATS - {name.upper()}")
            print("-"*60)
            print(f"  Capital Final    : ${result['final_capital']:,.2f}")
            print(f"  P&L Total        : ${result['total_pnl']:+,.2f} ({result['total_pnl_pct']:+.2%})")
            print(f"  Nombre de Trades : {result['total_trades']}")
            print(f"  Win Rate         : {result['win_rate']:.1%}")
            print(f"  Profit Factor    : {result['profit_factor']:.2f}")
            print(f"  Sharpe Ratio     : {result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown     : {result['max_drawdown']:.2%}")
            print(f"  Avg Win          : {result['avg_win']:.2%}")
            print(f"  Avg Loss         : {result['avg_loss']:.2%}")
            print("-"*60)
        
        # Sauvegarder les rÃ©sultats
        logger.info(f"\nðŸ’¾ Sauvegarde des rÃ©sultats dans {args.output}...")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for name, result in results.items():
            output_file = output_dir / f"backtest_{name}_{timestamp}.json"
            backtester.save_results(result, output_file)
            logger.info(f"  âœ… {output_file}")
        
        # GÃ©nÃ©rer le rapport comparatif
        logger.info("\nðŸ“ˆ GÃ©nÃ©ration du rapport comparatif...")
        report_file = output_dir / f"backtest_comparison_{timestamp}.html"
        backtester.generate_comparison_report(results, report_file)
        logger.info(f"  âœ… {report_file}")
        
        print("\n" + "="*80)
        print("âœ… BACKTEST TERMINÃ‰ AVEC SUCCÃˆS")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"âŒ Erreur lors du backtest: {e}", exc_info=True)
        return False


def main():
    """Point d'entrÃ©e principal"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    success = run_backtest(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
