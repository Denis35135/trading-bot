"""
Report Generator pour The Bot
Genere des rapports de performance formates et detailles
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generateur de rapports de performance
    
    Responsabilites:
    - Generer des rapports console formates
    - Creer des rapports JSON pour export
    - Generer des resumes quotidiens/hebdomadaires
    - Creer des rapports d'erreurs et d'alertes
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le generateur de rapports
        
        Args:
            config: Configuration
        """
        self.config = config
        self.reports_dir = Path(getattr(config, 'REPORTS_DIR', 'data/reports'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"""| Report Generator initialise (dir: {self.reports_dir})")
    
    def generate_live_dashboard(self,
    """
    capital: float,
    """
                               initial_capital: float,
                               positions: List[Dict],
                               performance: Dict,
                               risk_metrics: Dict,
                               system_health: Dict) -> str:
        """
        Genere un dashboard en temps reel pour la console
        
        Args:
            capital: Capital actuel
            initial_capital: Capital initial
            positions: Liste des positions ouvertes
            performance: Metriques de performance
            risk_metrics: Metriques de risque
            system_health: Sante du systeme
            
        Returns:
            String formate pour affichage console
        """
        # Calculs
        pnl_total = capital - initial_capital
        pnl_pct = (pnl_total / initial_capital) * 100
        
        # Header
        dashboard = "\n" + "="*80 + "\n"
        dashboard += f"{'"" THE BOT - LIVE DASHBOARD':^80}\n"
        dashboard += f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}\n"
        dashboard += "="*80 + "\n\n"
        
        # Section Capital & P&L
        dashboard += "' CAPITAL & P&L\n"
        dashboard += "-"*80 + "\n"
        dashboard += f"Capital Initial:    ${initial_capital:>15,.2f}\n"
        dashboard += f"Capital Actuel:     ${capital:>15,.2f}\n"
        dashboard += f"P&L Total:          ${pnl_total:>15,.2f} ({pnl_pct:+.2f}%)\n"
        dashboard += f"P&L Journalier:     ${performance.get('daily_pnl', 0):>15,.2f}\n"
        dashboard += "\n"
        
        # Section Positions
        dashboard += "" POSITIONS OUVERTES\n"
        dashboard += "-"*80 + "\n"
        if positions:
            dashboard += f"{'Symbole':<12} {'Strategie':<15} {'Cte':<6} {'Taille':<12} {'P&L Non Realise':<15}\n"
            dashboard += "-"*80 + "\n"
            for pos in positions[:10]:  # Max 10 positions affichees
                dashboard += f"{pos.get('symbol', 'N/A'):<12} "
                dashboard += f"{pos.get('strategy', 'N/A'):<15} "
                dashboard += f"{pos.get('side', 'N/A'):<6} "
                dashboard += f"${pos.get('size_usdc', 0):<11,.2f} "
                dashboard += f"${pos.get('unrealized_pnl', 0):>14,.2f}\n"
        else:
            dashboard += "Aucune position ouverte\n"
        dashboard += "\n"
        
        # Section Performance
        dashboard += "" PERFORMANCE\n"
        dashboard += "-"*80 + "\n"
        dashboard += f"Total Trades:       {performance.get('total_trades', 0):>15}\n"
        dashboard += f"Win Rate:           {performance.get('win_rate', 0):>14.1%}\n"
        dashboard += f"Profit Factor:      {performance.get('profit_factor', 0):>15.2f}\n"
        dashboard += f"Sharpe Ratio:       {performance.get('sharpe_ratio', 0):>15.2f}\n"
        dashboard += f"Trades/Heure:       {performance.get('trades_per_hour', 0):>15.1f}\n"
        dashboard += "\n"
        
        # Section Risque
        dashboard += "  RISQUE\n"
        dashboard += "-"*80 + "\n"
        dashboard += f"Risk Level:         {risk_metrics.get('risk_level', 'N/A'):>15}\n"
        dashboard += f"Drawdown Actuel:    {risk_metrics.get('current_drawdown', 0):>14.2%}\n"
        dashboard += f"Max Drawdown:       {risk_metrics.get('max_drawdown', 0):>14.2%}\n"
        dashboard += f"Exposition Totale:  {risk_metrics.get('total_exposure', 0):>14.2%}\n"
        dashboard += f"VaR (95%):          ${risk_metrics.get('var_95', 0):>14,.2f}\n"
        dashboard += "\n"
        
        # Section Systeme
        dashboard += """  SYSTME\n"
        dashboard += "-"*80 + "\n"
        dashboard += f"Statut:             {system_health.get('status', 'N/A'):>15}\n"
        dashboard += f"Uptime:             {system_health.get('uptime', 'N/A'):>15}\n"
        dashboard += f"CPU:                {system_health.get('cpu_percent', 0):>14.1f}%\n"
        dashboard += f"Memoire:            {system_health.get('memory_percent', 0):>14.1f}%\n"
        dashboard += f"Threads Actifs:     {system_health.get('active_threads', 0):>15}\n"
        
        dashboard += "\n" + "="*80 + "\n"
        
        return dashboard
    
    def generate_daily_summary(self,
    """
    date: datetime,
    """
                              performance: Dict,
                              trades: List[Dict],
                              top_performers: Dict,
                              alerts: List[Dict]) -> str:
        """
        Genere un resume quotidien
        
        Args:
            date: Date du resume
            performance: Metriques de performance
            trades: Liste des trades de la journee
            top_performers: Meilleures/pires performances
            alerts: Alertes de la journee
            
        Returns:
            String formate
        """
        summary = "\n" + "="*80 + "\n"
        summary += f"{'" R"SUM" QUOTIDIEN':^80}\n"
        summary += f"{date.strftime('%Y-%m-%d'):^80}\n"
        summary += "="*80 + "\n\n"
        
        # Performance du jour
        summary += "' PERFORMANCE DU JOUR\n"
        summary += "-"*80 + "\n"
        summary += f"P&L Journalier:     ${performance.get('daily_pnl', 0):>15,.2f}\n"
        summary += f"Return:             {performance.get('daily_return_pct', 0):>14.2f}%\n"
        summary += f"Trades Executes:    {len(trades):>15}\n"
        summary += f"Win Rate:           {performance.get('win_rate', 0):>14.1%}\n"
        summary += f"Profit Factor:      {performance.get('profit_factor', 0):>15.2f}\n"
        summary += "\n"
        
        # Top/Flop
        summary += "" TOP & FLOP\n"
        summary += "-"*80 + "\n"
        
        if 'best_trades' in top_performers:
            summary += "Meilleurs Trades:\n"
            for i, trade in enumerate(top_performers['best_trades'][:3], 1):
                summary += f"  {i}. {trade.get('symbol', 'N/A')}: ${trade.get('profit', 0):+.2f}\n"
        
        summary += "\n"
        
        if 'worst_trades' in top_performers:
            summary += "Pires Trades:\n"
            for i, trade in enumerate(top_performers['worst_trades'][:3], 1):
                summary += f"  {i}. {trade.get('symbol', 'N/A')}: ${trade.get('profit', 0):+.2f}\n"
        
        summary += "\n"
        
        # Par strategie
        if 'by_strategy' in performance:
            summary += "" PAR STRAT"GIE\n"
            summary += "-"*80 + "\n"
            for strategy, perf in performance['by_strategy'].items():
                summary += f"{strategy:.<20} "
                summary += f"Trades: {perf.get('trades', 0):>3} | "
                summary += f"Win Rate: {perf.get('win_rate', 0):>5.1%} | "
                summary += f"P&L: ${perf.get('total_pnl', 0):>+8.2f}\n"
            summary += "\n"
        
        # Alertes importantes
        if alerts:
            summary += "  ALERTES\n"
            summary += "-"*80 + "\n"
            for alert in alerts[:5]:  # Max 5 alertes
                summary += f"[{alert.get('timestamp', 'N/A')}] "
                summary += f"{alert.get('level', 'INFO')}: {alert.get('message', 'N/A')}\n"
            summary += "\n"
        
        summary += "="*80 + "\n"
        
        return summary
    
    def generate_weekly_report(self,
                              start_date: datetime,
                              end_date: datetime,
                              performance: Dict,
                              statistics: Dict) -> str:
        """
        Genere un rapport hebdomadaire
        
        Args:
            start_date: Debut de la semaine
            end_date: Fin de la semaine
            performance: Metriques de performance
            statistics: Statistiques detaillees
            
        Returns:
            String formate
        """
        report = "\n" + "="*80 + "\n"
        report += f"{'""| RAPPORT HEBDOMADAIRE':^80}\n"
        report += f"{start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d'):^80}\n"
        report += "="*80 + "\n\n"
        
        # Resume de la semaine
        report += "" R"SUM" HEBDOMADAIRE\n"
        report += "-"*80 + "\n"
        report += f"P&L Hebdomadaire:   ${performance.get('weekly_pnl', 0):>15,.2f}\n"
        report += f"Return:             {performance.get('weekly_return_pct', 0):>14.2f}%\n"
        report += f"Total Trades:       {performance.get('total_trades', 0):>15}\n"
        report += f"Trades/Jour:        {performance.get('avg_trades_per_day', 0):>15.1f}\n"
        report += f"Win Rate:           {performance.get('win_rate', 0):>14.1%}\n"
        report += f"Profit Factor:      {performance.get('profit_factor', 0):>15.2f}\n"
        report += f"Sharpe Ratio:       {performance.get('sharpe_ratio', 0):>15.2f}\n"
        report += f"Max Drawdown:       {performance.get('max_drawdown', 0):>14.2%}\n"
        report += "\n"
        
        # Statistiques avancees
        if statistics:
            report += "" STATISTIQUES AVANC"ES\n"
            report += "-"*80 + "\n"
            report += f"Avg Win:            ${statistics.get('avg_win', 0):>15,.2f}\n"
            report += f"Avg Loss:           ${statistics.get('avg_loss', 0):>15,.2f}\n"
            report += f"Largest Win:        ${statistics.get('largest_win', 0):>15,.2f}\n"
            report += f"Largest Loss:       ${statistics.get('largest_loss', 0):>15,.2f}\n"
            report += f"Win/Loss Ratio:     {statistics.get('win_loss_ratio', 0):>15.2f}\n"
            report += f"Recovery Factor:    {statistics.get('recovery_factor', 0):>15.2f}\n"
            report += "\n"
        
        # Performance journaliere
        if 'daily_breakdown' in performance:
            report += """| D"TAIL JOURNALIER\n"
            report += "-"*80 + "\n"
            report += f"{'Date':<12} {'P&L':<12} {'Trades':<8} {'Win Rate':<10}\n"
            report += "-"*80 + "\n"
            for day_data in performance['daily_breakdown']:
                report += f"{day_data.get('date', 'N/A'):<12} "
                report += f"${day_data.get('pnl', 0):<11,.2f} "
                report += f"{day_data.get('trades', 0):<8} "
                report += f"{day_data.get('win_rate', 0):<9.1%}\n"
            report += "\n"
        
        report += "="*80 + "\n"
        
        return report
    
    def generate_strategy_comparison(self, strategies_performance: Dict) -> str:
        """
        Genere un rapport comparatif des strategies
        
        Args:
            strategies_performance: Performance par strategie
            
        Returns:
            String formate
        """
        report = "\n" + "="*80 + "\n"
        report += f"{' COMPARAISON DES STRAT"GIES':^80}\n"
        report += "="*80 + "\n\n"
        
        if not strategies_performance:
            report += "Aucune donnee disponible\n"
            return report
        
        # Tableau comparatif
        report += f"{'Strategie':<15} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Avg P&L':<12} {'Sharpe':<8}\n"
        report += "-"*80 + "\n"
        
        for strategy, perf in sorted(strategies_performance.items(), 
                                     key=lambda x: x[1].get('total_pnl', 0), 
                                     reverse=True):
            report += f"{strategy:<15} "
            report += f"{perf.get('trades', 0):<8} "
            report += f"{perf.get('win_rate', 0):<7.1%} "
            report += f"${perf.get('total_pnl', 0):<11,.2f} "
            report += f"${perf.get('avg_pnl', 0):<11,.2f} "
            report += f"{perf.get('sharpe_ratio', 0):<8.2f}\n"
        
        report += "\n"
        
        # Analyse detaillee par strategie
        for strategy, perf in strategies_performance.items():
            report += f"\n{'"'*80}\n"
            report += f""' {strategy.upper()}\n"
            report += f"{'"'*80}\n"
            report += f"Total Trades:       {perf.get('trades', 0):>15}\n"
            report += f"Winning Trades:     {perf.get('wins', 0):>15}\n"
            report += f"Losing Trades:      {perf.get('losses', 0):>15}\n"
            report += f"Win Rate:           {perf.get('win_rate', 0):>14.1%}\n"
            report += f"Total P&L:          ${perf.get('total_pnl', 0):>14,.2f}\n"
            report += f"Avg P&L/Trade:      ${perf.get('avg_pnl', 0):>14,.2f}\n"
            report += f"Best Trade:         ${perf.get('best_trade', 0):>14,.2f}\n"
            report += f"Worst Trade:        ${perf.get('worst_trade', 0):>14,.2f}\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def generate_risk_report(self, risk_data: Dict) -> str:
        """
        Genere un rapport de risque detaille
        
        Args:
            risk_data: Donnees de risque
            
        Returns:
            String formate
        """
        report = "\n" + "="*80 + "\n"
        report += f"{'  RAPPORT DE RISQUE':^80}\n"
        report += f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}\n"
        report += "="*80 + "\n\n"
        
        # Niveau de risque actuel
        report += "| NIVEAU DE RISQUE\n"
        report += "-"*80 + "\n"
        risk_level = risk_data.get('risk_level', 'UNKNOWN')
        risk_emoji = {
            'NORMAL': '',
            'ELEVATED': '',
            'HIGH': '',
            'CRITICAL': '"',
            'EMERGENCY': ''
        }
        report += f"Status: {risk_emoji.get(risk_level, '"')} {risk_level}\n"
        report += "\n"
        
        # Metriques de risque
        report += "" M"TRIQUES DE RISQUE\n"
        report += "-"*80 + "\n"
        report += f"Drawdown Actuel:    {risk_data.get('current_drawdown', 0):>14.2%}\n"
        report += f"Max Drawdown:       {risk_data.get('max_drawdown', 0):>14.2%}\n"
        report += f"VaR 95%:            ${risk_data.get('var_95', 0):>14,.2f}\n"
        report += f"VaR 99%:            ${risk_data.get('var_99', 0):>14,.2f}\n"
        report += f"Exposition Totale:  {risk_data.get('total_exposure', 0):>14.2%}\n"
        report += f"Concentration Max:  {risk_data.get('max_concentration', 0):>14.2%}\n"
        report += f"Volatilite:         {risk_data.get('portfolio_volatility', 0):>14.2%}\n"
        report += "\n"
        
        # Positions  risque
        if 'risky_positions' in risk_data and risk_data['risky_positions']:
            report += "  POSITIONS  RISQUE\n"
            report += "-"*80 + "\n"
            report += f"{'Symbole':<12} {'Strategie':<15} {'Drawdown':<10} {'Taille':<12}\n"
            report += "-"*80 + "\n"
            for pos in risk_data['risky_positions'][:10]:
                report += f"{pos.get('symbol', 'N/A'):<12} "
                report += f"{pos.get('strategy', 'N/A'):<15} "
                report += f"{pos.get('drawdown', 0):<9.2%} "
                report += f"${pos.get('size_usdc', 0):<11,.2f}\n"
            report += "\n"
        
        # Actions recommandees
        if 'required_actions' in risk_data and risk_data['required_actions']:
            report += "" ACTIONS RECOMMAND"ES\n"
            report += "-"*80 + "\n"
            for action in risk_data['required_actions']:
                report += f"" {action}\n"
            report += "\n"
        
        # Historique des alertes
        if 'recent_alerts' in risk_data and risk_data['recent_alerts']:
            report += "" ALERTES R"CENTES\n"
            report += "-"*80 + "\n"
            for alert in risk_data['recent_alerts'][-10:]:
                timestamp = alert.get('timestamp', 'N/A')
                level = alert.get('level', 'INFO')
                message = alert.get('message', 'N/A')
                report += f"[{timestamp}] {level}: {message}\n"
            report += "\n"
        
        report += "="*80 + "\n"
        
        return report
    
    def generate_error_report(self, errors: List[Dict]) -> str:
        """
        Genere un rapport d'erreurs
        
        Args:
            errors: Liste des erreurs
            
        Returns:
            String formate
        """
        report = "\n" + "="*80 + "\n"
        report += f"{'" RAPPORT D\'ERREURS':^80}\n"
        report += f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}\n"
        report += "="*80 + "\n\n"
        
        if not errors:
            report += """| Aucune erreur detectee\n"
            report += "="*80 + "\n"
            return report
        
        # Resume
        report += f"Total Erreurs: {len(errors)}\n\n"
        
        # Erreurs par type
        error_types = {}
        for error in errors:
            error_type = error.get('type', 'UNKNOWN')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        report += "" ERREURS PAR TYPE\n"
        report += "-"*80 + "\n"
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            report += f"{error_type:<30} {count:>5}\n"
        report += "\n"
        
        # Dernieres erreurs
        report += "" DERNIRES ERREURS\n"
        report += "-"*80 + "\n"
        for error in errors[-20:]:  # 20 dernieres erreurs
            timestamp = error.get('timestamp', 'N/A')
            error_type = error.get('type', 'UNKNOWN')
            message = error.get('message', 'N/A')
            component = error.get('component', 'N/A')
            
            report += f"[{timestamp}] {error_type}\n"
            report += f"  Component: {component}\n"
            report += f"  Message: {message}\n"
            if 'traceback' in error:
                report += f"  Traceback: {error['traceback'][:200]}...\n"
            report += "\n"
        
        report += "="*80 + "\n"
        
        return report
    
    def save_report_to_file(self, report: str, filename: str):
        """
        Sauvegarde un rapport dans un fichier
        
        Args:
            report: Contenu du rapport
            filename: Nom du fichier
        """
        try:
            filepath = self.reports_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f""" Rapport sauvegarde: {filepath}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport {filename}: {e}")
    
    def export_json_report(self, data: Dict, filename: str):
        """
        Exporte un rapport en JSON
        
        Args:
            data: Donnees  exporter
            filename: Nom du fichier
        """
        try:
            filepath = self.reports_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f""" Rapport JSON exporte: {filepath}")
        except Exception as e:
            logger.error(f"Erreur export JSON {filename}: {e}")
    
    def generate_and_save_daily_report(self,
    """
    date: datetime,
    """
                                       performance: Dict,
                                       trades: List[Dict],
                                       top_performers: Dict,
                                       alerts: List[Dict]):
        """
        Genere et sauvegarde le rapport quotidien
        
        Args:
            date: Date du rapport
            performance: Metriques de performance
            trades: Liste des trades
            top_performers: Top performers
            alerts: Alertes
        """
        # Generer le rapport texte
        report = self.generate_daily_summary(date, performance, trades, top_performers, alerts)
        
        # Sauvegarder
        filename = f"daily_report_{date.strftime('%Y%m%d')}.txt"
        self.save_report_to_file(report, filename)
        
        # Exporter aussi en JSON
        json_data = {
            'date': date.isoformat(),
            'performance': performance,
            'trades_count': len(trades),
            'top_performers': top_performers,
            'alerts_count': len(alerts)
        }
        json_filename = f"daily_report_{date.strftime('%Y%m%d')}.json"
        self.export_json_report(json_data, json_filename)
    
    def generate_and_save_weekly_report(self,
                                        start_date: datetime,
                                        end_date: datetime,
                                        performance: Dict,
                                        statistics: Dict):
        """
        Genere et sauvegarde le rapport hebdomadaire
        
        Args:
            start_date: Debut de la semaine
            end_date: Fin de la semaine
            performance: Metriques de performance
            statistics: Statistiques
        """
        # Generer le rapport texte
        report = self.generate_weekly_report(start_date, end_date, performance, statistics)
        
        # Sauvegarder
        filename = f"weekly_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.txt"
        self.save_report_to_file(report, filename)
        
        # Exporter aussi en JSON
        json_data = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'performance': performance,
            'statistics': statistics
        }
        json_filename = f"weekly_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        self.export_json_report(json_data, json_filename)
    
    def cleanup_old_reports(self, days: int = 30):
        """
        Nettoie les anciens rapports
        
        Args:
            days: Nombre de jours  conserver
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted = 0
            
            for filepath in self.reports_dir.glob('*'):
                if filepath.is_file():
                    # Verifier la date de modification
                    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                    if mtime < cutoff_date:
                        filepath.unlink()
                        deleted += 1
            
            if deleted > 0:
                logger.info(f" Nettoyage: {deleted} anciens rapports supprimes")
        
        except Exception as e:
            logger.error(f"Erreur nettoyage rapports: {e}")
    
    def get_report_summary(self) -> Dict:
        """
        Retourne un resume des rapports disponibles
        
        Returns:
            Dict avec les infos sur les rapports
        """
        try:
            reports = {
                'daily': [],
                'weekly': [],
                'other': []
            }
            
            for filepath in self.reports_dir.glob('*.txt'):
                filename = filepath.name
                size = filepath.stat().st_size
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                
                report_info = {
                    'filename': filename,
                    'size_kb': size / 1024,
                    'modified': mtime.isoformat()
                }
                
                if filename.startswith('daily_report'):
                    reports['daily'].append(report_info)
                elif filename.startswith('weekly_report'):
                    reports['weekly'].append(report_info)
                else:
                    reports['other'].append(report_info)
            
            # Trier par date
            for category in reports:
                reports[category].sort(key=lambda x: x['modified'], reverse=True)
            
            return {
                'total_reports': sum(len(r) for r in reports.values()),
                'reports_dir': str(self.reports_dir),
                'by_type': reports
            }
        
        except Exception as e:
            logger.error(f"Erreur recuperation resume rapports: {e}")
            return {'error': str(e)}


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    """Test du report generator"""
    
    # Configuration de test
    config = {
        'reports_dir': 'data/reports_test'
    }
    
    generator = ReportGenerator(config)
    
    print("\n=== Test Report Generator ===\n")
    
    # Test Live Dashboard
    print("--- Live Dashboard ---")
    dashboard = generator.generate_live_dashboard(
        capital=1050.0,
        initial_capital=1000.0,
        positions=[
            {'symbol': 'BTCUSDT', 'strategy': 'scalping', 'side': 'BUY', 'size_usdc': 250.0, 'unrealized_pnl': 10.0},
            {'symbol': 'ETHUSDT', 'strategy': 'momentum', 'side': 'SELL', 'size_usdc': 150.0, 'unrealized_pnl': -5.0}
        ],
        performance={
            'daily_pnl': 25.0,
            'total_trades': 15,
            'win_rate': 0.67,
            'profit_factor': 2.1,
            'sharpe_ratio': 1.8,
            'trades_per_hour': 6.2
        },
        risk_metrics={
            'risk_level': 'NORMAL',
            'current_drawdown': 0.02,
            'max_drawdown': 0.05,
            'total_exposure': 0.4,
            'var_95': -20.0
        },
        system_health={
            'status': 'RUNNING',
            'uptime': '5h 23m',
            'cpu_percent': 45.2,
            'memory_percent': 62.1,
            'active_threads': 4
        }
    )
    print(dashboard)
    
    # Test Daily Summary
    print("\n--- Daily Summary ---")
    summary = generator.generate_daily_summary(
        date=datetime.now(),
        performance={
            'daily_pnl': 50.0,
            'daily_return_pct': 5.0,
            'win_rate': 0.70,
            'profit_factor': 2.5,
            'by_strategy': {
                'scalping': {'trades': 10, 'win_rate': 0.8, 'total_pnl': 35.0},
                'momentum': {'trades': 5, 'win_rate': 0.6, 'total_pnl': 15.0}
            }
        },
        trades=[],
        top_performers={
            'best_trades': [
                {'symbol': 'BTCUSDT', 'profit': 25.0},
                {'symbol': 'ETHUSDT', 'profit': 15.0}
            ],
            'worst_trades': [
                {'symbol': 'BNBUSDT', 'profit': -8.0}
            ]
        },
        alerts=[
            {'timestamp': '14:23:45', 'level': 'WARNING', 'message': 'Drawdown eleve detecte'}
        ]
    )
    print(summary)
    
    # Test Strategy Comparison
    print("\n--- Strategy Comparison ---")
    comparison = generator.generate_strategy_comparison({
        'scalping': {
            'trades': 100,
            'wins': 70,
            'losses': 30,
            'win_rate': 0.70,
            'total_pnl': 350.0,
            'avg_pnl': 3.5,
            'best_trade': 25.0,
            'worst_trade': -10.0,
            'sharpe_ratio': 2.1
        },
        'momentum': {
            'trades': 50,
            'wins': 32,
            'losses': 18,
            'win_rate': 0.64,
            'total_pnl': 200.0,
            'avg_pnl': 4.0,
            'best_trade': 30.0,
            'worst_trade': -15.0,
            'sharpe_ratio': 1.8
        }
    })
    print(comparison)
    
    # Test sauvegarde
    generator.save_report_to_file(dashboard, 'test_dashboard.txt')
    
    # Resume des rapports
    summary = generator.get_report_summary()
    print(f"\n--- Resume des rapports ---")
    print(f"Total rapports: {summary['total_reports']}")
    print(f"Repertoire: {summary['reports_dir']}")
    
    print("\n""| Tests termines")
