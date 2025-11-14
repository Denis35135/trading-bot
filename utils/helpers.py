"""
Helpers pour The Bot
Fonctions utilitaires et helpers pour diverses tÃƒÂ¢ches
"""

import os
import json
import logging
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# FORMATAGE
# ============================================================================

def format_price(price: float, decimals: int = 2) -> str:
    """
    Formate un prix pour affichage
    
    Args:
        price: Prix ÃƒÂ  formater
        decimals: Nombre de dÃƒÂ©cimales
        
    Returns:
        Prix formatÃƒÂ©
    """
    return f"${price:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2, show_sign: bool = True) -> str:
    """
    Formate un pourcentage
    
    Args:
        value: Valeur (0.05 = 5%)
        decimals: Nombre de dÃƒÂ©cimales
        show_sign: Afficher le signe +/-
        
    Returns:
        Pourcentage formatÃƒÂ©
    """
    sign = "+" if value > 0 and show_sign else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def format_duration(seconds: int) -> str:
    """
    Formate une durÃƒÂ©e en format lisible
    
    Args:
        seconds: DurÃƒÂ©e en secondes
        
    Returns:
        DurÃƒÂ©e formatÃƒÂ©e (ex: "2h 15m 30s")
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_timestamp(timestamp: Union[int, float, datetime], 
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formate un timestamp
    
    Args:
        timestamp: Timestamp (unix ou datetime)
        format_str: Format de sortie
        
    Returns:
        Timestamp formatÃƒÂ©
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
    else:
        dt = timestamp
    
    return dt.strftime(format_str)


def format_number_short(number: float) -> str:
    """
    Formate un nombre en format court (K, M, B)
    
    Args:
        number: Nombre ÃƒÂ  formater
        
    Returns:
        Nombre formatÃƒÂ© (ex: "1.5M")
    """
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return f"{number:.2f}"


# ============================================================================
# VALIDATION
# ============================================================================

def is_valid_symbol(symbol: str) -> bool:
    """
    VÃƒÂ©rifie si un symbole est valide
    
    Args:
        symbol: Symbole ÃƒÂ  vÃƒÂ©rifier (ex: BTCUSDT)
        
    Returns:
        True si valide
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Doit contenir au moins 6 caractÃƒÂ¨res
    if len(symbol) < 6:
        return False
    
    # Doit ÃƒÂªtre en majuscules
    if symbol != symbol.upper():
        return False
    
    # Doit se terminer par USDT, USDC ou BUSD
    valid_quotes = ['USDT', 'USDC', 'BUSD']
    if not any(symbol.endswith(quote) for quote in valid_quotes):
        return False
    
    return True


def is_valid_price(price: float, min_price: float = 0) -> bool:
    """
    VÃƒÂ©rifie si un prix est valide
    
    Args:
        price: Prix ÃƒÂ  vÃƒÂ©rifier
        min_price: Prix minimum acceptÃƒÂ©
        
    Returns:
        True si valide
    """
    if not isinstance(price, (int, float)):
        return False
    
    if price <= min_price:
        return False
    
    if not np.isfinite(price):
        return False
    
    return True


def is_valid_quantity(quantity: float, min_qty: float = 0) -> bool:
    """
    VÃƒÂ©rifie si une quantitÃƒÂ© est valide
    
    Args:
        quantity: QuantitÃƒÂ© ÃƒÂ  vÃƒÂ©rifier
        min_qty: QuantitÃƒÂ© minimum
        
    Returns:
        True si valide
    """
    if not isinstance(quantity, (int, float)):
        return False
    
    if quantity <= min_qty:
        return False
    
    if not np.isfinite(quantity):
        return False
    
    return True


def validate_order_params(symbol: str, 
                         side: str, 
                         quantity: float, 
                         price: Optional[float] = None) -> Tuple[bool, str]:
    """
    Valide les paramÃƒÂ¨tres d'un ordre
    
    Args:
        symbol: Symbole
        side: CÃƒÂ´tÃƒÂ© (BUY/SELL)
        quantity: QuantitÃƒÂ©
        price: Prix (optionnel pour market orders)
        
    Returns:
        Tuple (valide, message_erreur)
    """
    # Symbole
    if not is_valid_symbol(symbol):
        return False, f"Symbole invalide: {symbol}"
    
    # Side
    if side not in ['BUY', 'SELL']:
        return False, f"Side invalide: {side}"
    
    # QuantitÃƒÂ©
    if not is_valid_quantity(quantity):
        return False, f"QuantitÃƒÂ© invalide: {quantity}"
    
    # Prix (si fourni)
    if price is not None and not is_valid_price(price):
        return False, f"Prix invalide: {price}"
    
    return True, "OK"


# ============================================================================
# CALCULS FINANCIERS
# ============================================================================

def calculate_profit(entry_price: float, 
                    exit_price: float, 
                    quantity: float,
                    side: str = 'BUY',
                    fees: float = 0.001) -> Dict[str, float]:
    """
    Calcule le profit d'un trade
    
    Args:
        entry_price: Prix d'entrÃƒÂ©e
        exit_price: Prix de sortie
        quantity: QuantitÃƒÂ©
        side: BUY (long) ou SELL (short)
        fees: Frais (0.001 = 0.1%)
        
    Returns:
        Dict avec profit_usdc, profit_pct, fees_usdc
    """
    if side == 'BUY':
        # Long position
        gross_profit_pct = (exit_price - entry_price) / entry_price
    else:
        # Short position
        gross_profit_pct = (entry_price - exit_price) / entry_price
    
    # Valeur notionnelle
    notional_value = entry_price * quantity
    
    # Frais (entrÃƒÂ©e + sortie)
    entry_fee = notional_value * fees
    exit_fee = notional_value * fees
    total_fees = entry_fee + exit_fee
    
    # Profit brut
    gross_profit_usdc = notional_value * gross_profit_pct
    
    # Profit net
    net_profit_usdc = gross_profit_usdc - total_fees
    net_profit_pct = net_profit_usdc / notional_value
    
    return {
        'profit_usdc': net_profit_usdc,
        'profit_pct': net_profit_pct,
        'fees_usdc': total_fees,
        'gross_profit_usdc': gross_profit_usdc,
        'gross_profit_pct': gross_profit_pct
    }


def calculate_position_size(capital: float,
                           risk_percent: float,
                           entry_price: float,
                           stop_loss_price: float) -> Dict[str, float]:
    """
    Calcule la taille de position basÃƒÂ©e sur le risque
    
    Args:
        capital: Capital disponible
        risk_percent: Risque en % (0.02 = 2%)
        entry_price: Prix d'entrÃƒÂ©e
        stop_loss_price: Prix du stop loss
        
    Returns:
        Dict avec quantity, notional_value, risk_usdc
    """
    # Montant ÃƒÂ  risquer
    risk_usdc = capital * risk_percent
    
    # Distance au stop loss en %
    sl_distance_pct = abs(entry_price - stop_loss_price) / entry_price
    
    # Valeur notionnelle maximale
    notional_value = risk_usdc / sl_distance_pct
    
    # QuantitÃƒÂ©
    quantity = notional_value / entry_price
    
    return {
        'quantity': quantity,
        'notional_value': notional_value,
        'risk_usdc': risk_usdc,
        'sl_distance_pct': sl_distance_pct
    }


def calculate_roi(initial_capital: float, 
                 current_capital: float,
                 days: Optional[int] = None) -> Dict[str, float]:
    """
    Calcule le ROI
    
    Args:
        initial_capital: Capital initial
        current_capital: Capital actuel
        days: Nombre de jours (pour ROI annualisÃƒÂ©)
        
    Returns:
        Dict avec roi, roi_pct, annualized_roi (si days fourni)
    """
    roi = current_capital - initial_capital
    roi_pct = roi / initial_capital if initial_capital > 0 else 0
    
    result = {
        'roi': roi,
        'roi_pct': roi_pct
    }
    
    if days and days > 0:
        # ROI annualisÃƒÂ©
        annualized_roi = (1 + roi_pct) ** (365 / days) - 1
        result['annualized_roi'] = annualized_roi
    
    return result


def calculate_sharpe_ratio(returns: List[float], 
                          risk_free_rate: float = 0.02) -> float:
    """
    Calcule le ratio de Sharpe
    
    Args:
        returns: Liste des retours (en %)
        risk_free_rate: Taux sans risque annuel (0.02 = 2%)
        
    Returns:
        Ratio de Sharpe
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Moyenne et ÃƒÂ©cart-type
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    
    if std_return == 0:
        return 0.0
    
    # Sharpe annualisÃƒÂ©
    sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
    
    return sharpe


def calculate_max_drawdown(capital_history: List[float]) -> Dict[str, Any]:
    """
    Calcule le drawdown maximum
    
    Args:
        capital_history: Historique du capital
        
    Returns:
        Dict avec max_dd, max_dd_pct, dd_duration
    """
    if not capital_history or len(capital_history) < 2:
        return {'max_dd': 0, 'max_dd_pct': 0, 'dd_duration': 0}
    
    capital_array = np.array(capital_history)
    
    # Calculer les peaks cumulatifs
    running_max = np.maximum.accumulate(capital_array)
    
    # Drawdowns
    drawdowns = capital_array - running_max
    drawdown_pcts = drawdowns / running_max
    
    # Max drawdown
    max_dd_idx = np.argmin(drawdowns)
    max_dd = drawdowns[max_dd_idx]
    max_dd_pct = drawdown_pcts[max_dd_idx]
    
    # DurÃƒÂ©e du drawdown
    # Trouver le peak avant le max dd
    peak_idx = np.argmax(running_max[:max_dd_idx + 1])
    dd_duration = max_dd_idx - peak_idx
    
    return {
        'max_dd': max_dd,
        'max_dd_pct': abs(max_dd_pct),
        'dd_duration': dd_duration,
        'peak_idx': peak_idx,
        'trough_idx': max_dd_idx
    }


# ============================================================================
# CONVERSIONS
# ============================================================================

def round_step_size(quantity: float, step_size: float) -> float:
    """
    Arrondit une quantitÃƒÂ© selon le step size de Binance
    
    Args:
        quantity: QuantitÃƒÂ© ÃƒÂ  arrondir
        step_size: Step size (ex: 0.001)
        
    Returns:
        QuantitÃƒÂ© arrondie
    """
    precision = int(round(-np.log10(step_size)))
    return round(quantity, precision)


def round_price(price: float, tick_size: float) -> float:
    """
    Arrondit un prix selon le tick size de Binance
    
    Args:
        price: Prix ÃƒÂ  arrondir
        tick_size: Tick size (ex: 0.01)
        
    Returns:
        Prix arrondi
    """
    precision = int(round(-np.log10(tick_size)))
    return round(price, precision)


def convert_timeframe_to_seconds(timeframe: str) -> int:
    """
    Convertit un timeframe en secondes
    
    Args:
        timeframe: Timeframe (1m, 5m, 1h, etc.)
        
    Returns:
        Secondes
    """
    conversions = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '12h': 43200,
        '1d': 86400,
        '1w': 604800
    }
    
    return conversions.get(timeframe, 300)  # DÃƒÂ©faut: 5m


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convertit un timestamp en datetime
    
    Args:
        timestamp: Timestamp unix (en ms ou secondes)
        
    Returns:
        Datetime
    """
    # Si en millisecondes
    if timestamp > 1e10:
        timestamp = timestamp / 1000
    
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convertit un datetime en timestamp
    
    Args:
        dt: Datetime
        
    Returns:
        Timestamp unix en millisecondes
    """
    return int(dt.timestamp() * 1000)


# ============================================================================
# ANALYSE TECHNIQUE HELPERS
# ============================================================================

def detect_trend(prices: np.ndarray, window: int = 20) -> str:
    """
    DÃƒÂ©tecte la tendance
    
    Args:
        prices: Array des prix
        window: FenÃƒÂªtre d'analyse
        
    Returns:
        'uptrend', 'downtrend' ou 'sideways'
    """
    if len(prices) < window:
        return 'unknown'
    
    recent = prices[-window:]
    
    # RÃƒÂ©gression linÃƒÂ©aire simple
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    
    # Normaliser par le prix moyen
    slope_pct = slope / np.mean(recent)
    
    if slope_pct > 0.002:  # +0.2% par pÃƒÂ©riode
        return 'uptrend'
    elif slope_pct < -0.002:
        return 'downtrend'
    else:
        return 'sideways'


def calculate_volatility(prices: np.ndarray, window: int = 20) -> float:
    """
    Calcule la volatilitÃƒÂ©
    
    Args:
        prices: Array des prix
        window: FenÃƒÂªtre
        
    Returns:
        VolatilitÃƒÂ© (ÃƒÂ©cart-type des returns)
    """
    if len(prices) < window + 1:
        return 0.0
    
    recent = prices[-window-1:]
    returns = np.diff(recent) / recent[:-1]
    
    return np.std(returns)


def calculate_spread(bid: float, ask: float) -> float:
    """
    Calcule le spread en %
    
    Args:
        bid: Prix bid
        ask: Prix ask
        
    Returns:
        Spread en %
    """
    if bid <= 0:
        return 0.0
    
    return (ask - bid) / bid


def is_near_support_resistance(price: float, 
                               levels: List[float], 
                               threshold: float = 0.005) -> bool:
    """
    VÃƒÂ©rifie si le prix est proche d'un niveau de S/R
    
    Args:
        price: Prix actuel
        levels: Liste des niveaux S/R
        threshold: Seuil de proximitÃƒÂ© (0.005 = 0.5%)
        
    Returns:
        True si proche d'un niveau
    """
    for level in levels:
        distance_pct = abs(price - level) / price
        if distance_pct < threshold:
            return True
    
    return False


# ============================================================================
# UTILITAIRES FICHIERS
# ============================================================================

def ensure_dir_exists(path: str):
    """
    CrÃƒÂ©e un rÃƒÂ©pertoire s'il n'existe pas
    
    Args:
        path: Chemin du rÃƒÂ©pertoire
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, filepath: str, indent: int = 2):
    """
    Sauvegarde des donnÃƒÂ©es en JSON
    
    Args:
        data: DonnÃƒÂ©es ÃƒÂ  sauvegarder
        filepath: Chemin du fichier
        indent: Indentation
    """
    ensure_dir_exists(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """
    Charge des donnÃƒÂ©es JSON
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        DonnÃƒÂ©es chargÃƒÂ©es
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_order_id(prefix: str = "ORDER") -> str:
    """
    GÃƒÂ©nÃƒÂ¨re un ID d'ordre unique
    
    Args:
        prefix: PrÃƒÂ©fixe de l'ID
        
    Returns:
        ID unique
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{random_part}"


def hash_string(text: str) -> str:
    """
    CrÃƒÂ©e un hash d'une chaÃƒÂ®ne
    
    Args:
        text: Texte ÃƒÂ  hasher
        
    Returns:
        Hash MD5
    """
    return hashlib.md5(text.encode()).hexdigest()


# ============================================================================
# STATISTIQUES
# ============================================================================

def calculate_trade_stats(trades: List[Dict]) -> Dict[str, Any]:
    """
    Calcule les statistiques d'une liste de trades
    
    Args:
        trades: Liste des trades
        
    Returns:
        Dict avec statistiques
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'profit_factor': 0
        }
    
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.get('profit_usdc', 0) > 0]
    losing_trades = [t for t in trades if t.get('profit_usdc', 0) < 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    total_profit = sum(t.get('profit_usdc', 0) for t in winning_trades)
    total_loss = abs(sum(t.get('profit_usdc', 0) for t in losing_trades))
    
    stats = {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': win_count / total_trades if total_trades > 0 else 0,
        'avg_profit': total_profit / win_count if win_count > 0 else 0,
        'avg_loss': total_loss / loss_count if loss_count > 0 else 0,
        'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
        'total_pnl': sum(t.get('profit_usdc', 0) for t in trades)
    }
    
    return stats


# ============================================================================
# SÃƒâ€°CURITÃƒâ€°
# ============================================================================

def sanitize_symbol(symbol: str) -> str:
    """
    Nettoie un symbole
    
    Args:
        symbol: Symbole ÃƒÂ  nettoyer
        
    Returns:
        Symbole nettoyÃƒÂ©
    """
    # Enlever espaces et mettre en majuscules
    symbol = symbol.strip().upper()
    
    # Enlever caractÃƒÂ¨res spÃƒÂ©ciaux
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    symbol = ''.join(c for c in symbol if c in allowed)
    
    return symbol


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Limite une valeur entre min et max
    
    Args:
        value: Valeur
        min_value: Minimum
        max_value: Maximum
        
    Returns:
        Valeur clampÃƒÂ©e
    """
    return max(min_value, min(value, max_value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division sÃƒÂ©curisÃƒÂ©e (ÃƒÂ©vite division par zÃƒÂ©ro)
    
    Args:
        numerator: NumÃƒÂ©rateur
        denominator: DÃƒÂ©nominateur
        default: Valeur par dÃƒÂ©faut si division impossible
        
    Returns:
        RÃƒÂ©sultat ou valeur par dÃƒÂ©faut
    """
    if denominator == 0:
        return default
    
    return numerator / denominator


def get_env_variable(name: str, default: Any = None, cast_type: type = str) -> Any:
    """
    RÃƒÂ©cupÃƒÂ¨re une variable d'environnement avec cast
    
    Args:
        name: Nom de la variable
        default: Valeur par dÃƒÂ©faut
        cast_type: Type de cast
        
    Returns:
        Valeur castÃƒÂ©e
    """
    value = os.getenv(name)
    
    if value is None:
        return default
    
    try:
        if cast_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return cast_type(value)
    except (ValueError, TypeError):
        logger.warning(f"Impossible de caster {name}={value} en {cast_type.__name__}")
        return default
