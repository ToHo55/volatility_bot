"""
Volatility Trading Bot
"""

from .backtest import BacktestResult
from .strategies.mean_revert.signal import MeanRevertSignal
from .strategies.breakout_atr.signal import BreakoutATRSignal
from .strategies.ml_boost.signal import MLBoostSignal

__all__ = [
    'BacktestResult',
    'MeanRevertSignal',
    'BreakoutATRSignal',
    'MLBoostSignal'
]

__version__ = '0.1.0' 