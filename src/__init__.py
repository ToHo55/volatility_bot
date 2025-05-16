"""
Volatility Trading Bot
"""

from .strategies import MLBoostSignal, MeanRevertSignal, BreakoutATRSignal
from .backtest import BacktestEngine
from .executor import Executor

__all__ = [
    'MLBoostSignal',
    'MeanRevertSignal',
    'BreakoutATRSignal',
    'BacktestEngine',
    'Executor'
]

__version__ = '0.1.0' 