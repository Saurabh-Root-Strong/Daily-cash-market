"""
Domain type aliases.

Use these instead of bare primitives to make function signatures self-documenting
and to make future type narrowing easy.
"""
from __future__ import annotations

import datetime
from typing import TypeAlias

__all__ = [
    "TradeDate",
    "Symbol",
    "Sector",
    "Industry",
    "TurnoverLacs",
    "DeliveryPct",
    "PriceChangePct",
]

TradeDate: TypeAlias = datetime.date
Symbol: TypeAlias = str
Sector: TypeAlias = str
Industry: TypeAlias = str
TurnoverLacs: TypeAlias = float
DeliveryPct: TypeAlias = float
PriceChangePct: TypeAlias = float
