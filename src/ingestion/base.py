"""
BaseFetcher — contract every NSE data fetcher must honour.

Adding a new data source = subclass BaseFetcher, implement fetch() and name.
The orchestrator only talks to this interface, so new sources plug in without
touching orchestration logic.
"""
from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

import pandas as pd

__all__ = ["BaseFetcher"]


class BaseFetcher(ABC):
    """Abstract base for all NSE data fetchers."""

    def __init__(self, client: "NSEHttpClient") -> None:  # noqa: F821
        self._client = client

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label used in log messages."""

    @abstractmethod
    def fetch(self, trade_date: datetime.date) -> pd.DataFrame:
        """
        Fetch data for trade_date.

        Returns an empty DataFrame (not None) when data is legitimately
        unavailable (holiday, weekend, file not yet published).
        Raises FetchError or ParseError on genuine failures.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client={self._client!r})"
