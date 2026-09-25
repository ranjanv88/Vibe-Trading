"""Read-only technical indicator tool.

Computes RSI, MACD, Bollinger Bands, SMA, EMA, and volume statistics for a given
symbol using the existing market-data pipeline. All computation is pure Python (numpy/pandas);
no new dependencies.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.agent.tools import BaseTool
from src.market_data import fetch_market_data
from src.tools.market_data_tool import _canonicalize_interval

logger = logging.getLogger(__name__)

# ── Indicator defaults ────────────────────────────────────────────────────────
_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_BB_PERIOD = 20
_BB_STD = 2.0
_SMA_PERIODS = (20, 50, 200)
_EMA_PERIOD = 20
_ADX_PERIOD = 14
_ATR_PERIOD = 14
_OBV_MA_PERIOD = 20
_DEFAULT_LOOKBACK = 200
_MAX_LOOKBACK = 500
#: Calendar days fetched per requested bar. Two covers weekends and holidays
#: for daily bars; a weekly or monthly bar needs its whole period, or 200
#: weekly bars would be asked for over 400 days and come back as 57 (#1479).
_CALENDAR_DAYS_PER_BAR = {"1W": 7, "1M": 31}


_CLOSE_KEYS = ("close", "Close", "CLOSE", "adj_close")
_VOLUME_KEYS = ("volume", "Volume", "VOLUME")
_DATE_KEYS = ("trade_date", "date", "datetime", "timestamp", "time")


def _extract_series(
    payload: Any, keys: tuple[str, ...], *, keep_missing: bool = False
) -> pd.Series | None:
    """Extract one numeric column from a loader payload, on its dates.

    Loader results can be a ``DataFrame``, a plain column mapping, or the
    ``list[dict]`` shape returned by :func:`fetch_market_data`. Preserve a real
    date index when one is available; a payload without dates intentionally
    keeps a ``RangeIndex`` so callers do not report a row number as a date.

    Args:
        payload: The loader result for one symbol.
        keys: Column spellings to look for, first match wins.
        keep_missing: Keep a dated bar whose value is missing as NaN instead
            of dropping it, so a gap cannot slide an older value into "latest".

    Returns:
        A float series, or ``None`` when the payload has no such column.
    """
    if isinstance(payload, list):
        if not payload or not isinstance(payload[0], dict):
            return None
        frame = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        inner = payload.get("data")
        if isinstance(inner, list):
            return _extract_series(inner, keys, keep_missing=keep_missing)
        try:
            frame = pd.DataFrame(payload)
        except ValueError:
            return None
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        return None

    key = next((key for key in keys if key in frame.columns), None)
    if key is None:
        return None

    values = pd.to_numeric(frame[key], errors="coerce")
    date_key = next((key for key in _DATE_KEYS if key in frame.columns), None)
    if date_key is not None:
        dates = pd.to_datetime(frame[date_key], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(frame.index, index=frame.index)
    elif not isinstance(
        frame.index, pd.RangeIndex
    ) and not pd.api.types.is_integer_dtype(frame.index.dtype):
        dates = pd.Series(
            pd.to_datetime(frame.index, errors="coerce"), index=frame.index
        )
    else:
        kept = values if keep_missing else values.dropna()
        return kept.reset_index(drop=True).astype(float)

    valid = dates.notna() if keep_missing else values.notna() & dates.notna()
    normalized = pd.Series(
        values.loc[valid].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates.loc[valid]),
        dtype=float,
    )
    return normalized.sort_index(kind="stable")


def _extract_close_series(payload: Any) -> pd.Series | None:
    """Extract the close-price series from a loader payload."""
    return _extract_series(payload, _CLOSE_KEYS)


def _extract_volume_series(payload: Any) -> pd.Series | None:
    """Extract the volume series, keeping a dated bar with no volume as NaN."""
    return _extract_series(payload, _VOLUME_KEYS, keep_missing=True)


def _compute_volume_stats(
    volume: pd.Series, period: int = 20
) -> dict[str, float | None]:
    """Return latest volume, its trailing mean, and latest/mean ratio.

    The trailing mean is reported only when the latest period bars all carry
    finite volume values, so a missing bar cannot silently change the window.
    """
    if volume.empty:
        return {"latest": None, "sma_20": None, "ratio_20": None}

    latest_raw = volume.iloc[-1]
    latest = float(latest_raw) if pd.notna(latest_raw) else None
    window = volume.iloc[-period:]
    sma = None
    ratio = None
    if len(window) == period and window.notna().all():
        sma = float(window.mean())
        if latest is not None and sma != 0:
            ratio = float(latest / sma)

    return {"latest": latest, "sma_20": sma, "ratio_20": ratio}


def _compute_sma(close: pd.Series, period: int) -> float | None:
    """Simple moving average over the last *period* bars."""
    if len(close) < period:
        return None
    return float(close.iloc[-period:].mean())


def _compute_ema(close: pd.Series, period: int) -> float | None:
    """Exponential moving average over the full series."""
    if len(close) < period:
        return None
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def _compute_rsi(close: pd.Series, period: int = _RSI_PERIOD) -> float | None:
    """Relative Strength Index (Wilder smoothing) over *period* bars.

    Matches the Wilder-EWM convention already used for RSI elsewhere in this
    codebase (``shadow_account/extractor.py``, ``shadow_account/scanner.py``,
    ``skills/technical-basic/example_signal_engine.py``): a plain rolling mean
    of gains/losses is a materially different, non-Wilder technique and used
    to diverge from those siblings by several RSI points on ordinary data.
    """
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_adx(df: pd.DataFrame, period: int = _ADX_PERIOD) -> dict[str, float | None] | None:
    """Full Wilder ADX chain: TR → +DI/-DI → DX → ADX. Trend strength gauge."""
    if df is None or len(df) < period * 2 + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()
    return {
        "adx": round(float(adx_series.iloc[-1]), 2),
        "plus_di": round(float(plus_di.iloc[-1]), 2),
        "minus_di": round(float(minus_di.iloc[-1]), 2),
    }


def _compute_atr_pct(df: pd.DataFrame, period: int = _ATR_PERIOD) -> float | None:
    """Average True Range as % of price — position-sizing / stop distance."""
    if df is None or len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])
    return round(atr / float(close.iloc[-1]) * 100, 2)


def _compute_obv_signal(df: pd.DataFrame, ma_period: int = _OBV_MA_PERIOD) -> dict[str, Any] | None:
    """On-Balance Volume trend vs its own moving average — volume confirmation."""
    if df is None or len(df) < ma_period + 1:
        return None
    close, volume = df["close"], df["volume"]
    obv = (volume * np.sign(close.diff().fillna(0))).cumsum()
    obv_ma = obv.rolling(ma_period).mean()
    vol_ma = volume.rolling(ma_period).mean()
    rising = bool(obv.iloc[-1] > float(obv_ma.iloc[-1]))
    vol_ratio = float(volume.iloc[-1]) / max(float(vol_ma.iloc[-1]), 1e-9)
    return {
        "obv_above_ma": rising,
        "volume_ratio": round(vol_ratio, 2),
    }


def _compute_macd(
    close: pd.Series,
    fast: int = _MACD_FAST,
    slow: int = _MACD_SLOW,
    signal: int = _MACD_SIGNAL,
) -> dict[str, float | None] | None:
    """MACD line, signal line, and histogram."""
    if len(close) < slow + signal:
        return None
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd_line": round(float(macd_line.iloc[-1]), 4),
        "signal_line": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


def _compute_bollinger(
    close: pd.Series,
    period: int = _BB_PERIOD,
    num_std: float = _BB_STD,
) -> dict[str, float | None] | None:
    """Bollinger Bands: upper, middle (SMA), lower."""
    if len(close) < period:
        return None
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return {
        "upper": round(float(sma.iloc[-1] + num_std * std.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float(sma.iloc[-1] - num_std * std.iloc[-1]), 2),
    }


class TechnicalIndicatorTool(BaseTool):
    """Compute common technical indicators for a symbol.

    Fetches OHLCV data through the existing loader pipeline, then computes
    RSI, MACD, Bollinger Bands, SMA, EMA, and volume statistics. All math is
    pure Python; no new dependencies, no network calls beyond what the loaders
    already do.
    """

    name = "technical_indicators"
    description = (
        "Compute common technical indicators (RSI, MACD, Bollinger Bands, "
        "SMA, EMA, ADX/DI, ATR%, OBV/volume) plus a composite trend signal "
        "(bullish/bearish/neutral with strength and volume confirmation) plus "
        "latest/20-bar volume statistics for a trading symbol "
        "(volume.unit is the serving source's declared unit -- lots or shares -- "
        "or null when undeclared). "
        "Uses the project's data loaders "
        "to fetch price history, then computes indicators locally."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": (
                    "Trading symbol, e.g. AAPL for US stocks, "
                    "600519.SH for A-shares, BTC-USDT for crypto."
                ),
            },
            "interval": {
                "type": "string",
                "description": "Bar interval: 1d (default), 1wk, or 1mo.",
                "default": "1d",
            },
            "lookback": {
                "type": "integer",
                "description": (
                    "Number of bars to fetch. Default 200, max 500. "
                    "More bars = more accurate long-term indicators (SMA 200)."
                ),
                "default": _DEFAULT_LOOKBACK,
            },
        },
        "required": ["symbol"],
    }
    repeatable = True
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        symbol = str(kwargs.get("symbol", "")).strip()
        requested_interval = str(kwargs.get("interval", "1d")).strip()
        lookback_raw = kwargs.get("lookback", _DEFAULT_LOOKBACK)

        if not symbol:
            return json.dumps({"ok": False, "error": "symbol is required"})
        # The documented 1d / 1wk / 1mo spellings used to reach the loaders
        # verbatim, and most of them know neither weekly spelling.
        interval = _canonicalize_interval(requested_interval)
        if interval is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unsupported interval {requested_interval!r}; use 1d, 1wk or 1mo",
                }
            )

        try:
            lookback = int(lookback_raw)
        except (TypeError, ValueError):
            lookback = _DEFAULT_LOOKBACK
        lookback = max(10, min(lookback, _MAX_LOOKBACK))

        # Fetch enough bars to cover the longest indicator window + buffer.
        end_date = datetime.now().strftime("%Y-%m-%d")
        days = lookback * _CALENDAR_DAYS_PER_BAR.get(interval, 2)
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            data = fetch_market_data(
                codes=[symbol],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                # Indicators require consecutive bars. ``max_rows=lookback``
                # would make the shared helper even-stride sample long windows.
                max_rows=0,
                # For volume_unit: lots on the A-share sources, shares on the
                # Yahoo family (#1062) -- 100x apart, and nothing to infer from.
                include_provenance=True,
            )
        except Exception as exc:
            logger.debug("fetch_market_data failed for %s: %s", symbol, exc)
            return json.dumps({"ok": False, "error": f"Failed to fetch data: {exc}"})

        df = data.get(symbol)
        if df is None:
            return json.dumps({"ok": False, "error": f"No data returned for {symbol}"})
        if isinstance(df, pd.DataFrame) and df.empty:
            return json.dumps({"ok": False, "error": f"No data returned for {symbol}"})
        if isinstance(df, list) and not df:
            return json.dumps({"ok": False, "error": f"No data returned for {symbol}"})
        if isinstance(df, dict) and not df:
            return json.dumps({"ok": False, "error": f"No data returned for {symbol}"})
        if isinstance(df, dict) and df.get("truncated") is True:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Indicator calculation requires consecutive, untruncated bars",
                }
            )
        # Normalize list/wrapper payloads to a DataFrame so the full-OHLCV
        # indicators (ADX/ATR/OBV) below see high/low/close whatever shape
        # the loader returned.
        if isinstance(df, list):
            df = pd.DataFrame(df) if df and isinstance(df[0], dict) else None
        elif isinstance(df, dict) and isinstance(df.get("data"), list):
            df = pd.DataFrame(df["data"])
        if df is None or (hasattr(df, "empty") and df.empty):
            return json.dumps({"ok": False, "error": f"No data returned for {symbol}"})

        close = _extract_close_series(df)
        if close is None or close.empty:
            return json.dumps({"ok": False, "error": "No close price column in data"})
        close = close.sort_index(kind="stable").tail(lookback)

        volume = _extract_volume_series(df)
        if volume is not None:
            volume = volume.sort_index(kind="stable")
            if isinstance(close.index, pd.DatetimeIndex) and isinstance(
                volume.index, pd.DatetimeIndex
            ):
                volume = volume.reindex(close.index)
            else:
                volume = volume.tail(len(close)).reset_index(drop=True)
        volume_stats = (
            _compute_volume_stats(volume)
            if volume is not None
            else {"latest": None, "sma_20": None, "ratio_20": None}
        )
        provenance = (data.get("_provenance") or {}).get(symbol)
        volume_stats["unit"] = (
            provenance.get("volume_unit") if isinstance(provenance, dict) else None
        )

        # ── Compute indicators ────────────────────────────────────────────
        indicators: dict[str, Any] = {
            "rsi_14": _compute_rsi(close),
            "macd": _compute_macd(close),
            "bollinger": _compute_bollinger(close),
            "volume": volume_stats,
        }
        for period in _SMA_PERIODS:
            indicators[f"sma_{period}"] = _compute_sma(close, period)
        indicators[f"ema_{_EMA_PERIOD}"] = _compute_ema(close, _EMA_PERIOD)

        # Full-OHLCV indicators (ADX / ATR / OBV) need more than close alone.
        ohlcv = None
        if isinstance(df, pd.DataFrame) and {"high", "low", "close"}.issubset(
            {c.lower() for c in df.columns}
        ):
            ohlcv = df.copy()
            ohlcv.columns = [str(c).lower() for c in ohlcv.columns]
            if "volume" not in ohlcv.columns:
                ohlcv["volume"] = 0.0
        else:
            try:
                d = df.to_dict() if callable(getattr(df, "to_dict", None)) else dict(df)
                cols = {k.lower(): v for k, v in d.items()}
                if {"high", "low", "close"}.issubset(cols):
                    ohlcv = pd.DataFrame(cols)
                    if "volume" not in ohlcv.columns:
                        ohlcv["volume"] = 0.0
            except Exception:  # noqa: BLE001
                ohlcv = None

        if ohlcv is not None:
            indicators["adx"] = _compute_adx(ohlcv, _ADX_PERIOD)
            indicators["atr_pct"] = _compute_atr_pct(ohlcv, _ATR_PERIOD)
            indicators["obv"] = _compute_obv_signal(ohlcv, _OBV_MA_PERIOD)

            # ── Composite signal (3-dimensional voting, per technical-basic skill)
            rsi_val = indicators.get("rsi_14")
            macd = indicators.get("macd") or {}
            adx = indicators.get("adx") or {}
            obv = indicators.get("obv") or {}
            trend_bull = (
                indicators.get("ema_20") is not None
                and indicators.get("sma_50") is not None
                and indicators["ema_20"] > indicators["sma_50"]
                and (macd.get("histogram") or 0) > 0
                and (adx.get("plus_di") or 0) > (adx.get("minus_di") or -1)
            )
            trend_bear = (
                indicators.get("ema_20") is not None
                and indicators.get("sma_50") is not None
                and indicators["ema_20"] < indicators["sma_50"]
                and (macd.get("histogram") or 0) < 0
                and (adx.get("plus_di") or 1) < (adx.get("minus_di") or 0)
            )
            strong_trend = (adx.get("adx") or 0) >= 25.0
            vol_confirm = bool(obv.get("obv_above_ma"))
            if trend_bull and rsi_val is not None and rsi_val < 70 and vol_confirm:
                composite = 1
            elif trend_bear and rsi_val is not None and rsi_val > 30 and not vol_confirm:
                composite = -1
            else:
                composite = 0
            indicators["signal"] = {
                "composite": composite,  # 1 long, -1 short, 0 stand aside
                "trend": "bullish" if trend_bull else ("bearish" if trend_bear else "neutral"),
                "trend_strength": ("strong" if strong_trend else "weak"),
                "volume_confirms": vol_confirm,
                "overbought": rsi_val is not None and rsi_val >= 70,
                "oversold": rsi_val is not None and rsi_val <= 30,
            }

        latest_close = float(close.iloc[-1]) if len(close) > 0 else None
        latest_date = (
            str(close.index[-1])[:10]
            if isinstance(close.index, pd.DatetimeIndex) and len(close) > 0
            else None
        )

        return json.dumps(
            {
                "ok": True,
                "symbol": symbol,
                "interval": interval,
                "latest_close": latest_close,
                "latest_date": latest_date,
                "indicators": indicators,
            },
            ensure_ascii=False,
            default=str,
        )
