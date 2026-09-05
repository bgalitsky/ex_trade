#!/usr/bin/env python3
"""Backtest and optimize an ex-dividend-date trading strategy.

The program accepts either the supplied ZIP archive or a directory containing
price CSVs and matching ex-date files.  Expected pairs look like:

    ISPY (....).csv
    ISPY_ex_dividend_date.txt

It first evaluates the requested fixed rule (buy two trading sessions after
the ex-date and sell eight trading sessions after it). It then searches the
user-specified neighborhoods around that rule: buy 2 +/- 1 sessions after the
ex-date (1 through 3), and sell 8 +/- 4 sessions after the ex-date (4 through
12). The selected rule is evaluated on an untouched out-of-sample period.

Distributions are excluded because the calculation uses raw OHLC prices, not
adjusted prices.  Fractional shares are used and all proceeds are compounded.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd


FIXED_BUY_AFTER = 2
FIXED_SELL_AFTER = 8


@dataclass
class InstrumentData:
    ticker: str
    prices: pd.DataFrame
    ex_dates: pd.DatetimeIndex
    date_shift_days: int
    price_source: str
    ex_date_source: str


@dataclass
class BacktestResult:
    summary: dict[str, object]
    trades: pd.DataFrame
    equity: pd.DataFrame
    skipped: pd.DataFrame


def _ticker_from_price_name(name: str) -> str:
    """Infer a ticker from names such as 'JEPI (dates).csv'."""
    stem = Path(name).stem
    return re.split(r"[\s(]", stem, maxsplit=1)[0].strip().upper()


def _ticker_from_ex_name(name: str) -> str | None:
    match = re.match(
        r"^(.+?)_ex[_ -]?dividend[_ -]?date(?:s)?\.(?:txt|csv)$",
        Path(name).name,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().upper() if match else None


def _read_price_csv(
    source: BinaryIO | str | Path,
    source_name: str,
    date_shift: str,
) -> tuple[pd.DataFrame, int]:
    prices = pd.read_csv(source)
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"{source_name}: missing columns {sorted(missing)}")

    raw_dates = pd.to_datetime(prices["Date"], errors="coerce", utc=True)
    if raw_dates.isna().any():
        bad = int(raw_dates.isna().sum())
        raise ValueError(f"{source_name}: {bad} unparseable Date value(s)")

    if date_shift == "auto":
        # Some exports label each U.S. session at 21:00 UTC on the preceding
        # calendar day.  Pick the small shift producing the fewest weekend
        # observations; prefer zero when scores tie.
        candidates = (-1, 0, 1)
        scored: list[tuple[int, int, int]] = []
        for shift in candidates:
            shifted = (raw_dates + pd.Timedelta(days=shift)).dt.normalize()
            weekend_count = int((shifted.dt.dayofweek >= 5).sum())
            scored.append((weekend_count, abs(shift), shift))
        chosen_shift = min(scored)[2]
    else:
        chosen_shift = int(date_shift)

    normalized = (raw_dates + pd.Timedelta(days=chosen_shift)).dt.tz_localize(None)
    prices = prices.copy()
    prices["Date"] = normalized.dt.normalize()

    for column in ("Open", "High", "Low", "Close"):
        prices[column] = pd.to_numeric(
            prices[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    if "Volume" in prices.columns:
        prices["Volume"] = pd.to_numeric(
            prices["Volume"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    if prices[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError(f"{source_name}: unparseable OHLC value(s)")
    if (prices[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{source_name}: OHLC prices must be positive")

    prices = (
        prices.sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )
    return prices, chosen_shift


def _read_ex_dates(source: BinaryIO | str | Path, source_name: str) -> pd.DatetimeIndex:
    frame = pd.read_csv(source)
    if frame.empty or len(frame.columns) == 0:
        raise ValueError(f"{source_name}: no ex-dividend dates found")
    preferred = next(
        (c for c in frame.columns if c.strip().lower() == "ex_dividend_date"),
        frame.columns[0],
    )
    dates = pd.to_datetime(frame[preferred], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{source_name}: unparseable ex-dividend date(s)")
    return pd.DatetimeIndex(dates.dt.normalize().drop_duplicates().sort_values())


def load_instruments(input_path: Path, date_shift: str = "auto") -> dict[str, InstrumentData]:
    """Load all complete price/ex-date pairs from a ZIP or directory."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    loaded: dict[str, InstrumentData] = {}

    if input_path.is_file() and zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
            price_names = {
                _ticker_from_price_name(n): n
                for n in names
                if n.lower().endswith(".csv") and _ticker_from_ex_name(n) is None
            }
            ex_names = {
                ticker: n
                for n in names
                if (ticker := _ticker_from_ex_name(n)) is not None
            }
            for ticker in sorted(price_names.keys() & ex_names.keys()):
                price_name, ex_name = price_names[ticker], ex_names[ticker]
                with archive.open(price_name) as price_stream:
                    price_bytes = io.BytesIO(price_stream.read())
                with archive.open(ex_name) as ex_stream:
                    ex_bytes = io.BytesIO(ex_stream.read())
                prices, shift = _read_price_csv(price_bytes, price_name, date_shift)
                ex_dates = _read_ex_dates(ex_bytes, ex_name)
                loaded[ticker] = InstrumentData(
                    ticker=ticker,
                    prices=prices,
                    ex_dates=ex_dates,
                    date_shift_days=shift,
                    price_source=price_name,
                    ex_date_source=ex_name,
                )
    elif input_path.is_dir():
        files = [p for p in input_path.iterdir() if p.is_file()]
        price_paths = {
            _ticker_from_price_name(p.name): p
            for p in files
            if p.suffix.lower() == ".csv" and _ticker_from_ex_name(p.name) is None
        }
        ex_paths = {
            ticker: p
            for p in files
            if (ticker := _ticker_from_ex_name(p.name)) is not None
        }
        for ticker in sorted(price_paths.keys() & ex_paths.keys()):
            price_path, ex_path = price_paths[ticker], ex_paths[ticker]
            prices, shift = _read_price_csv(price_path, price_path.name, date_shift)
            ex_dates = _read_ex_dates(ex_path, ex_path.name)
            loaded[ticker] = InstrumentData(
                ticker=ticker,
                prices=prices,
                ex_dates=ex_dates,
                date_shift_days=shift,
                price_source=price_path.name,
                ex_date_source=ex_path.name,
            )
    else:
        raise ValueError("Input must be a ZIP archive or a directory")

    if not loaded:
        raise ValueError(
            "No complete ticker pairs found. Expected 'TICKER (...).csv' and "
            "'TICKER_ex_dividend_date.txt'."
        )
    return loaded


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "ex_date",
            "ex_session_date",
            "buy_date",
            "sell_date",
            "buy_after_sessions",
            "sell_after_sessions",
            "buy_price",
            "sell_price",
            "capital_before",
            "shares",
            "pnl",
            "trade_return",
            "capital_after",
        ]
    )


def _drawdown(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    peak = equity.cummax()
    return equity / peak - 1.0


def _make_equity_curve(
    instrument: InstrumentData,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_capital: float,
    mark_price_column: str = "Close",
) -> pd.DataFrame:
    prices = instrument.prices
    sessions = prices.loc[prices["Date"].between(start, end), ["Date", mark_price_column]].copy()
    sessions.rename(columns={mark_price_column: "mark_price"}, inplace=True)
    if sessions.empty:
        return pd.DataFrame(columns=["ticker", "date", "equity", "drawdown", "invested"])

    sessions["equity"] = starting_capital
    sessions["invested"] = False
    cash = starting_capital
    for row in trades.itertuples(index=False):
        mask = sessions["Date"].between(row.buy_date, row.sell_date)
        sessions.loc[mask, "equity"] = row.shares * sessions.loc[mask, "mark_price"]
        sessions.loc[mask, "invested"] = True
        after_sale = sessions["Date"] > row.sell_date
        sessions.loc[after_sale, "equity"] = row.capital_after
        cash = float(row.capital_after)

    if trades.empty:
        sessions["equity"] = cash
    sessions["drawdown"] = _drawdown(sessions["equity"])
    sessions.insert(0, "ticker", instrument.ticker)
    sessions.rename(columns={"Date": "date"}, inplace=True)
    return sessions[["ticker", "date", "equity", "drawdown", "invested"]]


def _summarize(
    instrument: InstrumentData,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_capital: float,
    buy_after: int,
    sell_after: int,
    skipped_count: int,
) -> dict[str, object]:
    ending_capital = (
        float(trades.iloc[-1]["capital_after"]) if not trades.empty else starting_capital
    )
    total_return = ending_capital / starting_capital - 1.0
    calendar_days = max((end - start).days + 1, 1)
    cagr = (ending_capital / starting_capital) ** (365.25 / calendar_days) - 1.0

    trade_returns = trades["trade_return"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = int((trade_returns > 0).sum())
    losses = int((trade_returns < 0).sum())
    win_rate = wins / len(trade_returns) if len(trade_returns) else np.nan
    average_trade = float(trade_returns.mean()) if len(trade_returns) else np.nan
    trade_vol = float(trade_returns.std(ddof=1)) if len(trade_returns) > 1 else np.nan
    years = calendar_days / 365.25
    trades_per_year = len(trade_returns) / years if years > 0 else np.nan
    trade_sharpe = (
        float(trade_returns.mean() / trade_vol * math.sqrt(trades_per_year))
        if len(trade_returns) > 1 and trade_vol > 0 and trades_per_year > 0
        else np.nan
    )
    max_drawdown = float(equity["drawdown"].min()) if not equity.empty else 0.0
    return_over_drawdown = (
        total_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    )
    exposure = float(equity["invested"].mean()) if not equity.empty else 0.0

    window = instrument.prices[instrument.prices["Date"].between(start, end)]
    if len(window) >= 2:
        buy_hold_return = float(window.iloc[-1]["Close"] / window.iloc[0]["Close"] - 1.0)
    else:
        buy_hold_return = np.nan

    return {
        "ticker": instrument.ticker,
        "period_start": start.date().isoformat(),
        "period_end": end.date().isoformat(),
        "buy_after_sessions": buy_after,
        "sell_after_sessions": sell_after,
        "starting_capital": starting_capital,
        "ending_capital": ending_capital,
        "profit": ending_capital - starting_capital,
        "total_return": total_return,
        "cagr": cagr,
        "buy_hold_price_return": buy_hold_return,
        "max_drawdown": max_drawdown,
        "return_over_drawdown": return_over_drawdown,
        "trade_sharpe": trade_sharpe,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "average_trade_return": average_trade,
        "exposure_fraction": exposure,
        "skipped_events": skipped_count,
        "date_shift_days": instrument.date_shift_days,
    }


def backtest(
    instrument: InstrumentData,
    buy_after: int,
    sell_after: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_capital: float = 20_000.0,
    buy_price_column: str = "Close",
    sell_price_column: str = "Close",
    ex_dates_override: Iterable[pd.Timestamp] | None = None,
) -> BacktestResult:
    """Run one fully compounded, single-position strategy."""
    if buy_after < 0 or sell_after <= buy_after:
        raise ValueError("Require 0 <= buy_after < sell_after")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")

    prices = instrument.prices
    session_dates = prices["Date"].to_numpy(dtype="datetime64[ns]")
    ex_dates = pd.DatetimeIndex(
        list(ex_dates_override) if ex_dates_override is not None else instrument.ex_dates
    )
    ex_dates = ex_dates[(ex_dates >= start) & (ex_dates <= end)]

    capital = float(starting_capital)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    last_sell_date: pd.Timestamp | None = None

    for ex_date in ex_dates:
        ex_pos = int(np.searchsorted(session_dates, np.datetime64(ex_date), side="left"))
        if ex_pos >= len(prices):
            skipped.append({"ticker": instrument.ticker, "ex_date": ex_date, "reason": "no price session on/after ex-date"})
            continue
        buy_pos = ex_pos + buy_after
        sell_pos = ex_pos + sell_after
        if sell_pos >= len(prices):
            skipped.append({"ticker": instrument.ticker, "ex_date": ex_date, "reason": "insufficient later price sessions"})
            continue

        ex_session_date = pd.Timestamp(prices.iloc[ex_pos]["Date"])
        buy_date = pd.Timestamp(prices.iloc[buy_pos]["Date"])
        sell_date = pd.Timestamp(prices.iloc[sell_pos]["Date"])
        if buy_date > end or sell_date > end:
            skipped.append({"ticker": instrument.ticker, "ex_date": ex_date, "reason": "trade does not finish inside period"})
            continue
        if last_sell_date is not None and buy_date < last_sell_date:
            raise ValueError(
                f"{instrument.ticker}: overlapping trades at {buy_date.date()}; "
                "the strategy cannot fully invest the same capital twice"
            )

        buy_price = float(prices.iloc[buy_pos][buy_price_column])
        sell_price = float(prices.iloc[sell_pos][sell_price_column])
        capital_before = capital
        shares = capital_before / buy_price
        capital = shares * sell_price
        trade_return = sell_price / buy_price - 1.0
        rows.append(
            {
                "ticker": instrument.ticker,
                "ex_date": ex_date,
                "ex_session_date": ex_session_date,
                "buy_date": buy_date,
                "sell_date": sell_date,
                "buy_after_sessions": buy_after,
                "sell_after_sessions": sell_after,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "capital_before": capital_before,
                "shares": shares,
                "pnl": capital - capital_before,
                "trade_return": trade_return,
                "capital_after": capital,
            }
        )
        last_sell_date = sell_date

    trades = pd.DataFrame(rows) if rows else _empty_trades()
    skipped_frame = pd.DataFrame(skipped, columns=["ticker", "ex_date", "reason"])
    equity = _make_equity_curve(
        instrument, trades, start, end, starting_capital, mark_price_column="Close"
    )
    summary = _summarize(
        instrument,
        trades,
        equity,
        start,
        end,
        starting_capital,
        buy_after,
        sell_after,
        len(skipped_frame),
    )
    return BacktestResult(summary=summary, trades=trades, equity=equity, skipped=skipped_frame)


def complete_ex_dates(
    instrument: InstrumentData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    maximum_sell_after: int,
) -> pd.DatetimeIndex:
    """Use a common event set so all optimization candidates are comparable."""
    session_dates = instrument.prices["Date"].to_numpy(dtype="datetime64[ns]")
    eligible: list[pd.Timestamp] = []
    for ex_date in instrument.ex_dates:
        if ex_date < start or ex_date > end:
            continue
        ex_pos = int(np.searchsorted(session_dates, np.datetime64(ex_date), side="left"))
        sell_pos = ex_pos + maximum_sell_after
        if ex_pos < len(session_dates) and sell_pos < len(session_dates):
            if pd.Timestamp(session_dates[sell_pos]) <= end:
                eligible.append(pd.Timestamp(ex_date))
    return pd.DatetimeIndex(eligible)


def optimize(
    instrument: InstrumentData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_capital: float,
    buy_min: int,
    buy_max: int,
    sell_min: int,
    sell_max: int,
    objective: str,
    buy_price_column: str,
    sell_price_column: str,
    min_trades: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    eligible = complete_ex_dates(instrument, start, end, sell_max)
    records: list[dict[str, object]] = []
    for buy_after in range(buy_min, buy_max + 1):
        for sell_after in range(sell_min, sell_max + 1):
            if sell_after <= buy_after:
                continue
            try:
                result = backtest(
                    instrument,
                    buy_after,
                    sell_after,
                    start,
                    end,
                    starting_capital,
                    buy_price_column,
                    sell_price_column,
                    ex_dates_override=eligible,
                )
            except ValueError as exc:
                # A long holding window can overlap the next monthly signal.
                # Such a rule cannot satisfy the fully-reused-capital premise,
                # so it is not an admissible optimization candidate.
                if "overlapping trades" in str(exc):
                    continue
                raise
            record = dict(result.summary)
            record["eligible_common_events"] = len(eligible)
            records.append(record)

    grid = pd.DataFrame(records)
    valid = grid[(grid["trades"] >= min_trades) & grid[objective].notna()].copy()
    if valid.empty:
        raise ValueError(
            f"{instrument.ticker}: no candidate has at least {min_trades} complete trades"
        )

    valid.sort_values(
        by=[objective, "max_drawdown", "buy_after_sessions", "sell_after_sessions"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    best = valid.iloc[0].to_dict()
    grid["selected"] = (
        (grid["buy_after_sessions"] == best["buy_after_sessions"])
        & (grid["sell_after_sessions"] == best["sell_after_sessions"])
    )
    grid.sort_values(
        by=[objective, "max_drawdown", "buy_after_sessions", "sell_after_sessions"],
        ascending=[False, False, True, True],
        inplace=True,
        na_position="last",
    )
    return grid.reset_index(drop=True), best


def yearly_summary(
    ticker: str,
    equity: pd.DataFrame,
    starting_capital: float,
) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(
            columns=["ticker", "year", "year_start_capital", "year_end_capital", "year_return"]
        )
    work = equity.copy()
    work["year"] = pd.to_datetime(work["date"]).dt.year
    rows: list[dict[str, object]] = []
    previous_end = starting_capital
    for year, group in work.groupby("year", sort=True):
        end_capital = float(group.iloc[-1]["equity"])
        rows.append(
            {
                "ticker": ticker,
                "year": int(year),
                "year_start_capital": previous_end,
                "year_end_capital": end_capital,
                "year_return": end_capital / previous_end - 1.0,
            }
        )
        previous_end = end_capital
    return pd.DataFrame(rows)


def _write_csv(frames: list[pd.DataFrame], path: Path) -> None:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    output = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    output.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.10f")


def _percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{100.0 * number:,.2f}%"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest fixed and optimized post-ex-date trading rules."
    )
    parser.add_argument("input", type=Path, help="ZIP archive or input directory")
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results"))
    parser.add_argument("--ticker", action="append", help="Ticker to run; repeat as needed")
    parser.add_argument("--capital", type=float, default=20_000.0)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-12-31", help="Capped automatically at last price date")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--fixed-buy-after", type=int, default=FIXED_BUY_AFTER)
    parser.add_argument("--fixed-sell-after", type=int, default=FIXED_SELL_AFTER)
    parser.add_argument(
        "--buy-min",
        type=int,
        default=1,
        help="Earliest buy offset; default 1 (= 2 - 1 trading sessions)",
    )
    parser.add_argument(
        "--buy-max",
        type=int,
        default=3,
        help="Latest buy offset; default 3 (= 2 + 1 trading sessions)",
    )
    parser.add_argument(
        "--sell-min",
        type=int,
        default=4,
        help="Earliest sell offset; default 4 (= 8 - 4 trading sessions)",
    )
    parser.add_argument(
        "--sell-max",
        type=int,
        default=12,
        help="Latest sell offset; default 12 (= 8 + 4 trading sessions)",
    )
    parser.add_argument("--min-trades", type=int, default=6)
    parser.add_argument(
        "--objective",
        choices=["ending_capital", "total_return", "cagr", "trade_sharpe", "return_over_drawdown", "win_rate"],
        default="total_return",
    )
    parser.add_argument("--buy-price", choices=["Open", "Close"], default="Close")
    parser.add_argument("--sell-price", choices=["Open", "Close"], default="Close")
    parser.add_argument(
        "--date-shift-days",
        choices=["auto", "-1", "0", "1"],
        default="auto",
        help="Calendar-day correction applied before identifying sessions",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Only run the fixed rule",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(args.start).normalize()
    requested_end = pd.Timestamp(args.end).normalize()
    train_end_requested = pd.Timestamp(args.train_end).normalize()
    test_start = pd.Timestamp(args.test_start).normalize()
    if requested_end < start:
        raise ValueError("--end must not precede --start")
    if args.buy_min < 0 or args.buy_max < args.buy_min:
        raise ValueError("Require 0 <= --buy-min <= --buy-max")
    if args.sell_min < 1 or args.sell_max < args.sell_min:
        raise ValueError("Require 1 <= --sell-min <= --sell-max")
    if args.sell_max <= args.buy_min:
        raise ValueError("The sell range must contain an offset later than the buy range")

    instruments = load_instruments(args.input, date_shift=args.date_shift_days)
    if args.ticker:
        wanted = {ticker.upper() for ticker in args.ticker}
        missing = sorted(wanted.difference(instruments))
        if missing:
            raise ValueError(f"Requested ticker(s) not found: {', '.join(missing)}")
        instruments = {k: v for k, v in instruments.items() if k in wanted}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_summaries: list[pd.DataFrame] = []
    fixed_trades: list[pd.DataFrame] = []
    fixed_equities: list[pd.DataFrame] = []
    fixed_yearly: list[pd.DataFrame] = []
    skipped_events: list[pd.DataFrame] = []
    grids: list[pd.DataFrame] = []
    selected_train: list[pd.DataFrame] = []
    selected_test: list[pd.DataFrame] = []
    selected_test_trades: list[pd.DataFrame] = []

    print("Fixed rule and walk-forward optimization")
    print(f"Starting capital per independent ticker: ${args.capital:,.2f}")
    print(f"Execution prices: buy={args.buy_price}, sell={args.sell_price}")
    if not args.no_optimize:
        print(
            f"Optimization grid: buy +{args.buy_min}..+{args.buy_max}, "
            f"sell +{args.sell_min}..+{args.sell_max} trading sessions from ex-date"
        )
    print()

    for ticker, instrument in instruments.items():
        data_end = pd.Timestamp(instrument.prices["Date"].max())
        end = min(requested_end, data_end)
        if end < start:
            print(f"Skipping {ticker}: no price data in requested period", file=sys.stderr)
            continue

        fixed = backtest(
            instrument,
            args.fixed_buy_after,
            args.fixed_sell_after,
            start,
            end,
            args.capital,
            args.buy_price,
            args.sell_price,
        )
        fixed_summaries.append(pd.DataFrame([fixed.summary]))
        fixed_trades.append(fixed.trades)
        fixed_equities.append(fixed.equity)
        fixed_yearly.append(yearly_summary(ticker, fixed.equity, args.capital))
        skipped_events.append(fixed.skipped.assign(run="fixed"))

        print(
            f"{ticker:5s} fixed {args.fixed_buy_after}/{args.fixed_sell_after}: "
            f"${fixed.summary['ending_capital']:,.2f}  "
            f"return={_percent(fixed.summary['total_return'])}  "
            f"maxDD={_percent(fixed.summary['max_drawdown'])}  "
            f"trades={fixed.summary['trades']}  "
            f"date_shift={instrument.date_shift_days:+d}d"
        )

        if args.no_optimize:
            continue

        train_end = min(train_end_requested, end)
        if train_end < start or test_start > end:
            print(f"      optimization skipped: training or test window unavailable")
            continue

        grid, best = optimize(
            instrument,
            start,
            train_end,
            args.capital,
            args.buy_min,
            args.buy_max,
            args.sell_min,
            args.sell_max,
            args.objective,
            args.buy_price,
            args.sell_price,
            args.min_trades,
        )
        grid.insert(1, "optimization_objective", args.objective)
        grids.append(grid)
        selected_train.append(pd.DataFrame([best]))

        best_buy = int(best["buy_after_sessions"])
        best_sell = int(best["sell_after_sessions"])
        test = backtest(
            instrument,
            best_buy,
            best_sell,
            test_start,
            end,
            args.capital,
            args.buy_price,
            args.sell_price,
        )
        selected_test.append(pd.DataFrame([test.summary]))
        selected_test_trades.append(test.trades)
        skipped_events.append(test.skipped.assign(run="optimized_test"))
        print(
            f"      selected on {start.date()}..{train_end.date()}: "
            f"{best_buy}/{best_sell}; 2026 OOS return={_percent(test.summary['total_return'])}, "
            f"ending=${test.summary['ending_capital']:,.2f}, trades={test.summary['trades']}"
        )

    if not fixed_summaries:
        raise ValueError("No ticker had data in the requested period")

    _write_csv(fixed_summaries, args.output_dir / "fixed_rule_summary.csv")
    _write_csv(fixed_trades, args.output_dir / "fixed_rule_trades.csv")
    _write_csv(fixed_equities, args.output_dir / "fixed_rule_equity_curves.csv")
    _write_csv(fixed_yearly, args.output_dir / "fixed_rule_yearly_summary.csv")
    _write_csv(skipped_events, args.output_dir / "skipped_events.csv")
    if grids:
        _write_csv(grids, args.output_dir / "optimization_grid_2024_2025.csv")
        _write_csv(selected_train, args.output_dir / "optimized_rules_training_summary.csv")
        _write_csv(selected_test, args.output_dir / "optimized_rules_2026_oos_summary.csv")
        _write_csv(selected_test_trades, args.output_dir / "optimized_rules_2026_oos_trades.csv")

    print()
    print(f"CSV results written to: {args.output_dir.resolve()}")
    print(
        "Reminder: the optimized rule is a research result, not a guarantee; "
        "2026 is reported separately as out-of-sample data."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
