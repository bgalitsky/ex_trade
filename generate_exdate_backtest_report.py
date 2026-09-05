#!/usr/bin/env python3
"""Generate a polished PDF report for the ex-date backtest archive.

This script is the reporting companion to ``ispy_exdate_optimizer.py``. It
loads the same ZIP archive (or an extracted directory), reruns the fixed and
optimized strategies, and writes one combined PDF containing:

* methodology and assumptions;
* a cross-fund executive summary;
* fixed 2/8 results for 2024 through 2026 YTD;
* the five best training-period offset pairs for each fund;
* annual fixed-rule results;
* 2026 out-of-sample comparison against the fixed rule;
* transaction-by-transaction 2026 results; and
* out-of-sample equity charts.

Default optimization ranges match the requested neighborhoods:

* buy 2 +/- 1 trading sessions after ex-date: +1 through +3;
* sell 8 +/- 4 trading sessions after ex-date: +4 through +12.

The PDF uses raw Close/Open prices from the supplied files. Distributions,
taxes, fees, and spreads are excluded. Capital is fully reused and compounded.
"""

from __future__ import annotations

import argparse
import html
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from ispy_exdate_optimizer import (
        BacktestResult,
        InstrumentData,
        backtest,
        load_instruments,
        optimize,
    )
except ImportError as exc:  # pragma: no cover - only reached when deployed incorrectly
    raise SystemExit(
        "Place generate_exdate_backtest_report.py beside "
        "ispy_exdate_optimizer.py, then run it again."
    ) from exc


NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#1F77B4")
PALE_BLUE = colors.HexColor("#EAF3FA")
PALE_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#6B7280")
GRID = colors.HexColor("#D6DCE2")
WHITE = colors.white
FONT = "ReportSans"
FONT_BOLD = "ReportSans-Bold"


@dataclass
class FundReport:
    instrument: InstrumentData
    period_end: pd.Timestamp
    fixed_full: BacktestResult
    optimization_grid: pd.DataFrame
    selected_training: dict[str, Any]
    fixed_test: BacktestResult
    optimized_test: BacktestResult
    annual: pd.DataFrame
    equity_chart_path: Path


def money(value: Any, signed: bool = False, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    sign = "+" if signed and number > 0 else "-" if number < 0 else ""
    return f"{sign}${abs(number):,.{digits}f}"


def percent(value: Any, signed: bool = False, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{100.0 * number:.{digits}f}%"


def integer(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def iso(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def short_date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%b %d, %Y")


def p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(str(text)), style)


def rich(text: str, style: ParagraphStyle) -> Paragraph:
    """Paragraph helper for trusted, locally composed ReportLab markup."""
    return Paragraph(text, style)


def register_fonts() -> None:
    """Embed Matplotlib's DejaVu fonts for consistent cross-platform rendering."""
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular = font_dir / "DejaVuSans.ttf"
    bold = font_dir / "DejaVuSans-Bold.ttf"
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Matplotlib DejaVu Sans font files were not found")
    if FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT, str(regular)))
    if FONT_BOLD not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_BOLD)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["title"] = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontName=FONT_BOLD,
        fontSize=25,
        leading=30,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    styles["subtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontName=FONT,
        fontSize=12,
        leading=17,
        textColor=MID_GRAY,
        spaceAfter=18,
    )
    styles["h1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName=FONT_BOLD,
        fontSize=19,
        leading=23,
        textColor=NAVY,
        spaceBefore=2,
        spaceAfter=10,
        keepWithNext=True,
    )
    styles["h2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName=FONT_BOLD,
        fontSize=12.5,
        leading=16,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    styles["body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName=FONT,
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#20262D"),
        spaceAfter=6,
    )
    styles["small"] = ParagraphStyle(
        "Small",
        parent=styles["body"],
        fontSize=7.8,
        leading=10.2,
        textColor=MID_GRAY,
        spaceAfter=3,
    )
    styles["bullet"] = ParagraphStyle(
        "Bullet",
        parent=styles["body"],
        leftIndent=14,
        firstLineIndent=-8,
        bulletIndent=3,
        spaceAfter=5,
    )
    styles["callout"] = ParagraphStyle(
        "Callout",
        parent=styles["body"],
        fontSize=9.1,
        leading=13,
        leftIndent=9,
        rightIndent=9,
        borderColor=BLUE,
        borderWidth=0.8,
        borderPadding=8,
        backColor=PALE_BLUE,
        spaceBefore=6,
        spaceAfter=9,
    )
    styles["table_header"] = ParagraphStyle(
        "TableHeader",
        parent=styles["small"],
        fontName=FONT_BOLD,
        fontSize=7.3,
        leading=8.5,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
    styles["table_cell"] = ParagraphStyle(
        "TableCell",
        parent=styles["small"],
        fontSize=7.3,
        leading=9,
        textColor=colors.HexColor("#20262D"),
        alignment=TA_CENTER,
    )
    styles["table_left"] = ParagraphStyle(
        "TableLeft",
        parent=styles["table_cell"],
        alignment=TA_LEFT,
    )
    styles["table_right"] = ParagraphStyle(
        "TableRight",
        parent=styles["table_cell"],
        alignment=TA_RIGHT,
    )
    styles["metric_label"] = ParagraphStyle(
        "MetricLabel",
        parent=styles["small"],
        fontSize=8,
        leading=10,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
    )
    styles["metric_value"] = ParagraphStyle(
        "MetricValue",
        parent=styles["body"],
        fontName=FONT_BOLD,
        fontSize=14,
        leading=17,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    return styles


def styled_table(
    data: list[list[Any]],
    col_widths: list[float],
    *,
    header: bool = True,
    repeat_rows: int = 1,
    long: bool = False,
    row_backgrounds: bool = True,
) -> Table:
    cls = LongTable if long else Table
    table = cls(data, colWidths=col_widths, repeatRows=repeat_rows if header else 0, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
            ]
        )
    if row_backgrounds:
        first = 1 if header else 0
        for row in range(first, len(data)):
            if (row - first) % 2:
                commands.append(("BACKGROUND", (0, row), (-1, row), PALE_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def metric_cards(items: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cells = [
        [[rich(f"<b>{html.escape(value)}</b>", styles["metric_value"])], [p(label, styles["metric_label"])]]
        for label, value in items
    ]
    row = [[Table(cell, colWidths=[1.48 * inch], rowHeights=[0.3 * inch, 0.25 * inch]) for cell in cells]]
    table = Table(row, colWidths=[1.55 * inch] * len(items), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, WHITE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_annual_table(
    ticker: str,
    result: BacktestResult,
    starting_capital: float,
) -> pd.DataFrame:
    trades = result.trades.copy()
    columns = [
        "ticker",
        "year",
        "year_start_capital",
        "year_end_capital",
        "year_return",
        "transactions",
        "wins",
        "win_rate",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    trades["year"] = pd.to_datetime(trades["ex_date"]).dt.year
    capital = float(starting_capital)
    rows: list[dict[str, Any]] = []
    for year, group in trades.groupby("year", sort=True):
        returns = group["trade_return"].astype(float)
        year_return = float((1.0 + returns).prod() - 1.0)
        ending = capital * (1.0 + year_return)
        wins = int((returns > 0).sum())
        rows.append(
            {
                "ticker": ticker,
                "year": int(year),
                "year_start_capital": capital,
                "year_end_capital": ending,
                "year_return": year_return,
                "transactions": len(group),
                "wins": wins,
                "win_rate": wins / len(group),
            }
        )
        capital = ending
    return pd.DataFrame(rows, columns=columns)


def create_equity_chart(
    fixed: BacktestResult,
    optimized: BacktestResult,
    selected_buy: int,
    selected_sell: int,
    ticker: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    if not fixed.equity.empty:
        ax.plot(
            pd.to_datetime(fixed.equity["date"]),
            fixed.equity["equity"],
            label="Fixed +2/+8",
            color="#6B7280",
            linewidth=1.7,
        )
    if not optimized.equity.empty:
        ax.plot(
            pd.to_datetime(optimized.equity["date"]),
            optimized.equity["equity"],
            label=f"Selected +{selected_buy}/+{selected_sell}",
            color="#1F77B4",
            linewidth=2.0,
        )
    ax.axhline(float(fixed.summary["starting_capital"]), color="#AAB2BA", linewidth=0.8, linestyle="--")
    ax.set_title(f"{ticker}: 2026 YTD equity, out-of-sample", fontweight="bold")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_fund_report(
    instrument: InstrumentData,
    args: argparse.Namespace,
    chart_dir: Path,
) -> FundReport | None:
    start = pd.Timestamp(args.start).normalize()
    requested_end = pd.Timestamp(args.end).normalize()
    period_end = min(requested_end, pd.Timestamp(instrument.prices["Date"].max()))
    train_end = min(pd.Timestamp(args.train_end).normalize(), period_end)
    test_start = pd.Timestamp(args.test_start).normalize()
    if period_end < start or test_start > period_end:
        return None

    fixed_full = backtest(
        instrument,
        args.fixed_buy_after,
        args.fixed_sell_after,
        start,
        period_end,
        args.capital,
        args.buy_price,
        args.sell_price,
    )
    grid, selected = optimize(
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
    selected_buy = int(selected["buy_after_sessions"])
    selected_sell = int(selected["sell_after_sessions"])
    fixed_test = backtest(
        instrument,
        args.fixed_buy_after,
        args.fixed_sell_after,
        test_start,
        period_end,
        args.capital,
        args.buy_price,
        args.sell_price,
    )
    optimized_test = backtest(
        instrument,
        selected_buy,
        selected_sell,
        test_start,
        period_end,
        args.capital,
        args.buy_price,
        args.sell_price,
    )
    annual = build_annual_table(instrument.ticker, fixed_full, args.capital)

    equity_chart_path = chart_dir / f"{instrument.ticker.lower()}_oos_equity.png"
    create_equity_chart(
        fixed_test,
        optimized_test,
        selected_buy,
        selected_sell,
        instrument.ticker,
        equity_chart_path,
    )
    return FundReport(
        instrument=instrument,
        period_end=period_end,
        fixed_full=fixed_full,
        optimization_grid=grid,
        selected_training=selected,
        fixed_test=fixed_test,
        optimized_test=optimized_test,
        annual=annual,
        equity_chart_path=equity_chart_path,
    )


def header_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, height = letter
    if document.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(0.65 * inch, height - 0.52 * inch, width - 0.65 * inch, height - 0.52 * inch)
        canvas.setFont(FONT_BOLD, 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(0.65 * inch, height - 0.4 * inch, "EX-DATE STRATEGY BACKTEST")
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(0.65 * inch, 0.38 * inch, "Historical simulation - distributions, taxes, fees, and spreads excluded")
    canvas.drawRightString(width - 0.65 * inch, 0.38 * inch, f"Page {document.page}")
    canvas.restoreState()


def assumptions_table(
    reports: list[FundReport],
    args: argparse.Namespace,
    styles: dict[str, ParagraphStyle],
) -> Table:
    shifts = sorted({r.instrument.date_shift_days for r in reports})
    end = max(r.period_end for r in reports)
    rows = [
        [p("Parameter", styles["table_header"]), p("Applied definition", styles["table_header"])],
        [p("Period", styles["table_left"]), p(f"{args.start} through {end.date().isoformat()} (latest available session)", styles["table_left"])],
        [p("Starting capital", styles["table_left"]), p(f"{money(args.capital)} independently for each fund", styles["table_left"])],
        [p("Fixed rule", styles["table_left"]), p(f"Buy +{args.fixed_buy_after}; sell +{args.fixed_sell_after} trading sessions from ex-date", styles["table_left"])],
        [p("Optimization", styles["table_left"]), p(f"Buy +{args.buy_min}..+{args.buy_max}; sell +{args.sell_min}..+{args.sell_max}; maximize {args.objective.replace('_', ' ')}", styles["table_left"])],
        [p("Validation", styles["table_left"]), p(f"Train through {args.train_end}; evaluate from {args.test_start}", styles["table_left"])],
        [p("Execution", styles["table_left"]), p(f"Buy at {args.buy_price}; sell at {args.sell_price}; fractional shares; fully compounded", styles["table_left"])],
        [p("Excluded", styles["table_left"]), p("Distributions, taxes, commissions, bid-ask spread, and slippage", styles["table_left"])],
        [p("Date normalization", styles["table_left"]), p(f"Automatic calendar correction selected: {', '.join(f'{x:+d} day' for x in shifts)}", styles["table_left"])],
    ]
    return styled_table(rows, [1.35 * inch, 5.45 * inch])


def executive_summary_table(
    reports: list[FundReport],
    styles: dict[str, ParagraphStyle],
) -> Table:
    header = ["Fund", "Selected rule", "Fixed full return", "Fixed 2026", "Selected 2026", "2026 difference"]
    rows: list[list[Any]] = [[p(x, styles["table_header"]) for x in header]]
    for report in reports:
        buy = int(report.selected_training["buy_after_sessions"])
        sell = int(report.selected_training["sell_after_sessions"])
        delta = float(report.optimized_test.summary["total_return"]) - float(report.fixed_test.summary["total_return"])
        rows.append(
            [
                rich(f"<b>{report.instrument.ticker}</b>", styles["table_cell"]),
                p(f"+{buy}/+{sell}", styles["table_cell"]),
                p(percent(report.fixed_full.summary["total_return"], signed=True), styles["table_right"]),
                p(percent(report.fixed_test.summary["total_return"], signed=True), styles["table_right"]),
                p(percent(report.optimized_test.summary["total_return"], signed=True), styles["table_right"]),
                p(percent(delta, signed=True), styles["table_right"]),
            ]
        )
    return styled_table(rows, [0.65 * inch, 0.9 * inch, 1.2 * inch, 1.05 * inch, 1.15 * inch, 1.15 * inch])


def overall_results_table(report: FundReport, styles: dict[str, ParagraphStyle]) -> Table:
    s = report.fixed_full.summary
    rows = [
        [p("Metric", styles["table_header"]), p("Result", styles["table_header"]), p("Metric", styles["table_header"]), p("Result", styles["table_header"])],
        [p("Complete trades", styles["table_left"]), p(integer(s["trades"]), styles["table_right"]), p("Ending capital", styles["table_left"]), p(money(s["ending_capital"]), styles["table_right"])],
        [p("Total return", styles["table_left"]), p(percent(s["total_return"], signed=True), styles["table_right"]), p("Net profit", styles["table_left"]), p(money(s["profit"], signed=True), styles["table_right"])],
        [p("CAGR", styles["table_left"]), p(percent(s["cagr"], signed=True), styles["table_right"]), p("Win rate", styles["table_left"]), p(percent(s["win_rate"]), styles["table_right"])],
        [p("Maximum drawdown", styles["table_left"]), p(percent(s["max_drawdown"], signed=True), styles["table_right"]), p("Average trade", styles["table_left"]), p(percent(s["average_trade_return"], signed=True), styles["table_right"])],
        [p("Price-only buy-and-hold", styles["table_left"]), p(percent(s["buy_hold_price_return"], signed=True), styles["table_right"]), p("Invested sessions", styles["table_left"]), p(percent(s["exposure_fraction"]), styles["table_right"])],
    ]
    return styled_table(rows, [1.5 * inch, 0.9 * inch, 1.55 * inch, 0.95 * inch])


def top_rules_table(report: FundReport, top_n: int, styles: dict[str, ParagraphStyle]) -> Table:
    top = report.optimization_grid.head(top_n)
    header = ["Rank", "Buy", "Sell", "Hold", "Train return", "Ending capital", "Win rate", "Max drawdown"]
    rows: list[list[Any]] = [[p(x, styles["table_header"]) for x in header]]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        rows.append(
            [
                p(rank, styles["table_cell"]),
                p(f"+{int(row['buy_after_sessions'])}", styles["table_cell"]),
                p(f"+{int(row['sell_after_sessions'])}", styles["table_cell"]),
                p(f"{int(row['sell_after_sessions'] - row['buy_after_sessions'])}", styles["table_cell"]),
                p(percent(row["total_return"], signed=True), styles["table_right"]),
                p(money(row["ending_capital"]), styles["table_right"]),
                p(percent(row["win_rate"]), styles["table_right"]),
                p(percent(row["max_drawdown"], signed=True), styles["table_right"]),
            ]
        )
    return styled_table(
        rows,
        [0.42 * inch, 0.46 * inch, 0.46 * inch, 0.46 * inch, 0.9 * inch, 1.08 * inch, 0.75 * inch, 0.95 * inch],
    )


def annual_results_table(report: FundReport, styles: dict[str, ParagraphStyle]) -> Table:
    header = ["Year", "Trades", "Start capital", "End capital", "Return", "Wins", "Win rate"]
    rows: list[list[Any]] = [[p(x, styles["table_header"]) for x in header]]
    for row in report.annual.itertuples(index=False):
        label = f"{row.year} YTD" if row.year == report.period_end.year else str(row.year)
        rows.append(
            [
                p(label, styles["table_left"]),
                p(integer(row.transactions), styles["table_cell"]),
                p(money(row.year_start_capital), styles["table_right"]),
                p(money(row.year_end_capital), styles["table_right"]),
                p(percent(row.year_return, signed=True), styles["table_right"]),
                p(integer(row.wins), styles["table_cell"]),
                p(percent(row.win_rate), styles["table_right"]),
            ]
        )
    return styled_table(rows, [0.72 * inch, 0.55 * inch, 1.12 * inch, 1.12 * inch, 0.78 * inch, 0.52 * inch, 0.78 * inch])


def oos_comparison_table(report: FundReport, styles: dict[str, ParagraphStyle]) -> Table:
    buy = int(report.selected_training["buy_after_sessions"])
    sell = int(report.selected_training["sell_after_sessions"])
    rows = [[p(x, styles["table_header"]) for x in ["Strategy", "Trades", "Ending capital", "Return", "Win rate", "Max drawdown"]]]
    for label, result in [
        (f"Fixed +2/+8", report.fixed_test),
        (f"Selected +{buy}/+{sell}", report.optimized_test),
    ]:
        s = result.summary
        rows.append(
            [
                p(label, styles["table_left"]),
                p(integer(s["trades"]), styles["table_cell"]),
                p(money(s["ending_capital"]), styles["table_right"]),
                p(percent(s["total_return"], signed=True), styles["table_right"]),
                p(percent(s["win_rate"]), styles["table_right"]),
                p(percent(s["max_drawdown"], signed=True), styles["table_right"]),
            ]
        )
    return styled_table(rows, [1.25 * inch, 0.62 * inch, 1.2 * inch, 0.86 * inch, 0.86 * inch, 1.0 * inch])


def transactions_table(report: FundReport, styles: dict[str, ParagraphStyle]) -> LongTable:
    header = ["Ex-date", "Buy date", "Sell date", "Buy", "Sell", "Return", "P&L", "Capital after"]
    rows: list[list[Any]] = [[p(x, styles["table_header"]) for x in header]]
    for row in report.optimized_test.trades.itertuples(index=False):
        rows.append(
            [
                p(short_date(row.ex_date), styles["table_left"]),
                p(short_date(row.buy_date), styles["table_left"]),
                p(short_date(row.sell_date), styles["table_left"]),
                p(money(row.buy_price), styles["table_right"]),
                p(money(row.sell_price), styles["table_right"]),
                p(percent(row.trade_return, signed=True), styles["table_right"]),
                p(money(row.pnl, signed=True), styles["table_right"]),
                p(money(row.capital_after), styles["table_right"]),
            ]
        )
    return styled_table(
        rows,
        [0.95 * inch, 0.95 * inch, 0.95 * inch, 0.63 * inch, 0.63 * inch, 0.72 * inch, 0.78 * inch, 0.97 * inch],
        long=True,
    )


def fund_story(report: FundReport, args: argparse.Namespace, styles: dict[str, ParagraphStyle]) -> list[Any]:
    ticker = report.instrument.ticker
    selected_buy = int(report.selected_training["buy_after_sessions"])
    selected_sell = int(report.selected_training["sell_after_sessions"])
    delta = float(report.optimized_test.summary["total_return"]) - float(report.fixed_test.summary["total_return"])
    if delta > 0.005:
        validation_text = (
            f"The selected +{selected_buy}/+{selected_sell} rule improved 2026 YTD return by "
            f"{percent(delta)} relative to the fixed +2/+8 rule."
        )
    elif delta >= 0:
        validation_text = (
            f"The selected +{selected_buy}/+{selected_sell} rule was only marginally better in 2026 YTD "
            f"({percent(delta)} difference)."
        )
    else:
        validation_text = (
            f"The selected +{selected_buy}/+{selected_sell} rule underperformed the fixed +2/+8 rule "
            f"by {percent(abs(delta))} in 2026 YTD, illustrating optimization risk."
        )

    story: list[Any] = [
        p(f"{ticker} backtest report", styles["h1"]),
        p(
            f"Fixed-rule period: {args.start} through {report.period_end.date().isoformat()}. "
            f"Training period ends {args.train_end}; 2026 is evaluated separately.",
            styles["subtitle"],
        ),
        metric_cards(
            [
                ("Fixed ending capital", money(report.fixed_full.summary["ending_capital"])),
                ("Fixed total return", percent(report.fixed_full.summary["total_return"], signed=True)),
                ("Selected rule", f"+{selected_buy} / +{selected_sell}"),
                ("Selected 2026 return", percent(report.optimized_test.summary["total_return"], signed=True)),
            ],
            styles,
        ),
        Spacer(1, 8),
        p("1. Fixed +2/+8 strategy", styles["h2"]),
        p(
            "Each completed transaction invests the full available portfolio at the selected buy price. "
            "Fractional shares are allowed, and sale proceeds become the capital for the next transaction. "
            "Raw prices are used, so distributions are not credited.",
            styles["body"],
        ),
        overall_results_table(report, styles),
        p("2. Optimization on 2024-2025", styles["h2"]),
        p(
            f"The grid tests all valid combinations from buy +{args.buy_min}..+{args.buy_max} and "
            f"sell +{args.sell_min}..+{args.sell_max}. Rules are ranked by "
            f"{args.objective.replace('_', ' ')}. All candidates use a common set of ex-dates so the "
            "ranking is not improved by selectively omitting difficult transactions.",
            styles["body"],
        ),
        top_rules_table(report, args.top_n, styles),
        rich(
            f"The highest-ranked training rule is <b>Buy +{selected_buy} / Sell +{selected_sell}</b>, "
            f"with a compounded training return of <b>{percent(report.selected_training['total_return'], signed=True)}</b> "
            f"and ending capital of <b>{money(report.selected_training['ending_capital'])}</b>.",
            styles["callout"],
        ),
        p("3. Annual fixed-rule breakdown", styles["h2"]),
        annual_results_table(report, styles),
        PageBreak(),
        p(f"{ticker}: 2026 out-of-sample validation", styles["h1"]),
        p(
            "The optimized offsets were selected without using 2026 returns. Both strategies below restart "
            f"with {money(args.capital)} on {args.test_start}, making their YTD results directly comparable.",
            styles["body"],
        ),
        oos_comparison_table(report, styles),
        p(validation_text, styles["callout"]),
        Image(str(report.equity_chart_path), width=6.85 * inch, height=2.42 * inch),
        p("2026 selected-rule transactions", styles["h2"]),
        transactions_table(report, styles),
        Spacer(1, 5),
        p(
            f"2026 result: {integer(report.optimized_test.summary['wins'])} winning and "
            f"{integer(report.optimized_test.summary['losses'])} losing transactions, with "
            f"{percent(report.optimized_test.summary['total_return'], signed=True)} compounded return and "
            f"{money(report.optimized_test.summary['profit'], signed=True)} profit.",
            styles["body"],
        ),
    ]
    return story


def build_pdf(
    reports: list[FundReport],
    output: Path,
    args: argparse.Namespace,
) -> None:
    register_fonts()
    styles = make_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.62 * inch,
        title=args.title,
        author=args.author,
        subject="Ex-date trading strategy backtest and optimization",
        allowSplitting=1,
    )

    tickers = ", ".join(report.instrument.ticker for report in reports)
    story: list[Any] = [
        Spacer(1, 0.55 * inch),
        p(args.title, styles["title"]),
        p(
            f"Fixed +2/+8 strategy and constrained offset optimization for {tickers}",
            styles["subtitle"],
        ),
        Spacer(1, 0.12 * inch),
        p("Executive summary", styles["h1"]),
        p(
            "This report applies one consistent methodology to every complete price/ex-date pair in the "
            "input archive. Each ticker is modeled independently with the same starting capital. The "
            "optimization result is selected on 2024-2025 and then tested on 2026 YTD.",
            styles["body"],
        ),
        executive_summary_table(reports, styles),
        Spacer(1, 12),
        p("Applied assumptions", styles["h2"]),
        assumptions_table(reports, args, styles),
        p(
            "Interpretation: a positive 2026 difference supports the selected rule only for this limited "
            "out-of-sample window. It is not evidence of a persistent market anomaly. The price-only "
            "simulation omits distributions and all implementation costs.",
            styles["callout"],
        ),
        Spacer(1, 0.15 * inch),
        p(f"Generated {date.today().isoformat()}", styles["small"]),
        PageBreak(),
    ]

    for index, report in enumerate(reports):
        story.extend(fund_story(report, args, styles))
        if index != len(reports) - 1:
            story.append(PageBreak())

    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PDF report from the ex-date backtest ZIP or directory."
    )
    parser.add_argument("input", type=Path, help="Original ZIP archive or extracted data directory")
    parser.add_argument("--output", type=Path, default=Path("exdate_backtest_report.pdf"))
    parser.add_argument("--ticker", action="append", help="Ticker to include; repeat for multiple tickers")
    parser.add_argument("--capital", type=float, default=20_000.0)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--fixed-buy-after", type=int, default=2)
    parser.add_argument("--fixed-sell-after", type=int, default=8)
    parser.add_argument("--buy-min", type=int, default=1)
    parser.add_argument("--buy-max", type=int, default=3)
    parser.add_argument("--sell-min", type=int, default=4)
    parser.add_argument("--sell-max", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--objective",
        choices=["ending_capital", "total_return", "cagr", "trade_sharpe", "return_over_drawdown", "win_rate"],
        default="total_return",
    )
    parser.add_argument("--buy-price", choices=["Open", "Close"], default="Close")
    parser.add_argument("--sell-price", choices=["Open", "Close"], default="Close")
    parser.add_argument("--date-shift-days", choices=["auto", "-1", "0", "1"], default="auto")
    parser.add_argument("--title", default="Ex-Date Strategy Backtest Report")
    parser.add_argument("--author", default="Boris Galitsky")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    if end < start:
        raise ValueError("--end must not precede --start")
    if not (start <= train_end < test_start <= end):
        raise ValueError("Require start <= train-end < test-start <= end")
    if args.capital <= 0:
        raise ValueError("--capital must be positive")
    if args.buy_min < 0 or args.buy_max < args.buy_min:
        raise ValueError("Require 0 <= --buy-min <= --buy-max")
    if args.sell_min < 1 or args.sell_max < args.sell_min:
        raise ValueError("Require 1 <= --sell-min <= --sell-max")
    if args.sell_max <= args.buy_min:
        raise ValueError("At least one sell offset must be later than a buy offset")
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    if args.fixed_buy_after < 0 or args.fixed_sell_after <= args.fixed_buy_after:
        raise ValueError("Require 0 <= fixed buy offset < fixed sell offset")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    instruments = load_instruments(args.input, date_shift=args.date_shift_days)
    if args.ticker:
        selected_tickers = {value.upper() for value in args.ticker}
        missing = sorted(selected_tickers.difference(instruments))
        if missing:
            raise ValueError(f"Ticker(s) not found: {', '.join(missing)}")
        instruments = {ticker: item for ticker, item in instruments.items() if ticker in selected_tickers}

    with tempfile.TemporaryDirectory(prefix="exdate_report_") as temporary:
        chart_dir = Path(temporary)
        reports: list[FundReport] = []
        for ticker, instrument in sorted(instruments.items()):
            print(f"Analyzing {ticker}...", flush=True)
            prepared = prepare_fund_report(instrument, args, chart_dir)
            if prepared is not None:
                reports.append(prepared)
        if not reports:
            raise ValueError("No ticker has data in the requested train/test period")
        build_pdf(reports, args.output, args)

    print(f"Report written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
