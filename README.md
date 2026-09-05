# Ex-Date Trading Strategy Backtester

This project backtests and optimizes short holding-period strategies around ETF ex-dividend dates. It also generates a polished PDF report containing assumptions, ranked trading rules, annual results, transaction tables, and equity charts.

The supplied dataset covers JEPI, JEPQ, PBP, QQQI, and QYLD from 2024 through August 31, 2026. The scripts also support additional funds, including ISPY, when matching price and ex-date files are added.

## Project files

```text
.
├── ispy_exdate_optimizer.py
├── generate_exdate_backtest_report.py
├── data for backtest.zip
├── exdate_backtest_report.pdf
└── README.md
```

| File | Purpose |
|---|---|
| `ispy_exdate_optimizer.py` | Loads the input data, runs the fixed strategy, searches the permitted buy/sell offsets, validates the selected rule on 2026 YTD, and exports detailed CSV results. |
| `generate_exdate_backtest_report.py` | Imports the optimizer functions and creates a combined PDF report for all selected funds. |
| `data for backtest.zip` | Price histories and ex-dividend-date lists for the supplied funds. |
| `exdate_backtest_report.pdf` | Example 11-page report generated from the supplied archive. |

## Strategy definition

The default fixed strategy is:

- Starting capital: **$20,000 per fund**.
- Buy: **2 trading sessions after** the ex-date.
- Sell: **8 trading sessions after** the ex-date.
- Execution: closing price on both transaction dates.
- Capital: fully reused and compounded after every completed trade.
- Fractional shares: allowed.
- Distributions: excluded.
- Taxes, fees, spreads, and slippage: ignored.
- Test period: January 1, 2024 through the latest available 2026 session.

An ex-date is session zero. For example, if the ex-date is Thursday, buy offset `+2` normally identifies the following Monday, assuming Friday and Monday are both present in the price dataset.

## Optimization and validation

The default parameter grid searches around the fixed 2/8 rule:

- Buy offsets: **+1, +2, and +3** trading sessions from the ex-date (`2 +/- 1`).
- Sell offsets: **+4 through +12** trading sessions from the ex-date (`8 +/- 4`).
- Constraint: the sell offset must be later than the buy offset.
- Default objective: highest compounded total return.
- Training sample: 2024-2025.
- Out-of-sample evaluation: 2026 YTD.

All candidate rules are evaluated using a common set of complete ex-date events. A candidate that would overlap the next trade is rejected because the same fully invested capital cannot support two positions simultaneously.

The separation between training and 2026 evaluation is important: the rule selected from 2024-2025 is not re-optimized using 2026 returns.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- ReportLab

Install the dependencies with:

```bash
python -m pip install numpy pandas matplotlib reportlab
```

## Input format

The input may be the original ZIP archive or a directory containing extracted files. Each ticker requires one price CSV and one ex-date file.

Example:

```text
JEPI (20260831000000000 _ 20231222000000000).csv
JEPI_ex_dividend_date.txt
```

### Price CSV

Required columns:

```csv
Date,Open,High,Low,Close,Volume
2024-01-02T21:00:00.000Z,55.09,55.10,54.81,54.81,"3,541,259"
```

The calculations use raw `Open` or `Close` values. Adjusted prices are not used because distributions are explicitly excluded.

### Ex-date file

```csv
ex_dividend_date
2024-02-01
2024-03-01
2024-04-01
```

The supplied price timestamps are one calendar day earlier than their actual U.S. trading sessions. Automatic date normalization detects and corrects this shift. For a conventional price file whose dates already identify the trading session, use `--date-shift-days 0` if automatic detection is not appropriate.

### Adding another ticker

To add ISPY or another fund, place both matching files in the ZIP or input directory:

```text
ISPY (date-range).csv
ISPY_ex_dividend_date.txt
```

The filename prefix before the first space or opening parenthesis is treated as the ticker.

## Running the optimizer

Run all supplied funds:

```bash
python ispy_exdate_optimizer.py "data for backtest.zip" \
  --output-dir backtest_results
```

Run only one fund:

```bash
python ispy_exdate_optimizer.py "data for backtest.zip" \
  --ticker QQQI \
  --output-dir qqqi_results
```

Use Open for entry and Close for exit:

```bash
python ispy_exdate_optimizer.py "data for backtest.zip" \
  --buy-price Open \
  --sell-price Close \
  --output-dir open_close_results
```

Change the optimization objective:

```bash
python ispy_exdate_optimizer.py "data for backtest.zip" \
  --objective return_over_drawdown \
  --output-dir risk_adjusted_results
```

Available objectives are:

- `ending_capital`
- `total_return`
- `cagr`
- `trade_sharpe`
- `return_over_drawdown`
- `win_rate`

Use `--no-optimize` to calculate only the fixed rule.

## Optimizer outputs

The optimizer writes the following CSV files:

| Output | Contents |
|---|---|
| `fixed_rule_summary.csv` | Overall fixed-rule metrics for every fund. |
| `fixed_rule_trades.csv` | One row per completed fixed-rule transaction. |
| `fixed_rule_equity_curves.csv` | Session-level portfolio value, drawdown, and investment status. |
| `fixed_rule_yearly_summary.csv` | Fixed-rule results grouped by calendar year. |
| `optimization_grid_2024_2025.csv` | Every admissible buy/sell combination and its training metrics. |
| `optimized_rules_training_summary.csv` | The selected training-period rule for each fund. |
| `optimized_rules_2026_oos_summary.csv` | Performance of each selected rule on 2026 YTD data. |
| `optimized_rules_2026_oos_trades.csv` | Transaction-level 2026 results for the selected rules. |
| `skipped_events.csv` | Ex-dates omitted because the required future sessions were unavailable. |

The principal summary metrics include ending capital, profit, total return, CAGR, price-only buy-and-hold return, maximum drawdown, win rate, average trade return, trade-level Sharpe ratio, and exposure fraction.

## Generating the PDF report

Keep `generate_exdate_backtest_report.py` and `ispy_exdate_optimizer.py` in the same directory. The report generator imports the tested data-loading, backtesting, and optimization functions from the optimizer.

Generate the combined report:

```bash
python generate_exdate_backtest_report.py "data for backtest.zip" \
  --output exdate_backtest_report.pdf
```

Generate a report for one fund:

```bash
python generate_exdate_backtest_report.py "data for backtest.zip" \
  --ticker QQQI \
  --output QQQI_backtest_report.pdf
```

Include several selected funds by repeating `--ticker`:

```bash
python generate_exdate_backtest_report.py "data for backtest.zip" \
  --ticker JEPI \
  --ticker JEPQ \
  --output JEPI_JEPQ_report.pdf
```

The PDF contains:

- A cross-fund executive summary.
- Complete assumptions and execution rules.
- Fixed 2/8 performance from 2024 through 2026 YTD.
- The five best 2024-2025 parameter combinations for each fund.
- Annual fixed-rule results.
- Fixed-versus-optimized 2026 comparisons.
- 2026 transaction-by-transaction tables.
- Out-of-sample equity charts.

## Example results

The following results were generated from the supplied archive through August 31, 2026. Each fund is an independent simulation starting with $20,000.

| Fund | Selected rule | Fixed 2/8 full-period return | Fixed 2026 return | Selected-rule 2026 return | 2026 difference |
|---|---:|---:|---:|---:|---:|
| JEPI | +3/+10 | +8.18% | -0.45% | +2.43% | +2.88% |
| JEPQ | +3/+9 | +25.46% | +3.34% | +6.72% | +3.38% |
| PBP | +3/+7 | +7.58% | +4.44% | +4.46% | +0.02% |
| QQQI | +1/+5 | -10.54% | +1.45% | -0.20% | -1.66% |
| QYLD | +3/+7 | +4.99% | +3.29% | +3.69% | +0.40% |

The optimized rule improved the limited 2026 result for four funds, but the improvement for PBP was negligible and the optimized QQQI rule underperformed the fixed strategy. These mixed results illustrate why the out-of-sample section should be reviewed separately from the stronger training-period results.

## Important limitations

- **Distributions are excluded.** These are income-oriented funds, so a price-only return is not the same as an investor's total return.
- The 2026 out-of-sample window contains only approximately seven or eight completed trades per fund.
- Selecting the best historical offsets creates overfitting risk even when a later validation period is reserved.
- The scripts assume that each CSV row represents one ordered dataset session after date normalization.
- OHLC data do not model intraday execution uncertainty.
- No taxes, commissions, spreads, liquidity constraints, or slippage are modeled.
- Fractional shares may not be available for every security or brokerage account.
- Results for the five funds must not be added together: each simulation independently starts with $20,000.
- Historical performance does not establish that the pattern will persist.

This software is for research and educational analysis and is not investment advice.

## Useful commands

Display all optimizer options:

```bash
python ispy_exdate_optimizer.py --help
```

Display all report-generator options:

```bash
python generate_exdate_backtest_report.py --help
```
