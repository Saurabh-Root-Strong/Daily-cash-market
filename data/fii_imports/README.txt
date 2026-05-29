FII Derivatives Statistics Import Folder
=========================================

Drop NSE FII Derivatives Statistics XLS files here, then run:
  python -m src.cli import-fii-stats

This is useful when NSE has not yet published the data via its API
(typical publication lag is 1-3 days after the trading date).


HOW TO DOWNLOAD
---------------
1. Open: https://www.nseindia.com/all-reports-derivatives
2. Find the section "F&O-FII derivatives statistics (xls)"
3. Click the orange download arrow for the date you want
4. Save the .xls file to THIS folder (data/fii_imports/)
5. Repeat for each missing date
6. Run: python -m src.cli import-fii-stats


FILE FORMAT EXPECTED
--------------------
The standard NSE file is named like:  ls_stats_26-May-2026.xls
The parser reads the date from row 0 (title row) automatically.
Any filename is fine — all .xls and .xlsx files in this folder are processed.

Re-importing the same file is safe — data is upserted (overwritten by date+category).


WHAT THIS DATA POWERS
---------------------
- Big Players (F&O) page -> FII Derivatives section
- Index Prediction -> Signals 9, 10, 11 (FII flow, cumulative, OI buildup)
