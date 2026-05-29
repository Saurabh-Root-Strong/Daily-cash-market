FPI NSDL Import Folder
======================

Drop NSDL FPI Investment Excel files here, then run:
  python -m src.cli import-fpi

Or use the "Import FPI Files Now" button on the FPI Capital Flow dashboard page.


HOW TO DOWNLOAD
---------------
1. Open: https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx
2. Select Report Type: "FPI Purchase/ Sales/ Net Investment"
3. Select the month/year range you want (download one month at a time, or multi-month)
4. Click Download → save the .xls or .xlsx file
5. Move the file to THIS folder (data/fpi_imports/)
6. Run: python -m src.cli import-fpi


FILE FORMAT EXPECTED
--------------------
The NSDL Excel must have:
  - A header row containing "Equity" and "Debt" category names
  - Sub-columns: Gross Purchase | Gross Sales | Net Investment (₹ Crores)
  - Categories: Equity, Debt, Debt-VRR, Hybrid, Others
  - Date column (first column): DD-MMM-YYYY or DD/MMM/YYYY format

The parser handles merged header cells automatically.
Re-importing the same file is safe — data is overwritten by (date, category).


RECOMMENDED COVERAGE
--------------------
Download the last 6 months of data for meaningful signals.
Each monthly file typically has 20-22 rows (one per trading day).

File naming: any name is fine — all .xls and .xlsx files in this folder are processed.
