"""
Exit 0  → market is open today, proceed with fetch.
Exit 1  → market is closed today (weekend or NSE holiday), skip fetch.

Update NSE_HOLIDAYS each January with the new year's list from:
  https://www.nseindia.com/resources/exchange-communication-holidays
"""
import sys
import datetime

NSE_HOLIDAYS: set[datetime.date] = {
    # ── 2025 ──────────────────────────────────────────────────────────────────
    datetime.date(2025, 2, 26),   # Mahashivratri
    datetime.date(2025, 3, 14),   # Holi
    datetime.date(2025, 3, 31),   # Id-Ul-Fitr (Ramzan Eid)
    datetime.date(2025, 4, 14),   # Dr. Ambedkar Jayanti / Ram Navami
    datetime.date(2025, 4, 18),   # Good Friday
    datetime.date(2025, 5, 1),    # Maharashtra Day
    datetime.date(2025, 6, 7),    # Bakri Id (Eid ul-Adha)
    datetime.date(2025, 7, 6),    # Muharram
    datetime.date(2025, 8, 15),   # Independence Day
    datetime.date(2025, 8, 27),   # Ganesh Chaturthi
    datetime.date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    datetime.date(2025, 10, 21),  # Diwali — Laxmi Pujan
    datetime.date(2025, 10, 22),  # Diwali — Balipratipada
    datetime.date(2025, 11, 5),   # Prakash Gurpurb (Guru Nanak Jayanti)
    datetime.date(2025, 12, 25),  # Christmas

    # ── 2026 ──────────────────────────────────────────────────────────────────
    datetime.date(2026, 1, 26),   # Republic Day
    datetime.date(2026, 3, 3),    # Holi
    datetime.date(2026, 3, 20),   # Gudi Padwa / Ugadi
    datetime.date(2026, 4, 3),    # Good Friday
    datetime.date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    datetime.date(2026, 4, 22),   # Ram Navami
    datetime.date(2026, 5, 1),    # Maharashtra Day
    datetime.date(2026, 5, 27),   # Bakri Id (Eid ul-Adha, approximate)
    datetime.date(2026, 8, 15),   # Independence Day
    datetime.date(2026, 9, 16),   # Ganesh Chaturthi
    datetime.date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    datetime.date(2026, 10, 20),  # Dussehra (approximate)
    datetime.date(2026, 11, 9),   # Diwali — Laxmi Pujan (approximate)
    datetime.date(2026, 11, 10),  # Diwali — Balipratipada (approximate)
    datetime.date(2026, 11, 24),  # Guru Nanak Jayanti (approximate)
    datetime.date(2026, 12, 25),  # Christmas
}

today = datetime.date.today()

if today.weekday() >= 5:
    print(f"[market_check] Skipping — {today.strftime('%A %d %b %Y')} is a weekend")
    sys.exit(1)

if today in NSE_HOLIDAYS:
    print(f"[market_check] Skipping — {today.strftime('%d %b %Y')} is an NSE public holiday")
    sys.exit(1)

print(f"[market_check] Market open — proceeding with fetch for {today.strftime('%d %b %Y (%A)')}")
sys.exit(0)
