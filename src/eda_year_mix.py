"""Measure how the field mix shifted over time.

Cross-tabulates submission year against primary archive and primary group, so the
sampling plan can be built from the distribution of the years we actually keep
rather than from the all-time distribution.

Input:  the arXiv snapshot
Output: reports/year_mix.json
Usage:  python eda_year_mix.py
"""

import json
from collections import Counter, defaultdict

from config import DATA_FILE, REPORTS, archive_of, group_of
from eda_scan import submission_year


def main():
    by_year_group = defaultdict(Counter)
    by_year_archive = defaultdict(Counter)
    total = 0

    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cats = (rec.get("categories") or "").split()
            year = submission_year(rec)
            if not cats or not year:
                continue

            total += 1
            by_year_group[year][group_of(cats[0])] += 1
            by_year_archive[year][archive_of(cats[0])] += 1

            if total % 500_000 == 0:
                print(f"  {total:,} records...", flush=True)

    out = {
        "by_year_group": {str(y): dict(c) for y, c in by_year_group.items()},
        "by_year_archive": {str(y): dict(c) for y, c in by_year_archive.items()},
    }
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "year_mix.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"\nWrote {path}")

    # Compare the all-time mix against the recent-years mix.
    for label, years in [("ALL YEARS", None), ("2016+", range(2016, 2027)),
                         ("2020+", range(2020, 2027))]:
        agg = Counter()
        for y, c in by_year_group.items():
            if years is None or y in years:
                agg.update(c)
        n = sum(agg.values())
        print(f"\n{label}  ({n:,} papers)")
        for g, c in agg.most_common():
            print(f"  {g:10} {c:>9,}  {100 * c / n:5.1f}%")


if __name__ == "__main__":
    main()
