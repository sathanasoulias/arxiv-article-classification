"""Single streaming pass over the arXiv snapshot, collecting EDA statistics.

The file is 5.65 GB of JSON Lines, so it is read line by line rather than loaded.
Counts are written to reports/eda_stats.json for plotting.
"""

import json
import re
import sys
from collections import Counter

from config import DATA_FILE, REPORTS, archive_of, group_of

YEAR = re.compile(r"\b(19|20)\d{2}\b")


def submission_year(rec):
    versions = rec.get("versions")
    if versions:
        m = YEAR.search(versions[0].get("created", ""))
        if m:
            return int(m.group(0))
    update = rec.get("update_date") or ""
    return int(update[:4]) if update[:4].isdigit() else None


def main():
    total = 0
    empty_abstract = 0
    abstract_words = Counter()      # word count -> frequency
    labels_per_paper = Counter()
    primary_archive = Counter()
    primary_group = Counter()
    all_groups = Counter()          # counts every label, so sums above total
    per_year = Counter()
    cross_group = 0                 # papers whose labels span >1 group

    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            abstract = (rec.get("abstract") or "").strip()
            if not abstract:
                empty_abstract += 1
                continue

            abstract_words[len(abstract.split())] += 1

            cats = (rec.get("categories") or "").split()
            if not cats:
                continue

            labels_per_paper[len(cats)] += 1
            primary_archive[archive_of(cats[0])] += 1
            primary_group[group_of(cats[0])] += 1

            groups = {group_of(c) for c in cats}
            all_groups.update(groups)
            if len(groups) > 1:
                cross_group += 1

            year = submission_year(rec)
            if year:
                per_year[year] += 1

            if total % 250_000 == 0:
                print(f"  {total:,} records...", flush=True)

    stats = {
        "total_records": total,
        "empty_abstract": empty_abstract,
        "cross_group_papers": cross_group,
        "abstract_words": dict(abstract_words),
        "labels_per_paper": dict(labels_per_paper),
        "primary_archive": dict(primary_archive),
        "primary_group": dict(primary_group),
        "all_groups": dict(all_groups),
        "per_year": dict(per_year),
    }

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "eda_stats.json"
    out.write_text(json.dumps(stats), encoding="utf-8")
    print(f"\nWrote {out}")

    # Console summary
    def pct(n):
        return f"{100 * n / total:.1f}%"

    words = sorted(abstract_words.elements())
    n = len(words)
    print(f"\nTotal records:        {total:,}")
    print(f"Empty abstracts:      {empty_abstract:,} ({pct(empty_abstract)})")
    print(f"Cross-group papers:   {cross_group:,} ({pct(cross_group)})")
    print("\nAbstract length (words):")
    for label, i in [("min", 0), ("p25", n // 4), ("median", n // 2),
                     ("p75", 3 * n // 4), ("p95", int(0.95 * n)), ("max", n - 1)]:
        print(f"  {label:7} {words[i]:,}")

    print("\nLabels per paper:")
    for k in sorted(labels_per_paper)[:8]:
        print(f"  {k}: {labels_per_paper[k]:,} ({pct(labels_per_paper[k])})")

    print(f"\nPrimary group ({len(primary_group)} classes):")
    for g, c in primary_group.most_common():
        print(f"  {g:22} {c:>9,}  {pct(c)}")

    print(f"\nPrimary archive ({len(primary_archive)} classes) - top 15:")
    for a, c in primary_archive.most_common(15):
        print(f"  {a:22} {c:>9,}  {pct(c)}")

    print("\nPapers per year (last 12):")
    for y in sorted(per_year)[-12:]:
        print(f"  {y}: {per_year[y]:,}")


if __name__ == "__main__":
    sys.exit(main())
