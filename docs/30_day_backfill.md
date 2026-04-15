# 30-Day NewsAPI Backfill Plan

## Goal

Build a reproducible rolling news corpus for stock-movement experiments while staying within the NewsAPI Developer quota.

Recommended ticker set:

- `AAPL`
- `MSFT`
- `NVDA`
- `TSLA`
- `CVNA`
- `KO`
- `PG`
- `JNJ`

## Quota Model

Use NewsAPI `everything` with:

- `pageSize=100`
- `max_pages_per_query=1`
- one request per `ticker x date`

This keeps cost predictable.

- `8` tickers x `1` request/day = `8` requests per date
- `30` days x `8` tickers = `240` ticker-days

Because the free tier is limited, the 30-day corpus is filled progressively rather than in a single run.

## Collection Workflow

Use the project CLI:

```bash
financial-sentiment-2026 fetch-news --query "Apple OR Microsoft OR Nvidia" --from-date 2026-04-10 --to-date 2026-04-14
```

For the parent project collector, the important behavior to preserve is:

1. save one file per `date + ticker`
2. skip existing files on reruns
3. deduplicate articles by `url`
4. keep a manifest of requests used and files written

## How To Avoid Duplicate Fetches

The safest pattern is:

- save files using deterministic names like `YYYY-MM-DD_TICKER.csv`
- check whether that file already exists before making the API call
- skip existing ticker-day files

That means the next collection run fills missing ticker-days instead of re-fetching the same slices repeatedly.

## Best Operating Mode

### Phase 1

Backfill oldest missing dates first until the rolling 30-day window is complete.

### Phase 2

Once the rolling window is full:

- collect `yesterday` only
- maintain the rolling window daily
- expected request cost stays low

## Why This Helps Training

This gives the project:

- multi-ticker news coverage
- cleaner chronological splits
- enough structure to compare financial-only, sentiment-only, and combined models

