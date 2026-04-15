from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import settings


class NewsApiClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or settings.news_api_key
        self.base_url = (base_url or settings.news_api_base_url).rstrip("/")
        if not self.api_key:
            raise ValueError("Missing NEWS_API_KEY in environment configuration.")

    def fetch_everything(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = "en",
        page_size: int = 100,
        max_pages: int = 5,
        sort_by: str = "publishedAt",
    ) -> pd.DataFrame:
        url = f"{self.base_url}/everything"
        records: list[dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            response = requests.get(
                url,
                params={
                    "q": query,
                    "from": from_date,
                    "to": to_date,
                    "language": language,
                    "sortBy": sort_by,
                    "pageSize": page_size,
                    "page": page,
                    "apiKey": self.api_key,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            articles = payload.get("articles", [])
            if not articles:
                break

            for article in articles:
                source = article.get("source") or {}
                records.append(
                    {
                        "source_id": source.get("id"),
                        "source_name": source.get("name"),
                        "author": article.get("author"),
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article.get("url"),
                        "url_to_image": article.get("urlToImage"),
                        "published_at": article.get("publishedAt"),
                        "content": article.get("content"),
                        "query": query,
                    }
                )

            if len(articles) < page_size:
                break

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return frame

        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
        frame = frame.drop_duplicates(subset=["url"]).sort_values("published_at").reset_index(drop=True)
        return frame

    def save_news(self, news_frame: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        news_frame.to_csv(output_path, index=False)
        return output_path

