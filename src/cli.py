from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from pipeline import build_dataset, convert_news_json, fetch_news, run_gru_training, run_lstm_training, save_torch_artifact


app = typer.Typer(help="Financial sentiment analysis pipeline.")


@app.command("fetch-news")
def fetch_news_command(
    query: str = typer.Option(..., help="NewsAPI query string."),
    from_date: str = typer.Option(..., help="Start date in YYYY-MM-DD format."),
    to_date: str = typer.Option(..., help="End date in YYYY-MM-DD format."),
) -> None:
    output_path = fetch_news(query=query, from_date=from_date, to_date=to_date)
    print(f"[green]Saved raw news to[/green] {output_path}")


@app.command("convert-news-json")
def convert_news_json_command(
    json_path: Path = typer.Option(..., exists=True, readable=True, help="Path to a raw NewsAPI JSON file."),
    output_path: Path | None = typer.Option(None, help="Optional output CSV path."),
) -> None:
    csv_path = convert_news_json(json_path=json_path, output_path=output_path)
    print(f"[green]Saved flattened news CSV to[/green] {csv_path}")


@app.command("build-dataset")
def build_dataset_command(
    ticker: str = typer.Option(..., help="Stock ticker symbol, for example AAPL."),
    from_date: str = typer.Option(..., help="Start date in YYYY-MM-DD format."),
    to_date: str = typer.Option(..., help="End date in YYYY-MM-DD format."),
    news_path: Path = typer.Option(..., exists=True, readable=True, help="Path to a news CSV."),
) -> None:
    output_path = build_dataset(ticker=ticker, from_date=from_date, to_date=to_date, news_path=news_path)
    print(f"[green]Saved processed dataset to[/green] {output_path}")


@app.command("train-lstm")
def train_lstm_command(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Path to processed dataset CSV."),
) -> None:
    result = run_lstm_training(dataset)
    artifact = save_torch_artifact(result.model, "hybrid_lstm.pt")
    print(f"[cyan]LSTM accuracy:[/cyan] {result.accuracy:.4f}")
    print(f"[green]Saved model to[/green] {artifact}")
    print(result.report)


@app.command("train-gru")
def train_gru_command(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Path to processed dataset CSV."),
) -> None:
    result = run_gru_training(dataset)
    artifact = save_torch_artifact(result.model, "hybrid_gru.pt")
    print(f"[cyan]GRU accuracy:[/cyan] {result.accuracy:.4f}")
    print(f"[green]Saved model to[/green] {artifact}")
    print(result.report)


if __name__ == "__main__":
    app()
