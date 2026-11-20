"""Write the README from artifact files so the numbers are not typed by hand."""

from __future__ import annotations

import json

from sessrec.paths import (
    CURVES_FILE,
    DATA_STATS_FILE,
    LATENCY_FILE,
    METRICS_FILE,
    ROOT,
    SETTINGS_FILE,
)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _ms(value: float) -> str:
    return f"{value:.2f}"


def write_readme() -> None:
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    stats = json.loads(DATA_STATS_FILE.read_text(encoding="utf-8"))
    curves = json.loads(CURVES_FILE.read_text(encoding="utf-8"))
    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    latency = json.loads(LATENCY_FILE.read_text(encoding="utf-8")) if LATENCY_FILE.exists() else None
    models = metrics["models"]
    passed = "met" if metrics["pass_rule_met"] else "not met"
    rows = []
    for key, label in (
        ("popularity", "Popularity"),
        ("item_knn", "Item-to-item similarity"),
        ("sasrec", "SASRec"),
    ):
        row = models[key]
        rows.append(
            f"| {label} | {_fmt(row['ndcg@10'])} | {_fmt(row['hr@10'])} | {_fmt(row['coverage@10'])} |"
        )
    if latency:
        hardware = latency["hardware"]
        length_rows = []
        for length, marks in latency["by_history_length"].items():
            length_rows.append(
                f"| {length} | {_ms(marks['p50'])} | {_ms(marks['p95'])} | {_ms(marks['p99'])} |"
            )
        latency_section = f"""
The timed path is the full HTTP round trip on localhost: {latency["warmup"]} warm-up requests are discarded, then {latency["requests_per_length"]} requests are sent for each history length. p95 means 95% of those requests finished in this time or less.

| Films in the request | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
{chr(10).join(length_rows)}

Hardware: {hardware["cpu"]}, {hardware["cores"]} cores, {hardware["os"]}, Python {hardware["python"]}, torch {hardware["torch"]}. Torch threads used by the service: {latency.get("torch_threads")}.

The 50 ms line is a budget for this catalog of {stats["items"]} films on one machine. It is not evidence about a catalog of millions of titles. The measurement used port {latency.get("port", 8000)}.
"""
        headline_p95 = latency["by_history_length"].get("20", next(iter(latency["by_history_length"].values())))
        latency_headline = f"At 20 films of history, p50 is {_ms(headline_p95['p50'])} ms and p95 is {_ms(headline_p95['p95'])} ms."
    else:
        latency_section = "\nLatency has not been measured yet.\n"
        latency_headline = "Latency has not been measured yet."

    best = models["sasrec"]["ndcg@10"]
    other = max(models["popularity"]["ndcg@10"], models["item_knn"]["ndcg@10"])
    text = f"""# Next-Movie Recommender: Rating-aware Self-Attention

Prefer the hand-maintained README.md when publishing; this generator still emits the baseline three-method table from `artifacts/metrics.json`.

Three known methods guess the next film a viewer would rate 4 or 5 stars. The sequence model has to beat a popularity list. A stronger neighbor method is reported whether it wins or loses.

Live explanation: https://ag-68.github.io/Streaming_Session_Recommender/

![Metric comparison](docs/figures/metrics.png)

## Results on the test split

The hidden film is ranked against every other film the viewer has not already seen. Higher is better. These numbers are written from `artifacts/metrics.json`.

| Method | NDCG@10 | HR@10 | Coverage@10 |
|---|---:|---:|---:|
{chr(10).join(rows)}

The pass rule ({metrics["pass_rule"]}) is **{passed}**. SASRec's NDCG@10 is {_fmt(best)}. The better of the two simpler methods is {_fmt(other)}. {latency_headline}

This is not a reproduction of the scores in Kang and McAuley (2018). That paper counted every rating as a positive event, trained against one random wrong film, and ranked the answer among 100 random wrong films. This project keeps only 4- and 5-star ratings, trains against the whole catalog, and ranks against the whole catalog.

## What this is

A viewer liked *Heat* and then *Collateral*. The home row has to suggest a next film. Popularity offers the same widely liked films to everyone. Item-to-item similarity suggests films that share an audience. SASRec also reads the order. All three see the same history, may not repeat a film already in that history, and are scored by the same code.

The service returns five films from SASRec. If the history is empty, or every id is unknown, it returns the popularity list and says so.

## Quickstart

Unix:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data
make baselines
make train
make evaluate
make serve
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts/download_data.py
python -m sessrec.data
python -m sessrec.baselines
python -m sessrec.train
python -m sessrec.evaluate
python -m uvicorn sessrec.serve.app:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/docs.

```bash
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{{\\"history\\":[1,2,3],\\"k\\":5}}"
```

Replace those ids with raw MovieLens ids. If `artifacts/sasrec.pt` is already in the checkout, `make serve` does not need a new training run. A fresh training run can move the numbers a little. The files in `artifacts/` are the run behind this README.

## Method

```mermaid
flowchart LR
  A[ML-1M ratings and movies] --> B[keep 4 and 5 star ratings, sort by time, remap ids]
  B --> C[leave-one-out split]
  C --> D[popularity and item-to-item similarity]
  C --> E[train SASRec in PyTorch]
  D --> F[full ranking]
  E --> F
  E --> G[checkpoint]
  G --> H[FastAPI]
  F --> I[artifact JSON]
  H --> I
```

SASRec is the 2018 sequence model: a film embedding shared with the output scores, a position embedding, and causal attention blocks. It is trained with full-catalog cross-entropy. The paper used one random wrong film. Item-to-item similarity keeps the {settings["item_knn_neighbors"]} nearest films by cosine similarity of who liked them in the training sequences. Popularity ranks films by those same training counts.

| Setting | Value |
|---|---|
| max history | {settings["max_len"]} |
| embedding size | {settings["hidden_dim"]} |
| blocks | {settings["num_blocks"]} |
| attention heads | {settings["num_heads"]} |
| dropout | {settings["dropout"]} |
| learning rate | {settings["lr"]} |
| batch size | {settings["batch_size"]} |
| patience | {settings["patience"]} epochs without a validation gain |
| neighbors | {settings["item_knn_neighbors"]} |
| seed | {settings["seed"]} |
| best epoch | {curves["best_epoch"]} |

![Training](docs/figures/training.png)

## Evaluation

MovieLens records ratings, not plays. A rating of 4 or 5 is treated as a liked film. Ratings of 1, 2, and 3 are left out. Viewers with fewer than 5 liked films are dropped. After that filter this run has {stats["users"]} viewers, {stats["items"]} films, and {stats["positive_ratings"]} liked ratings. The raw file still has {stats["raw_ratings"]} ratings and {stats["raw_users"]} viewers.

Each viewer's liked films are ordered by timestamp, with the original file order breaking ties. The last film is the test question. The one before it is the validation question, used only to stop training. The test question is scored once.

NDCG@10 is 1 when the hidden film is in position 1, `1/log2(3)` when it is in position 2, and 0 when it is outside the top 10. HR@10 only asks whether it appears anywhere in the top 10. Coverage@10 is the share of films that reach at least one viewer's top 10.

## API

`GET /health` returns the status and model version.

`POST /recommend` accepts a history of raw MovieLens ids, oldest first, and a list size `k` from 1 to 50. The response has exactly `k` films, none of them already in the history. Unknown ids come back in `ignored_ids`. `strategy` is `sasrec` or `popularity_fallback`. `latency_ms` covers the model path after the request has been checked. The table above is the full HTTP round trip, which is a different measurement.

## Latency
{latency_section}
![Latency](docs/figures/latency.png)

## Tests

| Test | What a pass shows |
|---|---|
| `test_metrics.py` | The rank scores match hand-worked cases. |
| `test_data.py` | Order, the hidden-film rule, contiguous ids, and the 4-and-5-star filter. |
| `test_model.py` | The shape, the no-peeking rule, no NaN on padding, and a tiny sequence rule can be learned. |
| `test_baseline_comparison.py` | All three test scores exist, and SASRec is above popularity. |
| `test_api.py` | The HTTP contract, including the empty-history fallback. |
| `test_latency.py` | In-process p95 under 50 ms. |

```bash
python -m pytest
python -m pytest -m "not latency"
```

## What this does not show

A rating is not a finished play. Finding a hidden 5-star film is not the same as a happier viewer on a real home row. Each viewer's last film happened on a different date, so training can include one viewer's later activity while the test asks about another viewer's earlier film. That is data leakage: the model can learn something that would not have been known at the prediction time. The comparison also uses one dataset and one protocol. Item-to-item similarity is allowed to win. That result is part of the table above.

Not built here: genre features, a single global time cutoff, an online experiment, and a two-stage retriever for a much larger catalog.

## Where a later project could go

- Re-rank the five films with business rules.
- Add genre or other catalog text as extra evidence.
- Test the lists with real viewers instead of a hidden rating.

## Related project

[Causal Decisioning Multi Objective Orchestrator](https://github.com/ag-68/Causal_Decisioning_Multi_Objective_Orchestrator)

## Citations

- Sarwar, Karypis, Konstan, and Riedl. "Item-based Collaborative Filtering Recommendation Algorithms." WWW 2001.
- Kang and McAuley. "Self-Attentive Sequential Recommendation." ICDM 2018.
- Hidasi, Karatzoglou, Baltrunas, and Tikk. "Session-based Recommendations with Recurrent Neural Networks." 2016.
- Sun, Liu, Wu, Pei, Lin, Ou, and Jiang. "BERT4Rec." CIKM 2019.
- Covington, Adams, and Sargin. "Deep Neural Networks for YouTube Recommendations." RecSys 2016.
- Dacrema, Cremonesi, and Jannach. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches." RecSys 2019.
- Krichene and Rendle. "On Sampled Metrics for Item Recommendation." KDD 2020.
- Harper and Konstan. "The MovieLens Datasets: History and Context." 2015. Data: https://grouplens.org/datasets/movielens/1m/
- Ji, Sun, Zhang, and Li. "A Critical Study on Data Leakage in Recommender System Offline Evaluation." ACM TOIS 2023.

## License

TODO(human): choose a license for the code. The MovieLens files stay under the GroupLens terms and are not committed.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    write_readme()


if __name__ == "__main__":
    main()
