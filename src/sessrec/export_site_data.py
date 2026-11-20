"""Copy generated results into the site folder."""

from __future__ import annotations

import json
import shutil

from sessrec.glossary import GLOSSARY
from sessrec.paths import (
    CURVES_FILE,
    DATA_STATS_FILE,
    DEMO_USERS_FILE,
    DOCS_DATA,
    HISTORY_LENGTH_FILE,
    LATENCY_FILE,
    METRICS_FILE,
    METRICS_ID_ONLY_FILE,
    SETTINGS_FILE,
    SLICES_FILE,
)


def export() -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    for source in (
        METRICS_FILE,
        METRICS_ID_ONLY_FILE,
        CURVES_FILE,
        LATENCY_FILE,
        HISTORY_LENGTH_FILE,
        SLICES_FILE,
        DEMO_USERS_FILE,
        DATA_STATS_FILE,
        SETTINGS_FILE,
    ):
        if source.exists():
            shutil.copyfile(source, DOCS_DATA / source.name)
    (DOCS_DATA / "glossary.json").write_text(json.dumps(GLOSSARY, indent=2), encoding="utf-8")


def main() -> None:
    export()


if __name__ == "__main__":
    main()
