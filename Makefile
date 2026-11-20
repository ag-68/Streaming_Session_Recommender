.PHONY: data baselines train evaluate serve bench test site-data figures readme all

export PYTHONPATH := src
PY := python

data:
	$(PY) scripts/download_data.py
	$(PY) -m sessrec.data

baselines:
	$(PY) -m sessrec.baselines

train:
	$(PY) -m sessrec.train

evaluate:
	$(PY) -m sessrec.evaluate

serve:
	$(PY) -m uvicorn sessrec.serve.app:app --host 127.0.0.1 --port 8000

bench:
	$(PY) scripts/benchmark_latency.py

test:
	$(PY) -m pytest

site-data:
	$(PY) -m sessrec.export_site_data

figures:
	$(PY) -m sessrec.figures

readme:
	$(PY) -m sessrec.write_readme

all: data baselines train evaluate bench site-data figures readme test
