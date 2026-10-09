# Quality evaluation (see eval/run.py)
eval:            ## synthetic clips + the real clips in testdata/, compared with the baseline
	.venv/bin/python -m eval.run

eval-ci:         ## what CI runs
	.venv/bin/python -m eval.run --synthetic-only

eval-baseline:   ## accept the current numbers as the baseline
	.venv/bin/python -m eval.run --update-baseline

.PHONY: eval eval-ci eval-baseline
