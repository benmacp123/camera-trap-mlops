# Every target runs from committed files and public downloads; none needs
# an account. `make help` lists them.

.PHONY: help install install-models lock manifest subset fetch spike-models spike test lint format format-check

help: ## List the targets and what each does
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'


# uv manages the venv at venv/ (the portfolio's convention) rather than
# .venv; activate it before the other targets.
export UV_PROJECT_ENVIRONMENT = venv

install: ## Sync venv/ to uv.lock: core and dev tools only (what CI runs)
	uv sync --frozen

install-models: ## Also install the model stack (CPU torch, BioCLIP 2, DINOv2, SpeciesNet)
	uv sync --frozen --group baselines

lock: ## Re-resolve uv.lock after editing pyproject.toml
	uv lock

manifest: ## Rebuild data/manifest/ from LILA's Orinoquía metadata and MDv5a results (5 MB download)
	PYTHONPATH=src python -m cameratrap.data.manifest

subset: ## Rewrite the spike's image selection from the committed manifest
	PYTHONPATH=src python -m cameratrap.data.subset

fetch: ## Download the spike's images (about 15k, 11 GB) to data/images/ and record checksums
	PYTHONPATH=src python -m cameratrap.data.fetch

# The spike's model runs need `make install-models` and write to data/cache/;
# each resumes if stopped. SpeciesNet takes about 11 hours on CPU.
spike-models: ## Run MDv5a check, BioCLIP 2 embeddings and SpeciesNet over the spike images
	PYTHONPATH=src python -m cameratrap.arms.speciesnet_arm --detector-check
	PYTHONPATH=src python -m cameratrap.embed.bioclip
	PYTHONPATH=src python -m cameratrap.arms.speciesnet_arm

spike: ## Score the spike from the caches; writes results/spike/
	PYTHONPATH=src python -m cameratrap.report.spike

test: ## Run the tests with coverage
	PYTHONPATH=src python -m pytest -vv --cov=cameratrap --cov=cloud tests/

lint: ## Pylint the packages; docstrings everywhere, tests included
	PYTHONPATH=src python -m pylint --disable=R,C src/cameratrap src/cloud
	# Every module, class and function needs a docstring, tests included.
	PYTHONPATH=src python -m pylint --disable=all \
		--enable=missing-module-docstring,missing-class-docstring,missing-function-docstring \
		src tests

format: ## Format with black
	black src/ tests/

format-check: ## Check formatting without changing files
	black --check src/ tests/
