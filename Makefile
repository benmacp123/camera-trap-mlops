# Every target runs from committed files and public downloads; none needs
# an account. `make help` lists them.

.PHONY: help install install-models lock manifest subset fetch spike-models spike crop-index fetch-crops crops crop-sizes detector-report embed embed-check speciesnet-crops baselines hub-stage hub-upload fetch-embeddings test lint format format-check

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

# Phase 1. The index and the report need only committed files (the index
# also reads data/raw/ from `make manifest`); crops need the images.
crop-index: ## Rewrite the crop index: top 3 animal frames of every sequence, person frames dropped
	PYTHONPATH=src python -m cameratrap.data.crops index

fetch-crops: ## Download the indexed images (about 26k, 20 GB) and record checksums
	PYTHONPATH=src python -m cameratrap.data.fetch --selection data/manifest/orinoquia_crops.csv

crops: ## Cut the indexed crops to data/crops/ (resumes; needs fetch-crops)
	PYTHONPATH=src python -m cameratrap.data.crops write

crop-sizes: ## Record each stored crop's side length in data/manifest/ (needs crops)
	PYTHONPATH=src python -m cameratrap.data.crops sizes

detector-report: ## Detector misses per site and class; writes results/detector/
	PYTHONPATH=src python -m cameratrap.report.detector

# Phase 2 model runs: need `make install-models` and the crop store; each
# resumes if stopped. Together about 14 hours on this laptop's CPU.
embed: ## Embed every stored crop with BioCLIP 2 then DINOv2 (about 6 h) into data/cache/embeddings/
	PYTHONPATH=src python -m cameratrap.embed.backbones bioclip2
	PYTHONPATH=src python -m cameratrap.embed.backbones dinov2

embed-check: ## Re-embed 50 cached crops per backbone and compare; writes results/embeddings/
	PYTHONPATH=src python -m cameratrap.embed.backbones bioclip2 --check 50 --threads 4
	PYTHONPATH=src python -m cameratrap.embed.backbones dinov2 --check 50 --threads 4

speciesnet-crops: ## Run SpeciesNet over every crop-index frame, reusing the spike's answers (about 8.5 h)
	PYTHONPATH=src python -m cameratrap.arms.speciesnet_arm --crop-index

# Everything below needs only committed files and the caches, which a fresh
# clone gets from `make fetch-embeddings` without an account.
fetch-embeddings: ## Download the published caches (data/manifest/hub.json) into data/cache/; no account
	PYTHONPATH=src python -m cameratrap.embed.hub download

baselines: ## Zero-shot baselines per fold site from the caches; writes results/baselines/ and figures/
	PYTHONPATH=src python -m cameratrap.report.baselines

hub-stage: ## Write the files and dataset card to publish into data/hub_stage/
	PYTHONPATH=src python -m cameratrap.embed.hub stage

hub-upload: ## Owner only: upload data/hub_stage/ (REPO_ID=user/name, needs `hf auth login`)
	PYTHONPATH=src python -m cameratrap.embed.hub upload --repo-id $(REPO_ID)

test: ## Run the tests with coverage
	PYTHONPATH=src python -m pytest -vv --cov=cameratrap --cov=cloud tests/

# CI installs only core and dev, so pylint cannot import the model stack
# that embed and arms use; the same libraries tests/test_layout.py confines.
MODEL_LIBRARIES = torch,torchvision,open_clip,timm,speciesnet

lint: ## Pylint the packages; docstrings everywhere, tests included
	PYTHONPATH=src python -m pylint --disable=R,C --ignored-modules=$(MODEL_LIBRARIES) \
		src/cameratrap src/cloud
	# Every module, class and function needs a docstring, tests included.
	PYTHONPATH=src python -m pylint --disable=all \
		--enable=missing-module-docstring,missing-class-docstring,missing-function-docstring \
		src tests

format: ## Format with black
	black src/ tests/

format-check: ## Check formatting without changing files
	black --check src/ tests/
