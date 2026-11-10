# Every target runs from committed files and public downloads; none needs
# an account. `make help` lists them.

.PHONY: help install install-models lock manifest subset fetch spike-models spike crop-index fetch-crops crops crop-sizes detector-report embed embed-check speciesnet-crops baselines challengers select-challenger freeze holdout-check holdout cnn-control control trust-report track gate-calibrate hub-stage hub-upload fetch-embeddings test lint format format-check

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

speciesnet-crops: ## Run SpeciesNet over every crop-index frame, reusing the spike's answers (about 13 h)
	PYTHONPATH=src python -m cameratrap.arms.speciesnet_arm --crop-index

# Everything below needs only committed files and the caches, which a fresh
# clone gets from `make fetch-embeddings` without an account.
fetch-embeddings: ## Download the published caches (data/manifest/hub.json) into data/cache/; no account
	PYTHONPATH=src python -m cameratrap.embed.hub download

baselines: ## Zero-shot baselines per fold site from the caches; writes results/baselines/ and figures/
	PYTHONPATH=src python -m cameratrap.report.baselines

challengers: ## Out-of-fold probes on BioCLIP 2 and DINOv2 embeddings per fold site; writes results/challengers/ and figures/
	PYTHONPATH=src python -m cameratrap.report.challengers

select-challenger: ## Choose the hold-out challenger by simulated gate pass rate on fold sites; writes results/selection/ and figures/
	PYTHONPATH=src python -m cameratrap.report.selection

freeze: ## Write and hash the hold-out freeze to config/holdout_freeze.json; refuses to overwrite one
	PYTHONPATH=src python -m cameratrap.report.holdout_freeze

holdout-check: ## Check the freeze, its inputs and the caches, without fitting or scoring anything
	PYTHONPATH=src python -m cameratrap.report.holdout --check

holdout: ## One shot: score the frozen challenger on the hold-out and run the gate; writes results/holdout/ and figures/
	PYTHONPATH=src python -m cameratrap.report.holdout

# The control is reported only and never touches the hold-out. Training needs
# `make install-models` and the crop store, and resumes if stopped.
cnn-control: ## Fine-tune EfficientNet-B0 out of fold by site (about 16 h on CPU); writes results/control/frame_logits.npz
	PYTHONPATH=src python -m cameratrap.arms.cnn

control: ## Report the CNN control from its committed logits; writes results/control/ and figures/
	PYTHONPATH=src python -m cameratrap.report.control

trust-report: ## Per-site trust report for the champion from committed predictions; writes results/trust/ and figures/
	PYTHONPATH=src python -m cameratrap.report.trust

drift: ## Replay the champion in site and date order and score six label-free signals; writes results/drift/ and figures/
	PYTHONPATH=src python -m cameratrap.report.drift

# Both outputs are git-ignored and derived from committed results.
track: ## Rebuild mlflow.db and mlruns/ from the committed results: runs, versions and aliases
	rm -rf mlflow.db mlruns
	PYTHONPATH=src python -m cameratrap.tracking.registry

gate-calibrate: ## Per-site tolerance false-block rates on fold sites; writes results/gate/ and figures/
	PYTHONPATH=src python -m cameratrap.report.gate_calibration

hub-stage: ## Write the files and dataset card to publish into data/hub_stage/
	PYTHONPATH=src python -m cameratrap.embed.hub stage

REPO_ID ?= benmacp123/orinoquia-crop-embeddings

hub-upload: ## Owner only: upload data/hub_stage/ to REPO_ID (needs `hf auth login`)
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
