# Every target runs from committed files and public downloads; none needs
# an account. `make help` lists them.

.PHONY: help install install-models lock manifest subset fetch spike-models spike crop-index fetch-crops crops crop-sizes detector-report embed embed-check speciesnet-crops baselines challengers select-challenger freeze holdout-check holdout cnn-control control trust-report drift review track gate-calibrate hub-stage hub-upload fetch-embeddings prices lambda-model lambda-image cloud-emulate tf-fmt tf-check tf-plan tf-apply tf-output lambda-push cloud-upload cloud-sweep cloud-run tf-destroy cloud-leftovers test lint format format-check

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

review: ## Simulate the review loop: queue, label, retrain and gate at new sites (about 20 min, 8 cores); writes results/review/ and figures/
	PYTHONPATH=src python -m cameratrap.report.review

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

# Phase 8, the AWS batch route. Everything down to tf-check runs without an
# account. Docker commands use --network host: the bridge network here has
# no DNS.
prices: ## Refresh the AWS list-price snapshot in data/prices/ (public files, no account)
	PYTHONPATH=src python -m cloud.prices

# The revision the DINOv2 embedding cache records, so the image holds the
# weights every committed prediction was made with.
DINOV2_REVISION = 4685c99dabffe5affac90bd99dbffd25801ae58d
LAMBDA_IMAGE = camera-trap-lambda

LAMBDA_WEIGHTS = build/lambda/model.safetensors

# A file target, so the 330 MB copy happens once.
$(LAMBDA_WEIGHTS):
	mkdir -p build/lambda
	python -c "from huggingface_hub import hf_hub_download; import shutil; \
		shutil.copyfile(hf_hub_download('timm/vit_base_patch14_dinov2.lvd142m', \
		'model.safetensors', revision='$(DINOV2_REVISION)'), '$(LAMBDA_WEIGHTS)')"

lambda-model: $(LAMBDA_WEIGHTS) ## Stage the DINOv2 weights in build/lambda/ (330 MB; from the Hugging Face cache or a download, no account)

# --provenance=false: Lambda rejects the multi-manifest image that Docker's
# default build attestation produces.
lambda-image: lambda-model ## Build the champion's Lambda container image (needs Docker)
	docker build --network host --provenance=false -f infra/lambda/Dockerfile \
		-t $(LAMBDA_IMAGE) .

# AWS's emulator stands in for the Lambda service; a folder stands in for
# S3. Read-only filesystem and a 512 MB /tmp as in Lambda. OMP_NUM_THREADS
# matches the two CPUs allowed, which the container cannot see for itself.
RIE_URL = https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.38/aws-lambda-rie-x86_64
EMULATOR = camera-trap-emulator

build/aws-lambda-rie:
	mkdir -p build
	curl -sSL -o $@ $(RIE_URL)
	chmod +x $@

cloud-emulate: lambda-image build/aws-lambda-rie ## Run 96 crops through the container under the Lambda emulator and compare with the local predictions (needs Docker and the crop store)
	mkdir -p build/emulator/bucket
	docker run -d --rm --name $(EMULATOR) -p 9000:8080 \
		--read-only --tmpfs /tmp:size=512m --memory 3008m --cpus 2 \
		--user $$(id -u):$$(id -g) -e OMP_NUM_THREADS=2 -e LOCAL_ROOT=/local \
		-v $(CURDIR)/build/aws-lambda-rie:/aws-lambda-rie:ro \
		-v $(CURDIR)/build/emulator/bucket:/local/bucket \
		-v $(CURDIR)/data/crops/orinoquia:/local/bucket/crops:ro \
		--entrypoint /aws-lambda-rie $(LAMBDA_IMAGE) \
		/opt/venv/bin/python -m awslambdaric cloud.handler.handler
	sleep 2
	PYTHONPATH=src python -m cloud.batch emulate; \
		status=$$?; docker stop $(EMULATOR) > /dev/null; exit $$status

# Terraform runs in a pinned image, so nothing needs installing.
TF_IMAGE = hashicorp/terraform:1.16.6@sha256:bfffc1ded5fbf0a17beaaf22eb4793980abdb19d0761ff6a3a0efb2ab4d9b316
DOCKER_RUN = docker run --rm --network host --user $$(id -u):$$(id -g) \
	-e HOME=/tmp -v $(CURDIR):/repo -w /repo/infra/terraform
TF = $(DOCKER_RUN) $(TF_IMAGE)

tf-fmt: ## Format the Terraform (needs Docker)
	$(TF) fmt -recursive

tf-check: ## terraform fmt -check and validate, no account needed (needs Docker)
	$(TF) fmt -check -recursive -diff
	$(TF) init -backend=false -input=false
	$(TF) validate

# Everything below reads or changes your AWS account, so run it yourself:
# it needs credentials in ~/.aws (set AWS_PROFILE) and
# infra/terraform/terraform.tfvars. apply and destroy run interactively
# (-it) so Terraform shows the plan and asks before changing anything.
#
# Order: tf-apply (budget alarm, bucket, registry) -> lambda-push ->
# tf-apply FUNCTIONS=true -> cloud-upload -> cloud-sweep -> cloud-run
# MEMORY=<the sweep's cheapest> -> tf-destroy -> cloud-leftovers.
TF_AWS = $(DOCKER_RUN) -v $(HOME)/.aws:/tmp/.aws:ro -e AWS_PROFILE
FUNCTIONS ?= false
OUTPUT = python -c "import json, sys; print(json.load(open('build/terraform-outputs.json'))[sys.argv[1]]['value'])"

# The provider is downloaded once; later runs skip the registry, so a slow
# network cannot fail a step that needs nothing from it.
infra/terraform/.terraform:
	$(TF) init -input=false

tf-plan: infra/terraform/.terraform ## terraform plan against your AWS account; FUNCTIONS=true includes the functions (credentials)
	$(TF_AWS) $(TF_IMAGE) plan -var deploy_functions=$(FUNCTIONS)

tf-apply: infra/terraform/.terraform ## Deploy; first without the functions, then FUNCTIONS=true after lambda-push (credentials)
	$(TF_AWS) -it $(TF_IMAGE) apply -var deploy_functions=$(FUNCTIONS)
	$(MAKE) tf-output

# The outputs include the account ID (in the registry URL): build/ is git-ignored.
tf-output: ## Save terraform output -json to build/terraform-outputs.json
	mkdir -p build
	$(TF) output -json > build/terraform-outputs.json

lambda-push: lambda-image ## Push the image to the stack's registry (credentials; after the first tf-apply)
	aws ecr get-login-password --region $$($(OUTPUT) region) | \
		docker login --username AWS --password-stdin $$($(OUTPUT) repository_url | cut -d/ -f1)
	docker tag $(LAMBDA_IMAGE) $$($(OUTPUT) repository_url):$$($(OUTPUT) image_tag)
	docker push $$($(OUTPUT) repository_url):$$($(OUTPUT) image_tag)

cloud-upload: ## Upload the indexed crops (1.9 GB) to the stack's bucket; resumes (credentials)
	PYTHONPATH=src python -m cloud.batch upload

cloud-sweep: ## Score one fixed 512-crop sample at each memory size; writes results/cloud/ (credentials)
	PYTHONPATH=src python -m cloud.batch sweep

cloud-run: ## The measured run: every crop at MEMORY=<MB>; resumes; writes results/cloud/ (credentials)
	PYTHONPATH=src python -m cloud.batch run --memory $(MEMORY)

# deploy_functions=false: destroy removes everything in the state whatever
# the variable says, and false stops Terraform looking up the pushed image,
# which fails the destroy if the push never finished.
tf-destroy: ## Remove everything deployed, bucket contents and image included (credentials)
	$(TF_AWS) -it $(TF_IMAGE) destroy -var deploy_functions=false

cloud-leftovers: ## After tf-destroy: list anything of the project still in the account (credentials)
	PYTHONPATH=src python -m cloud.batch leftovers

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
