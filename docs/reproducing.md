# Reproducing every step

[Back to the README](../README.md)

Needs [uv](https://docs.astral.sh/uv/) and nothing else: no accounts or
API keys. Python 3.14 on Linux x86-64 or Apple-silicon macOS.

```bash
make install          # core and dev tools: gate, drift, tracking, tests
make install-models   # adds CPU torch, BioCLIP 2, DINOv2 and SpeciesNet
source venv/bin/activate
make test lint format-check
make manifest         # optional: rebuild the manifest from LILA (5 MB download)
make fetch            # optional: download the spike's images (11 GB) and check them
make spike-models     # optional: MDv5a check, BioCLIP 2, SpeciesNet (about 12 h on CPU)
make spike            # score the spike from those caches; writes results/spike/
make detector-report  # detector misses per site and class; committed files only
make fetch-crops      # optional: download every indexed image (20 GB) and check them
make crops            # cut the 25,947 crops to data/crops/
make embed            # optional: BioCLIP 2 and DINOv2 over every crop (about 6 h on CPU)
make speciesnet-crops # optional: SpeciesNet over every crop frame (about 13 h on CPU)
make embed-check      # re-embed 50 cached crops per backbone and compare
make baselines        # zero-shot tables per fold site; writes results/baselines/ and figures/
make challengers      # probes and kNN per fold site (about 12 min); writes results/challengers/
make select-challenger # choose the hold-out challenger by simulated gate pass rate (about 8 min); writes results/selection/
make freeze           # already run: writes config/holdout_freeze.json once and refuses to overwrite it
make holdout-check    # check the freeze, inputs and caches; fits and scores nothing
make holdout          # one shot: score the hold-out and run the gate; writes results/holdout/
make cnn-control      # optional: fine-tune the CNN control (about 14 h on CPU, resumes); writes its logits to results/control/
make control          # report the control from its committed logits; writes results/control/ and figures/
make drift            # replay the champion and score six label-free signals (about 7 min, needs the embedding caches); writes results/drift/
make review           # simulate the review loop: queue, label, retrain, gate (about 20 min on 8 cores, needs the embedding caches); writes results/review/
make trust-report     # per-site trust report for the champion, with drift's predicted accuracy (about 3 min); writes results/trust/ and figures/
make cloud-report     # cost report from the committed run files; writes results/cloud/summary.json and figures/
make track            # rebuild the MLflow runs and aliases from committed results
make gate-calibrate   # fixed-tolerance false-block rates (stand-in pair, kept for the record); writes results/gate/
```

The model outputs are published on the Hugging Face Hub, so the scoring
can be reproduced without downloading images or running a model:

```bash
make fetch-embeddings # no account needed; checks each file's SHA-256
make baselines
```

`make help` lists every target.
