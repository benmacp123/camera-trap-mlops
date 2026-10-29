# Camera-Trap MLOps

Status: Phase 0 done. The first spike ran on Orinoquía's fold sites; the hold-out is untouched.

## The question

Can a small, reproducible lifecycle decide, site by site, whether to
trust a zero-shot camera-trap species classifier or replace it with an
adapted one, and can it tell from unlabelled images when that decision
needs revisiting?

Strong off-the-shelf classifiers already exist (SpeciesNet, BioCLIP 2).
What is rarely published is the accountability around them: how well
they do at camera sites they were never trained on, when an adapted
model has earned the right to replace them, and whether any label-free
signal warns that accuracy has dropped. This project builds that
layer. Finding that the off-the-shelf model should stay in place
everywhere would still be a complete result.

## Answer so far

From the first spike, on 38 fold sites (5,337 sequences, nine species;
none of the ten hold-out sites has been scored):

- BioCLIP 2 zero-shot, the champion, averages 0.84 balanced accuracy
  per site and falls below 0.85 at 18 of 38 sites; its weakest species
  is collared peccary.
- A linear probe on its frozen embeddings averages 0.89, +4.8 points.
- SpeciesNet averages 0.92 restricted to the nine species, and answers
  89% of sequences in its own open-set mode, 89% of them correctly. It
  has probably seen these images (they went through Wildlife Insights),
  so this is not held-out performance.
- Mean model confidence tracks per-site accuracy closely (Spearman 0.87,
  95% interval 0.73 to 0.93), a first hint that a label-free signal can
  flag weak sites.

The plan's rule for choosing the dataset asked for at least a quarter of
sites below 0.85 with the best zero-shot model. Read against SpeciesNet
that fails (3 of 38 sites); read against BioCLIP 2 it passes. The
project uses the BioCLIP 2 reading because SpeciesNet's scores here are
not held-out, a choice made after seeing the results and recorded as
such.

## How it works

The components are below. Solid parts are built; dashed parts are
planned, and the diagram is updated as each one lands.

```mermaid
flowchart TB
    lila[("LILA BC images<br>Orinoquía, 9 species")] --> manifest["Manifest and loader<br>site split, humans excluded"]
    mdv5a[("Precomputed<br>MegaDetector v5a boxes")] --> manifest
    mdv5a --> crops["Crops"]
    manifest --> crops
    crops --> embeddings[("Cached embeddings<br>local; Hugging Face Hub planned")]

    subgraph arms ["Classifier arms"]
        direction LR
        speciesnet["SpeciesNet<br>zero-shot, COL geofence"]
        bioclip["BioCLIP 2<br>zero-shot"]
        probe["Probe on BioCLIP 2<br>(kNN, DINOv2 planned)"]
        cnn["Naive CNN fine-tune<br>(control, planned)"]
    end

    crops --> speciesnet
    embeddings --> bioclip
    embeddings --> probe
    crops -.-> cnn

    subgraph lifecycle ["Lifecycle (planned)"]
        mlflow[("MLflow registry<br>@champion, @challenger")]
        gate{{"Promotion gate<br>per-site paired bootstrap"}}
        report["Per-site trust report"]
        replay["Replay by date and site"]
        drift["Drift signals scored<br>against held-back labels"]
        review["Review queue<br>sampled sequences"]
        mlflow -. "challenger" .-> gate
        gate -. "promote or reject" .-> mlflow
        gate -.-> report
        mlflow -. "champion" .-> replay -.-> drift -.-> review
    end

    arms -. "scores per site" .-> gate
    arms -. "logged runs" .-> mlflow
    review -. "new labels" .-> arms

    subgraph aws ["AWS batch route (planned)"]
        s3[("S3 crops")] -.-> lambda["Lambda container<br>classifier over crops"]
    end

    mlflow -. "champion" .-> lambda
```

## Data

Orinoquía Camera Traps (Vélez et al. 2023), from
[LILA BC](https://lila.science/datasets/orinoquia-camera-traps/),
released under the
[Community Data License Agreement (permissive)](https://cdla.io/permissive-1-0/).
Images are fetched from LILA's public mirrors and are not stored in this
repository.

The committed manifest, [`data/manifest/orinoquia.csv`](data/manifest/orinoquia.csv),
lists one row per image in a clean single-species sequence of the nine
classes: 65,042 images in 9,451 sequences at 49 sites. It is built from
LILA's metadata and precomputed MegaDetector v5a results, whose URLs and
SHA-256 checksums are recorded in
[`data/manifest/orinoquia.json`](data/manifest/orinoquia.json) along with
the counts of everything dropped. Any sequence that ever showed a person
is dropped whole (those images were removed upstream), as are
multi-species sequences and sequences of other classes.

Splits are by site, never by image. Ten sites with at least 30 sequences
each form the final hold-out, drawn with a fixed seed so that every class
has at least 15 hold-out sequences at two or more sites; the other 39
sites form five grouped folds of about 1,530 sequences each. The draw
uses label counts only, never model scores, and its hash is checked by
the tests so the hold-out cannot move after the freeze.

The first spike uses a smaller selection,
[`data/manifest/orinoquia_spike.csv`](data/manifest/orinoquia_spike.csv),
drawn from the fold sites only, so no hold-out image is downloaded before
the freeze. It holds up to 100 sequences per class per site, keeping in
each the three frames with the most confident MegaDetector animal box
(14,806 images), plus 500 random fold images of any confidence for
checking the precomputed detections against a fresh run. `make fetch`
downloads them (about 11 GB) and writes each image's SHA-256 into the
manifest, so a later download can be checked against it.

## Reproducing

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
```

The spike's model outputs are cached locally for now; publishing the
embeddings to the Hugging Face Hub, so the scoring can be reproduced
without the images or a model run, is part of Phase 2.

`make help` lists every target.

## Licence

Code: [MIT](LICENSE). Data keep their own licences (see Data).
