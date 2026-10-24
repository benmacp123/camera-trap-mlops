# Camera-Trap MLOps

Status: planning; nothing is built yet.

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

## How it works

The planned components are below. Dashed parts are not built yet; the
diagram is updated as each one lands.

```mermaid
flowchart TB
    lila[("LILA BC images<br>Orinoquía, 9 species")] -.-> manifest["Manifest and loader<br>site split, humans excluded"]
    mdv5a[("Precomputed<br>MegaDetector v5a boxes")] -.-> crops["Crops"]
    manifest -.-> crops
    crops -.-> embeddings[("Cached embeddings<br>Hugging Face Hub")]

    subgraph arms ["Classifier arms (planned)"]
        direction LR
        speciesnet["SpeciesNet<br>zero-shot, COL geofence"]
        bioclip["BioCLIP 2<br>zero-shot"]
        probe["Probe and kNN on<br>BioCLIP 2, DINOv2 embeddings"]
        cnn["Naive CNN fine-tune<br>(control)"]
    end

    crops -.-> arms
    embeddings -.-> arms

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

## Licence

To be added.
