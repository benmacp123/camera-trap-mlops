"""Tests for publishing the promoted head and its model card, without the network."""

import json
import shutil

import pytest

from cameratrap.arms import head_hub


@pytest.fixture(name="evidence")
def fixture_evidence():
    """Return the committed results the card is written from.

    Returns:
        The dict from ``head_hub.load_evidence``.
    """
    return head_hub.load_evidence()


def test_class_recall_counts_per_label():
    """Recall is hits over that label's sequences, and absent classes are left out."""
    rows = [
        {"label": "a", "pred": "a"},
        {"label": "a", "pred": "b"},
        {"label": "b", "pred": "b"},
    ]
    assert head_hub.class_recall(rows, ["b", "a", "c"]) == [
        ("b", 1, 1.0),
        ("a", 2, 0.5),
    ]


def test_card_reports_the_committed_holdout_result(evidence):
    """The card's headline figures are the hold-out summary's, not typed by hand."""
    text = head_hub.card(evidence, "u/r")
    summary = evidence["summary"]
    low, high = summary["interval"]
    assert (
        f"{summary['mean_gain']:+.3f} (95% interval {low:+.3f} to {high:+.3f}" in text
    )
    for row in evidence["per_site"]:
        assert f"| {row['site']} | {row['n']} |" in text
    assert evidence["freeze"]["sha256"] in text
    assert 'hf_hub_download("u/r", "probe_head.npz")' in text


def test_card_states_licence_and_limits(evidence):
    """The card carries the licence, the base model and the cautions a user needs."""
    text = head_hub.card(evidence, "u/r")
    assert text.startswith("---\nlicense: mit\n")
    assert "base_model: timm/vit_base_patch14_dinov2.lvd142m" in text
    assert "Any species outside the nine" in text
    assert "probably seen these images" in text
    assert "Spix's guan" in text


def test_evidence_refuses_a_result_from_another_freeze(tmp_path):
    """A hold-out result scored under a different freeze cannot back the card."""
    holdout = tmp_path / "holdout"
    shutil.copytree(head_hub.HOLDOUT_DIR, holdout)
    summary = json.loads((holdout / "summary.json").read_text())
    summary["freeze_sha256"] = "0" * 64
    (holdout / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="committed freeze"):
        head_hub.load_evidence(holdout_dir=holdout)


def test_stage_copies_the_head_byte_for_byte(tmp_path, evidence):
    """The staged head is the committed file, beside a card naming the repo."""
    files = head_hub.stage(tmp_path / "stage", "u/r", evidence)
    committed = head_hub.file_sha256(head_hub.HOLDOUT_DIR / head_hub.HEAD_NAME)
    assert files == {head_hub.HEAD_NAME: committed}
    assert "u/r" in (tmp_path / "stage" / "README.md").read_text()


def write_record(tmp_path, sha):
    """Write an upload record naming one head hash.

    Args:
        tmp_path: Folder to write into.
        sha: The SHA-256 the record claims for the head.

    Returns:
        Path of the record.
    """
    path = tmp_path / "hub_head.json"
    path.write_text(
        json.dumps(
            {
                "repo_id": "u/r",
                "repo_type": "model",
                "revision": "deadbeef",
                "files": {head_hub.HEAD_NAME: sha},
            }
        )
    )
    return path


def fake_hub(monkeypatch, body):
    """Make ``hf_hub_download`` write fixed bytes instead of using the network.

    Args:
        monkeypatch: pytest's patcher.
        body: The bytes the stand-in Hub serves for any file.
    """

    def download(repo_id, name, **kwargs):
        """Stand in for the Hub.

        Args:
            repo_id: The repo asked for.
            name: The file asked for.
            **kwargs: ``repo_type``, ``revision`` and ``local_dir``.

        Returns:
            The local path written.
        """
        assert (repo_id, kwargs["revision"]) == ("u/r", "deadbeef")
        target = f"{kwargs['local_dir']}/{name}"
        with open(target, "wb") as handle:
            handle.write(body)
        return target

    monkeypatch.setattr(head_hub, "hf_hub_download", download)


def test_check_passes_when_published_matches_committed(tmp_path, monkeypatch):
    """A published head identical to the committed one passes the check."""
    head = tmp_path / "head.npz"
    head.write_bytes(b"weights")
    record = write_record(tmp_path, head_hub.file_sha256(head))
    fake_hub(monkeypatch, b"weights")
    assert head_hub.check(record, head)["sha256"] == head_hub.file_sha256(head)


def test_check_refuses_a_different_published_head(tmp_path, monkeypatch):
    """A published head that differs from the record is an error."""
    head = tmp_path / "head.npz"
    head.write_bytes(b"weights")
    record = write_record(tmp_path, head_hub.file_sha256(head))
    fake_hub(monkeypatch, b"tampered")
    with pytest.raises(ValueError, match="published head"):
        head_hub.check(record, head)


def test_check_refuses_a_changed_committed_head(tmp_path, monkeypatch):
    """A committed head that no longer matches the record is an error."""
    head = tmp_path / "head.npz"
    head.write_bytes(b"retrained")
    record = write_record(tmp_path, "0" * 64)
    fake_hub(monkeypatch, b"weights")
    with pytest.raises(ValueError, match="committed head"):
        head_hub.check(record, head)
