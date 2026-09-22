"""Put the paradigm's marker names back into files converted before eeg_raw_to_bids kept them.

The BrainVision derivatives (BIDS and every decomb stage) share one marker file per run,
byte for byte, so each is rewritten from a freshly converted BIDS tree whose positions must
match. eegfeat bundles get the event name renamed in the recipe, the events ledger, the
epochs' event_id and the manifest's provenance, with the manifest hashes recomputed.

    python paradigm_specific/thermal_pain/repair_markers.py --fixed-bids FIXED \\
        --derivatives BIDS_EEG DECOMB_STAGE... --bundles EEGFEAT --archive ARCHIVE
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mne
import pandas as pd

from eegfeat.preprocessing.provenance import canonical_json, file_hash

GENERIC = {"Stimulus/S  1": "Trig_therm/T  1", "Stimulus/S  2": "Volume/V  1"}
RUN = re.compile(r"(sub-[A-Za-z0-9]+)_task-thermalactive_run-(\d+)")


def marker_lines(path: Path) -> list[str]:
    return [
        line.split("=", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Mk")
    ]


def splice_markers(fixed: Path, old: Path) -> None:
    # The generic file holds the same events at the same samples; only the names were lost.
    named = marker_lines(fixed)
    positions = {
        line.split(",")[2] for line in named if line.split(",")[0] in ("Volume", "Trig_therm")
    }
    generic = {line.split(",")[2] for line in marker_lines(old)}
    if generic != positions:
        raise ValueError(f"{old}: marker positions differ from {fixed}; not the same recording")
    kept = [
        line for line in old.read_text(encoding="utf-8").splitlines() if not line.startswith("Mk")
    ]
    numbered = [f"Mk{number}={marker}" for number, marker in enumerate(named, start=1)]
    old.write_text("\n".join([*kept, *numbered]) + "\n", encoding="utf-8")


def rename(text: str) -> str:
    for generic, named in GENERIC.items():
        text = text.replace(generic, named)
    return text


def repair_bundle(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    stem = manifest_path.name.removesuffix("_preprocessing.json")
    folder = manifest_path.parent
    epochs_file = folder / f"{stem}_epo.fif"
    epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
    epochs.event_id = {rename(name): code for name, code in epochs.event_id.items()}
    epochs.save(epochs_file, overwrite=True, fmt="double", verbose=False)
    ledger = folder / f"{stem}_events.tsv"
    frame = pd.read_csv(ledger, sep="\t")
    frame["label"] = frame["label"].map(rename)
    frame.to_csv(ledger, sep="\t", index=False)
    recipe = folder / f"{stem}_recipe.yaml"
    recipe.write_text(rename(recipe.read_text()))
    manifest["provenance"] = json.loads(rename(json.dumps(manifest["provenance"])))
    for name in manifest["files"]:
        manifest["files"][name] = file_hash(folder / name)
    manifest_path.write_text(canonical_json(manifest) + "\n")


# Parked generations stay as they were; only bundles still carrying the generic name qualify.
def live_bundles(root: Path) -> list[Path]:
    return [
        manifest
        for manifest in sorted(root.rglob("*_task-thermalactive_run-*_preprocessing.json"))
        if not any(part.startswith("_superseded") for part in manifest.relative_to(root).parts)
        and not manifest.name.startswith("._")  # exFAT AppleDouble sidecars match the glob
        and "Stimulus/S" in manifest.read_text()
    ]


def archive(path: Path, root: Path, archive_root: Path) -> None:
    target = archive_root / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(path, target)  # not copy2: xattrs become ._ sidecars on exFAT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fixed-bids", type=Path, required=True, help="BIDS tree written by the fixed converter"
    )
    parser.add_argument(
        "--derivatives",
        type=Path,
        nargs="*",
        default=[],
        help="trees holding *_eeg.vmrk to rewrite",
    )
    parser.add_argument(
        "--bundles", type=Path, default=None, help="EEGFeat tree holding *_preprocessing.json"
    )
    parser.add_argument(
        "--archive", type=Path, required=True, help="where originals are copied first"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    fixed = {
        RUN.search(p.name).groups(): p
        for p in args.fixed_bids.rglob("*_eeg.vmrk")
        if RUN.search(p.name)
    }
    spliced = 0
    for root in args.derivatives:
        for old in sorted(root.rglob("*_task-thermalactive_run-*_eeg.vmrk")):
            key = RUN.search(old.name)
            if key is None or old.name.startswith("._"):
                continue
            source = fixed.get(key.groups())
            if source is None:
                raise FileNotFoundError(f"{old}: no fixed conversion for {key.group(0)}")
            if marker_lines(old) == marker_lines(source):
                continue
            if not args.dry_run:
                archive(old, root, args.archive / root.name)
                splice_markers(source, old)
            spliced += 1
    repaired = 0
    if args.bundles is not None:
        for manifest in live_bundles(args.bundles):
            if not args.dry_run:
                stem = manifest.name.removesuffix("_preprocessing.json")
                for suffix in ("_preprocessing.json", "_epo.fif", "_events.tsv", "_recipe.yaml"):
                    archive(
                        manifest.with_name(stem + suffix),
                        args.bundles,
                        args.archive / args.bundles.name,
                    )
                repair_bundle(manifest)
            repaired += 1
    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"{verb} {spliced} marker files and {repaired} bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
