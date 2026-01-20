#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path


SUPERVISED_ROWS = [
    ("InterPro_Binding", "Fold", "AUROC%"),
    ("InterPro_Binding", "Superfamily", "AUROC%"),
    ("BioLIP2_Binding", "Fold", "AUROC%"),
    ("BioLIP2_Binding", "Superfamily", "AUROC%"),
    ("ProteinShake_Binding", "Test", "AUROC%"),
    ("InterPro_ActiveSite", "Fold", "AUROC%"),
    ("InterPro_ActiveSite", "Superfamily", "AUROC%"),
    ("BioLIP2_Catalytic", "Fold", "AUROC%"),
    ("BioLIP2_Catalytic", "Superfamily", "AUROC%"),
    ("InterPro_Conserved", "Fold", "AUROC%"),
    ("InterPro_Conserved", "Superfamily", "AUROC%"),
    ("InterPro_Repeats", "Fold", "AUROC%"),
    ("InterPro_Repeats", "Superfamily", "AUROC%"),
    ("ProteinGLUE_Epitope", "Fold", "AUROC%"),
    ("ProteinGLUE_Epitope", "Superfamily", "AUROC%"),
    ("Atlas_RMSF", "Fold", "Spearman%"),
    ("Atlas_RMSF", "Superfamily", "Spearman%"),
    ("Atlas_BFactor", "Fold", "Spearman%"),
    ("Atlas_BFactor", "Superfamily", "Spearman%"),
    ("Atlas_NEQ", "Fold", "Spearman%"),
    ("Atlas_NEQ", "Superfamily", "Spearman%"),
    ("RemoteHomology", "Fold", "MacroF1%"),
    ("RemoteHomology", "Superfamily", "MacroF1%"),
    ("RemoteHomology", "Family", "MacroF1%"),
]

UNSUPERVISED_ROWS = [
    ("ApoHolo", "Test", "PCC%"),
    ("ApoHolo", "Test", "Spearman%"),
    ("FoldSwitching", "Test", "PCC%"),
    ("FoldSwitching", "Test", "Spearman%"),
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def _extract_last_float(text: str, pattern: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    if isinstance(matches[-1], tuple):
        return [float(x) for x in matches[-1]]
    try:
        return float(matches[-1])
    except Exception:
        return None


def _fmt(val):
    if val is None:
        return ""
    try:
        return f"{float(val):.2f}"
    except Exception:
        return ""


def _load_log(path: Path):
    if not path.is_file():
        return ""
    return _read_text(path)


def collect_metrics(logs_dir: Path):
    supervised = {}
    unsupervised = {}

    def set_val(row_key, value):
        supervised[row_key] = _fmt(value)

    def set_val_unsup(row_key, value):
        unsupervised[row_key] = _fmt(value)

    # BioLIP2 binding
    text = _load_log(logs_dir / "eval_biolip2_binding.log")
    set_val(
        ("BioLIP2_Binding", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("BioLIP2_Binding", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # BioLIP2 catalytic
    text = _load_log(logs_dir / "eval_biolip2_catalytic.log")
    set_val(
        ("BioLIP2_Catalytic", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("BioLIP2_Catalytic", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # ProteinShake binding
    text = _load_log(logs_dir / "eval_proteinshake.log")
    set_val(
        ("ProteinShake_Binding", "Test", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+test AUROC% = ([0-9.]+)"),
    )

    # InterPro binding
    text = _load_log(logs_dir / "eval_interpro_binding.log")
    set_val(
        ("InterPro_Binding", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("InterPro_Binding", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # InterPro active site
    text = _load_log(logs_dir / "eval_interpro_activesite.log")
    set_val(
        ("InterPro_ActiveSite", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("InterPro_ActiveSite", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # InterPro conserved
    text = _load_log(logs_dir / "eval_interpro_conserved.log")
    set_val(
        ("InterPro_Conserved", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("InterPro_Conserved", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # InterPro repeats
    text = _load_log(logs_dir / "eval_interpro_repeats.log")
    set_val(
        ("InterPro_Repeats", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("InterPro_Repeats", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # ProteinGLUE epitope region
    text = _load_log(logs_dir / "eval_proteinglue.log")
    set_val(
        ("ProteinGLUE_Epitope", "Fold", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+fold_test AUROC% = ([0-9.]+)"),
    )
    set_val(
        ("ProteinGLUE_Epitope", "Superfamily", "AUROC%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test AUROC% = ([0-9.]+)"),
    )

    # Atlas metrics (Spearman; scale to percent)
    def set_atlas(task_name, log_name):
        text = _load_log(logs_dir / log_name)
        fold = _extract_last_float(text, r"\[final\]\s+fold_test spearman=([0-9.]+)")
        sfam = _extract_last_float(text, r"\[final\]\s+superfamily_test spearman=([0-9.]+)")
        fold = fold * 100.0 if fold is not None else None
        sfam = sfam * 100.0 if sfam is not None else None
        set_val((task_name, "Fold", "Spearman%"), fold)
        set_val((task_name, "Superfamily", "Spearman%"), sfam)

    set_atlas("Atlas_RMSF", "eval_atlas_rmsf.log")
    set_atlas("Atlas_BFactor", "eval_atlas_bfactor.log")
    set_atlas("Atlas_NEQ", "eval_atlas_neq.log")

    # Remote homology (Macro F1 already in %)
    text = _load_log(logs_dir / "eval_remote_homology.log")
    set_val(
        ("RemoteHomology", "Fold", "MacroF1%"),
        _extract_last_float(text, r"\[final\]\s+fold_test f1=([0-9.]+)"),
    )
    set_val(
        ("RemoteHomology", "Superfamily", "MacroF1%"),
        _extract_last_float(text, r"\[final\]\s+superfamily_test f1=([0-9.]+)"),
    )
    set_val(
        ("RemoteHomology", "Family", "MacroF1%"),
        _extract_last_float(text, r"\[final\]\s+family_test f1=([0-9.]+)"),
    )

    # Apo/Holo + Fold Switching (unsupervised)
    text = _load_log(logs_dir / "eval_apolo.log")
    table_vals = _extract_last_float(
        text,
        r"\[table\]\s+apo_holo PCC%=([0-9.]+)\s+Spearman%=([0-9.]+).*fold_switching PCC%=([0-9.]+)\s+Spearman%=([0-9.]+)",
    )
    if isinstance(table_vals, list) and len(table_vals) == 4:
        set_val_unsup(("ApoHolo", "Test", "PCC%"), table_vals[0])
        set_val_unsup(("ApoHolo", "Test", "Spearman%"), table_vals[1])
        set_val_unsup(("FoldSwitching", "Test", "PCC%"), table_vals[2])
        set_val_unsup(("FoldSwitching", "Test", "Spearman%"), table_vals[3])

    return supervised, unsupervised


def update_csv(path: Path, rows, run_tag: str, values: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Task", "Split", "Metric"]
    existing = {}
    extra_rows = []

    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = ["Task", "Split", "Metric"]
            for row in reader:
                if len(row) < 3:
                    continue
                key = (row[0], row[1], row[2])
                existing[key] = row

        if run_tag not in header:
            header.append(run_tag)
        run_idx = header.index(run_tag)

        for key, row in existing.items():
            if key not in rows:
                extra_rows.append(key)

        for key in rows + extra_rows:
            row = existing.get(key, [key[0], key[1], key[2]])
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            row[run_idx] = values.get(key, "")
            existing[key] = row

        ordered_rows = [existing[key] for key in rows + extra_rows]
    else:
        header = ["Task", "Split", "Metric", run_tag]
        ordered_rows = []
        for key in rows:
            ordered_rows.append([key[0], key[1], key[2], values.get(key, "")])

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(ordered_rows)


def parse_args():
    p = argparse.ArgumentParser(description="Append eval metrics to CSV tables.")
    p.add_argument("--logs-dir", required=True, help="Directory containing eval_*.log files")
    p.add_argument("--run-tag", required=True, help="Column name for this run (e.g., date tag)")
    p.add_argument("--out-supervised", required=True, help="Output CSV for supervised metrics")
    p.add_argument("--out-unsupervised", required=True, help="Output CSV for unsupervised metrics")
    p.add_argument("--skip-supervised", action="store_true", help="Do not update supervised CSV")
    p.add_argument("--skip-unsupervised", action="store_true", help="Do not update unsupervised CSV")
    return p.parse_args()


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    supervised, unsupervised = collect_metrics(logs_dir)
    if not args.skip_supervised:
        update_csv(Path(args.out_supervised), SUPERVISED_ROWS, args.run_tag, supervised)
    if not args.skip_unsupervised:
        update_csv(Path(args.out_unsupervised), UNSUPERVISED_ROWS, args.run_tag, unsupervised)


if __name__ == "__main__":
    main()
