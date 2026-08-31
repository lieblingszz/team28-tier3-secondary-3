import argparse
import csv
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import OFFICIAL_OUTCOMES        # noqa: E402

# code/scripts/ -> code/ -> code/results/. The submission keeps the model
# outputs beside the code, so this needs no configuration.
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

SCALE_RANGE = {name: spec["scale_range"] for name, spec in OFFICIAL_OUTCOMES.items()}

# Members grouped by MODEL. A group with more than one file is averaged within
# itself before the groups are averaged together.
MEMBERS = {
    "qwen38": [
        "tier3_submission_calibrated_tier3_qwen38_rag_hipporag_20260830_093608.csv",
        "tier3_submission_calibrated_tier3_qwen38_rag_hipporag_g70_20260830_102830.csv",
    ],
    "gpt4o":  ["tier3_submission_calibrated_tier3_gpt4o_rag_parallel_fulltext_20260825_132946.csv"],
    "gpt5":   ["tier3_submission_calibrated_tier3_gpt5_rag_parallel_fulltext_20260826_135120.csv"],
    "claude": ["tier3_submission_calibrated_tier3_claude_rag_parallel_fulltext_20260826_134941.csv"],
}

KEY = ["condition", "outcome"]


def _read(name):
    df = pd.read_csv(RESULTS_DIR / name)
    assert list(df.columns) == KEY + ["ate"], f"{name}: unexpected columns {list(df.columns)}"
    assert not df["ate"].isna().any(), f"{name}: missing values"
    assert len(df) == 208, f"{name}: expected 208 rows, got {len(df)}"
    return df.set_index(KEY)["ate"]


def build(flat: bool):
    groups = {}
    for model, files in MEMBERS.items():
        series = [_read(f) for f in files]
        # Every member must cover the identical grid, or a mean would be over
        # a different set of cells per model without saying so.
        for s in series[1:]:
            assert s.index.equals(series[0].index), f"{model}: grid mismatch between members"
        groups[model] = pd.concat(series, axis=1)

    all_files = pd.concat(list(groups.values()), axis=1)
    for name, g in groups.items():
        assert g.index.equals(all_files.index), f"{name}: grid mismatch across models"

    # The series the ensemble actually averages over: five files under --flat,
    # four model-level means otherwise. Reported alongside so the shrinkage
    # line compares the ensemble against the right thing.
    per_member = all_files if flat else pd.concat(
        [g.mean(axis=1).rename(name) for name, g in groups.items()], axis=1)
    ate = per_member.mean(axis=1)

    return ate.rename("ate").reset_index(), all_files, per_member


def write(df, path):
    """Match the rest of ../results/: CRLF, 4 dp, sorted by condition then
    outcome."""
    df = df.sort_values(KEY)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(KEY + ["ate"])
    for condition, outcome, ate in df.itertuples(index=False):
        w.writerow([condition, outcome, repr(round(float(ate), 4))])
    path.write_text(buf.getvalue(), newline="")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true",
                    help="equal weight per FILE rather than per model (five-way mean)")
    args = ap.parse_args()

    ate, members, per_member = build(args.flat)
    scheme = "flat" if args.flat else "nested"
    out = RESULTS_DIR / (f"tier3_submission_calibrated_tier3_ensemble_direct"
                       f"{'_flat' if args.flat else ''}.csv")
    write(ate, out)

    pp = ate["ate"] / ate["outcome"].map(SCALE_RANGE) * 100
    member_pp = per_member.div(
        per_member.index.get_level_values("outcome").map(SCALE_RANGE), axis=0) * 100
    weights = ("1/5 each (Qwen3.8 gets 2/5 = 40%)" if args.flat
               else "1/4 per model (Qwen3.8's two runs share one quarter)")

    print(f"{out.name}")
    print(f"  scheme      : {scheme} mean, weights {weights}")
    print(f"  members     : {members.shape[1]} files across {len(MEMBERS)} models, "
          f"averaged over {per_member.shape[1]} series")
    print(f"  cells       : {len(ate)}")
    print(f"  mean|pp|    : {pp.abs().mean():.4f}")
    print(f"  members' own mean|pp|, averaged: {member_pp.abs().mean().mean():.4f}   "
          f"(the ensemble is smaller by {member_pp.abs().mean().mean() - pp.abs().mean():.4f}"
          f" -- that gap is member disagreement cancelling)")
    print(f"  per-cell spread across members: mean SD = {member_pp.std(axis=1).mean():.4f} pp, "
          f"max = {member_pp.std(axis=1).max():.4f} pp")


if __name__ == "__main__":
    main()
