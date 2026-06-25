import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. Core Q-function
# ============================================================

def q_score_from_counts(domain_size: int, relation_size: int, image_size: int) -> float:
    if domain_size == 0 or image_size <= 1:
        return 0.0

    q = ((relation_size / domain_size) - 1.0) / (image_size - 1.0)
    return float(max(0.0, min(1.0, q)))


def compute_q_for_rule(df: pd.DataFrame, lhs: List[str], rhs: str, hyperedge_id: str) -> dict:
    domain_size = df[lhs].drop_duplicates().shape[0]
    image_size = df[rhs].drop_duplicates().shape[0]
    relation_size = df[lhs + [rhs]].drop_duplicates().shape[0]

    q = q_score_from_counts(domain_size, relation_size, image_size)
    strength = 1.0 - q
    is_fd = np.isclose(q, 0.0)

    return {
        "hyperedge_id": hyperedge_id,
        "lhs": tuple(lhs),
        "rhs": rhs,
        "layer": len(lhs),
        "domain_size": domain_size,
        "image_size": image_size,
        "relation_size": relation_size,
        "q_score": q,
        "strength": strength,
        "is_fd": is_fd,
        "type": "FD" if is_fd else "LD",
    }


# ============================================================
# 2. FD-only extraction
# ============================================================

def is_exact_fd(df: pd.DataFrame, lhs: List[str], rhs: str) -> bool:
    max_rhs_values = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False).max()
    return bool(max_rhs_values == 1)


def extract_fd_dependencies_for_hyperedge(
    df: pd.DataFrame,
    features: List[str],
    hyperedge_id: str,
    max_layer: int = 3,
    minimal_only: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:

    work_df = df[features].copy()
    if dropna:
        work_df = work_df.dropna()

    rows = []

    for rhs in features:
        lhs_candidates = [c for c in features if c != rhs]
        max_k = min(max_layer, len(lhs_candidates))
        kept_fd_lhs = []

        for k in range(1, max_k + 1):
            for lhs in combinations(lhs_candidates, k):
                lhs_tuple = tuple(lhs)

                if minimal_only:
                    if any(set(prev).issubset(lhs_tuple) for prev in kept_fd_lhs):
                        continue

                if is_exact_fd(work_df, list(lhs), rhs):
                    domain_size = work_df[list(lhs)].drop_duplicates().shape[0]
                    image_size = work_df[rhs].drop_duplicates().shape[0]

                    rows.append({
                        "hyperedge_id": hyperedge_id,
                        "lhs": lhs_tuple,
                        "rhs": rhs,
                        "layer": len(lhs_tuple),
                        "domain_size": domain_size,
                        "image_size": image_size,
                        "relation_size": domain_size,
                        "q_score": 0.0,
                        "strength": 1.0,
                        "is_fd": True,
                        "type": "FD",
                    })

                    kept_fd_lhs.append(lhs_tuple)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["hyperedge_id", "rhs", "layer", "lhs"],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

    return out


# ============================================================
# 3. Q-based candidate extraction
# ============================================================

def extract_candidate_dependencies_for_hyperedge(
    df: pd.DataFrame,
    features: List[str],
    hyperedge_id: str,
    max_layer: int = 3,
    min_strength: float = 0.8,
    max_q: Optional[float] = None,
    dropna: bool = False,
) -> pd.DataFrame:

    work_df = df[features].copy()
    if dropna:
        work_df = work_df.dropna()

    rows = []

    for rhs in features:
        lhs_candidates = [c for c in features if c != rhs]
        max_k = min(max_layer, len(lhs_candidates))

        for k in range(1, max_k + 1):
            for lhs in combinations(lhs_candidates, k):
                rule = compute_q_for_rule(work_df, list(lhs), rhs, hyperedge_id)

                if rule["strength"] >= min_strength:
                    if max_q is None or rule["q_score"] <= max_q:
                        rows.append(rule)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["hyperedge_id", "rhs", "layer", "q_score", "strength", "lhs"],
            ascending=[True, True, True, True, False, True]
        ).reset_index(drop=True)

    return out


# ============================================================
# 4. Minimality filtering
# ============================================================

def is_minimal_dependency_mixed(
    rule_row: pd.Series,
    all_rules_same_rhs: pd.DataFrame,
    ld_improvement_threshold: float = 0.2,
) -> bool:

    lhs = tuple(rule_row["lhs"])
    strength = float(rule_row["strength"])
    is_fd = bool(rule_row["is_fd"])

    if len(lhs) == 1:
        return True

    for k in range(1, len(lhs)):
        for subset in combinations(lhs, k):
            subset_match = all_rules_same_rhs[
                all_rules_same_rhs["lhs"].apply(lambda x: tuple(x) == tuple(subset))
            ]

            if subset_match.empty:
                continue

            subset_row = subset_match.iloc[0]

            if is_fd:
                if bool(subset_row["is_fd"]):
                    return False
            else:
                subset_strength = float(subset_row["strength"])
                if strength - subset_strength < ld_improvement_threshold:
                    return False

    return True


def filter_minimal_dependencies_mixed(
    dep_df: pd.DataFrame,
    ld_improvement_threshold: float = 0.2,
) -> pd.DataFrame:

    if dep_df.empty:
        return dep_df.copy()

    kept_rows = []

    for hyperedge_id in sorted(dep_df["hyperedge_id"].unique()):
        h_df = dep_df[dep_df["hyperedge_id"] == hyperedge_id].copy()

        for rhs in sorted(h_df["rhs"].unique()):
            rhs_df = h_df[h_df["rhs"] == rhs].copy()
            rhs_df = rhs_df.sort_values(by=["layer", "lhs"]).reset_index(drop=True)

            for _, row in rhs_df.iterrows():
                if is_minimal_dependency_mixed(
                    row,
                    rhs_df,
                    ld_improvement_threshold=ld_improvement_threshold,
                ):
                    kept_rows.append(row.to_dict())

    out = pd.DataFrame(kept_rows)

    if not out.empty:
        out = out.sort_values(
            by=["hyperedge_id", "type", "rhs", "layer", "q_score", "strength", "lhs"],
            ascending=[True, True, True, True, True, False, True]
        ).reset_index(drop=True)

    return out


# ============================================================
# 5. Hyperedge extraction
# ============================================================

def read_hyperedges(hyperedge_path: str) -> pd.DataFrame:
    hyperedges = pd.read_csv(hyperedge_path)

    required_cols = {"hyperedge_id", "features"}
    missing = required_cols - set(hyperedges.columns)

    if missing:
        raise ValueError(f"Hyperedge file is missing required columns: {missing}")

    return hyperedges


def extract_dependencies_for_hyperedges(
    df: pd.DataFrame,
    hyperedge_path: str,
    max_layer: int = 3,
    dependency_type: str = "both",
    min_strength: float = 0.8,
    max_q: Optional[float] = None,
    ld_improvement_threshold: float = 0.2,
    minimal_only: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:

    hyperedges = read_hyperedges(hyperedge_path)
    dependency_type = dependency_type.lower()

    all_results = []

    for _, row in hyperedges.iterrows():
        hyperedge_id = str(row["hyperedge_id"])
        features = [f.strip() for f in str(row["features"]).split(";") if f.strip()]

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: Skipping {hyperedge_id}; missing features: {missing_features}")
            continue

        if len(features) < 2:
            print(f"Warning: Skipping {hyperedge_id}; fewer than 2 features.")
            continue

        if dependency_type == "fd":
            dep_df = extract_fd_dependencies_for_hyperedge(
                df=df,
                features=features,
                hyperedge_id=hyperedge_id,
                max_layer=max_layer,
                minimal_only=minimal_only,
                dropna=dropna,
            )
        else:
            dep_df = extract_candidate_dependencies_for_hyperedge(
                df=df,
                features=features,
                hyperedge_id=hyperedge_id,
                max_layer=max_layer,
                min_strength=min_strength,
                max_q=max_q,
                dropna=dropna,
            )

            if minimal_only:
                dep_df = filter_minimal_dependencies_mixed(
                    dep_df,
                    ld_improvement_threshold=ld_improvement_threshold,
                )

            if dependency_type == "ld":
                dep_df = dep_df[dep_df["type"] == "LD"].copy()
            elif dependency_type == "both":
                pass
            else:
                raise ValueError("dependency_type must be one of: 'fd', 'ld', 'both'")

        if not dep_df.empty:
            all_results.append(dep_df)

    if not all_results:
        return pd.DataFrame(columns=[
            "hyperedge_id", "lhs", "rhs", "layer", "type", "strength", "q_score",
            "domain_size", "image_size", "relation_size"
        ])

    out = pd.concat(all_results, ignore_index=True)

    cols = [
        "hyperedge_id", "lhs", "rhs", "layer", "type", "strength", "q_score",
        "domain_size", "image_size", "relation_size"
    ]

    out = out[cols].reset_index(drop=True)
    return out


# ============================================================
# 6. Formatting helpers
# ============================================================

def format_feature_set(features) -> str:
    return "{ " + ", ".join(f'"{f}"' for f in features) + " }"


def build_text_table(rows, headers=None) -> List[str]:
    if not rows:
        return []

    left_width = max(len(str(r[0])) for r in rows)
    right_width = max(len(str(r[1])) for r in rows)

    if headers is not None:
        left_width = max(left_width, len(str(headers[0])))
        right_width = max(right_width, len(str(headers[1])))

    separator = "-" * (left_width + 3 + right_width)

    lines = []

    if headers is not None:
        header = f"{str(headers[0]).ljust(left_width)} | {str(headers[1]).ljust(right_width)}"
        lines.append(header)
        lines.append(separator)
    else:
        lines.append(separator)

    for left, right in rows:
        lines.append(f"{str(left).ljust(left_width)} | {str(right).ljust(right_width)}")

    lines.append(separator)
    return lines


def group_fd_rhs_by_lhs(fd_rules):
    grouped = defaultdict(list)

    for rule in fd_rules:
        grouped[tuple(rule["lhs"])].append(rule["rhs"])

    out = []

    for lhs, rhs_list in grouped.items():
        out.append({
            "lhs": lhs,
            "rhs_list": sorted(rhs_list)
        })

    return sorted(out, key=lambda x: (len(x["lhs"]), x["lhs"], x["rhs_list"]))


# ============================================================
# 7. Output formatting
# ============================================================

def build_pretty_hyperedge_output(
    dep_df: pd.DataFrame,
    dataset_name: str,
    hyperedge_file: str,
    n_rows: int,
    n_cols: int,
    show_fd: bool = True,
    show_ld: bool = True,
    group_rhs_for_fd: bool = True,
    show_q_for_ld: bool = True,
    show_strength_for_ld: bool = False,
) -> str:

    lines = []

    total_ld = dep_df[dep_df["type"] == "LD"].shape[0]

    fd_df = dep_df[dep_df["type"] == "FD"].copy()
    if group_rhs_for_fd and not fd_df.empty:
        total_fd = 0
        for _, h_fd in fd_df.groupby("hyperedge_id"):
            for _, layer_fd in h_fd.groupby("layer"):
                rules = [
                    {"lhs": tuple(r["lhs"]), "rhs": r["rhs"]}
                    for _, r in layer_fd.iterrows()
                ]
                total_fd += len(group_fd_rhs_by_lhs(rules))
    else:
        total_fd = fd_df.shape[0]

    lines.append(f"Check for hyperedge-based logical and functional dependencies in file '{dataset_name}'")
    lines.append("")

    summary_rows = [
        ("Data", dataset_name),
        ("Hyperedge file", Path(hyperedge_file).name),
        ("No. of rows", n_rows),
        ("No. of columns", n_cols),
    ]

    if show_ld:
        summary_rows.append(("No. of LDs", total_ld))
    if show_fd:
        summary_rows.append(("No. of FDs", total_fd))

    lines.extend(build_text_table(summary_rows, headers=None))

    if dep_df.empty:
        lines.append("")
        lines.append("No dependencies found.")
        return "\n".join(lines)

    for hyperedge_id in sorted(dep_df["hyperedge_id"].unique()):
        h_df = dep_df[dep_df["hyperedge_id"] == hyperedge_id].copy()

        lines.append("")
        lines.append("")
        lines.append("##################################################")
        lines.append(f"Hyperedge {hyperedge_id}")
        lines.append("##################################################")

        if show_ld:
            ld_df = h_df[h_df["type"] == "LD"].copy()

            if not ld_df.empty:
                lines.append("")
                lines.append("========================================")
                lines.append("Logical Dependencies (LDs)")
                lines.append("========================================")
                lines.append("")

                ld_desc_rows = [
                    ("Q = 0", "Functional dependency (FD)"),
                    ("0 < Q < 1", "Logical dependency (LD)"),
                    ("Q = 1", "No dependency"),
                    ("Lower Q", "Stronger dependency"),
                ]

                lines.extend(build_text_table(ld_desc_rows, headers=("Condition", "Implication")))
                lines.append("")
                lines.append("Note: Functional dependencies are a special case of logical dependencies.")
                lines.append("")

                for layer in sorted(ld_df["layer"].unique()):
                    layer_df = ld_df[ld_df["layer"] == layer].copy()
                    layer_df = layer_df.sort_values(by=["q_score", "lhs", "rhs"])

                    lines.append("")
                    lines.append(f"Layer {layer} ({len(layer_df)} dependencies)")
                    lines.append("-" * 40)

                    for _, row in layer_df.iterrows():
                        lhs_str = format_feature_set(tuple(row["lhs"]))
                        rhs_str = format_feature_set((row["rhs"],))

                        details = []
                        if show_q_for_ld:
                            details.append(f"q={row['q_score']:.4f}")
                        if show_strength_for_ld:
                            details.append(f"strength={row['strength']:.4f}")

                        if details:
                            lines.append(f"{lhs_str} ~> {rhs_str} ({', '.join(details)})")
                        else:
                            lines.append(f"{lhs_str} ~> {rhs_str}")

        if show_fd:
            fd_df_h = h_df[h_df["type"] == "FD"].copy()

            if not fd_df_h.empty:
                lines.append("")
                lines.append("")
                lines.append("========================================")
                lines.append("Functional Dependencies (FDs)")
                lines.append("========================================")

                for layer in sorted(fd_df_h["layer"].unique()):
                    layer_df = fd_df_h[fd_df_h["layer"] == layer].copy()

                    lines.append("")

                    if group_rhs_for_fd:
                        fd_rules = [
                            {"lhs": tuple(r["lhs"]), "rhs": r["rhs"]}
                            for _, r in layer_df.iterrows()
                        ]
                        grouped_rules = group_fd_rhs_by_lhs(fd_rules)

                        lines.append(f"Layer {layer} ({len(grouped_rules)} dependencies)")
                        lines.append("-" * 40)

                        for rule in grouped_rules:
                            lhs_str = format_feature_set(rule["lhs"])
                            rhs_str = format_feature_set(rule["rhs_list"])
                            lines.append(f"{lhs_str} -> {rhs_str}")
                    else:
                        lines.append(f"Layer {layer} ({len(layer_df)} dependencies)")
                        lines.append("-" * 40)

                        for _, row in layer_df.iterrows():
                            lhs_str = format_feature_set(tuple(row["lhs"]))
                            rhs_str = format_feature_set((row["rhs"],))
                            lines.append(f"{lhs_str} -> {rhs_str}")

    return "\n".join(lines)


def export_dependencies_as_text(
    dep_df: pd.DataFrame,
    dataset_name: str,
    hyperedge_file: str,
    output_path: Optional[str],
    n_rows: int,
    n_cols: int,
    show_fd: bool,
    show_ld: bool,
    group_rhs_for_fd: bool,
    show_q_for_ld: bool,
    show_strength_for_ld: bool,
    print_output: bool,
):
    text_output = build_pretty_hyperedge_output(
        dep_df=dep_df,
        dataset_name=dataset_name,
        hyperedge_file=hyperedge_file,
        n_rows=n_rows,
        n_cols=n_cols,
        show_fd=show_fd,
        show_ld=show_ld,
        group_rhs_for_fd=group_rhs_for_fd,
        show_q_for_ld=show_q_for_ld,
        show_strength_for_ld=show_strength_for_ld,
    )

    if print_output:
        print(text_output)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.write_text(text_output, encoding="utf-8")
        print(f"\nSaved dependency output to: {output_path.resolve()}")

    return text_output


# ============================================================
# 8. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract logical and functional dependencies within predefined hyperedges."
    )

    parser.add_argument("--input", required=True, help="Path to original input CSV file")
    parser.add_argument("--hyperedges", required=True, help="Path to hyperedge definition CSV file")
    parser.add_argument("--output", default=None, help="Path to output text file")

    parser.add_argument(
        "--type",
        default="both",
        choices=["fd", "ld", "both"],
        help="Dependency type to extract"
    )

    parser.add_argument("--max-layer", type=int, default=3, help="Maximum LHS size")
    parser.add_argument("--min-strength", type=float, default=0.8, help="Minimum dependency strength")
    parser.add_argument("--max-q", type=float, default=None, help="Maximum q-score to keep")
    parser.add_argument("--ld-threshold", type=float, default=0.2, help="Minimality threshold for LDs")

    parser.add_argument("--no-minimal", action="store_true", help="Disable minimality filtering")
    parser.add_argument("--dropna", action="store_true", help="Drop rows with missing values")
    parser.add_argument("--drop-first-column", action="store_true", help="Drop the first column from input data")
    parser.add_argument("--no-group-fd-rhs", action="store_true", help="Do not group identical FD LHS rules")
    parser.add_argument("--show-ld-strength", action="store_true", help="Show LD strength in text output")
    parser.add_argument("--hide-ld-q", action="store_true", help="Hide LD q-value in text output")
    parser.add_argument("--quiet", action="store_true", help="Do not print output to terminal")

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    hyperedge_path = Path(args.hyperedges)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not hyperedge_path.exists():
        raise FileNotFoundError(f"Hyperedge file not found: {hyperedge_path}")

    df = pd.read_csv(input_path)

    if args.drop_first_column:
        if df.shape[1] > 1:
            dropped_col = df.columns[0]
            df = df.iloc[:, 1:]
            print(f"Dropped first column: '{dropped_col}'")
        else:
            print("Warning: Cannot drop first column (only one column present)")

    deps_df = extract_dependencies_for_hyperedges(
        df=df,
        hyperedge_path=str(hyperedge_path),
        max_layer=args.max_layer,
        dependency_type=args.type,
        min_strength=args.min_strength,
        max_q=args.max_q,
        ld_improvement_threshold=args.ld_threshold,
        minimal_only=not args.no_minimal,
        dropna=args.dropna,
    )

    export_dependencies_as_text(
        dep_df=deps_df,
        dataset_name=input_path.name,
        hyperedge_file=str(hyperedge_path),
        output_path=args.output,
        n_rows=len(df),
        n_cols=df.shape[1],
        show_fd=args.type in ["fd", "both"],
        show_ld=args.type in ["ld", "both"],
        group_rhs_for_fd=not args.no_group_fd_rhs,
        show_q_for_ld=not args.hide_ld_q,
        show_strength_for_ld=args.show_ld_strength,
        print_output=not args.quiet,
    )


if __name__ == "__main__":
    main()