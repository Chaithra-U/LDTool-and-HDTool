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


def compute_q_for_rule(df: pd.DataFrame, lhs: List[str], rhs: str) -> dict:
    domain_size = df[lhs].drop_duplicates().shape[0]
    image_size = df[rhs].drop_duplicates().shape[0]
    relation_size = df[lhs + [rhs]].drop_duplicates().shape[0]

    q = q_score_from_counts(domain_size, relation_size, image_size)
    strength = 1.0 - q
    is_fd = np.isclose(q, 0.0)

    return {
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
    """
    Exact FD check:
    lhs -> rhs holds if each unique lhs-value combination maps to only one rhs value.
    """
    max_rhs_values = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False).max()
    return bool(max_rhs_values == 1)


def extract_fd_dependencies(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_layer: int = 3,
    minimal_only: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Faster FD-only extraction.

    This avoids computing full Q statistics for every candidate.
    If minimal_only=True, exact classical FD minimality is applied:
    X -> y is skipped if a proper subset of X is already an FD for y.
    """
    if features is None:
        features = list(df.columns)
    else:
        features = list(features)

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
                    has_fd_subset = any(set(prev).issubset(lhs_tuple) for prev in kept_fd_lhs)
                    if has_fd_subset:
                        continue

                if is_exact_fd(work_df, list(lhs), rhs):
                    domain_size = work_df[list(lhs)].drop_duplicates().shape[0]
                    image_size = work_df[rhs].drop_duplicates().shape[0]

                    rows.append({
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
            by=["rhs", "layer", "lhs"],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    return out


# ============================================================
# 3. Unified Q-based candidate extraction
# ============================================================

def extract_candidate_dependencies(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_layer: int = 3,
    min_strength: float = 0.8,
    max_q: Optional[float] = None,
    dropna: bool = False,
) -> pd.DataFrame:
    if features is None:
        features = list(df.columns)
    else:
        features = list(features)

    work_df = df[features].copy()
    if dropna:
        work_df = work_df.dropna()

    rows = []

    for rhs in features:
        lhs_candidates = [c for c in features if c != rhs]
        max_k = min(max_layer, len(lhs_candidates))

        for k in range(1, max_k + 1):
            for lhs in combinations(lhs_candidates, k):
                rule = compute_q_for_rule(work_df, list(lhs), rhs)

                if rule["strength"] >= min_strength:
                    if max_q is None or rule["q_score"] <= max_q:
                        rows.append(rule)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["rhs", "layer", "q_score", "strength", "lhs"],
            ascending=[True, True, True, False, True]
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

    for rhs in sorted(dep_df["rhs"].unique()):
        rhs_df = dep_df[dep_df["rhs"] == rhs].copy()
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
            by=["type", "rhs", "layer", "q_score", "strength", "lhs"],
            ascending=[True, True, True, True, False, True]
        ).reset_index(drop=True)

    return out


# ============================================================
# 5. Main extraction function
# ============================================================

def extract_dependencies(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_layer: int = 3,
    dependency_type: str = "both",
    min_strength: float = 0.8,
    max_q: Optional[float] = None,
    ld_improvement_threshold: float = 0.2,
    minimal_only: bool = True,
    include_stats: bool = True,
    dataset_name: str = "dataset",
    dropna: bool = False,
) -> pd.DataFrame:

    dependency_type = dependency_type.lower()

    if dependency_type == "fd":
        dep_df = extract_fd_dependencies(
            df=df,
            features=features,
            max_layer=max_layer,
            minimal_only=minimal_only,
            dropna=dropna,
        )
    else:
        dep_df = extract_candidate_dependencies(
            df=df,
            features=features,
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

    dep_df["dataset"] = dataset_name

    if dep_df.empty:
        return dep_df

    if include_stats:
        cols = [
            "dataset", "lhs", "rhs", "layer", "type", "strength", "q_score",
            "domain_size", "image_size", "relation_size"
        ]
    else:
        cols = ["dataset", "lhs", "rhs", "layer", "type", "strength"]

    dep_df = dep_df[cols].reset_index(drop=True)
    return dep_df


# ============================================================
# 6. Layer-wise grouping
# ============================================================

def group_dependencies_by_layer(dep_df: pd.DataFrame):
    result = {"FD": {}, "LD": {}}

    if dep_df.empty:
        return result

    for dep_type in ["FD", "LD"]:
        sub = dep_df[dep_df["type"] == dep_type].copy()
        if sub.empty:
            continue

        for layer in sorted(sub["layer"].unique()):
            layer_df = sub[sub["layer"] == layer].copy()
            rules = []

            for _, row in layer_df.iterrows():
                rule = {
                    "lhs": tuple(row["lhs"]),
                    "rhs": row["rhs"],
                }

                if dep_type == "LD":
                    if "q_score" in row.index:
                        rule["q_score"] = float(row["q_score"])
                    if "strength" in row.index:
                        rule["strength"] = float(row["strength"])

                rules.append(rule)

            result[dep_type][int(layer)] = rules

    return result


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

    out = sorted(out, key=lambda x: (len(x["lhs"]), x["lhs"], x["rhs_list"]))
    return out


# ============================================================
# 7. Pretty formatting
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


def build_pretty_dependency_output(
    dep_dict,
    dataset_name: str = "dataset.csv",
    n_rows: int = None,
    n_cols: int = None,
    show_fd: bool = True,
    show_ld: bool = True,
    group_rhs_for_fd: bool = True,
    show_q_for_ld: bool = True,
    show_strength_for_ld: bool = False,
) -> str:
    lines = []

    raw_ld_count = sum(len(v) for v in dep_dict.get("LD", {}).values())

    if group_rhs_for_fd:
        grouped_fd_count = sum(
            len(group_fd_rhs_by_lhs(v)) for v in dep_dict.get("FD", {}).values()
        )
    else:
        grouped_fd_count = sum(len(v) for v in dep_dict.get("FD", {}).values())

    if show_fd and not show_ld:
        check_text = "functional dependencies"
    elif show_ld and not show_fd:
        check_text = "logical dependencies"
    else:
        check_text = "logical and functional dependencies"

    lines.append(f"Check for {check_text} in file '{dataset_name}'")
    lines.append("")

    summary_rows = [
        ("Data", dataset_name),
        ("No. of rows", n_rows if n_rows is not None else "N/A"),
        ("No. of columns", n_cols if n_cols is not None else "N/A"),
    ]

    if show_ld:
        summary_rows.append(("No. of LDs", raw_ld_count))
    if show_fd:
        summary_rows.append(("No. of FDs", grouped_fd_count))

    lines.extend(build_text_table(summary_rows, headers=None))

    if show_ld and dep_dict.get("LD"):
        lines.append("")
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

        for layer in sorted(dep_dict["LD"].keys()):
            ld_rules = dep_dict["LD"][layer]

            lines.append("")
            lines.append(f"Layer {layer} ({len(ld_rules)} dependencies)")
            lines.append("-" * 40)

            ld_rules = sorted(ld_rules, key=lambda r: (len(r["lhs"]), r["lhs"], r["rhs"]))

            for rule in ld_rules:
                lhs_str = format_feature_set(rule["lhs"])
                rhs_str = format_feature_set((rule["rhs"],))

                details = []

                if show_q_for_ld and "q_score" in rule:
                    details.append(f"q={rule['q_score']:.4f}")

                if show_strength_for_ld and "strength" in rule:
                    details.append(f"strength={rule['strength']:.4f}")

                if details:
                    lines.append(f"{lhs_str} ~> {rhs_str} ({', '.join(details)})")
                else:
                    lines.append(f"{lhs_str} ~> {rhs_str}")

    if show_fd and dep_dict.get("FD"):
        lines.append("")
        lines.append("")
        lines.append("========================================")
        lines.append("Functional Dependencies (FDs)")
        lines.append("========================================")

        for layer in sorted(dep_dict["FD"].keys()):
            fd_rules = dep_dict["FD"][layer]

            lines.append("")
            grouped_rules = group_fd_rhs_by_lhs(fd_rules) if group_rhs_for_fd else fd_rules
            lines.append(f"Layer {layer} ({len(grouped_rules)} dependencies)")
            lines.append("-" * 40)

            if group_rhs_for_fd:
                for rule in grouped_rules:
                    lhs_str = format_feature_set(rule["lhs"])
                    rhs_str = format_feature_set(rule["rhs_list"])
                    lines.append(f"{lhs_str} -> {rhs_str}")
            else:
                for rule in fd_rules:
                    lhs_str = format_feature_set(rule["lhs"])
                    rhs_str = format_feature_set((rule["rhs"],))
                    lines.append(f"{lhs_str} -> {rhs_str}")

    return "\n".join(lines)


def export_dependencies_as_text(
    dep_df: pd.DataFrame,
    dataset_name: str = "dataset.csv",
    output_path: Optional[str] = None,
    n_rows: int = None,
    n_cols: int = None,
    show_fd: bool = True,
    show_ld: bool = True,
    group_rhs_for_fd: bool = True,
    show_q_for_ld: bool = True,
    show_strength_for_ld: bool = False,
    print_output: bool = True,
):
    dep_dict = group_dependencies_by_layer(dep_df)

    text_output = build_pretty_dependency_output(
        dep_dict=dep_dict,
        dataset_name=dataset_name,
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

    return dep_dict, text_output


# ============================================================
# 8. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multi-attribute logical and functional dependencies from a CSV file."
    )

    parser.add_argument("--input", required=True, help="Path to input CSV file")
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
    parser.add_argument("--no-group-fd-rhs", action="store_true", help="Do not group identical FD LHS rules")
    parser.add_argument("--show-ld-strength", action="store_true", help="Show LD strength in text output")
    parser.add_argument("--hide-ld-q", action="store_true", help="Hide LD q-value in text output")
    parser.add_argument("--quiet", action="store_true", help="Do not print output to terminal")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of features/columns to analyze"
    )
    parser.add_argument(
        "--drop-first-column",
        action="store_true",
        help="Drop the first column (useful for ID columns)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.drop_first_column:
        if df.shape[1] > 1:
            dropped_col = df.columns[0]
            df = df.iloc[:, 1:]
            print(f"Dropped first column: '{dropped_col}'")
        else:
            print("Warning: Cannot drop first column (only one column present)")

    deps_df = extract_dependencies(
        df=df,
        features=args.features,
        max_layer=args.max_layer,
        dependency_type=args.type,
        min_strength=args.min_strength,
        max_q=args.max_q,
        ld_improvement_threshold=args.ld_threshold,
        minimal_only=not args.no_minimal,
        include_stats=True,
        dataset_name=input_path.name,
        dropna=args.dropna,
    )

    export_dependencies_as_text(
        dep_df=deps_df,
        dataset_name=input_path.name,
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