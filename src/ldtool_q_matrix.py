import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd


# ============================================================
# 1. Q-matrix extraction: Layer 1 only
# ============================================================

def extract_dependencies_from_q_matrix(
    q_matrix_path: str,
    dependency_type: str = "both",
    min_strength: float = 0.8,
    max_q: Optional[float] = None,
    dataset_name: str = "q_matrix",
) -> pd.DataFrame:
    """
    Extract Layer 1 dependencies from an external pairwise Q-matrix.

    Expected CSV format:
        - first column contains row/index attribute names (LHS)
        - remaining columns are RHS attribute names
        - each cell contains Q(lhs -> rhs)

    Q interpretation:
        Q = 0       -> functional dependency (FD)
        0 < Q < 1   -> logical dependency (LD)
        Q = 1       -> no dependency
    """
    q_df = pd.read_csv(q_matrix_path, index_col=0)
    rows = []

    dependency_type = dependency_type.lower()

    for lhs in q_df.index:
        for rhs in q_df.columns:
            if lhs == rhs:
                continue

            q_val = q_df.loc[lhs, rhs]

            if pd.isna(q_val):
                continue

            q = float(q_val)
            q = max(0.0, min(1.0, q))

            if np.isclose(q, 1.0):
                continue

            strength = 1.0 - q
            is_fd = np.isclose(q, 0.0)
            rule_type = "FD" if is_fd else "LD"

            if dependency_type == "fd" and rule_type != "FD":
                continue
            if dependency_type == "ld" and rule_type != "LD":
                continue
            if dependency_type not in ["fd", "ld", "both"]:
                raise ValueError("dependency_type must be one of: 'fd', 'ld', 'both'")

            if strength < min_strength:
                continue

            if max_q is not None and q > max_q:
                continue

            rows.append({
                "dataset": dataset_name,
                "lhs": (lhs,),
                "rhs": rhs,
                "layer": 1,
                "type": rule_type,
                "strength": strength,
                "q_score": q,
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["type", "rhs", "q_score", "strength", "lhs"],
            ascending=[False, True, True, False, True]
        ).reset_index(drop=True)

    return out


# ============================================================
# 2. Grouping helpers
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
                    rule["q_score"] = float(row["q_score"])
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

    return sorted(out, key=lambda x: (len(x["lhs"]), x["lhs"], x["rhs_list"]))


# ============================================================
# 3. Formatting helpers
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
    dep_df: pd.DataFrame,
    dataset_name: str,
    n_attributes: int,
    show_fd: bool = True,
    show_ld: bool = True,
    group_rhs_for_fd: bool = True,
    show_q_for_ld: bool = True,
    show_strength_for_ld: bool = False,
) -> str:
    dep_dict = group_dependencies_by_layer(dep_df)

    raw_ld_count = sum(len(v) for v in dep_dict.get("LD", {}).values())

    if group_rhs_for_fd:
        grouped_fd_count = sum(
            len(group_fd_rhs_by_lhs(v)) for v in dep_dict.get("FD", {}).values()
        )
    else:
        grouped_fd_count = sum(len(v) for v in dep_dict.get("FD", {}).values())

    lines = []
    lines.append(f"Check for first-layer logical and functional dependencies in Q-matrix '{dataset_name}'")
    lines.append("")

    summary_rows = [
        ("Data", dataset_name),
        ("No. of attributes", n_attributes),
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

            ld_rules = sorted(ld_rules, key=lambda r: (r["q_score"], r["lhs"], r["rhs"]))

            for rule in ld_rules:
                lhs_str = format_feature_set(rule["lhs"])
                rhs_str = format_feature_set((rule["rhs"],))

                details = []

                if show_q_for_ld:
                    details.append(f"q={rule['q_score']:.4f}")

                if show_strength_for_ld:
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


# ============================================================
# 4. Export
# ============================================================

def export_dependencies_as_text(
    dep_df: pd.DataFrame,
    q_matrix_path: str,
    output_path: Optional[str] = None,
    show_fd: bool = True,
    show_ld: bool = True,
    group_rhs_for_fd: bool = True,
    show_q_for_ld: bool = True,
    show_strength_for_ld: bool = False,
    print_output: bool = True,
):
    q_df = pd.read_csv(q_matrix_path, index_col=0)

    text_output = build_pretty_dependency_output(
        dep_df=dep_df,
        dataset_name=Path(q_matrix_path).name,
        n_attributes=len(q_df.columns),
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
# 5. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract first-layer logical and functional dependencies from an external pairwise Q-matrix."
    )

    parser.add_argument("--q-matrix", required=True, help="Path to external pairwise Q-matrix CSV")
    parser.add_argument("--output", default=None, help="Path to output text file")

    parser.add_argument(
        "--type",
        default="both",
        choices=["fd", "ld", "both"],
        help="Dependency type to extract"
    )

    parser.add_argument("--min-strength", type=float, default=0.8, help="Minimum dependency strength")
    parser.add_argument("--max-q", type=float, default=None, help="Maximum Q-value to keep")

    parser.add_argument("--no-group-fd-rhs", action="store_true", help="Do not group identical FD LHS rules")
    parser.add_argument("--show-ld-strength", action="store_true", help="Show LD strength in text output")
    parser.add_argument("--hide-ld-q", action="store_true", help="Hide LD Q-value in text output")
    parser.add_argument("--quiet", action="store_true", help="Do not print output to terminal")

    return parser.parse_args()


def main():
    args = parse_args()

    q_matrix_path = Path(args.q_matrix)

    if not q_matrix_path.exists():
        raise FileNotFoundError(f"Q-matrix file not found: {q_matrix_path}")

    deps_df = extract_dependencies_from_q_matrix(
        q_matrix_path=str(q_matrix_path),
        dependency_type=args.type,
        min_strength=args.min_strength,
        max_q=args.max_q,
        dataset_name=q_matrix_path.name,
    )

    export_dependencies_as_text(
        dep_df=deps_df,
        q_matrix_path=str(q_matrix_path),
        output_path=args.output,
        show_fd=args.type in ["fd", "both"],
        show_ld=args.type in ["ld", "both"],
        group_rhs_for_fd=not args.no_group_fd_rhs,
        show_q_for_ld=not args.hide_ld_q,
        show_strength_for_ld=args.show_ld_strength,
        print_output=not args.quiet,
    )


if __name__ == "__main__":
    main()