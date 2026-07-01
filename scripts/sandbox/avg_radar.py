"""
Create averaged transformer-embedding prediction outputs.

Outputs produced:
1. avg_stats_across_embedding_models.csv
   - Same descriptor-level format as the input prediction CSVs, with numeric
     metrics averaged across the selected transformer encoder models.

2. descriptor_r2_by_model_with_average.csv
   - Regression descriptors only, with one r2 column per transformer model and
     an Average_r2 column across models.

3. group_performance_by_task.csv
   - Descriptor-group performance for regression, binary classification, and
     multiclass classification using the averaged embedding-model results.

4. bar_plots_by_group_and_task/
   - Separate bar plots for each descriptor group and task type, if populated:
     regression: r2
     binary classification: Balanced_Accuracy
     multiclass classification: Balanced_Accuracy

5. bar_plots_predicted_leq_0p5/
   - Bar plots highlighting descriptors performing at/below a threshold.

6. group_fraction_plots_by_task/
   - Stacked bar plots showing what fraction of each task's descriptors fall
     into each descriptor group, split by high/low performance.

This script follows the pathing/grouping conventions used in the existing
cross-feature analysis scripts.

---------------------------------------------------------------------------
Fixes applied in this revision (see inline comments for details):
---------------------------------------------------------------------------
- BUG: Binary classification descriptors were not appearing in any plots.
  Root cause was twofold:
    (a) `average_model_results` merged per-model CSVs on raw column names.
        If any transformer model's CSV used different casing for a metric
        column (e.g. "balanced_accuracy" vs "Balanced_Accuracy"), pandas
        treated them as two different columns, so the merged column was
        mostly NaN (only populated by whichever models happened to match
        the exact casing used downstream) -> classification rows looked
        empty and got filtered out everywhere.
    (b) `filter_task_rows` matched `task_type` values with an exact string
        equality check and had no tolerance for label variants (e.g. a CSV
        using "classification" instead of "binary_classification"). Its
        metric-based fallback also used metric lists that overlap between
        "classification" and "multiclass" (both share Accuracy,
        Balanced_Accuracy, MCC), which made the fallback unreliable.
  Fix: metric columns are now canonicalised (case-insensitively) at load
  time before averaging, and task_type matching now accepts common label
  aliases. The metric-based fallback uses each task's full (broad) metric
  list to decide inclusion, and only uses task-unique metrics (e.g.
  Sensitivity/Specificity for binary, F1_macro/AUC_OVR for multiclass) to
  EXCLUDE rows that clearly belong to the other task - not to require
  their presence. An earlier version of this fix required unique metrics
  for inclusion, which incorrectly dropped binary-classification
  descriptors that had Accuracy/AUC/Balanced_Accuracy populated but no
  Sensitivity/Specificity/PPV/NPV values.
- BUG: `plot_group_task_fraction_bars` referenced an undefined variable
  `metsubsetric` (a typo for `metric`), which would raise a NameError and
  crash the script the first time that plot had data to draw. Fixed.
- Deduplicated the three near-identical bar-plotting code blocks
  (regular group plots, low-performing plots, and the combined
  low-performing plot) into a single `render_bar_plot` helper.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

def add_project_imports(project_root: Path | None) -> None:
    """Add project src paths for getPaths/getGroups/getLowVarianceColumns."""
    if project_root is None:
        candidates = [
            Path("/users/yhb18174/TL_project/scripts/src/pathing"),
            Path("/users/yhb18174/TL_project/scripts/src/datasets"),
        ]
    else:
        candidates = [
            project_root / "scripts" / "src" / "pathing",
            project_root / "scripts" / "src" / "datasets",
        ]

    for p in candidates:
        sys.path.insert(0, str(p))


# -----------------------------------------------------------------------------
# Task configuration
# -----------------------------------------------------------------------------

# Canonical task_type values this script writes into its own outputs.
TASK_TYPE_MAP = {
    "regression": "regression",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
}

# Accepted variants for each task's `task_type` label when *reading* input
# CSVs. Matching is case-insensitive and whitespace-trimmed. This makes the
# script tolerant of upstream scripts using slightly different conventions
# (e.g. "classification" vs "binary_classification").
TASK_TYPE_ALIASES = {
    "regression": {"regression", "reg"},
    "classification": {
        "classification",
        "binary_classification",
        "binary",
        "binary_class",
        "bin_class",
    },
    "multiclass": {
        "multiclass",
        "multiclass_classification",
        "multi_class",
        "multiclass_class",
    },
}

TASK_METRICS = {
    "regression": ["Pearson_r", "r2", "RMSE", "Bias", "MSE", "SDEP"],
    "classification": [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "AUC",
        "MCC",
        "Balanced_Accuracy",
    ],
    "multiclass": [
        "Accuracy",
        "Balanced_Accuracy",
        "F1_macro",
        "AUC_OVR",
        "MCC",
    ],
}

# Every metric name this script knows about, used to canonicalise column
# casing across the different transformer-model CSVs (see
# `normalize_metric_columns`).
ALL_KNOWN_METRICS = sorted({m for metrics in TASK_METRICS.values() for m in metrics})

SORT_METRIC_BY_TASK = {
    "regression": "r2",
    "classification": "AUC",
    "multiclass": "Balanced_Accuracy",
}

PLOT_METRIC_BY_TASK = {
    "regression": "r2",
    "classification": "Balanced_Accuracy",
    "multiclass": "Balanced_Accuracy",
}

PLOT_LABEL_BY_TASK = {
    "regression": "Regression r2",
    "classification": "Binary classification balanced accuracy",
    "multiclass": "Multiclass balanced accuracy",
}


# -----------------------------------------------------------------------------
# Small generic helpers
# -----------------------------------------------------------------------------

def safe_name(name: str) -> str:
    """Make a safe filename from a descriptor group name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def clean_descriptor_label(label: str) -> str:
    """Remove common descriptor suffixes for cleaner plot labels."""
    label = str(label)
    for suffix in ("_mordred", "_rdkit"):
        label = label.replace(suffix, "")
    return label


def find_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    """
    Case-insensitive lookup of a metric column's actual name in `df`.

    Returns the real column name (preserving its original casing) if a
    case-insensitive match is found, otherwise None.
    """
    lookup = {c.lower(): c for c in df.columns}
    return lookup.get(metric.lower())


def get_metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    Fetch a metric column as numeric data, matched case-insensitively.

    Returns an all-NaN series (aligned to df's index) if the metric isn't
    present under any casing, so callers can use this safely without extra
    existence checks.
    """
    col = find_metric_column(df, metric)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def available_metrics(df: pd.DataFrame, wanted: Iterable[str]) -> list[str]:
    """Return the canonical metric names from `wanted` that are present in
    df as columns, matched case-insensitively."""
    lookup = {c.lower() for c in df.columns}
    return [m for m in wanted if m.lower() in lookup]


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to their canonical metric name where they match a known
    metric case-insensitively (e.g. 'balanced_accuracy' -> 'Balanced_Accuracy').

    Without this, the same metric produced by different transformer-model
    CSVs with inconsistent casing gets treated as two separate columns
    during averaging, silently dropping most of the data for that metric.
    """
    canonical_lookup = {m.lower(): m for m in ALL_KNOWN_METRICS}
    rename_map = {
        col: canonical_lookup[col.lower()]
        for col in df.columns
        if col.lower() in canonical_lookup and col != canonical_lookup[col.lower()]
    }
    return df.rename(columns=rename_map) if rename_map else df


def filter_task_rows(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """
    Filter rows belonging to a given task ("regression" / "classification" /
    "multiclass") using the `task_type` column, tolerant of casing and
    common label variants (see TASK_TYPE_ALIASES).

    Raises a clear error if `task_type` is missing entirely, rather than
    silently guessing from which metric columns happen to be populated -
    metric-based guessing was unreliable (classification and multiclass
    share several metric names) and previously caused descriptors to be
    dropped or misclassified.
    """
    if "task_type" not in df.columns:
        raise KeyError(
            "'task_type' column not found - cannot classify descriptors by "
            "task without it. Check that the prediction CSVs include a "
            "task_type column."
        )

    aliases = TASK_TYPE_ALIASES[task_name]
    normalized = df["task_type"].astype(str).str.strip().str.lower()
    return df.loc[normalized.isin(aliases)].copy()


def resolve_results_dir(paths: dict, result_dir_key: str):
    """Resolve a prediction-output directory dictionary from paths."""
    try:
        return paths["prediction_output_dirs"][result_dir_key]
    except KeyError as exc:
        valid = list(paths.get("prediction_output_dirs", {}).keys())
        raise KeyError(
            f"Could not find result-dir key '{result_dir_key}'. Valid keys: {valid}"
        ) from exc


def load_model_results(
    results_dir: dict,
    pred: str,
    models: list[str],
) -> dict[str, pd.DataFrame]:
    """Load prediction CSVs for each transformer model, with metric column
    names canonicalised so per-model casing differences don't break later
    averaging."""
    loaded = {}
    missing = []

    for model in models:
        exp = f"pred_{pred}_tr_{model}"
        if exp not in results_dir:
            missing.append(f"{model}: missing experiment key {exp}")
            continue

        csv_path = Path(results_dir[exp]) / f"{exp}.csv"
        if not csv_path.exists():
            missing.append(f"{model}: missing file {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0)
        loaded[model] = normalize_metric_columns(df)

    if missing:
        print("Warning: some model files were not loaded:")
        for m in missing:
            print(f"  - {m}")

    if not loaded:
        raise ValueError("No model result CSVs were loaded.")

    return loaded


def average_model_results(model_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Average numeric columns across model result dataframes and keep metadata.

    Metadata columns are taken as the first non-null value per descriptor.
    """
    combined = pd.concat(model_results, names=["model", "descriptor"], axis=0)

    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    metadata_cols = [c for c in combined.columns if c not in numeric_cols]

    avg_numeric = combined[numeric_cols].groupby(level="descriptor").mean(numeric_only=True)

    if metadata_cols:
        metadata = combined[metadata_cols].groupby(level="descriptor").first()
        avg_df = pd.concat([metadata, avg_numeric], axis=1)
    else:
        avg_df = avg_numeric

    return avg_df


def make_descriptor_r2_table(model_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a descriptor-level r2 table for each model plus the average."""
    r2_frames = []

    for model, df in model_results.items():
        if find_metric_column(df, "r2") is None:
            continue

        reg_df = filter_task_rows(df, "regression")
        s = get_metric_series(reg_df if not reg_df.empty else df, "r2")
        if reg_df.empty:
            s = s.loc[s.notna()]
        r2_frames.append(s.rename(model))

    if not r2_frames:
        raise ValueError("No r2 columns were available for regression descriptors.")

    out = pd.concat(r2_frames, axis=1)
    out["Average_r2"] = out.mean(axis=1, skipna=True)
    out["n_models_present"] = out.drop(columns=["Average_r2"]).notna().sum(axis=1)
    out = out.sort_values("Average_r2", ascending=False)
    return out


def group_descriptors_long(
    avg_df: pd.DataFrame,
    group_map: dict[str, list[str]],
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute descriptor-group averages for each task and metric."""
    exclude_cols = set(exclude_cols or [])
    rows = []

    for task_name, metrics in TASK_METRICS.items():
        task_df = filter_task_rows(avg_df, task_name)
        if task_df.empty:
            continue

        present_metrics = available_metrics(task_df, metrics)
        if not present_metrics:
            continue

        # Build a small numeric frame of just the metrics we need, matched
        # case-insensitively, so downstream indexing is safe regardless of
        # the original CSVs' column casing.
        metric_frame = pd.DataFrame(
            {m: get_metric_series(task_df, m) for m in present_metrics},
            index=task_df.index,
        )

        for group_name, members in group_map.items():
            present_members = [
                m for m in members
                if m in task_df.index and m not in exclude_cols
            ]
            if not present_members:
                continue

            subset = metric_frame.loc[present_members]
            if subset.dropna(how="all").empty:
                continue

            row = {
                "task": task_name,
                "task_type": TASK_TYPE_MAP[task_name],
                "descriptor_group": group_name,
                "n_descriptors": len(present_members),
            }

            for metric in present_metrics:
                row[f"avg_{metric}"] = subset[metric].mean(skipna=True)
                row[f"median_{metric}"] = subset[metric].median(skipna=True)

            sort_metric = SORT_METRIC_BY_TASK[task_name]
            sort_col = f"avg_{sort_metric}"
            row["primary_metric"] = sort_metric
            row["primary_metric_value"] = row.get(sort_col, np.nan)

            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["task", "primary_metric_value"], ascending=[True, False])
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def get_dynamic_bar_plot_layout(n_labels: int):
    """Return dynamic figure settings for readable descriptor bar plots,
    scaling figure size / label size / rotation with the number of bars."""
    if n_labels <= 10:
        fig_width, fig_height, label_size, rotation = 10, 5.5, 10, 45
        bottom_margin, bar_width = 0.28, 0.75
    elif n_labels <= 25:
        fig_width, fig_height, label_size, rotation = max(12, 0.55 * n_labels), 6.0, 9, 60
        bottom_margin, bar_width = 0.35, 0.75
    elif n_labels <= 60:
        fig_width, fig_height, label_size, rotation = max(16, 0.38 * n_labels), 7.0, 7, 90
        bottom_margin, bar_width = 0.42, 0.70
    elif n_labels <= 120:
        fig_width, fig_height, label_size, rotation = max(22, 0.30 * n_labels), 8.0, 5.5, 90
        bottom_margin, bar_width = 0.48, 0.65
    else:
        fig_width, fig_height, label_size, rotation = max(32, 0.24 * n_labels), 9.0, 4.5, 90
        bottom_margin, bar_width = 0.52, 0.60

    fig_width = min(fig_width, 60)  # avoid absurdly huge figures
    return fig_width, fig_height, label_size, rotation, bottom_margin, bar_width


def render_bar_plot(
    labels: list[str],
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    ylim: tuple[float, float] | None = (0, 1.05),
    threshold_line: float | None = None,
) -> None:
    """
    Render and save a single-series descriptor bar plot with dynamic sizing.

    Shared by the regular per-group/per-task plots and the low-performing
    descriptor plots, which previously duplicated this logic three times.
    """
    n_labels = len(labels)
    x = np.arange(n_labels)
    fig_width, fig_height, label_size, rotation, bottom_margin, bar_width = (
        get_dynamic_bar_plot_layout(n_labels)
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(x, values, width=bar_width)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if threshold_line is not None:
        ax.axhline(threshold_line, linestyle="--", linewidth=1)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Descriptor", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=rotation, ha="right" if rotation else "center", fontsize=label_size
    )
    ax.tick_params(axis="y", labelsize=9)
    # Numeric x positions (rather than categorical labels) so duplicate
    # cleaned labels never collapse onto the same bar.
    ax.set_xlim(-0.75, n_labels - 0.25)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=bottom_margin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_task_bars(
    avg_df: pd.DataFrame,
    group_map: dict[str, list[str]],
    out_dir: Path,
    min_members: int = 1,
) -> None:
    """
    Create separate bar plots for each descriptor group and task type:
    regression -> r2, classification -> Balanced_Accuracy,
    multiclass -> Balanced_Accuracy.

    A plot is only created when that group has at least one descriptor
    populated for that task/metric.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_name, metric in PLOT_METRIC_BY_TASK.items():
        task_df = filter_task_rows(avg_df, task_name)

        if task_df.empty or find_metric_column(task_df, metric) is None:
            print(f"Skipping {task_name}: no {metric} values available.")
            continue

        metric_values = get_metric_series(task_df, metric)
        task_out_dir = out_dir / task_name
        task_out_dir.mkdir(parents=True, exist_ok=True)

        for group_name, members in group_map.items():
            present_members = [
                m for m in members
                if m in metric_values.index and pd.notna(metric_values.loc[m])
            ]
            if len(present_members) < min_members:
                continue

            plot_df = metric_values.reindex(present_members).dropna()
            if plot_df.empty:
                continue

            plot_df = plot_df.sort_values(ascending=False)
            labels = [clean_descriptor_label(x) for x in plot_df.index]

            out_path = (
                task_out_dir
                / f"{safe_name(group_name)}_{task_name}_{safe_name(metric)}_bar.png"
            )
            render_bar_plot(
                labels=labels,
                values=plot_df.values,
                ylabel=metric,
                title=f"{group_name}: {PLOT_LABEL_BY_TASK[task_name]} ({len(labels)} descriptors)",
                out_path=out_path,
            )
            print(
                f"Saved {task_name} plot for {group_name}: "
                f"{len(labels)} descriptors -> {out_path}"
            )


def plot_low_performing_descriptors(
    avg_df: pd.DataFrame,
    group_map: dict[str, list[str]],
    out_dir: Path,
    threshold: float = 0.5,
) -> None:
    """
    Create bar plots for descriptors with averaged performance <= threshold.

    Creates:
    1. One combined plot per task type containing all descriptors <= threshold.
    2. One plot per descriptor group per task type, if that group has
       descriptors <= threshold.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reverse lookup: descriptor -> descriptor group(s)
    descriptor_to_groups: dict[str, list[str]] = {}
    for group_name, members in group_map.items():
        for member in members:
            descriptor_to_groups.setdefault(member, []).append(group_name)

    for task_name, metric in PLOT_METRIC_BY_TASK.items():
        task_df = filter_task_rows(avg_df, task_name)

        if task_df.empty or find_metric_column(task_df, metric) is None:
            print(f"Skipping low-performing {task_name}: no {metric} values available.")
            continue

        metric_values = get_metric_series(task_df, metric)
        low_df = task_df.copy()
        low_df[metric] = metric_values
        low_df = low_df.loc[low_df[metric].notna() & (low_df[metric] <= threshold)].copy()

        if low_df.empty:
            print(f"No {task_name} descriptors with {metric} <= {threshold}.")
            continue

        low_df["descriptor_group"] = [
            ";".join(descriptor_to_groups.get(idx, ["Unmapped"])) for idx in low_df.index
        ]
        low_df = low_df.sort_values(metric, ascending=True)

        task_out_dir = out_dir / task_name
        task_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Combined low-performing plot for this task type.
        labels = [
            f"{clean_descriptor_label(idx)} [{grp}]"
            for idx, grp in zip(low_df.index, low_df["descriptor_group"])
        ]
        values = low_df[metric].values
        combined_path = (
            task_out_dir
            / f"ALL_{task_name}_{safe_name(metric)}_leq_{str(threshold).replace('.', 'p')}_bar.png"
        )
        render_bar_plot(
            labels=labels,
            values=values,
            ylabel=metric,
            title=f"All {task_name} descriptors with {metric} <= {threshold} ({len(labels)} descriptors)",
            out_path=combined_path,
            ylim=(min(0, np.nanmin(values) - 0.05), max(threshold + 0.05, np.nanmax(values) + 0.05)),
            threshold_line=threshold,
        )
        print(f"Saved combined low-performing {task_name} plot: {len(labels)} descriptors -> {combined_path}")

        low_df.to_csv(
            task_out_dir
            / f"ALL_{task_name}_{safe_name(metric)}_leq_{str(threshold).replace('.', 'p')}.csv"
        )

        # 2. Group-specific low-performing plots for this task type.
        for group_name, members in group_map.items():
            present_members = [m for m in members if m in low_df.index]
            if not present_members:
                continue

            group_low_df = low_df.loc[present_members].sort_values(metric, ascending=True)
            labels = [clean_descriptor_label(x) for x in group_low_df.index]
            values = group_low_df[metric].values

            group_path = (
                task_out_dir
                / f"{safe_name(group_name)}_{task_name}_{safe_name(metric)}_leq_{str(threshold).replace('.', 'p')}_bar.png"
            )
            render_bar_plot(
                labels=labels,
                values=values,
                ylabel=metric,
                title=f"{group_name}: {task_name} descriptors with {metric} <= {threshold} ({len(labels)} descriptors)",
                out_path=group_path,
                ylim=(min(0, np.nanmin(values) - 0.05), max(threshold + 0.05, np.nanmax(values) + 0.05)),
                threshold_line=threshold,
            )
            print(
                f"Saved low-performing {task_name} plot for {group_name}: "
                f"{len(labels)} descriptors -> {group_path}"
            )


def get_group_task_fraction_summary(
    avg_df: pd.DataFrame,
    group_map: dict[str, list[str]],
    threshold: float = 0.7,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a group/task summary table: for each task, what fraction of its
    descriptors fall into each descriptor group, and what fraction of those
    are at/below `threshold`.
    """
    exclude_cols = set(exclude_cols or [])
    rows = []

    for task_name, metric in PLOT_METRIC_BY_TASK.items():
        task_df = filter_task_rows(avg_df, task_name)

        if task_df.empty or find_metric_column(task_df, metric) is None:
            print(f"Skipping group fraction summary for {task_name}: no {metric}.")
            continue

        metric_values = get_metric_series(task_df, metric)
        valid_descriptors = [
            idx for idx in task_df.index
            if idx not in exclude_cols and pd.notna(metric_values.loc[idx])
        ]

        total_task_descriptors = len(valid_descriptors)
        if total_task_descriptors == 0:
            print(f"Skipping group fraction summary for {task_name}: no valid descriptors.")
            continue

        valid_set = set(valid_descriptors)

        for group_name, members in group_map.items():
            present_members = [m for m in members if m in valid_set]
            if not present_members:
                continue

            member_values = metric_values.reindex(present_members)
            n_group_task = len(present_members)
            n_low = int((member_values <= threshold).sum())
            n_high = n_group_task - n_low

            rows.append({
                "task": task_name,
                "task_type": TASK_TYPE_MAP[task_name],
                "metric": metric,
                "descriptor_group": group_name,
                "n_total_task_descriptors": total_task_descriptors,
                "n_group_task_descriptors": n_group_task,
                "n_group_low": n_low,
                "n_group_high": n_high,
                "fraction_group_total": n_group_task / total_task_descriptors,
                "fraction_group_low": n_low / total_task_descriptors,
                "fraction_group_high": n_high / total_task_descriptors,
                "low_threshold": threshold,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["task", "fraction_group_total"], ascending=[True, False])
    return out


def plot_group_task_fraction_bars(
    summary_df: pd.DataFrame,
    out_dir: Path,
    threshold: float = 0.7,
) -> None:
    """
    Create one stacked bar plot per task type: x-axis is descriptor group,
    y-axis is the fraction of that task's descriptors in the group, stacked
    by high-performing vs low-performing (metric <= threshold) share.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        print("No group-task fraction summary data available for plotting.")
        return

    for task_name in ["regression", "classification", "multiclass"]:
        task_plot_df = summary_df.loc[summary_df["task"] == task_name].copy()
        if task_plot_df.empty:
            print(f"No group fraction data for {task_name}.")
            continue

        task_plot_df = task_plot_df.sort_values("fraction_group_total", ascending=False)

        labels = task_plot_df["descriptor_group"].tolist()
        frac_high = task_plot_df["fraction_group_high"].values
        frac_low = task_plot_df["fraction_group_low"].values
        total_counts = task_plot_df["n_group_task_descriptors"].values
        low_counts = task_plot_df["n_group_low"].values
        total_task = int(task_plot_df["n_total_task_descriptors"].iloc[0])
        metric = task_plot_df["metric"].iloc[0]

        n_labels = len(labels)
        x = np.arange(n_labels)
        fig_width, fig_height, label_size, rotation, bottom_margin, bar_width = (
            get_dynamic_bar_plot_layout(n_labels)
        )
        # This summary plot doesn't need to be as tall as descriptor-level plots.
        fig_height = max(6.5, min(fig_height, 8.0))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # NOTE: this line previously referenced an undefined variable
        # ("metsubsetric" instead of "metric"), which raised a NameError
        # and crashed the script whenever this plot had data to draw.
        ax.bar(x, frac_high, width=bar_width, label=f"{metric} > {threshold}")
        ax.bar(x, frac_low, width=bar_width, bottom=frac_high, label=f"{metric} <= {threshold}")

        total_heights = frac_high + frac_low

        ax.set_ylabel("Fraction of descriptors in task", fontsize=11)
        ax.set_xlabel("Descriptor group", fontsize=11)
        ax.set_title(
            f"{PLOT_LABEL_BY_TASK[task_name]}: group fraction and low-performing share\n"
            f"Total {task_name} descriptors = {total_task}",
            fontsize=12,
        )

        ymax = max(0.1, float(np.nanmax(total_heights)) * 1.20)
        ax.set_ylim(0, min(1.05, ymax))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=rotation, ha="right" if rotation else "center", fontsize=label_size
        )
        ax.tick_params(axis="y", labelsize=9)

        # Count labels above each bar: total descriptor count, with
        # low-performing count shown in parentheses.
        for i, (total_n, low_n, total_frac) in enumerate(zip(total_counts, low_counts, total_heights)):
            ax.text(i, total_frac + 0.01, f"{total_n}\n({low_n})", ha="center", va="bottom", fontsize=7)

        ax.legend(fontsize=9)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=bottom_margin)

        threshold_label = str(threshold).replace(".", "p")
        out_path = out_dir / f"{task_name}_group_fraction_stacked_threshold_{threshold_label}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved group fraction stacked plot for {task_name}: {out_path}")


def write_run_summary(out_dir: Path, model_results: dict[str, pd.DataFrame], pred: str) -> None:
    """Write a small text summary of the run."""
    lines = [
        "Average embedding output generation summary",
        "===========================================",
        f"Predicted descriptor set: {pred}",
        f"Models averaged: {', '.join(model_results.keys())}",
        "",
        "Created files:",
        "- avg_stats_across_embedding_models.csv",
        "- descriptor_r2_by_model_with_average.csv",
        "- group_performance_by_task.csv",
        "- bar_plots_by_group_and_task/regression/*_regression_r2_bar.png",
        "- bar_plots_by_group_and_task/classification/*_classification_Balanced_Accuracy_bar.png",
        "- bar_plots_by_group_and_task/multiclass/*_multiclass_Balanced_Accuracy_bar.png",
    ]
    (out_dir / "README_outputs.txt").write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average transformer encoder results and create descriptor/group outputs."
    )
    parser.add_argument("--pred", default="mordred", help="Predicted descriptor set, e.g. mordred")
    parser.add_argument(
        "--result-dir",
        default="lipinski_cross_feature_predictions",
        help="Key under paths['prediction_output_dirs'] containing prediction outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["chemberta", "molformer", "molformer-c3-1b", "selformer", "chembertasey"],
        help="Transformer encoder model names to average.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If omitted, saves under results/embedding_average_outputs.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional TL_project root. If omitted, uses the absolute imports from existing scripts.",
    )
    parser.add_argument(
        "--exclude-low-var",
        action="store_true",
        help="Exclude low-variance descriptors from group averages and plots.",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=0.8,
        help="Low-variance threshold passed to getLowVarianceColumns.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else None
    add_project_imports(project_root)

    from get_paths import getPaths
    from group_descriptors import getGroups
    from analyse_datasets import getLowVarianceColumns

    paths = getPaths()
    results_dir = resolve_results_dir(paths, args.result_dir)
    group_map = getGroups(args.pred)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(paths["imp_dirs"]["results_dir"])
            / "embedding_average_outputs"
            / f"pred_{args.pred}_avg_transformers"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_cols = []
    if args.exclude_low_var:
        pred_ft_df = Path(paths["full_features"]["all"][args.pred])
        exclude_cols = getLowVarianceColumns(pred_ft_df, threshold=args.var_threshold)

    # 1. Load all model result CSVs (column names canonicalised on load).
    model_results = load_model_results(results_dir, args.pred, args.models)

    # 2. Average all numeric performance columns across transformer models.
    avg_df = average_model_results(model_results)

    if exclude_cols:
        avg_df = avg_df.drop(index=[c for c in exclude_cols if c in avg_df.index], errors="ignore")

    avg_df.to_csv(out_dir / "avg_stats_across_embedding_models.csv")

    # 3. r2 table for each model and average across regression descriptors.
    r2_table = make_descriptor_r2_table(model_results)
    if exclude_cols:
        r2_table = r2_table.drop(index=[c for c in exclude_cols if c in r2_table.index], errors="ignore")
    r2_table.to_csv(out_dir / "descriptor_r2_by_model_with_average.csv")

    # 4. Group performances for regression, classification, and multiclass.
    group_perf = group_descriptors_long(avg_df, group_map, exclude_cols=exclude_cols)
    group_perf.to_csv(out_dir / "group_performance_by_task.csv", index=False)

    # 5. Group-level fraction summary + stacked plots.
    group_fraction_summary = get_group_task_fraction_summary(
        avg_df=avg_df, group_map=group_map, threshold=0.7, exclude_cols=exclude_cols
    )
    group_fraction_summary.to_csv(
        out_dir / "group_task_fraction_summary_threshold_0p7.csv", index=False
    )
    plot_group_task_fraction_bars(
        summary_df=group_fraction_summary,
        out_dir=out_dir / "group_fraction_plots_by_task",
        threshold=0.7,
    )

    # 6. Separate bar plots for each descriptor group and task type.
    plot_group_task_bars(avg_df=avg_df, group_map=group_map, out_dir=out_dir / "bar_plots_by_group_and_task")

    # 7. Low-performing descriptor plots, threshold <= 0.5.
    plot_low_performing_descriptors(
        avg_df=avg_df, group_map=group_map, out_dir=out_dir / "bar_plots_predicted_leq_0p5", threshold=0.5
    )

    # 8. Write summary after all outputs have been created.
    write_run_summary(out_dir, model_results, args.pred)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()