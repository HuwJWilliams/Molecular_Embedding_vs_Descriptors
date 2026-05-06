"""Similarity analysis module for molecular feature sets."""
 
from __future__ import annotations
 
import re
from pathlib import Path
from typing import Union
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import RobustScaler, StandardScaler
 
 
class Similarity:
    """Compute and visualise molecular similarity across feature sets.
 
    Parameters
    ----------
    colour_map : dict, optional
        Mapping from lower-case feature/source name to an RGBA or
        matplotlib-compatible colour tuple.  Keys are matched
        boundary-aware (word-boundary regex) so ``"morgan"`` will
        not accidentally match ``"FpDensityMorgan1"``.
    default_colour : tuple, optional
        Fallback colour used when a name is not found in
        ``colour_map``.  Defaults to mid-grey ``(0.5, 0.5, 0.5, 1.0)``.
    save_all : bool, optional
        When *True* every ``_savePlot`` call saves regardless of the
        per-call ``save_plot`` flag.  Defaults to ``False``.
    """
 
    def __init__(
        self,
        colour_map: dict | None = None,
        default_colour: tuple = (0.5, 0.5, 0.5, 1.0),
        save_all: bool = False,
    ) -> None:
        self.colour_map: dict = {k.lower(): v for k, v in (colour_map or {}).items()}
        self.default_colour: tuple = default_colour
        self.save_all: bool = save_all
 
    # ------------------------------------------------------------------
    # Public similarity metrics
    # ------------------------------------------------------------------
 
    @staticmethod
    def getRBFSim(df: pd.DataFrame) -> pd.DataFrame:
        """Return an RBF (Gaussian) kernel similarity matrix.
 
        Features are z-score standardised before computing squared
        Euclidean distances.  The bandwidth ``gamma`` is set to the
        reciprocal of the median non-zero squared distance (the
        *median heuristic*).
 
        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with molecules as rows.
 
        Returns
        -------
        pd.DataFrame
            Square similarity matrix sharing ``df``'s index.
        """
        df_scaled = StandardScaler().fit_transform(df)
        sq_dist = pairwise_distances(df_scaled, metric="sqeuclidean")
        gamma = 1.0 / np.median(sq_dist[sq_dist > 0])
        sim = np.exp(-gamma * sq_dist)
        return pd.DataFrame(sim, index=df.index, columns=df.index)
 
    @staticmethod
    def getJACCSim(df: pd.DataFrame) -> pd.DataFrame:
        """Return a continuous (generalised) Jaccard similarity matrix.
 
        Features are robust-scaled then min-shifted to ``[0, ∞)``
        before the element-wise ``min / max`` formula is applied.
 
        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with molecules as rows.
 
        Returns
        -------
        pd.DataFrame
            Square similarity matrix sharing ``df``'s index.
        """
        arr = RobustScaler().fit_transform(df)
        arr = arr - arr.min(axis=0)
 
        n = arr.shape[0]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                num = np.sum(np.minimum(arr[i], arr[j]))
                den = np.sum(np.maximum(arr[i], arr[j]))
                sim[i, j] = 1.0 if den == 0 else num / den
 
        return pd.DataFrame(sim, index=df.index, columns=df.index)
 
    @staticmethod
    def getTanimotoSim(df: pd.DataFrame) -> pd.DataFrame:
        """Return Tanimoto (Jaccard) similarity on binary fingerprints.
 
        Non-zero entries are treated as present (``True``).
 
        Parameters
        ----------
        df : pd.DataFrame
            Binary or count fingerprint matrix.
 
        Returns
        -------
        pd.DataFrame
            Square similarity matrix sharing ``df``'s index.
        """
        arr_bool = (df.to_numpy() > 0).astype(bool)
        dist = pairwise_distances(arr_bool, metric="jaccard")
        return pd.DataFrame(1.0 - dist, index=df.index, columns=df.index)
 
    # ------------------------------------------------------------------
    # Dendrogram / ordering helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def getDendrogramOrder(
        sim_df: pd.DataFrame,
    ) -> tuple[pd.Index, np.ndarray]:
        """Return the leaf order and linkage matrix for a similarity matrix.
 
        Converts the similarity matrix to a distance matrix
        (``dist = 1 − sim``), computes average-linkage hierarchical
        clustering, and extracts the dendrogram leaf order so that
        rows/columns of a heatmap can be arranged to match the
        dendrogram.
 
        Parameters
        ----------
        sim_df : pd.DataFrame
            Square, symmetric similarity matrix (values in ``[0, 1]``).
 
        Returns
        -------
        ordered_index : pd.Index
            ``sim_df.index`` reordered according to the dendrogram leaves.
        linkage_matrix : np.ndarray
            SciPy linkage matrix (shape ``(n-1, 4)``).
        """
        dist = np.clip(1.0 - sim_df.to_numpy(dtype=float), 0.0, None)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        dendro = dendrogram(Z, no_plot=True)
        return sim_df.index[dendro["leaves"]], Z
 
    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
 
    def plotHeatmapWithYDendrogram(
        self,
        heatmap_df: pd.DataFrame,
        linkage_matrix: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
        cbar_label: str,
        title: str,
        out_file: Path,
    ) -> None:
        """Save a heatmap with a matching left-side dendrogram.
 
        The dendrogram is drawn in its own axes that shares the heatmap's
        y-extent, so the tree branches align precisely with the heatmap
        rows.
 
        Parameters
        ----------
        heatmap_df : pd.DataFrame
            Square data frame whose rows/columns are already in
            dendrogram order (as returned by :meth:`getDendrogramOrder`).
        linkage_matrix : np.ndarray
            SciPy linkage matrix produced from the *same* ordering.
        cmap : str
            Matplotlib colormap name (e.g. ``"viridis"``, ``"Greys"``).
        vmin, vmax : float
            Colour-scale limits.
        cbar_label : str
            Label for the colour-bar.
        title : str
            Heatmap title.
        out_file : Path
            Destination file path (PNG recommended).
        """
        n = heatmap_df.shape[0]
 
        fig = plt.figure(figsize=(16, 14))
        # Reserve a narrow column for the dendrogram and a wide one for
        # the heatmap.  The two axes must share exactly the same y range
        # so that dendrogram leaves map 1-to-1 onto heatmap rows.
        gs = fig.add_gridspec(1, 2, width_ratios=(1, 8), wspace=0.02)
        ax_dendro = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
 
        # --- dendrogram ---------------------------------------------------
        # Draw into a throw-away axes first so we can extract the raw
        # icoord/dcoord values (scipy leaf units: centres at 5, 15, …).
        # We then rescale them to seaborn's heatmap coordinate system
        # (row centres at 0.5, 1.5, … within [0, n]) and re-draw the
        # lines manually.  This is the only approach that is robust to
        # seaborn resetting axis limits after heatmap() returns.
        dendro_data = dendrogram(linkage_matrix, no_plot=True)
 
        # scipy unit: leaf i sits at 10*i + 5  →  heatmap unit: i + 0.5
        # generalised linear map: heatmap_y = (scipy_y / 10)
        scipy_max = 10 * n
        def _to_heatmap(y_scipy):
            return np.asarray(y_scipy) / 10.0
 
        ax_dendro.set_xlim(
            min(min(d) for d in dendro_data["dcoord"]),
            max(max(d) for d in dendro_data["dcoord"]),
        )
        ax_dendro.set_ylim(0, n)
        ax_dendro.invert_xaxis()   # branches grow left → right toward heatmap
 
        for xs, ys in zip(dendro_data["dcoord"], dendro_data["icoord"]):
            # dcoord = branch lengths (x in "left" orientation)
            # icoord = leaf positions  (y in "left" orientation)
            ax_dendro.plot(_to_heatmap(ys), xs, color="black", linewidth=0.8)
 
        ax_dendro.axis("off")
 
        # --- heatmap ------------------------------------------------------
        sns.heatmap(
            heatmap_df,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"label": cbar_label, "shrink": 0.8},
            ax=ax_heat,
        )
        ax_heat.set_title(title, pad=10)
        ax_heat.set_xlabel("Molecules")
        ax_heat.set_ylabel("")
 
        # Tick labels become unreadable at large n; suppress them.
        if n > 40:
            ax_heat.set_xticklabels([])
            ax_heat.set_yticklabels([])
 
        fig.tight_layout()
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")
 
    @staticmethod
    def savesSmilesGrid(
        smiles_list: list[str],
        legend_list: list[str],
        out_path: Union[str, Path] = "molecule_grid.png",
        mols_per_row: int = 4,
        sub_img_size: tuple[int, int] = (320, 260),
    ) -> None:
        """Render a grid of 2-D molecular structures and save to disk.
 
        Invalid SMILES strings are skipped with a warning.
 
        Parameters
        ----------
        smiles_list : list of str
            SMILES strings to draw.
        legend_list : list of str
            Per-molecule legend strings (same length as *smiles_list*).
        out_path : str or Path
            Output image file path.
        mols_per_row : int
            Number of structures per grid row.
        sub_img_size : tuple of int
            ``(width, height)`` in pixels for each sub-image cell.
        """
        if len(smiles_list) != len(legend_list):
            raise ValueError("smiles_list and legend_list must be the same length.")
 
        mols, legends = [], []
        for smi, leg in zip(smiles_list, legend_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Skipping invalid SMILES: {smi}")
                continue
            mols.append(mol)
            legends.append(str(leg))
 
        if not mols:
            raise ValueError("No valid SMILES to draw.")
 
        img = Draw.MolsToGridImage(
            mols,
            legends=legends,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            useSVG=False,
        )
        img.save(str(out_path))
 
    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
 
    def plotSimilarities(
        self,
        feature_sets: list[str],
        feature_paths: dict[str, Path],
        n_mols: int = 100,
        show_top_n_pairs: int = 3,
        molids: list[str] | None = None,
        save_dir: Path = Path("results") / "similarity",
        molid2Smiles: callable | None = None,
    ) -> tuple[dict, dict]:
        """Compute and visualise similarities across multiple feature sets.
 
        For every feature set the method computes RBF and continuous-
        Jaccard similarity matrices, then produces:
 
        * Molecule-grid images for the most-similar pairs per metric.
        * Combined upper/lower-triangle heatmaps (with dendrogram) for
          every pair of feature sets.
        * Absolute-difference heatmaps between feature sets.
        * KPCA scatter / KDE panels for all feature sets on one canvas.
        * Scatter plots comparing each metric against Tanimoto similarity
          (requires ``"maccs"`` or ``"morgan"`` in *feature_sets*).
 
        Parameters
        ----------
        feature_sets : list of str
            Names of the feature sets to analyse.  Must be keys in
            *feature_paths*.
        feature_paths : dict
            Mapping from feature-set name to the CSV file path containing
            that feature matrix (molecules as rows, features as columns).
        n_mols : int, optional
            Number of molecules to sample when *molids* is not provided.
            Default ``100``.
        show_top_n_pairs : int, optional
            Number of most-similar molecule pairs to display per metric /
            feature combination.  Default ``3``.
        molids : list of str, optional
            Explicit list of molecule IDs to use.  When *None*, a random
            sample of *n_mols* molecules is drawn from the first feature
            set.
        save_dir : Path, optional
            Directory for all output files.
        molid2Smiles : callable, optional
            Function mapping a molecule ID to its SMILES string.  Used
            when rendering top-pair grids.  If *None*, grids are skipped.
 
        Returns
        -------
        all_sim_matrices : dict
            Nested dict ``{metric_key: {feature_name: pd.DataFrame}}``.
        mol_comp : dict
            Dict ``{"feat_sim": {key: list_of_top_pair_records}}``.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
 
        if not feature_sets:
            raise ValueError("feature_sets is empty.")
 
        molids = list(molids) if molids else []
 
        if not molids:
            sample_feat = feature_sets[0]
            molids = list(
                pd.read_csv(feature_paths[sample_feat], index_col=0)
                .sample(n_mols)
                .index
            )
 
        all_sim_matrices: dict[str, dict[str, pd.DataFrame]] = {
            "rbf": {},
            "jacc": {},
        }
        mol_comp: dict = {"feat_sim": {}}
 
        # ------------------------------------------------------------------
        # Build similarity matrices & collect top pairs
        # ------------------------------------------------------------------
        for feat in feature_sets:
            feat_df = pd.read_csv(feature_paths[feat], index_col=0).loc[molids]
 
            all_sim_matrices["rbf"][feat] = self.getRBFSim(feat_df)
            all_sim_matrices["jacc"][feat] = self.getJACCSim(feat_df)
            ids = list(feat_df.index)
 
            for metric_key in ("rbf", "jacc"):
                sim = all_sim_matrices[metric_key][feat].to_numpy(dtype=float)
                if sim.shape[0] < 2:
                    continue
 
                iu = np.triu_indices(sim.shape[0], k=1)
                pair_vals = sim[iu]
                if pair_vals.size == 0:
                    continue
 
                k = min(show_top_n_pairs, pair_vals.size)
                top_idx = np.argsort(pair_vals)[-k:][::-1]
 
                top_pairs = []
                for rank, idx_flat in enumerate(top_idx, start=1):
                    i = int(iu[0][idx_flat])
                    j = int(iu[1][idx_flat])
                    score = float(pair_vals[idx_flat])
                    id1, id2 = ids[i], ids[j]
 
                    smi1 = molid2Smiles(id1) if molid2Smiles else None
                    smi2 = molid2Smiles(id2) if molid2Smiles else None
 
                    print(
                        f"[TOP-{rank} {metric_key.upper()}] {feat}: "
                        f"({id1}, {id2}) -> {score:.6f}"
                    )
                    top_pairs.append(
                        {
                            "rank": rank,
                            "score": score,
                            "id_pair": (id1, id2),
                            "smi_pair": (smi1, smi2),
                        }
                    )
 
                mol_comp["feat_sim"][f"{feat}_{metric_key}"] = top_pairs
 
        # ------------------------------------------------------------------
        # Molecule grid images for top pairs
        # ------------------------------------------------------------------
        if molid2Smiles is not None:
            for key, val in mol_comp.get("feat_sim", {}).items():
                if not isinstance(val, list):
                    continue
 
                label = key.replace("_", " ").upper()
                smiles_list, legend_list = [], []
 
                for rec in val:
                    id1, id2 = rec["id_pair"]
                    smi1, smi2 = rec["smi_pair"]
                    rank = rec.get("rank", "?")
                    score = rec.get("score")
 
                    smiles_list.extend([smi1, smi2])
                    suffix = f"TOP-{rank} ({score:.3f})" if score is not None else f"TOP-{rank}"
                    legend_list.extend(
                        [f"{id1}\n{label}\n{suffix}", f"{id2}\n{label}\n{suffix}"]
                    )
 
                if not smiles_list:
                    continue
 
                out_path = save_dir / f"{key}_top{show_top_n_pairs}.png"
                self.savesSmilesGrid(smiles_list, legend_list, out_path=out_path, mols_per_row=2)
                print(f"Saved: {out_path}")
 
        # ------------------------------------------------------------------
        # Combined upper/lower-triangle heatmaps for every feature-set pair
        # ------------------------------------------------------------------
        if len(feature_sets) >= 2:
            for i in range(len(feature_sets)):
                for j in range(i + 1, len(feature_sets)):
                    feat_a, feat_b = feature_sets[i], feature_sets[j]
 
                    for metric_key, cbar_label in [
                        ("rbf", "RBF Similarity"),
                        ("jacc", "Continuous Jaccard"),
                    ]:
                        sim_a_df = all_sim_matrices[metric_key][feat_a]
                        sim_b_df = all_sim_matrices[metric_key][feat_b]
 
                        common_ids = sim_a_df.index.intersection(sim_b_df.index)
                        if len(common_ids) < 2:
                            print(
                                f"Skipping combined heatmap {metric_key} "
                                f"{feat_a} vs {feat_b}: not enough shared IDs."
                            )
                            continue
 
                        order_ids, Z = self.getDendrogramOrder(
                            sim_a_df.loc[common_ids, common_ids]
                        )
 
                        sim_a = sim_a_df.loc[order_ids, order_ids].to_numpy(dtype=float)
                        sim_b = sim_b_df.loc[order_ids, order_ids].to_numpy(dtype=float)
 
                        n = sim_a.shape[0]
                        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                        lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
 
                        combined = np.zeros((n, n), dtype=float)
                        combined[upper_mask] = sim_a[upper_mask]
                        combined[lower_mask] = sim_b[lower_mask]
                        np.fill_diagonal(combined, 1.0)
 
                        combined_df = pd.DataFrame(combined, index=order_ids, columns=order_ids)
 
                        # Absolute difference (lower triangle only)
                        diff = np.abs(sim_a - sim_b)
                        diff_lower = np.full((n, n), np.nan, dtype=float)
                        diff_lower[lower_mask] = diff[lower_mask]
                        diff_lower_df = pd.DataFrame(diff_lower, index=order_ids, columns=order_ids)
 
                        self.plotHeatmapWithYDendrogram(
                            heatmap_df=combined_df,
                            linkage_matrix=Z,
                            cmap="viridis",
                            vmin=0,
                            vmax=1,
                            cbar_label=cbar_label,
                            title=f"{metric_key.upper()} | Upper: {feat_a} | Lower: {feat_b}",
                            out_file=save_dir / f"combined_triangles_{metric_key}_{feat_a}_vs_{feat_b}.png",
                        )
 
                        self.plotHeatmapWithYDendrogram(
                            heatmap_df=diff_lower_df,
                            linkage_matrix=Z,
                            cmap="Greys",
                            vmin=0,
                            vmax=0.5,
                            cbar_label=f"|{feat_a} - {feat_b}| ({metric_key.upper()})",
                            title=f"{metric_key.upper()} absolute difference | {feat_a} vs {feat_b}",
                            out_file=save_dir / f"difference_heatmap_{metric_key}_{feat_a}_vs_{feat_b}.png",
                        )
 
        # ------------------------------------------------------------------
        # KPCA panels
        # ------------------------------------------------------------------
        for metric_key in ("rbf", "jacc"):
            common_ids = None
            for feat in feature_sets:
                idx = all_sim_matrices[metric_key][feat].index
                common_ids = idx if common_ids is None else common_ids.intersection(idx)
 
            if common_ids is None or len(common_ids) < 3:
                print(f"Skipping KPCA for {metric_key}: not enough shared molecule IDs.")
                continue
 
            try:
                plot_frames = []
                for feat in feature_sets:
                    S = all_sim_matrices[metric_key][feat].loc[common_ids, common_ids].to_numpy()
                    coords = KernelPCA(n_components=2, kernel="precomputed").fit_transform(S)
                    tmp = pd.DataFrame(coords, columns=["KPCA1", "KPCA2"], index=common_ids)
                    tmp["Source"] = feat
                    plot_frames.append(tmp.reset_index(names="MolID"))
 
                plot_df = pd.concat(plot_frames, ignore_index=True)
                source_labels = list(plot_df["Source"].dropna().astype(str).unique())
                fallback_palette = sns.color_palette("tab10", n_colors=max(1, len(source_labels)))
                source_palette = {
                    src: (
                        c if (c := self._getColour(src)) != self.default_colour else fallback_palette[idx % len(fallback_palette)]
                    )
                    for idx, src in enumerate(source_labels)
                }
 
                # Variance explained from the first feature set's kernel
                kpca_ref = KernelPCA(n_components=2, kernel="precomputed")
                kpca_ref.fit(
                    all_sim_matrices[metric_key][feature_sets[0]]
                    .loc[common_ids, common_ids]
                    .to_numpy()
                )
                eigvals = np.asarray(kpca_ref.eigenvalues_, dtype=float)
                var_ratio = eigvals / eigvals.sum() if eigvals.sum() > 0 else np.array([np.nan, np.nan])
                xlab = f"KPCA1 ({var_ratio[0] * 100:.1f}% var)"
                ylab = f"KPCA2 ({var_ratio[1] * 100:.1f}% var)"
 
                fig, axes = plt.subplots(
                    2, 2, figsize=(11, 8.5), gridspec_kw={"wspace": 0.18, "hspace": 0.18}
                )
 
                sns.kdeplot(
                    data=plot_df, x="KPCA1", hue="Source", fill=True,
                    common_norm=False, alpha=0.25, ax=axes[0, 0],
                    legend=False, palette=source_palette,
                )
                axes[0, 0].set_xlabel(xlab)
                axes[0, 0].set_ylabel("Density")
 
                sns.scatterplot(
                    data=plot_df, x="KPCA1", y="KPCA2", hue="Source",
                    s=28, alpha=0.75, edgecolor=None, ax=axes[1, 0],
                    palette=source_palette,
                )
                axes[1, 0].set_xlabel(xlab)
                axes[1, 0].set_ylabel(ylab)
 
                sns.kdeplot(
                    data=plot_df, x="KPCA2", hue="Source", fill=True,
                    common_norm=False, alpha=0.25, ax=axes[1, 1],
                    legend=False, palette=source_palette,
                )
                axes[1, 1].set_xlabel(ylab)
                axes[1, 1].set_ylabel("Density")
                axes[0, 1].axis("off")
 
                # Detach scatter legend so we can place it at figure level
                handles, labels = axes[1, 0].get_legend_handles_labels()
                leg = axes[1, 0].get_legend()
                if leg is not None:
                    leg.remove()
 
                # Stats table
                stats_rows, stats_numeric = [], []
                for src, grp in plot_df.groupby("Source"):
                    kpca1 = grp["KPCA1"].dropna().to_numpy()
                    kpca2 = grp["KPCA2"].dropna().to_numpy()
                    q1_1, q3_1 = np.percentile(kpca1, [25, 75]) if len(kpca1) else (np.nan, np.nan)
                    q1_2, q3_2 = np.percentile(kpca2, [25, 75]) if len(kpca2) else (np.nan, np.nan)
                    std1 = float(np.std(kpca1, ddof=1)) if len(kpca1) > 1 else np.nan
                    iqr1 = float(q3_1 - q1_1) if len(kpca1) else np.nan
                    std2 = float(np.std(kpca2, ddof=1)) if len(kpca2) > 1 else np.nan
                    iqr2 = float(q3_2 - q1_2) if len(kpca2) else np.nan
                    stats_numeric.append([src, std1, iqr1, std2, iqr2])
                    stats_rows.append(
                        [
                            src,
                            f"{std1:.3f}" if np.isfinite(std1) else "nan",
                            f"{iqr1:.3f}" if np.isfinite(iqr1) else "nan",
                            f"{std2:.3f}" if np.isfinite(std2) else "nan",
                            f"{iqr2:.3f}" if np.isfinite(iqr2) else "nan",
                        ]
                    )
 
                sort_idx = sorted(range(len(stats_rows)), key=lambda i: str(stats_rows[i][0]))
                stats_rows = [stats_rows[i] for i in sort_idx]
                stats_numeric = [stats_numeric[i] for i in sort_idx]
 
                tbl = axes[0, 1].table(
                    cellText=stats_rows,
                    colLabels=["Source", "STD1", "IQR1", "STD2", "IQR2"],
                    colWidths=[0.36, 0.16, 0.16, 0.16, 0.16],
                    cellLoc="center",
                    colLoc="center",
                    loc="lower center",
                    bbox=[0.02, 0.00, 0.96, 0.78],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1.0, 1.15)
 
                metric_vals = np.array([row[1:] for row in stats_numeric], dtype=float)
                if metric_vals.size > 0:
                    best_vals = np.nanmax(metric_vals, axis=0)
                    highlight = "#fff3b0"
                    for r_idx in range(metric_vals.shape[0]):
                        for c_idx in range(metric_vals.shape[1]):
                            val = metric_vals[r_idx, c_idx]
                            if np.isfinite(val) and np.isclose(val, best_vals[c_idx], rtol=1e-9, atol=1e-12):
                                cell = tbl[(r_idx + 1, c_idx + 1)]
                                cell.set_facecolor(highlight)
                                cell.get_text().set_weight("bold")
 
                axes[0, 1].text(
                    0.5, 0.82,
                    "KDE Stats\n(STD/IQR for KPCA1 and KPCA2)",
                    ha="center", va="bottom", fontsize=9,
                    transform=axes[0, 1].transAxes,
                )
 
                fig.suptitle(
                    f"KPCA across all feature sets ({metric_key.upper()})",
                    y=0.985, fontsize=14,
                )
                if handles:
                    fig.legend(
                        handles, labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.955),
                        ncol=min(len(labels), 5),
                        frameon=False,
                        title="Source",
                        fontsize=9,
                        title_fontsize=10,
                    )
 
                plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.86])
                out_kpca = save_dir / f"kpca_{metric_key}_all_feature_sets_panel.png"
                plt.savefig(out_kpca, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved: {out_kpca}")
 
            except Exception as exc:
                print(f"Skipping KPCA for {metric_key} due to error: {exc}")
 
        # ------------------------------------------------------------------
        # Metric-vs-Tanimoto comparison plots
        # ------------------------------------------------------------------
        tanimoto_feat = next(
            (f for f in ("maccs", "morgan") if f in feature_sets), None
        )
 
        if tanimoto_feat is None:
            print(
                "Skipping TANIMOTO comparison plots: "
                "include 'maccs' or 'morgan' in feature_sets."
            )
        else:
            tanimoto_raw = pd.read_csv(feature_paths[tanimoto_feat], index_col=0).loc[molids]
            tanimoto_sim = self.getTanimotoSim(tanimoto_raw)
 
            for metric_key in ("jacc", "rbf"):
                for feat in feature_sets:
                    sim_df = all_sim_matrices[metric_key][feat]
                    common_ids = sim_df.index.intersection(tanimoto_sim.index)
                    if len(common_ids) < 2:
                        print(
                            f"Skipping {metric_key.upper()} vs TANIMOTO for {feat}: "
                            "not enough shared IDs."
                        )
                        continue
 
                    sim_arr = sim_df.loc[common_ids, common_ids].to_numpy(dtype=float)
                    tan_arr = tanimoto_sim.loc[common_ids, common_ids].to_numpy(dtype=float)
                    iu = np.triu_indices(len(common_ids), k=1)
                    if len(iu[0]) == 0:
                        continue
 
                    comp_df = pd.DataFrame(
                        {"Tanimoto": tan_arr[iu], metric_key.upper(): sim_arr[iu]}
                    ).dropna()
                    if comp_df.empty:
                        continue
 
                    point_colour = self._getColour(str(feat))
                    if point_colour == self.default_colour:
                        point_colour = sns.color_palette("tab10", n_colors=1)[0]
 
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        data=comp_df,
                        x="Tanimoto",
                        y=metric_key.upper(),
                        color=point_colour,
                        s=18,
                        alpha=0.35,
                        edgecolor=None,
                    )
                    plt.xlabel(f"Tanimoto Similarity ({tanimoto_feat})")
                    plt.ylabel(f"{metric_key.upper()} Similarity")
                    plt.title(f"{metric_key.upper()} vs TANIMOTO ({feat})")
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black", alpha=0.8)
                    plt.grid(axis="both", linestyle="--", alpha=0.25)
                    plt.tight_layout()
                    out_file = save_dir / f"{metric_key}_vs_tanimoto_{feat}_ref-{tanimoto_feat}.png"
                    plt.savefig(out_file, dpi=300)
                    plt.close()
                    print(f"Saved: {out_file}")
 
        return all_sim_matrices, mol_comp
 
    # ------------------------------------------------------------------
    # Hidden worker functions
    # ------------------------------------------------------------------
 
    def _getColour(self, name: str) -> tuple:
        """Return the RGBA colour for *name* from the internal colour map.
 
        Lookup order:
 
        1. Exact lower-case key match.
        2. Boundary-aware substring match (longer keys checked first to
           avoid short-key collisions such as ``"morgan"`` matching
           ``"FpDensityMorgan1"``).
        3. :attr:`default_colour` if no match is found.
 
        Parameters
        ----------
        name : str
            Label to look up.
 
        Returns
        -------
        tuple
            RGBA colour tuple.
        """
        lower_name = name.lower()
        if lower_name in self.colour_map:
            return self.colour_map[lower_name]
 
        for key in sorted(self.colour_map.keys(), key=len, reverse=True):
            pattern = rf"(?<![a-z0-9]){re.escape(key.lower())}(?![a-z0-9])"
            if re.search(pattern, lower_name):
                return self.colour_map[key]
 
        return self.default_colour
 
    def _savePlot(
        self,
        save_plot: bool,
        save_path: Union[str, Path],
        save_fname: str,
        dpi: int,
        description: str = "Saved plot",
        fig: plt.Figure | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Save a matplotlib figure to disk.
 
        Parameters
        ----------
        save_plot : bool
            When *True* (or when :attr:`save_all` is *True*) the figure
            is written to disk.
        save_path : str or Path
            Directory in which to save the file.  Created if absent.
        save_fname : str
            File name.  A ``.png`` extension is appended when no
            recognised image extension is present.
        dpi : int
            Output resolution in dots per inch.
        description : str, optional
            Label used in the confirmation message printed to stdout.
        fig : plt.Figure, optional
            Figure to save.  Defaults to ``plt.gcf()``.
        metadata : dict, optional
            Metadata dict forwarded to ``fig.savefig``.
 
        Returns
        -------
        None
        """
        if not (save_plot or self.save_all):
            return
 
        metadata = metadata or {}
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
 
        if not any(
            str(save_fname).lower().endswith(ext)
            for ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf")
        ):
            save_fname = str(save_fname) + ".png"
 
        full_path = save_path / save_fname
        fig_to_save = fig or plt.gcf()
        fig_to_save.savefig(full_path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        print(f"Saved {description} to\n{full_path}")
