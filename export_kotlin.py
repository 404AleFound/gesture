"""Compile a trained sklearn Pipeline into pure Kotlin source.

Usage:
    python export_kotlin.py logger/latest/models/RandomForest.joblib \
        [--out RandomForestModel.kt] [--package com.example.gesture]

Output is a single self-contained Kotlin file:
    - `object RandomForestModel` holds the scaler stats, per-tree arrays,
      feature names and label names as plain FloatArray/IntArray literals.
    - `fun predict(features: FloatArray): String` runs standardization,
      walks every tree, tallies votes and returns the winning label.

No external dependencies (no ONNX runtime, no math libraries). The file
drops straight into an Android module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train import load_model


def _tree_arrays(sk_tree) -> dict:
    """Extract the four arrays we need to run inference on a decision tree.

    sklearn stores the tree in a bunch of parallel arrays indexed by node id.
    We keep only what predict() needs:
        feature[i]         -- which feature to split on at node i (-2 for leaves)
        threshold[i]       -- split threshold at node i
        left[i] / right[i] -- child node ids for the two branches
        leaf_class[i]      -- class index the leaf votes for (only meaningful
                              for leaf nodes, but we fill for all rows)
    """
    t = sk_tree
    # value: shape (n_nodes, 1, n_classes). Predicted class is argmax at each leaf.
    leaf_class = t.value[:, 0, :].argmax(axis=1).astype(np.int32)
    return {
        "feature": t.feature.astype(np.int32),
        "threshold": t.threshold.astype(np.float32),
        "left": t.children_left.astype(np.int32),
        "right": t.children_right.astype(np.int32),
        "leaf_class": leaf_class,
    }


def _float_array_literal(
    name: str, arr: np.ndarray, indent: str = "    ", visibility: str = "private ",
) -> str:
    values = ", ".join(f"{v:.9g}f" for v in arr.tolist())
    return f"{indent}{visibility}val {name}: FloatArray = floatArrayOf({values})"


def _int_array_literal(
    name: str, arr: np.ndarray, indent: str = "    ", visibility: str = "private ",
) -> str:
    values = ", ".join(str(int(v)) for v in arr.tolist())
    return f"{indent}{visibility}val {name}: IntArray = intArrayOf({values})"


def render_kotlin(joblib_path: Path, package: str) -> str:
    pipe, labels, feature_names = load_model(joblib_path)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    n_features = int(scaler.mean_.shape[0])
    n_classes = int(clf.n_classes_)
    n_trees = int(clf.n_estimators)
    labels_str = [str(l) for l in labels]

    # Each tree becomes its own nested `object` in the output; that keeps
    # each tree's array initialization in a separate <clinit>, so we don't
    # bust the JVM 65535-byte method-size limit that a single flat init
    # block hits once you have ~10+ trees of non-trivial size.
    trees = [_tree_arrays(t.tree_) for t in clf.estimators_]

    lines: list[str] = []
    lines.append(f"package {package}")
    lines.append("")
    lines.append('/** Auto-generated from a scikit-learn RandomForest pipeline.')
    lines.append(f' *  source: {joblib_path.name}')
    lines.append(f' *  {n_features} features / {n_trees} trees / {n_classes} classes')
    lines.append(' *  Do not edit by hand -- regenerate via export_kotlin.py.')
    lines.append(' */')
    lines.append("object RandomForestModel {")
    lines.append("")

    # ---- metadata ------------------------------------------------------
    feat_lit = ", ".join(f'"{n}"' for n in feature_names)
    lines.append(f"    val featureNames: List<String> = listOf({feat_lit})")
    lbl_lit = ", ".join(f'"{n}"' for n in labels_str)
    lines.append(f"    val labels: List<String> = listOf({lbl_lit})")
    lines.append(f"    private const val N_FEATURES = {n_features}")
    lines.append(f"    private const val N_CLASSES = {n_classes}")
    lines.append(f"    private const val N_TREES = {n_trees}")
    lines.append("")

    # ---- scaler stats --------------------------------------------------
    lines.append(_float_array_literal("scalerMean", scaler.mean_.astype(np.float32)))
    lines.append(_float_array_literal("scalerScale", scaler.scale_.astype(np.float32)))
    lines.append("")

    # ---- tree arrays ---------------------------------------------------
    # Each tree gets its own nested `object` so its arrays are initialized
    # in a separate <clinit>. Emitting all trees as top-level vals inside
    # RandomForestModel piles every array literal into one static
    # initializer and blows past the JVM 65535-byte method-size limit.
    # The nested object itself is private, so the vals inside can stay public
    # (default) -- otherwise the outer registry can't reference them.
    for i, tree in enumerate(trees):
        lines.append(f"    private object Tree{i} {{")
        lines.append(_int_array_literal("feature", tree["feature"], indent="        ", visibility=""))
        lines.append(_float_array_literal("threshold", tree["threshold"], indent="        ", visibility=""))
        lines.append(_int_array_literal("left", tree["left"], indent="        ", visibility=""))
        lines.append(_int_array_literal("right", tree["right"], indent="        ", visibility=""))
        lines.append(_int_array_literal("leafClass", tree["leaf_class"], indent="        ", visibility=""))
        lines.append("    }")
        lines.append("")

    # ---- registry of trees ---------------------------------------------
    lines.append("    private val treeFeatures: Array<IntArray> = arrayOf(")
    lines.append(",\n".join(f"        Tree{i}.feature" for i in range(n_trees)))
    lines.append("    )")
    lines.append("    private val treeThresholds: Array<FloatArray> = arrayOf(")
    lines.append(",\n".join(f"        Tree{i}.threshold" for i in range(n_trees)))
    lines.append("    )")
    lines.append("    private val treeLefts: Array<IntArray> = arrayOf(")
    lines.append(",\n".join(f"        Tree{i}.left" for i in range(n_trees)))
    lines.append("    )")
    lines.append("    private val treeRights: Array<IntArray> = arrayOf(")
    lines.append(",\n".join(f"        Tree{i}.right" for i in range(n_trees)))
    lines.append("    )")
    lines.append("    private val treeLeafClasses: Array<IntArray> = arrayOf(")
    lines.append(",\n".join(f"        Tree{i}.leafClass" for i in range(n_trees)))
    lines.append("    )")
    lines.append("")

    # ---- predict fn ----------------------------------------------------
    lines.append("""\
    /** Predict a gesture label from an unscaled feature vector.
     *  The vector must be aligned with [featureNames]. */
    fun predict(features: FloatArray): String {
        require(features.size == N_FEATURES) {
            "expected $N_FEATURES features, got ${features.size}"
        }

        // 1. StandardScaler: (x - mean) / scale
        val scaled = FloatArray(N_FEATURES)
        for (i in 0 until N_FEATURES) {
            scaled[i] = (features[i] - scalerMean[i]) / scalerScale[i]
        }

        // 2. Majority vote across every tree
        val votes = IntArray(N_CLASSES)
        for (t in 0 until N_TREES) {
            val feature = treeFeatures[t]
            val threshold = treeThresholds[t]
            val left = treeLefts[t]
            val right = treeRights[t]
            val leafClass = treeLeafClasses[t]

            var node = 0
            while (left[node] != -1) {                // -1 marks a leaf
                node = if (scaled[feature[node]] <= threshold[node])
                    left[node] else right[node]
            }
            votes[leafClass[node]]++
        }

        // 3. Argmax of the vote tally
        var bestClass = 0
        var bestVotes = votes[0]
        for (c in 1 until N_CLASSES) {
            if (votes[c] > bestVotes) {
                bestVotes = votes[c]; bestClass = c
            }
        }
        return labels[bestClass]
    }

    /** Same as [predict] but also returns the raw class probabilities
     *  estimated as tree-vote fractions. */
    fun predictWithProbs(features: FloatArray): Pair<String, FloatArray> {
        require(features.size == N_FEATURES) {
            "expected $N_FEATURES features, got ${features.size}"
        }
        val scaled = FloatArray(N_FEATURES)
        for (i in 0 until N_FEATURES) {
            scaled[i] = (features[i] - scalerMean[i]) / scalerScale[i]
        }
        val votes = IntArray(N_CLASSES)
        for (t in 0 until N_TREES) {
            val feature = treeFeatures[t]
            val threshold = treeThresholds[t]
            val left = treeLefts[t]
            val right = treeRights[t]
            val leafClass = treeLeafClasses[t]
            var node = 0
            while (left[node] != -1) {
                node = if (scaled[feature[node]] <= threshold[node])
                    left[node] else right[node]
            }
            votes[leafClass[node]]++
        }
        val probs = FloatArray(N_CLASSES) { votes[it].toFloat() / N_TREES }
        var best = 0
        for (c in 1 until N_CLASSES) if (probs[c] > probs[best]) best = c
        return labels[best] to probs
    }
}
""")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compile a joblib model to Kotlin.")
    p.add_argument("model", help="path to a RandomForest .joblib file")
    p.add_argument("--out", default="RandomForestModel.kt",
                   help="output Kotlin file (default: RandomForestModel.kt)")
    p.add_argument("--package", default="com.example.gesture",
                   help="Kotlin package (default: com.example.gesture)")
    args = p.parse_args()

    joblib_path = Path(args.model)
    src = render_kotlin(joblib_path, args.package)

    out = Path(args.out)
    out.write_text(src, encoding="utf-8")
    print(f"wrote {out} ({len(src):,} bytes)")


if __name__ == "__main__":
    main()
