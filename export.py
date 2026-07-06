"""Compile a trained sklearn RandomForest pipeline into a single Kotlin file.

Uses m2cgen to turn each tree into pure code, then post-processes the Java
output into idiomatic Kotlin. The scaler in the sklearn Pipeline is applied
manually inside the generated `predict` -- m2cgen does not accept Pipeline
objects, so we lift the StandardScaler stats out and let m2cgen handle the
raw RandomForestClassifier.

Usage:
    python export.py logger/latest/models/RandomForest.joblib \
        [--out GestureRandomForest.kt] [--package dev.gc.gesture]

Every tree becomes its own `private fun tree{i}(x: DoubleArray): DoubleArray`
so no single method blows past the JVM 65535-byte size limit. `predict`
standardizes, sums the tree vectors, and returns the winning label.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import m2cgen as m2c
import numpy as np

from train import load_model


_TREE_BODY_RE = re.compile(
    r"public static double\[\] score\(double\[\] input\) \{\s*"
    r"double\[\] var0;\s*(.*)return var0;\s*\}",
    re.DOTALL,
)


def _java_body_to_kotlin(body: str) -> str:
    """Rewrite the innards of a m2cgen `score()` method into Kotlin.

    m2cgen emits nested `if (cond) { ... } else { ... }` where each terminal
    branch does `var0 = new double[] {..};`. Kotlin's `if` is an expression,
    but rewriting the whole tree that way is fragile -- we keep the block
    structure and just replace the leaf assignments and terminal return.
    """
    kt = body
    kt = re.sub(
        r"var0 = new double\[\] \{([^}]*)\};",
        r"return doubleArrayOf(\1)",
        kt,
    )
    kt = kt.replace("input[", "x[")
    return kt


def _tree_to_kotlin_fn(idx: int, java_src: str) -> str:
    m = _TREE_BODY_RE.search(java_src)
    if not m:
        raise RuntimeError(f"could not parse m2cgen output for tree {idx}")
    body = _java_body_to_kotlin(m.group(1))
    body = "\n".join("    " + line if line.strip() else "" for line in body.splitlines())
    return (
        f"    private fun tree{idx}(x: DoubleArray): DoubleArray {{\n"
        f"{body}\n"
        f"        error(\"unreachable\")\n"
        f"    }}"
    )


def _double_array_literal(name: str, arr: np.ndarray, indent: str = "    ") -> str:
    values = ", ".join(f"{v!r}" for v in arr.astype(float).tolist())
    return f"{indent}private val {name}: DoubleArray = doubleArrayOf({values})"


def render_kotlin(joblib_path: Path, package: str, class_name: str) -> str:
    pipe, labels, feature_names = load_model(joblib_path)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    n_features = int(scaler.mean_.shape[0])
    n_classes = int(clf.n_classes_)
    n_trees = int(clf.n_estimators)
    labels_str = [str(l) for l in labels]

    tree_fns = [
        _tree_to_kotlin_fn(i, m2c.export_to_java(est, class_name="T"))
        for i, est in enumerate(clf.estimators_)
    ]

    lines: list[str] = []
    lines.append(f"package {package}")
    lines.append("")
    lines.append("/** Auto-generated from a scikit-learn RandomForest pipeline.")
    lines.append(f" *  source: {joblib_path.name}")
    lines.append(f" *  {n_features} features / {n_trees} trees / {n_classes} classes")
    lines.append(" *  Do not edit by hand -- regenerate via export.py.")
    lines.append(" */")
    lines.append(f"object {class_name} {{")
    lines.append("")

    feat_lit = ", ".join(f'"{n}"' for n in feature_names)
    lines.append(f"    val featureNames: List<String> = listOf({feat_lit})")
    lbl_lit = ", ".join(f'"{n}"' for n in labels_str)
    lines.append(f"    val labels: List<String> = listOf({lbl_lit})")
    lines.append(f"    private const val N_FEATURES = {n_features}")
    lines.append(f"    private const val N_CLASSES = {n_classes}")
    lines.append(f"    private const val N_TREES = {n_trees}")
    lines.append("")

    lines.append(_double_array_literal("scalerMean", scaler.mean_))
    lines.append(_double_array_literal("scalerScale", scaler.scale_))
    lines.append("")

    lines.extend(tree_fns)
    lines.append("")

    lines.append("    /** Predict a gesture label from an unscaled feature vector. */")
    lines.append("    fun predict(features: DoubleArray): String {")
    lines.append("        require(features.size == N_FEATURES) {")
    lines.append("            \"expected $N_FEATURES features, got ${features.size}\"")
    lines.append("        }")
    lines.append("        val scaled = DoubleArray(N_FEATURES) {")
    lines.append("            (features[it] - scalerMean[it]) / scalerScale[it]")
    lines.append("        }")
    lines.append("        val scores = DoubleArray(N_CLASSES)")
    lines.append("        for (t in 0 until N_TREES) {")
    lines.append("            val v = tree(t, scaled)")
    lines.append("            for (c in 0 until N_CLASSES) scores[c] += v[c]")
    lines.append("        }")
    lines.append("        var best = 0")
    lines.append("        for (c in 1 until N_CLASSES) if (scores[c] > scores[best]) best = c")
    lines.append("        return labels[best]")
    lines.append("    }")
    lines.append("")
    lines.append("    /** Same as [predict] but also returns class probabilities. */")
    lines.append("    fun predictWithProbs(features: DoubleArray): Pair<String, DoubleArray> {")
    lines.append("        require(features.size == N_FEATURES) {")
    lines.append("            \"expected $N_FEATURES features, got ${features.size}\"")
    lines.append("        }")
    lines.append("        val scaled = DoubleArray(N_FEATURES) {")
    lines.append("            (features[it] - scalerMean[it]) / scalerScale[it]")
    lines.append("        }")
    lines.append("        val scores = DoubleArray(N_CLASSES)")
    lines.append("        for (t in 0 until N_TREES) {")
    lines.append("            val v = tree(t, scaled)")
    lines.append("            for (c in 0 until N_CLASSES) scores[c] += v[c]")
    lines.append("        }")
    lines.append("        val probs = DoubleArray(N_CLASSES) { scores[it] / N_TREES }")
    lines.append("        var best = 0")
    lines.append("        for (c in 1 until N_CLASSES) if (probs[c] > probs[best]) best = c")
    lines.append("        return labels[best] to probs")
    lines.append("    }")
    lines.append("")

    lines.append("    private fun tree(t: Int, x: DoubleArray): DoubleArray = when (t) {")
    for i in range(n_trees):
        lines.append(f"        {i} -> tree{i}(x)")
    lines.append("        else -> error(\"tree index out of range: $t\")")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compile a joblib RandomForest to Kotlin.")
    p.add_argument("model", help="path to a RandomForest .joblib file")
    p.add_argument("--out", default="GestureRandomForest.kt",
                   help="output Kotlin file (default: GestureRandomForest.kt)")
    p.add_argument("--package", default="dev.gc.gesture",
                   help="Kotlin package (default: dev.gc.gesture)")
    p.add_argument("--class-name", default="GestureRandomForest",
                   help="Kotlin object name (default: GestureRandomForest)")
    args = p.parse_args()

    src = render_kotlin(Path(args.model), args.package, args.class_name)
    out = Path(args.out)
    out.write_text(src, encoding="utf-8")
    print(f"wrote {out} ({len(src):,} bytes)")


if __name__ == "__main__":
    main()
