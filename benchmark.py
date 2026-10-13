"""
benchmark.py — Quantitative Benchmarking & Evaluation Suite for Receipt OCR.

Measures:
  1. Latency & Throughput (per stage: Preprocessing, OCR Extraction, Validation, Export)
  2. Hardware Acceleration Efficiency (Metal GPU vs CPU)
  3. Mathematical Integrity Rate (Qty * Price == Total)
  4. Ground-Truth Accuracy Metrics (when paired with _truth.json from tools/synthesize.py):
     - Item Count Recall
     - Field-level Match Rate
     - Financial Total Absolute Error

Usage:
    # Benchmark single image with active GPU engine:
    python benchmark.py --image receipts/first.jpg

    # Benchmark and compare engines (PaddleOCR-VL MLX vs PP-Structure CPU):
    python benchmark.py --image receipts/first.jpg --compare-engines

    # Benchmark a directory of synthetic test receipts against ground truth:
    python benchmark.py --folder data/synthetic/
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from src.pipeline import ReceiptOCRPipeline
from config.settings import OCR_ENGINE, VL_BACKEND, VL_PIPELINE_VERSION

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")


def check_math_consistency(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Checks if Quantity * Price == Total for all numeric rows.
    Returns: (valid_rows, total_rows, consistency_rate)
    """
    if df.empty or "Количество" not in df.columns or "Цена" not in df.columns:
        return 0, 0, 0.0

    valid = 0
    total = 0

    for _, row in df.iterrows():
        try:
            qty = float(str(row["Количество"]).replace(" ", "").replace(",", "."))
            price = float(str(row["Цена"]).replace(" ", "").replace(",", "."))
            total_price = float(str(row["Сумма"]).replace(" ", "").replace(",", "."))

            if qty > 0 and price > 0:
                total += 1
                expected = qty * price
                # Allow minor rounding difference (within 1%)
                if abs(expected - total_price) <= max(1.0, 0.01 * expected):
                    valid += 1
        except Exception:
            continue

    rate = (valid / total) if total > 0 else 0.0
    return valid, total, rate


def evaluate_against_ground_truth(df: pd.DataFrame, gt_data: dict) -> dict:
    """
    Evaluates extracted DataFrame against synthetic ground-truth metadata.
    """
    gt_products = gt_data.get("products", [])
    expected_count = len(gt_products)
    actual_count = len(df)

    # Item count match
    count_error = abs(actual_count - expected_count)

    # Name similarity score
    name_scores = []
    extracted_names = [str(n) for n in df["Наименование"].tolist()] if "Наименование" in df.columns else []

    for gt_p in gt_products:
        gt_name = gt_p["name"]
        if extracted_names:
            best_score = max(fuzz.ratio(gt_name.lower(), ex.lower()) for ex in extracted_names)
            name_scores.append(best_score)
        else:
            name_scores.append(0.0)

    avg_name_similarity = sum(name_scores) / len(name_scores) if name_scores else 0.0

    # Total Sum Error
    extracted_sum = 0.0
    if "Сумма" in df.columns:
        for s in df["Сумма"]:
            try:
                extracted_sum += float(str(s).replace(" ", "").replace(",", "."))
            except Exception:
                pass

    expected_sum = float(gt_data.get("total_sum", 0.0))
    sum_abs_error = abs(extracted_sum - expected_sum)
    sum_acc_pct = max(0.0, 100.0 - (sum_abs_error / expected_sum * 100.0)) if expected_sum > 0 else 0.0

    return {
        "expected_items": expected_count,
        "extracted_items": actual_count,
        "count_accuracy_pct": round(max(0.0, 100.0 - (count_error / expected_count * 100.0)), 1) if expected_count else 0.0,
        "avg_name_similarity_pct": round(avg_name_similarity, 1),
        "expected_total_sum": expected_sum,
        "extracted_total_sum": extracted_sum,
        "sum_accuracy_pct": round(sum_acc_pct, 1),
    }


def run_benchmark_on_image(
    image_path: Path,
    engine: str = OCR_ENGINE,
    vl_backend: str = VL_BACKEND,
) -> dict:
    """Runs a timed extraction on a single image and measures performance."""
    pipeline = ReceiptOCRPipeline(
        engine_type=engine,
        vl_backend=vl_backend,
    )

    t0 = time.perf_counter()
    clean_img = pipeline.preprocessor.process(str(image_path))
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_df = pipeline.extractor.extract(clean_img)
    t_extract = time.perf_counter() - t1

    t2 = time.perf_counter()
    products, failed_df = pipeline.validator.validate(raw_df)
    t_val = time.perf_counter() - t2

    total_time = t_prep + t_extract + t_val

    gbs_df = pipeline.validator.to_gbs_dataframe(products)
    valid_math, total_math, math_rate = check_math_consistency(gbs_df)

    metrics = {
        "image": image_path.name,
        "engine": engine,
        "backend": vl_backend if engine == "vl" else "cpu",
        "preprocess_sec": round(t_prep, 3),
        "extract_sec": round(t_extract, 3),
        "validate_sec": round(t_val, 3),
        "total_latency_sec": round(total_time, 3),
        "throughput_docs_per_min": round(60.0 / total_time, 1) if total_time > 0 else 0.0,
        "items_extracted": len(products),
        "failed_rows": len(failed_df),
        "math_consistency_rate_pct": round(math_rate * 100, 1),
    }

    # Check for accompanying ground-truth file
    truth_file = image_path.parent / f"{image_path.stem}_truth.json"
    if truth_file.exists():
        with open(truth_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_metrics = evaluate_against_ground_truth(gbs_df, gt_data)
        metrics["ground_truth"] = gt_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark Receipt OCR Pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single receipt image.")
    group.add_argument("--folder", type=str, help="Path to directory containing receipt images.")
    parser.add_argument("--compare-engines", action="store_true", help="Compare PaddleOCR-VL (MLX GPU) vs PP-Structure (CPU).")
    parser.add_argument("--output-report", type=str, default="output/benchmark_report.md", help="Markdown report output path.")
    args = parser.parse_args()

    results = []

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: {img_path} not found.")
            return

        print(f"\n--- Running Benchmark on {img_path.name} ---")

        if args.compare_engines:
            print("1/2: Evaluating PaddleOCR-VL (Apple Silicon GPU via MLX)...")
            m_vl = run_benchmark_on_image(img_path, engine="vl", vl_backend="mlx-vlm-server")
            results.append(m_vl)

            print("2/2: Evaluating PP-StructureV3 (CPU)...")
            m_struct = run_benchmark_on_image(img_path, engine="structure", vl_backend="native")
            results.append(m_struct)
        else:
            m = run_benchmark_on_image(img_path)
            results.append(m)

    elif args.folder:
        folder = Path(args.folder)
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not images:
            print(f"No .jpg or .png images found in {folder}")
            return

        print(f"\n--- Running Benchmark on {len(images)} images in {folder} ---")
        for idx, img in enumerate(images, 1):
            print(f"[{idx}/{len(images)}] Benchmarking {img.name} ...")
            try:
                m = run_benchmark_on_image(img)
                results.append(m)
            except Exception as e:
                print(f"  ✗ Error on {img.name}: {e}")

    # ── Print Formatted Summary ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("                     RECEIPT OCR BENCHMARK RESULTS")
    print("=" * 80)

    headers = ["Image", "Engine", "Backend", "Latency", "Throughput", "Items", "Math Rate"]
    row_fmt = "{:<20} {:<15} {:<12} {:<10} {:<12} {:<7} {:<10}"
    print(row_fmt.format(*headers))
    print("-" * 80)

    for r in results:
        print(row_fmt.format(
            r["image"][:19],
            r["engine"],
            r["backend"],
            f"{r['total_latency_sec']}s",
            f"{r['throughput_docs_per_min']} doc/m",
            r["items_extracted"],
            f"{r['math_consistency_rate_pct']}%",
        ))

        if "ground_truth" in r:
            gt = r["ground_truth"]
            print(f"  └─ GT Eval: Name Sim: {gt['avg_name_similarity_pct']}% | "
                  f"Item Recall: {gt['count_accuracy_pct']}% | Total Sum Acc: {gt['sum_accuracy_pct']}%")

    print("=" * 80)

    # ── Export Markdown Report ───────────────────────────────────────
    out_path = Path(args.output_report)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    report_lines = [
        "# Receipt OCR Pipeline — Benchmark & Evaluation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Samples Evaluated:** {len(results)}  ",
        "",
        "## Performance & Latency Summary",
        "",
        "| Document | Engine | Backend | Latency (s) | Throughput (docs/min) | Extracted Items | Math Integrity |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        report_lines.append(
            f"| `{r['image']}` | `{r['engine']}` | `{r['backend']}` | "
            f"**{r['total_latency_sec']}** | {r['throughput_docs_per_min']} | "
            f"{r['items_extracted']} | {r['math_consistency_rate_pct']}% |"
        )

    # If ground truth was evaluated
    gt_results = [r for r in results if "ground_truth" in r]
    if gt_results:
        report_lines.extend([
            "",
            "## Ground Truth Accuracy Metrics",
            "",
            "| Document | Expected Items | Extracted Items | Name Similarity | Sum Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for r in gt_results:
            gt = r["ground_truth"]
            report_lines.append(
                f"| `{r['image']}` | {gt['expected_items']} | {gt['extracted_items']} | "
                f"**{gt['avg_name_similarity_pct']}%** | **{gt['sum_accuracy_pct']}%** |"
            )

    report_lines.append("")
    report_lines.append("> *Generated automatically by `benchmark.py`.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✓ Markdown benchmark report saved → {out_path}\n")


if __name__ == "__main__":
    main()
