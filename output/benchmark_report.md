# Receipt OCR Pipeline — Benchmark & Evaluation Report

**Date:** 2026-10-13 20:49:08  
**Total Samples Evaluated:** 4  

## Performance & Latency Summary

| Document | Engine | Backend | Latency (s) | Throughput (docs/min) | Extracted Items | Math Integrity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `receipt_synth_001.png` | `vl` | `mlx-vlm-server` | **8.003** | 7.5 | 4 | 0.0% |
| `receipt_synth_002.png` | `vl` | `mlx-vlm-server` | **4.425** | 13.6 | 4 | 100.0% |
| `receipt_synth_003.png` | `vl` | `mlx-vlm-server` | **4.956** | 12.1 | 4 | 100.0% |
| `receipt_synth_004.png` | `vl` | `mlx-vlm-server` | **4.107** | 14.6 | 4 | 100.0% |

## Ground Truth Accuracy Metrics

| Document | Expected Items | Extracted Items | Name Similarity | Sum Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `receipt_synth_001.png` | 4 | 4 | **29.0%** | **100.0%** |
| `receipt_synth_002.png` | 4 | 4 | **93.4%** | **26.3%** |
| `receipt_synth_003.png` | 4 | 4 | **99.1%** | **100.0%** |
| `receipt_synth_004.png` | 4 | 4 | **19.8%** | **91.5%** |

> *Generated automatically by `benchmark.py`.*