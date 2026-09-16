# Qwen3-8B GiGPO ALFWorld raw `score` curve

Sampled at optimizer step 1 and every 20 steps through 400 from the public rollout dumps. `score_mean` is the mean over JSONL rows, so it is a **row-weighted dump statistic**, not necessarily identical to the trainer's episode-level success-rate metric.

| optimizer step | mean score | high score (>=9) | zero | -0.1 | rows |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.182 | 22.0% | 62.7% | 15.3% | 5120 |
| 20 | 2.294 | 22.9% | 76.8% | 0.3% | 4480 |
| 40 | 3.255 | 32.5% | 67.4% | 0.0% | 3008 |
| 60 | 1.733 | 17.3% | 82.6% | 0.0% | 4160 |
| 80 | 2.242 | 22.4% | 77.6% | 0.0% | 4544 |
| 100 | 2.818 | 28.2% | 71.8% | 0.0% | 3392 |
| 120 | 4.206 | 42.1% | 57.9% | 0.0% | 3200 |
| 140 | 3.804 | 38.0% | 62.0% | 0.0% | 2944 |
| 160 | 1.755 | 17.6% | 82.4% | 0.0% | 3584 |
| 180 | 4.262 | 42.6% | 57.4% | 0.0% | 2752 |
| 200 | 2.798 | 28.0% | 71.9% | 0.1% | 2752 |
| 220 | 7.580 | 75.8% | 24.2% | 0.0% | 2112 |
| 240 | 6.265 | 62.6% | 37.4% | 0.0% | 2048 |
| 260 | 4.918 | 49.2% | 50.8% | 0.0% | 2560 |
| 280 | 6.011 | 60.1% | 39.9% | 0.0% | 2304 |
| 300 | 7.337 | 73.4% | 26.6% | 0.0% | 1536 |
| 320 | 8.933 | 89.3% | 10.6% | 0.1% | 1472 |
| 340 | 7.126 | 71.3% | 28.6% | 0.2% | 1792 |
| 360 | 4.902 | 49.0% | 51.0% | 0.0% | 2560 |
| 380 | 4.371 | 43.7% | 56.1% | 0.2% | 2624 |
| 400 | 7.578 | 75.8% | 24.2% | 0.0% | 1920 |

## Interpretation

- In these dumps, scores are concentrated near `10`, `9.9`, `0`, and `-0.1`.
- `10/9.9` are high-outcome rows; `0` are unsuccessful rows without the small penalty; `-0.1` indicates an unsuccessful row with the penalty used by this run.
- Because multi-turn rollouts are flattened into dump rows, longer trajectories can contribute more rows. Use this curve for training-behavior diagnostics; use the model card / trainer metrics for episode-level success rate.

Source: `wckwan/PRM-agent-rl-artifacts/rollouts/alfworld_qwen3_8b_gigpo` (public Hugging Face dataset).