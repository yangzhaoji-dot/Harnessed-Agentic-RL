# Qwen3-8B GiGPO ALFWorld token-length statistics

> Re-tokenized from the public JSONL `input` / `output` strings with the official `Qwen/Qwen3-8B` tokenizer (`add_special_tokens=False`). Each JSONL row is one model decision/action turn, not a full trajectory. Exact original trainer token counts may differ slightly if the dump decode/re-encode path was not perfectly lossless.

## Aggregate over sampled optimizer dumps (step 1 + every 20 through 400)

| unit | average | median | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|
| input prompt | 503.4 | 471.0 | 685.7 | 756.0 | 1293 |
| assistant output (reasoning + action) | 708.5 | 667.0 | 1148.0 | 1371.0 | 2049 |
| decision/action turn total (input + output) | 1211.9 | 1161.0 | 1695.0 | 1911.0 | 2937 |
| action command only | 6.0 | 6.0 | 9.0 | 10.0 | 14 |

Total sampled decision rows: **60,864**.

## By optimizer step — average / max decision-turn tokens

| optimizer step | rows | input avg | input max | output avg | output max | turn avg | turn max | action cmd avg | action cmd max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5120 | 472.0 | 1020 | 1202.3 | 2049 | 1674.4 | 2914 | 4.7 | 12 |
| 20 | 4480 | 460.8 | 1097 | 818.2 | 2048 | 1279.0 | 2635 | 5.6 | 13 |
| 40 | 3008 | 507.8 | 1118 | 760.3 | 2048 | 1268.1 | 2604 | 5.6 | 13 |
| 60 | 4160 | 481.2 | 1020 | 686.0 | 2043 | 1167.3 | 2508 | 5.8 | 14 |
| 80 | 4544 | 517.8 | 1152 | 718.1 | 2048 | 1235.9 | 2749 | 5.7 | 11 |
| 100 | 3392 | 485.1 | 998 | 677.2 | 2048 | 1162.3 | 2625 | 5.8 | 11 |
| 120 | 3200 | 472.4 | 912 | 729.7 | 2048 | 1202.1 | 2546 | 6.0 | 14 |
| 140 | 2944 | 474.5 | 1017 | 722.0 | 2048 | 1196.6 | 2534 | 6.1 | 12 |
| 160 | 3584 | 556.8 | 1093 | 760.9 | 2048 | 1317.7 | 2713 | 6.1 | 13 |
| 180 | 2752 | 512.0 | 1034 | 621.3 | 2048 | 1133.3 | 2937 | 6.5 | 14 |
| 200 | 2752 | 474.7 | 953 | 596.7 | 2048 | 1071.4 | 2688 | 6.1 | 14 |
| 220 | 2112 | 547.6 | 1044 | 554.1 | 1584 | 1101.7 | 2191 | 6.4 | 11 |
| 240 | 2048 | 494.6 | 1105 | 488.0 | 1328 | 982.6 | 2053 | 6.2 | 11 |
| 260 | 2560 | 563.6 | 1185 | 517.5 | 2048 | 1081.1 | 2545 | 5.9 | 11 |
| 280 | 2304 | 484.4 | 1070 | 512.9 | 2048 | 997.3 | 2572 | 6.6 | 11 |
| 300 | 1536 | 472.9 | 863 | 552.1 | 2048 | 1025.0 | 2615 | 6.7 | 14 |
| 320 | 1472 | 509.0 | 1167 | 547.7 | 2048 | 1056.7 | 2637 | 6.7 | 14 |
| 340 | 1792 | 567.5 | 1293 | 650.4 | 2048 | 1217.9 | 2606 | 6.6 | 14 |
| 360 | 2560 | 574.1 | 1048 | 655.8 | 1789 | 1229.8 | 2264 | 6.5 | 12 |
| 380 | 2624 | 503.6 | 987 | 633.5 | 2048 | 1137.1 | 2616 | 6.6 | 14 |
| 400 | 1920 | 518.3 | 1039 | 638.5 | 1862 | 1156.8 | 2655 | 6.7 | 11 |

## By ALFWorld environment-step bucket — all sampled dumps

| env step | rows | input avg | input P95 | input max | output avg | output max | turn avg | turn max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | 2738 | 630.3 | 963.0 | 1267 | 496.1 | 2048 | 1126.4 | 2688 |
| 3-5 | 7914 | 560.5 | 867.0 | 1293 | 584.1 | 2048 | 1144.6 | 2937 |
| 6-10 | 10118 | 494.1 | 707.0 | 990 | 635.7 | 2048 | 1129.8 | 2741 |
| 11-20 | 11938 | 489.8 | 707.0 | 995 | 769.9 | 2048 | 1259.8 | 2902 |
| 21-30 | 9375 | 486.7 | 702.0 | 995 | 795.6 | 2049 | 1282.3 | 2700 |
| 31-40 | 8334 | 484.6 | 699.3 | 1053 | 782.4 | 2048 | 1267.0 | 2914 |
| 41-50+ | 7715 | 484.3 | 701.0 | 994 | 777.8 | 2048 | 1262.1 | 2817 |
| unknown | 2732 | 472.5 | 846.0 | 877 | 564.1 | 1808 | 1036.6 | 2218 |

## Interpretation note

- `turn_total` is one decision/action turn: the prompt visible before the decision plus that assistant response. It is **not** the cumulative full-trajectory token count.
- Because this GiGPO run keeps only a short recent history window, prompt length should become roughly bounded after the early steps rather than growing linearly with trajectory length. The env-step table tests this directly.
- The 4B full-history recipe uses a different accounting regime: its `12K` cap is on the cumulative full trajectory, so do not compare `12K` directly against the 8B per-turn maximum.