from pathlib import Path
from collections import Counter
import re

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "papers" / "papers.csv"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def split_field(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def save_bar(counter, title, xlabel, filename, topn=12):
    items = counter.most_common(topn)
    if not items:
        return
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.45 * len(labels))))
    y = range(len(labels))
    ax.barh(list(y), values)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def normalize_rl(text):
    t = str(text).lower()
    labels = []
    if "grpo" in t:
        labels.append("GRPO")
    if "ppo" in t and "grpo" not in t:
        labels.append("PPO")
    if "capo" in t:
        labels.append("CAPO")
    if "distill" in t or "reverse-kl" in t or "opd" in t:
        labels.append("Distillation / OPD")
    if "none" in t or "not" in t and "rl" in t:
        labels.append("No base-model RL")
    if not labels:
        labels.append("Other / TBD")
    return labels


def classify_internalization(text):
    t = str(text).lower()
    if any(k in t for k in ["explicitly evaluates without runtime skill", "harness-free", "standalone model", "reattaching harness"]):
        return "Harness-free / standalone eval"
    if any(k in t for k in ["paired", "without-component", "performance gap", "skill-injected"]):
        return "Paired removal / utility test"
    if any(k in t for k in ["usage anneal", "usage decay", "calls decrease", "indirect"]):
        return "Usage decay only"
    if any(k in t for k in ["claims stronger autonomous", "distilled into model", "supporting real capability transfer"]):
        return "Claimed transfer, limited causal test"
    return "None / not studied"


def normalize_model_scales(value):
    out = []
    for part in split_field(value):
        p = part.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?B", p, flags=re.I):
            out.append(p.upper())
        elif "api" in p.lower() or "frontier" in p.lower() or "mixed" in p.lower():
            out.append("API / mixed")
        elif p and p.lower() != "tbd":
            out.append(p)
    return out


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 1) Papers over time
    timeline = df.dropna(subset=["date"]).copy()
    timeline["month"] = timeline["date"].dt.to_period("M").astype(str)
    counts = timeline.groupby("month").size().sort_index()
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(counts.index, counts.values, marker="o")
        ax.set_title("Harness × Agentic RL Papers Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Papers")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "papers_over_time.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    # 2) Research categories
    categories = Counter()
    for v in df["category"]:
        categories.update(split_field(v))
    save_bar(categories, "Research Categories", "Number of papers", "research_categories.svg", topn=14)

    # 3) RL algorithms / training paradigms
    rl = Counter()
    for v in df["rl_algorithm"]:
        rl.update(normalize_rl(v))
    save_bar(rl, "RL / Optimization Algorithms", "Number of papers", "rl_algorithms.svg", topn=12)

    # 4) Model scales
    scales = Counter()
    for v in df["model_scale"]:
        scales.update(normalize_model_scales(v))
    save_bar(scales, "Model Scale Distribution", "Number of paper entries", "model_scale.svg", topn=14)

    # 5) Benchmarks
    benchmarks = Counter()
    for v in df["benchmark"]:
        for b in split_field(v):
            if b and b.lower() not in {"tbd", "varies"}:
                benchmarks[b] += 1
    save_bar(benchmarks, "Most Used Benchmarks", "Number of papers", "benchmarks.svg", topn=15)

    # 6) Internalization evidence
    evidence = Counter(classify_internalization(v) for v in df["internalization_evidence"])
    save_bar(evidence, "Internalization Evidence Strength", "Number of papers", "internalization_evidence.svg", topn=10)

    print(f"Generated statistics for {len(df)} papers in {OUT_DIR}")


if __name__ == "__main__":
    main()
