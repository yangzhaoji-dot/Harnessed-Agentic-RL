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
    if "none" in t or ("not" in t and "rl" in t):
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


# Conceptual landscape coordinates for papers we have manually reviewed.
# x: 0=fixed harness, 3=structural/evolutionary harness
# y: 0=served model fixed, 3=strong/joint model learning
LANDSCAPE_OVERRIDES = {
    "2605.08741": (0.4, 1.4),   # OPHSD
    "2607.22688": (2.7, 1.6),   # Co-Harness
    "2608.17528": (0.5, 2.4),   # Agent Lightning
    "2607.21557": (0.5, 2.3),   # OpenForgeRL
    "2608.29641": (0.8, 2.5),   # Harness-RL
    "2608.05446": (2.0, 2.6),   # EvoHarness-RL
    "2607.25825": (2.3, 0.4),   # CHILL-Harness
    "2609.02786": (2.9, 2.8),   # SafeEvolve
    "2608.25593": (3.0, 0.6),   # JIT-Agent
    "2608.23041": (3.0, 0.4),   # AutoSaddler
    "2608.24804": (3.0, 0.3),   # StarHarness
    "2608.19880": (2.8, 2.2),   # EnvHarness
    "2608.15763": (2.3, 2.5),   # TaoLive HAT
    "2606.01619": (2.7, 2.6),   # ReSkill
    "2603.28716": (2.5, 2.5),   # D2Skill
    "2605.27899": (1.8, 2.7),   # SKILLC
}

LANDSCAPE_LABELS = {
    "2605.08741": "OPHSD",
    "2607.22688": "Co-Harness",
    "2608.17528": "Agent Lightning",
    "2607.21557": "OpenForgeRL",
    "2608.29641": "Harness-RL",
    "2608.05446": "EvoHarness-RL",
    "2607.25825": "CHILL",
    "2609.02786": "SafeEvolve",
    "2608.25593": "JIT-Agent",
    "2608.23041": "AutoSaddler",
    "2608.24804": "StarHarness",
    "2608.19880": "EnvHarness",
    "2608.15763": "TaoLive HAT",
    "2606.01619": "ReSkill",
    "2603.28716": "D2Skill",
    "2605.27899": "SKILLC",
}


def infer_harness_score(row):
    text = " ".join(
        str(row.get(k, ""))
        for k in ["category", "harness_design", "harness_change", "training_method"]
    ).lower()
    if any(k in text for k in ["co-evolution", "co-evolving", "self-evolved", "evolution", "evolving", "stratified search", "code patches", "created/revised/versioned/pruned"]):
        return 2.8
    if any(k in text for k in ["adaptive", "runtime workflow intervention", "state evolves", "skill bank", "learned controller", "router"]):
        return 2.0
    if "fixed" in text:
        return 0.5
    return 1.2


def infer_model_score(row):
    rl = str(row.get("rl_algorithm", "")).lower()
    train = str(row.get("training_method", "")).lower()
    category = str(row.get("category", "")).lower()
    if any(k in rl for k in ["grpo", "ppo", "capo", "reinforcement", "agentic rl"]) or " rl" in train:
        score = 2.4
    elif any(k in train for k in ["distill", "opd", "fine-tun", "sft"]):
        score = 1.4
    else:
        score = 0.4
    if "co-evolution" in category and score >= 2.0:
        score = 2.8
    return score


def short_label(row):
    arxiv = str(row.get("arxiv", "")).strip()
    if arxiv in LANDSCAPE_LABELS:
        return LANDSCAPE_LABELS[arxiv]
    title = str(row.get("title", "Paper"))
    words = title.split()
    return " ".join(words[:3]) + ("…" if len(words) > 3 else "")


def save_landscape(df):
    points = []
    for _, row in df.iterrows():
        arxiv = str(row.get("arxiv", "")).strip()
        if arxiv in LANDSCAPE_OVERRIDES:
            x, y = LANDSCAPE_OVERRIDES[arxiv]
        else:
            x, y = infer_harness_score(row), infer_model_score(row)
        points.append((x, y, short_label(row)))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=70, alpha=0.85)

    offsets = [(6, 6), (6, -12), (-8, 8), (-8, -14), (10, 0)]
    for i, (x, y, label) in enumerate(points):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5)

    ax.axvline(1.75, alpha=0.18)
    ax.axhline(1.75, alpha=0.18)
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 3.15)
    ax.set_xticks([0, 1, 2, 3], ["Fixed", "Adaptive use", "Learned / optimized", "Structural evolution"])
    ax.set_yticks([0, 1, 2, 3], ["Model fixed", "SFT / KD", "RL-trained", "Joint / strong learning"])
    ax.set_xlabel("Harness mutability / optimization degree →")
    ax.set_ylabel("Served-model learning degree →")
    ax.set_title("Harness × Agentic RL Research Landscape")
    ax.grid(alpha=0.15)

    ax.text(0.12, 2.96, "Harnessed RL / internalization", fontsize=9, alpha=0.65)
    ax.text(2.02, 2.96, "Co-evolution frontier", fontsize=9, alpha=0.65)
    ax.text(2.02, 0.12, "Harness-side optimization", fontsize=9, alpha=0.65)
    ax.text(0.12, 0.12, "Fixed-system baseline", fontsize=9, alpha=0.65)
    fig.text(
        0.5,
        0.01,
        "Conceptual map: manually curated coordinates for reviewed papers; heuristic fallback for new entries. Not a performance ranking.",
        ha="center",
        fontsize=8,
        alpha=0.65,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT_DIR / "research_landscape.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 1) Conceptual research landscape
    save_landscape(df)

    # 2) Papers over time
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

    # 3) Research categories
    categories = Counter()
    for v in df["category"]:
        categories.update(split_field(v))
    save_bar(categories, "Research Categories", "Number of papers", "research_categories.svg", topn=14)

    # 4) RL algorithms / training paradigms
    rl = Counter()
    for v in df["rl_algorithm"]:
        rl.update(normalize_rl(v))
    save_bar(rl, "RL / Optimization Algorithms", "Number of papers", "rl_algorithms.svg", topn=12)

    # 5) Model scales
    scales = Counter()
    for v in df["model_scale"]:
        scales.update(normalize_model_scales(v))
    save_bar(scales, "Model Scale Distribution", "Number of paper entries", "model_scale.svg", topn=14)

    # 6) Benchmarks
    benchmarks = Counter()
    for v in df["benchmark"]:
        for b in split_field(v):
            if b and b.lower() not in {"tbd", "varies"}:
                benchmarks[b] += 1
    save_bar(benchmarks, "Most Used Benchmarks", "Number of papers", "benchmarks.svg", topn=15)

    # 7) Internalization evidence
    evidence = Counter(classify_internalization(v) for v in df["internalization_evidence"])
    save_bar(evidence, "Internalization Evidence Strength", "Number of papers", "internalization_evidence.svg", topn=10)

    print(f"Generated statistics for {len(df)} papers in {OUT_DIR}")


if __name__ == "__main__":
    main()
