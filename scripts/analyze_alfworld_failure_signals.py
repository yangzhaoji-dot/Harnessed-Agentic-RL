#!/usr/bin/env python3
"""Quantify observable failure signals in public ALFWorld GiGPO rollout dumps.

The public JSONL dump omits traj_uid/uid/anchor_obs, so this script only reports
signals that are identifiable from each model-visible decision row. It does NOT
claim to label latent causes such as planning failure or memory failure.

Reliable / directly observable signals:
- protocol invalidity under verl-agent's ALFWorld projection convention
- parsed action not in the prompt's admissible-action list
- current observation == 'Nothing happens.'
- two-step state revisit visible in the history window
- strong ABA loop continuation (state revisit + current action repeats action t-2)
- late-step occupancy (>=40 / >=50)
- exact-context branch candidates (same prompt context, different actions, mixed outcome)
- same-context + same-action + mixed outcome negative controls
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TASK_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.I)
ENV_STEP_RE = re.compile(r"You are now at step\s+(\d+)", re.I)
CUR_OBS_RE = re.compile(
    r"current observation is:\s*(.*?)(?:\nYour admissible actions of the current situation are:)",
    re.I | re.S,
)
ADM_RE = re.compile(
    r"Your admissible actions of the current situation are:\s*(.*?)\.(?:\n\nNow it's your turn|\nNow it's your turn)",
    re.I | re.S,
)
HIST_PAIR_RE = re.compile(
    r"\[Observation\s+(\d+):\s*'(.*?)',\s*Action\s+\1:\s*'([^']*)'\]",
    re.I | re.S,
)
ACTION_CLOSED_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
ACTION_OPEN_RE = re.compile(r"<action>\s*([^\n<]+)", re.I)
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def extract_task(inp: str) -> str | None:
    m = TASK_RE.search(inp)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def extract_env_step(inp: str) -> int | None:
    m = ENV_STEP_RE.search(inp)
    if m:
        return int(m.group(1))
    # The first step uses a no-history template in some dumps.
    if "Welcome to TextWorld, ALFRED" in inp and "Prior to this step" not in inp:
        return 1
    return None


def extract_obs(inp: str) -> str | None:
    m = CUR_OBS_RE.search(inp)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    # First-step no-history prompt: use the initial room description up to admissible actions.
    marker = "Your admissible actions of the current situation are:"
    if marker in inp:
        pre = inp.split(marker, 1)[0]
        # Keep the task-bearing full prompt prefix as a conservative state text.
        return re.sub(r"\s+", " ", pre).strip()
    return None


def extract_admissible(inp: str) -> list[str]:
    m = ADM_RE.search(inp)
    if not m:
        return []
    return [norm(x) for x in re.findall(r"'([^']+)'", m.group(1))]


def extract_history(inp: str) -> list[tuple[int, str, str]]:
    return [
        (int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip(), re.sub(r"\s+", " ", m.group(3)).strip())
        for m in HIST_PAIR_RE.finditer(inp)
    ]


def extract_action(out: str) -> tuple[str | None, bool]:
    m = ACTION_CLOSED_RE.search(out)
    if m:
        return norm(m.group(1)), True
    m = ACTION_OPEN_RE.search(out)
    if m:
        return norm(m.group(1)), False
    return None, False


def protocol_valid(out: str) -> bool:
    # Mirrors the verl-agent projection validity convention: both tag pairs exist,
    # and no Chinese characters are present. The environment may still execute a
    # fallback string even when this returns False.
    return (
        "<action>" in out.lower()
        and "</action>" in out.lower()
        and "<think>" in out.lower()
        and "</think>" in out.lower()
        and CHINESE_RE.search(out) is None
    )


def score_class(score: float) -> str:
    if score >= 9:
        return "high"
    if score <= 0:
        return "low"
    return "mid"


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def analyze_file(path: Path, optimizer_step: int) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            x = json.loads(line)
            inp = str(x.get("input", ""))
            out = str(x.get("output", ""))
            score = float(x.get("score", 0.0))
            env_step = extract_env_step(inp)
            obs = extract_obs(inp)
            hist = extract_history(inp)
            adm = extract_admissible(inp)
            action, action_closed = extract_action(out)
            pvalid = protocol_valid(out)
            semantic_admissible = (action in adm) if (action is not None and adm) else None
            hist_norm = [(st, norm(o), norm(a)) for st, o, a in hist]
            obs_norm = norm(obs)
            current_action_norm = norm(action) if action else None

            nothing = obs_norm == "nothing happens."
            state_revisit_2 = len(hist_norm) >= 2 and obs_norm == hist_norm[-2][1]
            repeat_prev = len(hist_norm) >= 1 and current_action_norm is not None and current_action_norm == hist_norm[-1][2]
            repeat_2back = len(hist_norm) >= 2 and current_action_norm is not None and current_action_norm == hist_norm[-2][2]
            strong_aba = state_revisit_2 and repeat_2back

            rows.append(
                {
                    "row": row_idx,
                    "optimizer_step": optimizer_step,
                    "task": extract_task(inp),
                    "env_step": env_step,
                    "obs": obs,
                    "history": hist,
                    "action": action,
                    "action_closed": action_closed,
                    "protocol_valid": pvalid,
                    "semantic_admissible": semantic_admissible,
                    "nothing_happens": nothing,
                    "state_revisit_2": state_revisit_2,
                    "repeat_prev": repeat_prev,
                    "repeat_2back": repeat_2back,
                    "strong_aba": strong_aba,
                    "score": score,
                    "score_class": score_class(score),
                    "context_sig": sha1_text(re.sub(r"\s+", " ", inp).strip()),
                }
            )

    n = len(rows)
    low = [r for r in rows if r["score_class"] == "low"]
    high = [r for r in rows if r["score_class"] == "high"]
    parsed = [r for r in rows if r["action"] is not None]
    semantic_checkable = [r for r in rows if r["semantic_admissible"] is not None]

    # Exact visible-context grouping: the normalized input prompt is identical.
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["context_sig"]].append(r)

    branch_candidates = 0
    mixed_contexts = 0
    same_action_mixed_controls = 0
    repeated_contexts = 0
    multi_action_contexts = 0
    for rr in groups.values():
        if len(rr) >= 2:
            repeated_contexts += 1
        actions = {r["action"] for r in rr}
        if len(actions) >= 2:
            multi_action_contexts += 1
        has_hi = any(r["score_class"] == "high" for r in rr)
        has_lo = any(r["score_class"] == "low" for r in rr)
        if has_hi and has_lo:
            mixed_contexts += 1
            if len(actions) >= 2:
                branch_candidates += 1
        by_action: dict[str | None, list[dict]] = defaultdict(list)
        for r in rr:
            by_action[r["action"]].append(r)
        if any(
            any(x["score_class"] == "high" for x in aa)
            and any(x["score_class"] == "low" for x in aa)
            for aa in by_action.values()
        ):
            same_action_mixed_controls += 1

    def count(flag: str, subset: list[dict]) -> int:
        return sum(bool(r[flag]) for r in subset)

    result = {
        "optimizer_step": optimizer_step,
        "rows": n,
        "mean_score": statistics.fmean(r["score"] for r in rows) if rows else 0.0,
        "high_rows": len(high),
        "high_pct": pct(len(high), n),
        "low_rows": len(low),
        "low_pct": pct(len(low), n),
        "protocol_invalid_rows": n - count("protocol_valid", rows),
        "protocol_invalid_pct": pct(n - count("protocol_valid", rows), n),
        "protocol_invalid_low_pct": pct(len(low) - count("protocol_valid", low), len(low)),
        "semantic_nonadmissible_rows": sum(r["semantic_admissible"] is False for r in rows),
        "semantic_nonadmissible_pct_checkable": pct(sum(r["semantic_admissible"] is False for r in rows), len(semantic_checkable)),
        "semantic_nonadmissible_low_pct_checkable": pct(sum(r["semantic_admissible"] is False for r in low), sum(r["semantic_admissible"] is not None for r in low)),
        "nothing_happens_pct": pct(count("nothing_happens", rows), n),
        "nothing_happens_low_pct": pct(count("nothing_happens", low), len(low)),
        "state_revisit_2_pct": pct(count("state_revisit_2", rows), n),
        "state_revisit_2_low_pct": pct(count("state_revisit_2", low), len(low)),
        "repeat_prev_pct": pct(count("repeat_prev", rows), n),
        "repeat_prev_low_pct": pct(count("repeat_prev", low), len(low)),
        "strong_aba_pct": pct(count("strong_aba", rows), n),
        "strong_aba_low_pct": pct(count("strong_aba", low), len(low)),
        "late40_pct": pct(sum((r["env_step"] or 0) >= 40 for r in rows), n),
        "late40_low_pct": pct(sum((r["env_step"] or 0) >= 40 for r in low), len(low)),
        "step50_pct": pct(sum((r["env_step"] or 0) >= 50 for r in rows), n),
        "contexts": len(groups),
        "repeated_contexts": repeated_contexts,
        "multi_action_contexts": multi_action_contexts,
        "mixed_high_low_contexts": mixed_contexts,
        "branch_candidate_contexts": branch_candidates,
        "same_action_mixed_contexts": same_action_mixed_controls,
        "parsed_action_pct": pct(len(parsed), n),
        "env_step_median": statistics.median([r["env_step"] for r in rows if r["env_step"] is not None]) if any(r["env_step"] is not None for r in rows) else None,
        "env_step_mean": statistics.fmean([r["env_step"] for r in rows if r["env_step"] is not None]) if any(r["env_step"] is not None for r in rows) else None,
    }
    return result, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="STEP=PATH entries")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rows = []
    for spec in args.inputs:
        step_s, path_s = spec.split("=", 1)
        res, rows = analyze_file(Path(path_s), int(step_s))
        summaries.append(res)
        all_rows.extend(rows)
    summaries.sort(key=lambda x: x["optimizer_step"])

    (args.out / "failure_signal_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = list(summaries[0].keys()) if summaries else []
    with (args.out / "failure_signal_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)

    md = [
        "# Qwen3-8B GiGPO ALFWorld — observable failure-signal evolution",
        "",
        "> Public rollout dumps omit `traj_uid/uid/anchor_obs`, so this report only quantifies signals directly identifiable from the dumped model-visible decision rows. It does **not** claim automatic ground-truth labels for planning, memory, or goal-relevance failures.",
        "",
        "## Main table — all rows",
        "",
        "| train step | rows | high score | protocol invalid | semantic non-admissible* | `Nothing happens` | 2-step state revisit | strong ABA loop | env step >=40 | branch candidates | same-action mixed controls |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        md.append(
            f"| {r['optimizer_step']} | {r['rows']} | {r['high_pct']:.1f}% | {r['protocol_invalid_pct']:.1f}% | {r['semantic_nonadmissible_pct_checkable']:.1f}% | {r['nothing_happens_pct']:.1f}% | {r['state_revisit_2_pct']:.1f}% | {r['strong_aba_pct']:.1f}% | {r['late40_pct']:.1f}% | {r['branch_candidate_contexts']} | {r['same_action_mixed_contexts']} |"
        )
    md += [
        "",
        "\* Semantic non-admissible = parsed action text is not in the prompt's current admissible-action list. This is separate from verl-agent's protocol-validity flag.",
        "",
        "## Conditional on low outcome (`score <= 0`)",
        "",
        "| train step | low rows | protocol invalid | semantic non-admissible* | `Nothing happens` | 2-step state revisit | repeat previous action | strong ABA loop | env step >=40 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        md.append(
            f"| {r['optimizer_step']} | {r['low_rows']} | {r['protocol_invalid_low_pct']:.1f}% | {r['semantic_nonadmissible_low_pct_checkable']:.1f}% | {r['nothing_happens_low_pct']:.1f}% | {r['state_revisit_2_low_pct']:.1f}% | {r['repeat_prev_low_pct']:.1f}% | {r['strong_aba_low_pct']:.1f}% | {r['late40_low_pct']:.1f}% |"
        )
    md += [
        "",
        "## Definitions",
        "",
        "- **Protocol invalid**: missing `<think>...</think>` or `<action>...</action>` pair, or contains Chinese characters, matching the verl-agent ALFWorld projection validity convention.",
        "- **Semantic non-admissible**: the extracted action string is absent from the current prompt's admissible-action list.",
        "- **2-step state revisit**: current observation text equals the observation from two decisions earlier, a conservative visible-window loop signal.",
        "- **Strong ABA loop**: a 2-step state revisit plus the current sampled action repeats the action from two decisions earlier.",
        "- **Branch candidate**: identical visible prompt context appears with different sampled actions and both high and low eventual scores. It is an intervention candidate, not causal proof.",
        "- **Same-action mixed control**: identical visible context and identical current action can still have high and low eventual scores, demonstrating downstream-confounding of trajectory score.",
        "",
        "## What this dump cannot identify reliably",
        "",
        "- `history loss / memory failure`: full trajectory identity and pre-window observations are not preserved in the public dump.",
        "- `planning failure` and `goal-affordance failure`: these require replay, a task-state parser, or manual/LLM labeling; they should not be inferred solely from final score.",
        "",
    ]
    (args.out / "failure_signal_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
