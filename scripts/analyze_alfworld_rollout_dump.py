#!/usr/bin/env python3
"""Reconstruct ALFWorld episodes from a flattened GiGPO rollout JSONL dump.

Expected row schema for the public wckwan ALFWorld artifacts:
  input: prompt containing recent observation/action history and current observation
  output: one model turn containing <think>...</think> and <action>...</action>
  score: trajectory-level outcome score repeated on all steps of the episode
  step: optimizer-step id (NOT environment-step id)

The script deliberately keeps automatic failure labels conservative. It identifies
where successful and failed episodes first diverge, whether the failed action was
admissible, and whether it looks like a repeat/stuck event. Semantic labels such as
Information/Plan/Action/Recovery Harness should be assigned after manual review.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASK_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.I)
ENV_STEP_RE = re.compile(r"You are now at step\s+(\d+)", re.I)
PRIOR_RE = re.compile(r"Prior to this step, you have already taken\s+(\d+)\s+step", re.I)
CUR_OBS_RE = re.compile(
    r"current observation is:\s*(.*?)(?:\nYour admissible actions of the current situation are:)",
    re.I | re.S,
)
ADM_RE = re.compile(
    r"Your admissible actions of the current situation are:\s*(.*?)\.(?:\n\nNow it's your turn|\nNow it's your turn)",
    re.I | re.S,
)
HIST_ACTION_RE = re.compile(r"Action\s+(\d+):\s*'([^']*)'", re.I)
HIST_OBS_RE = re.compile(r"Observation\s+(\d+):\s*'(.*?)'\s*,\s*Action\s+\1:", re.I | re.S)
ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
ACTION_OPEN_RE = re.compile(r"<action>\s*([^\n<]+)", re.I)


def extract_task(text: str) -> str | None:
    m = TASK_RE.search(text or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def extract_env_step(text: str) -> int | None:
    m = ENV_STEP_RE.search(text or "")
    return int(m.group(1)) if m else None


def extract_prior(text: str) -> int | None:
    m = PRIOR_RE.search(text or "")
    return int(m.group(1)) if m else None


def extract_current_observation(text: str) -> str | None:
    m = CUR_OBS_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_admissible(text: str) -> list[str]:
    m = ADM_RE.search(text or "")
    if not m:
        return []
    block = m.group(1)
    # ALFWorld prompt serializes a numpy-ish array of single-quoted strings.
    vals = re.findall(r"'([^']+)'", block)
    return [re.sub(r"\s+", " ", x).strip() for x in vals]


def extract_action(output: str) -> tuple[str | None, bool]:
    m = ACTION_TAG_RE.search(output or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(), True
    m = ACTION_OPEN_RE.search(output or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(), False
    return None, False


def extract_history(text: str) -> tuple[dict[int, str], dict[int, str]]:
    actions: dict[int, str] = {}
    observations: dict[int, str] = {}
    for m in HIST_ACTION_RE.finditer(text or ""):
        actions[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    for m in HIST_OBS_RE.finditer(text or ""):
        observations[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    return actions, observations


def score_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    # In this dump success is 10, success with one invalid penalty is 9.9,
    # failure is 0, and failure with one invalid penalty is -0.1.
    if score >= 9.0:
        return "success"
    if score <= 0.0:
        return "failure"
    return "other"


def lcp_actions(a: dict[int, str], b: dict[int, str]) -> tuple[int, int | None]:
    """Return number of shared action steps and first divergent action step."""
    common = sorted(set(a) & set(b))
    shared = 0
    for step in common:
        if a[step] == b[step]:
            shared += 1
            continue
        return shared, step
    all_steps = sorted(set(a) | set(b))
    for step in all_steps:
        if a.get(step) != b.get(step):
            return shared, step
    return shared, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    key_types: dict[str, Counter[str]] = defaultdict(Counter)
    with args.src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            for k, v in obj.items():
                key_types[k][type(v).__name__] += 1
            inp = str(obj.get("input", ""))
            out = str(obj.get("output", ""))
            score = obj.get("score")
            score = float(score) if isinstance(score, (int, float)) else None
            action, action_closed = extract_action(out)
            admissible = extract_admissible(inp)
            hist_actions, hist_obs = extract_history(inp)
            env_step = extract_env_step(inp)
            rows.append({
                "row": idx,
                "optimizer_step": obj.get("step"),
                "task": extract_task(inp),
                "score": score,
                "label": score_label(score),
                "env_step": env_step,
                "prior_steps": extract_prior(inp),
                "observation": extract_current_observation(inp),
                "admissible": admissible,
                "action": action,
                "action_tag_closed": action_closed,
                "action_admissible": (action in admissible) if action is not None and admissible else None,
                "history_actions": hist_actions,
                "history_observations": hist_obs,
                "input": inp,
                "output": out,
            })

    # The dump is flattened episode-by-episode. A new episode is detected when
    # the environment step counter resets/non-increases, task changes, or score changes.
    episodes: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for r in rows:
        new_ep = False
        if cur:
            p = cur[-1]
            if r["task"] != p["task"]:
                new_ep = True
            elif r["score"] != p["score"]:
                new_ep = True
            elif r["env_step"] is not None and p["env_step"] is not None and r["env_step"] <= p["env_step"]:
                new_ep = True
        if new_ep:
            episodes.append({"rows": cur})
            cur = []
        cur.append(r)
    if cur:
        episodes.append({"rows": cur})

    for eid, ep in enumerate(episodes):
        rr = ep["rows"]
        ep["episode_id"] = eid
        ep["task"] = rr[0]["task"]
        ep["score"] = rr[0]["score"]
        ep["label"] = score_label(ep["score"])
        ep["start_row"] = rr[0]["row"]
        ep["end_row"] = rr[-1]["row"]
        ep["first_env_step"] = rr[0]["env_step"]
        ep["last_env_step"] = rr[-1]["env_step"]

        actions: dict[int, str] = {}
        observations: dict[int, str] = {}
        admissible_by_step: dict[int, list[str]] = {}
        # Recover historical prefix from all row prompts (helps restore step 1).
        for r in rr:
            actions.update(r["history_actions"])
            observations.update(r["history_observations"])
            if r["env_step"] is not None:
                if r["action"] is not None:
                    actions[r["env_step"]] = r["action"]
                if r["observation"] is not None:
                    observations[r["env_step"]] = r["observation"]
                admissible_by_step[r["env_step"]] = r["admissible"]
        ep["actions"] = dict(sorted(actions.items()))
        ep["observations"] = dict(sorted(observations.items()))
        ep["admissible_by_step"] = dict(sorted(admissible_by_step.items()))
        ep["n_actions"] = len(actions)
        ep["invalid_output_steps"] = [
            r["env_step"] for r in rr
            if r["action_admissible"] is False
        ]
        ep["malformed_action_steps"] = [
            r["env_step"] for r in rr
            if r["action"] is None or not r["action_tag_closed"]
        ]

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_task[ep["task"] or "<unknown>"].append(ep)

    # For each task, select success/failure pair with the longest identical action prefix.
    pairs: list[dict[str, Any]] = []
    for task, ee in by_task.items():
        succ = [e for e in ee if e["label"] == "success"]
        fail = [e for e in ee if e["label"] == "failure"]
        candidates = []
        for s in succ:
            for f in fail:
                shared, div = lcp_actions(s["actions"], f["actions"])
                candidates.append((shared, div if div is not None else 10**9, s, f))
        if not candidates:
            continue
        shared, _, s, f = max(candidates, key=lambda x: (x[0], -x[1]))
        _, div = lcp_actions(s["actions"], f["actions"])
        fail_action = f["actions"].get(div) if div is not None else None
        succ_action = s["actions"].get(div) if div is not None else None
        fail_adm = f["admissible_by_step"].get(div, []) if div is not None else []
        fail_is_adm = (fail_action in fail_adm) if fail_action is not None and fail_adm else None
        prev_fail = [f["actions"].get(x) for x in range(max(1, (div or 1)-3), div or 1)]
        repeated = fail_action in prev_fail if fail_action is not None else False
        pairs.append({
            "task": task,
            "task_group_size": len(ee),
            "success_episode": s["episode_id"],
            "failure_episode": f["episode_id"],
            "success_score": s["score"],
            "failure_score": f["score"],
            "shared_action_prefix": shared,
            "divergence_step": div,
            "success_action": succ_action,
            "failure_action": fail_action,
            "success_observation": s["observations"].get(div) if div is not None else None,
            "failure_observation": f["observations"].get(div) if div is not None else None,
            "failure_action_admissible": fail_is_adm,
            "failure_action_repeats_recent": repeated,
            "success_actions": s["actions"],
            "failure_actions": f["actions"],
            "success_invalid_steps": s["invalid_output_steps"],
            "failure_invalid_steps": f["invalid_output_steps"],
        })
    pairs.sort(key=lambda x: (-x["shared_action_prefix"], x["task"]))

    scores = [e["score"] for e in episodes if e["score"] is not None]
    labels = Counter(e["label"] for e in episodes)
    lengths = [e["n_actions"] for e in episodes]
    group_sizes = Counter(len(v) for v in by_task.values())
    row_env_steps = Counter(r["env_step"] for r in rows if r["env_step"] is not None)

    summary = {
        "source": str(args.src),
        "row_schema": {k: dict(v) for k, v in sorted(key_types.items())},
        "rows": len(rows),
        "optimizer_steps_seen": sorted({r["optimizer_step"] for r in rows}),
        "episodes_reconstructed": len(episodes),
        "unique_tasks": len(by_task),
        "task_group_size_distribution": dict(sorted(group_sizes.items())),
        "episode_labels": dict(labels),
        "episode_score_distribution": dict(Counter(str(x) for x in scores)),
        "episode_action_length_min": min(lengths) if lengths else None,
        "episode_action_length_median": statistics.median(lengths) if lengths else None,
        "episode_action_length_mean": statistics.fmean(lengths) if lengths else None,
        "episode_action_length_max": max(lengths) if lengths else None,
        "rows_with_closed_action_tag": sum(r["action_tag_closed"] for r in rows),
        "rows_with_parsed_action": sum(r["action"] is not None for r in rows),
        "rows_with_admissibility_check": sum(r["action_admissible"] is not None for r in rows),
        "rows_with_non_admissible_action": sum(r["action_admissible"] is False for r in rows),
        "mixed_success_failure_tasks": sum(
            bool([e for e in ee if e["label"] == "success"]) and bool([e for e in ee if e["label"] == "failure"])
            for ee in by_task.values()
        ),
        "selected_divergence_pairs": len(pairs),
        "environment_step_row_distribution": {str(k): v for k, v in sorted(row_env_steps.items())},
    }
    (args.out / "episode_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.out / "episodes_index.jsonl").open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps({
                "episode_id": e["episode_id"],
                "task": e["task"],
                "score": e["score"],
                "label": e["label"],
                "start_row": e["start_row"],
                "end_row": e["end_row"],
                "first_env_step": e["first_env_step"],
                "last_env_step": e["last_env_step"],
                "n_actions": e["n_actions"],
                "invalid_output_steps": e["invalid_output_steps"],
                "malformed_action_steps": e["malformed_action_steps"],
                "actions": e["actions"],
            }, ensure_ascii=False) + "\n")

    with (args.out / "divergence_pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    md = [
        "# ALFWorld Qwen3-8B GiGPO — optimizer step 1 episode reconstruction",
        "",
        "> Important: `step` in the raw JSONL is the **optimizer step**. Each JSONL row is one environment-turn training sample, not one whole ALFWorld episode. Episodes below are reconstructed from the environment-step counter inside `input`.",
        "",
        "## Reconstructed batch",
        "",
        f"- raw rows: **{summary['rows']}**",
        f"- reconstructed episodes: **{summary['episodes_reconstructed']}**",
        f"- unique tasks: **{summary['unique_tasks']}**",
        f"- task group-size distribution: `{summary['task_group_size_distribution']}`",
        f"- episode outcomes: `{summary['episode_labels']}`",
        f"- episode score distribution: `{summary['episode_score_distribution']}`",
        f"- action length min/median/mean/max: **{summary['episode_action_length_min']} / {summary['episode_action_length_median']} / {summary['episode_action_length_mean']:.2f} / {summary['episode_action_length_max']}**",
        f"- parsed actions: **{summary['rows_with_parsed_action']} / {summary['rows']}**",
        f"- non-admissible emitted actions (where check possible): **{summary['rows_with_non_admissible_action']}**",
        f"- tasks containing both success and failure episodes: **{summary['mixed_success_failure_tasks']}**",
        "",
        "## Highest-leverage success/failure pairs",
        "",
        "For each task, this selects the success/failure pair sharing the longest identical action prefix. These are useful candidate intervention points because the trajectories stay behaviorally matched as long as possible before diverging.",
        "",
    ]

    for rank, p in enumerate(pairs, 1):
        md += [
            f"### {rank}. {p['task']}",
            "",
            f"- task group size: **{p['task_group_size']}**",
            f"- success episode / score: **{p['success_episode']} / {p['success_score']}**",
            f"- failure episode / score: **{p['failure_episode']} / {p['failure_score']}**",
            f"- shared action prefix: **{p['shared_action_prefix']} step(s)**",
            f"- first divergence: **step {p['divergence_step']}**",
            f"- success action: `{p['success_action']}`",
            f"- failure action: `{p['failure_action']}`",
            f"- failed action admissible: **{p['failure_action_admissible']}**",
            f"- failed action repeats a recent action: **{p['failure_action_repeats_recent']}**",
            "",
            "**Observation on success branch at divergence**",
            "```text",
            str(p['success_observation']),
            "```",
            "**Observation on failure branch at divergence**",
            "```text",
            str(p['failure_observation']),
            "```",
            "",
        ]
    (args.out / "divergence_pairs.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
