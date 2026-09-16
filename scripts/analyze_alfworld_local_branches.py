#!/usr/bin/env python3
"""Analyze context-matched local branches in flattened ALFWorld GiGPO dumps.

The public generation dump omits traj_uid/uid/anchor_obs, so exact trajectory
identity cannot be reconstructed losslessly. Instead, group rows by the exact
agent-visible local context (task, environment step, recent history, current
observation). Rows in one group are genuine matched decision situations from the
point of view of the model. Their sampled actions and rollout scores can therefore
be used to discover candidate intervention states.

Important: score differences are ASSOCIATIONAL, not a causal effect of the current
action, because later actions also affect the final outcome.
"""
from __future__ import annotations

import argparse, hashlib, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASK_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.I)
ENV_STEP_RE = re.compile(r"You are now at step\s+(\d+)", re.I)
CUR_OBS_RE = re.compile(r"current observation is:\s*(.*?)(?:\nYour admissible actions of the current situation are:)", re.I|re.S)
ADM_RE = re.compile(r"Your admissible actions of the current situation are:\s*(.*?)\.(?:\n\nNow it's your turn|\nNow it's your turn)", re.I|re.S)
HIST_PAIR_RE = re.compile(r"\[Observation\s+(\d+):\s*'(.*?)',\s*Action\s+\1:\s*'([^']*)'\]", re.I|re.S)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I|re.S)
ACTION_OPEN_RE = re.compile(r"<action>\s*([^\n<]+)", re.I)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.I|re.S)

def norm(s: str|None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def task(inp: str) -> str|None:
    m=TASK_RE.search(inp); return norm(m.group(1)) if m else None

def env_step(inp: str) -> int|None:
    m=ENV_STEP_RE.search(inp); return int(m.group(1)) if m else None

def obs(inp: str) -> str|None:
    m=CUR_OBS_RE.search(inp); return norm(m.group(1)) if m else None

def admissible(inp: str) -> list[str]:
    m=ADM_RE.search(inp)
    return [norm(x) for x in re.findall(r"'([^']+)'", m.group(1))] if m else []

def history(inp: str) -> list[tuple[int,str,str]]:
    return [(int(m.group(1)), norm(m.group(2)), norm(m.group(3))) for m in HIST_PAIR_RE.finditer(inp)]

def action(out: str) -> tuple[str|None,bool]:
    m=ACTION_RE.search(out)
    if m: return norm(m.group(1)), True
    m=ACTION_OPEN_RE.search(out)
    if m: return norm(m.group(1)), False
    return None, False

def think_excerpt(out: str, n=600) -> str:
    m=THINK_RE.search(out)
    s=norm(m.group(1)) if m else norm(out)
    return s[:n] + ('…' if len(s)>n else '')

def score_class(x: float|None) -> str:
    if x is None: return 'unknown'
    if x >= 9.0: return 'high'
    if x <= 0.0: return 'low'
    return 'mid'

def sig_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('src',type=Path); ap.add_argument('out',type=Path); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with a.src.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            x=json.loads(line); inp=str(x.get('input','')); out=str(x.get('output',''))
            sc=float(x['score']) if isinstance(x.get('score'),(int,float)) else None
            ac,closed=action(out); adm=admissible(inp); hist=history(inp)
            cur_obs=obs(inp); st=env_step(inp); ta=task(inp)
            state_key=(ta,st,cur_obs)
            # Agent-visible context in this prompt: exact recent obs/action history + current obs.
            context_key=(ta,st,tuple(hist),cur_obs)
            rows.append({
                'row':i,'task':ta,'env_step':st,'observation':cur_obs,'history':hist,
                'action':ac,'action_closed':closed,'admissible':adm,
                'action_admissible': (ac in adm) if ac is not None and adm else None,
                'score':sc,'score_class':score_class(sc),'output':out,
                'state_sig':sig_hash(state_key),'context_sig':sig_hash(context_key),
            })

    def summarize_groups(key_name: str):
        groups=defaultdict(list)
        for r in rows: groups[r[key_name]].append(r)
        out=[]
        for gid, rr in groups.items():
            scores=[r['score'] for r in rr if r['score'] is not None]
            acts=defaultdict(list)
            for r in rr: acts[r['action']].append(r)
            cls=Counter(r['score_class'] for r in rr)
            action_stats=[]
            for ac, aa in acts.items():
                ss=[r['score'] for r in aa if r['score'] is not None]
                action_stats.append({
                    'action':ac,'n':len(aa),'mean_score':statistics.fmean(ss) if ss else None,
                    'high':sum(r['score_class']=='high' for r in aa),
                    'low':sum(r['score_class']=='low' for r in aa),
                    'admissible_all': all(r['action_admissible'] is not False for r in aa),
                    'closed_tag_all': all(r['action_closed'] for r in aa),
                    'rows':[r['row'] for r in aa[:8]],
                })
            action_stats.sort(key=lambda z:(-(z['mean_score'] if z['mean_score'] is not None else -999),-z['n'],str(z['action'])))
            out.append({
                'sig':gid,'n':len(rr),'task':rr[0]['task'],'env_step':rr[0]['env_step'],
                'observation':rr[0]['observation'],'history':rr[0]['history'],
                'unique_actions':len(acts),'score_classes':dict(cls),
                'score_min':min(scores) if scores else None,'score_max':max(scores) if scores else None,
                'score_mean':statistics.fmean(scores) if scores else None,
                'mixed_high_low': cls['high']>0 and cls['low']>0,
                'action_stats':action_stats,
            })
        return groups,out

    state_groups,state_summ=summarize_groups('state_sig')
    ctx_groups,ctx_summ=summarize_groups('context_sig')

    candidates=[]
    for g in ctx_summ:
        if not g['mixed_high_low'] or g['unique_actions']<2: continue
        best_high=None; best_low=None
        for ast in g['action_stats']:
            if ast['high']>0:
                if best_high is None or (ast['high']/ast['n'], ast['mean_score'] or -999, ast['n']) > (best_high['high']/best_high['n'], best_high['mean_score'] or -999,best_high['n']):
                    best_high=ast
            if ast['low']>0:
                if best_low is None or (ast['low']/ast['n'], -(ast['mean_score'] or 999), ast['n']) > (best_low['low']/best_low['n'], -(best_low['mean_score'] or 999), best_low['n']):
                    best_low=ast
        if not best_high or not best_low or best_high['action']==best_low['action']: continue
        high_rows=[r for r in ctx_groups[g['sig']] if r['action']==best_high['action'] and r['score_class']=='high']
        low_rows=[r for r in ctx_groups[g['sig']] if r['action']==best_low['action'] and r['score_class']=='low']
        if not high_rows or not low_rows: continue
        h=high_rows[0]; l=low_rows[0]
        prev_actions=[x[2] for x in g['history']]
        candidates.append({
            'context_sig':g['sig'],'task':g['task'],'env_step':g['env_step'],'n':g['n'],
            'unique_actions':g['unique_actions'],'observation':g['observation'],'history':g['history'],
            'high_action':h['action'],'high_score':h['score'],'high_row':h['row'],'high_action_admissible':h['action_admissible'],
            'low_action':l['action'],'low_score':l['score'],'low_row':l['row'],'low_action_admissible':l['action_admissible'],
            'low_action_repeats_recent': l['action'] in prev_actions if l['action'] else False,
            'high_think_excerpt':think_excerpt(h['output']),'low_think_excerpt':think_excerpt(l['output']),
            'action_stats':g['action_stats'],
        })
    candidates.sort(key=lambda z:(-len(z['history']),-z['env_step'] if z['env_step'] else 0,-z['n'],z['task'] or ''))

    # Same action + mixed outcome is a useful negative control: the current action alone cannot explain the outcome.
    same_action_mixed=[]
    for g in ctx_summ:
        for ast in g['action_stats']:
            if ast['high']>0 and ast['low']>0:
                same_action_mixed.append({
                    'context_sig':g['sig'],'task':g['task'],'env_step':g['env_step'],'n_context':g['n'],
                    'action':ast['action'],'n_action':ast['n'],'high':ast['high'],'low':ast['low'],
                    'observation':g['observation'],'history':g['history'],
                })
    same_action_mixed.sort(key=lambda z:(-z['n_action'],-z['env_step'] if z['env_step'] else 0))

    summary={
        'rows':len(rows),'tasks':len(set(r['task'] for r in rows)),
        'state_groups':len(state_groups),'context_groups':len(ctx_groups),
        'context_groups_n_ge_2':sum(g['n']>=2 for g in ctx_summ),
        'context_groups_multiple_actions':sum(g['unique_actions']>=2 for g in ctx_summ),
        'context_groups_mixed_high_low':sum(g['mixed_high_low'] for g in ctx_summ),
        'context_groups_mixed_high_low_multiple_actions':sum(g['mixed_high_low'] and g['unique_actions']>=2 for g in ctx_summ),
        'candidate_branch_points':len(candidates),
        'same_action_mixed_outcome_controls':len(same_action_mixed),
        'parsed_actions':sum(r['action'] is not None for r in rows),
        'closed_action_tags':sum(r['action_closed'] for r in rows),
        'non_admissible_parsed_actions':sum(r['action_admissible'] is False for r in rows),
        'score_distribution':dict(Counter(str(r['score']) for r in rows)),
        'env_step_distribution':dict(Counter(str(r['env_step']) for r in rows)),
    }
    (a.out/'local_branch_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.out/'local_branch_candidates.jsonl').open('w',encoding='utf-8') as f:
        for c in candidates: f.write(json.dumps(c,ensure_ascii=False)+'\n')
    with (a.out/'same_action_mixed_controls.jsonl').open('w',encoding='utf-8') as f:
        for c in same_action_mixed[:200]: f.write(json.dumps(c,ensure_ascii=False)+'\n')

    md=['# Context-matched local branch analysis — Qwen3-8B GiGPO, optimizer step 1','',
        '> This is the reliable analysis surface for the public dump. Exact `traj_uid` is not included in the JSONL, so we do **not** claim to reconstruct trajectory identities. We compare rows only when the model sees the same task, same ALFWorld step, same recent history, and same current observation.','',
        '> A high-vs-low score difference at one state is **not yet causal credit for the current action**: later decisions remain a confounder. These are candidate intervention states for replay/branch experiments.','',
        '## Summary','']
    for k,v in summary.items():
        if k not in ('env_step_distribution','score_distribution'): md.append(f'- {k}: **{v}**')
    md += [f"- score_distribution: `{summary['score_distribution']}`",'',
           '## Candidate matched branch points','']
    for i,c in enumerate(candidates[:30],1):
        md += [f"### {i}. {c['task']} — env step {c['env_step']}",'',
               f"- matched-context samples: **{c['n']}**; unique sampled actions: **{c['unique_actions']}**",
               f"- HIGH branch: row `{c['high_row']}`, score `{c['high_score']}`, action `{c['high_action']}`, admissible `{c['high_action_admissible']}`",
               f"- LOW branch: row `{c['low_row']}`, score `{c['low_score']}`, action `{c['low_action']}`, admissible `{c['low_action_admissible']}`",
               f"- low action repeats recent history: **{c['low_action_repeats_recent']}**",'',
               '**Recent history**','```text']
        for st,o,ac in c['history']:
            md.append(f"step {st}: OBS={o}\n        ACT={ac}")
        md += ['```','**Current observation**','```text',str(c['observation']),'```',
               '**HIGH reasoning excerpt**','```text',c['high_think_excerpt'],'```',
               '**LOW reasoning excerpt**','```text',c['low_think_excerpt'],'```','']
    if not candidates:
        md.append('No exact-context mixed-outcome/different-action branch points were found. Use state-level grouping or a later optimizer step, or regenerate rollouts with `traj_uid/uid/anchor_obs` preserved.')
    md += ['','## Negative controls: same context + same action + different eventual score','',
           'These cases show why `score(high)-score(low)` cannot be naively attributed to the current action: identical local actions can still lead to different eventual outcomes because later actions differ.','']
    for c in same_action_mixed[:20]:
        md.append(f"- `{c['task']}` step {c['env_step']}: action `{c['action']}` appears {c['n_action']} times with high={c['high']}, low={c['low']} under the same visible context.")
    (a.out/'local_branch_report.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
