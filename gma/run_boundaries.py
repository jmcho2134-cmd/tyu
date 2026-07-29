#!/usr/bin/env python
"""
run_boundaries.py — Stage 4 실행기: cache -> artifacts/boundaries.json
================================================================================

    cache/demo_XXX.npz  --segment_hier-->  artifacts/boundaries.json
                                           artifacts/boundaries_diag.json

boundaries.json (PIPELINE_v4.md Sec 3.3) — Stage 5 subgoal_discover 의 입력:

    { "demo_000": { "T": 471,
                    "bounds": [151, 179, 330, 377],
                    "labels": ["approach","grasp","transport","place","retreat"],
                    "subgoal_per_phase": [["eef_object_dist", 1.0], ...] } }

GATE G3 — 라벨 시퀀스 일치율 >= 70%
------------------------------------
모든 데모의 라벨 시퀀스 중 최빈 시퀀스에 일치하는 비율. 이 게이트를 통과하기
전에는 Stage 5 이후의 결과가 의미 없다 (문서 Sec 5: "G3가 지금 서 있는 자리").
실패 시 L1 파라미터(임계/디바운스) 조정이 처방이다.

    python run_boundaries.py                       # ./cache -> ./artifacts
    python run_boundaries.py --plot                # + phase overlay png
    python run_boundaries.py --baseline            # flat segmenter 와 비교 출력
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

import feature_select as fs
from extract_features import load_cache
from segment_hier import segment_hierarchical, segment_object_centric

GATE_G3 = 0.70

# 기본은 객체 중심 pre|move|post. 경계 2개가 모두 물리 이벤트(객체 이동
# 시작/정착)라 안정적이고, 계층 모드는 ablation 비교용으로 남긴다.
SEGMENTERS = dict(object=segment_object_centric, hier=segment_hierarchical)


def run(cache_dir="./cache", out_dir="./artifacts", *, enforce_gate=True,
        plot=False, baseline=False, mode="object"):
    entries = load_cache(cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    segmenter = SEGMENTERS[mode]
    print(f"[mode] {mode} ({segmenter.__name__})")

    boundaries, diags, seqs = {}, {}, []
    for e in entries:
        seg = segmenter(e["F"], e["names"], dt=e["dt"])
        seqs.append(" | ".join(seg["labels"]))
        boundaries[e["demo_id"]] = dict(
            T=int(len(e["F"])),
            bounds=[int(b) for b in seg["bounds"]],
            labels=seg["labels"],
            subgoal_per_phase=[[n, float(s)] for n, s in seg["subgoal_per_phase"]],
        )
        diags[e["demo_id"]] = seg["diag"]
        print(f"{e['demo_id']}: T={len(e['F']):>4}  bounds={seg['bounds']}  "
              f"{' | '.join(seg['labels'])}")
        if plot:
            _plot(e, seg, out_dir)
            _plot3d(e, seg, out_dir, mode)
        if baseline:
            from phase_segment import segment_features
            flat = segment_features(e["F"], e["names"])
            print(f"          [baseline flat] {len(flat['bounds']) + 1} phases  "
                  f"bounds={flat['bounds']}")

    # ---- GATE G3 --------------------------------------------------------
    tally = Counter(seqs)
    modal, n_modal = tally.most_common(1)[0]
    agree = n_modal / len(seqs)
    meta = dict(n_demos=len(entries),
                sequence_tally={k: int(v) for k, v in tally.items()},
                modal_sequence=modal, agreement=round(agree, 3),
                gate_g3=GATE_G3, gate_pass=bool(agree >= GATE_G3))

    with open(os.path.join(out_dir, "boundaries.json"), "w") as f:
        json.dump(boundaries, f, indent=2)
    with open(os.path.join(out_dir, "boundaries_diag.json"), "w") as f:
        json.dump(dict(meta=meta, demos=diags), f, indent=2)

    print(f"\n[G3] modal sequence: {modal}")
    for s, c in tally.most_common():
        print(f"     {c}/{len(seqs)}  {s}")
    if agree >= GATE_G3:
        print(f"[G3] PASS: agreement {agree:.0%} >= {GATE_G3:.0%}")
    else:
        msg = f"[G3] FAIL: agreement {agree:.0%} < {GATE_G3:.0%} — L1 파라미터 조정 필요"
        if enforce_gate:
            raise SystemExit(msg)
        print(msg)
    print(f"[out] {os.path.join(out_dir, 'boundaries.json')} (+ _diag.json)")
    return boundaries, meta


def _plot(entry, seg, out_dir):
    """phase overlay: Ψ 3채널 + contact 위에 경계선. 사람 검증용."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib unavailable; skipping plot")
        return
    F, names = entry["F"], entry["names"]
    T = len(F)
    chans = ["eef_object_dist", "object_goal_dist", "gripper_open", "contact"]
    fig, axes = plt.subplots(len(chans), 1, figsize=(10, 7), sharex=True)
    edges = [0] + list(seg["bounds"]) + [T]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, ch in zip(axes, chans):
        y = F[:, names.index(ch)]
        ax.plot(y, lw=1.2, color="k")
        for s, lab in enumerate(seg["labels"]):
            ax.axvspan(edges[s], edges[s + 1], alpha=0.15, color=colors[s % 10])
        for b in seg["bounds"]:
            ax.axvline(b, color="r", lw=0.8, ls="--")
        ax.set_ylabel(ch, fontsize=8)
    mids = [(edges[s] + edges[s + 1]) / 2 for s in range(len(seg["labels"]))]
    for m, lab in zip(mids, seg["labels"]):
        axes[0].text(m, axes[0].get_ylim()[1], lab, ha="center", va="bottom",
                     fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"{entry['demo_id']}  bounds={seg['bounds']}")
    fig.tight_layout()
    p = os.path.join(out_dir, f"boundaries_{entry['demo_id']}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"          [plot] {p}")


def _plot3d(entry, seg, out_dir, mode="object"):
    """phase 색으로 칠한 3D eef 궤적 (phase_segment.visualize 재사용).
    캐시에 raw 좌표(eef_pos/obj_pos)가 있어야 한다 — 없으면 재추출 안내."""
    if entry.get("eef_pos") is None or entry.get("obj_pos") is None:
        print(f"          [plot3d] {entry['demo_id']}: cache has no eef_pos/"
              f"obj_pos — re-run extract_features.py to enable 3D plots")
        return
    from phase_segment import visualize
    p = os.path.join(out_dir, f"traj3d_{entry['demo_id']}.png")
    visualize(entry["eef_pos"], entry["obj_pos"], seg, entry["F"],
              entry["names"], entry["goal"], entry["dt"], p,
              title=f"{entry['demo_id']} ({mode})")
    print(f"          [plot3d] {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--out-dir", default="./artifacts")
    ap.add_argument("--no-gate", action="store_true",
                    help="report G3 but do not abort on failure")
    ap.add_argument("--plot", action="store_true",
                    help="write a phase-overlay png per demo (human check)")
    ap.add_argument("--baseline", action="store_true",
                    help="also print the flat segmenter's cut for comparison")
    ap.add_argument("--mode", default="object", choices=sorted(SEGMENTERS),
                    help="object = pre|move|post (기본), hier = 5-phase 계층")
    args = ap.parse_args()
    run(args.cache_dir, args.out_dir, enforce_gate=not args.no_gate,
        plot=args.plot, baseline=args.baseline, mode=args.mode)


if __name__ == "__main__":
    main()
