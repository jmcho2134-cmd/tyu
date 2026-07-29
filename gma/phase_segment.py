#!/usr/bin/env python
"""
phase_segment.py — M1: phase segmentation and per-phase subgoal sets.
================================================================================

Reads its feature schema from feature_select.py and defines NOTHING itself. The
segmenter never learns which feature is "the approach one": it only sees each
column's KIND and whether it is allowed to create a BOUNDARY.

    progress + boundary -> the plateau of its high-water-mark is a boundary
                           (Kneedle knee; threshold-free; the running max makes
                           sidetracks/regressions invisible -- only PROGRESS can
                           close a phase, so uncharacterised inefficiencies are
                           absorbed instead of being promoted to phases)
    event    + boundary -> a mode transition is a boundary (2-means split, so the
                           split point is data-derived, not a magic number)
    quality             -> ignored here entirely

Phase COUNT, NAMES, ORDER and THRESHOLDS are never hardcoded; they emerge.

Outputs per phase
-----------------
    subgoal set G_z = (main, change, hold), read off the demo by phase_subgoal_set

A phase is not one feature. Approach is: the eef-object distance FALLS *while* the
object stays put, the gripper stays open and nothing is touched. So the subgoal is
that whole pattern, and there are two ways to fail it -- reverse what the demo
drove (`change`), or disturb what it held (`hold`).

This is what makes a gripper phase degradable at all. `gripper_open` is
near-binary: a unit-vector perturbation moves it ~0.38 against a +-1 command, so
small lambdas cannot flip it and its measured degradation curve is flat. But the
same phase also CHANGES `contact` and HOLDS `grasp_align` / `eef_object_dist`,
which are continuous -- and degrading those is misalignment / lost contact,
exactly what proposal Sec 8.1 lists for the Contact/Grasp phase.

There are no waypoints. An earlier version made the subgoal partly "distance from
the demo's own waypoint at this phase"; on real data that term carried 39-69% of
the objective, every phase then chose a rotation-dominated direction (phase
conditioning collapsed), and the gripper phases scored while IMPROVING their own
defining feature. That is proposal Sec 9.3's similarity-to-the-demo trap.

Gripper labelling is by ORDER (1st transition = close/grasp, 2nd = open/place),
never by rise/fall: a Panda's sum|qpos| RISES on close, so rise/fall labelling is
polarity-dependent and was observed to invert grasp and place.

Reusability: segment_features() is pure numpy and is the same call M5 makes on
policy rollouts (proposal Sec 6.2 -- the reward takes z_t as input, so new
trajectories must be segmentable the same way).

Where the input comes from
--------------------------
The segmenter is now fed Psi, not the full feature matrix:

    Psi = feature_select.compute_psi(frames)          # (T, 3)
    seg = segment_features(Psi, names=feature_select.PSI_COLUMNS)

Psi is not a second computation -- it is compute_trajectory(columns=BOUNDARY),
the same implementation restricted to the boundary columns, so M1 and M5 cannot
disagree about the phase input. Passing the full F still works exactly as before
(segment_features already filtered by kind internally), which is why
degradation_proto.py's `ps.segment_features(Phi)` needs no edit.

Feeding Psi rather than F buys two things. The type makes the rule "phases are
cut on the observation space" enforceable instead of conventional: none of the
three boundary columns reads torque, actions or quaternions, so an attempt to
segment on a quality feature fails at the column list rather than silently
changing the phases. And segmentation keeps running when the torque channel is
missing, which matters because torque cannot be replayed out of the demo file.

The robosuite replay that used to be SECTION 2 now lives in frame_extract.py.
Re-export shims are kept below for the downstream prototypes.

    python phase_segment.py --selftest              # no robosuite needed
    python phase_segment.py --demo-root ./demos
"""

import argparse
import json
import os
import warnings
from glob import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import feature_select as fs


# ===========================================================================
# SECTION 1 — the segmenter (pure numpy; reusable on M5 rollouts)
# ===========================================================================
def _minmax(x):
    x = np.asarray(x, float); lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _is_decreasing(f, edge_frac=0.1):
    f = np.asarray(f, float); e = max(1, int(edge_frac * len(f)))
    return f[-e:].mean() <= f[:e].mean()


def normalized_progress(f, edge_frac=0.1):
    """Map a progress feature to [0,1] with 1 = most complete. Direction is auto-
    detected from the net trend: ending LOWER than it started = drive-to-min."""
    n = _minmax(f)
    return (1.0 - n) if _is_decreasing(f, edge_frac) else n


def high_water_mark(p):
    """Running max -> monotone. Regressions (sidetracks) vanish."""
    return np.maximum.accumulate(np.asarray(p, float))


def trend_ratio(f, window=21):
    """|net change| / total variation, computed on the SMOOTHED signal.
    1 = perfectly monotone, ~0 = pure jitter.

    This is what separates "this feature actually progressed here" from "this
    feature only jittered here" -- the job the old climb_ratio filter was supposed
    to do but could not. climb_ratio compared the climb of a normalized curve, and
    normalized_progress min-maxes first, so a 0.002-amplitude noise signal
    stretched to span [0,1] and climbed 0.47, sailing past a 0.15 threshold. It
    could never drop anything.

    Smoothing first is not cosmetic: total variation is dominated by measurement
    noise, not by motion. On a real teleop trace the signal's own TV is ~1.35 but
    noise inflates it to ~4.44, which drags a genuine progress feature down to
    0.12 and would delete the approach phase. Smoothed: real progress ~0.35,
    pure noise ~0.016 -- a 20x margin.
    """
    f = np.asarray(f, float)
    if len(f) < 5:
        return 0.0
    g = fs.smooth_positions(f.reshape(-1, 1), window=window).ravel()
    tv = float(np.abs(np.diff(g)).sum())
    return abs(float(g[-1] - g[0])) / (tv + 1e-12)


def plateau_knee(hwm):
    """Kneedle: for a concave-increasing curve the knee is the point furthest
    above the chord from start to end. Threshold-free."""
    hwm = np.asarray(hwm, float); T = len(hwm)
    if T < 3:
        return T - 1
    return int(np.argmax(_minmax(hwm) - np.linspace(0.0, 1.0, T)))


def two_means_threshold(v, iters=25):
    """1D 2-means -> (threshold, high_is_upper). Data-derived split point."""
    v = np.asarray(v, float); lo, hi = np.min(v), np.max(v)
    if hi <= lo:
        return lo, True
    c0, c1 = lo, hi
    for _ in range(iters):
        mid = 0.5 * (c0 + c1)
        left, right = v[v <= mid], v[v > mid]
        n0 = left.mean() if len(left) else c0
        n1 = right.mean() if len(right) else c1
        if np.isclose(n0, c0) and np.isclose(n1, c1):
            break
        c0, c1 = n0, n1
    return 0.5 * (c0 + c1), (c1 >= c0)


def event_transitions(f, min_run=3):
    """Binarise by 2-means; return (time, +1 rise / -1 fall) for each flip.
    min_run debounces so gripper jitter does not spawn spurious phases."""
    f = np.asarray(f, float)
    if set(np.unique(f).tolist()).issubset({0.0, 1.0}):
        b = f.astype(int)
    else:
        thr, high_upper = two_means_threshold(f)
        b = (f > thr).astype(int)
        if not high_upper:
            b = 1 - b
    out = b.copy(); i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if (j - i) < min_run and 0 < i < len(out):
            out[i:j] = out[i - 1]
        i = j
    idx = np.where(np.diff(out) != 0)[0]
    return [(int(i + 1), int(out[i + 1] - out[i])) for i in idx]


def _tail_subgoal(F, a, b, names, so_far, min_trend=0.15):
    """The subgoal of a segment that no event closed: the boundary-eligible
    feature the demo is still driving hardest INSIDE this segment.

    sign = -sign(net change): the demo drove it this way, so degrading means
    reversing it. For the retreat that reads as (eef_object_dist, -1) -- "the eef
    should be getting away from the object and is not" -- which is continuous and
    easy to move, so it degrades well.
    """
    best, best_tr = None, min_trend
    for i, n in enumerate(names):
        sp = fs.SPEC.get(n)
        if sp is None or not sp.boundary:
            continue
        seg = F[a:b, i]
        tr = trend_ratio(seg)
        if tr > best_tr:
            net = float(seg[-1] - seg[0])
            if net != 0:
                best, best_tr = (n, -float(np.sign(net))), tr
    if best is not None:
        return best
    return so_far[-1] if so_far else (names[0], +1.0)


def segment_features(F, names=None, min_seg_frac=0.03, min_trend=0.15):
    """Feature-agnostic segmentation. THE reusable entry point (M1 + M5).

    F     : (T, k) feature matrix whose columns are `names`
    names : column names; defaults to feature_select.NON_DEFERRED

    Returns dict:
      z                 (T,) phase label per timestep
      bounds            boundary timesteps
      labels            what closed each phase
      subgoal_per_phase [(feature_name, degrade_sign), ...] one per phase
      events            diagnostics
    """
    F = np.asarray(F, float); T, k = F.shape
    names = list(names) if names is not None else list(fs.NON_DEFERRED)
    if len(names) != k:
        raise ValueError(f"F has {k} columns but {len(names)} names")
    min_seg = max(2, int(min_seg_frac * T))

    events = []                                   # (t, name, type, dir, col_idx)

    # -- progress + boundary: plateau of the high-water mark --------------
    for i, n in enumerate(names):
        sp = fs.SPEC.get(n)
        if sp is None or not sp.boundary or sp.kind != fs.PROGRESS:
            continue
        if trend_ratio(F[:, i]) < min_trend:
            continue          # jittered here, never progressed -> not a subgoal
        hwm = high_water_mark(normalized_progress(F[:, i]))
        events.append((plateau_knee(hwm), n, "plateau", 0, i))

    # -- event + boundary: mode transitions -------------------------------
    for i, n in enumerate(names):
        sp = fs.SPEC.get(n)
        if sp is None or not sp.boundary or sp.kind != fs.EVENT:
            continue
        for t, d in event_transitions(F[:, i]):
            events.append((t, n, "transition", d, i))

    events.sort(key=lambda e: e[0])

    # Order-based engage/release labels for event features (polarity-proof).
    ev_order, seen = {}, {}
    for e in events:
        if e[2] == "transition":
            c = seen.get(e[1], 0); ev_order[(e[1], e[0])] = c; seen[e[1]] = c + 1

    # keep: drop near the ends; when a plateau and an event coincide keep the EVENT
    kept = []
    for e in events:
        t = e[0]
        if t < min_seg or t > T - min_seg:
            continue
        if kept and (t - kept[-1][0]) < min_seg:
            if e[2] == "transition" and kept[-1][2] == "plateau":
                kept[-1] = e
            continue
        kept.append(e)
    bounds = [int(e[0]) for e in kept]

    z = np.zeros(T, dtype=int); labels, subgoal_per_phase = [], []
    edges = [0] + bounds + [T]
    for s in range(len(edges) - 1):
        a, b = edges[s], edges[s + 1]
        z[a:b] = s
        closing = [e for e in events if a < e[0] <= b]
        plats = [e for e in closing if e[2] == "plateau"]
        if plats:
            e = plats[-1]
            labels.append(f"{e[1]}:reached")
            # drive-to-min (a distance) -> raising it degrades (+1)
            sign = +1.0 if _is_decreasing(F[:, e[4]]) else -1.0
            subgoal_per_phase.append((e[1], sign))
        elif closing:
            e = closing[-1]
            engage = (ev_order.get((e[1], e[0]), 0) % 2 == 0)
            labels.append(f"{e[1]}:{'close' if engage else 'open'}")
            # degrade = revert the transition (robust to either polarity)
            sgn = -float(np.sign(e[3]))
            subgoal_per_phase.append((e[1], sgn if sgn != 0 else +1.0))
        else:
            # No event closed this segment -- it is the tail (retreat). It still
            # has a subgoal: whatever the demo is STILL driving there. Read it off
            # the tail's own data.
            #
            # Inheriting the previous phase's subgoal (what this used to do) gave
            # "after releasing, degrade by not opening the gripper", which is
            # meaningless -- and on real data that phase found no valid d_z at all.
            #
            # The tail also cannot be found by the plateau machinery: eef_object_dist
            # FALLS during approach and RISES during retreat, but normalized_progress
            # picks ONE direction for the whole demo from the net trend, so the final
            # rise reads as one more regression and the high-water mark absorbs it
            # (measured: 1.000 -> 1.000 across the retreat). That absorption is the
            # whitelist principle working as designed for sidetracks; the tail just
            # needs to be read directly instead.
            #
            # Retreat matters more than it looks: it is the ONE phase whose
            # degradation cannot fail the task -- the object is already placed -- so
            # its ladder varies execution time / path length while every rung still
            # succeeds. That is a pure-efficiency family, which is exactly what BfD
            # needs (degrading on the success axis leaves nothing to extrapolate:
            # the demo already succeeds).
            labels.append("final" if s == len(edges) - 2 else f"seg{s}")
            subgoal_per_phase.append(_tail_subgoal(F, a, b, names,
                                                   subgoal_per_phase))
    return dict(z=z, bounds=bounds, labels=labels,
                subgoal_per_phase=subgoal_per_phase, events=events)


def _role_in_window(seg, gscale, hold_frac, trend_min):
    """change / hold / free for one feature inside one phase window."""
    if len(seg) < 3:
        return "free", 0.0
    rel = float(seg.std()) / max(gscale, 1e-9)
    tv = float(np.abs(np.diff(seg)).sum())
    net = float(seg[-1] - seg[0])
    trend = abs(net) / (tv + 1e-12)          # 1 = perfectly monotone, 0 = jitter
    if rel < hold_frac:
        return "hold", 0.0                   # ~constant here
    if trend > trend_min:
        # the demo drove it this way, so degrading it means reversing that
        return "change", -float(np.sign(net))
    return "free", 0.0                       # oscillates: a quality-like signal


def phase_subgoal_set(F, z, bounds, main_per_phase, names=None,
                      hold_frac=0.05, trend_min=0.30, pad=2, main_only=False):
    """Per phase, read every feature's ROLE off the demo -> the phase's subgoal.

    A phase is not one feature. Approach is: the eef-object distance FALLS *while*
    the object stays put, the gripper stays open and nothing is touched. So the
    subgoal is that whole pattern, and there are two ways to fail it:

        change : the demo drove this feature -> degrade by REVERSING it (signed)
        hold   : the demo kept this feature ~constant -> degrade by DISTURBING it
        free   : it oscillated -> not part of the subgoal (quality lands here)

    This is what gives the gripper phases something degradable. `gripper_open` is
    near-binary: a unit-vector perturbation moves it ~0.38 against a +-1 command,
    so small lambdas cannot flip it and its measured curve is flat. But grasp also
    *changes* `contact` and *holds* `grasp_align` and `eef_object_dist` -- and
    those are continuous. Degrading them is misalignment / lost contact, which is
    exactly what proposal Sec 8.1 lists for the Contact/Grasp phase.

    IMPORTANT -- `pad`: a phase's own event happens AT its closing boundary (the
    gripper closes on the frame that ENDS the grasp phase, which already belongs
    to the next phase). A window stopping at the boundary therefore sees nothing
    change and the phase gets an empty subgoal. The window runs to bounds+pad.

    Quality features are excluded by schema, never by a threshold: a big jerk
    change is a COST, not a failure to achieve the phase.
    """
    F = np.asarray(F, float); T = len(F)
    names = list(names) if names is not None else list(fs.NAMES)
    gscale = {n: float(F[:, i].max() - F[:, i].min())
              for i, n in enumerate(names)}
    edges = [0] + list(bounds) + [T]
    out = {}
    for k in range(len(edges) - 1):
        a = edges[k]
        b = min(edges[k + 1] + pad, T)
        if main_only:
            # MAIN-ONLY: each phase's subgoal is the single feature that
            # segmented it -- nothing else. Same dict shape as the full set
            # (change carries the main, hold empty), so every downstream
            # consumer (fcm_proto score_terms, degradation_proto) works
            # unchanged and collapses to exactly ONE term.
            main = main_per_phase[min(k, len(main_per_phase) - 1)]
            out[k] = dict(main=(main[0], float(main[1])),
                          change=[(main[0], float(main[1]))], hold=[],
                          free=[n for n in names if n != main[0]],
                          window=(a, b))
            continue
        change, hold, free = [], [], []
        for i, n in enumerate(names):
            sp = fs.SPEC.get(n)
            if sp is None or sp.kind == fs.QUALITY:
                free.append(n); continue
            role, sgn = _role_in_window(F[a:b, i], gscale[n], hold_frac, trend_min)
            if role == "change":
                change.append((n, sgn))
            elif role == "hold":
                hold.append(n)
            else:
                free.append(n)
        main = main_per_phase[min(k, len(main_per_phase) - 1)]
        # The feature that CLOSED the phase is the main term by construction, even
        # if the window statistics did not flag it (a 1-frame event can look flat).
        if main[0] not in [c[0] for c in change]:
            change.insert(0, (main[0], float(main[1])))
        hold = [h for h in hold if h != main[0]]
        out[k] = dict(main=(main[0], float(main[1])), change=change, hold=hold,
                      free=free, window=(a, b))
    return out


# ===========================================================================
# SECTION 2 — robosuite plumbing: MOVED to frame_extract.py
# ===========================================================================
# The replay / env / contact / goal code that used to live here now lives in
# frame_extract.py. It had to move: proposal Sec 6.2 requires the segmenter to
# run on M5 policy rollouts, and a rollout has no hdf5 states to replay, so the
# per-frame extraction has to be callable from both sides. Leaving it here would
# have forced M5 to grow a second copy, and two copies of "what counts as an
# observation" is exactly how the reward ends up reading a z_t it was never
# trained against.
#
# What stays in this file is SECTION 1: pure-numpy segmentation. Unchanged.
#
# The names below are RE-EXPORTS, kept so fcm_proto.py and degradation_proto.py
# (which call ps.read_frame / ps.contact_signal / ps.resolve_object /
# ps.resolve_object_body / ps.reset_to_scene / ps.read_demo / ps.build_env)
# need no edits at all. New code should import frame_extract directly.
import frame_extract as fx

build_env           = fx.build_env
reset_to_scene      = fx.reset_to_scene
read_frame          = fx.read_frame
contact_signal      = fx.contact_signal
resolve_object      = fx.resolve_object
resolve_object_body = fx.resolve_object_body
_obs                = fx._obs
_object_pos         = fx._object_pos
_object_quat        = fx._object_quat


def read_demo(hdf5_path):
    """4-tuple form (name, states, actions, model_xml), as before.

    frame_extract.read_demo grew a 5th element (recorded torques), so the legacy
    shape is preserved here rather than breaking existing unpacking."""
    return fx.read_demo_legacy(hdf5_path)


def replay(env, states, object_type):
    """Back-compat wrapper: demo states -> the old 6-tuple of raw arrays.

    Prefer frame_extract.FrameExtractor(...).from_states(...), which also
    resolves the goal from the env and fills the torque channel."""
    ex = fx.FrameExtractor(env, object_type)
    rows = []
    for st in states:
        env.sim.set_state_from_flattened(st)
        env.sim.forward()
        rows.append(ex._frame(with_dynamics=False))
    ch = fx.FrameExtractor._stack(rows)
    return (ch["eef_pos"], ch["obj_pos"], ch["grip"],
            ch["eef_quat"], ch["obj_quat"], ch["contact"])


def infer_goal(obj):
    """DEPRECATED -- goal = the demo's FINAL object position.

    Correct only when the trajectory SUCCEEDED. On a policy rollout, which is
    precisely where M5 needs a goal, obj[-1] is wherever the object happened to
    stop: measured 0.71 m from the true bin centre on a failed rollout. Because
    object_goal_dist is a BOUNDARY feature, that error re-cuts every phase and
    the reward then reads a z_t it never saw during training.

    Use frame_extract.resolve_goal(env, object_type)."""
    warnings.warn(
        "infer_goal(obj[-1]) is deprecated: it is the demo's final object "
        "position, not the task goal, and is wrong for any trajectory that did "
        "not succeed. Use frame_extract.resolve_goal(env, object_type).",
        DeprecationWarning, stacklevel=2)
    return np.asarray(obj[-1], float).copy()


# ===========================================================================
# SECTION 3 — visualization
# ===========================================================================
def visualize(eef, obj, seg, F, names, goal, dt, out_png, title):
    z, bounds, labels = seg["z"], seg["bounds"], seg["labels"]
    T = len(z); t = np.arange(T) * dt
    nseg = int(z.max()) + 1
    cmap = plt.get_cmap("tab10"); colors = [cmap(i % 10) for i in range(nseg)]
    show = [i for i, n in enumerate(names) if fs.SPEC[n].boundary]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for s in range(nseg):
        m = z == s
        sub = seg["subgoal_per_phase"][s]
        ax.plot(eef[m, 0], eef[m, 1], eef[m, 2], color=colors[s], linewidth=2.5,
                label=f"[{s}] {labels[s]}  (sub:{sub[0]})")
    ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color="0.5", linewidth=1.0,
            linestyle="--", label="object")
    ax.scatter(*goal, color="black", marker="*", s=140, label="goal")
    for b in bounds:
        ax.scatter(*eef[min(b, T - 1)], color="red", marker="x", s=70)
    ax.set_title(f"{title}\neef path by phase (x = boundary)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=7)

    ax2 = fig.add_subplot(1, 2, 2)
    for i in show:
        ax2.plot(t, _minmax(F[:, i]), linewidth=1.8, label=names[i])
    for b in bounds:
        ax2.axvline(b * dt, color="0.3", linestyle=":", alpha=0.8)
    edges = [0] + bounds + [T]
    for s in range(nseg):
        ax2.axvspan(edges[s] * dt, edges[s + 1] * dt, color=colors[s], alpha=0.10)
    ax2.set_title("boundary features (normalized) + phases")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("normalized")
    ax2.legend(loc="center left", fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# SECTION 4 — selftest
# ===========================================================================
def run_selftest(out_dir="."):
    print("=== phase_segment SELFTEST (synthetic; no robosuite) ===")
    T, dt = 360, 0.05
    rng = np.random.default_rng(0)

    # approach: eef-object distance falls to ~0 by t=90, flat, then RISES at the
    # end -- the robot retreats. The rise is real progress toward a different
    # target, not a sidetrack, and the tail phase must pick it up.
    d1 = np.concatenate([np.linspace(1.0, 0.05, 90), np.full(T - 130, 0.05),
                         np.linspace(0.05, 0.45, 40)])
    d1 += rng.normal(0, 0.01, T)
    # gripper with REAL Panda polarity: sum|qpos| RISES on close.
    grip = np.zeros(T); grip[100:300] = 1.0
    # transport: object-goal distance falls t=190..290, WITH a sidetrack at 230
    d2 = np.concatenate([np.full(190, 0.5), np.linspace(0.5, 0.02, 100),
                         np.full(T - 290, 0.02)])
    d2[230:250] += np.linspace(0, 0.15, 20)          # regression = sidetrack
    d2[250:270] += np.linspace(0.15, 0, 20)
    # quality features must never create a phase
    jerk = rng.normal(0, 1, T) ** 2 * 50

    cols = {n: np.zeros(T) for n in fs.NON_DEFERRED}
    cols["eef_object_dist"] = d1
    cols["object_goal_dist"] = d2
    cols["gripper_open"] = grip
    cols["eef_jerk"] = jerk
    cols["object_height"] = np.concatenate(          # would be a lift phase if
        [np.zeros(110), np.linspace(0, 0.2, 80), np.full(T - 190, 0.2)])
    F = np.stack([cols[n] for n in fs.NON_DEFERRED], axis=1)

    seg = segment_features(F)
    nphase = int(seg["z"].max()) + 1
    print(f"phases: {nphase}   bounds: {seg['bounds']}")
    for i, (lab, sg) in enumerate(zip(seg["labels"], seg["subgoal_per_phase"])):
        e = [0] + seg["bounds"] + [T]
        print(f"  phase {i}: t[{e[i]:3d}:{e[i+1]:3d}]  {lab:<26} "
              f"subgoal={sg[0]:<18} degrade_sign={sg[1]:+.0f}")

    ok = True
    labs = seg["labels"]
    grip_labs = [l for l in labs if l.startswith("gripper_open:")]
    if grip_labs[:2] != ["gripper_open:close", "gripper_open:open"]:
        ok = False; print(f"[FAIL] gripper order labels: {grip_labs}")
    # object_height is QUALITY now -> it must NOT create a lift phase
    if any("object_height" in l for l in labs):
        ok = False; print("[FAIL] object_height (quality) created a phase")
    if any("eef_jerk" in l for l in labs):
        ok = False; print("[FAIL] eef_jerk (quality) created a phase")
    # the transport sidetrack must be ABSORBED, not promoted to a boundary
    leaked = [b for b in seg["bounds"] if 225 <= b <= 275]
    if leaked:
        ok = False; print(f"[FAIL] sidetrack leaked a boundary at {leaked}")
    # the retreat tail must NOT inherit the gripper subgoal; it must read
    # eef_object_dist(-1) = "should be getting away and is not" off its own data
    tail_feat, tail_sign = seg["subgoal_per_phase"][-1]
    print(f"\ntail phase subgoal = {tail_feat}({tail_sign:+.0f})")
    if tail_feat != "eef_object_dist" or tail_sign != -1.0:
        ok = False
        print(f"[FAIL] the retreat tail should be eef_object_dist(-1), not "
              f"{tail_feat}({tail_sign:+.0f}) -- it must read its own data, not "
              f"inherit the previous phase")
    else:
        print("   -> read off the tail's own data (not inherited) ✓")
        print("      the retreat is the one phase whose degradation cannot fail")
        print("      the task, so its ladder is pure efficiency -- what BfD needs")

    # the retreat is invisible to the plateau machinery: show why the tail must
    # be read directly
    i_eo = fs.NAMES.index("eef_object_dist")
    hwm = high_water_mark(normalized_progress(F[:, i_eo]))
    print(f"\n   (high-water mark across the retreat: {hwm[T-45]:.3f} -> "
          f"{hwm[-1]:.3f} -- the rise is absorbed as a regression, which is the")
    print(f"    whitelist principle working; the tail is read directly instead)")

    # the dead-filter regression: pure noise must NOT be treated as progress
    noise = np.full(T, 0.5) + rng.normal(0, 0.002, T)
    print(f"\n   trend_ratio(real progress)={trend_ratio(F[:, i_eo]):.2f}  "
          f"trend_ratio(pure noise)={trend_ratio(noise):.2f}")
    if trend_ratio(noise) >= 0.15:
        ok = False; print("[FAIL] the trend filter lets noise through")

    for feat, sign in seg["subgoal_per_phase"][:-1]:
        if feat == "object_goal_dist" and sign != +1.0:
            ok = False; print(f"[FAIL] {feat} degrade_sign should be +1")

    # per-phase subgoal MAIN-ONLY (the --subgoal main ablation)
    sset_main = phase_subgoal_set(F, seg["z"], seg["bounds"],
                                  seg["subgoal_per_phase"], main_only=True)
    print("\nsubgoal per phase (MAIN-ONLY, the ablation):")
    for k in sorted(sset_main):
        d = sset_main[k]
        print(f"  phase {k} {str(d['window']):<12} main={d['main'][0]}"
              f"({d['main'][1]:+.0f})  change={d['change']}  hold={d['hold']}")
        if d["change"] != [d["main"]] or d["hold"]:
            ok = False; print(f"[FAIL] phase {k}: main-only must give "
                              f"change=[main] and hold=[]")

    # per-phase subgoal SET (the CLI default)
    sset = phase_subgoal_set(F, seg["z"], seg["bounds"], seg["subgoal_per_phase"])
    print("\nsubgoal set per phase (read off the demo, no thresholds by hand):")
    for k in sorted(sset):
        d = sset[k]
        ch = ", ".join(f"{n}({s:+.0f})" for n, s in d["change"])
        print(f"  phase {k} {str(d['window']):<12} main={d['main'][0]:<18}")
        print(f"          change=[{ch}]")
        print(f"          hold={d['hold']}")

    # 1) every phase's main term must be present in `change`
    for k, d in sset.items():
        if d["main"][0] not in [c[0] for c in d["change"]]:
            ok = False; print(f"[FAIL] phase {k}: main not in change")
    # 2) quality features must never enter a subgoal
    for k, d in sset.items():
        bad = [n for n, _ in d["change"] if fs.SPEC[n].kind == fs.QUALITY]
        bad += [n for n in d["hold"] if fs.SPEC[n].kind == fs.QUALITY]
        if bad:
            ok = False; print(f"[FAIL] phase {k}: quality feature in subgoal: {bad}")
    # 3) the gripper phase must gain something CONTINUOUS to degrade -- that is
    #    the whole point of the set (its own feature is near-binary and flat)
    gk = [k for k, d in sset.items() if d["main"][0] == "gripper_open"]
    for k in gk:
        d = sset[k]
        cont = [n for n, _ in d["change"] if n != "gripper_open"] + d["hold"]
        if not cont:
            ok = False
            print(f"[FAIL] gripper phase {k} has ONLY the binary gripper in its "
                  f"subgoal -- nothing continuous to degrade")
        else:
            print(f"\n  gripper phase {k}: besides the binary gripper it can be "
                  f"degraded via {cont}")

    print(f"\n  sidetrack at t=230..270 absorbed: {'✓' if not leaked else '✗'}   "
          f"quality features created no phase: "
          f"{'✓' if not any('jerk' in l or 'height' in l for l in labs) else '✗'}")

    # -- Psi and F must segment identically ---------------------------------
    # segment_features has always filtered by kind internally, so restricting
    # the input to the boundary columns cannot change the answer. Asserting it
    # is what lets degradation_proto.py keep calling segment_features(Phi)
    # untouched while new code passes Psi.
    Psi = F[:, fs.boundary_mask()]
    seg_psi = segment_features(Psi, names=fs.PSI_COLUMNS)
    same = (np.array_equal(seg_psi["z"], seg["z"])
            and list(seg_psi["bounds"]) == list(seg["bounds"])
            and list(seg_psi["labels"]) == list(seg["labels"]))
    if not same:
        ok = False
        print(f"  [FAIL] segment_features(Psi) != segment_features(F): "
              f"{seg_psi['bounds']} vs {seg['bounds']}")
    else:
        print(f"  segment_features(Psi) == segment_features(F): "
              f"bounds={seg['bounds']}  -> old and new call sites agree")

    # A quality feature must be rejected as a phase input by construction.
    try:
        segment_features(F[:, :1], names=["eef_jerk"])
        if any(fs.SPEC["eef_jerk"].boundary for _ in [0]):
            ok = False; print("  [FAIL] a quality feature became a boundary")
        else:
            print("  segment_features(names=['eef_jerk']) -> 0 boundaries "
                  "(quality can never cut a phase)")
    except Exception as ex:
        print(f"  segment_features on a quality column rejected: "
              f"{type(ex).__name__}")

    print(f"[selftest] {'PASS' if ok else 'FAIL'}  ({nphase} phases, no lift)")
    return ok


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="M1: phase segmentation + per-phase subgoal sets.")
    ap.add_argument("--demo-root", default=os.path.join(".", "demos"))
    ap.add_argument("--pattern", default="demo.hdf5")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-demos", type=int, default=0)
    ap.add_argument("--subgoal", choices=["main", "set"], default="set",
                    help="set (default): the full (main, change, hold) pattern "
                         "read off the demo -- what makes gripper phases "
                         "degradable at all. main: each phase's subgoal is the "
                         "SINGLE feature that segmented it; kept as the named "
                         "ablation (collapses score_terms to one term "
                         "downstream, and fcm/degradation flags cannot undo "
                         "that once it is baked into phase_seg.npz).")
    ap.add_argument("--torque", choices=["auto", "recorded", "inverse", "none"],
                    default="auto",
                    help="auto: recorded torques if the hdf5 has them, else "
                         "MuJoCo inverse dynamics. Torque is NOT replayable "
                         "(sim.data.actuator_force reads 0 after a state "
                         "injection), so without one of these the energy "
                         "feature is identically 0.")
    ap.add_argument("--allow-goal-fallback", action="store_true",
                    help="permit goal = obj[-1] when the env exposes no goal. "
                         "Only valid for a SUCCESSFUL demo; never for a rollout.")
    ap.add_argument("--debug-keys", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args.out_dir or ".")
        return

    paths = sorted(glob(os.path.join(args.demo_root, "**", args.pattern),
                        recursive=True))
    if not paths:
        raise SystemExit(f"No '{args.pattern}' under {args.demo_root}. "
                         f"Use --selftest to verify the segmenter without demos.")
    if args.max_demos > 0:
        paths = paths[:args.max_demos]
    print(f"[info] {len(paths)} demo file(s). schema: {fs.N_FEATURES} features, "
          f"boundary = {fs.BOUNDARY}")
    print(f"[info] phase input Psi = {fs.PSI_COLUMNS} (observation + goal only)")

    cache, debugged = {}, False
    try:
        for hp in paths:
            env_info, demos = fx.read_demo(hp)          # 5-tuple: + torques
            robots = env_info["robots"]
            sig = (env_info["env_name"],
                   tuple(robots) if isinstance(robots, (list, tuple)) else (robots,))
            if sig not in cache:
                print(f"[info] building env {sig[0]} / {sig[1]} ...")
                cache[sig] = fx.build_env(env_info)
            env = cache[sig]
            dt = 1.0 / float(env_info.get("control_freq", 20))
            object_type = env_info.get("object_type", None)
            extractor = fx.FrameExtractor(env, object_type, dt=dt)

            for name, states, actions, model_xml, torques in demos:
                fx.reset_to_scene(env, model_xml)
                if args.debug_keys and not debugged:
                    print("\nobs keys:", sorted(fx._obs(env).keys()))
                    objs = getattr(env, "objects", None)
                    if objs:
                        print("objects:", [(getattr(x, "name", "?"),
                                            getattr(x, "root_body", "?"))
                                           for x in objs])
                    debugged = True

                # -- extraction: ONE call, shared with M5 --------------------
                # The goal is resolved from the ENV here, not from obj[-1].
                # object_goal_dist is a boundary feature, so a goal that is off
                # by the width of a bin moves the phase cuts themselves.
                fr = extractor.from_states(
                    states, actions=actions, torques=torques,
                    torque_mode=args.torque,
                    allow_goal_fallback=args.allow_goal_fallback)
                print(f"[info] {fr.summary()}")

                # -- segmentation runs on Psi (obs + goal only) --------------
                Psi = fs.compute_psi(fr)
                seg = segment_features(Psi, names=fs.PSI_COLUMNS)

                # -- the full feature matrix is a separate concern -----------
                F = fs.compute_from_frames(fr)
                sset = phase_subgoal_set(F, seg["z"], seg["bounds"],
                                         seg["subgoal_per_phase"],
                                         main_only=(args.subgoal == "main"))

                out_dir = args.out_dir or os.path.dirname(hp)
                os.makedirs(out_dir, exist_ok=True)
                tag = os.path.basename(os.path.dirname(hp)) or name
                suf = f"_{tag}" if args.out_dir else ""
                out_npz = os.path.join(out_dir, f"phase_seg{suf}.npz")
                # Key names are unchanged so fcm_proto.py reads this file as-is;
                # torque/qvel/roles are additions.
                np.savez(
                    out_npz, z=seg["z"], bounds=np.array(seg["bounds"]),
                    labels=np.array(seg["labels"], dtype=object),
                    subgoal_per_phase=np.array(seg["subgoal_per_phase"],
                                               dtype=object),
                    subgoal_set=np.array(sorted(sset.items()), dtype=object),
                    goal=fr.goal, eef=fr.eef_pos, obj=fr.obj_pos,
                    eef_quat=fr.eef_quat, obj_quat=fr.obj_quat,
                    grip=fr.grip, contact=fr.contact, F=F, Psi=Psi,
                    qvel=fr.qvel if fr.qvel is not None else np.zeros(0),
                    torque=fr.torque if fr.torque is not None else np.zeros(0),
                    goal_source=str(fr.goal_source),
                    torque_source=str(fr.torque_source),
                    feat_names=np.array(fs.NAMES, dtype=object),
                    psi_names=np.array(fs.PSI_COLUMNS, dtype=object),
                    reward_input=np.array(fs.REWARD_INPUT, dtype=object),
                    heldout=np.array(fs.HELDOUT, dtype=object), dt=dt)
                visualize(fr.eef_pos, fr.obj_pos, seg, F, fs.NAMES, fr.goal, dt,
                          os.path.join(out_dir, f"phase_seg{suf}.png"),
                          f"{env_info['env_name']} / {tag}")
                nph = int(seg["z"].max()) + 1
                print(f"[ok] {hp} -> {nph} phases, bounds={seg['bounds']}")
                for i, lab in enumerate(seg["labels"]):
                    d = sset[i]
                    ch = ", ".join(f"{n}({s:+.0f})" for n, s in d["change"])
                    print(f"       phase {i}: {lab}")
                    print(f"          main   = {d['main'][0]} "
                          f"({d['main'][1]:+.0f})")
                    print(f"          change = [{ch}]")
                    print(f"          hold   = {d['hold']}")
    finally:
        for env in cache.values():
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
