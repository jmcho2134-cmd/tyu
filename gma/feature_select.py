#!/usr/bin/env python
"""
feature_select.py — THE feature schema. The single human prior of this project.
================================================================================

Everything downstream (phase_segment.py, fcm.py, later degradation/reward) imports
its feature list, kinds and computation from here. Nothing else defines features.

What a human specifies (and ONLY this)
--------------------------------------
For each feature: its NAME, its KIND, and whether it may create a phase BOUNDARY.
Nothing else. Phase count, phase names, phase order, thresholds, subgoals,
degradation directions and their magnitudes are all derived from data downstream.

    kind = "progress"  a quantity that monotonically approaches a target over a
                       successful demo (a distance -> 0). Subgoal achieved = it
                       reaches a PLATEAU. Direction is auto-detected from the net
                       trend, never hardcoded.
    kind = "event"     a discrete mode (gripper open/closed). Boundary = a mode
                       transition. The split point comes from 2-means on the data.
    kind = "quality"   HOW WELL / HOW COSTLY the motion was (jerk, effort, slip).
                       NEVER a boundary, NEVER a subgoal -- only a consequence to
                       predict. Quality features are free to degrade.

    boundary = True    this feature is allowed to cut the demo into phases.
    deferred = True    cannot be computed until phase_segment has produced the
                       waypoints (see compute_deferred).

Recorded design decisions (each was a real decision, not a default)
------------------------------------------------------------------
* object_height is "quality", NOT "progress". Lifting is part of transport, not a
  separate subgoal; as a progress feature it resurrects a "lift" phase whose
  degradation direction tangles with rotation. This is a SCHEMA judgement (what
  counts as a subgoal), not a threshold, so it does not violate "thresholds must
  come from data".
* contact is "event" with boundary=False. Its shape is a clean two-state signal
  (a data-driven shape classifier calls it an event, with 0.000 of the time spent
  in transit -- exactly like the gripper). It joins phase SUBGOALS -- "grasp = the
  gripper closes AND contact is established" -- without creating phases of its
  own, so the phase-count question stays a separate experiment.
* grasp_align is "progress" with boundary=False. Bread is non-axisymmetric, so
  the wrist must align to grasp it: the angle falls through approach and plateaus.
  It carries ORIENTATION into the subgoals, which is what lets a gripper phase be
  degraded by MISALIGNMENT (proposal Sec 8.1) instead of by the near-binary
  gripper command, which cannot be moved by a small additive perturbation.
* There are NO waypoint features. An earlier version scored degradation partly by
  distance from the demo's own waypoint. Measured on real data that term carried
  39-69% of the objective, every phase then chose a rotation-dominated direction
  (phase conditioning collapsed to "rotate the wrist" 5 times out of 5), and the
  gripper phases scored NEGATIVE on their own boundary feature -- the chosen
  direction IMPROVED the actual subgoal and only "won" by differing from the demo.
  That is the similarity-to-the-demo trap proposal Sec 9.3 warns about.

Proposal mapping: this is the Feature Bank's schema (Sec 7.1/7.2). It is NOT a
reward. No weights live here.

    python feature_select.py --selftest
"""

import argparse
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as _R
    _HAS_SCIPY = True
except Exception:                                             # pragma: no cover
    _HAS_SCIPY = False

PROGRESS, EVENT, QUALITY = "progress", "event", "quality"


# ===========================================================================
# THE SCHEMA
# ===========================================================================
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str            # progress | event | quality
    boundary: bool       # may cut the demo into phases
    channel: str         # proposal Sec 7.1 channel
    doc: str
    # --- role: the A / B / C partition (proposal Sec 12.3) -------------------
    # Defaults keep every pre-existing call site working: anything constructed
    # without these fields behaves as before (a plain reward input).
    reward_input: bool = True   # may be fed to R_theta
    heldout: bool = False       # evaluation-only; NEVER a reward input


FEATURES = [
    # ---- boundary features: these three define the phases -------------------
    FeatureSpec("eef_object_dist",  PROGRESS, True,  "interaction",
                "gripper-object distance; plateau = the reach subgoal is met"),
    FeatureSpec("object_goal_dist", PROGRESS, True,  "goal",
                "object-goal distance; plateau = the transport subgoal is met"),
    FeatureSpec("gripper_open",     EVENT,    True,  "robot",
                "gripper aperture proxy sum|qpos|; transitions = grasp / release"),

    # ---- subgoal members, but never boundaries -----------------------------
    FeatureSpec("contact",          EVENT,    False, "interaction",
                "gripper-object contact / grasp flag; part of the grasp subgoal"),
    FeatureSpec("grasp_align",      PROGRESS, False, "interaction",
                "eef-object relative rotation angle; carries ORIENTATION into the "
                "subgoals, so a gripper phase can be degraded by misalignment"),

    # ---- quality features: predicted, never a boundary or a subgoal ---------
    FeatureSpec("object_height",    QUALITY,  False, "object",
                "object lift height (demoted from progress: no separate lift phase)",
                reward_input=False, heldout=True),
    FeatureSpec("eef_speed",        QUALITY,  False, "robot", "eef speed"),
    FeatureSpec("object_speed",     QUALITY,  False, "object", "object speed"),
    FeatureSpec("action_magnitude", QUALITY,  False, "robot",
                "control effort. reward_input=False ON PURPOSE: the degradation "
                "operator is a'=a+lambda*d_z, so ||a'||^2 = ||a||^2 + "
                "2*lambda*<a,d_z> + lambda^2*||d_z||^2 -- the last term is "
                "monotone in lambda and state-independent, making this an almost "
                "perfect proxy for the rung label. A reward that reads it scores "
                "the ranking with R = -w||a|| and every other feature's gradient "
                "dies. That is 'merely undo the operator' (proposal Sec 13) while "
                "still passing the ordinal-consistency check (Sec 9.5).",
                reward_input=False, heldout=True),
    FeatureSpec("eef_accel",        QUALITY,  False, "robot", "eef acceleration"),
    FeatureSpec("eef_jerk",         QUALITY,  False, "robot", "eef jerk"),
    FeatureSpec("object_slip",      QUALITY,  False, "interaction",
                "object motion in the eef frame (slip / drift proxy)"),
    FeatureSpec("eef_ang_speed",    QUALITY,  False, "robot", "eef angular speed"),
    FeatureSpec("object_ang_speed", QUALITY,  False, "object",
                "object angular speed (tumbling proxy)"),
    FeatureSpec("energy",           QUALITY,  False, "robot",
                "mechanical power proxy sum_i |tau_i * qdot_i| * dt. Needs the "
                "TORQUE channel, which is NOT in the observation and is NOT "
                "recoverable by state replay (sim.data.actuator_force reads 0 "
                "after set_state_from_flattened, because the flattened state is "
                "[time, qpos, qvel] and carries no ctrl). Held out of the reward "
                "on purpose: if a policy trained on kinematic features alone also "
                "spends less energy, that is evidence the reward generalised "
                "rather than reversing the operator (proposal Sec 12.3).",
                reward_input=False, heldout=True),
]

NAMES = [f.name for f in FEATURES]
KINDS = [f.kind for f in FEATURES]
SPEC = {f.name: f for f in FEATURES}
BOUNDARY = [f.name for f in FEATURES if f.boundary]
SUBGOAL_ELIGIBLE = [f.name for f in FEATURES if f.kind != QUALITY]
N_FEATURES = len(FEATURES)
# Back-compat alias: every feature is computed directly now, so "non-deferred"
# is the whole list. Kept so older call sites keep working.
NON_DEFERRED = NAMES

# --- the A / B / C partition (proposal Sec 12.3) ----------------------------
# PSI_COLUMNS is exactly BOUNDARY: the phase input is not a second computation,
# it is the boundary COLUMNS of the same feature matrix. One implementation
# means M1 (demo replay) and M5 (policy rollout) cannot drift apart, which is
# what proposal Sec 6.2 requires of the phase module.
PSI_COLUMNS = BOUNDARY
REWARD_INPUT = [f.name for f in FEATURES if f.reward_input]
HELDOUT = [f.name for f in FEATURES if f.heldout]

# Channels that a Frames object carries but that do NOT come from the
# observation. Psi must never touch these (see compute_psi / the selftest).
NON_OBS_CHANNELS = ("torque", "qvel")


def index_of(name):
    return NAMES.index(name)


def boundary_mask():
    return np.array([f.boundary for f in FEATURES], dtype=bool)


def reward_input_mask():
    """Columns of F that may be fed to R_theta. Excludes action_magnitude and
    energy -- see their FeatureSpec docs for why."""
    return np.array([f.reward_input for f in FEATURES], dtype=bool)


def heldout_mask():
    """Columns reserved for evaluation only (proposal Sec 12.3's 'metrics not
    directly targeted by the Degenerator')."""
    return np.array([f.heldout for f in FEATURES], dtype=bool)


def subgoal_eligible_mask():
    """Quality features can never be part of a subgoal: a large jerk change is a
    COST, not a failure to achieve the phase."""
    return np.array([f.kind != QUALITY for f in FEATURES], dtype=bool)


# ===========================================================================
# rotation / kinematics helpers
# ===========================================================================
def qnorm(q):
    q = np.asarray(q, float); n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])


def canonicalize(quats):
    """Sign-consistent quaternion sequence (q and -q are the same rotation)."""
    q = np.array(quats, float).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def rel_angle(q1, q2):
    """Angle (rad, [0,pi]) of the relative rotation q1->q2. Sign-flip safe."""
    if not _HAS_SCIPY:
        d = abs(float(np.dot(qnorm(q1), qnorm(q2))))
        return float(2 * np.arccos(min(1.0, d)))
    r = _R.from_quat(qnorm(q1)).inv() * _R.from_quat(qnorm(q2))
    return float(r.magnitude())


def angular_speed(quats, dt):
    q = canonicalize(quats); T = len(q); out = np.zeros(T)
    for t in range(1, T):
        out[t] = rel_angle(q[t - 1], q[t]) / dt
    return out


def smooth_positions(pos, window=9, poly=2):
    """Savgol-smooth (T,3) positions before differentiating (teleop noise is
    instrumentation, not motion)."""
    pos = np.asarray(pos, float); T = len(pos)
    if not _HAS_SCIPY or T < poly + 2:
        return pos.copy()
    w = window if window <= T else (T if T % 2 == 1 else T - 1)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return pos.copy()
    return savgol_filter(pos, window_length=w, polyorder=poly, axis=0, mode="interp")


def eef_kinematics(eef, dt):
    """Smoothed velocity/accel/jerk VECTORS (each (T,3)); first frames padded."""
    eef = np.asarray(eef, float); T = len(eef)
    es = smooth_positions(eef)
    vel = np.zeros((T, 3)); acc = np.zeros((T, 3)); jrk = np.zeros((T, 3))
    if T > 1:
        vel[1:] = np.diff(es, axis=0) / dt; vel[0] = vel[1]
    if T > 2:
        acc[2:] = np.diff(vel[1:], axis=0) / dt; acc[:2] = acc[2]
    if T > 3:
        jrk[3:] = np.diff(acc[2:], axis=0) / dt; jrk[:3] = jrk[3]
    return vel, acc, jrk


def compose_axis_angle(a1, a2):
    """rotvec(R(a1)*R(a2)) -- the correct analogue of a1+a2 on SO(3). Linear add
    is only first-order for small angles; large-lambda rotation perturbations
    should compose properly."""
    if not _HAS_SCIPY:
        return np.asarray(a1, float) + np.asarray(a2, float)
    r = _R.from_rotvec(np.asarray(a1, float)) * _R.from_rotvec(np.asarray(a2, float))
    return r.as_rotvec()


def perturb_action(a, delta, adim, proper_rotation=False, clip=True):
    """a + delta, optionally composing the rotation dims (3..adim-2) properly.
    Clipping to the controller's [-1,1] box is ON by default: an unclipped
    perturbation saturates the controller and what actually executed is then
    unknown (this silently broke earlier lambda=2 checks)."""
    a = np.asarray(a, float).copy(); delta = np.asarray(delta, float)
    out = a + delta
    if proper_rotation and adim > 4:
        out[3:adim - 1] = compose_axis_angle(a[3:adim - 1], delta[3:adim - 1])
    return np.clip(out, -1.0, 1.0) if clip else out


def clip_fraction(a, delta, adim):
    """How much of a+delta falls outside the action box (0 = fully feasible).
    Used as the feasibility guard when ranking candidate directions."""
    raw = np.asarray(a, float) + np.asarray(delta, float)
    excess = np.maximum(np.abs(raw) - 1.0, 0.0)
    denom = np.abs(np.asarray(delta, float)).sum() + 1e-9
    return float(excess.sum() / denom)


# ===========================================================================
# feature computation
# ===========================================================================
_ENERGY_WARNED = False


def compute_trajectory(eef, obj, grip, dt, eef_quat=None, obj_quat=None,
                       goal=None, obj0_z=None, actions=None, contact=None,
                       torque=None, qvel=None, columns=None):
    """Frames -> (T, k) matrix. One call; the old assemble()/compute_deferred()
    two-step is gone with the waypoint features.

    The positional signature is UNCHANGED from the version fcm_proto.py and
    degradation_proto.py call, so those files keep working untouched.

    columns : None            -> the full (T, N_FEATURES) matrix, ordered as NAMES
              list of names   -> only those columns, in the order given.
                                 columns=BOUNDARY yields Psi (T, 3).

    Psi is deliberately NOT a separate function. Computing it as a column subset
    of this one implementation is what makes M1 (demo replay) and M5 (policy
    rollout) structurally unable to disagree about the phase input -- proposal
    Sec 6.2. A side benefit: because none of the boundary columns touch torque,
    actions or quaternions, Psi still computes when those channels are missing,
    so phase segmentation keeps running before the torque channel is wired up.

    Rotation channels are 0 when quats are omitted (4-dim OSC_POSITION).
    action_magnitude[t] = ||actions[t-1]||: the effort that PRODUCED state t, so
    demo and perturbed rows share one rule.
    energy[t] = sum_i |tau_i * qdot_i| * dt; 0 (with one warning) when the torque
    channel is absent -- see the FeatureSpec doc for why it cannot be replayed.
    """
    global _ENERGY_WARNED

    out_names = list(NAMES) if columns is None else list(columns)
    unknown = [n for n in out_names if n not in SPEC]
    if unknown:
        raise ValueError(f"unknown feature column(s): {unknown}")
    want = set(out_names)

    eef = np.asarray(eef, float); obj = np.asarray(obj, float)
    grip = np.asarray(grip, float); T = len(eef)
    goal = obj[-1] if goal is None else np.asarray(goal, float)
    obj0_z = obj[0, 2] if obj0_z is None else float(obj0_z)

    col = {}

    # ---- boundary columns (Psi): observation + goal ONLY --------------------
    if "eef_object_dist" in want:
        col["eef_object_dist"] = np.linalg.norm(eef - obj, axis=1)
    if "object_goal_dist" in want:
        col["object_goal_dist"] = np.linalg.norm(obj - goal[None, :], axis=1)
    if "gripper_open" in want:
        col["gripper_open"] = grip

    # ---- remaining subgoal members -----------------------------------------
    if "contact" in want:
        col["contact"] = (np.zeros(T) if contact is None
                          else np.asarray(contact, float))
    if want & {"grasp_align", "eef_ang_speed", "object_ang_speed"}:
        if eef_quat is None or obj_quat is None:
            eef_as = np.zeros(T); obj_as = np.zeros(T); align = np.zeros(T)
        else:
            eef_as = angular_speed(eef_quat, dt)
            obj_as = angular_speed(obj_quat, dt)
            eq = canonicalize(eef_quat); oq = canonicalize(obj_quat)
            align = np.array([rel_angle(eq[t], oq[t]) for t in range(T)])
        if "grasp_align" in want:      col["grasp_align"] = align
        if "eef_ang_speed" in want:    col["eef_ang_speed"] = eef_as
        if "object_ang_speed" in want: col["object_ang_speed"] = obj_as

    # ---- quality columns ----------------------------------------------------
    if "object_height" in want:
        col["object_height"] = obj[:, 2] - obj0_z
    if "eef_speed" in want:
        v = np.zeros(T)
        if T > 1:
            v[1:] = np.linalg.norm(np.diff(eef, axis=0), axis=1) / dt
        col["eef_speed"] = v
    if "object_speed" in want:
        v = np.zeros(T)
        if T > 1:
            v[1:] = np.linalg.norm(np.diff(obj, axis=0), axis=1) / dt
        col["object_speed"] = v
    if "action_magnitude" in want:
        act_mag = np.zeros(T)
        if actions is not None and T > 1:
            an = np.linalg.norm(np.asarray(actions, float), axis=1)
            m = min(T - 1, len(an)); act_mag[1:1 + m] = an[:m]
        col["action_magnitude"] = act_mag
    if want & {"eef_accel", "eef_jerk"}:
        _, acc, jrk = eef_kinematics(eef, dt)
        if "eef_accel" in want: col["eef_accel"] = np.linalg.norm(acc, axis=1)
        if "eef_jerk" in want:  col["eef_jerk"] = np.linalg.norm(jrk, axis=1)
    if "object_slip" in want:
        rel = obj - eef
        slip = np.zeros(T)
        if T > 1:
            slip[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        col["object_slip"] = slip
    if "energy" in want:
        e = np.zeros(T)
        if torque is None or qvel is None:
            if not _ENERGY_WARNED:
                print("[warn] feature_select: torque/qvel missing -> energy "
                      "column is identically 0. Torque is NOT recoverable by "
                      "state replay; record it during collection or fall back "
                      "to frame_extract.inverse_dynamics_torque().")
                _ENERGY_WARNED = True
        else:
            tau = np.atleast_2d(np.asarray(torque, float))
            qd = np.atleast_2d(np.asarray(qvel, float))
            j = min(tau.shape[1], qd.shape[1])          # arm joints in common
            n = min(T, len(tau), len(qd))
            e[:n] = np.abs(tau[:n, :j] * qd[:n, :j]).sum(axis=1) * dt
        col["energy"] = e

    return np.stack([col[n] for n in out_names], axis=1)


# ---------------------------------------------------------------------------
# Frames-based entry points (what frame_extract.py hands over)
# ---------------------------------------------------------------------------
def compute_from_frames(fr, columns=None):
    """Frames dataclass -> feature matrix. A thin delegate to compute_trajectory
    so there is exactly one implementation of every feature."""
    return compute_trajectory(
        fr.eef_pos, fr.obj_pos, fr.grip, fr.dt,
        eef_quat=fr.eef_quat, obj_quat=fr.obj_quat,
        goal=fr.goal, obj0_z=fr.obj0_z,
        actions=fr.actions, contact=fr.contact,
        torque=fr.torque, qvel=fr.qvel, columns=columns)


def compute_psi(fr):
    """Psi = the phase input. Observation + goal only, by construction.

    THE call M1 and M5 share: phase_segment.segment_features(compute_psi(fr),
    names=feature_select.PSI_COLUMNS)."""
    return compute_from_frames(fr, columns=PSI_COLUMNS)


# ===========================================================================
# selftest
# ===========================================================================
def run_selftest():
    print("=== feature_select SELFTEST ===")
    ok = True
    print(f"{N_FEATURES} features\n")
    print(f"{'name':<18}{'kind':<10}{'bound':<7}{'subgoal':<9}"
          f"{'reward_in':<11}heldout")
    for f in FEATURES:
        print(f"{f.name:<18}{f.kind:<10}{str(f.boundary):<7}"
              f"{str(f.kind != QUALITY):<9}{str(f.reward_input):<11}{f.heldout}")

    # schema invariants the rest of the pipeline relies on
    if BOUNDARY != ["eef_object_dist", "object_goal_dist", "gripper_open"]:
        ok = False; print(f"[FAIL] boundary set changed: {BOUNDARY}")
    if SPEC["object_height"].kind != QUALITY:
        ok = False; print("[FAIL] object_height must be quality (no lift phase)")
    if SPEC["contact"].kind != EVENT or SPEC["contact"].boundary:
        ok = False; print("[FAIL] contact must be event AND boundary=False -- it "
                          "joins subgoals without creating phases")
    if SPEC["grasp_align"].kind != PROGRESS or SPEC["grasp_align"].boundary:
        ok = False; print("[FAIL] grasp_align must be progress AND boundary=False")
    for f in FEATURES:
        if f.kind == QUALITY and f.boundary:
            ok = False; print(f"[FAIL] quality {f.name} may not be a boundary")
        if "waypoint" in f.name:
            ok = False; print(f"[FAIL] {f.name}: waypoint features are gone for "
                              "good (they turned degradation into 'differs from "
                              "the demo')")
    print(f"\nboundary -> phases:  {BOUNDARY}")
    print(f"subgoal-eligible:    {SUBGOAL_ELIGIBLE}")
    print("   -> contact & grasp_align are eligible but create no phase, so a")
    print("      gripper phase can be degraded by lost contact / misalignment")
    print("      instead of by the near-binary gripper command (proposal Sec 8.1).")

    # -- the A / B / C partition (proposal Sec 12.3) -------------------------
    print(f"\nreward inputs (phi): {REWARD_INPUT}")
    print(f"held out (eval only):{HELDOUT}")
    if SPEC["action_magnitude"].reward_input:
        ok = False
        print("[FAIL] action_magnitude must NOT be a reward input: it is an "
              "almost perfect proxy for the perturbation level lambda, so a "
              "ranking loss solves itself with R = -w||a|| and every other "
              "feature's gradient dies (proposal Sec 13).")
    if SPEC["energy"].reward_input:
        ok = False; print("[FAIL] energy must be held out, not a reward input")
    both = set(REWARD_INPUT) & set(HELDOUT)
    if both:
        ok = False; print(f"[FAIL] a feature cannot be both reward input and "
                          f"held out: {sorted(both)}")
    if not set(BOUNDARY) <= set(REWARD_INPUT):
        ok = False; print("[FAIL] every boundary feature should also be a "
                          "reward input (z_t is derived from them)")

    # -- Psi invariants: the M1/M5 consistency guarantee (proposal Sec 6.2) ---
    if PSI_COLUMNS != BOUNDARY:
        ok = False; print(f"[FAIL] PSI_COLUMNS != BOUNDARY: {PSI_COLUMNS}")

    # computation on a synthetic pick-and-place
    T, dt = 120, 0.05
    eef = np.zeros((T, 3)); obj = np.zeros((T, 3))
    eef[:, 0] = np.linspace(0.0, 0.5, T)
    eef[:, 2] = 1.0 - 0.15 * np.sin(np.linspace(0, np.pi, T))
    obj[:60] = [0.25, 0.0, 0.85]
    obj[60:, 0] = np.linspace(0.25, 0.5, T - 60); obj[60:, 2] = 0.85
    grip = np.zeros(T); grip[60:110] = 1.0
    con = np.zeros(T); con[60:110] = 1.0
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))
    acts = np.zeros((T - 1, 7))

    F = compute_trajectory(eef, obj, grip, dt, quat, quat, goal=obj[-1],
                           actions=acts, contact=con)
    if F.shape != (T, N_FEATURES):
        ok = False; print(f"[FAIL] shape {F.shape} != {(T, N_FEATURES)}")
    if not np.allclose(F[:, index_of("contact")], con):
        ok = False; print("[FAIL] contact column not wired through")
    print(f"\ncompute_trajectory -> {F.shape} in ONE call "
          f"(the assemble/compute_deferred two-step is gone)")

    # (a) Psi is computable from observation + goal ALONE. No quats, no actions,
    #     no contact, no torque, no qvel -- exactly what a policy rollout can
    #     always produce. If this ever fails, a non-observation channel has
    #     leaked into a boundary feature and M5 can no longer segment.
    Psi_min = compute_trajectory(eef, obj, grip, dt, goal=obj[-1],
                                 columns=PSI_COLUMNS)
    if Psi_min.shape != (T, len(PSI_COLUMNS)) or not np.isfinite(Psi_min).all():
        ok = False; print(f"[FAIL] Psi not computable from obs+goal alone: "
                          f"{Psi_min.shape}")
    else:
        print(f"compute_trajectory(columns=PSI_COLUMNS) -> {Psi_min.shape} with "
              f"quats/actions/contact/torque all None  (obs+goal suffice)")

    # (b) ONE implementation: Psi must equal the boundary columns of the full F.
    #     This is the guarantee that M1 and M5 cannot disagree (Sec 6.2).
    if not np.allclose(Psi_min, F[:, boundary_mask()]):
        ok = False; print("[FAIL] Psi != F[:, boundary_mask()] -- the column "
                          "subset and the full matrix disagree")
    else:
        print("Psi == F[:, boundary_mask()]  -> single implementation verified")

    # (c) energy is 0 without torque, non-zero with it, and warns once.
    E = index_of("energy")
    if not np.allclose(F[:, E], 0.0):
        ok = False; print("[FAIL] energy should be 0 when torque is absent")
    tau = np.ones((T, 7)) * 2.0; qd = np.ones((T, 7)) * 0.5
    F2 = compute_trajectory(eef, obj, grip, dt, quat, quat, goal=obj[-1],
                            actions=acts, contact=con, torque=tau, qvel=qd)
    exp = 7 * abs(2.0 * 0.5) * dt
    if not np.allclose(F2[:, E], exp):
        ok = False; print(f"[FAIL] energy {F2[0, E]} != expected {exp}")
    else:
        print(f"energy with torque -> {F2[0, E]:.4f} J/step (expected {exp:.4f})")

    cf = clip_fraction(np.array([0.9, 0, 0, 0, 0, 0, 0]),
                       np.array([0.5, 0, 0, 0, 0, 0, 0]), 7)
    if not (cf > 0):
        ok = False; print("[FAIL] clip_fraction should flag 0.9+0.5 > 1")
    print(f"clip_fraction(0.9 + 0.5) = {cf:.2f} (>0 => infeasible)")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
    else:
        print(f"{N_FEATURES} features: {NAMES}")
        print(f"boundary: {BOUNDARY}")
        print("(this module is the schema; run --selftest to verify it)")


if __name__ == "__main__":
    main()
