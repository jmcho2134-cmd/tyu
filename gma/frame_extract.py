#!/usr/bin/env python
"""
frame_extract.py — the ONE place raw channels are produced.
================================================================================

Everything that touches robosuite lives here. `phase_segment.py` used to own
this code (its old SECTION 2); it was moved out because the segmenter must run
on M5 policy rollouts too (proposal Sec 6.2), and a rollout does not come from
an hdf5 file. With the extraction split off, M1 and M5 call the SAME per-frame
code and cannot drift apart.

What a demo.hdf5 actually holds
-------------------------------
    states  (T, 71)   flattened MuJoCo state = [time, qpos, qvel]   <- NOT an obs
    actions (T, adim) exactly what was fed to env.step, already in [-1, 1]
    torques (T, nu)   OPTIONAL, only if collect_demo.py recorded it
    model_file        the scene XML (attrs)

The 71-dim state is a simulator save-file, not an observation. `robot0_eef_pos`
does not appear anywhere in it -- it is a forward-kinematics result. So an
observation is not sliced out of the state, it is REGENERATED: inject the state,
run sim.forward(), then read the observables. Skipping sim.forward() returns the
PREVIOUS frame's values with no error, which is why _frame() never omits it.

Three channels are not observations
-----------------------------------
    goal    NOT in obs for robosuite manipulation envs. Comes from the env
            (resolve_goal). The old infer_goal(obj[-1]) is correct only for a
            SUCCESSFUL demo; on a failed rollout obj[-1] is wherever the object
            stopped, which was measured ~0.71 m from the true bin centre. Since
            object_goal_dist is a BOUNDARY feature, a wrong goal silently
            re-cuts the phases and R_theta(phi, z, g) then reads a z it was
            never trained against.
    torque  NOT in obs and NOT recoverable by replay: the flattened state has no
            `ctrl`, so sim.data.actuator_force reads exactly 0 after
            set_state_from_flattened + forward (measured: live std ~25 N.m per
            joint, replayed std 0.000). Either record it during collection or
            reconstruct it with inverse_dynamics_torque().
    qvel    IS in the state, so it replays exactly; kept next to torque because
            the energy feature needs both.

    python frame_extract.py --selftest      # no robosuite needed
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field, fields
from glob import glob

import numpy as np

import feature_select as fs


# ===========================================================================
# SECTION 1 — the channel container
# ===========================================================================
@dataclass
class Frames:
    """Raw per-timestep channels. The boundary between "what the simulator gave
    us" and "what we computed" -- features are never stored here.

    The Optional-ness is load-bearing documentation: eef_pos/obj_pos/grip/goal
    are always available (M1 and M5 alike), while torque/qvel/contact may be
    absent, and feature_select degrades those columns to 0 rather than crashing.
    """

    # --- always present: observation channels + the env-provided goal --------
    eef_pos: np.ndarray                      # (T, 3)  obs robot0_eef_pos
    obj_pos: np.ndarray                      # (T, 3)  obs {Obj}_pos
    grip: np.ndarray                         # (T,)    obs sum|robot0_gripper_qpos|
    goal: np.ndarray                         # (3,)    env  <- NOT obs
    dt: float

    # --- optional observation channels --------------------------------------
    eef_quat: np.ndarray = None              # (T, 4)  obs
    obj_quat: np.ndarray = None              # (T, 4)  obs
    contact: np.ndarray = None               # (T,)    sim contact query

    # --- optional non-observation channels (never touched by Psi) -----------
    actions: np.ndarray = None               # (T, adim)
    qvel: np.ndarray = None                  # (T, nv)  sim
    torque: np.ndarray = None                # (T, nu)  sim  <- see module doc

    # --- provenance ----------------------------------------------------------
    obj0_z: float = None
    goal_source: str = "unknown"             # which resolve_goal branch fired
    torque_source: str = "none"              # recorded | inverse_dynamics | none
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.eef_pos = np.asarray(self.eef_pos, float)
        self.obj_pos = np.asarray(self.obj_pos, float)
        self.grip = np.asarray(self.grip, float).ravel()
        self.goal = np.asarray(self.goal, float).ravel()
        if self.obj0_z is None:
            self.obj0_z = float(self.obj_pos[0, 2])
        for n in ("eef_quat", "obj_quat", "contact", "actions", "qvel", "torque"):
            v = getattr(self, n)
            if v is not None:
                setattr(self, n, np.asarray(v, float))

    def __len__(self):
        return len(self.eef_pos)

    # -- the obs/non-obs split, machine-checkable ----------------------------
    OBS_FIELDS = ("eef_pos", "obj_pos", "grip", "eef_quat", "obj_quat", "contact")
    NON_OBS_FIELDS = ("qvel", "torque")

    def obs_only(self):
        """A copy with every non-observation channel dropped. Used by the
        selftest to PROVE Psi never reads torque/qvel/actions."""
        return Frames(eef_pos=self.eef_pos, obj_pos=self.obj_pos, grip=self.grip,
                      goal=self.goal, dt=self.dt, eef_quat=self.eef_quat,
                      obj_quat=self.obj_quat, contact=self.contact,
                      obj0_z=self.obj0_z, goal_source=self.goal_source)

    def to_legacy_dict(self):
        """The dict shape fcm_proto.py / degradation_proto.py already pass
        around (`frames["eef"]`, ...). Kept so those files need no edits."""
        return dict(eef=self.eef_pos, obj=self.obj_pos, grip=self.grip,
                    eef_quat=self.eef_quat, obj_quat=self.obj_quat,
                    contact=self.contact)

    def summary(self):
        have = [f.name for f in fields(self)
                if f.name in self.OBS_FIELDS + self.NON_OBS_FIELDS + ("actions",)
                and getattr(self, f.name) is not None]
        return (f"Frames(T={len(self)}, dt={self.dt:.4f}, "
                f"goal={np.round(self.goal, 4)} [{self.goal_source}], "
                f"torque={self.torque_source}, channels={have})")


# ===========================================================================
# SECTION 2 — hdf5 + env plumbing  (moved verbatim from phase_segment.py)
# ===========================================================================
def read_demo(hdf5_path):
    """-> (env_info, [(name, states, actions, model_xml, torques), ...])

    NOTE the 5-tuple: the old phase_segment.read_demo returned 4. Older callers
    unpack 4, so read_demo_legacy() below keeps them working.
    """
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        env_info = json.loads(data.attrs["env_info"])
        demos = []
        for name in data:
            g = data[name]
            if "states" in g and "actions" in g:
                tq = g["torques"][()] if "torques" in g else None
                demos.append((name, g["states"][()], g["actions"][()],
                              g.attrs.get("model_file", None), tq))
    return env_info, demos


def read_demo_legacy(hdf5_path):
    """4-tuple form, for call sites written before torques existed."""
    env_info, demos = read_demo(hdf5_path)
    return env_info, [(n, s, a, x) for n, s, a, x, _ in demos]


def _robosuite_make(**kwargs):
    """robosuite.make is not always exposed by a bare `import robosuite`
    (circular-import timing); it is defined in robosuite.environments.base."""
    try:
        from robosuite import make
    except (ImportError, AttributeError):
        from robosuite.environments.base import make
    return make(**kwargs)


def build_env(env_info):
    return _robosuite_make(
        env_name=env_info["env_name"], robots=env_info["robots"],
        controller_configs=env_info.get("controller_configs"),
        control_freq=env_info.get("control_freq", 20),
        has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_camera_obs=False, reward_shaping=True)


def reset_to_scene(env, model_xml):
    env.reset()
    if model_xml is None:
        return
    xml = model_xml
    try:
        from robosuite.utils.mjcf_utils import postprocess_model_xml
        import inspect
        if len(inspect.signature(postprocess_model_xml).parameters) == 1:
            xml = postprocess_model_xml(model_xml)
    except Exception:
        xml = model_xml
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()


def _obs(env):
    try:
        return env._get_observations(force_update=True)
    except TypeError:
        return env._get_observations()


def _object_pos(o, object_type):
    """The object's TRUE pose comes from the observable '{Name}_pos'. The body
    xpos of '{Name}_main' can be a static reference point that does not track the
    object (this silently flattened height/goal_dist once)."""
    if object_type:
        key = f"{object_type.capitalize()}_pos"
        if key in o:
            return np.asarray(o[key], float)
    cand = [k for k in o if k.endswith("_pos")
            and not k.startswith("robot") and "gripper" not in k]
    return np.asarray(o[cand[0]], float) if cand else np.zeros(3)


def _object_quat(o, object_type):
    if object_type:
        key = f"{object_type.capitalize()}_quat"
        if key in o:
            return np.asarray(o[key], float)
    cand = [k for k in o if k.endswith("_quat")
            and not k.startswith("robot") and "gripper" not in k]
    return np.asarray(o[cand[0]], float) if cand else np.array([0.0, 0.0, 0.0, 1.0])


def read_frame(env, object_type):
    """(eef, obj, grip, eef_quat, obj_quat) from the CURRENT sim state.

    Unchanged from phase_segment.read_frame so fcm_proto / degradation_proto,
    which import it through the shim, behave identically."""
    o = _obs(env)
    eef = np.asarray(o["robot0_eef_pos"], float)
    grip = float(np.sum(np.abs(np.asarray(o.get("robot0_gripper_qpos", [0.0]),
                                          float))))
    obj = _object_pos(o, object_type)
    eef_q = np.asarray(o.get("robot0_eef_quat", [0.0, 0.0, 0.0, 1.0]), float)
    obj_q = _object_quat(o, object_type)
    return eef, obj, grip, eef_q, obj_q


def resolve_object(env, object_type):
    objs = getattr(env, "objects", None)
    if not objs:
        return None
    if object_type:
        for ob in objs:
            if object_type.lower() in getattr(ob, "name", "").lower():
                return ob
    return objs[0]


def resolve_object_body(env, object_type):
    ob = resolve_object(env, object_type)
    return getattr(ob, "root_body", None) if ob is not None else None


def _gripper_geom_ids(env):
    ids = set()
    try:
        g = env.robots[0].gripper
        names = []
        for attr in ("contact_geoms", "important_geoms"):
            v = getattr(g, attr, None)
            if isinstance(v, dict):
                for vv in v.values():
                    names.extend(vv)
            elif v:
                names.extend(v)
        for n in names:
            try:
                ids.add(env.sim.model.geom_name2id(n))
            except Exception:
                pass
    except Exception:
        pass
    return ids


def _body_geom_ids(env, body_name):
    ids = set()
    try:
        bid = env.sim.model.body_name2id(body_name)
        for gi in range(env.sim.model.ngeom):
            if env.sim.model.geom_bodyid[gi] == bid:
                ids.add(gi)
    except Exception:
        pass
    return ids


_CONTACT_WARNED = False


def contact_signal(env, obj_model, obj_body):
    """1.0 if the gripper touches/grasps the object. Three-layer fallback:
    _check_grasp (both fingerpads, matches the segmenter) -> check_contact ->
    manual geom-pair scan. Without the chain this returns a constant and its R^2
    goes nan."""
    global _CONTACT_WARNED
    try:
        if obj_model is not None and hasattr(env, "_check_grasp"):
            return 1.0 if env._check_grasp(env.robots[0].gripper, obj_model) else 0.0
    except Exception:
        pass
    try:
        if obj_body is not None and hasattr(env, "check_contact"):
            return 1.0 if env.check_contact(env.robots[0].gripper, obj_body) else 0.0
    except Exception:
        pass
    try:
        gg = _gripper_geom_ids(env)
        og = _body_geom_ids(env, obj_body) if obj_body else set()
        if gg and og:
            d = env.sim.data
            for i in range(int(d.ncon)):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in gg and g2 in og) or (g2 in gg and g1 in og):
                    return 1.0
            return 0.0
    except Exception:
        pass
    if not _CONTACT_WARNED:
        print("[warn] contact undecidable -> filling 0")
        _CONTACT_WARNED = True
    return 0.0


# ===========================================================================
# SECTION 3 — goal resolution  (replaces infer_goal(obj[-1]))
# ===========================================================================
class GoalUnavailable(NotImplementedError):
    """Raised when the environment has no spatial goal this module knows how to
    read. Better than silently falling back to obj[-1], which is right only for
    a successful demo and wrong for every failed rollout."""


def _goal_pickplace(env, _obs_dict):
    return np.asarray(env.target_bin_placements[env.object_id], float), \
        "env.target_bin_placements[object_id]"


def _goal_nut(env, _obs_dict):
    name = type(env).__name__.lower()
    attr = "peg2_body_id" if "round" in name else "peg1_body_id"
    if not hasattr(env, attr):
        attr = "peg1_body_id" if hasattr(env, "peg1_body_id") else None
    if attr is None:
        raise GoalUnavailable("NutAssembly env exposes no peg body id")
    return np.asarray(env.sim.data.body_xpos[getattr(env, attr)], float), \
        f"sim.data.body_xpos[{attr}]"


def _goal_stack(env, obs_dict):
    """cubeA's goal is wherever cubeB is; it moves with the scene, so it must be
    re-read every reset rather than cached."""
    if "cubeB_pos" not in obs_dict:
        raise GoalUnavailable("Stack env has no cubeB_pos observable")
    return np.asarray(obs_dict["cubeB_pos"], float), "obs['cubeB_pos']"


# Checked in order; first predicate that matches wins.
GOAL_REGISTRY = [
    (lambda env: hasattr(env, "target_bin_placements"), _goal_pickplace),
    (lambda env: hasattr(env, "peg1_body_id") or hasattr(env, "peg2_body_id"),
     _goal_nut),
    (lambda env: type(env).__name__.lower().startswith("stack"), _goal_stack),
]


def resolve_goal(env, object_type=None, *, allow_fallback=False, obj=None):
    """The task's spatial goal, read from the ENV.

    allow_fallback=True re-enables the old obj[-1] behaviour, with a warning.
    It is off by default on purpose: obj[-1] equals the goal only when the
    trajectory SUCCEEDED, so using it on a policy rollout (which is exactly
    where M5 needs a goal) silently mis-cuts every phase.
    """
    if env is not None:
        od = None
        for pred, fn in GOAL_REGISTRY:
            try:
                if not pred(env):
                    continue
            except Exception:
                continue
            if od is None:
                od = _obs(env)
            return fn(env, od)

    if allow_fallback and obj is not None and len(obj):
        warnings.warn(
            "resolve_goal fell back to obj[-1]. This is the demo's final object "
            "position, NOT the task goal: it is correct only if the trajectory "
            "succeeded. Never use it for a policy rollout.", RuntimeWarning)
        return np.asarray(obj[-1], float).copy(), "FALLBACK obj[-1]"

    raise GoalUnavailable(
        f"no goal source for env={type(env).__name__ if env else None}. "
        f"Add a branch to GOAL_REGISTRY, or pass allow_fallback=True with obj= "
        f"if (and only if) this trajectory is a successful demo.")


# ===========================================================================
# SECTION 4 — torque  (not in obs, not replayable)
# ===========================================================================
def inverse_dynamics_torque(env, states, dt, n_joints=7):
    """Reconstruct joint torque from the saved states via MuJoCo inverse
    dynamics: tau = mj_inverse(qpos, qvel, qddot) with qddot central-differenced
    from qvel.

    Measured against live torque on a random-action rollout: correlation 0.74,
    total-energy ratio 0.71. It is an APPROXIMATION -- it misses actuator
    saturation and contact forces, and the finite-differenced qddot is noisy.
    Prefer torques recorded at collection time; this exists so demos gathered
    before that change are still usable.
    """
    import mujoco
    M = env.sim.model._model
    D = env.sim.data._data
    nq, nv = M.nq, M.nv
    states = np.asarray(states, float)

    qpos = states[:, 1:1 + nq]
    qvel = states[:, 1 + nq:1 + nq + nv]
    qacc = np.zeros_like(qvel)
    if len(qvel) > 2:
        qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2.0 * dt)

    out = np.zeros((len(states), n_joints))
    for t in range(len(states)):
        D.qpos[:] = qpos[t]
        D.qvel[:] = qvel[t]
        D.qacc[:] = qacc[t]
        mujoco.mj_inverse(M, D)
        out[t] = np.asarray(D.qfrc_inverse[:n_joints], float)
    return out


def resolve_torque(env, states, dt, recorded=None, mode="auto", n_joints=7):
    """-> (torque (T, n_joints) or None, source string)

    mode: 'auto'      recorded if present, else inverse dynamics
          'recorded'  recorded only; None if absent
          'inverse'   always inverse dynamics
          'none'      never produce torque (energy column stays 0)
    """
    if mode == "none":
        return None, "none"
    if recorded is not None and mode in ("auto", "recorded"):
        tq = np.asarray(recorded, float)
        if tq.ndim == 1:
            tq = tq.reshape(-1, 1)
        if len(tq) < len(states):
            # Rows align head-first (torque[i] pairs with action[i]/state[i]
            # from index 0), and collection writes exactly one row per step --
            # so a shortfall means appends were DROPPED at collection time, and
            # every index after the first drop is misaligned in a way no
            # trimming can repair. The energy column past len(tq) stays 0.
            print(f"[warn] recorded torque has {len(tq)} rows for "
                  f"{len(states)} states -- rows were dropped at collection; "
                  f"treat this demo's torque/energy as suspect")
        return tq[:len(states), :n_joints], "recorded"
    if mode == "recorded":
        return None, "none"
    if env is None:
        return None, "none"
    try:
        return inverse_dynamics_torque(env, states, dt, n_joints), "inverse_dynamics"
    except Exception as ex:
        print(f"[warn] inverse dynamics failed ({type(ex).__name__}: {ex}); "
              f"energy column will be 0")
        return None, "none"


# ===========================================================================
# SECTION 5 — the extractor  (ONE per-frame core, two entry points)
# ===========================================================================
class FrameExtractor:
    """Produces Frames from either a saved demo (M1) or a live rollout (M5).

    Both paths funnel through _frame(), which is the whole point: proposal
    Sec 6.2 needs the reward's z_t to mean the same thing at training time and
    at deployment time, and that only holds if the observation channels are
    produced by identical code.
    """

    def __init__(self, env, object_type=None, dt=None):
        self.env = env
        self.object_type = object_type
        self.dt = float(dt) if dt is not None else float(
            getattr(env, "control_timestep", 0.05))
        self.obj_model = resolve_object(env, object_type)
        self.obj_body = resolve_object_body(env, object_type)

    # -- the shared core -----------------------------------------------------
    def _frame(self, with_dynamics=True):
        """One frame from whatever state the sim is in RIGHT NOW.

        Callers are responsible for putting the sim in that state (inject +
        forward for replay, env.step for rollout). torque is read here but is
        meaningless after a state injection -- from_states overwrites it.
        """
        eef, obj, grip, eq, oq = read_frame(self.env, self.object_type)
        f = dict(eef_pos=eef, obj_pos=obj, grip=grip, eef_quat=eq, obj_quat=oq,
                 contact=contact_signal(self.env, self.obj_model, self.obj_body))
        if with_dynamics:
            d = self.env.sim.data
            f["qvel"] = np.asarray(d.qvel[:7], float).copy()
            f["torque"] = np.asarray(d.actuator_force[:7], float).copy()
        return f

    @staticmethod
    def _stack(rows):
        out = {}
        for k in rows[0]:
            out[k] = np.array([r[k] for r in rows], float)
        return out

    # -- M1: replay a saved demo ---------------------------------------------
    def from_states(self, states, actions=None, torques=None, *,
                    goal=None, torque_mode="auto", allow_goal_fallback=False):
        """Demo states -> Frames.

        The goal is resolved from the ENV once, before the loop: it is a
        property of the task, not of the trajectory. torque read during replay
        is discarded (it is identically 0) and refilled by resolve_torque.
        """
        rows = []
        for st in states:
            self.env.sim.set_state_from_flattened(st)
            self.env.sim.forward()          # never omit: obs are stale without it
            rows.append(self._frame(with_dynamics=True))
        ch = self._stack(rows)

        if goal is None:
            obj_seq = ch["obj_pos"]
            goal, gsrc = resolve_goal(self.env, self.object_type,
                                      allow_fallback=allow_goal_fallback,
                                      obj=obj_seq)
        else:
            goal, gsrc = np.asarray(goal, float), "caller-supplied"

        tq, tsrc = resolve_torque(self.env, states, self.dt,
                                  recorded=torques, mode=torque_mode)

        return Frames(eef_pos=ch["eef_pos"], obj_pos=ch["obj_pos"],
                      grip=ch["grip"], goal=goal, dt=self.dt,
                      eef_quat=ch["eef_quat"], obj_quat=ch["obj_quat"],
                      contact=ch["contact"], actions=actions,
                      qvel=ch["qvel"], torque=tq,
                      goal_source=gsrc, torque_source=tsrc,
                      meta=dict(mode="from_states", n_states=len(states)))

    # -- M5: live policy rollout ---------------------------------------------
    def from_rollout(self, policy, horizon, *, goal=None, reset=True,
                     model_xml=None):
        """Roll `policy` out and return Frames. torque IS valid here, because
        the sim was actually stepped rather than teleported.

        policy: callable(obs_dict) -> action, or an object with .predict(obs).
        """
        if reset:
            if model_xml is not None:
                reset_to_scene(self.env, model_xml)
            else:
                self.env.reset()

        if goal is None:
            goal, gsrc = resolve_goal(self.env, self.object_type)
        else:
            goal, gsrc = np.asarray(goal, float), "caller-supplied"

        act = getattr(policy, "predict", policy)
        rows, acts = [self._frame(with_dynamics=True)], []
        for _ in range(int(horizon)):
            o = _obs(self.env)
            a = act(o)
            a = a[0] if isinstance(a, tuple) else a
            a = np.asarray(a, float).ravel()
            self.env.step(a)
            rows.append(self._frame(with_dynamics=True))
            acts.append(a)
        ch = self._stack(rows)

        return Frames(eef_pos=ch["eef_pos"], obj_pos=ch["obj_pos"],
                      grip=ch["grip"], goal=goal, dt=self.dt,
                      eef_quat=ch["eef_quat"], obj_quat=ch["obj_quat"],
                      contact=ch["contact"], actions=np.array(acts, float),
                      qvel=ch["qvel"], torque=ch["torque"],
                      goal_source=gsrc, torque_source="live",
                      meta=dict(mode="from_rollout", horizon=int(horizon)))


# ===========================================================================
# SECTION 6 — selftest (pure numpy; no robosuite)
# ===========================================================================
def _synthetic_frames(T=120, dt=0.05, with_dyn=True):
    eef = np.zeros((T, 3)); obj = np.zeros((T, 3))
    eef[:, 0] = np.linspace(0.0, 0.5, T)
    eef[:, 2] = 1.0 - 0.15 * np.sin(np.linspace(0, np.pi, T))
    obj[:60] = [0.25, 0.0, 0.85]
    obj[60:, 0] = np.linspace(0.25, 0.5, T - 60)
    obj[60:, 2] = 0.85
    grip = np.zeros(T); grip[60:110] = 1.0
    con = np.zeros(T); con[60:110] = 1.0
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))
    kw = {}
    if with_dyn:
        kw = dict(qvel=np.ones((T, 7)) * 0.5, torque=np.ones((T, 7)) * 2.0,
                  torque_source="synthetic")
    return Frames(eef_pos=eef, obj_pos=obj, grip=grip, goal=np.array([0.5, 0, 0.85]),
                  dt=dt, eef_quat=quat, obj_quat=quat, contact=con,
                  actions=np.zeros((T - 1, 7)), goal_source="synthetic", **kw)


def run_selftest():
    print("=== frame_extract SELFTEST ===")
    ok = True

    fr = _synthetic_frames()
    print(fr.summary())

    # 1. Psi from the FULL frames and from the obs-only copy must be identical.
    #    This is the machine-checkable form of "phases are cut on the
    #    observation space": if any boundary feature ever reads torque or qvel,
    #    these two disagree and M5 loses the ability to segment its rollouts.
    psi_full = fs.compute_psi(fr)
    psi_obs = fs.compute_psi(fr.obs_only())
    if not np.allclose(psi_full, psi_obs):
        ok = False
        print("[FAIL] Psi changed when non-obs channels were dropped -> a "
              "boundary feature is reading torque/qvel")
    else:
        print(f"Psi(full) == Psi(obs_only)  {psi_full.shape}  "
              f"-> boundary features are observation+goal only")

    # 2. Psi is the boundary columns of the full F (single implementation).
    F = fs.compute_from_frames(fr)
    if not np.allclose(psi_full, F[:, fs.boundary_mask()]):
        ok = False; print("[FAIL] Psi != F[:, boundary_mask()]")
    else:
        print(f"Psi == F[:, boundary_mask()]   F={F.shape}")

    # 3. energy is live when torque is present, 0 when it is not.
    E = fs.index_of("energy")
    if np.allclose(F[:, E], 0.0):
        ok = False; print("[FAIL] energy is 0 despite torque being present")
    F0 = fs.compute_from_frames(fr.obs_only())
    if not np.allclose(F0[:, E], 0.0):
        ok = False; print("[FAIL] energy should be 0 without torque")
    print(f"energy: with torque {F[0, E]:.4f} / without {F0[0, E]:.4f}")

    # 4. resolve_goal must REFUSE rather than guess.
    try:
        resolve_goal(None)
        ok = False; print("[FAIL] resolve_goal(None) should raise GoalUnavailable")
    except GoalUnavailable:
        print("resolve_goal(None) -> GoalUnavailable (no silent obj[-1] guess)")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g, src = resolve_goal(None, allow_fallback=True, obj=fr.obj_pos)
        if not w or "obj[-1]" not in str(w[0].message):
            ok = False; print("[FAIL] fallback must warn")
        elif not np.allclose(g, fr.obj_pos[-1]):
            ok = False; print("[FAIL] fallback returned the wrong point")
        else:
            print(f"resolve_goal(allow_fallback=True) -> {np.round(g,3)} "
                  f"[{src}] + RuntimeWarning")

    # 5. the legacy dict shape fcm_proto / degradation_proto expect.
    d = fr.to_legacy_dict()
    if set(d) != {"eef", "obj", "grip", "eef_quat", "obj_quat", "contact"}:
        ok = False; print(f"[FAIL] legacy dict keys changed: {sorted(d)}")
    else:
        print(f"to_legacy_dict keys OK: {sorted(d)}")

    # 6. Frames tolerates a bare 4-dim OSC_POSITION demo (no quats at all).
    bare = Frames(eef_pos=fr.eef_pos, obj_pos=fr.obj_pos, grip=fr.grip,
                  goal=fr.goal, dt=fr.dt)
    if not np.allclose(fs.compute_psi(bare), psi_full):
        ok = False; print("[FAIL] Psi differs on a quat-less demo")
    else:
        print("Psi unchanged on a quat-less (OSC_POSITION) demo")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo-root", default=None,
                    help="print a Frames summary for each demo found (needs robosuite)")
    ap.add_argument("--pattern", default="demo.hdf5")
    args = ap.parse_args()

    if args.selftest or args.demo_root is None:
        run_selftest()
        return

    paths = sorted(glob(os.path.join(args.demo_root, "**", args.pattern),
                        recursive=True))
    if not paths:
        raise SystemExit(f"No '{args.pattern}' under {args.demo_root}")
    for hp in paths:
        env_info, demos = read_demo(hp)
        env = build_env(env_info)
        try:
            ex = FrameExtractor(env, env_info.get("object_type"),
                                dt=1.0 / env_info.get("control_freq", 20))
            for name, states, actions, xml, tq in demos:
                reset_to_scene(env, xml)
                fr = ex.from_states(states, actions, tq)
                print(f"{hp} :: {name}\n    {fr.summary()}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
