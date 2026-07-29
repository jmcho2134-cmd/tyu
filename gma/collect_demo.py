#!/usr/bin/env python
"""
collect_demo.py
===============

Interactive keyboard-teleop demo collector for **any robosuite 1.5 environment
and robot**, generalized from this project's ``collect_pickplace_can.py``.

What changed vs. collect_pickplace_can.py
-----------------------------------------
``collect_pickplace_can.py`` was hard-wired to *PickPlaceCan + Panda* with an
enforced *OSC_POSITION* (4-dim, position-only) action space, so the arm could
not roll and the scene/robot could not be changed. This script instead follows
robosuite's own reference scripts:

* ``robosuite/demos/demo_random_action.py`` -> **terminal menus** to pick the
  environment, robot(s) (and the two-arm configuration) via
  ``choose_environment`` / ``choose_robots`` / ``choose_multi_arm_config`` from
  ``robosuite.utils.input_utils``.
* ``robosuite/scripts/collect_human_demonstrations.py`` -> the actual teleop
  collection loop (``collect_human_trajectory``), reused verbatim. Its HDF5
  writer is intentionally NOT reused: we save with our own ``save_episode_as_hdf5``
  so an accepted demo is written whether or not the env auto-flagged success
  (the same layout robosuite's writer produces).

Two things the user explicitly asked for
-----------------------------------------
1. **The camera moves.** Default renderer is ``mjviewer`` (the native MuJoCo
   passive viewer): drag with the mouse to orbit / pan / zoom the camera
   freely while teleoperating. ``--renderer mujoco`` (the OpenCV viewer with a
   fixed *named* ``--camera``) is still available.
2. **The arm can roll (rotate).** Default arm controller is **OSC_POSE**
   (6-DOF pose -> 7-dim action). With OSC_POSE the stock ``Keyboard`` device's
   rotation keys (e/r/y/h/p/o) are live, so the end-effector can roll/pitch/yaw.
   ``--controller OSC_POSITION`` restores the 4-dim, position-only behaviour
   that the downstream feature-bank / reward-net pipeline expects (rotation
   disabled) via the ``OSCPositionKeyboard`` shim kept from the original file.

Nothing under the robosuite install is modified; all customization lives here.

Usage (activate the conda ``robosuite`` env first)::

    python collect_demo.py                       # fully interactive menus
    python collect_demo.py --environment Lift --robots Panda
    python collect_demo.py --controller OSC_POSITION   # 4-dim, no roll (pipeline)
    python collect_demo.py --renderer mujoco --camera agentview

Quit: press ``q`` in the keyboard listener to end the current episode. After
each episode the terminal asks whether to save it (``y`` -> write demo_<N>,
``n`` -> discard and reset to re-collect). Press ``Ctrl+C`` in the terminal to
stop the whole program.
"""

import argparse
import datetime
import inspect
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite

# 1.5 API: load_composite_controller_config replaces the deprecated
# load_controller_config. Import path verified in robosuite/controllers/__init__.py.
from robosuite.controllers import load_composite_controller_config

# Reuse robosuite's own teleop collection loop (NOT its gatherer). Importing the
# module does NOT run its __main__ block. collect_human_trajectory's signature
# differs across 1.5.x patch releases (1.5.1 has no goal_update_mode; 1.5.2 adds
# it), so we call it version-robustly below via inspect.signature.
#
# We deliberately do NOT use robosuite's gather_demonstrations_as_hdf5: it writes
# an episode ONLY when env._check_success() latched during collection, so a demo
# the human wants to keep but that the env never flagged successful would be
# silently dropped. Instead we save the just-collected episode ourselves (see
# save_episode_as_hdf5) whenever the user answers 'y', regardless of the flag.
from robosuite.scripts.collect_human_demonstrations import collect_human_trajectory

# Interactive terminal menus, exactly as robosuite's demo_random_action.py uses.
from robosuite.utils.input_utils import (
    choose_environment,
    choose_multi_arm_config,
    choose_robots,
)

# Stock keyboard device (used as-is for OSC_POSE so rotation keys are live;
# subclassed below for the position-only OSC_POSITION mode).
from robosuite.devices import Keyboard

# The built-in collection wrappers we reuse verbatim.
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


# ---------------------------------------------------------------------------
# Torque recording: the one channel a state replay cannot give back
# ---------------------------------------------------------------------------
class TorqueRecordingWrapper(DataCollectionWrapper):
    """DataCollectionWrapper that also logs sim.data.actuator_force each step.

    Why this exists. The flattened MuJoCo state robosuite saves is
    [time, qpos, qvel] -- it carries no `ctrl`. So when the pipeline later
    replays a demo with set_state_from_flattened + sim.forward(), MuJoCo
    recomputes actuator_force from a ctrl that reset() zeroed, and the torque
    reads exactly 0 at every frame (measured: live torque std ~25 N.m per joint,
    replayed std 0.000). qpos/qvel replay perfectly; torque does not replay at
    all. Any energy / effort feature therefore has to be captured HERE, at
    collection time, or reconstructed afterwards by inverse dynamics
    (frame_extract.inverse_dynamics_torque, correlation ~0.74 against live).

    The torques land in a sidecar npz per flush rather than inside the
    DataCollectionWrapper's own state_*.npz, so nothing about robosuite's file
    format changes and an older reader still parses the episode directory.
    """

    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        super().__init__(env, directory, collect_freq=collect_freq,
                         flush_freq=flush_freq)
        self.torques = []

    def step(self, action):
        ret = super().step(action)
        try:
            self.torques.append(
                np.array(self.env.sim.data.actuator_force, dtype=float))
        except Exception as ex:
            # Never let logging break collection -- but say it at least once:
            # torque[i] must pair with action[i], so one dropped append silently
            # misaligns every later torque row of this episode.
            if not getattr(self, "_torque_warned", False):
                self._torque_warned = True
                print(f"[warn] torque logging failed ({type(ex).__name__}: {ex})"
                      f"; torque rows are misaligned from this step on")
        return ret

    def _flush(self):
        tq = (np.array(self.torques, dtype=float) if self.torques
              else np.zeros((0, 0)))
        ep_dir = self.ep_directory       # cleared by super()._flush() in some versions
        super()._flush()
        if ep_dir is not None:
            np.savez(os.path.join(ep_dir,
                                  "torque_{}.npz".format(
                                      str(time.time()).replace(".", "_"))),
                     torques=tq)
        self.torques = []


def load_episode_torques(ep_dir):
    """Concatenate the torque sidecars written for one episode, in flush order.
    Returns None when the episode predates torque recording."""
    files = sorted(glob(os.path.join(ep_dir, "torque_*.npz")))
    if not files:
        return None
    chunks = [np.load(f)["torques"] for f in files]
    chunks = [c for c in chunks if c.size]
    return np.concatenate(chunks, axis=0) if chunks else None


# ---------------------------------------------------------------------------
# Custom keyboard device: position-only (OSC_POSITION) teleoperation
# ---------------------------------------------------------------------------
class OSCPositionKeyboard(Keyboard):
    """Keyboard device that emits a 3-DOF position-only arm delta.

    Only used when ``--controller OSC_POSITION`` is selected. It bypasses the
    stock ``Device.input2action`` (which asserts the arm controller is
    OSC_POSE/JOINT_POSITION and builds a 6-DOF delta) so we can drive an
    OSC_POSITION arm. Rotation keys (e/r/y/h/p/o) are intentionally ignored in
    this mode -- i.e. the arm does NOT roll. Everything else (pynput listener,
    key bindings, gripper toggle, reset key) is inherited from Keyboard.

    Kept identical to collect_pickplace_can.py so demos collected in
    OSC_POSITION mode remain byte-for-byte compatible with the downstream
    4-dim pipeline.
    """

    # Signature must tolerate the collection loop calling
    # device.input2action(goal_update_mode=...) (1.5.2) or with no kwargs (1.5.1).
    def input2action(self, mirror_actions=False, goal_update_mode="target"):
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        state = self.get_controller_state()
        dpos = state["dpos"]
        raw_drotation = state["raw_drotation"]
        grasp = state["grasp"]
        reset = state["reset"]

        # A reset (the 'q' key) is signalled by returning None -> ends episode.
        if reset:
            return None

        # Reproduce the stock device's per-teleop scaling so "feel" matches the
        # OSC_POSE keyboard: dpos is scaled (*75) and clipped to [-1, 1] inside
        # _postprocess_device_outputs. drotation is computed then discarded.
        drotation = raw_drotation[[1, 0, 2]]
        drotation[2] = -drotation[2]
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)

        # Map gripper toggle state -> +1 (closed) / -1 (open), as the stock device does.
        grasp = 1 if grasp else -1

        ac_dict = {}
        # Fill zero deltas for every arm so create_action_vector has an entry for
        # each controlled arm part; the active arm is overwritten below.
        for arm in robot.arms:
            ctrl = robot.part_controllers[arm]
            # Guard: this device only makes sense for a 3-DOF OSC_POSITION arm.
            assert ctrl.name == "OSC_POSITION", (
                "OSCPositionKeyboard only supports OSC_POSITION arms; got "
                f"'{ctrl.name}' for arm '{arm}'. Use the stock Keyboard for OSC_POSE."
            )
            ac_dict[f"{arm}_delta"] = np.zeros(ctrl.control_dim)  # control_dim == 3
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        # Active arm: 3-DOF position delta only (rotation dropped), clipped to [-1, 1].
        active_ctrl = robot.part_controllers[active_arm]
        ac_dict[f"{active_arm}_delta"] = np.clip(dpos[: active_ctrl.control_dim], -1.0, 1.0)

        gripper = robot.gripper[active_arm]
        if hasattr(gripper, "grasp_qpos"):
            # Some grippers map the discrete grasp signal to a qpos target.
            ac_dict[f"{active_arm}_gripper"] = getattr(gripper, "grasp_qpos")[grasp]
        else:
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper.dof)

        return ac_dict


# ---------------------------------------------------------------------------
# Controller config: choose the arm controller (OSC_POSE default -> rolling)
# ---------------------------------------------------------------------------
def make_arm_controller_config(arm_controller, robot):
    """Return a BASIC composite-controller config with the arm part(s) set to
    ``arm_controller``.

    * ``OSC_POSE``      -> 6-DOF pose command -> 7-dim action (roll enabled).
    * ``OSC_POSITION``  -> 3-DOF position command -> 4-dim action (no roll).

    robosuite 1.5's ``load_composite_controller_config`` FLATTENS
    ``body_parts.arms`` into ``body_parts['right'] / ['left']`` (there is no
    'arms' key in the returned dict). We iterate those arm parts and swap the
    type. For OSC_POSITION we must ALSO resize output_max/output_min to length 3,
    because the BASIC (OSC_POSE) arm keeps length-6 output arrays and robosuite's
    ``nums2array`` does NOT truncate them to control_dim -> a shape mismatch
    during action scaling if left at length 6.
    """
    # BASIC composite controller (generic). load_composite_controller_config
    # accepts a composite name ("BASIC") or a .json path -- NOT an arm-part name.
    config = load_composite_controller_config(controller="BASIC", robot=robot)

    if arm_controller == "OSC_POSITION":
        out_max, out_min = [0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]
    elif arm_controller == "OSC_POSE":
        # Leave OSC_POSE at its BASIC defaults (length 6); nothing to resize.
        out_max = out_min = None
    else:
        out_max = out_min = None
        print(f"[warn] unrecognized arm controller '{arm_controller}'; not resizing output_max/min.")

    n_overridden = 0
    for part_name, part in config["body_parts"].items():
        # Only touch arm parts (they start out as OSC_POSE in BASIC).
        if isinstance(part, dict) and part.get("type") in ("OSC_POSE", "OSC_POSITION"):
            part["type"] = arm_controller
            if out_max is not None:
                part["output_max"] = list(out_max)
                part["output_min"] = list(out_min)
            n_overridden += 1

    if n_overridden == 0:
        print("[warn] no OSC arm parts found in BASIC config to override.")
    return config


# ---------------------------------------------------------------------------
# Terminal menus: environment / robot(s), mirroring demo_random_action.py
# ---------------------------------------------------------------------------
def choose_options_interactively(args):
    """Resolve env_name / robots / env_configuration.

    If the user passed --environment / --robots on the CLI we honour them;
    otherwise we fall back to robosuite's interactive terminal menus, following
    the exact branching used by robosuite/demos/demo_random_action.py so that
    two-arm and humanoid environments pick sensible robots.
    """
    options = {}

    # print welcome info (same as demo_random_action.py)
    print("Welcome to robosuite v{}!".format(suite.__version__))
    if hasattr(suite, "__logo__"):
        print(suite.__logo__)

    # --- environment ---
    if args.environment is not None:
        options["env_name"] = args.environment
    else:
        options["env_name"] = choose_environment()

    # --- robot(s), with the multi-arm / humanoid branching from the demo ---
    if args.robots:
        # CLI override: one or more robot names given explicitly.
        options["robots"] = args.robots if len(args.robots) > 1 else args.robots[0]
        if "TwoArm" in options["env_name"] and args.env_configuration is not None:
            options["env_configuration"] = args.env_configuration
    elif "TwoArm" in options["env_name"]:
        # Choose env config and add it to options.
        options["env_configuration"] = (
            args.env_configuration if args.env_configuration is not None else choose_multi_arm_config()
        )
        # A bimanual config -> Baxter; otherwise the user picks two single arms.
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []
            print("A multiple single-arm configuration was chosen.\n")
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    return options


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def build_env(args, options, controller_config):
    """Create the (unwrapped) env, mirroring the stock collectors' suite.make.

    ``render_camera`` names the initial camera; with ``--renderer mjviewer`` you
    can then move that camera freely with the mouse. robosuite auto-enables the
    offscreen renderer only when needed, so has_offscreen_renderer=False is fine
    for on-screen teleop.
    """
    # first robot name (str) -> used to load the composite controller defaults.
    env = suite.make(
        **options,
        controller_configs=controller_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
    )

    # env.action_dim is None until the first reset(); env.action_spec is derived
    # from robot.action_limits and is valid immediately after make(). Use it.
    action_dim = int(env.action_spec[0].shape[0])
    arm_part = env.robots[0].part_controllers.get(args.arm)
    arm_controller_name = arm_part.name if arm_part is not None else "<unknown>"

    return env, action_dim, arm_controller_name


# ---------------------------------------------------------------------------
# Pretty launch banner
# ---------------------------------------------------------------------------
def print_launch_banner(args, options, out_dir, action_dim, arm_controller_name):
    rolling = arm_controller_name == "OSC_POSE"
    controls = [
        "  arrow keys  : move end-effector in x (up/down) and y (left/right)",
        "  . / ;       : move end-effector down / up (z)",
        "  spacebar    : toggle gripper open/close",
        "  q           : end the current episode (종료 후 저장 여부를 y/n 로 물어봄)",
    ]
    if rolling:
        controls.append("  e/r y/h p/o : ROLL / PITCH / YAW the end-effector  (OSC_POSE 회전키 활성)")
    else:
        controls.append("  (rotation keys e/r/y/h/p/o are IGNORED in OSC_POSITION mode -> 팔 롤링 없음)")

    print("\n" + "=" * 72)
    print(" robosuite keyboard demo collector  (환경/로봇 선택 + 카메라 이동 + 팔 롤링)")
    print("=" * 72)
    print(f" output directory : {out_dir}")
    print(f" environment      : {options['env_name']}")
    print(f" robot(s)         : {options.get('robots')}")
    if "env_configuration" in options:
        print(f" arm config       : {options['env_configuration']}")
    print(f" renderer / camera: {args.renderer} / {args.camera}")
    if args.renderer == "mjviewer":
        print("                    (마우스로 카메라 orbit/pan/zoom -> '카메라 이동' 가능)")
    print(f" controller       : {arm_controller_name}  ->  action_dim = {action_dim}")
    if rolling:
        print("                    (3 EE pos + 3 EE rot deltas + 1 gripper) -> 팔 롤링 O")
    elif action_dim == 4:
        print("                    (3 EE position deltas + 1 gripper) -> 팔 롤링 X, 4-dim 파이프라인 호환")
    print(f" control_freq     : {args.control_freq} Hz   (loop cap max_fr = {args.max_fr})")
    print(f" pos/rot sens.    : {args.pos_sensitivity} / {args.rot_sensitivity}")
    print("-" * 72)
    print(" keyboard controls:")
    for line in controls:
        print(line)
    print("-" * 72)
    print(" 키보드 리스너는 전역 pynput 훅이라 렌더 창에 포커스가 없어도 키가 잡힙니다.")
    print(" 프로그램 전체를 멈추려면 이 터미널에서 Ctrl+C 를 누르세요.")
    print(" 매 에피소드가 끝나면(q 또는 성공) 터미널에서 저장 여부(y/n)를 물어봅니다.")
    print("   y -> 저장 후 다음 데모 계속   /   n -> 저장 안 하고 리셋 후 재수집")
    print(f" 저장 위치: {os.path.join(out_dir, 'demo_<N>', 'demo.hdf5')}  (승인한 데모마다 번호별 폴더)")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Per-demo helpers: numbered demo_<N> folders + ask-to-save prompt
# (kept from collect_pickplace_can.py so each accepted demo lands in its own
#  demos/<Env>_<Robot>/demo_<N>/demo.hdf5)
# ---------------------------------------------------------------------------
def list_episode_dirs(tmp_directory):
    """Return the set of 'ep_*' episode sub-directory names in tmp_directory."""
    if not os.path.isdir(tmp_directory):
        return set()
    return {d for d in os.listdir(tmp_directory) if d.startswith("ep_")}


def newest_episode_dir(tmp_directory, before):
    """Return the full path of the ep_* dir created since the `before` snapshot.

    collect_human_trajectory() creates exactly one new episode directory per
    call (via DataCollectionWrapper._on_first_interaction), so the set
    difference against a pre-call snapshot pins down the just-collected episode.
    Returns None if nothing new was written (e.g. the user quit before taking a
    single step, so no interaction was ever logged).
    """
    new = sorted(list_episode_dirs(tmp_directory) - before)
    if not new:
        return None
    return os.path.join(tmp_directory, new[-1])


def episode_successful(ep_dir):
    """True if any state_*.npz in ep_dir recorded a successful=True flag.

    Mirrors gather_demonstrations_as_hdf5's own success criterion (OR of the
    per-flush `successful` flags) so what we ask about matches what would be
    written to the hdf5.
    """
    ok = False
    for state_file in glob(os.path.join(ep_dir, "state_*.npz")):
        dic = np.load(state_file, allow_pickle=True)
        ok = ok or bool(dic["successful"])
    return ok


def save_episode_as_hdf5(ep_dir, out_dir, env_info):
    """Write a single collected episode (ep_dir) into out_dir/demo.hdf5.

    This is a trimmed copy of robosuite's gather_demonstrations_as_hdf5 for ONE
    episode, with the crucial difference that it does NOT gate on the success
    flag: whatever the human accepted with 'y' is saved. The resulting file has
    the same layout robosuite produces (a single 'demo_1' group with `states`
    and `actions`, plus `data` attrs incl. env_info), so downstream tooling reads
    it unchanged. Returns (num_states, was_flagged_successful).
    """
    state_paths = os.path.join(ep_dir, "state_*.npz")
    states, actions = [], []
    env_name = None
    success = False
    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file, allow_pickle=True)
        env_name = str(dic["env"])
        states.extend(dic["states"])
        for ai in dic["action_infos"]:
            actions.append(ai["actions"])
        success = success or bool(dic["successful"])

    if len(states) == 0:
        return 0, success

    # Drop the trailing state: DataCollectionWrapper records the state AFTER the
    # action, so there is one extra state at the end (same fix robosuite applies).
    del states[-1]
    assert len(states) == len(actions), (len(states), len(actions))

    # Torques are logged once per step, so they align with `actions`, i.e. with
    # `states` after the trailing state is dropped. Trim rather than assert:
    # a mismatch should not throw away a demo the human already accepted.
    torques = load_episode_torques(ep_dir)
    if torques is not None and len(torques):
        n = min(len(states), len(torques))
        if n != len(torques):
            print(f"[warn] torque rows {len(torques)} != states {len(states)}; "
                  f"trimming to {n}")
        torques = torques[:n]
    else:
        torques = None
        print("[warn] no torque sidecar in this episode -> the energy feature "
              "will fall back to inverse dynamics downstream.")

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")
        ep_grp = grp.create_group("demo_1")  # one demo per demo_<N>/ folder
        # model.xml is written by DataCollectionWrapper on first interaction, so
        # it should always exist; guard anyway so a missing xml doesn't crash.
        xml_path = os.path.join(ep_dir, "model.xml")
        if os.path.exists(xml_path):
            with open(xml_path, "r") as xml_f:
                ep_grp.attrs["model_file"] = xml_f.read()
        else:
            print(f"[warn] model.xml not found in {ep_dir}; saving without model_file attr.")
        ep_grp.create_dataset("states", data=np.array(states))
        ep_grp.create_dataset("actions", data=np.array(actions))
        if torques is not None:
            # Optional dataset: frame_extract.read_demo returns None when it is
            # absent, so demos collected before this change still load.
            ep_grp.create_dataset("torques", data=np.array(torques))

        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info

    return len(states), success


def next_demo_dir(root):
    """Return the path to the next unused 'demo_<N>' folder under root.

    Scans root for existing 'demo_<int>' folders and returns 'demo_<max+1>'
    (or 'demo_1' if none exist). The folder is NOT created here so that quitting
    without saving leaves no empty stub behind.
    """
    os.makedirs(root, exist_ok=True)
    used = []
    for name in os.listdir(root):
        if name.startswith("demo_") and name[len("demo_"):].isdigit():
            used.append(int(name[len("demo_"):]))
    n = (max(used) + 1) if used else 1
    return os.path.join(root, f"demo_{n}")


def ask_keep_demo():
    """Prompt (Korean) whether to save the just-collected demo.

    Loops until a clear yes/no is given. Returns True to keep+save, False to
    discard and re-collect. Ctrl+D (EOF) is treated as 'no' (re-collect).
    """
    while True:
        try:
            ans = input(">> 이번 데모를 저장하겠습니까? [y/n]: ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("   y(저장) 또는 n(리셋 후 재수집) 으로 답해주세요.")


def ask_num_demos():
    """Prompt (Korean) for how many demos to collect. Returns a positive int.

    Loops until a positive integer is entered. Ctrl+D (EOF) defaults to 1.
    """
    while True:
        try:
            ans = input(">> 몇 개의 데모를 수집하시겠습니까? (정수 입력): ").strip()
        except EOFError:
            print("   (입력 없음 -> 1개로 진행합니다.)")
            return 1
        if ans.isdigit() and int(ans) > 0:
            return int(ans)
        print("   1 이상의 정수를 입력해주세요.")


# ---------------------------------------------------------------------------
# Version-robust wrapper around robosuite's collect_human_trajectory
# ---------------------------------------------------------------------------
def run_one_episode(env, device, args):
    """Call robosuite's collect_human_trajectory, passing goal_update_mode only
    if this robosuite version's signature accepts it (1.5.2+)."""
    params = inspect.signature(collect_human_trajectory).parameters
    kwargs = {}
    if "goal_update_mode" in params:
        kwargs["goal_update_mode"] = args.goal_update_mode
    collect_human_trajectory(env, device, args.arm, args.max_fr, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive keyboard-teleop demo collection for any robosuite env/robot "
                    "(camera movable, arm rolling via OSC_POSE)."
    )
    parser.add_argument("--directory", type=str, default=os.path.join(".", "demos"),
                        help="Root directory to store collected demos.")
    parser.add_argument("--environment", type=str, default=None,
                        help="robosuite environment name. Omit to pick from a terminal menu.")
    parser.add_argument("--robots", nargs="+", type=str, default=None,
                        help="Robot model(s). Omit to pick from a terminal menu.")
    parser.add_argument("--num-demos", type=int, default=None,
                        help="How many demos to collect (saved with 'y'). Omit to be asked in the "
                             "terminal. Collection stops once this many demos are saved.")
    parser.add_argument("--env-configuration", type=str, default=None,
                        help="Two-arm configuration (e.g. 'bimanual'). Omit to pick from a menu "
                             "for TwoArm envs.")
    parser.add_argument("--device", type=str, default="keyboard",
                        help="Teleop device (only 'keyboard' is wired here).")
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"],
                        help="'mjviewer' = native viewer with a FREE, mouse-movable camera (default); "
                             "'mujoco' = OpenCV viewer locked to a named --camera.")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Initial camera view.")
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        choices=["OSC_POSE", "OSC_POSITION"],
                        help="Arm controller. OSC_POSE (default) enables arm rolling (7-dim). "
                             "OSC_POSITION forces the 4-dim position-only pipeline action (no roll).")
    parser.add_argument("--control-freq", type=int, default=20,
                        help="Environment control frequency (Hz).")
    parser.add_argument("--max-fr", type=int, default=20,
                        help="Cap the collection loop to this many frames/sec.")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0,
                        help="Position input sensitivity.")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0,
                        help="Rotation input sensitivity (used only in OSC_POSE mode).")
    parser.add_argument("--arm", type=str, default="right",
                        help="Which arm to control (single-arm robots have only 'right').")
    parser.add_argument("--no-torque", action="store_true",
                        help="Do NOT record joint torques. Torque is absent from "
                             "the saved MuJoCo state and cannot be recovered by "
                             "replay, so without it the downstream energy feature "
                             "must fall back to inverse dynamics or stays 0.")
    parser.add_argument("--goal-update-mode", type=str, default="target", choices=["target", "achieved"],
                        help="Passed through to device.input2action on robosuite versions that accept it.")
    args = parser.parse_args()

    if args.device != "keyboard":
        raise SystemExit(
            f"--device '{args.device}' is not supported by this utility. Only 'keyboard' is wired here. "
            "Extend it with robosuite's SpaceMouse device if needed."
        )

    # --- interactive (or CLI) environment/robot selection ---
    options = choose_options_interactively(args)

    # --- how many demos to collect (asked in the terminal unless given on CLI) ---
    num_demos = args.num_demos if (args.num_demos and args.num_demos > 0) else ask_num_demos()

    # --- controller config: set the chosen arm controller ---
    # load_composite_controller_config wants a single robot name; use the first.
    first_robot = options["robots"][0] if isinstance(options["robots"], (list, tuple)) else options["robots"]
    controller_config = make_arm_controller_config(args.controller, first_robot)

    # --- build env ---
    env, action_dim, arm_controller_name = build_env(args, options, controller_config)

    # If the user explicitly asked for the 4-dim pipeline action, fail loudly on
    # a mismatch (mirrors collect_pickplace_can.py's hard check).
    if args.controller == "OSC_POSITION" and action_dim != 4:
        env.close()
        raise SystemExit(
            "\n=================== ACTION-SPACE CHECK FAILED ===================\n"
            f"Expected a 4-dim action (OSC_POSITION: 3 pos + 1 gripper) but got "
            f"action_dim = {action_dim} (arm controller: {arm_controller_name}).\n"
            "================================================================="
        )

    # --- output root: ./demos/<Env>_<Robot>/  (each accepted demo -> demo_<N>/) ---
    # e.g. demos/PickPlaceCan_Panda/demo_1/demo.hdf5, demos/PickPlaceMilk_Panda/demo_1/...
    # The Module-1 pipeline's demo_root is ./demos and it discovers these
    # recursively, so demos collected here feed build_feature_bank.py directly.
    robots_tag = "-".join(options["robots"]) if isinstance(options["robots"], (list, tuple)) else options["robots"]
    run_name = "{}_{}".format(options["env_name"], robots_tag)
    root_dir = os.path.join(args.directory, run_name)
    os.makedirs(root_dir, exist_ok=True)

    # --- env metadata for the hdf5 (so downstream code follows THIS demo) ---
    # We also record object_type for PickPlace-family envs (PickPlaceCan -> "can",
    # PickPlaceMilk -> "milk", ...) so build_feature_bank.py picks the right object
    # per demo without any hard-coded "can". Non-PickPlace envs leave it null.
    env_name = options["env_name"]
    object_type = None
    if env_name.startswith("PickPlace") and len(env_name) > len("PickPlace"):
        object_type = env_name[len("PickPlace"):].lower()   # PickPlaceMilk -> "milk"
    env_info_dict = {
        "env_name": env_name,
        "robots": options["robots"] if isinstance(options["robots"], (list, tuple)) else [options["robots"]],
        "controller_configs": controller_config,
        "action_dim": action_dim,
        "arm_controller": arm_controller_name,
        "control_freq": args.control_freq,
        # Recorded so downstream never has to guess whether the energy feature
        # is real, inverse-dynamics-reconstructed, or identically zero.
        "torque_recorded": (not args.no_torque),
    }
    if object_type is not None:
        env_info_dict["object_type"] = object_type
    env_info = json.dumps(env_info_dict)

    # --- wrap: Visualization -> DataCollection (as in the stock collector) ---
    # TorqueRecordingWrapper is DataCollectionWrapper plus a torque sidecar; see
    # its docstring for why torque cannot simply be replayed back out later.
    env = VisualizationWrapper(env)
    tmp_directory = os.path.join("/tmp", "rs_collect_{}".format(str(time.time()).replace(".", "_")))
    env = (DataCollectionWrapper(env, tmp_directory) if args.no_torque
           else TorqueRecordingWrapper(env, tmp_directory))

    # --- keyboard device: stock Keyboard for OSC_POSE (rolling), shim for OSC_POSITION ---
    if args.controller == "OSC_POSITION":
        device = OSCPositionKeyboard(
            env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
    else:
        device = Keyboard(
            env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )

    print_launch_banner(args, options, root_dir, action_dim, arm_controller_name)

    # --- collection loop: collect exactly `num_demos` accepted demos ---
    # Each iteration runs a single episode (collect_human_trajectory resets the
    # env at its start and calls env.close() when the episode ends -- via 'q' or
    # a 10-step success hold). We then:
    #   * if no data was recorded (quit before moving) -> re-collect silently.
    #   * otherwise ALWAYS prompt to save (independent of the env's auto-success
    #     detection, so the prompt always appears):
    #       'y' -> save this one episode to the NEXT demos/<Env>_<Robot>/demo_<N>/
    #              demo.hdf5 (one demo per folder); count it toward num_demos.
    #       'n' -> discard it; the env resets on the next loop = re-collect.
    # Only accepted ('y') demos count, so a bad take that you reject does not use
    # up one of your num_demos. The loop ends once `saved` reaches num_demos.
    #
    # IMPORTANT: we do NOT delete the episode folders under tmp_directory. On the
    # next env.reset(), DataCollectionWrapper flushes any leftover buffer into the
    # PREVIOUS episode's folder; deleting that folder here would make the next
    # reset write into a missing path -> FileNotFoundError. Each demo_<N>/demo.hdf5
    # still holds exactly one demo because we save one specific ep_dir per accept.
    # tmp_directory lives under /tmp and is best-effort cleaned on exit.
    print(f"[info] 이번 세션에서 {num_demos}개의 데모를 수집합니다.\n")
    saved = 0
    try:
        while saved < num_demos:
            print(f"===== 데모 {saved + 1}/{num_demos} 수집 시작 "
                  f"(창에서 조작, 끝나면 q) =====")
            before = list_episode_dirs(tmp_directory)
            run_one_episode(env, device, args)

            ep_dir = newest_episode_dir(tmp_directory, before)
            if ep_dir is None:
                print("[info] 이번 에피소드에서 기록된 데이터가 없습니다. 다시 수집합니다.\n")
                continue

            # Informational only: whether the env auto-detected task success.
            auto_ok = episode_successful(ep_dir)
            print("[info] 이번 에피소드 task 자동 성공 감지: "
                  f"{'성공(success)' if auto_ok else '미감지(not flagged)'}")

            if ask_keep_demo():
                out_dir = next_demo_dir(root_dir)          # demos/<Env>_<Robot>/demo_<N>
                os.makedirs(out_dir, exist_ok=True)
                n_states, _ = save_episode_as_hdf5(ep_dir, out_dir, env_info)
                if n_states == 0:
                    # No usable transitions -> remove the empty stub folder, retry.
                    shutil.rmtree(out_dir, ignore_errors=True)
                    print("[info] 저장할 스텝이 없어 저장을 건너뜁니다. 다시 수집합니다.\n")
                    continue
                saved += 1
                print(f"\n[saved {saved}/{num_demos}] {os.path.join(out_dir, 'demo.hdf5')}  "
                      f"(states={n_states})\n")
            else:
                print("[info] 저장하지 않고 리셋 후 다시 수집합니다.\n")

        print(f"[done] 목표한 {num_demos}개 데모 수집 완료. 저장 위치: {root_dir}")
    except KeyboardInterrupt:
        print(f"\n[done] 사용자에 의해 중단되었습니다. ({saved}/{num_demos} 저장) "
              f"저장 위치: {root_dir}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        # best-effort cleanup of the raw episode dumps under /tmp
        shutil.rmtree(tmp_directory, ignore_errors=True)


if __name__ == "__main__":
    main()