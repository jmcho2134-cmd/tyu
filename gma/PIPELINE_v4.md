# BfD 파이프라인 아키텍처 v4

**Structured Degradation from Suboptimal Demonstrations**
robosuite PickPlaceBread / UR5e / OSC_POSE (action_dim = 7)

---

## 0. 한 줄 요약

```text
사람은 feature 계산 함수만 정의한다
  → 여러 suboptimal 데모를 동일한 phase 로 분할한다
  → 각 phase 의 subgoal 을 관측 state 만으로 추론한다 (하드코딩 없음)
  → FCM 이 action perturbation 의 feature 결과를 학습한다
  → FCM 으로 각 phase 의 subgoal 을 방해하는 action 집합을 찾는다
  → 데모추종 정책에 그 방향 노이즈를 λ 램프로 주입해 열화 사다리를 만든다
  → simulator 실측으로 phase 별 선호 랭킹을 만든다
  → 그 랭킹으로 학습한 R_θ 를 SAC+HER 로 최대화해 데모를 넘어선다
```

---

# 1. 전체 파이프라인

```text
┌─ Stage 1 ── 데모 수집 ────────────────────────────────────────┐
│  collect_demo.py                                     [기존]   │
│  teleop, 성공하되 비효율적, N ≥ 10 (권장 20)                   │
└───────────────────────────────────────────────────────────────┘
      ↓  demos/PickPlaceBread_UR5e/demo_N/demo.hdf5
      ↓
┌─ Stage 2 ── 사람의 정의 (유일한 human prior) ─────────────────┐
│  feature_select.py                                   [기존]   │
│  FeatureSpec(name, kind, boundary, reward_input, heldout)     │
│  kind ∈ {progress, event, quality}                            │
└───────────────────────────────────────────────────────────────┘
      ↓  FEATURES / BOUNDARY / SUBGOAL_ELIGIBLE / PSI_COLUMNS
      ↓
┌─ Stage 3 ── feature 추출 ─────────────────────────────────────┐
│  extract_features.py                                 [신규 ✓] │
│  ★ robosuite 가 필요한 유일한 지점. 이후는 전부 순수 numpy     │
│  read_demo → build_env → reset_to_scene → FrameExtractor      │
└───────────────────────────────────────────────────────────────┘
      ↓  cache/demo_XXX.npz   (F, names, actions, goal, dt)
      ↓
┌─ Stage 4 ── 객체 중심 phase 분할 ─────────────────────────────┐
│  segment_hier.py + run_boundaries.py                 [구현 ✓] │
│  변위 이중임계 hysteresis: |d_og − rest| > ε_hi 탐지 후        │
│  ε_lo(노이즈 바닥)로 온셋 역추적 → t_start, t_settle           │
│  → pre | move | post  (3 phase; grasp 는 pre→move 경계의       │
│    subgoal 조건으로 옮겨짐, D11)                               │
│  5-phase 계층 모드는 --mode hier 로 ablation 보존              │
│  ▶ GATE: 라벨 시퀀스 일치율 ≥ 70%                              │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/boundaries.json  (+ _diag.json)
      ↓
┌─ Stage 5 ── subgoal 자동 추론 ────────────────────────────────┐
│  subgoal_discover.py                                 [신규 ✓] │
│  v = Var_i[φ(t_k)] / Var_{i,t}[φ]  → 조건 feature             │
│  ε = c·σ                           → 허용 오차                 │
│  persistence                       → satisfies 용 조건 분리    │
│  ▶ GATE: V2 순환성 ablation 에서 조건 생존                     │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/subgoal.json  ★ 단일 진리원
      ↓  rho.py → ρ_k(s,g), satisfies(k,s,g), phase_of(s,g)
      ↓
┌─ Stage 6 ── FCM 데이터 수집 + 학습 ───────────────────────────┐
│  fcm.py                                        [신규 ✓] │
│  동일 state 분기: Branch A = demo action                      │
│                  Branch B = demo action + perturbation        │
│  주입 종류: Gaussian / 좌표축 / 랜덤 단위벡터 / 데모 평행      │
│  출력: residual  r(t,h) = φ_pert(t+h) − φ_demo(t+h)           │
│  ▶ GATE: λ=0 → residual ≈ 0                                   │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/fcm_dataset.hdf5 → fcm_ensemble.pt
      ↓
┌─ Stage 7 ── 방해 action 집합 도출 ────────────────────────────┐
│  fcm.py (screening)                            [신규]   │
│  d*_z = argmin_d  ρ_z( φ + FCM(s, a, λ_probe·d) )             │
│  불확실성 penalty + diversity 선택 → phase 별 Top-K            │
│  ▶ GATE: Top-K recall > random                                │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/action_sets.json
      ↓   ← 여기까지가 "각 phase 에 어떻게 주입할까"의 학습
      ↓
┌─ Stage 8 ── 열화 궤적 생성 (SSRR 골격) ───────────────────────┐
│  degradation.py                                [신규]   │
│  정책 = 데모추종 closed-loop (BC 아님, λ=0 이 데모 재현)       │
│  a'_t = clip( a_demo(t) + λ·mask(t)·d*_z )                    │
│  λ 램프: bracketing → binary search → λ_max                   │
│  levels: 0 / .25 / .50 / .75 / 1.0 × λ_max                    │
│  ✗ sigmoid 캘리브레이션 없음 (Stage 9 실측이 대체)             │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/degradation.npz
      ↓
┌─ Stage 9 ── simulator 실측 + 선호 랭킹 ───────────────────────┐
│  rollout_exec.py                                     [신규]   │
│  실측: success, 최종 goal 오차, path length, slip, jerk ...    │
│  라벨 = 실측 Δ  (λ 는 생성 노브일 뿐 라벨 아님)                │
│  ▶ GATE: monotonicity + gradedness + success preservation     │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/preference.hdf5
      ↓
┌─ Stage 10 ── reward 학습 ─────────────────────────────────────┐
│  reward_learning.py                            [신규 ✓]  │
│  pairwise ranking loss, phase-conditioned R_θ(s,a,s',g,z)     │
│  입력에서 제외: action_magnitude, energy (reward_input=False)  │
│  ▶ GATE: heldout family preference accuracy                   │
└───────────────────────────────────────────────────────────────┘
      ↓  artifacts/reward.pt
      ↓
┌─ Stage 11 ── SAC + HER → BTD ────────────────────────────────┐
│  미구현                                                       │
│  r = r_env(sparse) + γΦ(s') − Φ(s),   Φ = c·(k + ρ_k)         │
│  (선택) FCM argmax 가속 — on-policy 갱신 필요                  │
│  ▶ GATE: 같은 시드에서 학습 정책 vs 데모 직접 비교             │
└───────────────────────────────────────────────────────────────┘
```

---

# 2. 모듈 맵

| Stage | 파일 | 상태 | 핵심 진입점 |
| --- | --- | --- | --- |
| 1 | `collect_demo.py` | 기존 | `save_episode_as_hdf5` (L455) |
| 2 | `feature_select.py` | 기존 | `FEATURES` (L92), `compute_from_frames` (L408) |
| 3 | `frame_extract.py` | 기존 | `FrameExtractor.from_states` (L526) |
| 3 | `extract_features.py` | **신규 ✓** | `extract()`, `synth()`, `load_cache()` |
| 4 | `segment_hier.py` | **신규 ✓** | `segment_hierarchical()` |
| 4 | `run_boundaries.py` | **신규 ✓** | `run()` |
| 4 | `phase_segment.py` | 기존, baseline | `segment_features` (L218) |
| 5 | `subgoal_discover.py` | **신규 ✓** | `discover()`, `select_features()` |
| 5 | `rho.py` | **신규 ✓** | `Subgoal.rho / satisfies / phase_of` |
| 5 | `verify_subgoal.py` | **신규 ✓** | V1~V4 |
| 6·7 | `fcm.py` | 기존, 개조 | `score_terms` (L103) ← **통합 지점** |
| 8 | `degradation.py` | 기존, 개조 | — |
| 9 | `rollout_exec.py` | 기존 | — |
| 10 | `reward_learning.py` | 기존, 개조 | — |
| 11 | — | 미구현 | — |

**통합 지점 하나**: `fcm.score_terms(sset, mode)`가 지금은 단일 데모 `phase_subgoal_set` 출력을 받습니다. `rho.Subgoal`을 받도록 시그니처만 바꾸면 Stage 6 이후 전부가 새 subgoal을 쓰게 됩니다. 내부 스코어링 로직은 손대지 않아도 됩니다.

---

# 3. 데이터 구조

## 3.1 Stage 1 — `demo.hdf5`

```text
data/                                   attrs: env, env_info, date, time
  demo_N/                               attrs: model_file (XML 전문)
    states    (T, 77)   float64         MuJoCo flattened state
    actions   (T,  7)   float64         [dx,dy,dz,drx,dry,drz,gripper]
    torques   (T,  8)   float64         선택. 리플레이 불가라 수집 시 기록
```

## 3.2 Stage 3 — `cache/demo_XXX.npz`

```python
F        : (T, N_FEATURES) float64   # feature_select.NAMES 순서 고정
names    : (N_FEATURES,)   str
actions  : (T, 7)          float64
goal     : (3,)            float64
dt       : float                      # 1 / control_freq
demo_id  : str
```

Ψ는 별도 저장하지 않습니다. `F[:, [names.index(n) for n in fs.PSI_COLUMNS]]`로 복원되고, 역방향은 불가능하기 때문입니다.

## 3.3 Stage 4 — `boundaries.json`

```json
{
  "demo_000": {
    "T": 460,
    "bounds": [266, 415],
    "labels": ["pre", "move", "post"],
    "subgoal_per_phase": [["eef_object_dist", 1.0],
                          ["object_goal_dist", 1.0],
                          ["eef_object_dist", -1.0]]
  }
}
```

`boundaries_diag.json`에 경계가 흔들릴 때 원인을 짚을 근거가 들어갑니다.

```json
{ "demo_000": {
    "level1": {"mode": "object_centric", "eps_hi": 0.005, "eps_lo": 0.001,
               "t_detect": 287, "t_start": 266, "t_settle": 415},
    "level2": {} } }
```

## 3.4 Stage 5 — `subgoal.json` ★

```json
{
  "meta": {
    "n_demos_used": 18, "demos_used": ["demo_000", ...],
    "demos_excluded": {"demo_007": ["approach","grasp","transport","retreat"]},
    "sequence_tally": {"approach | grasp | transport | place | retreat": 18},
    "canonical_labels": ["approach","grasp","transport","place","retreat"],
    "theta": 0.15, "c": 2.0, "alpha": 1.0, "max_feat": 4,
    "pad": 2, "sd_floor": 0.001, "persist_min": 0.80,
    "boundary_features_dropped": false,
    "constant_features_dropped": []
  },
  "phases": {
    "1": {
      "label": "grasp",
      "features":    ["contact", "eef_object_dist", "grasp_align"],
      "mean":        [1.0,   0.0121, 0.183],
      "std":         [0.0,   0.0038, 0.0142],
      "v":           [0.000, 0.021,  0.087],
      "type":        ["eq",  "le",   "le"],
      "thresh":      [1.0,   0.0197, 0.2114],
      "persistence": [0.95,  0.98,   0.31],
      "persistent":  [true,  true,   false],
      "boundary_t":  [153, 149, 161, ...],
      "d_start": 3.84, "sd_floor": 0.001
    }
  }
}
```

필드가 왜 필요한지:

| 필드 | 쓰는 곳 | 없으면 |
| --- | --- | --- |
| `features`, `mean`, `std` | `ρ_k` 표준화 거리 | ρ 계산 불가 |
| `type`, `thresh` | `satisfies` | 조건 판정 불가 |
| `persistent` | `satisfies`, `phase_of` | phase 라벨이 뒤로 되돌아감 |
| `v` | 진단·리포트 | 왜 선택됐는지 추적 불가 |
| `d_start` | `ρ` 정규화 | ρ 스케일 없음 |

## 3.5 Stage 6 — `FCMSample`

```python
@dataclass
class FCMSample:
    episode_id: str
    timestep: int
    phase_id: int                 # rho.phase_of 로 계산
    phase_progress: float         # rho.rho
    feature_t: np.ndarray         # (F,)
    action_t: np.ndarray          # (7,)
    perturbation: np.ndarray      # (7,)  λ·d
    window: int
    goal: np.ndarray              # (3,)
    horizons: np.ndarray          # (H,)  sec
    residual: np.ndarray          # (H, F)  φ_pert − φ_demo
    feasibility: dict             # clipping / joint limit / drop / success
```

`λ=0` 샘플을 10% 섞습니다. residual이 0이 아니면 state 분기나 리플레이가 깨진 것이므로 학습을 진행하지 않습니다.

## 3.6 Stage 7 — `action_sets.json`

```json
{ "phase_1": [
    { "candidate_id": "p1_c003",
      "direction": [0.12, -0.83, 0.41, 0.0, 0.0, 0.0, 0.0],
      "subspace": "position",
      "start_fraction": 0.25, "duration_fraction": 0.50,
      "predicted_drho": -0.34, "uncertainty": 0.06,
      "screening_score": 0.28 } ] }
```

`λ`는 여기 들어가지 않습니다. 같은 방향이라도 phase와 데모에 따라 유효 크기가 달라서 Stage 8에서 개별 탐색합니다.

## 3.7 Stage 8 — `DegradationFamily`

```python
@dataclass
class DegradationFamily:
    family_id: str
    candidate: dict               # action_sets.json 의 한 항목
    demo_id: str
    phase_id: int
    lambda_max: float
    lambda_levels: list           # [0, .25, .50, .75, 1.0] × λ_max
    trajectories: list            # 각 level 의 (states, actions, F)
    rho_endpoint: list            # 각 level 의 ρ_k(끝점)
```

## 3.8 Stage 9 — `preference.hdf5`

```text
pairs/
  i          (P,)  int    # 우세 궤적 인덱스
  j          (P,)  int    # 열세 궤적 인덱스
  phase_id   (P,)  int
  margin     (P,)  float  # 실측 Δ 차이
trajs/
  F          (M, T, N_FEATURES)
  actions    (M, T, 7)
  goal       (M, 3)
  measured/                      # ★ 라벨의 출처
    success        (M,)  bool
    final_goal_err (M,)  float
    path_length    (M,)  float
    ...
  lam        (M,)  float         # 생성 노브. 라벨 아님. 학습 입력 아님
```

## 3.9 Stage 10~11

```python
R_θ(s, a, s', g, z) -> float          # phase-conditioned
Φ(s, g) = c · (phase_of(s,g) + ρ_k(s,g))
r_total = r_env_sparse + γ·Φ(s') − Φ(s)
```

`Φ`가 상태만의 함수이므로 최적 정책이 원래 task의 최적 정책과 같습니다. 데모 수준에 상한이 걸리지 않습니다.

---

# 4. 확정된 설계 결정

| # | 결정 | 근거 |
| --- | --- | --- |
| D1 | AIRL 미사용 | 복원 대상이 "데모를 최적으로 만드는 reward". BfD와 충돌 |
| D2 | segment는 계층 (객체 → 상호작용) | flat은 병합 창이 `0.03·T`라 데모 길이가 phase 개수를 바꿈 |
| D3 | subgoal은 cross-demo 분산비 | 단일 데모는 필연/우연 구분 불가 |
| D4 | 허용 오차 = `c·σ` | 데모 산포가 곧 관용 범위. 튜닝 불필요 |
| D5 | 조건을 persistence로 분리 | 순간 마커(그리퍼 열림)가 `phase_of`를 되돌림 |
| D6 | Stage 8 정책 = 데모추종 closed-loop | λ=0이 데모 재현 + 상태 반응. 둘 다 공짜 |
| D7 | sigmoid 캘리브레이션 미사용 | Stage 9 실측이 대체. 단조성 가정도 함께 소거 |
| D8 | λ는 생성 노브, 라벨 아님 | 실측 Δ가 라벨 |
| D9 | reward 입력에서 `action_magnitude`, `energy` 제외 | 열화 연산자를 되돌리는 지름길 |
| D10 | `subgoal.json` 단일 진리원 | 모듈별 재계산은 정의 드리프트를 만듦 |
| D11 | 분할은 객체 중심 pre\|move\|post | 5-phase 계층의 b2(knee)는 물리 이벤트 없는 곡률 경계라 실데이터에서 grasp 가 transport 를 침투. 3-phase 는 경계 2개가 모두 물리 이벤트(객체 이동 시작/정착)이고, "파지됨"은 phase 가 아니라 pre→move 경계의 subgoal 조건으로 Stage 5 가 추론 |

---

# 5. 게이트

| Gate | Stage | 조건 | 실패 시 |
| --- | --- | --- | --- |
| G1 | 1 | 성공 데모 ≥ 10, action layout 일관 | 데모 추가 수집 |
| G2 | 3 | state 복원 오차 ≤ tol | 이후 전부 중단 |
| G3 | 4 | 라벨 시퀀스 일치율 ≥ 70% | L1 파라미터 조정 |
| G4 | 5 | V2 순환성 ablation에서 조건 생존 | 방법 재검토 |
| G5 | 5 | V3 phase 역행 < 5% | `c` 또는 persistence 조정 |
| G6 | 6 | λ=0 → residual ≈ 0 | 분기·리플레이 수정 |
| G7 | 7 | Top-K recall > random | FCM 재학습 |
| G8 | 9 | monotonicity + gradedness + success 유지 | candidate reject |
| G9 | 10 | heldout preference accuracy | feature ablation 재검토 |
| G10 | 11 | 학습 정책 > 데모 (같은 시드) | BTD 주장 보류 |

**G3가 지금 서 있는 자리입니다.** 여기 통과 전에는 Stage 5 이후 결과가 의미 없습니다.

---

# 6. FCM의 두 가지 사용

같은 모델을 부호만 바꿔 두 번 씁니다.

```text
Stage 7 (열화 탐색)
    d* = argmin_d  ρ_z( φ_t + FCM(s_t, a_t, λ_probe·d) )
         ← subgoal 을 가장 방해하는 방향

Stage 11 (개선 탐색, 선택)
    a* = argmax_a  R_θ( φ_t + FCM(s_t, a, 0) )
         ← R_θ 를 가장 올리는 방향
```

역할 분담을 혼동하지 않는 것이 중요합니다.

| | 아는 것 | 모르는 것 |
| --- | --- | --- |
| FCM | 물리적 결과 (feature가 어떻게 변하는가) | 그게 좋은지 나쁜지 |
| ρ / R_θ | 좋고 나쁨 | 어떤 action이 그 상태를 만드는지 |

Stage 11에서 쓰려면 **SAC rollout 데이터로 FCM을 주기적으로 갱신**해야 합니다. Stage 6의 FCM은 데모 근방에서만 정확하고 SAC는 거기서 멀어집니다. 짧은 horizon만 쓰고 on-policy로 갱신하는 것이 표준 처리입니다.

---

# 7. SSRR과의 관계

| SSRR 부품 | 본 파이프라인 |
| --- | --- |
| 정책 + 노이즈 주입 구조 | **채택** (Stage 8) |
| λ 램프 | **채택** (Stage 8) |
| 등방 Gaussian 방향 | 기각 → FCM 방향 (Stage 7) |
| ε→성능 sigmoid 캘리브레이션 | 기각 → simulator 실측 (Stage 9) |
| 랭킹 → reward | 유지 (T-REX/D-REX 공통 골격) |

논문 서술:

> SSRR의 policy-with-injected-noise 구조를 채택하되, 등방 Gaussian 방향을 phase-conditioned FCM 방향으로 대체하고, ε–성능 회귀 캘리브레이션을 simulator 실측으로 대체한다. 후자는 조작 태스크에서 grasp 실패로 인해 λ–성능 단조성이 깨지는 문제를 구조적으로 회피한다.

**Ablation A**: 등방 Gaussian(=SSRR) vs phase-conditioned structured. 랭킹 정확도, 사다리 계단 균등성, cliff 발생률로 비교합니다.

---

# 8. 사람이 정하는 것 / 데이터가 정하는 것

| 항목 | 주체 | task마다 다시? |
| --- | --- | --- |
| feature 계산 함수, kind, boundary 플래그 | 사람 | 예 (얇음) |
| observation / action space | 사람 | 예 |
| quality feature 지정 | 사람 | 예 |
| θ, c, α, persist_min | 사람 (하이퍼파라미터) | **아니오** |
| phase 개수 · 이름 · 순서 | **데이터** | 자동 |
| phase 경계 | **데이터** | 자동 |
| subgoal 조건 feature | **데이터** | 자동 |
| 허용 오차 ε | **데이터** | 자동 |
| 조건 형태 (le/ge/box/eq) | **데이터** | 자동 |
| ρ_k | subgoal에서 도출 | 자동 |
| 열화 방향 | **FCM** | 자동 |
| λ_max | **simulator** | 자동 |
| 선호 랭킹 | **simulator 실측** | 자동 |

---

# 9. 남은 작업

| 순서 | 작업 | 규모 |
| --- | --- | --- |
| 1 | Stage 3~5 실데모 실행, G3·G4·G5 판정 | 실행만 |
| 2 | `fcm.score_terms` → `rho.Subgoal` 어댑터 | 함수 1개 |
| 3 | Stage 7 screening을 ρ 기준으로 전환 | 중간 |
| 4 | Stage 8 λ 탐색 유지, sigmoid 제거 확인 | 소 |
| 5 | Stage 9 실측 라벨 경로 확인 | 소 |
| 6 | Stage 10 R_θ에 phase 조건 z 입력 | 중간 |
| 7 | Stage 11 SAC+HER + PBRS | 대 |
| 8 | BTD 비교 스크립트 (같은 시드, 정책 vs 데모) | 소, **미리** |

8번을 Stage 11 도달 전에 만들어두시는 것을 권합니다. 최종 주장의 근거인데 마지막에 만들면 급해집니다.
