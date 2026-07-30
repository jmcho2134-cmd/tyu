PS C:\Users\Administrator\Desktop\tyu-main\gma> python .\subgoal_discover.py
[discover] 20 demos, canonical = pre | move | post
  phase 0 [pre      ] d_start=9.3221   contact eq 1*, eef_object_dist le 0.01809*
  phase 1 [move     ] d_start=6.7393   object_goal_dist le 0.0464*, eef_object_dist box 0.008994
  phase 2 [post     ] d_start=4.5137   contact eq 0*, gripper_open le 0.9894*, object_goal_dist box 0.002189*, eef_object_dist ge 0.1398*
  (* = persistent -> phase_of/satisfies 에 사용)
[out] ./artifacts/subgoal.json

==============================================================================
추론된 SUBGOAL  (Stage 5 산출물 · 단일 진리원)
==============================================================================
데모 20편 사용 / 제외 없음      canonical: pre | move | post
하이퍼파라미터: theta=0.15  c=2.0  alpha=1.0  max_feat=4  persist_min=0.8

── phase 0  [pre]  ─────────────────────────────────────────────────────── 경계 t=[211, 217, 248, 208, 215, 145, 211, 156, 153, 102, 234, 192, 198, 124, 214, 190, 196, 204, 201, 202]
  달성 조건 (AND — 하나라도 깨지면 subgoal 미달):
    ★ contact == 1                       v=0.0000  persist=0.84  scale=0.5
    ★ eef_object_dist <= 0.01809         v=0.0019  persist=0.80  scale=0.024375
    ★ = persistent (2/2) → phase_of · satisfies 판정에 사용
    satisfies(0) ≡ contact == 1  AND  eef_object_dist <= 0.01809
  ρ 정규화: d_start=9.3221 (RMS·전역)   d_start_cond=[1.0, 12.676] (조건별)
  민감도 (달성 상태 ρ=1.000 / ρ_worst=1.000). 각 조건을 'phase 시작과 같은 수준'으로
           위반시킨다 — 조건들 사이에 동등하게 심각한 사건이다:
    contact            Δρ=-0.152   Δρ_worst=-0.841   satisfies→False
    eef_object_dist    Δρ=-0.988   Δρ_worst=-0.841   satisfies→False
    → 동등 심각도 조건 2개의 편차:  ρ 6.5배  vs  ρ_worst 1.0배  (1.0 에 가까울수록 공정)

── phase 1  [move]  ────────────────────────────────────────────────────── 경계 t=[360, 369, 374, 369, 331, 332, 363, 316, 330, 354, 404, 334, 385, 251, 357, 334, 293, 314, 408, 364]
  달성 조건 (AND — 하나라도 깨지면 subgoal 미달):
    ★ object_goal_dist <= 0.0464         v=0.0001  persist=0.91  scale=0.039412
      |eef_object_dist − 0.009855| <= 0.008994 v=0.0021  persist=0.61  scale=0.024375
    ★ = persistent (1/2) → phase_of · satisfies 판정에 사용
    satisfies(1) ≡ object_goal_dist <= 0.0464
  ρ 정규화: d_start=6.7393 (RMS·전역)   d_start_cond=[9.454, 0.945] (조건별)
  민감도 (달성 상태 ρ=1.000 / ρ_worst=1.000). 각 조건을 'phase 시작과 같은 수준'으로
           위반시킨다 — 조건들 사이에 동등하게 심각한 사건이다:
    object_goal_dist   Δρ=-1.000   Δρ_worst=-0.841   satisfies→False
    eef_object_dist    Δρ=-0.138   Δρ_worst=-0.841   satisfies→False
    → 동등 심각도 조건 2개의 편차:  ρ 7.2배  vs  ρ_worst 1.0배  (1.0 에 가까울수록 공정)

── phase 2  [post]  ────────────────────────────────────────────────────── 경계 t=[400, 403, 414, 398, 367, 368, 400, 355, 366, 393, 464, 387, 419, 311, 413, 363, 335, 367, 432, 390]
  달성 조건 (AND — 하나라도 깨지면 subgoal 미달):
    ★ contact == 0                       v=0.0000  persist=1.00  scale=0.5
    ★ gripper_open <= 0.9894             v=0.0000  persist=1.00  scale=0.1524
    ★ |object_goal_dist − 0.04339| <= 0.002189 v=0.0000  persist=0.95  scale=0.039412
    ★ eef_object_dist >= 0.1398          v=0.0017  persist=1.00  scale=0.024375
    ★ = persistent (4/4) → phase_of · satisfies 판정에 사용
    satisfies(2) ≡ contact == 0  AND  gripper_open <= 0.9894  AND  |object_goal_dist − 0.04339| <= 0.002189  AND  eef_object_dist >= 0.1398
  ρ 정규화: d_start=4.5137 (RMS·전역)   d_start_cond=[0.75, 6.198, 0.304, 5.33] (조건별)
  민감도 (달성 상태 ρ=1.000 / ρ_worst=1.000). 각 조건을 'phase 시작과 같은 수준'으로
           위반시킨다 — 조건들 사이에 동등하게 심각한 사건이다:
    contact            Δρ=-0.222   Δρ_worst=-0.943   satisfies→False   [시작의 1.3배 — 이진 반전]
    gripper_open       Δρ=-0.688   Δρ_worst=-0.707   satisfies→False
    object_goal_dist   Δρ=-0.040   Δρ_worst=-0.707   satisfies→False
    eef_object_dist    Δρ=-0.627   Δρ_worst=-0.707   satisfies→False
    → 동등 심각도 조건 3개의 편차:  ρ 17.3배  vs  ρ_worst 1.0배  (1.0 에 가까울수록 공정)

==============================================================================
Δρ_worst 가 Stage 7 의 최적화 신호입니다 (rho.py 헤더 참조).
|Δρ_worst| 가 작은 조건은 FCM screening 이 사실상 볼 수 없습니다.
==============================================================================
PS C:\Users\Administrator\Desktop\tyu-main\gma> 



[verify] G4 PASS / G5 PASS
PS C:\Users\Administrator\Desktop\tyu-main\gma> python .\verify_subgoal.py
[V1] 경계 커버리지: 97%  pre=100%  move=95%  post=95%
[V2] pre       ablation 생존 조건: contact(v=0.000)
[V2] move      ablation 생존 조건: —
[V2] post      ablation 생존 조건: contact(v=0.000)
[G4] PASS: 필수 경계 ['pre', 'post'] 생존 확인 (move 는 객체 채널 정의라 참고만)
[V3] phase 역행률 0.30% (스텝 7765개), 세그먼터 z 일치율 89.8%
[G5] PASS: 역행 0.30% < 5%
[V4] leave-one-out 경계 커버리지: 95% (20 folds; 데모 수가 적으면 참고용)

[verify] G4 PASS / G5 PASS
PS C:\Users\Administrator\Desktop\tyu-main\gma> 




[warn] feature_select: torque/qvel missing -> energy column is identically 0. Torque is NOT recoverable by state replay; record it during collection or fall back to frame_extract.inverse_dynamics_torque().
  demo_000: rows so far 1600
  demo_001: rows so far 3336
  demo_002: rows so far 4976
  demo_003: rows so far 6600
  demo_004: rows so far 8304
  demo_005: rows so far 9768
  demo_006: rows so far 11464
  demo_007: rows so far 13000
  demo_008: rows so far 14640
  demo_009: rows so far 16240
  demo_010: rows so far 17840
  demo_011: rows so far 19592
  demo_012: rows so far 21384
  demo_013: rows so far 23144
  demo_014: rows so far 24944
  demo_015: rows so far 26728
  demo_016: rows so far 28544
  demo_017: rows so far 30280
  demo_018: rows so far 31888
  demo_019: rows so far 33624
[collect] 33624 rows (606 branches dropped; anchors kept 229/240)
[G6] FAIL: λ=0 drift/effect (subgoal 조건 채널) 최대 0.627 (gripper_open) > 0.3, anchor drop 5% <= 50%
     판정 대상: eef_object_dist=0.076, object_goal_dist=0.030, gripper_open=0.627, contact=0.505
[plot] ./artifacts\fcm_g6_residual.png
[out] ./artifacts\fcm_dataset.hdf5
[G6] FAIL — 분기/리플레이 수정 전 학습 금지
PS C:\Users\Administrator\Desktop\tyu-main\gma> 



[collect] 33840 rows (600 branches dropped; anchors kept 230/240, ρ 필터로 1개)
[G6] FAIL: ρ_worst 공간 drift/effect 최대 0.686 (phase 0 [pre]) > 0.3, anchor drop 4% <= 50%
     phase 0 [pre  ] drift_ρ=0.0870 effect_ρ=0.1268 ratio=0.686 (λ=0 752행 / λ>0 12968행)  <-- 이 phase 데이터는 못 쓴다
     phase 1 [move ] drift_ρ=0.0008 effect_ρ=1.0844 ratio=0.001 (λ=0 520행 / λ>0 9064행)
     phase 2 [post ] drift_ρ=0.0023 effect_ρ=0.3212 ratio=0.007 (λ=0 568행 / λ>0 9968행)
     [진단] raw feature 비율 (분모가 δ 무관 방향까지 포함해 희석됨): eef_object_dist=0.074, object_goal_dist=0.026, gripper_open=0.523, contact=0.200
[plot] ./artifacts\fcm_g6_residual.png
[out] ./artifacts\fcm_dataset.hdf5
[G6] FAIL — 분기/리플레이 수정 전 학습 금지
PS C:\Users\Administrator\Desktop\tyu-main\gma> 



ollect] 33288 rows (606 branches dropped; anchors kept 227/240, ρ 필터로 2개)
[G6] PASS: ρ_worst 공간 drift/effect 최대 0.075 (phase 0 [pre]) <= 0.3, anchor drop 5% <= 50%
     phase 0 [pre  ] drift_ρ=0.0072 effect_ρ=0.0955 ratio=0.075 (λ=0 736행 / λ>0 12752행)
           drift 분포: 중앙 0.00000  95% 0.0005  최대 0.0535
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00718 ratio=0.075
     phase 1 [move ] drift_ρ=0.0025 effect_ρ=1.0484 ratio=0.002 (λ=0 504행 / λ>0 8648행)
           drift 분포: 중앙 0.00026  95% 0.0015  최대 0.0235
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00255 ratio=0.002
     phase 2 [post ] drift_ρ=0.0023 effect_ρ=0.7415 ratio=0.003 (λ=0 576행 / λ>0 10072행)
           drift 분포: 중앙 0.00078  95% 0.0041  최대 0.0110
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00231 ratio=0.003
     [진단] raw feature 비율 (분모가 δ 무관 방향까지 포함해 희석됨): eef_object_dist=0.080, object_goal_dist=0.035, gripper_open=0.489, contact=0.000
[plot] ./artifacts\fcm_g6_residual.png
[out] ./artifacts\fcm_dataset.hdf5
PS C:\Users\Administrator\Desktop\tyu-main\gma> 


PS C:\Users\Administrator\Desktop\tyu-main\gma> python .\fcm.py train
[G6] PASS: ρ_worst 공간 drift/effect 최대 0.075 (phase 0 [pre]) <= 0.3, anchor drop 5% <= 50%
     phase 0 [pre  ] drift_ρ=0.0072 effect_ρ=0.0955 ratio=0.075 (λ=0 736행 / λ>0 12752행)
           drift 분포: 중앙 0.00000  95% 0.0005  최대 0.0535
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00718 ratio=0.075
     phase 1 [move ] drift_ρ=0.0025 effect_ρ=1.0484 ratio=0.002 (λ=0 504행 / λ>0 8648행)
           drift 분포: 중앙 0.00026  95% 0.0015  최대 0.0235
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00255 ratio=0.002
     phase 2 [post ] drift_ρ=0.0023 effect_ρ=0.7415 ratio=0.003 (λ=0 576행 / λ>0 10072행)
           drift 분포: 중앙 0.00078  95% 0.0041  최대 0.0110
           > 0.15 인 행 0개 (0.00%) 제외 시 drift_ρ=0.00231 ratio=0.003
     [진단] raw feature 비율 (분모가 δ 무관 방향까지 포함해 희석됨): eef_object_dist=0.080, object_goal_dist=0.035, gripper_open=0.489, contact=0.000
  member 0: fitted (val loss 0.8049)
  member 1: fitted (val loss 0.7908)
  member 2: fitted (val loss 0.7949)
  member 3: fitted (val loss 0.7884)
  member 4: fitted (val loss 0.7928)
[train] heldout R^2 (rollout 단위 분할):
    eef_object_dist    +0.365
    object_goal_dist   +0.300
    gripper_open       +0.623
    contact            +0.333
    grasp_align        +0.517
    object_height      +0.416
    eef_speed          +0.775
    object_speed       +0.485
    action_magnitude   +0.887
    eef_accel          +0.719
    eef_jerk           +0.693
    object_slip        +0.686
    eef_ang_speed      +0.971
    object_ang_speed   +0.738
    energy               n/a
[plot] ./artifacts\fcm_fit_r2.png
[out] ./artifacts\fcm_ensemble.pkl
PS C:\Users\Administrator\Desktop\tyu-main\gma> 

