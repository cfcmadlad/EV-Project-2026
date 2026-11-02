# Project summary

This file summarises the whole study in one place: the three balancing techniques, how they were modelled and checked, what was learned from data, and every extension, with the numbers that matter and the results that failed. Every number is taken from `results/metrics.json`; the claim behind each one, with its evidence key, threshold and status, is in `results/claims.md`. The full write-up is `README.md`, and `results/FIGURES.md` lists every figure.

## 1. What the project is

When cells in a series string drift to different charge levels, usable capacity shrinks and some cells are stressed. The project simulates three ways to rebalance a 4-cell pack that starts with a 10 % SOC spread:

- the **passive shunt**, which burns the excess charge of the fullest cells in resistors;
- the **inductor buck-boost**, which moves charge between neighbouring cells through an inductor;
- the **flying capacitor**, a switched capacitor that shuttles charge between neighbouring cells.

The cell model is fitted to measured data of one LG HG2 18650 cell (Mendeley Data, DOI 10.17632/b5mj79w5w9.2, licence CC BY 4.0). The converters are checked against switching-level LTspice circuits with real MOSFETs, gate drive and dead time. On top of the simulator sit learned surrogates, SOC estimators, reinforcement-learning and model-predictive controllers, and a technique selector. Everything is deterministic: one simulation seed (2026) and seeds 0 to 2 for the learned models.

Ground rules that shape every result: no threshold was loosened to make a claim pass, FAIL claims are kept and described, and anything that rests on an assumption or on a missing measurement is marked NOT SUPPORTED.

Scoreboard: 322 claims, of which 183 PASS, 113 FAIL and 26 NOT SUPPORTED; 174 tests pass.

## 2. The three techniques side by side

Baseline: the same pack with a 10 % SOC spread, mid-range component values and the circuit-calibrated converter models.

| Property | Passive shunt | Inductor buck-boost | Flying capacitor |
| --- | --- | --- | --- |
| How it works | A resistor and a MOSFET per cell bleed the cells above the pack mean | An inductor between adjacent cells moves charge from the higher to the lower cell, in discontinuous conduction | A capacitor switched between adjacent cells by a fixed complementary clock equalises them |
| Component values | R_b = 60 Ω | L = 55 µH, f = 30 kHz | C = 5.5 mF, f = 5.5 kHz |
| Balancing time | 4.19 h (15068 s) | 1.08 h (3897 s), fastest | 1.99 h (7174 s) |
| Energy dissipated | 7176 J | 220.5 J | 39.5 J, lowest |
| of which cell side (conduction, R0 heating, switching) | 7176 J | 205.1 J | 29.9 J |
| of which gate drive, drawn from the pack | 0 J | 15.4 J | 9.6 J |
| Efficiency | 0 % | 96.2 % | 99.3 % |
| Component count | 4 resistors, 4 MOSFETs | 3 inductors, 6 MOSFETs | 3 capacitors, 8 MOSFETs |
| Cost | Low | High | Medium |
| Control | Low: on or off per cell | High: PWM, direction and current limits per link | Low to medium: fixed clock |

The ranking is buck-boost fastest, flying capacitor lowest loss, passive slowest and wasteful. The gate drivers are powered from the source cell of each active link, so their energy leaves the pack as extra source current and is booked as its own loss line. Sweeping the passive bleed resistor from 20 Ω to 100 Ω moves the passive balancing time from 5031 s to 25106 s. Across the 6000 dataset runs the techniques rank the same way. Cost and control complexity are qualitative engineering judgements.

## 3. Cell model and verification

**Fitted cell.** The C/20 discharge gives the OCV curve (2.88 V at 0 % to 4.19 V at 100 %) and a capacity of 2.781 Ah; HPPC pulses give R0 (11.6 mΩ at 50 % SOC) and the RC branches. UDDS and LA92 are held out of the fit and replayed with limits of 30 mV RMSE over the whole cycle and 20 mV in the 20 to 80 % SOC window.

| Model | UDDS whole | UDDS window | LA92 whole |
| --- | --- | --- | --- |
| Assumed generic cell (before the fit) | 92.9 mV | 56.6 mV | 106.7 mV |
| 1RC | 37.8 mV (FAIL) | 28.1 mV | 42.5 mV |
| 2RC (kept) | 28.5 mV | 16.63 mV | 28.8 mV |
| Single-branch equivalent of the 2RC (what the simulator uses) | 31.5 mV (FAIL) | | 35.0 mV (FAIL) |

Cell-to-cell spreads (capacity plus or minus 5 %, R0 plus or minus 10 %) are assumptions, because the data set has one cell. Fits at 0, 10, 25 and 40 degrees C follow an Arrhenius law for R0 with an activation energy of 0.201 eV, but its largest deviation is 13.48 % against a 10 % limit (FAIL); a log-quadratic form does better (0.7 % leave-one-temperature-out). The kept model fails the voltage limits at 0 and 10 degrees C, a nonlinear calibrated model reduces the error (UDDS at 0 degrees C: 21.3 mV, LA92: 32.4 mV), and balancing results at 0, 10 and 40 degrees C are NOT SUPPORTED.

**Simulator checks.** These run on every `run_all.py`:

- passive balancing time follows the closed form t = max_i 3600 Q_i (R_b + R0_i + R1) times the OCV integral to 0.06486 % over 200 random packs;
- flying-capacitor charge is conserved to a worst relative imbalance of 5.06e-13;
- an independent element-by-element energy ledger closes within 1 % of the energy removed for all three techniques and within 0.1 % per link;
- two consecutive runs give byte-identical metrics, CSV files and figures.

One check fails and is explained: the default per-link model books the cell resistance once per link, so the dissipated energy differs from a strict net-current ledger by up to 1.967 % (buck-boost) and 7.276 % (flying capacitor) over 200 random packs. The optional net-current link model removes the difference (ledger gap 0.11 % and 0.09 %).

## 4. LTspice validation and calibrated converter models

Each converter link was built in LTspice with real parts (flying capacitor: BSZ100N03LS MOSFETs with 5 V, 2.2 Ω gate drivers, 0.25 µs dead time, 5 mΩ ESR; buck-boost: FDS4410 switches with a synchronous or MBRS340 Schottky return path, 80 mΩ inductor; passive: resistor and MOSFET). 434 runs sweep C, f, L, operating point and SOC spread. Measurement quality checks pass; the worst energy balance residual of any run is 0.89 % of the removed power.

- **Flying capacitor.** The textbook series formula 1/(f C) plus series resistances is refuted: with real parts it is off by up to 160.48 % (median 75 %). The exact switched-capacitor form R_eq = (1/(f C)) coth(1/(4 f R C)) holds within 3.37 % on real parts and 1.01 % on ideal switches. At the baseline the series form gives 152 % more link resistance and a balancing time ratio of 1.66.
- **Buck-boost.** The converter stays in discontinuous conduction over the whole grid. The averaged currents hold within 4.88 % where the peak current stays below an inductor saturation rule; over the whole grid the sink current error reaches 28 %, so 56 of 96 combinations are unrealistic and the dataset range is pruned. The averaged loss error is 16.63 % in range against a 15 % tolerance (FAIL).
- **Decision.** The rule fixed in advance was: if the averaged formulas miss the 5 % current or 15 % loss tolerance, correction maps built from the LTspice results become the default. They did. The maps lower the buck-boost loss error to 6.87 % and the flying-capacitor current error to 0.10 % (in sample). The uncalibrated models are kept as a named option, and their claim is a FAIL.
- Python audits add that booking the buck-boost loss with the RMS ripple current raises the baseline energy to 236.1 J, and that the worst peak current over the L and f range is 16.8 A.

## 5. Learned surrogates

Task: predict balancing time and energy from the initial pack, so a BMS could choose a technique without simulating. The dataset is 6000 random packs (2000 per technique) with 19 features.

- **Plain boosting** (log targets, 500 trees): balancing-time R2 0.9836 and energy R2 0.9952 on one seed and an 80/20 split; the flying capacitor time is the hard case (R2 0.8691). On the 70/15/15 three-seed protocol of the model comparison the boosting R2 is 0.9825, short of the 0.99 target (FAIL).
- **Against physics.** Closed-form formulas written for the same simulator match the surrogate closely. Plain boosting does not beat them. A hybrid that learns the log residual over the closed form is the better design.
- **Deep models.** Plain MLP, cell-sequence GRU and cell-token Transformer all lose to boosting (GRU 0.9530). Learning only the residual over the physics lifts every model to 0.9997 or better (0.9998 for the GRU); boosting on the residual is as good as the networks to within 1e-4, so the value is in the physics, not the architecture.
- **Trajectories.** GRU and Transformer decoders predict the whole SOC-spread trajectory within 0.219 percentage points RMSE and reproduce the three shapes (linear passive, steady buck-boost, exponential-tail flying capacitor), but the absolute time to a 1 % spread is off by 13 to 15 % median (FAIL against 10 %).
- **Intervals and range.** 90 % conformal intervals reach nominal coverage for most targets; the passive-energy interval undercovers at 85.25 %. Out-of-range packs raise the error for every technique.

## 6. SOC estimation

Measured UDDS and LA92 data, Test A at 25 degrees C and Test B at 10 degrees C (unseen), with the splits written before training.

| Estimator | Test A RMSE | Test B RMSE |
| --- | --- | --- |
| Coulomb counting, 0.5 % full-scale offset | 25.2 % | |
| Coulomb counting, 0.1 % full-scale offset | 5.05 % | 4.03 % |
| 1RC EKF | 4.96 % | 6.18 % |
| 2RC EKF with hysteresis | 2.06 % | 2.11 % |
| GRU | 2.49 % | 2.97 % |
| EKF plus GRU hybrid | 0.89 % | 1.43 % |

The hybrid keeps within 5 % of its float32 RMSE at int8 (4560 bytes of weights). The 2RC EKF recovers from a 20 % wrong start to 2.39 % after 300 s against a 2 % limit (FAIL), and the 1RC EKF does not beat tester-grade coulomb counting (FAIL). Balancing on an estimated SOC still leaves most packs above the 1 % true-spread target (hybrid, passive: 72.5 %); the median final true spread is 2.9 % for passive on the 2RC EKF and 3.6 % for the buck-boost on the GRU. Balancing on cell rest voltage instead works only with a precise sensor: with a 0.5 % stopping threshold the largest sensor error that still brings 95 % of packs to the 1 % spread is about 1.7 mV (passive) and 1.6 mV (buck-boost), and the requirement fails at 2, 5 and 10 mV.

## 7. Control under load

- **Drive cycles.** Balancing during the measured Mixed 1 driving current: usable capacity is 1.232 Ah with no balancing, 1.324 Ah with the buck-boost (gain 0.092 Ah) and 1.358 Ah with the flying capacitor (gain 0.126 Ah); a charge-equalised pack reaches 1.395 Ah. Over 200 random packs the median driving gain is 0.072 Ah (buck-boost) and 0.074 Ah (flying capacitor); passive balancing does not help while driving.
- **Scaling to longer strings.** For 4 to 96 cells with a step-shaped initial profile the balancing time follows the derived laws (flying capacitor slope 1.87 against a theory of 2, buck-boost 0.90 against 1). With random initial SOCs the buck-boost slope is 0.22 against a theory of 0.5 (FAIL) and the flying-capacitor N-squared law is NOT SUPPORTED.
- **Reinforcement learning.** Eight PPO variants (residual on the rule or imitation start, with and without shaping, 3 seeds, 4 and 8 cells) are compared with no balancing, the rule, greedy transfer and an ideal reference. None beats the rule on every cycle; the best 8-cell policy gains 0.0023 Ah per pack, within noise. This null result stands.
- **Controller headroom bound.** The room any controller has over the rule is measured against a lossless ideal-transfer reference. At 4 cells the rule is within 0.88 % (buck-boost) and 0.79 % (flying capacitor) of it, so a learned controller has no meaningful room there. From 8 cells the median gap exceeds 1 % and the 90th percentile 2 %, reaching 1.45 % and 2.57 % at 96 cells. SOC estimation error costs the buck-boost a further 3.29 percentage points at 96 cells and the flying capacitor almost nothing. The passive shunt leaves a median gap of 5.1 % to 8.3 %.

## 8. Extensions

All thresholds, budgets and parameters were fixed in `src/params.py` and in README section s before any extension result was computed. Null and failed results are reported as they came out.

**Abstaining technique selector.** The selector chooses the technique with the least predicted energy among those predicted to finish within a parking-time budget. With normalised conformal intervals calibrated on rows the predictor was not fitted on, all six technique and target intervals fall in the 87 to 93 % coverage window (time 87.25, 88.5 and 88.5 %; energy 87.25, 89.5 and 88.25 % for passive, buck-boost and flying capacitor); the fixed-width interval missed it for passive energy. At the 1 h budget the boosting selector misses the budget on 6.8 % of feasible packs. The same rule on the hybrid predictor misses 0.3 % with point forecasts and 0 % when planning with the upper bound (98.7 % agreement with the simulation oracle). Abstaining on wide intervals (5.7 % of packs) changes nothing at 1 h and above, so the gain comes from the better predictor, not from the interval.

**Thermal limit.** With assumed thermal parameters (none exist in the repository; every absolute temperature is NOT SUPPORTED), continuous passive bleeding at the baseline reaches a 23.9 K resistor rise, above the 20 K ceiling (FAIL). Duty-cycling the bleed to hold the ceiling lengthens balancing from 4.19 h to 5.23 h (ratio 1.25) at the same 7171 J. Cell heating from R0 losses stays below 0.01 K for all three techniques. The thermally aware selector never picks passive, so the constraint changes no decision, and its 6.8 % violations at 1 h are inherited from the boosting predictor (FAIL). The heat balance checks are identities of the reconstruction, not independent validation.

**Out-of-range stress test.** With the SOC spread limit and the capacity and resistance tolerances widened 1, 1.25, 1.5 and 2 times (no retraining, 300 fresh packs per factor), the boosting surrogate's median balancing-time error grows from 6.0 % to 7.6 %, 9.7 % and 16.6 %, and its selector's 1 h violations from 3.7 % to 5.6 %, 6.5 % and 8.6 %: it crosses the 5 % limit at 1.25 times (FAIL). The hybrid keeps its median error at 0.6 to 0.7 % and its interval coverage at 91.6, 91.3, 91.0 and 89.3 %, and its abstaining selector stays at 0 % up to 1.5 times and 1.0 % at 2 times. The hybrid extrapolates because it multiplies a closed form by a small learned residual.

**Differentiable twin and gradient control.** A PyTorch twin of the per-link averaged pack step reproduces the NumPy step to machine precision (SOC exactly, energies within 5e-16 relative) and the NumPy rule's delivered Ah exactly. A receding-horizon controller replans every 120 s, throttles each link between 0 and 1 by 25 Adam steps over a 360 s horizon with the exact upcoming load as preview, and aims to maximise the softmin of the cell terminal voltage margin. It does not beat the rule: mean gain over the rule at 8, 16 and 32 cells is +0.41, -0.14 and -2.33 mAh for the buck-boost (win fractions 82, 73 and 59 %) and -0.90, -0.97 and -1.77 mAh for the flying capacitor (72, 71 and 65 %), against 15.9 to 30.1 mAh left by the rule. The 90 % win-fraction requirement and the 25 % gap-closed requirement both fail. Throttling lowers the buck-boost converter loss (by up to 520 J per run at 32 cells) without delivering more Ah. This is a null result for that one design.

## 9. What fails and what is not supported

- 113 claims FAIL: among them the 1RC cell model and the single-branch equivalent at 30 mV, the Arrhenius law at 10 %, the averaged buck-boost loss at 15 %, the net-current ledger gaps, plain boosting at R2 0.99, trajectory absolute time at 10 %, the 2RC EKF wrong-start recovery, all PPO variants, the random-profile scaling slopes, rest-voltage balancing at 2 mV and above, the boosting selector at 1 h, continuous passive bleeding under the thermal ceiling, the boosting selector at 1.25 times the training ranges and the gradient controller.
- 26 claims are NOT SUPPORTED, because nothing in the repository could support them: that converter efficiencies, losses and component values match a built converter, that the vendor datasheets validate the model, that results hold at 0, 10 and 40 degrees C or for other cells, lots, aged cells and temperatures, that learned control improves balancing under load, that results hold for strings longer than 4 cells or for an SOC-estimated pack in general, that the N-squared law holds for random strings, that the surrogate fits a microcontroller, and that any absolute temperature or the thermal ceiling holds on a built balancer.
- No converter was built or measured, the data set holds one cell, and nothing ran on a microcontroller. The circuits use ideal DC cell sources and a linear inductor without core loss.

## 10. Where to look

| Question | File |
| --- | --- |
| The full write-up, sections a to t | `README.md` |
| Every claim, evidence key, threshold and status | `results/claims.md` |
| What each figure shows | `results/FIGURES.md` |
| Every number | `results/metrics.json` |
| Reproduce everything | `python scripts/fetch_hg2.py --group all`, `python run_all.py`, `python -m scripts.report_claims`, `python -m scripts.check_readme`, `python -m pytest`, `ruff check .` |
