# Project summary (for review and planning)

Cell balancing for an EV battery pack: passive shunt, inductor buck-boost and flying capacitor, simulated on a 4-cell series pack with an LG HG2 cell model fitted to measured data (Mendeley dataset, DOI 10.17632/b5mj79w5w9.2, CC BY 4.0), plus an ML surrogate, SOC estimation and RL control. Branch: `cloud/endsem-final`. Python 3.13, everything deterministic from one seed; `python run_all.py` takes about 2 minutes on 4 cores.

## 1. What is committed and pushed (last push: commit "RL balancing control, N=96 scaling and final consolidation")

| Block | What was done | Key numbers |
| --- | --- | --- |
| A. Drive-cycle load | Reusable `step(state, pack_current, control)`; measured Mixed 1 current (1 s average) applied to the baseline pack; rest, driving, 0.5C charging | Driving usable Ah: none 1.232, passive 1.232, buck-boost 1.324, flying capacitor 1.380 (ideal-transfer reference 1.395). SOC conservation 1.7e-13, ledger under load 0.09 % |
| B. Temperature and SOC error | Phase 1 fit pipeline at 0, 10, 25, 40 °C; Arrhenius R0 fit; SOC-bias study (200 packs, sigma 0/1/3 %) | UDDS RMSE 38.5, 38.4, 28.5, 27.3 mV (0 and 10 °C FAIL the 30 mV limit). Ea 0.201 eV, 13.5 % misfit. 1 % bias leaves 86-87 % of passive and buck-boost packs above the 1 % target. 10 mV voltage error = 1.07 % SOC |
| C. SOC estimation | Coulomb counting, 1RC EKF, GRU (3 seeds) on fixed splits written before training; feed-through to balancing | Test A / Test B (10 °C, unseen) RMSE: GRU 2.49 / 2.97 %, EKF 4.96 / 6.18 %, coulomb counting 25.2 / 20.2 % (0.5 % FS offset) or 5.05 / 4.03 % (0.1 % FS). Balancing on these estimates still leaves at least 97 % of passive and buck-boost packs above 1 % |
| D. Scaling | N = 4, 8, 16, 32 (96 in the pushed version), 200 packs per N, random and step initial profiles | Step profile slopes match theory (flying capacitor 2.00 vs 2.00, buck-boost 0.92 vs 1.0, passive 0.00 vs 0). Random profiles do not follow N^2 (flying capacitor 0.90, buck-boost 0.22) |
| E. RL | Gymnasium env on the step function, buck-boost links with DCM-clipped duty, PPO 3 seeds, compared with no balancing, the midsem rule, greedy max transfer, an ideal-transfer reference | Null result. Mixed 7 delivered Ah: none 1.291, rule 1.335, greedy 1.334, PPO 1.289, reference 1.356. PPO never beat no balancing or the rule. Two training configurations were tried and both are reported |
| F. Consolidation | README sections r to v, `results/claims.md` (65 PASS, 34 FAIL, 26 NOT SUPPORTED), `results/FIGURES.md`, privacy grep clean, two `run_all` runs byte-identical | |

Data caveats: Mixed 3 missing at 0 and 10 °C; 40 °C Mixed 4-8 are aborted runs (0.19 Ah, case at 23 °C) and rejected; 10 °C LA92 is a spliced file and rejected; 25 °C HWFET has no current; labels come from integrated tester current (0.1 % of 75 A full scale).

## 2. What is in the working tree but NOT committed or pushed

These were done in the last session and are consistent with each other but have not been run through `run_all`, claims, tests and commit:

- `src/cold_model.py`, `scripts/identify_temperature.py`: calibrated nonlinear 2RC model (tables from HPPC, six scale factors fitted on other drive cycles, validated on UDDS/LA92). Full-rate replay: 0 °C UDDS 21.3/21.2 mV, LA92 32.4/34.0 mV (FAIL); 10 °C UDDS 22.0/20.03 mV (misses the 20 mV window by 0.03 mV, FAIL); 25 and 40 °C pass at 13-17 mV. Not usable inside the simulator (nonlinear).
- `src/temperature.py`: R0(T) form comparison (log-quadratic 0.7 % leave-one-temperature-out error vs 32 % for Arrhenius) and baseline balancing at the four temperatures.
- `src/coupled_links.py` plus tests: net-current link model (shared cell resistance solved: tridiagonal for the flying capacitor, fixed point for the buck-boost). Net-current ledger closes to about 0.05 % of dissipated energy. The per-link model remains the default so the midsem regression still passes.
- `src/drive_cycle.py` (`population_block`): 200 random packs, measured Mixed 1: buck-boost and flying capacitor raise usable Ah in 98-100 % of packs; passive does not help while driving.
- `src/soc_estimation/ekf2.py` and `scripts/evaluate_soc.py`: 2RC EKF with SOC-dependent parameters and one-state hysteresis, tuned on validation only. RMSE 2.06 % (Test A) and 2.11 % (Test B), which beats the GRU (2.49 / 2.97 %) and the 1RC EKF.
- `scripts/report_claims.py` edits for the above were NOT applied (the last edit was interrupted). Some existing claims still point at keys that exist; new claim rows for EKF2, coupling and the population study are still to be added.

Reverted or dropped (decided to cut scope): GRU variant retraining (augmentation, hidden 64, streaming), N = 96 rerun with a minimax buck-boost baseline, splits v2, N = 8 surrogate, technique selection, extra RL algorithms (SAC, TD3, imitation, MPC), all LTspice/circuit work (to be done on the desktop).

To finish: re-apply the claims edits, run `python run_all.py`, `python scripts/report_claims.py`, `pytest`, `ruff`, write a short README section for the above, update the feed-through numbers in README section t (they changed because EKF2 shifted the random draws), then commit and push.

## 3. Open limitations and ideas (ranked by value for effort)

1. **SOC-error loop is the main engineering gap.** Balancing on estimated SOC does not reach the 1 % true spread even with the best estimator. Try voltage-based balancing at rest with a sensor-error model (a few mV of cell voltage error maps to a few tenths of a percent SOC) and a stopping threshold below the target. This also gives a required ADC accuracy.
2. **Cold cell model.** The calibrated model nearly passes at 10 °C; a third slow RC branch or a SOC-dependent current exponent might close it. It cannot enter the simulator until it is linear again.
3. **Make net-current the default** after re-checking the ML dataset and midsem reference, so the reported dissipated energy has no known per-link approximation error.
4. **Scaling theory for random strings.** The heat-kernel plus extreme-value argument predicts (ln N)^2 growth for the flying capacitor; a quick test showed it is close but not within the 0.25 slope tolerance. The buck-boost baseline over-predicts at large N because it fixes one final state; a linear program over final states inside the 1 % band brought median error to about 9 % at N = 8-32 in a quick test.
5. **RL.** PPO collapsed to near-zero actions because the benefit of balancing arrives only at the episode end. Options: imitate the rule then fine-tune, a residual policy on top of the rule, dense shaping with a spread potential (training only), SAC/TD3, or a model-predictive controller over the exact simulator. On 4 cells the rule is already within about 0.02 Ah of the reference, so N = 8-16 and noisier SOC estimates are where learning has room.
6. **GRU.** Augmentation with noise and current offset, a stateful streaming variant, int8 quantisation and MACs per second for an embedded budget. The 2RC EKF now beats the GRU at 25 and 40 °C, so a hybrid (EKF plus learned correction) is more promising than a bigger GRU.
7. **ML surrogate for N = 8** and a technique-selection policy (pick the technique by predicted time and energy under a time budget).
8. **Hardware-facing work (desktop, LTspice):** switching-level loss breakdown, DCM/CCM boundary and inductor saturation, flying-capacitor chain at circuit level, one bench measurement of a buck-boost link.

## 4. Where to look

- `README.md` sections r to v (final phase), `results/claims.md` (every claim with evidence key, threshold, PASS/FAIL), `results/FIGURES.md` (one line per figure), `results/metrics.json` (all numbers), `data/processed/` (small committed caches), `scripts/` (everything that downloads, trains or runs long jobs).
- Ground rules used: one seed, explicit RNGs, no wall-clock numbers in `metrics.json`, new metrics keys only, no invented data, never loosen a threshold to pass.
