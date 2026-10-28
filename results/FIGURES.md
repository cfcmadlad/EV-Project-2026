# Figures

One line per figure: what it shows and its key number (all from `metrics.json`, fitted 25 °C parameters unless stated).

## Balancing at rest (sections f to i)

- `soc_passive.png`: per-cell SOC of the baseline 10 % spread pack during passive shunt balancing; the pack balances (spread below 1 %) after 4.19 h.
- `soc_buck_boost.png`: the same pack with the inductor buck-boost; balanced after 1.09 h.
- `soc_flying_capacitor.png`: the same pack with the flying capacitor; balanced after 2.00 h with a slow tail near the end.
- `soc_spread_comparison.png`: SOC spread against time for the three techniques on a log axis; energy dissipated 7176 J (passive), 200 J (buck-boost), 29.9 J (flying capacitor).
- `sweep_passive_r_bleed.png`: passive balancing time against bleed resistance (20 to 100 Ω); 1.40 h to 6.97 h.
- `sweep_flying_capacitor_frequency.png`: flying-capacitor balancing time against switching frequency; 2.70 h at 1 kHz down to 1.94 h at 10 kHz.
- `ml_parity_balancing_time_s.png`: predicted against simulated balancing time on the held-out set; test R² 0.989.
- `ml_parity_energy_dissipated_j.png`: predicted against simulated dissipated energy; test R² 0.995.
- `ml_importance_balancing_time_s.png`: permutation importance of the model inputs for balancing time.
- `ml_importance_energy_dissipated_j.png`: permutation importance of the model inputs for dissipated energy.

## Cell model from measured data (section o)

- `hg2_raw_overview.png`: overview of the measured LG HG2 25 °C tests (C/20, HPPC, drive cycles); C/20 capacity 2.78 Ah.
- `hppc_params_vs_soc.png`: R0 and polarisation parameters fitted from the HPPC pulses against SOC; R0 median in the constant window 11.8 mΩ with a 95 % interval of 11.4 to 12.2 mΩ.
- `cell_validation_udds.png`: measured against modelled terminal voltage on the UDDS cycle; 2RC whole-cycle RMSE 28.5 mV.
- `cell_validation_summary.png`: RMSE per drive cycle and model against the 30 mV and 20 mV limits; 2RC passes (UDDS 28.5, LA92 28.8, US06 29.9 mV), 1RC fails.
- `baseline_assumed_vs_fitted.png`: baseline time and energy with the assumed and the fitted parameters; passive time ratio 0.90.

## Converter models (section p) and ML evaluation (section q)

- `spice_vs_averaged.png`: averaged-model mean currents against LTspice at six operating points; buck-boost worst current error 2.9 %, two-phase flying-capacitor 0.03 %, midsem flying-capacitor formula 152 %.
- `spice_buckboost_waveform.png`: steady-state LTspice waveforms of the buck-boost at the three operating points.
- `spice_flyingcap_waveform.png`: steady-state LTspice waveforms of the flying capacitor at the three operating points.
- `ml_cv_comparison.png`: cross-validated MAE of physics formulas, gradient boosting, log-linear regression and the hybrid; the hybrid is best for both targets.
- `ml_out_of_range.png`: predictions for packs outside the training range (largest 20 % of component values, spread above 8 %).
- `ml_conformal_coverage.png`: conformal 90 % interval coverage per technique against the 87 to 93 % band; passive balancing-time coverage 88 %.

## Final phase

- `drive_cycle_spread.png`: SOC spread against time while driving (Mixed 1, repeated) and charging at 0.5C; only the flying capacitor reaches 1.0 % while driving, only the buck-boost reaches 0.86 % while charging.
- `drive_cycle_usable_capacity.png`: pack Ah until the first cell hits a limit in three scenarios; driving 1.232 (none) and 1.232 (passive), 1.324 (buck-boost), 1.358 Ah (flying capacitor), ideal-transfer reference 1.395 Ah.
- `params_vs_temperature.png`: R0, capacity, OCV at 50 % SOC, validation RMSE and polarisation against temperature; R0 30.2 mΩ at 0 °C to 10.1 mΩ at 40 °C, Arrhenius Ea 0.201 eV with a 13.5 % worst misfit, UDDS RMSE 38.5, 38.4, 28.5, 27.3 mV at 0, 10, 25, 40 °C.
- `soc_error_true_spread.png`: true final SOC spread (median with 5th to 95th percentile) against the SOC bias sigma for each technique; at sigma 1 % the median is 1.9 % (passive, buck-boost) and 86 to 87 % of packs end above the 1 % target.
- `soc_est_test_traces.png`: reference SOC, coulomb counting with a 0.5 % FS offset, EKF and GRU on UDDS at 25 °C (Test A) and at 10 °C (Test B, unseen).
- `soc_est_error_summary.png`: RMSE and maximum error per estimator and temperature; GRU RMSE 2.49 % (Test A) and 2.97 % (Test B) against EKF 4.96 % and 6.18 %.
- `soc_est_wrong_init.png`: mean absolute error during the first 20 minutes from a start 20 % too low; the EKF is at 1.8 % (Test A) and 1.95 % (Test B) after 300 s, coulomb counting stays near 21 %.
- `scaling_time_vs_n.png`: median balancing time against string length N (4 to 96) for random and step initial profiles with the theory lines; step slopes 0.90 (buck-boost, theory 1.0) and 2.00 (flying capacitor, theory 2.00), random slopes 0.22 and 0.90.
- `rl_training_curves.png`: PPO mean training return for the three seeds over 300 000 steps; no seed improves clearly.
- `rl_vs_baselines.png`: delivered Ah and reward on the four test cycles for no balancing, rule-based, greedy, PPO and the ideal-transfer reference; PPO equals or trails no balancing (for example 1.289 against 1.291 Ah on Mixed 7) while the rule gives 1.335 Ah.
- `voltage_balancing.png`: fraction of 200 packs below 1 % true SOC spread against cell voltage sensor error for passive, buck-boost and flying capacitor at three stop thresholds; 100 % at 1 mV, 94 % and 93 % at 2 mV, 11 % at 10 mV with the 0.5 % stop.
- `technique_selector.png`: budget violations among feasible packs and mean energy against parking time budget for the surrogate selector, physics selector, oracle and fixed policies; surrogate agreement 91.7 to 100 %, 6.5 % violations at 1 h.

## Circuit-level validation and deep learning (sections ab to ad)

- `ltspice_buck_boost_waveform.png`: two switching periods of the LTspice real-part synchronous buck-boost at 55 µH and 30 kHz (inductor current, switch node, cell currents); discontinuous conduction with D + D2 at most 0.961 over the grid.
- `ltspice_flying_capacitor_waveform.png`: two periods of the flying-capacitor link at 5.5 mF and 5.5 kHz (capacitor current, top plate voltage, cell currents); LTspice sink current 0.3214 A.
- `ltspice_vs_averaged.png`: model against LTspice sink current and link efficiency over the real-part sweeps, averaged flying capacitor and buck-boost and calibrated buck-boost; flying-capacitor worst current error 3.37 %, buck-boost in-range worst 4.88 % averaged and 3.90 % calibrated.
- `ltspice_loss_breakdown.png`: loss per component at the baseline link (buck-boost 55 µH, 30 kHz, synchronous and Schottky; flying capacitor 5.5 mF, 5.5 kHz); the synchronous buck-boost loses 25.8 mW in total with 1.76 mW of gate drive, the Schottky version 74.8 mW with 52.1 mW in the diode.
- `baseline_assumed_vs_fitted.png` (regenerated): baseline time and energy with the assumed and the fitted parameters, now with the circuit-calibrated models; buck-boost 1.09 h and flying capacitor 2.00 h.
- `deep_surrogates.png`: test R² of boosting, MLP, cell GRU, cell Transformer and the physics-residual versions for balancing time (all techniques and flying capacitor) and energy; boosting 0.9818, cell GRU 0.9524, boosting on the physics residual 0.9997, cell GRU on the physics residual 0.9997.
- `deep_trajectories.png`: simulated against predicted SOC spread trajectories of three held-out packs per technique for the GRU and Transformer decoders; spread RMSE 0.218 and 0.185 percentage points.
- `deep_soc.png`: nominal SOC RMSE of the 2RC EKF, the augmented GRU and the EKF plus GRU hybrid on Test A and Test B, and the hybrid and GRU RMSE against weight bit width; hybrid 0.89 % and 1.43 %, EKF 2.06 % and 2.11 %.
- `deep_rl_second_attempt.png`: usable pack capacity per controller (none, rule, greedy, residual PPO, imitation PPO, ideal) on the four test cycles for 4 and 8 cells; the learned controllers stay within 0.0006 reward units of the rule on 4 cells.
