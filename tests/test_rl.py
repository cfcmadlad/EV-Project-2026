import numpy as np

from src.balancing import buck_boost_currents
from src.cell import build_ocv_curve, open_circuit_voltage
from src.config import CONFIG
from src.drive_cycle import constant_program
from src.pack import sample_packs
from src.rl_env import BalancingEnv, no_balancing, rollout, rule_based
from src.rl_sim import advance, build_sim, commanded_currents, observe

PACKS = sample_packs(6, np.random.default_rng(11), CONFIG.cell, CONFIG.pack)
CURVE = build_ocv_curve(CONFIG.cell)


def flow_inputs():
    soc = PACKS.soc0
    ocv = open_circuit_voltage(soc, CURVE)
    inverse_lf = np.full(6, 1.0 / (55e-6 * 30e3))
    return soc, ocv, ocv.copy(), inverse_lf


def test_full_rule_command_reproduces_the_averaged_link_model() -> None:
    soc, ocv, emf, inverse_lf = flow_inputs()
    reference = buck_boost_currents(
        soc, ocv, emf, PACKS.r0_ohm, inverse_lf, CONFIG.buck_boost, CONFIG.sim.link_soc_deadband
    )
    difference = soc[:, :-1] - soc[:, 1:]
    active = np.abs(difference) > CONFIG.sim.link_soc_deadband
    command = np.sign(difference) * active * (CONFIG.buck_boost.duty / CONFIG.rl.d_max)
    flow, _ = commanded_currents(
        ocv, emf, PACKS.r0_ohm, command, inverse_lf, CONFIG.buck_boost, CONFIG.rl
    )
    assert np.allclose(flow.cell_current_a, reference.cell_current_a, rtol=1e-12, atol=1e-15)


def test_duty_is_clipped_to_discontinuous_conduction() -> None:
    soc, ocv, emf, inverse_lf = flow_inputs()
    emf = emf.copy()
    emf[:, 1] *= 0.6
    command = np.ones((6, 3))
    _, duty = commanded_currents(
        ocv, emf, PACKS.r0_ohm, command, inverse_lf, CONFIG.buck_boost, CONFIG.rl
    )
    ratio = np.where(command > 0, emf[:, :-1] / emf[:, 1:], emf[:, 1:] / emf[:, :-1])
    assert np.all(duty * (1.0 + ratio) <= CONFIG.rl.dcm_margin + 1e-12)
    assert duty.max() <= CONFIG.rl.d_max + 1e-12


def test_zero_command_leaves_charge_untouched_and_reward_matches_ledger() -> None:
    program = constant_program(3.0, 6, 3600)
    sim = build_sim(PACKS, program, np.zeros(PACKS.soc0.shape), CONFIG)
    total = np.zeros(6)
    for _ in range(40):
        reward, done = advance(sim, np.zeros((6, 3)))
        total += reward
    assert np.allclose(sim.state.removed_j, 0.0)
    assert np.allclose(total, sim.state.pack_charge_ah)


def test_rule_based_beats_no_balancing_on_a_constant_load() -> None:
    program = constant_program(3.0, 6, 3600)
    outputs = {}
    for name, controller in (("none", no_balancing), ("rule", rule_based)):
        sim = build_sim(PACKS, program, np.zeros(PACKS.soc0.shape), CONFIG)
        outputs[name] = rollout(sim, controller, 400)
    assert outputs["rule"]["pack_ah"].mean() >= outputs["none"]["pack_ah"].mean()
    assert outputs["rule"]["outside_s"].max() == 0.0


def test_gym_environment_runs_an_episode() -> None:
    profiles = {"profiles": {"Mixed1": {"current_a": [3.0] * 4000}}}
    env = BalancingEnv(CONFIG, profiles, ("Mixed1",))
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    done = truncated = False
    steps = 0
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        steps += 1
    assert done and info["pack_ah"] > 0.3 and steps < 500
    assert observe(env.sim).shape[1] == obs.shape[0]


def test_cached_rl_results_are_current_and_policies_are_small() -> None:
    from src.rl_report import POLICY_DIR, rl_block

    block = rl_block(CONFIG)
    assert block["status"] == "ok"
    assert block["limit_violation_seconds_total"] == 0.0
    assert all(path.stat().st_size < 5_000_000 for path in POLICY_DIR.glob("*.zip"))
