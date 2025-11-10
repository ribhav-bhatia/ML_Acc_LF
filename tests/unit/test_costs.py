
import os, json, yaml
from accel_sim.core.config import load_config
from accel_sim.ir.layers import model_from_json
from accel_sim.sims.analytical import run_analytical

def test_demo_model():
    cfg = load_config(os.path.join('configs','accel','example.yaml'))
    layers = model_from_json(os.path.join('examples','models','demo.json'))
    est = run_analytical(layers, cfg)
    assert est.total_latency_s > 0
    assert est.total_energy_pJ > 0
    assert len(est.layers) == 2
