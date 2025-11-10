
import argparse, os
from accel_sim.core.config import load_config
from accel_sim.ir.layers import model_from_json
from accel_sim.sims.analytical import run_analytical
from accel_sim.reports.typing import print_report, to_csv

def main():
    p = argparse.ArgumentParser(description="Low-fidelity ML accelerator analytical simulator")
    p.add_argument('--config', required=True, help='YAML accelerator config')
    p.add_argument('--model', required=True, help='JSON model description')
    p.add_argument('--csv_out', default='', help='Optional CSV output path')
    args = p.parse_args()

    cfg = load_config(args.config)
    layers = model_from_json(args.model)
    est = run_analytical(layers, cfg)
    print_report(est)
    if args.csv_out:
        to_csv(est, args.csv_out)
        print("\nSaved CSV:", args.csv_out)

if __name__ == '__main__':
    main()
