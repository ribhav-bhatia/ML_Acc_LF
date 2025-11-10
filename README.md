
# Low-Fidelity ML Accelerator Simulator (Analytical)

A minimal, **analytical** simulator in Python that estimates **latency** and **energy** for neural-network layers (Conv2D, MatMul) on a parameterized accelerator.

## Features
- Layer primitives: **Conv2D**, **MatMul (GEMM)**
- Configurable accelerator (PE array, frequency, DRAM bandwidth, energy params)
- Low-fidelity timing: `layer_time = max(compute_time, memory_time)`
- Energy model: `E = MACs * e_mac + bytes_dram * e_dram_per_B`
- CLI reporting + CSV export

## Install & Run
```bash
python3 main.py --config configs/accel/example.yaml --model examples/models/demo.json --csv_out results.csv
```

## Config (YAML)
```yaml
name: TinySIM
pe_array: {rows: 16, cols: 16, mac_per_cycle: 1}
frequency_GHz: 1.0
dram:
  bw_Bps: 100e9      # 100 GB/s
energy:
  e_mac_pJ: 0.2      # pJ per MAC (toy)
  e_dram_pJ_per_B: 200  # pJ per byte (toy)
```

## Model (JSON)
```json
{
  "dtype_bytes": 2,
  "ops": [
    {"type": "Conv2D", "name": "conv1", "N": 1, "C": 3, "H": 224, "W": 224, "K": 64, "R": 7, "S": 7, "stride": 2, "padding": 3},
    {"type": "MatMul", "name": "fc", "M": 1000, "K": 2048, "N": 1}
  ]
}
```

## Notes
- This is a **teaching/reference** baseline. It ignores on-chip SRAM, tiling, and reuse. Extend by adding SRAM BW/energy and splitting DRAM traffic by tensor.
- For fidelity upgrades, add: double-buffered tiles, NoC BW, SRAM banks, and per-level energy.
- Unit tests under `tests/unit` provide sanity checks.
