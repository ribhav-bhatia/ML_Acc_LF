
from typing import List
from ..mapping.costing import ModelEstimate, LayerEstimate

def humanize_time(s: float) -> str:
    if s < 1e-6: return f"{s*1e9:.2f} ns"
    if s < 1e-3: return f"{s*1e6:.2f} µs"
    if s < 1: return f"{s*1e3:.2f} ms"
    return f"{s:.3f} s"

def humanize_energy(pJ: float) -> str:
    if pJ < 1e3: return f"{pJ:.2f} pJ"
    if pJ < 1e6: return f"{pJ/1e3:.2f} nJ"
    if pJ < 1e9: return f"{pJ/1e6:.2f} µJ"
    if pJ < 1e12: return f"{pJ/1e9:.2f} mJ"
    return f"{pJ/1e12:.2f} J"

def print_report(est: ModelEstimate):
    print("Layer, MACs, Bytes(DRAM), ComputeTime, MemoryTime, LayerTime, Energy")
    for L in est.layers:
        print(f"{L.name}, {L.macs:,}, {L.bytes_dram:,}, {humanize_time(L.t_compute_s)}, {humanize_time(L.t_memory_s)}, {humanize_time(L.t_layer_s)}, {humanize_energy(L.energy_pJ)}")
    print("\nTOTAL Latency:", humanize_time(est.total_latency_s))
    print("TOTAL Energy:", humanize_energy(est.total_energy_pJ))

def to_csv(est: ModelEstimate, path: str):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["layer","macs","flops","bytes_dram","t_compute_s","t_memory_s","t_layer_s","energy_pJ"])
        for L in est.layers:
            w.writerow([L.name, L.macs, L.flops, L.bytes_dram, L.t_compute_s, L.t_memory_s, L.t_layer_s, L.energy_pJ])
        # add totals as a final row
        w.writerow(["TOTAL", sum(L.macs for L in est.layers), sum(L.flops for L in est.layers), sum(L.bytes_dram for L in est.layers), "", "", est.total_latency_s, est.total_energy_pJ])
