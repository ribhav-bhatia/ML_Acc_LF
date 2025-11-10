
from typing import Dict, Any, List
import math
from dataclasses import dataclass
from ..core.config import AcceleratorConfig
from ..ir.layers import LayerLike

@dataclass
class LayerEstimate:
    name: str
    macs: int
    flops: int
    bytes_dram: int
    t_compute_s: float
    t_memory_s: float
    t_layer_s: float
    energy_pJ: float

def estimate_layer(layer: LayerLike, cfg: AcceleratorConfig) -> LayerEstimate:
    macs = layer.macs()
    flops = layer.flops()
    # Compute time
    cycles = math.ceil(macs / max(1, cfg.macs_per_cycle))
    t_compute = cycles / cfg.cycles_per_second

    # Memory time (single-level DRAM model; low-fidelity)
    bytes_total = layer.bytes_moved()
    t_mem = bytes_total / cfg.dram_bw_Bps

    # Layer time is the overlap-aware max
    t_layer = max(t_compute, t_mem)

    # Energy: MAC energy + DRAM traffic energy
    e_mac = macs * cfg.e_mac_pJ
    e_mem = bytes_total * cfg.e_dram_pJ_per_B
    energy = e_mac + e_mem

    return LayerEstimate(
        name=layer.name,
        macs=macs,
        flops=flops,
        bytes_dram=bytes_total,
        t_compute_s=t_compute,
        t_memory_s=t_mem,
        t_layer_s=t_layer,
        energy_pJ=energy,
    )

@dataclass
class ModelEstimate:
    layers: List[LayerEstimate]

    @property
    def total_latency_s(self) -> float:
        return sum(l.t_layer_s for l in self.layers)

    @property
    def total_energy_pJ(self) -> float:
        return sum(l.energy_pJ for l in self.layers)

def estimate_model(layers: List[LayerLike], cfg: AcceleratorConfig) -> ModelEstimate:
    ests = [estimate_layer(L, cfg) for L in layers]
    return ModelEstimate(ests)
