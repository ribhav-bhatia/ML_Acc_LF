
from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class AcceleratorConfig:
    pe_rows: int
    pe_cols: int
    macs_per_pe_per_cycle: int
    frequency_GHz: float

    # Memory system (low-fidelity, single-level BW model + energy params)
    dram_bw_Bps: float
    e_mac_pJ: float
    e_dram_pJ_per_B: float

    # Optional bookkeeping
    name: str = "GenericSim"

    @property
    def macs_per_cycle(self) -> int:
        return self.pe_rows * self.pe_cols * self.macs_per_pe_per_cycle

    @property
    def cycles_per_second(self) -> float:
        return self.frequency_GHz * 1e9

def load_config(path: str) -> AcceleratorConfig:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return AcceleratorConfig(
        pe_rows=cfg['pe_array']['rows'],
        pe_cols=cfg['pe_array']['cols'],
        macs_per_pe_per_cycle=cfg['pe_array'].get('mac_per_cycle', 1),
        frequency_GHz=cfg.get('frequency_GHz', 1.0),
        dram_bw_Bps=cfg['dram']['bw_Bps'],
        e_mac_pJ=cfg['energy']['e_mac_pJ'],
        e_dram_pJ_per_B=cfg['energy']['e_dram_pJ_per_B'],
        name=cfg.get('name','GenericSim')
    )
