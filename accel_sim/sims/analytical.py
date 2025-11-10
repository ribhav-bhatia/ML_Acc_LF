
from typing import List
from ..core.config import AcceleratorConfig
from ..ir.layers import LayerLike
from ..mapping.costing import estimate_model, ModelEstimate

def run_analytical(layers: List[LayerLike], cfg: AcceleratorConfig) -> ModelEstimate:
    return estimate_model(layers, cfg)
