
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
import math

@dataclass
class Layer:
    name: str
    dtype_bytes: int = 2  # default FP16

    def flops(self) -> int:
        raise NotImplementedError

    def bytes_moved(self) -> int:
        """Very low-fidelity: assume inputs, weights, and outputs each read/written once from DRAM."""
        raise NotImplementedError

@dataclass
class Conv2D(Layer):
    N: int = 1
    C: int = 1
    H: int = 1
    W: int = 1
    K: int = 1
    R: int = 1
    S: int = 1
    stride: int = 1
    padding: int = 0
    dilation: int = 1

    def out_dims(self):
        H_eff = (self.H + 2*self.padding - self.dilation*(self.R-1) - 1) // self.stride + 1
        W_eff = (self.W + 2*self.padding - self.dilation*(self.S-1) - 1) // self.stride + 1
        return self.N, self.K, H_eff, W_eff

    def flops(self) -> int:
        N, K, Ho, Wo = self.out_dims()
        # Conv FLOPs (MACs*2). For latency we often count MACs; keep FLOPs = MACs*2
        macs = N * K * Ho * Wo * self.C * self.R * self.S
        return 2 * macs

    def macs(self) -> int:
        N, K, Ho, Wo = self.out_dims()
        return N * K * Ho * Wo * self.C * self.R * self.S

    def bytes_moved(self) -> int:
        # Input tensor
        B_in = self.N * self.C * self.H * self.W * self.dtype_bytes
        # Weights
        B_w = self.K * self.C * self.R * self.S * self.dtype_bytes
        # Output
        N, K, Ho, Wo = self.out_dims()
        B_out = N * K * Ho * Wo * self.dtype_bytes
        return B_in + B_w + B_out

@dataclass
class MatMul(Layer):
    # Computes: [M x K] @ [K x N] = [M x N]
    M: int = 1
    K: int = 1
    Nn: int = 1  # 'N' conflicts with batch above; use Nn

    def flops(self) -> int:
        # GEMM FLOPs: 2*M*K*N
        return 2 * self.M * self.K * self.Nn

    def macs(self) -> int:
        return self.M * self.K * self.Nn

    def bytes_moved(self) -> int:
        A = self.M * self.K * self.dtype_bytes
        B = self.K * self.Nn * self.dtype_bytes
        C = self.M * self.Nn * self.dtype_bytes
        return A + B + C

LayerLike = Union[Conv2D, MatMul]

def layer_from_dict(d: Dict[str, Any], default_dtype_bytes: int = 2) -> LayerLike:
    t = d['type'].lower()
    common = {'name': d.get('name', t), 'dtype_bytes': d.get('dtype_bytes', default_dtype_bytes)}
    if t == 'conv2d':
        return Conv2D(
            name=common['name'], dtype_bytes=common['dtype_bytes'],
            N=d.get('N',1), C=d['C'], H=d['H'], W=d['W'],
            K=d['K'], R=d['R'], S=d['S'],
            stride=d.get('stride',1), padding=d.get('padding',0), dilation=d.get('dilation',1)
        )
    elif t == 'matmul' or t == 'gemm':
        return MatMul(
            name=common['name'], dtype_bytes=common['dtype_bytes'],
            M=d['M'], K=d['K'], Nn=d.get('N', d.get('Nn',1))
        )
    else:
        raise ValueError(f"Unsupported layer type: {d['type']}")

def model_from_json(path: str) -> List[LayerLike]:
    import json
    with open(path, 'r') as f:
        model = json.load(f)
    dtype_bytes = model.get('dtype_bytes', 2)
    layers = [layer_from_dict(op, default_dtype_bytes=dtype_bytes) for op in model['ops']]
    return layers
