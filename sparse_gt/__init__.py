"""BP + sparse graph transformer decoding for LDPC codes (built from scratch).

Step 2 layer: code loading, BPSK/AWGN channel, batched differentiable
sum-product BP, error-rate simulation, and the top-k coverage analysis.
"""

from .bp import BPResult, SumProductBP
from .channel import awgn_llr, ebn0_to_sigma
from .codes import LDPCCode, available_codes, load_code

__all__ = ["BPResult", "SumProductBP", "awgn_llr", "ebn0_to_sigma",
           "LDPCCode", "available_codes", "load_code"]
