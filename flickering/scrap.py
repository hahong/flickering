"""NOT USED"""
import numpy as np
from .base import grouper

if False:
    def mse(signal, r=0.15, scales=None, scale_num=6, scale_min=1,
            sampen_func=None, output='sum'):
        """Perform DFA analysis.

    Input
    -----
    signal: signal
    r: tolerance for template matching
    scales: a vector of scales

    Reference
    ---------
    http://www.physionet.org/physiotools/mse/tutorial/tutorial.pdf
    """
        if len(signal.shape) != 1:
            raise ValueError('"signal" must be 1D.')

        if scales in None:
            scales = range(scale_min, scale_min + scale_num)

        
