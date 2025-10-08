"""
DAMA-BAX Example

This directory contains the implementation of the DAMA (Dynamic and Momentum Aperture)
optimization problem as an example of using the BAX framework.

All DAMA-specific modules are contained in this directory.
Core BAX framework modules (bax_core, da_NN) are imported from ../../core.
"""

import os
import sys

# Add local directory first, then core (so local modules take precedence)
local_path = os.path.dirname(__file__)
core_path = os.path.join(os.path.dirname(__file__), '../../core')
if local_path not in sys.path:
    sys.path.insert(0, local_path)
if core_path not in sys.path:
    sys.path.insert(1, core_path)
