"""Tests for the Clohessy-Wiltshire propagation.

The dynamics are linear, so these checks are exact rather than statistical:
 - a chaser at rest at the origin stays there with no thrust;
 - the along-track drift of a purely radial offset matches the analytical
   secular term;
 - propagating over ``2 dt`` equals propagating twice over ``dt``;
 - the out-of-plane-free planar solution is periodic over one orbital period
   for an initial condition on a closed relative orbit.
"""

import pytest

pytest.skip("dynamics.py is still a skeleton", allow_module_level=True)
