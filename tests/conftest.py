"""Shared test setup: draw off-screen, so the suite never opens a window."""

import matplotlib

matplotlib.use("Agg")
