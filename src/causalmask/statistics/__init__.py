"""Statistical tools for CausalMask evaluation: bootstrap, paired tests, multiplicity."""

from causalmask.statistics.bootstrap import (
    group_aware_bootstrap_ci,
    bootstrap_mean_ci,
    paired_bootstrap_diff,
)
