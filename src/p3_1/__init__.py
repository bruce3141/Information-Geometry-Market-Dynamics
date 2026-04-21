"""
p3_1 - Reproducibility code for Dean (2026), 'Information Geometry of Market
Dynamics: A Pareto Frontier from Contact Geometry'.

Modules:
  phase_space               - Shared phase-space construction (with fencepost fix)
  falsification_test        - Appendix B, 17-market cross-market validation
  consistency_tests         - Appendix B §B.5, four-way Lyapunov-consistency battery
  figure3                   - Appendix B §B.6, phase-plane flow alignment
  underdamped_test          - §6.2 underdamped-regime candidate sweep
  pareto_bound_verification - Appendix C, 4 numerical Pareto-frontier tests
  sharpe_optimization       - §4.2 footnote, Sharpe-vs-R^2 attractor demo
  figures                   - All paper figures (1a, 1b, 2)
"""
__version__ = "0.60.0"
