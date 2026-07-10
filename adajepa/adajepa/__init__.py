"""Isolated reproduction of AdaJEPA (arXiv 2606.32026).

An adaptive latent world model: JEPA encoder+predictor trained on reward-free
offline trajectories, planned with MPC (CEM or GD), and adapted at test time
inside the plan-execute-adapt-replan loop.

This package is deliberately self-contained (no imports from the ActionFleet
``lab/`` or ``backend/`` trees) so the findings can back a standalone paper.
"""

__version__ = "0.1.0"
