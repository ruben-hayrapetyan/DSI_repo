from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy

eps, _ = compute_dp_sgd_privacy(50000, 64, 0.68, 25, 1e-5)
print("Epsilon:", eps)