from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy

eps, _ = compute_dp_sgd_privacy(60000, 64, 0.623, 10, 1e-5)
print("Epsilon:", eps)