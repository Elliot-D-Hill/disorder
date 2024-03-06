import torch

counts = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
abundance = counts / counts.sum()
similarity = torch.tensor(
    [
        [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
        [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
        [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
        [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
        [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
    ]
)
frequency_results = {
    "alpha": [6.0],
    "rho": [1.0],
    "beta": [1.0],
    "gamma": [6.0],
    "normalized_alpha": [3.0],
    "normalized_rho": [0.5],
    "normalized_beta": [2.0],
}
similarity_results = {
    "alpha": [3.0],
    "rho": [2.05],
    "beta": [0.487805],
    "gamma": [1.463415],
    "normalized_alpha": [1.5],
    "normalized_rho": [1.025],
    "normalized_beta": [0.97561],
}
