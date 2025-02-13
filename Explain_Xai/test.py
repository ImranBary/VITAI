#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

# Number of integration steps for Integrated Gradients
IG_N_STEPS = 35

def integrated_gradients(model, X, baseline=None, device="cpu"):
    """
    Compute Integrated Gradients attributions for all rows in X using IG_N_STEPS steps.
    If no baseline is provided, use the median of X as reference.
    """
    # Use the underlying model stored in 'network'
    if hasattr(model, "network"):
        underlying_model = model.network
    else:
        raise AttributeError("TabNetRegressor object has no attribute 'network'")
    
    ig = IntegratedGradients(underlying_model)
    X_t = torch.tensor(X, dtype=torch.float, device=device)
    
    if baseline is None:
        baseline_array = np.median(X, axis=0)
        baseline_t = torch.tensor(baseline_array, dtype=torch.float, device=device)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float, device=device)
    
    # Use the correct keyword argument "baselines" (not "baseline")
    attrs_t = ig.attribute(X_t, baselines=baseline_t, n_steps=IG_N_STEPS)
    return attrs_t.detach().cpu().numpy()

# Create a dummy TabNetRegressor-like class
class DummyTabNetRegressor:
    def __init__(self):
        # Underlying network: a simple linear layer with 10 inputs and 1 output.
        self.network = nn.Linear(10, 1)
    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            return self.network(X_t).numpy()

def main():
    # Generate dummy data: 5 samples, 10 features each
    X_sample = np.random.rand(5, 10)
    dummy_model = DummyTabNetRegressor()
    
    # Compute Integrated Gradients attributions on the sample data
    attrs = integrated_gradients(dummy_model, X_sample, baseline=None, device="cpu")
    print("Integrated Gradients Attributions:")
    print(attrs)

if __name__ == '__main__':
    main()
