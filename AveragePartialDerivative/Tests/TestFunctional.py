import torch
from AveragePartialDerivative.AveragePartialDerivativeFunctional import avg_der_functional
def test_avg_der_functional():
    # Create test input: shape (N, 2)
    data = torch.tensor([
        [2.0, 3.0],
        [4.0, 5.0],
        [-1.0, 0.0]
    ])

    # Define evaluator function: f(x) = x0^2 + x1
    def evaluator(x):
        return x[:, 0]**2 + x[:, 1]

    grads = avg_der_functional(data, evaluator)

    # Expected gradient: df/dx0 = 2 * x0
    expected = 2 * data[:, 0].view(-1, 1)

    assert torch.allclose(grads, expected)
