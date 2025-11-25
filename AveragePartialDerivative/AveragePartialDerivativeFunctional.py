import torch

def avg_der_functional(data, evaluator):
    T = torch.autograd.Variable(data[:, [0]], requires_grad=True)
    input = torch.cat([T, data[:, 1:]], dim=1)
    output = evaluator(input)
    gradients = torch.autograd.grad(outputs=output, inputs=T,
                          grad_outputs=torch.ones(output.size()),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return gradients

