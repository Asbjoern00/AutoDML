import torch

# def avg_der_functional(data, evaluator):
#    T = torch.autograd.Variable(data.Treatments[:, [0]], requires_grad=True)
#    covariates = data.covariates_tensor
#    input = torch.cat([T, covariates], dim=1)
#    output = evaluator(input)
#    gradients = torch.autograd.grad(
#        outputs=output,
#        inputs=T,
#        grad_outputs=torch.ones(output.size()),
#        create_graph=True,
#        retain_graph=True,
#        only_inputs=True
#    )[0]
#    return gradients


def avg_der_fuctional(data,evaluator):
    epsilon = 0.001
    shift_down, shift_up = data.get_counterfactual_datasets(epsilon)
    out_up, out_down = evaluator(shift_up), evaluator(shift_down)
    return (out_up - out_down)/(2*epsilon)

