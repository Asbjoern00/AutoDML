import torch




def avg_der_fuctional(data,evaluator):
    epsilon = 0.0001
    shift_down, shift_up = data.get_counterfactual_datasets(epsilon)
    out_up, out_down = evaluator(shift_up), evaluator(shift_down)
    return (out_up - out_down)/(2*epsilon)

