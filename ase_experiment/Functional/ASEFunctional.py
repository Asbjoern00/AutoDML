def ase_functional(data, evaluator):

    data_shifted,data_control = data.get_counterfactual_datasets()
    yshift = evaluator(data_shifted)
    y0 = evaluator(data_control)

    return yshift - y0