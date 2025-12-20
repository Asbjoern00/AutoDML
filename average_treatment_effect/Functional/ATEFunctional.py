def ate_functional(data, evaluator):

    data_treated,data_control = data.get_counterfactual_datasets()
    y1 = evaluator(data_treated)
    y0 = evaluator(data_control)

    return y1 - y0