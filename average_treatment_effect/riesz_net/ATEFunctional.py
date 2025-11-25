def ate_functional(data, evaluator, treatment_index=0):
    x = data.clone()

    # Predict outcome under treatment T=1
    x_treated = x.clone()
    x_treated[:, treatment_index] = 1
    y1 = evaluator(x_treated)

    # Predict outcome under treatment T=0
    x_control = x.clone()
    x_control[:, treatment_index] = 0
    y0 = evaluator(x_control)

    return y1 - y0