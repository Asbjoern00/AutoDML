from ate_experiment.LASSO_experiment.main_crossfit_highdim import run as run_main_crossfit
from ate_experiment.LASSO_experiment.main_no_crossfit_highdim import run as run_main_no_crossfit

if __name__ == "__main__":
    run_main_crossfit()
    run_main_no_crossfit()
