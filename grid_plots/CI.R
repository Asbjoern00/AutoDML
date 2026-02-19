library(tidyverse)

n_intervals <- 100

# Helper function to load a single experiment
load_ci_data <- function(file_path, type_label, cross_label) {
  read_csv(file_path, show_col_types = FALSE) %>%
    select(truth, propensity_estimate, riesz_estimate,
           propensity_lower, propensity_upper, riesz_lower, riesz_upper) %>%
    pivot_longer(
      cols = -truth,
      names_to = c("estimator", ".value"),
      names_sep = "_"
    ) %>%
    mutate(
      type = type_label,
      c_fit = cross_label,
      estimator = ifelse(estimator == "riesz", "Direct", "Indirect")
    )
}

# Load all 500-sample experiments
xg_ate_no_cross <- load_ci_data(
  "ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results_500.csv",
  "Gradient Boosting ATE", "& No cross-fitting"
)
xg_ate_cross <- load_ci_data(
  "ate_experiment/gradient_boosting_experiment/results/cross_fit_results_500.csv",
  "Gradient Boosting ATE", "& 5 fold cross-fitting"
)
xg_ase_no_cross <- load_ci_data(
  "ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results_500.csv",
  "Gradient Boosting ASE", "& No cross-fitting"
)
xg_ase_cross <- load_ci_data(
  "ase_experiment/gradient_boosting_experiment/results/cross_fit_results_500.csv",
  "Gradient Boosting ASE", "& 5 fold cross-fitting"
)
lasso_ate_no_cross <- load_ci_data(
  "ate_experiment/LASSO_experiment/results/no_cross_fit_results_500.csv",
  "LASSO ATE", "& No cross-fitting"
)
lasso_ate_cross <- load_ci_data(
  "ate_experiment/LASSO_experiment/results/cross_fit_results_500.csv",
  "LASSO ATE", "& 5 fold cross-fitting"
)
lasso_ase_no_cross <- load_ci_data(
  "ase_experiment/LASSO_experiment/results/no_cross_fit_results_500.csv",
  "LASSO ASE", "& No cross-fitting"
)
lasso_ase_cross <- load_ci_data(
  "ase_experiment/LASSO_experiment/results/cross_fit_results_500.csv",
  "LASSO ASE", "& 5 fold cross-fitting"
)

# Combine all data
data <- bind_rows(
  xg_ate_no_cross, xg_ate_cross,
  xg_ase_no_cross, xg_ase_cross,
  lasso_ate_no_cross, lasso_ate_cross,
  lasso_ase_no_cross, lasso_ase_cross
) %>%
  mutate(estimator = paste(estimator, c_fit)) %>%
  arrange(type, estimator)

# Compute coverage and median CI length
summary_data <- data %>%
  group_by(type, estimator) %>%
  summarise(
    coverage = mean(lower <= truth & upper >= truth),
    med = median(upper - lower),
    .groups = "drop"
  )

# Subset for plotting (n_intervals per simulation)
plot_data <- data %>%
  group_by(type, estimator) %>%   # keep each estimator block separate
  slice(1:n_intervals) %>%        # take the first `n_intervals` rows per group
  ungroup() %>%
  mutate(Estimator = estimator)


# Label positions
label_data <- summary_data %>%
  mutate(label = sprintf("Coverage: %.2f\nMedian length: %.2f", coverage, med))

y_positions <- plot_data %>%
  group_by(type) %>%
  summarise(y_pos = max(upper), .groups = "drop")

x_positions <- plot_data %>%
  mutate(index = seq_along(lower)) %>%
  group_by(type, estimator) %>%
  summarise(x_pos = min(index), .groups = "drop")

label_data <- label_data %>%
  left_join(y_positions, by = "type") %>%
  left_join(x_positions, by = c("type", "estimator"))

# Plot
ggplot(plot_data) +
  geom_linerange(aes(x = seq_along(lower), ymin = lower, ymax = upper, color = Estimator)) +
  theme_classic() +
  xlab('') +
  facet_wrap(~type, scales = "free") +
  geom_text(
    data = label_data,
    aes(x = x_pos, y = y_pos * 1.1, label = label, color = estimator),
    hjust = 0, vjust = 1, size = 3, show.legend = FALSE
  )
