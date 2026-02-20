library(tidyverse)

process_results <- function(base_path, sizes, type_label, cross = TRUE) {
  
  fit_type <- ifelse(cross, "cross_fit_results_", "no_cross_fit_results_")
  
  map_dfr(sizes, function(n) {
    
    file_path <- paste0(base_path, fit_type, n, ".csv")
    
    df <- read_csv(file_path, show_col_types = FALSE)
    
    # ---- Handle inconsistent column names ----
    if ("propensity_estimate" %in% names(df)) {
      df <- df %>% rename(propensity = propensity_estimate)
    } else if ("indirect" %in% names(df)) {
      df <- df %>% rename(propensity = indirect)
    } else {
      stop(paste("No propensity/indirect column in", file_path))
    }
    
    df %>%
      mutate(
        riesz_residual = (riesz_estimate - truth)^2,
        propensity_residual = (propensity - truth)^2
      ) %>%
      summarise(
        riesz_mse = mean(riesz_residual),
        riesz_sd = sd(riesz_residual) / sqrt(1000),
        indirect_mse = mean(propensity_residual),
        indirect_sd = sd(propensity_residual) / sqrt(1000)
      ) %>%
      mutate(n = n, type = type_label) %>%
      pivot_longer(
        cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
        names_to = c("estimator", ".value"),
        names_sep = "_"
      )
  })
}



sizes <- c(125, 500, 2000)

# CROSS-FIT
cross_data <- bind_rows(
  
  process_results("ate_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ATE", cross = TRUE),
  
  process_results("ase_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ASE", cross = TRUE),
  
  process_results("ate_experiment/LASSO_experiment/results/",
                  sizes, "LASSO ATE", cross = TRUE),
  
  process_results("ase_experiment/LASSO_experiment/results/",
                  sizes, "LASSO ASE", cross = TRUE)
)

cross_data <- cross_data %>%
  mutate(Estimator = ifelse(estimator == "riesz",
                            "Direct with cross-fitting",
                            "Indirect with cross-fitting"))


no_cross_data <- bind_rows(
  
  process_results("ate_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ATE", cross = FALSE),
  
  process_results("ase_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ASE", cross = FALSE),
  
  process_results("ate_experiment/LASSO_experiment/results/",
                  sizes, "LASSO ATE", cross = FALSE),
  
  process_results("ase_experiment/LASSO_experiment/results/",
                  sizes, "LASSO ASE", cross = FALSE)
)

no_cross_data <- no_cross_data %>%
  mutate(Estimator = ifelse(estimator == "riesz",
                            "Direct without cross-fitting",
                            "Indirect without cross-fitting"))



plot_data <- bind_rows(cross_data, no_cross_data)

ggplot(plot_data) +
  geom_errorbar(
    aes(colour = Estimator,
        x = n,
        ymin = sqrt(n * (mse - 1.96 * sd)),
        ymax = sqrt(n * (mse + 1.96 * sd))),
    width = 100,
    alpha = 0.5
  ) +
  geom_line(
    aes(x = n,
        y = sqrt(n * mse),
        colour = Estimator),
    linetype = "dashed"
  ) +
  geom_point(
    aes(x = n,
        y = sqrt(n * mse),
        colour = Estimator),
    size = 3
  ) +
  theme_classic() +
  ylab(expression(sqrt(n) * RMSE)) +
  facet_wrap(~type, scales = "free") +
  ggtitle("RMSE at different sample sizes")

