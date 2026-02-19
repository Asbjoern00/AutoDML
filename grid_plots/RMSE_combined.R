library(tidyverse)
xg_125_ate_cross = read_csv("ate_experiment/gradient_boosting_experiment/results/cross_fit_results_125.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_500_ate_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/cross_fit_results_500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1000_ate_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/cross_fit_results.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1000, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1500_ate_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/cross_fit_results_1500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1500, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_2000_ate_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/cross_fit_results_2000.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )



xg_125_ase_cross = read_csv("ase_experiment/gradient_boosting_experiment/results/cross_fit_results_125.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_500_ase_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/cross_fit_results_500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1000_ase_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/cross_fit_results.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1000, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1500_ase_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/cross_fit_results_1500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1500, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_2000_ase_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/cross_fit_results_2000.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )


lasso_125_ate_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='LASSO ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_500_ate_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='LASSO ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_2000_ate_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='LASSO ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )


lasso_125_ase_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='LASSO ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_500_ase_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='LASSO ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_2000_ase_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='LASSO ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )


cross_data = rbind(xg_125_ate_cross, xg_500_ate_cross, xg_1000_ate_cross, xg_1500_ate_cross, xg_2000_ate_cross, xg_125_ase_cross, xg_500_ase_cross, xg_1000_ase_cross, xg_1500_ase_cross, xg_2000_ase_cross,
                   lasso_125_ate_cross, lasso_500_ate_cross, lasso_2000_ate_cross, lasso_125_ase_cross, lasso_500_ase_cross, lasso_2000_ase_cross)

cross_data = cross_data %>% mutate(Estimator = ifelse(estimator=='riesz', 'Direct with cross-fitting', 'Indirect with cross-fitting'))



xg_125_ate_no_cross = read_csv("ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results_125.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_500_ate_no_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results_500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1000_ate_no_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1000, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1500_ate_no_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results_1500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1500, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_2000_ate_no_cross <- read_csv("ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results_2000.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='Gradient Boosting ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )



xg_125_ase_no_cross = read_csv("ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results_125.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_500_ase_no_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results_500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1000_ase_no_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1000, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_1500_ase_no_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results_1500.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 1500, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
xg_2000_ase_no_cross <- read_csv("ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results_2000.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='Gradient Boosting ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )


lasso_125_ate_no_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='LASSO ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_500_ate_no_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='LASSO ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_2000_ate_no_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='LASSO ATE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )


lasso_125_ase_no_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 125, type='LASSO ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_500_ase_no_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 500, type='LASSO ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )
lasso_2000_ase_no_cross <- read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv")%>%
  mutate(riesz_residual = (riesz_estimate-truth)^2, propensity_residual = (propensity_estimate-truth)^2) %>%
  summarise(riesz_mse = mean(riesz_residual), riesz_sd = sd(riesz_residual) / sqrt(1000), indirect_mse = mean(propensity_residual), indirect_sd = sd(propensity_residual) / sqrt(1000)) %>%
  mutate(n = 2000, type='LASSO ASE') %>%
  pivot_longer(
    cols = c(riesz_mse, riesz_sd, indirect_mse, indirect_sd),
    names_to = c("estimator", ".value"),
    names_sep = "_"
  )


no_cross_data = rbind(xg_125_ate_no_cross, xg_500_ate_no_cross, xg_1000_ate_no_cross, xg_1500_ate_no_cross, xg_2000_ate_no_cross, xg_125_ase_no_cross, xg_500_ase_no_cross, xg_1000_ase_no_cross, xg_1500_ase_no_cross, xg_2000_ase_no_cross,
                      lasso_125_ate_no_cross, lasso_500_ate_no_cross, lasso_2000_ate_no_cross, lasso_125_ase_no_cross, lasso_500_ase_no_cross, lasso_2000_ase_no_cross)

no_cross_data = no_cross_data %>% mutate(Estimator = ifelse(estimator=='riesz', 'Direct without cross-fitting', 'Indirect without cross-fitting'))

plot_data = rbind(cross_data, no_cross_data)


ggplot(plot_data) +
  geom_errorbar(aes(colour=Estimator, x=n, ymin = sqrt(n*(mse-1.96*sd)), ymax = sqrt(n*(mse+1.96*sd))), width = 100, alpha=0.5)+
  geom_line(aes(x=n, y=sqrt(n*mse), colour = Estimator), linetype='dashed')+
  geom_point(aes(x=n, y = sqrt(n*mse), colour = Estimator), size=3)+
  theme_classic()+
  ylab(expression(sqrt(n) * RMSE))+
  facet_wrap(~type, scale='free')+
  ggtitle('RMSE at different sample sizes')
