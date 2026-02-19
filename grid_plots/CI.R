library(tidyverse)
n_intervals = 100
xg_ate_no_cross = read_csv("ate_experiment/gradient_boosting_experiment/results/no_cross_fit_results_500.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='Gradient boosting ATE', c_fit = '& No cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))
xg_ate_cross = read_csv("ate_experiment/gradient_boosting_experiment/results/cross_fit_results_500.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='Gradient boosting ATE', c_fit = '& 5 fold cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))
xg_ase_no_cross = read_csv("ase_experiment/gradient_boosting_experiment/results/no_cross_fit_results_500.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='Gradient boosting ASE', c_fit = '& No cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))
xg_ase_cross = read_csv("ase_experiment/gradient_boosting_experiment/results/cross_fit_results_500.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='Gradient boosting ASE', c_fit = '& 5 fold cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))





lasso_ate_no_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='LASSO ATE', c_fit = '& No cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))
lasso_ate_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='LASSO ATE', c_fit = '& 5 fold cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))
lasso_ase_no_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='LASSO ASE', c_fit = '& No cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))
lasso_ase_cross = read_csv("ate_experiment/LASSO_experiment/results/no_cross_fit_results_300.csv") %>%
  select(truth, propensity_estimate, riesz_estimate, propensity_lower, propensity_upper, riesz_lower, riesz_upper
  ) %>%
  pivot_longer(cols = -truth,names_to = c("estimator", ".value"),names_sep = "_") %>%
  mutate(type='LASSO ASE', c_fit = '& 5 fold cross-fitting', estimator=ifelse(estimator=='riesz', 'Direct', 'Indirect'))



truths = tibble(value=c(xg_ase_cross$truth[1], xg_ate_cross$truth[1], lasso_ase_cross$truth[1], lasso_ate_cross$truth[1]), type=c('Gradient boosting ASE', 'Gradient boosting ATE', 'LASSO ASE', 'LASSO ATE'))


data = rbind(xg_ate_no_cross, xg_ate_cross, xg_ase_cross, xg_ase_no_cross,
             lasso_ate_no_cross, lasso_ate_cross, lasso_ase_cross, lasso_ase_no_cross) %>%
  mutate(estimator = paste(estimator, c_fit)) %>%
  arrange(type, estimator)
summary_data = data %>% 
  group_by(type, estimator) %>%
  summarise(coverage=mean(lower <= truth & upper >= truth), med = median(upper-lower), .groups = "drop")

rows = c()
for (i in 1:(nrow(data)/1000)){
  rows = c(rows, 1:n_intervals + (i-1)*1000)
}
plot_data = data[rows,] %>% mutate(Estimator=estimator)

label_data <- summary_data %>%
  mutate(
    label = sprintf("Coverage: %.2f\nMedian length: %.2f", coverage, med)
  )

y_positions <- plot_data %>%
  group_by(type) %>%
  summarise(
    y_pos = max(upper),
    .groups = "drop"
  )
x_positions <- plot_data %>%
  mutate(index = seq_along(lower)) %>%
  group_by(type, estimator) %>%
  summarise(
    x_pos = min(index),
    .groups = "drop"
  )

label_data <- label_data %>%
  left_join(y_positions, by = 'type') %>%
  left_join(x_positions, by=c('type', 'estimator'))

ggplot(plot_data)+
  #geom_point(aes(x=seq_along(lower), y=estimate, color=Estimator))+
  geom_linerange(aes(x=seq_along(lower), ymin=lower, ymax=upper, color=Estimator),
                 )+
  theme_classic()+
  xlab('')+
  facet_wrap(~type, scale='free')+
  #geom_hline(data=truths, aes(yintercept=value), color='darkblue', size=0.7)+
  #theme(legend.position = "bottom")+
  geom_text(data = label_data,
            aes(x = x_pos,
                y = y_pos*1.1,
                label = label,
                color = estimator
                ),
            hjust = 0,
            vjust = 1,
            size = 3,
            show.legend = FALSE)
  
