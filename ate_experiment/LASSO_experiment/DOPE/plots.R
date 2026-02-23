library(tidyverse)
res <- read_csv("ate_experiment/LASSO_experiment/DOPE/InformedLasso.csv")
df <- res %>% select(-c(truth, covered_outcome_adapted, covered_riesz_adapted)) %>% pivot_longer(cols = everything(), names_to = "Estimator",values_to = "Estimate") %>% mutate(Estimator = str_replace(Estimator,"_", " ")) %>% filter(Estimate != 0) %>% 
  mutate(Estimator = case_when(Estimator == "Riesz" ~ "Separate", TRUE ~ Estimator))

stats <- df %>% group_by(Estimator) %>% summarise(Bias = mean(Estimate)-1, RMSE = sqrt(mean((Estimate-1)^2)), Variance =  var(Estimate))
df %>%
  left_join(stats, by = "Estimator") %>% 
  mutate(Estimator = fct_reorder(Estimator, RMSE)) %>%
  ggplot(aes(Estimator, Estimate, fill = Estimator)) + geom_violin() + geom_hline(yintercept = 1, linetype = "dashed") + 
  geom_text(
    data = stats,
    aes(
      x = Estimator,
      y = Inf,
      label = sprintf(
        "Bias = %.3f\nRMSE = %.3f",
        Bias, RMSE
      )
    ),
    vjust = 1.1,
    size = 5
  )+
  ylim(0.4,1.7)+
  theme_classic()  + 
  ggtitle("Distribution of estimators for different methods") + 
  theme(
    plot.title   = element_text(size = 16, hjust = 0.5),
    axis.title   = element_text(size = 16),
    axis.text    = element_text(size = 14),
    strip.text   = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 12)
  )
