library(tidyverse)
library(patchwork)
no_cross_fit_results <- read_csv("ase_experiment/lasso_experiment/Results/no_cross_fit_results.csv")
cross_fit_results <- read_csv("ase_experiment/lasso_experiment/Results/cross_fit_results.csv")


no_cf_bias <- mean(no_cross_fit_results$plugin_estimate-no_cross_fit_results$truth)
cf_bias <- mean(cross_fit_results$plugin_estimate-no_cross_fit_results$truth)

no_cf_rmse <- sqrt(mean((no_cross_fit_results$plugin_estimate-no_cross_fit_results$truth)^2))
cf_rmse <- sqrt(mean((cross_fit_results$plugin_estimate-no_cross_fit_results$truth)^2))



a_rmse = mean((no_cross_fit_results$indirect_estimate - no_cross_fit_results$truth)^2)^(1/2)
a_bias = mean(no_cross_fit_results$indirect_estimate-no_cross_fit_results$truth)
a = ggplot(no_cross_fit_results)+
  geom_histogram(aes(x=(indirect_estimate-truth)/sqrt(indirect_variance/1000), y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  xlab(
    expression(
      frac(
        hat(psi) - psi(P),
        sqrt(hat(V) / n)
      )
    )
  ) +
  ggtitle('Indirect Riesz representer & No cross-fitting') +
  xlim(-5,5) +
  ylim(0, 1/2) +
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("RMSE: ", round(a_rmse,5)), 
           hjust = 1.1, vjust = 2.1, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Bias: ", round(a_bias,5)), 
           hjust = 1.1, vjust = 4.2, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in Bias: ", round(no_cf_bias,5)), 
           hjust = 1.1, vjust = 6.3, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in RMSE: ", round(no_cf_rmse,5)), 
           hjust = 1.1, vjust = 8.4, 
           size = 4)+
  theme_classic()

b_rmse = mean((no_cross_fit_results$riesz_estimate - no_cross_fit_results$truth)^2)^(1/2)
b_bias = mean(no_cross_fit_results$riesz_estimate - no_cross_fit_results$truth)
b = ggplot(no_cross_fit_results)+
  geom_histogram(aes(x=(riesz_estimate-truth)/sqrt(riesz_variance/1000), y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  xlab(
    expression(
      frac(
        hat(psi) - psi(P),
        sqrt(hat(V) / n)
      )
    )
  ) +
  ggtitle('Direct Riesz representer & No cross-fitting') +
  xlim(-5,5) +
  ylim(0, 1/2) +
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("RMSE: ", round(b_rmse,5)), 
           hjust = 1.1, vjust = 2.1, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Bias: ", round(b_bias,5)), 
           hjust = 1.1, vjust = 4.2, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in Bias: ", round(no_cf_bias,5)), 
           hjust = 1.1, vjust = 6.3, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in RMSE: ", round(no_cf_rmse,5)), 
           hjust = 1.1, vjust = 8.4, 
           size = 4)+
  theme_classic()

c_rmse = mean((cross_fit_results$indirect_estimate - cross_fit_results$truth)^2)^(1/2)
c_bias = mean(cross_fit_results$indirect_estimate - cross_fit_results$truth)
c = ggplot(cross_fit_results)+
  geom_histogram(aes(x=(indirect_estimate-truth)/sqrt(indirect_variance/1000), y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  xlab(
    expression(
      frac(
        hat(psi) - psi(P),
        sqrt(hat(V) / n)
      )
    )
  ) +
  ggtitle('Indirect Riesz representer & Cross-fitting') +
  xlim(-5,5) +
  ylim(0, 1/2) +
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("RMSE: ", round(c_rmse,5)), 
           hjust = 1.1, vjust = 2.1, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Bias: ", round(c_bias,5)), 
           hjust = 1.1, vjust = 4.2, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in Bias: ", round(cf_bias,5)), 
           hjust = 1.1, vjust = 6.3, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in RMSE: ", round(cf_rmse,5)), 
           hjust = 1.1, vjust = 8.4, 
           size = 4)+
  theme_classic()

d_rmse = mean((cross_fit_results$riesz_estimate - cross_fit_results$truth)^2)^(1/2)
d_bias = mean(cross_fit_results$riesz_estimate - cross_fit_results$truth)
d = ggplot(cross_fit_results)+
  geom_histogram(aes(x=(riesz_estimate-truth)/sqrt(riesz_variance/1000), y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  xlab(
    expression(
      frac(
        hat(psi) - psi(P),
        sqrt(hat(V) / n)
      )
    )
  ) +
  ggtitle('Direct Riesz representer & Cross-fitting') +
  xlim(-5,5) +
  ylim(0, 1/2) +
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("RMSE: ", round(d_rmse,5)), 
           hjust = 1.1, vjust = 2.1, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Bias: ", round(d_bias,5)), 
           hjust = 1.1, vjust = 4.2, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in Bias: ", round(cf_bias,5)), 
           hjust = 1.1, vjust = 6.3, 
           size = 4)+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Plug-in RMSE: ", round(cf_rmse,5)), 
           hjust = 1.1, vjust = 8.4, 
           size = 4)+
  theme_classic()

a+b+c+d
