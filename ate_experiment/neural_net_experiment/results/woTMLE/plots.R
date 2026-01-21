library(tidyverse)
res_files <- list.files("ate_experiment/neural_net_experiment/results/woTMLE/",pattern = "*arch.csv", full.names = TRUE)
results <- tibble()
for(file in res_files){
  results <- read_csv(file) %>% mutate(TMLE = 0) %>% bind_rows(results)
}
res_files <- list.files("ate_experiment/neural_net_experiment/results/wTMLE/",pattern = "*arch.csv", full.names = TRUE)
for(file in res_files){
  results <- read_csv(file) %>% mutate(TMLE = 1) %>% bind_rows(results)
}

results <- results %>% filter(rr_weight <= 1)





agg_res <- results %>% group_by(rr_weight,TMLE) %>% 
  summarise(bias = mean(truth-riesz_estimate), variance = mean((riesz_estimate-mean(riesz_estimate))^2), rmse = sqrt(mean((truth-riesz_estimate)^2)), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)))


results %>%
  mutate(normalized = sqrt(1000)*(riesz_estimate-truth)/sqrt(riesz_variance)) %>%
  ggplot() +
  geom_histogram(aes(x=normalized, y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  geom_text(data = agg_res, aes(x = -3.3, y = 0.4, label = paste0( "RMSE: ",as.character(round(rmse,4)))), size = 4) +
  geom_text(data = agg_res, aes(x = -3.3, y = 0.375, label = paste0( "Bias: ",as.character(round(bias,4)))), size = 4) + 
  geom_text(data = agg_res, aes(x = -3.3, y = 0.35, label = paste0( "Variance: ",as.character(round(variance,4)))), size = 4) +
  facet_wrap(~TMLE*rr_weight) +
  theme_bw() +
  xlim(-5,5) + 
  xlab(
    expression(
      frac(
        hat(psi) - psi(P),
        sqrt(hat(V) / n)
      )
    )
  )


