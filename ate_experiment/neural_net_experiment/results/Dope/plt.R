library(tidyverse)
res_files <- list.files("ate_experiment/neural_net_experiment/results/Dope/",pattern = "*.csv", full.names = TRUE)
results <- tibble()
for(file in res_files){
  results <- results %>% bind_rows(read_csv(file))
}

agg_res <- results  %>% 
  summarise(bias = mean(truth-riesz_estimate), variance = mean((riesz_estimate-mean(riesz_estimate))^2), rmse = sqrt(mean((truth-riesz_estimate)^2)), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)))


agg_res %>% ggplot() + geom_line(aes(x = rr_weight, y = variance, color = "variance")) + geom_line(aes(x= rr_weight, y = bias^2, color = "bias^2")) + scale_x_log10() + theme_bw()

results %>%
  mutate(normalized = sqrt(1000)*(riesz_estimate-truth)/sqrt(riesz_variance)) %>%
  ggplot() +
  geom_histogram(aes(x=normalized, y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  geom_text(data = agg_res, aes(x = -3.5, y = 0.4, label = paste0( "RMSE: ",as.character(round(rmse,4)))), size = 4) +
  geom_text(data = agg_res, aes(x = -3.5, y = 0.35, label = paste0( "Bias: ",as.character(round(bias,4)))), size = 4) +
  #facet_wrap(~rr_weight) +
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


