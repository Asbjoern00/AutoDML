library(tidyverse)
res_files <- list.files("ate_experiment/neural_net_experiment/results/",pattern = "*.csv", full.names = TRUE)
results <- tibble()
for(file in res_files){
  results <- results %>% bind_rows(read_csv(file))
}

agg_res <- results %>% group_by(rr_weight) %>% 
  summarise(bias = mean(truth-riesz_estimate), variance = mean((riesz_estimate-mean(riesz_estimate))^2), rmse = sqrt(mean((truth-riesz_estimate)^2)), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)))


agg_res %>% ggplot() + geom_line(aes(x = rr_weight, y = variance, color = "variance")) + geom_line(aes(x= rr_weight, y = bias^2, color = "bias^2")) + scale_x_log10() + theme_bw()

results %>%
  mutate(normalized = sqrt(1000)*(riesz_estimate-truth)/sqrt(riesz_variance)) %>%
  ggplot(aes(x = normalized)) + 
  geom_histogram() +
  facet_wrap(~rr_weight) +
  theme_bw()
  

