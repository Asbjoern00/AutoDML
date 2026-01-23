library(tidyverse)
library(patchwork)
#res_files <- list.files("ate_experiment/neural_net_experiment/results/Dope/",pattern = "*.csv", full.names = TRUE)
res_files <- list.files("ate_experiment/neural_net_experiment/results/Dope/",pattern = "*_3.csv", full.names = TRUE)
res_files <- c(res_files, list.files("ate_experiment/neural_net_experiment/results/Dope/",pattern = "sep_nets.csv", full.names = TRUE))

results <- tibble()
for(file in res_files){
  res <-  read_csv(file)
  if(stringr::str_detect(file, "outcome")){
    res$Informed <- "Outcome"
  }else if(stringr::str_detect(file, "riesz")){
    res$Informed <- "Riesz"
    res <- res
  }else{
    res$Informed <- "Separate"
    res <- res
  }
  res$index <- 1:1000
  results <- bind_rows(results,res)
  results <- results %>% mutate(riesz_coverage = riesz_lower <= truth & truth <= riesz_upper) 
}

agg_res <- results  %>% group_by(Informed) %>% 
  summarise(bias = mean(truth-riesz_estimate), variance = var(riesz_estimate), rmse = sqrt(mean((truth-riesz_estimate)^2)), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)),
            est_as_var = mean(riesz_variance), avg_len = mean(riesz_upper-riesz_lower))


res_weights <- tibble()

res_files <- list.files("ate_experiment/neural_net_experiment/results/wTMLE/",pattern = "*.csv", full.names = TRUE)
for(file in res_files){
  res <-read_csv(file)
  res$TMLE <- 1
  res_weights <- bind_rows(res,res_weights)
}


res_files <- list.files("ate_experiment/neural_net_experiment/results/woTMLE/",pattern = "*.csv", full.names = TRUE)
for(file in res_files){
  res <-read_csv(file)
  res$TMLE <- 0
  res_weights <- bind_rows(res,res_weights)
}

agg_res_w <- res_weights %>% filter(rr_weight < 2) %>% group_by(rr_weight,TMLE) %>%
  summarise(bias = mean(truth-riesz_estimate), variance = mean((riesz_estimate-mean(riesz_estimate))^2), rmse = sqrt(mean((truth-riesz_estimate)^2)), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)),
            est_as_var = mean(riesz_variance), avg_len = mean(riesz_upper-riesz_lower)) %>% 
  mutate(TMLE = case_when(TMLE == 1 ~ "RieszNet, TMLE weight 1",
                   TRUE ~ "RieszNet, TMLE weight 0"))


tmle_colors <- c(
  "RieszNet, TMLE weight 1" = "#1b9e77",
  "RieszNet, TMLE weight 0" = "#d95f02",
  "Outcome Informed" = "#7570b3",
  "Riesz Informed"   = "#e7298a",
  "Separate Neural Nets" = "#66a61e"
)

a <- agg_res_w %>% ggplot(aes(x = rr_weight, color = as.factor(TMLE), y = rmse)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous(transform = "log2") +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Outcome") %>% pull("rmse"),color = "Outcome Informed")) +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Riesz") %>% pull("rmse"),color = "Riesz Informed")) +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Separate") %>% pull("rmse"),color = "Separate Neural Nets")) +
  labs(color = NULL) + 
  theme_bw()

b <- agg_res_w %>% ggplot(aes(x = rr_weight, color = TMLE, y = bias)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous(transform = "log2") +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Outcome") %>% pull("bias"),color = "Outcome Informed")) +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Riesz") %>% pull("bias"),color = "Riesz Informed")) +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Separate") %>% pull("bias"),color = "Separate Neural Nets")) +
  labs(color = NULL) + 
  theme_bw()


c <- agg_res_w %>% ggplot(aes(x = rr_weight, color = TMLE, y = variance)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous(transform = "log2") +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Outcome") %>% pull("variance"),color = "Outcome Informed")) +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Riesz") %>% pull("variance"),color = "Riesz Informed")) +
  geom_hline(aes(yintercept = agg_res %>% filter(Informed == "Separate") %>% pull("variance"),color = "Separate Neural Nets")) +
  labs(color = NULL) + 
  theme_bw()



a + b + c















#agg_res %>% ggplot() + geom_line(aes(x = rr_weight, y = variance, color = "variance")) + geom_line(aes(x= rr_weight, y = bias^2, color = "bias^2")) + scale_x_log10() + theme_bw()

results %>%
  mutate(normalized = sqrt(1000)*(riesz_estimate-truth)/sqrt(riesz_variance)) %>%
  ggplot() +
  geom_histogram(aes(x=normalized, y=after_stat(density)), color='white')+
  geom_function(fun=function(x) dnorm(x)) +
  geom_vline(xintercept=0, color='red', size=1) +
  geom_text(data = agg_res, aes(x = -3.5, y = 0.4, label = paste0( "RMSE: ",as.character(round(rmse,4)))), size = 4) +
  geom_text(data = agg_res, aes(x = -3.5, y = 0.38, label = paste0( "Bias: ",as.character(round(bias,4)))), size = 4) +
  geom_text(data = agg_res, aes(x = -3.5, y = 0.36, label = paste0( "Variance: ",as.character(round(variance,4)))), size = 4) +
  facet_wrap(~Informed) + 
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

ymin <- 1.5
ymax <- 3.2

ggplot(results %>% group_by(Informed) %>% slice_head(n=100))+
  geom_segment(aes(x=index, y=pmax(riesz_lower,ymin),xend=index, yend=pmin(riesz_upper,ymax), color = as.factor(riesz_coverage)))+
  geom_point(aes(x=index, y=riesz_estimate))+
  geom_text(data = agg_res, aes(x = 50, y = 3.2, label = paste0( "Coverage: ",as.character(round(cvg,4)))), size = 4) +
  geom_text(data = agg_res, aes(x = 50, y = 3.1, label = paste0( "Avg. length: ",as.character(round(avg_len,4)))), size = 4) +
  facet_wrap(~Informed) + 
  theme_bw()+
  scale_color_manual(                      
    values = c('FALSE' = "red", "TRUE" = "black"),
    name = "Coverage",
    labels = c("Missed", "Covered")
  )+
  geom_hline(yintercept = results$truth[1],color='blue')+
  ylab('Point estimate and CI')+
  ylim(ymin,ymax)






