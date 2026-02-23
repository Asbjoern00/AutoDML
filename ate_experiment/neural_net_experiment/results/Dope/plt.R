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
  summarise(bias = mean(truth-riesz_estimate), variance = var(riesz_estimate), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)),
            mse = mean((truth-riesz_estimate)^2),  var_mse = var((truth-riesz_estimate)^2)) %>% 
  mutate(rmse = sqrt(mse), var_rmse = 1/(4*mse)*var_mse, rmse_u = rmse + 1.96*sqrt(var_mse/1000), rmse_l = rmse - 1.96*sqrt(var_mse/1000)) 
              
              
              


res_weights <- tibble()

res_files <- list.files("ate_experiment/neural_net_experiment/results/wTMLE/",pattern = "*0.csv", full.names = TRUE)
for(file in res_files){
  res <-read_csv(file) %>% filter(riesz_variance > 0)
  res$TMLE <- 1
  res_weights <- bind_rows(res,res_weights)
}


res_files <- list.files("ate_experiment/neural_net_experiment/results/woTMLE/",pattern = "*0.csv", full.names = TRUE)
for(file in res_files){
  res <- read_csv(file) %>% filter(riesz_variance > 0)
  res$TMLE <- 0
  res_weights <- bind_rows(res,res_weights)
}
mask <- 2^seq(-7,1, by = 2)
agg_res_w <- res_weights %>% filter(rr_weight %in% mask) %>% group_by(rr_weight,TMLE) %>%
  summarise(bias = mean(truth-riesz_estimate), variance = var(riesz_estimate), cvg = mean((riesz_upper>truth)*(riesz_lower<truth)),
            mse = mean((truth-riesz_estimate)^2),  var_mse = var((truth-riesz_estimate)^2)) %>% 
  mutate(rmse = sqrt(mse), var_rmse = 1/(4*mse)*var_mse, rmse_u = rmse + 1.96*sqrt(var_mse/1000), rmse_l = rmse - 1.96*sqrt(var_mse/1000)) %>% 
  mutate(TMLE = case_when(TMLE == 1 ~ "RieszNet, TMLE weight 1",
                   TRUE ~ "RieszNet, TMLE weight 0"))


tmle_colors <- c(
  "RieszNet, TMLE weight 1" = "#1b9e77",
  "RieszNet, TMLE weight 0" = "#d95f02",
  "Outcome Informed" = "#7570b3",
  "Riesz Informed"   = "#e7298a",
  "Separate Neural Nets" = "#66a61e"
)
linetypes <- c(  "Outcome Informed" = "dashed",
                 "Riesz Informed"   = "dotted",
                 "Separate Neural Nets" = "dotdash")


hline_df <- agg_res %>%
  filter(Informed %in% c("Outcome", "Riesz", "Separate")) %>%
  mutate(
    label = case_when(
      Informed == "Outcome"  ~ "Outcome Informed",
      Informed == "Riesz"    ~ "Riesz Informed",
      Informed == "Separate" ~ "Separate Neural Nets"
    )
  ) %>%
  select(label, rmse, rmse_l, rmse_u) %>% mutate(rr_weight = 2^(-3))

a <- agg_res_w %>% 
    ggplot(aes(x = rr_weight, y = rmse, color = as.factor(TMLE))) + 
    
    geom_point(size = 3) + 
    
    #geom_line(linetype = "dashed") + 
    
    geom_errorbar(
      aes(ymin = rmse_l, ymax = rmse_u),
      width = 0.35,
    ) +
    
    # Horizontal point estimates
    geom_hline(
      data = hline_df,
      aes(
        yintercept = rmse,
        color = label
      ),
      linewidth = 1
    ) +
  geom_hline(
    data = hline_df,
    aes(
      yintercept = rmse_u,
      color = label,
      linetype = paste0(label, " (CI)")
    ),
    linewidth = 1
  ) +
  geom_hline(
    data = hline_df,
    aes(
      yintercept = rmse_l,
      color = label,
      linetype = paste0(label, " (CI)")
    ),
    linewidth = 1
  )+
    
    scale_color_manual(values = tmle_colors) +
    scale_fill_manual(values = tmle_colors) +
  scale_linetype_manual(
    values = c(
      "Outcome Informed (CI)" = "dashed",
      "Riesz Informed (CI)"   = "dotted",
      "Separate Neural Nets (CI)" = "dotdash"
    )
  ) + 
    
    scale_x_continuous(
      trans = "log2",
      breaks = mask,
      labels = function(x) {
        parse(text = paste0("2^", log2(x)))
      }
    ) + 
    
    labs(color = NULL, fill = NULL) + 
    ylab("RMSE") + 
    xlab("Weight on Riesz loss") + 
  theme_classic() + 
      theme(
      plot.title   = element_text(size = 16, hjust = 0.5),
      axis.title   = element_text(size = 16),
      axis.text    = element_text(size = 14),
      strip.text   = element_text(size = 14),
      legend.title = element_blank(),
      legend.text  = element_text(size = 14)
    ) + 
    ggtitle("RMSE for varying weights on Riesz loss") + 
     ylim(0.10, 0.75)

  
a
