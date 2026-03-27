library(tidyverse)
library(stringr)
files <- list.files("doper_neural_nets/var_gap/",full.names = TRUE)
files <- files[str_detect(files,"csv")]
res <- tibble()
for (f in files){
  res_tib <- read_csv(f)
  if(str_detect(f, "outcome")){
    res_tib$type <- "Outcome Informed Neural Network"
  }
  else{
    res_tib$type <- "Separate Neural Network"
  }
  res_tib$i <- 1:nrow(res_tib)
  res <- res %>% bind_rows(res_tib)
}
res %>% pivot_wider(names_from = type,values_from = c(estimate)) %>% 
  mutate(res_sq_oi = (`Outcome Informed Neural Network`-truth)^2 , res_sq_sep = (`Separate Neural Network`-truth)^2, gap_res = res_sq_sep-res_sq_oi) %>% 
  group_by(beta) %>% 
  summarise(empirical_gap = 1000*mean(gap_res,na.rm = TRUE), se_empirical_gap = sqrt(1000)*sd(gap_res, na.rm = TRUE)) %>% 
  mutate(theo_gap = 2*(exp(beta^2/2)-1), gap_l = empirical_gap - 1.96*se_empirical_gap, gap_u = empirical_gap + 1.96*se_empirical_gap )





res %>% mutate(res_sq = 1000*(estimate-truth)^2) %>% 
  group_by(type, beta) %>% 
  summarise(nMSE = mean(res_sq,na.rm=TRUE), se = sd(res_sq)/sqrt(1000), ci_l = nMSE - 1.96*se, ci_u = nMSE + 1.96*se) %>% 
  mutate(theo_gap = 2*(exp(beta^2/2)-1)) %>% 
  ggplot() + geom_line(aes(x=beta,y=nMSE, color=type),linetype = "dashed")+
  geom_point(aes(x=beta,y=nMSE, color=type)) + 
  geom_errorbar(aes(x=beta,ymin=ci_l,ymax=ci_u, color=type)) + 
  geom_line(aes(x=beta,y=theo_gap+4, color = "Theoretical MSE, bad"), linetype = "longdash") +
  geom_hline(aes(yintercept = 4, color = "Theoritcal MSE, good"), linetype = "longdash") +
  xlab(expression(beta)) +
  ylab("Asymptotic MSE") + 
  theme_classic()
  




library(ggnewscale)

beta_grid <- tibble(
  beta = seq(min(res$beta), max(res$beta), length.out = 400)
) %>%
  mutate(theo_gap = 2*(exp(beta^2/2)-1))

res %>% 
  mutate(res_sq = 1000*(estimate-truth)^2) %>% 
  group_by(type, beta) %>% 
  summarise(
    nMSE = mean(res_sq,na.rm=TRUE),
    se = sd(res_sq)/sqrt(1000),
    ci_l = nMSE - 1.96*se,
    ci_u = nMSE + 1.96*se,
    .groups = "drop"
  ) %>% 
  mutate(theo_gap = 2*(exp(beta^2/2)-1)) %>% 
  ggplot() +
  
  # Estimator layers
  geom_line(aes(x=beta,y=nMSE,color=type), linetype="dashed") +
  geom_point(aes(x=beta,y=nMSE,color=type)) +
  geom_errorbar(aes(x=beta,ymin=ci_l,ymax=ci_u,color=type,width = 0.1)) +
  labs(color="Estimator") +
  scale_color_manual(values= c("Outcome Informed Neural Network"="#00BFC4" ,
                     "Separate Neural Network" = "#F8766D"),
                     labels = c("Outcome Informed Neural Network" = "DOPE Auto-DML with\nNeural Network with LASSO Layer" ,
                                "Separate Neural Network" ="Auto-DML with \nSeparate Neural Networks")) + 
  
  new_scale_color() +
  
  # Representation layers
  geom_line(
    data = beta_grid,
    aes(x = beta, y = theo_gap + 4, color = "Theoretical MSE, bad"),
  ) +
  geom_hline(aes(yintercept=4,color="Theoretical MSE, good")) +
  scale_color_manual(
    name="Representation",
    values=c(
      "Theoretical MSE, bad"="#F8766D",
      "Theoretical MSE, good"="#00BFC4"
    ),
    labels = c(
      "Theoretical MSE, bad"=expression(X),
      "Theoretical MSE, good"=expression(Z)
    )
  ) +
  
  xlab(expression(beta)) +
  ylab("Asymptotic Variance / Estimator MSE") +
  theme_classic(base_size = 22)