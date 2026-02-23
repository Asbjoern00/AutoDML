library(tidyverse)
data0 <- read_csv("dope_neural_nets/different_riesz_weights/tmle0.csv") %>%
  mutate(type = 'TMLE weight 0, Outcome MSE weight 1', index=1:1000, tmle_w = 0)
data05 <- read_csv("dope_neural_nets/different_riesz_weights/tmle0.5.csv") %>%
  mutate(type = 'TMLE weight 0.5, Outcome MSE weight 1.5', index=1:1000, tmle_w = 0.5)
data1 <- read_csv("dope_neural_nets/different_riesz_weights/tmle1.csv") %>%
  mutate(type = 'TMLE weight 0.5, Outcome MSE weight 0.5', index=1:1000, tmle_w = 1)
data15 <- read_csv("dope_neural_nets/different_riesz_weights/tmle1.5.csv") %>%
  mutate(type = 'TMLE weight 1.5, Outcome MSE weight 0.5', index=1:1000, tmle_w = 1.5)
data2 <- read_csv("dope_neural_nets/different_riesz_weights/tmle2.csv") %>%
  mutate(type = 'TMLE weight 2, Outcome MSE weight 0', index=1:1000, tmle_w = 2)




data = rbind(data0, data05, data1,data15, data2)

d = data %>%
  mutate(residual = (estimate - truth)) %>%
  select(index, tmle_w, residual) %>%
  pivot_wider(
    names_from = tmle_w,
    values_from = residual
  ) %>%
  select(-index)

corrplot::corrplot(cor(d), method = 'number')



data = data %>%
  mutate(residual_sq = (estimate-truth)^2) %>%
  group_by(tmle_w) %>%
  summarise(mse = mean(residual_sq), sd_mse = sd(residual_sq)/(1000^0.5)) %>%
  mutate(rmse = mse^0.5, l = (mse-1.96*sd_mse)^0.5, u = (mse+1.96*sd_mse)^0.5)

ggplot(data) +
  geom_point(aes(x=tmle_w, y = rmse)) + 
  geom_errorbar(aes(x=tmle_w, ymin=l, ymax=u), width=.1)+
  geom_line(aes(x=tmle_w, y = rmse), linetype='dashed') + 
  theme_classic()+
  ylim(0,0.2)+
  xlab('Weight on TMLE component')+
  ylab('RMSE')+
  ggtitle('RMSE with constant total weight on TMLE and MSE loss components')+
  theme(plot.title = element_text(hjust = 0.5))
data
