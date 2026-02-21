library(tidyverse)
data0 <- read_csv("different_riesz_weights/tmle_w_0_n_riesz_0.csv") %>%
  mutate(type = 'TMLE weight 0, Outcome MSE weight 2', index=1:100)
data1 <- read_csv("different_riesz_weights/tmle_w_1_n_riesz_0.csv") %>%
  mutate(type = 'TMLE weight 1, Outcome MSE weight 1', index=1:100)
data2 <- read_csv("different_riesz_weights/tmle_w_2_n_riesz_0.csv") %>%
  mutate(type = 'TMLE weight 2, Outcome MSE weight 0', index=1:100)
data_outcome <- read_csv("dope_neural_nets/outcome_informed_ihdp/outcome_informed_ihdp_full.csv") %>%
  mutate(type = 'Outcome informed net', index=1:1000) 


k = 100
data = rbind(data0[1:k,],data1[1:k,], data2[1:k,])

text_data = data %>% group_by(type) %>% summarise(value = mean(abs(riesz_estimate-truth)^2)^0.5)





ggplot(data) +
  geom_line(aes(x=index, y=riesz_estimate-truth, color=type), size=0.3, linetype='dashed')+
  geom_point(aes(x=index, y=riesz_estimate-truth, color=type), size=3)+
  xlab('Iteration')+
  ylab('Estimation error')+
  theme_classic()+
  ggtitle('RieszNet estimation error in first 100 iterations')






text_data = data %>%
  group_by(type) %>%
  summarise(value = mean(abs(riesz_estimate-truth)),
            .groups = "drop") %>%
  mutate(
    label = paste0('MAE', ": ", sprintf("%.3f", value)),
    x = Inf,
    y = (3:1)/20+0.3
  )

ggplot(data) +
  geom_line(aes(x=index, y=riesz_estimate-truth, color=type),
            size=0.3, linetype='dashed')+
  geom_point(aes(x=index, y=riesz_estimate-truth, color=type),
             size=3)+
  geom_text(
    data = text_data,
    aes(x=x, y=y, color=type, label=label),
    hjust = 1.05,
    vjust = 1.1,
    size = 5,
    show.legend = FALSE
  )+
  xlab('Iteration')+
  ylab('Estimation error')+
  theme_classic()+
  ggtitle('RieszNet estimation error in first 100 iterations') +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.08)))+
  theme(
    legend.text = element_text(size = 12), 
    legend.title = element_blank()
  )
