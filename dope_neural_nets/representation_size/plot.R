library(tidyverse)
data1 = read_csv('dope_neural_nets/representation_size/1_shared.csv') %>%
  mutate(n_shared = 1)
data2 = read_csv('dope_neural_nets/representation_size/2_shared.csv') %>%
  mutate(n_shared = 2)
data3 = read_csv('dope_neural_nets/representation_size/3_shared.csv') %>%
  mutate(n_shared = 3)
data4 = read_csv('dope_neural_nets/representation_size/4_shared.csv') %>%
  mutate(n_shared = 4)
data4 = data4[c(1:771, 773:1000), ]
data5 = read_csv('dope_neural_nets/representation_size/5_shared.csv') %>%
  mutate(n_shared = 5)
data = bind_rows(data1, data2, data3, data4, data5) %>%
  group_by(n_shared) %>%
  summarise(rmse = sqrt(mean((estimate-truth)^2)))

ggplot(data) +
  geom_line(aes(x=n_shared, y=rmse))+
  geom_point(aes(x=n_shared, y=rmse))+
  theme_classic()+
  ylab('RMSE')+
  xlab('Number of shared layers')+
  ggtitle('RMSE With Varying Number of Shared Layers')
