library(tidyverse)
data1 = read_csv("dope_neural_nets/representation_size/shared_size_experiment/1_shared_neuron.csv") %>%
  mutate(n_shared = 1)
data3 = read_csv("dope_neural_nets/representation_size/shared_size_experiment/3_shared_neuron.csv") %>%
  mutate(n_shared = 3)
data10 = read_csv("dope_neural_nets/representation_size/shared_size_experiment/10_shared_neuron.csv") %>%
  mutate(n_shared = 10)
data50 = read_csv("dope_neural_nets/representation_size/shared_size_experiment/50_shared_neuron.csv") %>%
  mutate(n_shared = 50)
data100 = read_csv("dope_neural_nets/representation_size/shared_size_experiment/100_shared_neuron.csv") %>%
  mutate(n_shared = 100)

data = bind_rows(data1, data3, data10, data50, data100) %>%
  group_by(n_shared) %>%
  summarise(rmse = sqrt(mean((estimate-truth)^2)))

ggplot(data) +
  geom_line(aes(x=n_shared, y=rmse))+
  geom_point(aes(x=n_shared, y=rmse))+
  theme_classic()+
  ylab('RMSE')+
  xlab('Number of leurons in shared layer')+
  ggtitle('RMSE With Varying Number of Neurons in Shared Layer')
