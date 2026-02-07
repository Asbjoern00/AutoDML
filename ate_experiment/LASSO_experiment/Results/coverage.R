library(tidyverse)
library(patchwork)
no_cross_fit_results <- read_csv("ate_experiment/LASSO_experiment/Results/no_cross_fit_results.csv")
cross_fit_results <- read_csv("ate_experiment/LASSO_experiment/Results/cross_fit_results.csv")
no_cross_fit_results = no_cross_fit_results %>% mutate(index = 1:1000, propensity_coverage = propensity_lower <= truth & truth <= propensity_upper)
no_cross_fit_results = no_cross_fit_results %>% mutate(riesz_coverage = riesz_lower <= truth & truth <= riesz_upper)

cross_fit_results = cross_fit_results %>% mutate(index = 1:1000, propensity_coverage = propensity_lower <= truth & truth <= propensity_upper)
cross_fit_results = cross_fit_results %>% mutate(riesz_coverage = riesz_lower <= truth & truth <= riesz_upper)

ymax = 2.7
ymin = 0


a = ggplot(no_cross_fit_results[1:100,])+
  geom_segment(aes(x=index, y=propensity_lower,xend=index, yend=propensity_upper, color = as.factor(propensity_coverage)))+
  geom_point(aes(x=index, y=propensity_estimate))+
  theme_classic()+
  scale_color_manual(                      
    values = c('FALSE' = "red", "TRUE" = "black"),
    name = "Coverage",
    labels = c("Missed", "Covered")
  )+
  geom_hline(yintercept = no_cross_fit_results$truth[1],color='blue')+
  ylab('Point estimate and CI')+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Coverage: ", round(mean(no_cross_fit_results$propensity_coverage),5)), 
           hjust = 1.1, vjust = 2, 
           size = 4)+
  ylim(ymin,ymax)+
  ggtitle('Indirect Riesz representer & No cross-fitting')



b = ggplot(no_cross_fit_results[1:100,])+
  geom_segment(aes(x=index, y=propensity_lower,xend=index, yend=riesz_upper, color = as.factor(riesz_coverage)))+
  geom_point(aes(x=index, y=riesz_estimate))+
  theme_classic()+
  scale_color_manual(                      
    values = c('FALSE' = "red", "TRUE" = "black"),
    name = "Coverage",
    labels = c("Missed", "Covered")
  )+
  geom_hline(yintercept = no_cross_fit_results$truth[1],color='blue')+
  ylab('Point estimate and CI')+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Coverage: ", round(mean(no_cross_fit_results$riesz_coverage),5)), 
           hjust = 1.1, vjust = 2, 
           size = 4)+
  ylim(ymin, ymax)+
  ggtitle('Direct Riesz representer & No cross-fitting')


c = ggplot(cross_fit_results[1:100,])+
  geom_segment(aes(x=index, y=pmax(ymin,propensity_lower),xend=index, yend=pmin(ymax,propensity_upper), color = as.factor(propensity_coverage)))+
  geom_point(aes(x=index, y=propensity_estimate))+
  theme_classic()+
  scale_color_manual(                      
    values = c('FALSE' = "red", "TRUE" = "black"),
    name = "Coverage",
    labels = c("Missed", "Covered")
  )+
  geom_hline(yintercept = no_cross_fit_results$truth[1],color='blue')+
  ylab('Point estimate and CI')+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Coverage: ", round(mean(cross_fit_results$propensity_coverage),5)), 
           hjust = 1.1, vjust = 2, 
           size = 4)+
  ylim(ymin, ymax)+
  ggtitle('Indirect Riesz representer & Cross-fitting')

d = ggplot(cross_fit_results[1:100,])+
  geom_segment(aes(x=index, y=riesz_lower,xend=index, yend=riesz_upper, color = as.factor(riesz_coverage)))+
  geom_point(aes(x=index, y=riesz_estimate))+
  theme_classic()+
  scale_color_manual(                      
    values = c('FALSE' = "red", "TRUE" = "black"),
    name = "Coverage",
    labels = c("Missed", "Covered")
  )+
  geom_hline(yintercept = no_cross_fit_results$truth[1],color='blue')+
  ylab('Point estimate and CI')+
  annotate("text", 
           x = Inf, y = Inf, 
           label = paste0("Coverage: ", round(mean(cross_fit_results$riesz_coverage),5)), 
           hjust = 1.1, vjust = 2, 
           size = 4)+
  ylim(ymin, ymax)+
  ggtitle('Direct Riesz representer & Cross-fitting')



a+b+c+d