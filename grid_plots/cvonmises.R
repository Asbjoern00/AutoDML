library(tidyverse)



cvm_test <- function(df, include_pval=FALSE){
  propensity <- goftest::cvm.test(df$propensity_normalized, null = "pnorm")
  riesz <- goftest::cvm.test(df$riesz_normalized, null = "pnorm")
  out <- tibble(estimator = c("indirect","riesz"), statistic =c(propensity$statistic[["omega2"]], riesz$statistic[["omega2"]]))
  
  if (include_pval){
    out <- out %>% mutate(p = c(propensity$p.value,riesz$p.value))
  }
  
  return(out)
}

resample_frame <- function(df,m=1000){
  n <- nrow(df)
  res <- tibble()
  for (i in 1:m){
    idx <- sample(seq_len(n), n, replace = TRUE)
    tst <- df[idx, , drop = FALSE] %>% cvm_test()
    res <- bind_rows(res,tst)
  }
  res

}

process_results <- function(base_path, sizes, type_label, cross = TRUE) {
  
  fit_type <- ifelse(cross, "cross_fit_results_", "no_cross_fit_results_")
  
  
  map_dfr(sizes, function(n) {

    
    file_path <- paste0(base_path, fit_type, n, ".csv")
    
    df <- read_csv(file_path, show_col_types = FALSE)

    # ---- Handle inconsistent column names ----
    if ("propensity_estimate" %in% names(df)) {
      df <- df %>% rename(propensity = propensity_estimate)
    } else if (sum(stringr::str_detect("indirect",names(df))) > 0 ) {
      df <- df %>% rename_with(~gsub("indirect", "propensity",.x)) %>% rename(propensity = propensity_estimate)
    } else {
      stop(paste("No propensity/indirect column in", file_path))
    }
    if(stringr::str_detect(base_path, "gradient_boosting")){
      df <- df %>% mutate(propensity_normalized = (propensity-truth)/sqrt(propensity_variance), 
                    riesz_normalized = (riesz_estimate-truth)/sqrt(riesz_variance)) %>% 
        select(propensity_normalized,riesz_normalized)
      
    }
    else{
      df <- df %>% mutate(propensity_normalized = (propensity-truth)/sqrt(propensity_variance/n), 
                    riesz_normalized = (riesz_estimate-truth)/sqrt(riesz_variance/n)) %>% 
        select(propensity_normalized,riesz_normalized)
    }
    df_cvm_test <- cvm_test(df,include_pval = TRUE) %>% mutate(n = n, type = type_label)
    
    df_resample <- resample_frame(df)
    df_resample %>% group_by(estimator) %>% 
      summarise(ci_l = quantile(statistic, 0.025), ci_u = quantile(statistic, 0.975)) %>% 
      inner_join(df_cvm_test)
  })
}



sizes <- c(125, 500, 1000,1500, 2000)

# CROSS-FIT
cross_data <- bind_rows(
  
  process_results("ate_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ATE", cross = TRUE),
  
  process_results("ase_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ASE", cross = TRUE),
  
  process_results("ate_experiment/LASSO_experiment/Results/",
                  sizes, "LASSO ATE", cross = TRUE),
  
  process_results("ase_experiment/LASSO_experiment/Results/",
                  sizes, "LASSO ASE", cross = TRUE)
)

cross_data <- cross_data %>%
  mutate(Estimator = ifelse(estimator == "riesz",
                            "Direct with cross-fitting",
                            "Indirect with cross-fitting"))


no_cross_data <- bind_rows(
  
  process_results("ate_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ATE", cross = FALSE),
  
  process_results("ase_experiment/gradient_boosting_experiment/results/",
                  sizes, "Gradient Boosting ASE", cross = FALSE),
  
  process_results("ate_experiment/LASSO_experiment/Results/",
                  sizes, "LASSO ATE", cross = FALSE),
  
  process_results("ase_experiment/LASSO_experiment/Results/",
                  sizes, "LASSO ASE", cross = FALSE)
)

no_cross_data <- no_cross_data %>%
  mutate(Estimator = ifelse(estimator == "riesz",
                            "Direct without cross-fitting",
                            "Indirect without cross-fitting"))





ggplot(plot_data %>% filter(n > 500)) +
  geom_errorbar(
    aes(colour = Estimator,
        x = n,
        ymin = ci_l,
        ymax = ci_u),
    width = 100,
    alpha = 0.5
  ) +
  geom_line(
    aes(x = n,
        y = statistic,
        colour = Estimator),
    linetype = "dashed"
  ) +
  geom_point(
    aes(x = n,
        y = statistic,
        shape = p < 0.05,
        colour = Estimator),
    size = 3
  ) +
  scale_shape_manual(
    values = c(`TRUE` = 4,  # cross (x)
               `FALSE` = 16) # solid circle
  ) +
  geom_hline(
    aes(yintercept = 0.47),
    linetype = "solid",
    linewidth = 0.5
  ) +
  xlab(expression(n)) + 
  theme_classic() +
  ylab("Cramér-von Mises statistic") +
  facet_wrap(~type, scales = "free") +
  ggtitle("Cramér-von Mises statistic at different sample sizes") + 
  theme(
    plot.title   = element_text(size = 16, hjust = 0.5),
    axis.title   = element_text(size = 16),
    axis.text    = element_text(size = 14),
    strip.text   = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 12)
  )


vec <- numeric(10000)

for (i in 1:10000){
  vec[i] <- cvm.test(rnorm(1000), null = "pnorm")$statistic[["omega2"]]
}













