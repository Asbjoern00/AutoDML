library(MASS)
library(tidyverse)
library(broom)

df <- bind_cols(
  read_csv("dope_neural_nets/different_riesz_weights/tmle0_full_x.csv") %>% 
    mutate(TMLE0 = (truth - estimate)/sqrt(variance)) %>% 
    select(TMLE0),
  
  read_csv("dope_neural_nets/different_riesz_weights/tmle05_full_x.csv") %>% 
    mutate(TMLE0.5 = (truth - estimate)/sqrt(variance)) %>% 
    select(TMLE0.5),
  
  read_csv("dope_neural_nets/different_riesz_weights/tmle1_full_x.csv") %>% 
    mutate(TMLE1 = (truth - estimate)/sqrt(variance)) %>% 
    select(TMLE1)
)

# Fit models
mod_05 <- lm(TMLE0.5 ~ TMLE0, data = df)
mod_1  <- lm(TMLE1  ~ TMLE0, data = df)
mod_id <- lm(TMLE0  ~  a, data = df %>% mutate(a = TMLE0))

extract_label <- function(model, label) {
  tibble(
    label = label,
    intercept = coef(model)[1],
    slope = coef(model)[2],
    r2 = summary(model)$r.squared
  )
}

stats_df <- bind_rows(
  extract_label(mod_05, "TMLE weight 1, MSE weight 1"),
  extract_label(mod_1,  "TMLE weight 2, MSE weight 0"),
  extract_label(mod_id, "TMLE weight 0, MSE weight 2")
) %>%
  mutate(
    text = sprintf("Slope = %.3f\nR² = %.3f",
                   slope, r2)
  )

cols <- c(
  "TMLE weight 1, MSE weight 1" = "#1b9e77",
  "TMLE weight 2, MSE weight 0" = "#d95f02",
  "TMLE weight 0, MSE weight 2" = "#7570b3"
)

# Add a y-position for each label
stats_df <- stats_df %>%
  mutate(
    y_pos = seq(from = max(df$TMLE1), 
                to = max(df$TMLE1) - 0.03 * (length(label) - 1), 
                length.out = n())
  )


ggplot(df) + 
  geom_point(aes(x = TMLE0, y = TMLE0.5, color = "TMLE weight 1, MSE weight 1")) +
  geom_smooth(aes(x = TMLE0, y = TMLE0.5, color = "TMLE weight 1, MSE weight 1"),
              method = "lm", se = FALSE) +
  
  geom_point(aes(x = TMLE0, y = TMLE1, color = "TMLE weight 2, MSE weight 0")) +
  geom_smooth(aes(x = TMLE0, y = TMLE1, color = "TMLE weight 2, MSE weight 0"),
              method = "lm", se = FALSE) +
  
  geom_point(aes(x = TMLE0, y = TMLE0, color = "TMLE weight 0, MSE weight 2")) +
  geom_smooth(aes(x = TMLE0, y = TMLE0, color = "TMLE weight 0, MSE weight 2"),
              method = "lm", se = FALSE) +
  
  geom_text(
    data = stats_df,
    aes(x = min(df$TMLE0) - 0.03 * diff(range(df$TMLE0)),  # shift left a little
        y = y_pos,
        label = text,
        color = label),
    hjust = 0,
    size = 4
  ) +
  
  scale_color_manual(values = cols) +
  xlab("Standardized error, TMLE weight 0, MSE weight 2") +
  ylab("Standardized error, other estimator") +
  theme_classic()