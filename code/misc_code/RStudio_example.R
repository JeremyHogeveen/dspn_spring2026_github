# Load tidyverse
library(tidyverse)

# Simulate some RTs on an explore-exploit task
set.seed(42)
task_data <- tibble(
  trial = 1:100,
  condition = rep(c("Explore", "Exploit"), each = 50),
  rt_ms = c(rnorm(50, mean = 650, sd = 100), rnorm(50, mean = 400, sd = 80))
)

# Look at the first few rows
head(task_data)

# Plot the results
ggplot(task_data, aes(x = condition, y = rt_ms, fill = condition)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Reaction Times: Explore vs. Exploit", y = "Reaction Time (ms)")