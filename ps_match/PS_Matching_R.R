rm(list = ls())
library("MatchIt")
library("data.table")
library("moonBook")
library('tidyverse')

setwd("./Desktop/Request/")

df_orig <- fread("./for_match.csv")

"df_orig <- df_orig[with(df_orig, order(HPCID, SM_DATE))]"

"case <- df_orig %>% 
  filter(AGE > 40 | RER_over_gs == 1) %>%
  group_by('HPCID') %>%
  top_n(1)"

mod_match <- matchit(
  HVA ~ AGE, 
  method="nearest", 
  data=df_orig, 
  ratio = 1
)

matched_case <- match.data(mod_match)

mytable(HVA~AGE, data=matched_case)

fwrite(matched_case, file = "./matched_sample_by_R.csv", bom = TRUE)
