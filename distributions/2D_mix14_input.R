library(LaplacesDemon)
library(TruncatedNormal)
source('functions_for_mixtures.R')


set.seed(21)
cov1 <- matrix(c(0.01, 0, 0, 0.03), nrow = 2)
cov2 <- matrix(c(0.02, 0, 0, 0.02), nrow = 2)
#cov3 <- matrix(c(0.12, 0, 0, 0.12), nrow = 2)
cov4 <- matrix(c(0.01, 0, 0, 0.01), nrow = 2)
#cov5 <- matrix(c(0.01, 0, 0, 0.01), nrow = 2)

mu1 = c(0.2, 0.5)
mu2 = c(0.4, 0.3)
#mu3 = c(0.6, 0.7)
mu4 = c(0.8, 0.4)
#mu5 = c(0.5, 0.9)

a0 = 100
b0 = 1
a1 = 250
b1 = 1
a2 = 250
b2 = 1


component1_sample <- function(n=1){
  return(rtmvnorm(n, mu1, cov1, c(0, 0), c(1, 1)))
}

component1_pdf <- function(x){
  return(dtmvnorm(x, mu1, cov1, c(0, 0), c(1, 1)))
}

component2_sample <- function(n=1){
  return(rtmvnorm(n, mu2, cov2, c(0, 0), c(1, 1)))
}

component2_pdf <- function(x){
  return(dtmvnorm(x, mu2, cov2, c(0, 0), c(1, 1)))
}

component3_sample <- function(n=1){
  return(rtmvnorm(n, mu3, cov3, c(0, 0), c(1, 1)))
}

component3_pdf <- function(x){
  return(dtmvnorm(x, mu3, cov3, c(0, 0), c(1, 1)))
}



component5_sample <- function(n=1){
  return(gen_beta_sample(n, a0, b0, a1, b1, a2, b2))
}

component5_pdf <- function(x){
  return(gen_beta_density(x[,1], x[,2], a0, b0, a1, b1, a2, b2))
}


component_sample = list(component1_sample, component2_sample, component5_sample) 
component_pdf = c(component1_pdf, component2_pdf, component5_pdf)  
weights = c(0.40, 0.40, 0.2)

extract_samples <- function(n){
  return(sample_from_mixture(component_sample, weights, n))
}


compute_pdf <-function(points){
  return(pdf_for_mixture(component_pdf, weights, points))
}