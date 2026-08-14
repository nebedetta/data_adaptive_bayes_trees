source('functions_for_mixtures.R')

a0 = 5
b0 = 10
a1 = 3
b1 = 10
a2 = 3
b2 = 10


extract_samples <- function(n){
  return(gen_beta_sample(n, a0, b0, a1, b1, a2, b2))
}


compute_pdf <-function(points){
  
  den = rep(0, nrow(points))
  for (i in 1:nrow(points)){
    den[i] <- gen_beta_density(points[i,1], points[i,2], a0, b0, a1, b1, a2, b2)
  }
  
  return(den)
}