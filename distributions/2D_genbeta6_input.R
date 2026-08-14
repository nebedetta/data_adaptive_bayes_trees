source('functions_for_mixtures.R')

a0 = 3
b0 = 1
a1 = 6
b1 = 1
a2 = 9
b2 = 1


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