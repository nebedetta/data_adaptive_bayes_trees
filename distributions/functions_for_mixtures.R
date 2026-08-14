
# Function to sample from a mixture distribution 

sample_from_mixture <- function(components_sample, weights, sample_size){
  
  weights <- weights / sum(weights)
  component_indices <- sample(1:length(components_sample), size = sample_size, 
                              replace = TRUE, prob = weights)
  
  
  samples <- lapply(component_indices, function(i) components_sample[[i]]())
  
  return(do.call(rbind, samples))
  
}

# Function to evaluate the pdf from a mixture distribution 

pdf_for_mixture <- function(components_pdf, weights, points){
  
  pdf_values <- sapply(1:length(components_pdf), function(i) components_pdf[[i]](points))
  
  return(pdf_values%*%weights)
  
}

#############################
# Bivariate Generalized Beta Distribution

gen_beta_sample <- function(n, a0, b0, a1, b1, a2, b2){
  
  g0<- rgamma(n, a0, b0)
  g1<- rgamma(n, a1, b1)
  g2<- rgamma(n, a2, b2)
  
  x = g1/(g1+g0)
  y = g2/(g2+g0)
  
  return(cbind(x, y))
}


gen_beta_density <- function(x, y, a0, b0, a1, b1, a2, b2){
  
  #a0 = 5
  #b0 = 10
  #a1 = 3
  #b1 = 10
  #a2 = 3
  #b2 = 10
  
  logB = lgamma(a0) + lgamma(a1) + lgamma(a2) - lgamma(a0 + a1 + a2)
  lambda1 = b1/b0
  lambda2 = b2/b0
  
  #x = 0.3
  #y = 0.4
  
  log_pdf_num = a1*log(lambda1) + (a1-1)*log(x) -(a1+1)*log(1-x) + 
    a2*log(lambda2) + (a2-1)*log(y) -(a2+1)*log(1-y)
  log_pdf_den = (a0+a1+a2)*log(1 + lambda1*x/(1-x) + lambda2*y/(1-y))
  
  log_pdf = log_pdf_num - log_pdf_den - logB
  log_pdf
  
  return(exp(log_pdf))
}
