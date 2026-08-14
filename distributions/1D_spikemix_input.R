# ============================================================
# 1D_spikemix_input.R
# 0.1*U(0,1) + 0.3*U(0.25,0.5) + 0.4*Beta_[0.25,0.5](2,2) + 0.2*Beta(6000,4000)
# Sharp spike (Beta(6000,4000), std ~0.005) centered at x=0.6.
# ============================================================

weights_spikemix <- c(0.1, 0.3, 0.4, 0.2)

.comp1_sample <- function() runif(1, 0, 1)
.comp1_pdf    <- function(x) dunif(x, 0, 1)

.comp2_sample <- function() runif(1, 0.25, 0.5)
.comp2_pdf    <- function(x) dunif(x, 0.25, 0.5)

.comp3_sample <- function() 0.25 + 0.25 * rbeta(1, 2, 2)
.comp3_pdf    <- function(x) dbeta((x - 0.25) / 0.25, 2, 2) / 0.25

.comp4_sample <- function() rbeta(1, 6000, 4000)
.comp4_pdf    <- function(x) dbeta(x, 6000, 4000)

extract_samples <- function(n) {
  comp    <- sample(1:4, size = n, replace = TRUE, prob = weights_spikemix)
  samples <- numeric(n)
  samples[comp == 1] <- runif(sum(comp == 1), 0, 1)
  samples[comp == 2] <- runif(sum(comp == 2), 0.25, 0.5)
  samples[comp == 3] <- 0.25 + 0.25 * rbeta(sum(comp == 3), 2, 2)
  samples[comp == 4] <- rbeta(sum(comp == 4), 6000, 4000)
  return(matrix(samples, ncol = 1))
}

compute_pdf <- function(x) {
  weights_spikemix[1] * .comp1_pdf(x) +
    weights_spikemix[2] * .comp2_pdf(x) +
    weights_spikemix[3] * .comp3_pdf(x) +
    weights_spikemix[4] * .comp4_pdf(x)
}
