# ============================================================
# 1D_beta50020_input.R
# Beta(500, 20)
# ============================================================

extract_samples <- function(n) {
  return(matrix(rbeta(n, 500, 20), ncol = 1))
}

compute_pdf <- function(x) {
  dbeta(x, 500, 20)
}
