# ============================================================
# 1D_beta91_input.R
# Beta(9, 1)
# ============================================================

extract_samples <- function(n) {
  return(matrix(rbeta(n, 9, 1), ncol = 1))
}

compute_pdf <- function(x) {
  dbeta(x, 9, 1)
}
