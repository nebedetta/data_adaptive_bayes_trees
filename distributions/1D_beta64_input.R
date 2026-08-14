# ============================================================
# 1D_beta64_input.R
# Beta(6, 4)
# ============================================================

extract_samples <- function(n) {
  return(matrix(rbeta(n, 6, 4), ncol = 1))
}

compute_pdf <- function(x) {
  dbeta(x, 6, 4)
}
