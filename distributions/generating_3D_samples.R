library(ggplot2)
library(LaplacesDemon)
library(MASS)
library(TruncatedNormal)
library(plotly)
library(GGally)

source('distributions/functions_for_mixtures.R')



d <- 3

# Run this whole script once per scenario by setting `name` below and
# re-sourcing, or loop over all three via `name_list` further down.
name_list <- c("3D_smooth1", "3D_genbetaspike1", "3D_mix1")

for (name in name_list) {

  source(paste0('distributions/', name, '_input.R'))

  ########################################################
  #################### QUICK PLOTTING ####################

  # --- quick sample for plotting only ---
  set.seed(42)
  n_plot  <- 5000
  sample_plot <- extract_samples(n_plot)
  df_plot <- as.data.frame(sample_plot)
  colnames(df_plot) <- paste0("x", 0:(d-1))

  # --- marginal densities (generic per-dimension labels; scenarios differ
  # in parameterization so we don't hardcode shape params here) ---
  par(mfrow = c(1, 3))
  for (j in 0:(d-1)) {
    plot(density(df_plot[[paste0("x", j)]]),
         main = paste0(name, " — x", j),
         xlab = paste0("x", j),
         xlim = c(0, 1),
         col  = "steelblue", lwd = 2)
  }
  par(mfrow = c(1, 1))

  # --- pairwise scatterplots + marginals + correlations ---
  print(
    ggpairs(df_plot,
            lower = list(continuous = wrap("density", alpha = 0.6)),
            diag  = list(continuous = wrap("densityDiag", color = "steelblue")),
            upper = list(continuous = wrap("cor", size = 3))) +
      theme_minimal() +
      ggtitle(paste0(name, " — pairwise structure (n=", n_plot, ")"))
  )

}

for (name in name_list){

  source(paste0('distributions/', name, '_input.R'))

  ########################################################
  ################## GENERATING SAMPLES ##################

  # --- true density on regular grid ---
  grid_points <- read.csv("distributions/grid_points_3D.csv.gz")
  grid_points <- as.matrix(grid_points)
  z <- compute_pdf(grid_points)

  col_names <- c(paste0("x", 0:(d-1)), "true_density")
  new_data  <- cbind(grid_points, z)
  colnames(new_data) <- col_names

  if (!dir.exists('distributions/true_density')) {
    dir.create('distributions/true_density', recursive = TRUE)
  }
  write.csv(new_data,
            file = paste0("distributions/true_density/", name, "_true_density.csv"),
            row.names = FALSE)

  # --- samples ---
  seed_list        <- seq(0, 199)
  sample_size_list <- c(50000)

  folder_path <- paste0("distributions/samples/", name, "/")
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }

  for (seed in seed_list) {
    for (sample_size in sample_size_list) {

      filename <- paste0(folder_path, name, "_n", sample_size, "_iter", seed, ".csv")

      set.seed(50000 * seed)
      sample <- extract_samples(sample_size)
      colnames(sample) <- paste0("x", 0:(d-1))
      write.csv(sample, file = filename, row.names = FALSE)

    }
  }

  cat("Finished scenario:", name, "\n")
}
