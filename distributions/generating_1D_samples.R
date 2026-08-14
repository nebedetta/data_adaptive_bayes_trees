library(ggplot2)

source('distributions/functions_for_mixtures.R')
names_list <- c("1D_beta91", "1D_beta82", 
  "1D_multispikevariedwidth", 
  "1D_fourspikedyadic", "1D_overlappingscales", 
  "1D_smoothvsskewed", "1D_spikemix")



for (name in c("1D_beta91", "1D_beta82", 
               "1D_multispikevariedwidth")) {
  
  source(paste0('distributions/', name, '_input.R'))
  
  grid_points <- read.csv("distributions/grid_points_1D.csv")
  grid_points <- grid_points$y
  z           <- compute_pdf(grid_points)
  
  new_data           <- data.frame(y = grid_points, true_density = z)
  
  #if (!dir.exists('distributions/true_density')) {
  #  dir.create('distributions/true_density', recursive = TRUE)
  #}
  #write.csv(new_data,
  #          file      = paste0("distributions/true_density/", name, "_true_density.csv"),
  #          row.names = FALSE)
  
  # --- true density plot ---
  plot_dir <- "distributions/true_density/plots"
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  
  p <- ggplot(new_data, aes(x = y, y = true_density)) +
    geom_line() +
    labs(title = paste0(name, " — true density"), x = "y", y = "density") +
    theme_minimal(base_size = 12)
  
  print(p)
  
  ggsave(paste0(plot_dir, "/", name, "_true_density.png"),
         plot = p, width = 6, height = 4, dpi = 150)
  
  seed_list        <- seq(0, 199)
  sample_size_list <- c(500, 5000)
  
  folder_path <- paste0("distributions/samples/", name, "/")
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }
  
  for (seed in seed_list) {
    for (sample_size in sample_size_list) {
      
      filename <- paste0(folder_path, name, "_n", sample_size, "_iter", seed, ".csv")
      
      set.seed(50000 * seed)
      samp           <- extract_samples(sample_size)
      colnames(samp) <- c("y")
      write.csv(samp, file = filename, row.names = FALSE)
    }
  }


}



