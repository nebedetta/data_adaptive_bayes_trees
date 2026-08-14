library(ggplot2)
library(LaplacesDemon)
library(MASS)
library(TruncatedNormal)
library(plotly)

source('distributions/functions_for_mixtures.R')
source('distributions/2D_mix14_input.R')

name = "2D_mix14"


grid_points <- read.csv("distributions/grid_points_2D.csv.gz")
grid_points <- as.matrix(grid_points)
z <- compute_pdf(grid_points)

#ggplot(data = data.frame(x = grid_points[,1], y = grid_points[,2], z = z), aes(x = x, y = y, fill = z)) +
#  geom_tile() + theme_minimal() + scale_fill_gradient(low = "blue", high = "red")

density_data = data.frame("x" = grid_points[,1], "y" = grid_points[,2], "z" = z)
new_data <- cbind(grid_points, z)
colnames(new_data) <- c("x", "y", "true_density")

if (!dir.exists('distributions/true_density')) {
  dir.create('distributions/true_density', recursive = TRUE)
}
write.csv(new_data, file = paste0("distributions/true_density/", name, "_true_density.csv"), row.names = FALSE)



seed_list <- seq(0, 199)
sample_size_list <- c(500, 5000, 50000)
#sample_size_list <- c(5000)

for (seed in seed_list){
  
  folder_path <- paste0("distributions/samples/", name, "/")
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }
  
  
  for (sample_size in sample_size_list){
    
    filename <- paste0(folder_path, name, "_n", sample_size, "_iter", seed, ".csv")

    if (!file.exists(filename)) {
      set.seed(50000*seed)
      sample <- extract_samples(sample_size)
      colnames(sample) <- c("x", "y")
      filename <- paste0(folder_path, name, "_n", sample_size, "_iter", seed, ".csv")
      write.csv(sample, file = filename, row.names = FALSE)
      
      
    }else{
      print(paste0("File ", filename, " already exists"))
    }
    
  }
}


