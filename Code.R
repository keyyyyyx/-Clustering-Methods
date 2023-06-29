
### -----------------------------------------------------------
### Setups
# load packages
library(meanShiftR)
library(ks)
library(tidyverse)
library(Rtsne)
library(caret)
library(plot.matrix)
library(randomForest)
library(RColorBrewer)
library(cluster)
library(factoextra)
library(fpc)
library(fossil)
# load dataset
dat <- read.csv("project-data.csv", header = FALSE)

### -----------------------------------------------------------
### Data processing
# check for missings
sum(is.na(dat))
# check data distributions
par(mfrow = c(3,3))
for (i in 1:ncol(dat)) {
  hist(dat[, i])
}
## try dimension reduction with PCA
PCs <- prcomp(dat, scale. = TRUE)
PCs$x
# Visualize the Results with a Biplot
png(filename = "biplot.png")
biplot(PCs)
dev.off()
# Find Variance Explained by Each Principal Component
var_explained <- PCs$sdev^2 / sum(PCs$sdev^2)
# create and save  scree plot
png(filename = "ScreePlot.png")
ggplot(mapping  = aes(x = 1:64, y = var_explained)) +
  geom_line() +
  xlab("Principal Component") +
  ylab("Variance explained") +
  theme_bw()
dev.off()
## try dimension reduction with t-SNE
# convert data into matrix
dat_matrix <- as.matrix(dat)
set.seed(527)
tsne_out <- Rtsne(dat_matrix)
dat_tsne <- data.frame(x = tsne_out$Y[,1],
                       y = tsne_out$Y[,2])
# Plotting the 2D visualization of data created by t-SNE
png(filename = "tSNEVisual.png")
ggplot2::ggplot(dat_tsne, mapping = aes(x = x, y = y)) +
  geom_point() +
  theme_bw()
dev.off()
# check kde estimate
tsne_kde <- kde(dat_tsne)
png(filename = "tsneKDE.png")
plot(tsne_kde, main = NULL)
dev.off()

### -----------------------------------------------------------
### Mean Shift Clustering
## mean shift with original data with default h; too slow & too many clusters
ms1 <- meanShift(as.matrix(dat), iterations = 300)
png(filename = "ms1.png")
plot(dat_tsne$x, dat_tsne$y, col = ms1$assignment, main = NULL)
dev.off()
# length(unique(ms1$assignment))

## mean shift with original data with (optimal) h by CV; too many clusters
## choose optimal bandwidth
H <- numeric(64)
for (i in 1:64) {
  H[i] <- bw.ucv(dat[, i], )
}
H
ms2 <- meanShift(as.matrix(dat), iterations = 300)
png(filename = "ms2.png")
plot(dat_tsne$x, dat_tsne$y, col = ms2$assignment, main = NULL)
dev.off()
# length(unique(ms2$assignment))

# mean shift with dimension-reudced data with cv h; 
H <- numeric(2)
for (i in 1:2) {
  H[i] <- bw.bcv(dat_tsne[, i])
}
H
ms3 <- meanShift(as.matrix(dat_tsne), iterations = 150)
png(filename = "ms3.png")
ggplot(mapping = aes(x = dat_tsne$x, y = dat_tsne$y, color = as.factor(ms3$assignment))) +
  geom_point() +
  theme_bw() +
  guides(color = "none") +
  xlab("x") +
  ylab("y") 
dev.off()
length(unique(ms3$assignment))

## mean shift with dimension-reudced data with optimal H chosen by Hpi.diag; optimal
# choose optimal bandwidth
set.seed(527)
H <- Hpi.diag(dat_tsne)
H
plot(kde(dat_tsne, H = H))
# number of clusters vs number of iterations
nite <- seq(50, 500, by = 25)
nclust <- numeric(19)
for (i in 1:19) {
  out <- meanShift(as.matrix(dat_tsne), bandwidth = diag(H), iterations = nite[i])
  nclust[i] <- out$assignment %>% unique() %>% length()
  print(nclust)
}
png(filename = "nclustByITE.png")
plot(x = nite, y = nclust, xlab = "number of iterations", ylab = "number of clusters", type = "b")
dev.off()
# cluster using mean shift, with selected parameters
ms4 <- meanShift(as.matrix(dat_tsne), bandwidth = c(H[1, 1], H[2, 2]), iterations = 150)
write.table(ms4$assignment, file = "alg1_out.txt", sep = "\t", row.names = FALSE, col.names = FALSE)
# sil4 <- silhouette(ms4$assignment, dist = dist(dat))
cluster <- as.factor(ms4$assignment)
my_colors <- c(RColorBrewer::brewer.pal(12, "Paired"), "#000000")
png(filename = "ms4.png")
ggplot(mapping = aes(x = dat_tsne$x, y = dat_tsne$y, color = cluster)) +
  geom_point() +
  scale_color_manual(values = my_colors) +
  theme_bw() +
  xlab("x") +
  ylab("y") 
dev.off()
# length(unique(ms4$assignment))

### -----------------------------------------------------------
### Hierarchical Clustering
# With dimension reduced data
clusters <- hclust(dist(dat_tsne), method = "ward.D2")
png(filename = "dendrogram1.png")
plot(clusters, main = NULL)
dev.off()
# within sum of squares by number of clusters
png(filename = "ncluster1.png")
fviz_nbclust(dat_tsne, hcut, method = "wss", k.max = 20)
dev.off()
clusterCut <- cutree(clusters, 10) %>% as.factor()
png(filename = "HC1.png")
ggplot(mapping = aes(x = dat_tsne$x, y = dat_tsne$y, color = clusterCut)) +
  geom_point() +
  scale_color_manual(values = my_colors) +
  theme_bw() +
  xlab("x") +
  ylab("y") 
dev.off()

# without dimension reduction
clusters2 <- hclust(dist(dat), method = "ward.D2")
# plot(clusters2)
# plot within sum of squares by number of clusters
png(filename = "ncluster2.png")
fviz_nbclust(dat, hcut, method = "wss", k.max = 20)
dev.off()
clusterCut2 <- cutree(clusters2, 10)
write.table(clusterCut2, file = "alg2_out.txt", sep = "\t", row.names = FALSE, col.names = FALSE)
clusterCut2 <- clusterCut2 %>% as.factor()
png(filename = "HC2.png")
ggplot(mapping = aes(x = dat_tsne$x, y = dat_tsne$y, color = clusterCut2)) +
  geom_point() +
  scale_color_manual(values = my_colors) +
  theme_bw() +
  xlab("x") +
  ylab("y") 
dev.off()


### -----------------------------------------------------------
### Model Evaluation
## Stability Analysis

# Mean shift: change seed set 1
set.seed(533)
tsne_out_seed1 <- Rtsne(dat_matrix)
dat_tsne_seed1 <- data.frame(x = tsne_out_seed1$Y[,1],
                       y = tsne_out_seed1$Y[,2])
H <- Hpi.diag(dat_tsne_seed1)
ms_seed1 <- meanShift(as.matrix(dat_tsne_seed1), bandwidth = c(H[1, 1], H[2, 2]), iterations = 150)
cluster_seed1 <- ms_seed1$assignment
table(cluster, cluster_seed1)
# compute external evaluation metrices
rand_seed1 <- rand.index(as.numeric(cluster), cluster_seed1)
J_seed1 <- jaccard(as.numeric(cluster), cluster_seed1)

# Mean shift: change seed set 2
set.seed(579)
tsne_out_seed2 <- Rtsne(dat_matrix)
dat_tsne_seed2 <- data.frame(x = tsne_out_seed2$Y[,1],
                             y = tsne_out_seed2$Y[,2])
H <- Hpi.diag(dat_tsne_seed2)
ms_seed2 <- meanShift(as.matrix(dat_tsne_seed2), bandwidth = c(H[1, 1], H[2, 2]), iterations = 150)
cluster_seed2 <- ms_seed2$assignment
table(cluster, cluster_seed2)
rand_seed2 <- rand.index(as.numeric(cluster), cluster_seed2)

# Mean shift: change bandwidth 1
set.seed(527)
ms_H1 <- meanShift(as.matrix(dat_tsne), bandwidth = c(4.5, 5), iterations = 150)
cluster_H1 <- ms_H1$assignment
table(cluster, cluster_H1)
rand_H1 <- rand.index(as.numeric(cluster), cluster_H1)

# Mean shift: change bandwidth 2
set.seed(527)
ms_H2 <- meanShift(as.matrix(dat_tsne), bandwidth = c(4, 4.5), iterations = 150)
cluster_H2 <- ms_H2$assignment
table(cluster, cluster_H2)
rand_H2 <- rand.index(as.numeric(cluster), cluster_H2)

# Mean shift: Bootstrapping
set.seed(527)
rand_ms <- numeric(20)
for (i in 1:20) {
  index <- sample(12000, 2000, TRUE)
  boot <- dat_tsne[index, ]
  # H <- Hpi.diag(dat_tsne)
  dissim <- meanShift(as.matrix(boot), bandwidth = c(H[1, 1], H[2, 2]), iterations = 150)
  clust <- dissim$assignment
  rand_ms[i] <- rand.index(as.numeric(cluster[index]), clust)
}
mean(rand_ms)

# Hierarchical: Bootsrtap
set.seed(527)
rand_hier <- numeric(20)
for (i in 1:20) {
  index <- sample(12000, 2000, TRUE)
  boot <- dat[index, ]
  dissim <- hclust(dist(boot), method = "ward.D2")
  clust <- cutree(dissim, 10)
  rand_hier[i] <- rand.index(as.numeric(clusterCut2[index]), clust)
}
mean(rand_hier)

# Hierarchical on t-SNE: Bootsrtap
set.seed(527)
rand_hier <- numeric(20)
for (i in 1:20) {
  index <- sample(12000, 2000, TRUE)
  boot <- dat_tsne[index, ]
  dissim <- hclust(dist(boot), method = "ward.D2")
  clust <- cutree(dissim, 10)
  rand_hier[i] <- rand.index(as.numeric(clusterCut[index]), clust)
}
mean(rand_hier)

## Internal Evaluation
# mean shift
MS_eval <- cluster.stats(dist(dat), as.numeric(cluster))
MS_eval$within.cluster.ss
MS_eval$avg.silwidth
MS_eval$separation
MS_eval$average.between
MS_eval$ch
# hierarchical
HC_eval <- cluster.stats(dist(dat), as.numeric(clusterCut2))
HC_eval$within.cluster.ss
HC_eval$avg.silwidth
HC_eval$separation
HC_eval$average.between
HC_eval$ch
# hierarchical: t-SNE
HC_eval2 <- cluster.stats(dist(dat), as.numeric(clusterCut))
HC_eval2$within.cluster.ss
HC_eval2$avg.silwidth
HC_eval2$separation
HC_eval2$average.between
HC_eval2$ch

t1 <- table(cluster, clusterCut2)
M <- matrix(nrow = 11, ncol = 10)
for (i in 1:ncol(t1)) {
  maxi <- which.max(t1[, i])
  M[, maxi] <- t1[, i]
}
M[, 2] <- t1[, 1]

## Compare MS and Hierarchical
rand_MSH1 <- rand.index(as.numeric(cluster), as.numeric(clusterCut))
rand_MSH1
rand_H1H2 <- rand.index(as.numeric(clusterCut2), as.numeric(clusterCut))
rand_H1H2
rand_MSH <- rand.index(as.numeric(cluster), as.numeric(clusterCut2))
rand_MSH





