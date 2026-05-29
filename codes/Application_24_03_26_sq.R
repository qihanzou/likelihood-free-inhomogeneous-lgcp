rm(list = ls())
set.seed(2026)
library(spatstat.geom)
library(spatstat.random)
library(spatstat.model)
library(spatstat.explore)  
library(spatstat.random) 
library(reticulate)
library(future)
library(future.apply)
use_python("C:/Users/qihan/anaconda3/envs/py39env/python.exe", required = TRUE)
plan(multisession, workers = parallelly::availableCores() - 1)
data(gorillas, package = "spatstat.data")
X1 <- unique(unmark(gorillas))
all_dists <- nndist(gorillas)
scale_interval <- range(all_dists[all_dists != 0])

# Center and scale numeric covariates
gorillas.extra$elevation  = (gorillas.extra$elevation  - mean(gorillas.extra$elevation,  na.rm=TRUE))/sd(gorillas.extra$elevation,  na.rm=TRUE)
gorillas.extra$slopeangle = (gorillas.extra$slopeangle - mean(gorillas.extra$slopeangle, na.rm=TRUE))/sd(gorillas.extra$slopeangle, na.rm=TRUE)
gorillas.extra$waterdist  = (gorillas.extra$waterdist  - mean(gorillas.extra$waterdist,  na.rm=TRUE))/sd(gorillas.extra$waterdist,  na.rm=TRUE)

# need first order estimation.
fit_kppm <- kppm(X1 ~  elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", data = gorillas.extra)

# get the poisson trend 
ppm_trend <- as.ppm(fit_kppm)


elevation = as.im(ppm_trend$covariates$elevation)
waterdist = as.im(ppm_trend$covariates$waterdist)
slopeangle = as.im(ppm_trend$covariates$slopeangle)
heat_im = as.im(ppm_trend$covariates$heat)
slopetype_im = as.im(ppm_trend$covariates$slopetype)
vegetation_im = as.im(ppm_trend$covariates$vegetation)

heat_moderate = eval.im(heat_im == "Moderate")
heat_coolest = eval.im(heat_im == "Coolest")
slopetype_toe = eval.im(slopetype_im == "Toe")
slopetype_flat = eval.im(slopetype_im == "Flat")
slopetype_midslope = eval.im(slopetype_im == "Midslope")
slopetype_upper = eval.im(slopetype_im == "Upper")
slopetype_ridge = eval.im(slopetype_im == "Ridge")
vegetation_colonising = eval.im(vegetation_im == "Colonising")
vegetation_grassland = eval.im(vegetation_im == "Grassland")
vegetation_primary = eval.im(vegetation_im == "Primary")
vegetation_secondary = eval.im(vegetation_im == "Secondary")
vegetation_transition = eval.im(vegetation_im == "Transition")

beta0 = ppm_trend$coef[1]
beta_elevation = ppm_trend$coef[2]
beta_waterdist = ppm_trend$coef[3]
beta_slopeangle = ppm_trend$coef[4]
beta_heat_moderate = ppm_trend$coef[5]
beta_heat_coolest = ppm_trend$coef[6]
beta_slopetype_toe = ppm_trend$coef[7]
beta_slopetype_flat = ppm_trend$coef[8]
beta_slopetype_midslope = ppm_trend$coef[9]
beta_slopetype_upper = ppm_trend$coef[10]
beta_slopetype_ridge = ppm_trend$coef[11]
beta_vegetation_colonising = ppm_trend$coef[12]
beta_vegetation_grassland = ppm_trend$coef[13]
beta_vegetation_primary = ppm_trend$coef[14]
beta_vegetation_secondary = ppm_trend$coef[15]
beta_vegetation_transition = ppm_trend$coef[16]


beta_test = cbind(beta0, beta_elevation, beta_waterdist, beta_slopeangle, beta_heat_moderate, beta_heat_coolest, beta_slopetype_toe, beta_slopetype_flat,beta_slopetype_midslope,beta_slopetype_upper,beta_slopetype_ridge,beta_vegetation_colonising,beta_vegetation_grassland,beta_vegetation_primary,beta_vegetation_secondary,beta_vegetation_transition)


mean_trend = beta0 + beta_elevation * elevation +  beta_waterdist * waterdist + beta_slopeangle * slopeangle + beta_heat_moderate*heat_moderate + beta_heat_coolest*heat_coolest+ beta_slopetype_toe*slopetype_toe  + beta_slopetype_flat*slopetype_flat + beta_slopetype_midslope*slopetype_midslope  + beta_slopetype_upper*slopetype_upper + beta_slopetype_ridge*slopetype_ridge + beta_vegetation_colonising*vegetation_colonising + beta_vegetation_grassland*vegetation_grassland  + beta_vegetation_primary*vegetation_primary + beta_vegetation_secondary*vegetation_secondary + beta_vegetation_transition*vegetation_transition 
lambda_mean_trend <- eval.im(exp(mean_trend))
log_lam_im <- eval.im(log(lambda_mean_trend))

W <- Window(log_lam_im) 
rmax  <- rmax.rule("K", W)
r     <- seq(0, rmax, length.out = 513)

Lobs <- Linhom(X1, lambda = lambda_mean_trend, r = r, correction = "border")
Lobs <- as.numeric(Lobs$border - Lobs$r)


summary_ppm_trend = summary(ppm_trend)
CIs = cbind(summary_ppm_trend$coefs.SE.CI$CI95.lo, summary_ppm_trend$coefs.SE.CI$CI95.hi)




# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------
ntrain = 10000

beta0_train = runif(ntrain,  CIs[1,1], CIs[1,2]) 
beta_elevation_train = runif(ntrain,  CIs[2,1], CIs[2,2]) 
beta_waterdist_train = runif(ntrain,  CIs[3,1], CIs[3,2])
beta_slopeangle_train = runif(ntrain,  CIs[4,1], CIs[4,2]) 

beta_heatModerate_train = runif(ntrain,  CIs[5,1], CIs[5,2]) 
beta_heatCoolest_train = runif(ntrain,  CIs[6,1], CIs[6,2])

beta_slopetypeToe_train = runif(ntrain,  CIs[7,1], CIs[7,2]) 
beta_slopetypeFlat_train = runif(ntrain,  CIs[8,1], CIs[8,2]) 
beta_slopetypeMidslope_train = runif(ntrain,  CIs[9,1], CIs[9,2])
beta_slopetypeUpper_train = runif(ntrain,  CIs[10,1], CIs[10,2]) 
beta_slopetypeRidge_train = runif(ntrain,  CIs[11,1], CIs[11,2])

beta_vegetationColonising_train = runif(ntrain,  CIs[12,1], CIs[12,2]) 
beta_vegetationGrassland_train = runif(ntrain,  CIs[13,1], CIs[13,2]) 
beta_vegetationPrimary_train = runif(ntrain,  CIs[14,1], CIs[14,2])
beta_vegetationSecondary_train = runif(ntrain,  CIs[15,1], CIs[15,2]) 
beta_vegetationTransition_train = runif(ntrain,  CIs[16,1], CIs[16,2])

var_lo = 0.001
var_up = 4
scale_lo = scale_interval[1]
scale_up = scale_interval[2]
var_train   <- runif(ntrain, var_lo, var_up)
scale_train <- runif(ntrain, scale_lo, scale_up)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))
library(progressr)
handlers(global = TRUE)           
handlers("txtprogressbar")         

with_progress({
  p <- progressor(along = seq_len(ntrain))
  res_train <- future_lapply(
    seq_len(ntrain),
    function(j) {
      p(sprintf("iteration %d/%d", j, ntrain))
      mean_trend_train = beta0_train[j] + beta_elevation_train[j]*elevation + beta_waterdist_train[j]*waterdist + beta_slopeangle_train[j]*slopeangle + beta_heatModerate_train[j]*heat_moderate + beta_heatCoolest_train[j]*heat_coolest+ beta_slopetypeToe_train[j]*slopetype_toe + beta_slopetypeFlat_train[j]*slopetype_flat + beta_slopetypeMidslope_train[j]*slopetype_midslope  + beta_slopetypeUpper_train[j]*slopetype_upper + beta_slopetypeRidge_train[j]*slopetype_ridge + beta_vegetationColonising_train[j]*vegetation_colonising + beta_vegetationGrassland_train[j]*vegetation_grassland + beta_vegetationPrimary_train[j]*vegetation_primary + beta_vegetationSecondary_train[j]*vegetation_secondary + beta_vegetationTransition_train[j]*vegetation_transition 
      lambda_mean_trend_train <- eval.im(exp(mean_trend_train))
      log_lam_im_train <- eval.im(log(lambda_mean_trend_train))
      mu_j <- log_lam_im_train - 0.5*var_train[j]
      X_j <- rLGCP(model = "exponential", mu = mu_j, win = W, saveLambda = FALSE, var = var_train[j], scale = scale_train[j])
      Lfun <- Linhom(X_j, lambda = lambda_mean_trend_train, r = r, correction = "border")
      Lvec <- as.numeric(Lfun$border - Lfun$r)
      list(iter = j, L = Lvec)  
    },
    future.seed = 2026,
    future.packages = c("spatstat.geom", "spatstat.explore")
  )
})






L_list <- sapply(res_train, function(x) x[["L"]])
train_par <- cbind(var_train, scale_train) 
keep <- colSums(is.na(L_list)) == 0  
L_list  <- L_list[, keep, drop = FALSE]
train_par <- train_par[keep, ]
beta_train = cbind(beta0_train[keep], beta_elevation_train[keep], beta_waterdist_train[keep], beta_slopeangle_train[keep], beta_heatModerate_train[keep], beta_heatCoolest_train[keep], beta_slopetypeToe_train[keep], beta_slopetypeFlat_train[keep], beta_slopetypeMidslope_train[keep], beta_slopetypeUpper_train[keep], beta_slopetypeRidge_train[keep], beta_vegetationColonising_train[keep], beta_vegetationGrassland_train[keep], beta_vegetationPrimary_train[keep], beta_vegetationSecondary_train[keep], beta_vegetationTransition_train[keep])


m_L   <- rowMeans(L_list, na.rm = TRUE)                 
std_L <- apply(L_list, 1, sd) 
L_mat_std <- sweep(L_list, 1, m_L, "-")
L_mat_std <- sweep(L_mat_std, 1, std_L, "/")      
L_train <- array(as.numeric(t(L_mat_std)), dim = c(ncol(L_mat_std), nrow(L_mat_std), 1))

Y_train <- sqrt(train_par)


# Obs
L_test1 <- (Lobs - m_L) / std_L
L_test_mat <- t(as.matrix(L_test1))    
L_test <- array(as.numeric(L_test_mat), dim = c(nrow(L_test_mat), ncol(L_test_mat), 1))


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
epochs = 20
epochsm1 = epochs - 1
# ------------------------------------------------------------------------------
# M1: 
# ------------------------------------------------------------------------------
py_run_string("
import os, random
import numpy as np

random.seed(2026)
np.random.seed(2026)

import torch
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
")
source_python("C:/Users/qihan/Desktop/LGCP/record10_19_03_2026/NN_M1_v3_APP.py")
pred_test_M1 = NN_model_est_M1(L_train, Y_train, L_test, beta_train, beta_test, batch_size=100, epochs=epochs, lr=1e-3)
Y_pred_unscaled_M1 = t(t(pred_test_M1[[1]]^2))

m1 = min(c(pred_test_M1[[2]]))
m2 = max(c(pred_test_M1[[2]]))
plot(0:epochsm1, pred_test_M1[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (M1)", type = "b", col = "red", xlim = c(0,epochsm1), ylim = c(m1, m2))
legend("topright", legend = c("Train MSE"), col = c("red"), lty = 1, pch = 1)                 











# 4 KPPM methods:
fit_mincon <- kppm(X1 ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "mincon", data = gorillas.extra)
fit_clik2  <- kppm(X1 ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "clik2", data = gorillas.extra)
fit_palm   <- kppm(X1 ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "palm", data = gorillas.extra)
fit_adapcl <- kppm(X1 ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "adapcl", data = gorillas.extra)



# envelope test:
env_lgcp1 = function(theta, method_name = "mincon"){
  sigma2 = theta[1]     
  xi = theta[2]
  env_lgcp <- envelope(
    X1,
    fun = function(Y, r) Kinhom(Y, r=r, lambda=lambda_mean_trend, correction="border"),
    r = r,
    simulate = expression({rLGCP(model="exponential", mu = log_lam_im - 0.5*sigma2, win = W, var = sigma2, scale = xi, saveLambda = FALSE)}),
    nsim = 100,
    savefuns = TRUE, global = FALSE, verbose = FALSE
  )
  title_text <- paste0("Envelope of K-function (", method_name, ")")
  plot(env_lgcp, main = title_text)
  dclf.test(env_lgcp)
}

set.seed(2026)
plot.res = 600
png("app_envtest_mincon.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
env_lgcp1(fit_mincon$par, "mincon")
dev.off()

set.seed(2026)
plot.res = 600
png("app_envtest_clik2.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
env_lgcp1(fit_clik2$par, "clik2")
dev.off()

set.seed(2026)
plot.res = 600
png("app_envtest_palm.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
env_lgcp1(fit_palm$par, "palm")
dev.off()

set.seed(2026)
plot.res = 600
png("app_envtest_adapcl.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
env_lgcp1(fit_adapcl$par, "adapcl")
dev.off()

set.seed(2026)
plot.res = 600
png("app_envtest_dsbi_sq.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
env_lgcp1(as.numeric(Y_pred_unscaled_M1), "DSBI (sqrt)")
dev.off()




# plot.res = 600
# png("gorillasdata.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# ppm_trend <- as.ppm(fit_kppm)
# lambda_im <- predict(ppm_trend, type = "trend") 
# plot(lambda_im, main = "gorillas data in fitted first-order surface", col = terrain.colors(256))
# plot(X1, add = TRUE, cex = 0.6, col = "black")
# dev.off()
# 
# 
# # plot.res = 600
# # png("gorillasdata.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# # plot(X1, main = "gorillas data")
# # dev.off()
# 
# 
# plot.res = 600
# png("waterdist.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$waterdist, main = "Distance from nearest water source (metres)")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()
# 
# 
# plot.res = 600
# png("waterdist.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$waterdist, main = "Distance from nearest water source (metres)")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()
# 
# 
# plot.res = 600
# png("elevation.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$elevation, main = "Elevation of terrain (metres)")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()
# 
# plot.res = 600
# png("slopeangle.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$slopeangle, main = "Terrain slope (degrees)")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()
# 
# plot.res = 600
# png("heat.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$heat, main = "Heat Load Index")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()
# 
# plot.res = 600
# png("slopetype.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$slopetype, main = "Slope type")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()
# 
# 
# plot.res = 600
# png("vegetation.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
# plot(gorillas.extra$vegetation, main = "Vegetation type")
# #plot(X1, add = TRUE, pch = 16, cex = 0.5, col = "black")
# dev.off()



