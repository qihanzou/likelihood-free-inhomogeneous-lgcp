rm(list = ls())

set.seed(1)
seed <- sample.int(1e9, 1)

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

# train and test
n <- npoints(X1)
idx_keep <- sample(1:n, size = floor(0.8*n), replace = FALSE)
X_train <- X1[idx_keep]
X_test <- X1[-idx_keep]

npoints(X1)  # total
npoints(X_train)  # 80%
npoints(X_test)  # 20%
plot(X_train, main = "Observed (80%)")
plot(X_test, main = "Unobserved (20%)")

# Use train only
all_dists <- nndist(X_train)
scale_interval <- range(all_dists[all_dists != 0])

# Center and scale numeric covariates, scale numeric image covariates directly
gorillas.extra$elevation  = (gorillas.extra$elevation  - mean(gorillas.extra$elevation,  na.rm=TRUE))/sd(gorillas.extra$elevation,  na.rm=TRUE)
gorillas.extra$slopeangle = (gorillas.extra$slopeangle - mean(gorillas.extra$slopeangle, na.rm=TRUE))/sd(gorillas.extra$slopeangle, na.rm=TRUE)
gorillas.extra$waterdist  = (gorillas.extra$waterdist  - mean(gorillas.extra$waterdist,  na.rm=TRUE))/sd(gorillas.extra$waterdist,  na.rm=TRUE)

fit_kppm <- kppm(X_train ~  elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "mincon", data = gorillas.extra)
# get the mean trend 
ppm_trend <- as.ppm(fit_kppm)
lambda_im <- predict(ppm_trend, type = "trend")  
log_lam_im <- eval.im(log(lambda_im))
W <- Window(fit_kppm) 

# set r
rmax  <- rmax.rule("K", W)
r     <- seq(0, rmax, length.out = 513)
# L obs
Lobs <- Linhom(X_train, lambda = lambda_im, r = r, correction = "border")
Lobs <- as.numeric(Lobs$border - Lobs$r)

# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------
ntrain = 10000

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
      mu_j <- log_lam_im - 0.5*var_train[j]
      X_j <- rLGCP(
        model = "exponential",
        mu = mu_j,
        win = W,
        saveLambda = FALSE,
        var = var_train[j],
        scale = scale_train[j]
      )
      Lfun <- Linhom(X_j, lambda = lambda_im, r = r, correction = "border")
      Lvec <- as.numeric(Lfun$border - Lfun$r)
      list(iter = j, L = Lvec)  
    },
    future.seed = 0,
    future.packages = c("spatstat.geom", "spatstat.explore")
  )
})

L_list <- sapply(res_train, function(x) x[["L"]])
train_par <- cbind(var_train, scale_train) 
keep <- colSums(is.na(L_list)) == 0  
L_list  <- L_list[, keep, drop = FALSE]
train_par <- train_par[keep, ]
m_L   <- rowMeans(L_list, na.rm = TRUE)                 
std_L <- apply(L_list, 1, sd) 
L_mat_std <- sweep(L_list, 1, m_L, "-")
L_mat_std <- sweep(L_mat_std, 1, std_L, "/")      
L_train <- array(as.numeric(t(L_mat_std)), dim = c(ncol(L_mat_std), nrow(L_mat_std), 1))
m_par  <- apply(train_par, 2, mean)
std_par <- apply(train_par, 2, sd)
Y_train <- scale(train_par, center = m_par, scale = std_par)
L_test1 <- (Lobs - m_L) / std_L
L_test_mat <- t(as.matrix(L_test1))    
L_test <- array(as.numeric(L_test_mat), dim = c(nrow(L_test_mat), ncol(L_test_mat), 1))


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
epochs = 20
epochsm1 = epochs - 1
# ------------------------------------------------------------------------------
# DSBI
py_run_string("
import os, random
import numpy as np

random.seed(0)
np.random.seed(0)

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
")
source_python("NN_DSBI_app.py")
pred_test_DSBI = NN_model_est_DSBI(L_train, Y_train, L_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_unscaled_DSBI = t(t(pred_test_DSBI[[1]])*std_par + m_par)
m11 = min(c(pred_test_DSBI[[2]]))
m22 = max(c(pred_test_DSBI[[2]]))
plot(0:epochsm1, pred_test_DSBI[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (DSBI)", type = "b", col = "red", xlim = c(0,epochsm1), ylim = c(m11, m22))
legend("topright", legend = c("Train MSE"), col = c("red"), lty = 1, pch = 1)                 


fit_mincon <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "mincon", data = gorillas.extra)
fit_clik2  <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "clik2", data = gorillas.extra)
fit_palm   <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "palm", data = gorillas.extra)
fit_adapcl <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "adapcl", data = gorillas.extra)




