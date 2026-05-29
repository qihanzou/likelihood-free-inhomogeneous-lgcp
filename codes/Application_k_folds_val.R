rm(list = ls())

set.seed(1)
seed <- sample.int(1e9, 1)
k = 1


t0 <- Sys.time()
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

# "method" is not important here. We only care first order estimation here.
fit_kppm <- kppm(X_train ~  elevation + waterdist + slopeangle + heat + slopetype + vegetation, 
                 clusters = "LGCP", method = "mincon", data = gorillas.extra)


# get the poisson trend 
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
Nobs <- X_train$n

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
      list(iter = j, L = Lvec, N = X_j$n)  # keep j in the result
    },
    future.seed = 0,
    future.packages = c("spatstat.geom", "spatstat.explore")
  )
})

L_list <- sapply(res_train, function(x) x[["L"]])
N_train <- sapply(res_train, function(x) x[["N"]])
train_par <- cbind(var_train, scale_train) 
# delet Na
keep <- colSums(is.na(L_list)) == 0  
L_list  <- L_list[, keep, drop = FALSE]
N_train <- as.matrix(N_train[keep])
train_par <- train_par[keep, ]
sum(keep)


m_L   <- rowMeans(L_list, na.rm = TRUE)                 
std_L <- apply(L_list, 1, sd) 
L_mat_std <- sweep(L_list, 1, m_L, "-")
L_mat_std <- sweep(L_mat_std, 1, std_L, "/")      
L_train <- array(as.numeric(t(L_mat_std)), dim = c(ncol(L_mat_std), nrow(L_mat_std), 1))

m_par  <- apply(train_par, 2, mean)
std_par <- apply(train_par, 2, sd)
Y_train <- scale(train_par, center = m_par, scale = std_par)

# For M1 only
m_N = mean(N_train)
std_N = sd(N_train)
N_train = as.matrix((N_train - m_N)/std_N)


# Obs
L_test1 <- (Lobs - m_L) / std_L
L_test_mat <- t(as.matrix(L_test1))    
L_test <- array(as.numeric(L_test_mat), dim = c(nrow(L_test_mat), ncol(L_test_mat), 1))
N_test <- as.matrix((Nobs - m_N) / std_N)


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

random.seed(0)
np.random.seed(0)

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
")
source_python("NN_M1_v3_APP.py")
pred_test_M1 = NN_model_est_M1(L_train, Y_train, L_test, N_train, N_test, batch_size=100, epochs=epochs, lr=1e-3)
Y_pred_unscaled_M1 = t(t(pred_test_M1[[1]])*std_par + m_par)

m1 = min(c(pred_test_M1[[2]]))
m2 = max(c(pred_test_M1[[2]]))
plot(0:epochsm1, pred_test_M1[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (M1)", type = "b", col = "red", xlim = c(0,epochsm1), ylim = c(m1, m2))
legend("topright", legend = c("Train MSE"), col = c("red"), lty = 1, pch = 1)                 


# ------------------------------------------------------------------------------
# M2:
# ------------------------------------------------------------------------------
py_run_string("
import os, random
import numpy as np

random.seed(0)
np.random.seed(0)

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
")
source_python("NN_M2_v3_APP.py")
pred_test_M2 = NN_model_est_M2(L_train, Y_train, L_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_unscaled_M2 = t(t(pred_test_M2[[1]])*std_par + m_par)
m11 = min(c(pred_test_M2[[2]]))
m22 = max(c(pred_test_M2[[2]]))
plot(0:epochsm1, pred_test_M2[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (M2)", type = "b", col = "red", xlim = c(0,epochsm1), ylim = c(m11, m22))
legend("topright", legend = c("Train MSE"), col = c("red"), lty = 1, pch = 1)                 


t1 <- Sys.time()
difftime(t1, t0, units = "mins") 


# 4 KPPM methods:
fit_mincon <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "mincon", data = gorillas.extra)
fit_clik2  <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "clik2", data = gorillas.extra)
fit_palm   <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "palm", data = gorillas.extra)
fit_adapcl <- kppm(X_train ~ elevation + waterdist + slopeangle + heat + slopetype + vegetation, clusters = "LGCP", method = "adapcl", data = gorillas.extra)


# envelope test:
env_lgcp1 = function(theta, method_name = "mincon", k){
  sigma2 = theta[1]     
  xi = theta[2]
  env_lgcp <- envelope(
    X_test,
    fun = function(Y, r) Kinhom(Y, r=r, lambda=lambda_im, correction="border"),
    r = r,
    simulate = expression({rLGCP(model="exponential", mu = log_lam_im - 0.5*sigma2, win = Window(X_train), var = sigma2, scale = xi, saveLambda = FALSE)}),
    nsim = 100,
    savefuns = TRUE, global = FALSE, verbose = FALSE
  )
  title_text <- paste0("Envelope of K-function (", method_name, ", k = ", k, ")")
  plot(env_lgcp, main = title_text)
  dclf.test(env_lgcp)
}

set.seed(0)
env_lgcp1(fit_mincon$par, "mincon", k)
set.seed(0)
env_lgcp1(fit_clik2$par, "clik2", k)
set.seed(0)
env_lgcp1(fit_palm$par, "palm", k)
set.seed(0)
env_lgcp1(fit_adapcl$par, "adapcl", k)
set.seed(0)
env_lgcp1(as.numeric(Y_pred_unscaled_M2), "DSBI", k)


scale_interval



