rm(list = ls())
set.seed(2025)
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
plot(X1, main = "gorillas data")


# kppm for obs data, beta and kppm var and scale:
fit_kppm <- kppm(
  X1 ~ aspect + elevation + heat + slopeangle + slopetype + vegetation + waterdist,
  clusters = "LGCP",
  method   = "mincon",
  data     = gorillas.extra)

sumfit = summary(fit_kppm)
tab = sumfit$coefs.SE.CI




# simple simulation, unable to change parameters:
# sim1 <- simulate(fit_kppm, nsim = 1, saveLambda = TRUE)
# plot(sim1)

# get the poisson trend 
ppm_trend <- as.ppm(fit_kppm)
lambda_im <- predict(ppm_trend, type = "trend")  
log_lam_im <- eval.im(log(lambda_im))
# simulate 1 new realization based on beta hat and var hat, scale hat (exact ouput from kppm)
sigma2 <- unname(fit_kppm$clustpar[1])
alpha  <- unname(fit_kppm$clustpar[2])
W <- Window(fit_kppm)  
X_new <- rLGCP(model = "exponential", mu = log_lam_im, win = W, saveLambda = FALSE, var = sigma2, scale = alpha)
plot(X_new)
X_new$n

# set r
rmax  <- rmax.rule("K", W)
r     <- seq(0, rmax, length.out = 513)
# L obs
Lobs <- Linhom(X1, r = r, lambda = lambda_im, correction = "border")
Lobs <- as.numeric(Lobs$border - Lobs$r)
Nobs <- X1$n

# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------
ntrain = 20000
var_train   <- runif(ntrain, 0.001, 4)
scale_train <- runif(ntrain, 100, 2000)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

res_train <- future_lapply(
  seq_len(ntrain),
  function(j) {
    X_j <- rLGCP(model = "exponential", mu = log_lam_im, win = W, saveLambda = FALSE, var = var_train[j], scale = scale_train[j])
    Lfun <- Linhom(X_j, r = r, lambda = lambda_im, correction = "border")
    Lvec <- as.numeric(Lfun$border - Lfun$r)
    list(L = Lvec, N = X_j$n) 
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore")
)



L_list <- sapply(res_train, function(x) x[["L"]])
m_L   <- rowMeans(L_list, na.rm = TRUE)                 
std_L <- apply(L_list, 1, sd) 
L_mat_std <- sweep(L_list, 1, m_L, "-")
L_mat_std <- sweep(L_mat_std, 1, std_L, "/")      
L_train <- array(as.numeric(t(L_mat_std)), dim = c(ncol(L_mat_std), nrow(L_mat_std), 1))

train_par <- cbind(var_train, scale_train)        
m_par  <- apply(train_par, 2, mean)
std_par <- apply(train_par, 2, sd)
Y_train <- scale(train_par, center = m_par, scale = std_par)

# For M1 only
N_train <- sapply(res_train, function(x) x[["N"]])
m_N = mean(N_train)
std_N = sd(N_train)
N_train = as.matrix((N_train - m_N)/std_N)


# Obs
L_test1 <- (Lobs - m_L) / std_L
L_test_mat <- t(as.matrix(L_test1))    
L_test <- array(as.numeric(L_test_mat), dim = c(nrow(L_test_mat), ncol(L_test_mat), 1))
N_test <- as.matrix((Nobs - m_N) / std_N)

# ------------------------------------------------------------------------------
# M1: 
# ------------------------------------------------------------------------------
py_run_string("
import os, random
import numpy as np

random.seed(2025)
np.random.seed(2025)

import torch
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
")
source_python("NN_M1_padding0.py")
pred_test_M1 = NN_model_est_M1(L_train, Y_train, L_test, N_train, N_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_unscaled_M1 = t(t(pred_test_M1)*std_par + m_par)


# ------------------------------------------------------------------------------
# M2:
# ------------------------------------------------------------------------------
py_run_string("
import os, random
import numpy as np

random.seed(2025)
np.random.seed(2025)

import torch
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
")
source_python("NN_M2_padding0.py")
pred_test_M2 = NN_model_est_M2(L_train, Y_train, L_test, epochs=20, lr=1e-2)
Y_pred_unscaled_M2 = t(t(pred_test_M2)*std_par + m_par)


t1 <- Sys.time()
difftime(t1, t0, units = "mins") 


