set.seed(2025)
t0 <- Sys.time()
library(spatstat.geom)     
library(spatstat.explore)  
library(spatstat.random)   
library(spatstat.model)
#library(pracma)
library(reticulate)
library(future)
library(future.apply)
use_python("C:/Users/qihan/anaconda3/envs/py39env/python.exe", required = TRUE)
plan(multisession, workers = parallelly::availableCores() - 1)

simulate_grf = function(nx, ny, L, range, var) {
  mu0 = as.im(0, owin(xrange = c(0, L), yrange = c(0, L)), dimyx = c(ny, nx))
  sim_field = rLGCP(model = "exponential", mu = mu0, var = var, scale = range,
                    win = as.owin(mu0), saveLambda = TRUE)
  field_im  = attr(sim_field, "Lambda")
  field_im  = eval.im(log(field_im))   
  as.matrix(field_im)
}

simulate_LGCP_point_pattern <- function(params) {
  Y = simulate_grf(params$nx, params$ny, params$L, range = params$xi, var = params$alpha)
  
  Lambda = exp(params$gamma0 - 0.5*params$alpha + params$gamma1 * params$z + Y)
  
  Lambda_im = im(mat = Lambda, xrange = c(0, params$L), yrange = c(0, params$L))
  Xi = rpoispp(Lambda_im)
  list(points = cbind(Xi$x, Xi$y), Lambda = Lambda, fields = list(z = params$z, Y = Y))
}

z = simulate_grf(256, 256, 1, range = 0.05, var = 1)
z = z - mean(z)   # center z so the intercept is for average z


# L_inhom_exp <- function(r, sigma2, scale) {
#   g <- exp(sigma2*exp(-r/scale))
#   K <- 2*pi*cumtrapz(r, r*g) 
#   L <- sqrt(K / pi)
# }


W     <- owin(c(0,1), c(0,1))
rmax  <- rmax.rule("K", W)
r     <- seq(0, rmax, length.out = 513)

# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------
ntrain = 10000
beta0_train = runif(ntrain, 5, 5) # ALL 5
beta1_train = runif(ntrain, 0, 0) # ALL 0
var_train   <- runif(ntrain, 0.001, 4)
scale_train <- runif(ntrain, 0.001, 0.1)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

res_train <- future_lapply(
  seq_len(ntrain),
  function(j) {
    params <- list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                   xi = scale_train[j], alpha = var_train[j],
                   gamma0 = beta0_train[j],  gamma1 = beta1_train[j], z = z)
    sim <- simulate_LGCP_point_pattern(params)
    X1 <- ppp(x = sim$points[,1], y = sim$points[,2], window = owin(c(0, params$L), c(0, params$L)))
    Lfv <- Linhom(X1, r = r, correction = "border")
    Lvec <- as.numeric(Lfv$border - Lfv$r)
    
    #L_the = L_inhom_exp(r, sigma2 = var_train[j], scale = scale_train[j]) - r
    
    list(L = Lvec, N = length(sim$points[,1])) #, L_the = L_the)
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore")
)

# L_the <- sapply(res_train, function(x) x[["L_the"]]) 
# m_L_the   <- rowMeans(L_the, na.rm = TRUE)                 
# std_L_the <- apply(L_the, 1, sd)  
# L_the_std <- sweep(L_the, 1, m_L_the, "-")
# L_the_std[1, ] <- 0                  
# std_L_the[1]  <- 1
# L_the_std <- sweep(L_the_std, 1, std_L_the, "/")
# L_the_train <- array(as.numeric(t(L_the_std)),dim = c(ncol(L_the_std), nrow(L_the_std), 1))


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


# ------------------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------------------
ntest = 5000
beta0 = runif(ntest, 4, 6)
beta1 = runif(ntest, 0, 1)
var_test   <- runif(ntest, 0.001, 4)
scale_test <- runif(ntest, 0.001, 0.1)

plan(sequential); gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

res <- future_lapply(
  seq_len(ntest),
  function(j) {
    params <- list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                   xi = scale_test[j], alpha = var_test[j],
                   gamma0 = beta0[j],  gamma1 = beta1[j], z = z)
    sim <- simulate_LGCP_point_pattern(params)
    
    z_im <- im(mat  = sim$fields$z, xcol = seq(0, params$L, length.out = params$nx), yrow = seq(0, params$L, length.out = params$ny))
    X1 <- ppp(x = sim$points[,1], y = sim$points[,2], window = owin(c(0, params$L), c(0, params$L)))
    
    fits_kppm <- kppm(X1 ~ z_im, clusters = "LGCP", method = "mincon")
    Lfv <- Linhom(X1, r = r, correction = "border")
    L_test1 <- as.numeric(Lfv$border - Lfv$r)
    L_test1 <- (L_test1 - m_L) / std_L
    
    #L_the_test <- (L_test1 - m_L_the) / std_L_the
    
    beta_KPPM <- coef(fits_kppm)
    list(
      L      = L_test1,
      #L_the  = L_the_test,
      beta1  = as.numeric(beta_KPPM["z_im"]),
      beta0  = as.numeric(beta_KPPM["(Intercept)"]),
      var   <- as.numeric(fits_kppm$par[[1]]),
      scale <- as.numeric(fits_kppm$par[[2]]),
      N = length(sim$points[,1])
    )
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore", "spatstat.model")
)

L_test_list <- sapply(res, function(x) x[["L"]]) 
L_the_test <- sapply(res, function(x) x[["L_the"]]) 
beta0_KPPM  <- sapply(res, function(x) x[["beta0"]])
beta1_KPPM  <- sapply(res, function(x) x[["beta1"]])
scale_KPPM  <- sapply(res, function(x) x[[5]])
var_KPPM    <- sapply(res, function(x) x[[4]])

L_test_mat <- t(as.matrix(L_test_list))       
L_test <- array(as.numeric(L_test_mat), dim = c(nrow(L_test_mat), ncol(L_test_mat), 1))

# L_the_test <- t(as.matrix(L_the_test))       
# L_the_test <- array(as.numeric(L_the_test), dim = c(nrow(L_the_test), ncol(L_the_test), 1))

# For M1 only:
N_test <- sapply(res, function(x) x[["N"]])
N_test <- as.matrix((N_test - m_N) / std_N)


# ------------------------------------------------------------------------------
# Training:
# ------------------------------------------------------------------------------
plot(beta0, beta0_KPPM, xlab = "True beta0", ylab = "Pred beta0 (KPPM)", main = "beta0 (KPPM)")
usr <- c(4, 6, 4, 6)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(beta1, beta1_KPPM, xlab = "True beta1", ylab = "Pred beta1 (KPPM)", main = "beta1 (KPPM)")
usr <- c(0, 1, 0, 1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(var_test, var_KPPM, xlab = "True var", ylab = "Pred var (KPPM)", main = "var (KPPM)")
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, scale_KPPM, xlab = "True scale", ylab = "Pred scale (KPPM)", main = "scale (KPPM)")
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

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
# torch.cuda.manual_seed_all(2025)
source_python("NN_M1.py")
pred_test_M1 = NN_model_est_M1(L_train, Y_train, L_test, N_train, N_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_unscaled_M1 = t(t(pred_test_M1)*std_par + m_par)
# plot(var_test, Y_pred_unscaled_M1[,1])
# plot(scale_test, Y_pred_unscaled_M1[,2])
plot(var_test, Y_pred_unscaled_M1[,1], xlab = "True var", ylab = "Pred var (M1)", main = "var (M1)")
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, Y_pred_unscaled_M1[,2], xlab = "True scale", ylab = "Pred scale (M1)", main = "scale (M1)")
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
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
source_python("NN_M2.py")
pred_test_M2 = NN_model_est_M2(L_train, Y_train, L_test, epochs=20, lr=1e-2)
Y_pred_unscaled_M2 = t(t(pred_test_M2)*std_par + m_par)
# plot(var_test, Y_pred_unscaled_M2[,1])
# plot(scale_test, Y_pred_unscaled_M2[,2])
plot(var_test, Y_pred_unscaled_M2[,1], xlab = "True var", ylab = "Pred var (M2)", main = "var (M2)")
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, Y_pred_unscaled_M2[,2], xlab = "True scale", ylab = "Pred scale (M2)", main = "scale (M2)")
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
# ------------------------------------------------------------------------------
# M3:
# ------------------------------------------------------------------------------
# pred_test_M3 = NN_model_est_M2(L_the_train, Y_train, L_the_test, epochs=20, lr=1e-2)
# Y_pred_unscaled_M3 = t(t(pred_test_M3)*std_par + m_par)
# plot(var_test, Y_pred_unscaled_M3[,1])
# plot.new()
# plot(scale_test, Y_pred_unscaled_M3[,2])
# plot.new()

t1 <- Sys.time()
difftime(t1, t0, units = "mins")  


error_beta0_KPPM <- mean((beta0_KPPM - beta0)^2)
error_beta1_KPPM <- mean((beta1_KPPM - beta1)^2)
error_var_KPPM <- mean((var_KPPM - var_test)^2)
error_scale_KPPM <- mean((scale_KPPM - scale_test)^2)

error_var_M1 <- mean((Y_pred_unscaled_M1[,1] - var_test)^2)
error_scale_M1 <- mean((Y_pred_unscaled_M1[,2] - scale_test)^2)
error_var_M2 <- mean((Y_pred_unscaled_M2[,1] - var_test)^2)
error_scale_M2 <- mean((Y_pred_unscaled_M2[,2] - scale_test)^2)





