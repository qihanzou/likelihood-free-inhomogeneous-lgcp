set.seed(2025)
library(spatstat.geom)     
library(spatstat.explore)  
library(spatstat.random)   
library(spatstat.model)
library(reticulate)
library(future)
library(future.apply)
use_python("C:/Users/qihan/anaconda3/envs/py39env/python.exe", required = TRUE)
plan(multisession, workers = parallelly::availableCores() - 1)

simulate_grf = function(nx, ny, L, range, var) {
  mu0 = as.im(0, owin(xrange = c(0, L), yrange = c(0, L)), dimyx = c(ny, nx))
  sim_field = rLGCP(model = "exponential", mu = mu0, var = var, scale = range, win = as.owin(mu0), saveLambda = TRUE)
  field_im  = attr(sim_field, "Lambda")
  field_im  = eval.im(log(field_im))   
  as.matrix(field_im)
}

simulate_LGCP_point_pattern = function(params) {
  Y = simulate_grf(params$nx, params$ny, params$L, range = params$xi, var = params$alpha)
  Lambda = exp(params$gamma0 - 0.5*params$alpha + params$gamma1 * params$z + Y)
  Lambda_im = im(mat = Lambda, xrange = c(0, params$L), yrange = c(0, params$L))
  Xi = rpoispp(Lambda_im)
  list(points = cbind(Xi$x, Xi$y), Lambda = Lambda, fields = list(z = params$z, Y = Y))
}

z = simulate_grf(256, 256, 1, range = 0.05, var = 1)
z = z - mean(z) 
W     = owin(c(0,1), c(0,1))
rmax  = rmax.rule("K", W)
r     = seq(0, rmax, length.out = 513)

# ------------------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------------------
ntest = 5000
beta0 = runif(ntest, 5, 5)
beta1 = runif(ntest, 1, 1)
var_test   = runif(ntest, 0.001, 4)
scale_test = runif(ntest, 0.001, 0.1)

plan(sequential); gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

res = future_lapply(
  seq_len(ntest),
  function(j) {
    params = list(L = 1, nx = 256, ny = 256, z_scale = 0.05, xi = scale_test[j], alpha = var_test[j], gamma0 = beta0[j],  gamma1 = beta1[j], z = z)
    sim = simulate_LGCP_point_pattern(params)
    z_im = im(mat  = sim$fields$z, xcol = seq(0, params$L, length.out = params$nx), yrow = seq(0, params$L, length.out = params$ny))
    X1 = ppp(x = sim$points[,1], y = sim$points[,2], window = owin(c(0, params$L), c(0, params$L)))
    fits_kppm = kppm(X1 ~ z_im, clusters = "LGCP", method = "mincon")
    beta_KPPM = coef(fits_kppm)
    fits_clik2 = kppm(X1 ~ z_im, clusters = "LGCP", method = "clik2")
    beta_clik2 = coef(fits_clik2)
    fits_palm = kppm(X1 ~ z_im, clusters = "LGCP", method = "palm")
    beta_palm = coef(fits_palm)
    fits_adapcl = kppm(X1 ~ z_im, clusters = "LGCP", method = "adapcl")
    beta_adapcl = coef(fits_palm)
    Lfv = Linhom(X1, r = r, correction = "border")
    L_test0 = as.numeric(Lfv$border - Lfv$r)

    list(
      L_os   = L_test0,
      beta1  = as.numeric(beta_KPPM["z_im"]),
      beta0  = as.numeric(beta_KPPM["(Intercept)"]),
      var   = as.numeric(fits_kppm$par[[1]]),
      scale = as.numeric(fits_kppm$par[[2]]),
      beta1_clik2  = as.numeric(beta_clik2["z_im"]),
      beta0_clik2  = as.numeric(beta_clik2["(Intercept)"]),
      var_clik2   = as.numeric(fits_clik2$par[[1]]),
      scale_clik2 = as.numeric(fits_clik2$par[[2]]),
      beta1_palm  = as.numeric(beta_palm["z_im"]),
      beta0_palm  = as.numeric(beta_palm["(Intercept)"]),
      var_palm   = as.numeric(fits_palm$par[[1]]),
      scale_palm = as.numeric(fits_palm$par[[2]]),
      beta1_adapcl  = as.numeric(beta_adapcl["z_im"]),
      beta0_adapcl  = as.numeric(beta_adapcl["(Intercept)"]),
      var_adapcl   = as.numeric(fits_adapcl$par[[1]]),
      scale_adapcl = as.numeric(fits_adapcl$par[[2]])
    )
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore", "spatstat.model")
)

beta0_mincon  = sapply(res, function(x) x[["beta0"]])
beta1_mincon  = sapply(res, function(x) x[["beta1"]])
var_mincon    = sapply(res, function(x) x[[4]])
scale_mincon  = sapply(res, function(x) x[[5]])
beta0_clik2  = sapply(res, function(x) x[["beta0_clik2"]])
beta1_clik2  = sapply(res, function(x) x[["beta1_clik2"]])
var_clik2    = sapply(res, function(x) x[[8]])
scale_clik2  = sapply(res, function(x) x[[9]])
beta0_palm  = sapply(res, function(x) x[["beta0_palm"]])
beta1_palm  = sapply(res, function(x) x[["beta1_palm"]])
var_palm    = sapply(res, function(x) x[[12]])
scale_palm  = sapply(res, function(x) x[[13]])
beta0_adapcl  = sapply(res, function(x) x[["beta0_adapcl"]])
beta1_adapcl  = sapply(res, function(x) x[["beta1_adapcl"]])
var_adapcl    = sapply(res, function(x) x[[16]])
scale_adapcl  = sapply(res, function(x) x[[17]])
test_par = cbind(var_test, scale_test) 

# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------
ntrain = 10000
beta0_train = rnorm(ntrain, mean = mean(beta0_mincon), sd = sd(beta0_mincon)) 
beta1_train = rnorm(ntrain, mean = mean(beta1_mincon), sd = sd(beta1_mincon)) 
var_train   = runif(ntrain, 0.001, 4)
scale_train = runif(ntrain, 0.001, 0.1)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

res_train = future_lapply(
  seq_len(ntrain),
  function(j) {
    params = list(L = 1, nx = 256, ny = 256, z_scale = 0.05, xi = scale_train[j], alpha = var_train[j], gamma0 = beta0_train[j],  gamma1 = beta1_train[j], z = z)
    sim = simulate_LGCP_point_pattern(params)
    X1 = ppp(x = sim$points[,1], y = sim$points[,2], window = owin(c(0, params$L), c(0, params$L)))
    Lfv = Linhom(X1, r = r, correction = "border")
    Lvec = as.numeric(Lfv$border - Lfv$r)
    list(L = Lvec) 
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore")
)
# For train NN input
L_list = sapply(res_train, function(x) x[["L"]])
train_par = cbind(var_train, scale_train) 
# delete Na if exist
keep = colSums(is.na(L_list)) == 0  
L_list  = L_list[, keep, drop = FALSE]
train_par = train_par[keep, ]
m_L   = rowMeans(L_list, na.rm = TRUE)                 
std_L = apply(L_list, 1, sd) 
L_mat_std = sweep(L_list, 1, m_L, "-")
L_mat_std = sweep(L_mat_std, 1, std_L, "/")      
L_train = array(as.numeric(t(L_mat_std)), dim = c(ncol(L_mat_std), nrow(L_mat_std), 1))
m_par  = apply(train_par, 2, mean)
std_par = apply(train_par, 2, sd)
Y_train = scale(train_par, center = m_par, scale = std_par)
# ----------------------------------------------------------------------------------------#
# For test NN input
L_test_list = sapply(res, function(x) (x[["L_os"]]- m_L) / std_L) 
L_test_mat = t(as.matrix(L_test_list))       
L_test = array(as.numeric(L_test_mat), dim = c(nrow(L_test_mat), ncol(L_test_mat), 1))
Y_test = scale(test_par, center = m_par, scale = std_par)
# delete Na if exist
keep2 = colSums(is.na(L_test_list)) == 0
L_test_list = L_test_list[, keep2, drop = FALSE]
test_par = test_par[keep2, ]
beta0_mincon = beta0_mincon[keep2]
beta1_mincon = beta1_mincon[keep2]
var_mincon   = var_mincon[keep2]
scale_mincon = scale_mincon[keep2]
beta0_clik2  = beta0_clik2[keep2]
beta1_clik2  = beta1_clik2[keep2]
var_clik2    = var_clik2[keep2]
scale_clik2  = scale_clik2[keep2]
beta0_palm   = beta0_palm[keep2]
beta1_palm   = beta1_palm[keep2]
var_palm     = var_palm[keep2]
scale_palm   = scale_palm[keep2]
beta0_adapcl = beta0_adapcl[keep2]
beta1_adapcl = beta1_adapcl[keep2]
var_adapcl   = var_adapcl[keep2]
scale_adapcl = scale_adapcl[keep2]
beta0 = beta0[keep2]
beta1 = beta1[keep2]
var_test = var_test[keep2]
scale_test = scale_test[keep2]

# mincon:
plot(var_test, var_mincon, xlab = "True variance parameters", ylab = "Predicted variance parameters (mincon)", main = "Variance parameter (mincon)", xlim = c(0, 4), ylim = c(0, 5))
usr = c(0.001, 4, 0.001, 4)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, scale_mincon, xlab = "True scale parameters", ylab = "Predicted scale parameters (mincon)", main = "Scale parameter (mincon)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr = c(0.001, 0.1, 0.001, 0.1)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

# clik2
plot(var_test, var_clik2, xlab = "True variance parameters", ylab = "Predicted variance parameters (clik2)", main = "Variance parameter (clik2)", xlim = c(0, 4), ylim = c(0, 5))
usr = c(0.001, 4, 0.001, 4)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, scale_clik2, xlab = "True scale parameters", ylab = "Predicted scale parameters (clik2)", main = "Scale parameter (clik2)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr = c(0.001, 0.1, 0.001, 0.1)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

# palm
plot(var_test, var_palm, xlab = "True variance parameters", ylab = "Predicted variance parameters (palm)", main = "Variance parameter (palm)", xlim = c(0, 4), ylim = c(0, 5))
usr = c(0.001, 4, 0.001, 4)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, scale_palm, xlab = "True scale parameters", ylab = "Predicted scale parameters (palm)", main = "Scale parameter (palm)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr = c(0.001, 0.1, 0.001, 0.1)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

# adapcl
plot(var_test, var_adapcl, xlab = "True variance parameters", ylab = "Predicted variance parameters (adapcl)", main = "Variance parameter (adapcl)", xlim = c(0, 4), ylim = c(0, 5))
usr = c(0.001, 4, 0.001, 4)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, scale_adapcl, xlab = "True scale parameters", ylab = "Predicted scale parameters (adapcl)", main = "Scale parameter (adapcl)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr = c(0.001, 0.1, 0.001, 0.1)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 


# ------------------------------------------------------------------------------
epochs = 20
epochs_n1 = epochs-1
# ------------------------------------------------------------------------------
# DSBI
py_run_string("
import random
import numpy as np
import torch
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)")
source_python("NN_DSBI_sim.py")
pred_test_DSBI = NN_model_est_DSBI(L_train, Y_train, L_test, Y_test, batch_size=100, epochs=epochs, lr=1e-3)
Y_pred_unscaled_DSBI = t(t(pred_test_DSBI[[1]])*std_par + m_par)
m11 = min(c(pred_test_DSBI[[2]],pred_test_DSBI[[3]]))
m22 = max(c(pred_test_DSBI[[2]],pred_test_DSBI[[3]]))
plot(0:epochs_n1, pred_test_DSBI[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (DSBI)", type = "b", col = "red", xlim = c(0,epochs_n1), ylim = c(m11, m22))
points(0:epochs_n1, pred_test_DSBI[[3]], col = "blue", type = "b")
legend("topright", legend = c("Train MSE", "Validation MSE"), col = c("red", "blue"), lty = 1, pch = 1)                 

plot(var_test, Y_pred_unscaled_DSBI[,1], xlab = "True variance parameters", ylab = "Predicted variance parameters (DSBI)", main = "Variance parameter (DSBI)", xlim = c(0, 4), ylim = c(0, 5))
usr = c(0.001, 4, 0.001, 4)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, Y_pred_unscaled_DSBI[,2], xlab = "True scale parameters", ylab = "Predicted scale parameters (DSBI)", main = "Scale parameter (DSBI)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr = c(0.001, 0.1, 0.001, 0.1)
x0  = max(usr[1], usr[3])
x1  = min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
# ------------------------------------------------------------------------------
error_var_mincon = mean((var_mincon - var_test)^2)
error_scale_mincon = mean((scale_mincon - scale_test)^2)
error_var_clik2 = mean((var_clik2 - var_test)^2)
error_scale_clik2 = mean((scale_clik2 - scale_test)^2)
error_var_palm = mean((var_palm - var_test)^2)
error_scale_palm = mean((scale_palm - scale_test)^2)
error_var_adapcl = mean((var_adapcl - var_test)^2)
error_scale_adapcl = mean((scale_adapcl - scale_test)^2)
error_var_DSBI = mean((Y_pred_unscaled_DSBI[,1] - var_test)^2)
error_scale_DSBI = mean((Y_pred_unscaled_DSBI[,2] - scale_test)^2)

# check larger predictions:
sum(var_mincon>5)
sum(scale_mincon>0.15)
sum(var_clik2>5)
sum(scale_clik2>0.15)
sum(var_palm>5)
sum(scale_palm>0.15)
sum(var_adapcl>5)
sum(scale_adapcl>0.15)
sum(Y_pred_unscaled_DSBI[,1]>5)
sum(Y_pred_unscaled_DSBI[,2]>0.15)

# remove extreme values
mean((var_mincon[var_mincon<5] - var_test[var_mincon<5])^2)
mean((scale_mincon[scale_mincon<0.15] - scale_test[scale_mincon<0.15])^2)
mean((var_clik2[var_clik2<5] - var_test[var_clik2<5])^2)
mean((scale_clik2[scale_clik2<0.15] - scale_test[scale_clik2<0.15])^2)
mean((var_palm[var_palm<5] - var_test[var_palm<5])^2)
mean((scale_palm[scale_palm<0.15] - scale_test[scale_palm<0.15])^2)
mean((var_adapcl[var_adapcl<5] - var_test[var_adapcl<5])^2)
mean((scale_adapcl[scale_adapcl<0.15] - scale_test[scale_adapcl<0.15])^2)
mean((Y_pred_unscaled_DSBI[,1][Y_pred_unscaled_DSBI[,1]<5] - var_test[Y_pred_unscaled_DSBI[,1]<5])^2)
mean((Y_pred_unscaled_DSBI[,2][Y_pred_unscaled_DSBI[,2]<0.15] - scale_test[Y_pred_unscaled_DSBI[,2]<0.15])^2)


hist(beta0_mincon, main = expression(paste("Histogram of estimated ",beta[0], " KPPM")), xlab = expression(paste("Estimated ",beta[0], " KPPM")), breaks = 20)
abline(v = 5, col = "red", lwd = 2)
abline(v = mean(beta0_mincon), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c(expression(paste("True ",beta[0], " ")), expression(paste("KPPM ",E(hat(beta)[0])))), col = c("red", "blue"), lty = 1, pch = 1)                 

hist(beta1_mincon, main = expression(paste("Histogram of estimated ",beta[1], " KPPM")), xlab = expression(paste("Estimated ",beta[1], " KPPM")), breaks = 20)
abline(v = 1, col = "red", lwd = 2)
abline(v = mean(beta1_mincon), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c(expression(paste("True ",beta[1], " ")), expression(paste("KPPM ",E(hat(beta)[1])))), col = c("red", "blue"), lty = 1, pch = 1)                 

hist(beta0_train, probability = T, main = expression(paste("Histogram of simulated ",beta[0], " train")), xlab = expression(paste("Simulated ",beta[0], " train")), breaks = 20)
abline(v = 5, col = "red", lwd = 2)
abline(v = mean(beta0_mincon), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c(expression(paste("True ",beta[0], " ")), expression(paste("KPPM ",E(hat(beta)[0])))), col = c("red", "blue"), lty = 1, pch = 1)                 

hist(beta1_train,  probability = T, main = expression(paste("Histogram of simulated ",beta[1], " train")), xlab = expression(paste("Simulated ",beta[1], " train")), breaks = 20)
abline(v = 1, col = "red", lwd = 2)
abline(v = mean(beta1_mincon), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c(expression(paste("True ",beta[1], " ")), expression(paste("KPPM ",E(hat(beta)[1])))), col = c("red", "blue"), lty = 1, pch = 1)                 

