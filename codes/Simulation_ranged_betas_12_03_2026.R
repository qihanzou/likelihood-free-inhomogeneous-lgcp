set.seed(2025)
t0 <- Sys.time()
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

simulate_LGCP_point_pattern <- function(params) {
  Y = simulate_grf(params$nx, params$ny, params$L, range = params$xi, var = params$alpha)
  Lambda = exp(params$gamma0 - 0.5*params$alpha + params$gamma1 * params$z + Y)
  Lambda_im = im(mat = Lambda, xrange = c(0, params$L), yrange = c(0, params$L))
  Xi = rpoispp(Lambda_im)
  list(points = cbind(Xi$x, Xi$y), Lambda = Lambda, fields = list(z = params$z, Y = Y))
}

z = simulate_grf(256, 256, 1, range = 0.05, var = 1)
z = z - mean(z)   # center z so the intercept is for average z
W     <- owin(c(0,1), c(0,1))
rmax  <- rmax.rule("K", W)
r     <- seq(0, rmax, length.out = 513)


# ------------------------------------------------------------------------------
# TRAINing 1
# ------------------------------------------------------------------------------
ntrain1 = 10000
beta0_train1 = runif(ntrain1, 4, 6) # ranges
beta1_train1 = runif(ntrain1, 0, 1) 
var_train1   = runif(ntrain1, 0.001, 4)
scale_train1 = runif(ntrain1, 0.001, 0.1)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

library(progressr)
handlers(global = TRUE)           
handlers("txtprogressbar") 

with_progress({
p <- progressor(along = seq_len(ntrain1))
res_train1 = future_lapply(
  seq_len(ntrain1),
  function(j) {
    p(sprintf("iteration %d/%d", j, ntrain1))
    params_train1 = list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                   xi = scale_train1[j], alpha = var_train1[j],
                   gamma0 = beta0_train1[j],  gamma1 = beta1_train1[j], z = z)
    sim_train1 = simulate_LGCP_point_pattern(params_train1)
    X_train1 = ppp(x = sim_train1$points[,1], y = sim_train1$points[,2], window = owin(c(0, params_train1$L), c(0, params_train1$L)))
    z_im = im(mat  = sim_train1$fields$z, xcol = seq(0, params_train1$L, length.out = params_train1$nx), yrow = seq(0, params_train1$L, length.out = params_train1$ny))
    
    fits_train1 = kppm(X_train1 ~ z_im, clusters = "LGCP", method = "mincon")
    beta_hat_train1 = coef(fits_train1)
    Lambda_hat_train1 = exp(beta_hat_train1[1] + beta_hat_train1[2]*z_im)
    Lambda_true_train1 = exp(params_train1$gamma0 + params_train1$gamma1*z_im)
    
    Lfv1 = Linhom(X_train1, lambda = Lambda_true_train1, r = r, correction = "border") # true beta
    Lfv2 = Linhom(X_train1, lambda = Lambda_hat_train1, r = r, correction = "border") # plug-in estimated beta
    Lfv3 = Linhom(X_train1, r = r, correction = "border") # kernel smoother
    
    Lvec1 = as.numeric(Lfv1$border - Lfv1$r)
    Lvec2 = as.numeric(Lfv2$border - Lfv2$r)
    Lvec3 = as.numeric(Lfv3$border - Lfv3$r)
    
    list(L1 = Lvec1, 
         L2 = Lvec2,
         L3 = Lvec3) 
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore")
)
})
L_list1 <- sapply(res_train1, function(x) x[["L1"]])
L_list2 <- sapply(res_train1, function(x) x[["L2"]])
L_list3 <- sapply(res_train1, function(x) x[["L3"]])
train_par1 <- cbind(var_train1, scale_train1) 


# delete Na
keep <- (colSums(is.na(L_list1)) == 0) &
  (colSums(is.na(L_list2)) == 0) &
  (colSums(is.na(L_list3)) == 0)

# filter all L_lists (columns)
L_list1 <- L_list1[, keep, drop = FALSE]
L_list2 <- L_list2[, keep, drop = FALSE]
L_list3 <- L_list3[, keep, drop = FALSE]

# filter train_par (rows)
train_par1 <- train_par1[keep, , drop = FALSE]
m_par  <- apply(train_par1, 2, mean)
std_par <- apply(train_par1, 2, sd)
Y_train <- scale(train_par1, center = m_par, scale = std_par)


###
m_L1   <- rowMeans(L_list1, na.rm = TRUE)                 
std_L1 <- apply(L_list1, 1, sd) 
L_mat_std1 <- sweep(L_list1, 1, m_L1, "-")
L_mat_std1 <- sweep(L_mat_std1, 1, std_L1, "/")      
L_train1 <- array(as.numeric(t(L_mat_std1)), dim = c(ncol(L_mat_std1), nrow(L_mat_std1), 1))

m_L2   <- rowMeans(L_list2, na.rm = TRUE)                 
std_L2 <- apply(L_list2, 1, sd) 
L_mat_std2 <- sweep(L_list2, 1, m_L2, "-")
L_mat_std2 <- sweep(L_mat_std2, 1, std_L2, "/")      
L_train2 <- array(as.numeric(t(L_mat_std2)), dim = c(ncol(L_mat_std2), nrow(L_mat_std2), 1))

m_L3   <- rowMeans(L_list3, na.rm = TRUE)                 
std_L3 <- apply(L_list3, 1, sd) 
L_mat_std3 <- sweep(L_list3, 1, m_L3, "-")
L_mat_std3 <- sweep(L_mat_std3, 1, std_L3, "/")      
L_train3 <- array(as.numeric(t(L_mat_std3)), dim = c(ncol(L_mat_std3), nrow(L_mat_std3), 1))

# ------------------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------------------
ntest = 5000
beta0_test = runif(ntest, 4, 6)
beta1_test = runif(ntest, 0, 1)
var_test   <- runif(ntest, 0.001, 4)
scale_test <- runif(ntest, 0.001, 0.1)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

with_progress({
p2 <- progressor(along = seq_len(ntest))
res <- future_lapply(
  seq_len(ntest),
  function(j) {
    p2(sprintf("iteration %d/%d", j, ntest))
    
    params <- list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                   xi = scale_test[j], alpha = var_test[j],
                   gamma0 = beta0_test[j],  gamma1 = beta1_test[j], z = z)
    sim <- simulate_LGCP_point_pattern(params)
    z_im <- im(mat  = sim$fields$z, xcol = seq(0, params$L, length.out = params$nx), yrow = seq(0, params$L, length.out = params$ny))
    X1 <- ppp(x = sim$points[,1], y = sim$points[,2], window = owin(c(0, params$L), c(0, params$L)))
    
    fits_kppm <- kppm(X1 ~ z_im, clusters = "LGCP", method = "mincon")
    beta_KPPM <- coef(fits_kppm)
    fits_clik2 <- kppm(X1 ~ z_im, clusters = "LGCP", method = "clik2")
    beta_clik2 <- coef(fits_clik2)
    fits_palm <- kppm(X1 ~ z_im, clusters = "LGCP", method = "palm")
    beta_palm <- coef(fits_palm)
    fits_adapcl <- kppm(X1 ~ z_im, clusters = "LGCP", method = "adapcl")
    beta_adapcl <- coef(fits_palm)
    
    Lambda_hat_test = exp(beta_KPPM[1] + beta_KPPM[2]*z_im)
    Lambda_true_test = exp(params$gamma0 + params$gamma1*z_im)


    Lfv_test1 <- Linhom(X1, lambda = Lambda_true_test, r = r, correction = "border")
    Lfv_test2 <- Linhom(X1, lambda = Lambda_hat_test, r = r, correction = "border")
    Lfv_test3 <- Linhom(X1, r = r, correction = "border")
    
    L_test1 <- as.numeric(Lfv_test1$border - Lfv_test1$r)
    L_test2 <- as.numeric(Lfv_test1$border - Lfv_test1$r)
    L_test3 <- as.numeric(Lfv_test1$border - Lfv_test1$r)
    
    L_test1_train1 <- (L_test1 - m_L1) / std_L1
    L_test2_train1 <- (L_test2 - m_L1) / std_L1
    L_test3_train1 <- (L_test3 - m_L1) / std_L1
    
    L_test1_train2 <- (L_test1 - m_L2) / std_L2
    L_test2_train2 <- (L_test2 - m_L2) / std_L2
    L_test3_train2 <- (L_test3 - m_L2) / std_L2
    
    L_test1_train3 <- (L_test1 - m_L3) / std_L3
    L_test2_train3 <- (L_test2 - m_L3) / std_L3
    L_test3_train3 <- (L_test3 - m_L3) / std_L3
    
    list(
      L_test1_train1 = L_test1_train1,
      L_test2_train1 = L_test2_train1,
      L_test3_train1 = L_test3_train1,
      
      L_test1_train2 = L_test1_train2,
      L_test2_train2 = L_test2_train2,
      L_test3_train2 = L_test3_train2,
      
      L_test1_train3 = L_test1_train3,
      L_test2_train3 = L_test2_train3,
      L_test3_train3 = L_test3_train3,

      beta1  = as.numeric(beta_KPPM["z_im"]),
      beta0  = as.numeric(beta_KPPM["(Intercept)"]),
      var   <- as.numeric(fits_kppm$par[[1]]),
      scale <- as.numeric(fits_kppm$par[[2]]),
      
      beta1_clik2  = as.numeric(beta_clik2["z_im"]),
      beta0_clik2  = as.numeric(beta_clik2["(Intercept)"]),
      var_clik2   <- as.numeric(fits_clik2$par[[1]]),
      scale_clik2 <- as.numeric(fits_clik2$par[[2]]),
      
      beta1_palm  = as.numeric(beta_palm["z_im"]),
      beta0_palm  = as.numeric(beta_palm["(Intercept)"]),
      var_palm   <- as.numeric(fits_palm$par[[1]]),
      scale_palm <- as.numeric(fits_palm$par[[2]]),
      
      beta1_adapcl  = as.numeric(beta_adapcl["z_im"]),
      beta0_adapcl  = as.numeric(beta_adapcl["(Intercept)"]),
      var_adapcl   <- as.numeric(fits_adapcl$par[[1]]),
      scale_adapcl <- as.numeric(fits_adapcl$par[[2]]),
      
      N = length(sim$points[,1])
    )
  },
  future.seed = 2025,
  future.packages = c("spatstat.geom", "spatstat.explore", "spatstat.model")
)
})

L_test1_train1_list <- sapply(res, function(x) x[["L_test1_train1"]]) 
L_test2_train1_list <- sapply(res, function(x) x[["L_test2_train1"]]) 
L_test3_train1_list <- sapply(res, function(x) x[["L_test3_train1"]]) 

L_test1_train2_list <- sapply(res, function(x) x[["L_test1_train2"]]) 
L_test2_train2_list <- sapply(res, function(x) x[["L_test2_train2"]]) 
L_test3_train2_list <- sapply(res, function(x) x[["L_test3_train2"]])

L_test1_train3_list <- sapply(res, function(x) x[["L_test1_train3"]]) 
L_test2_train3_list <- sapply(res, function(x) x[["L_test2_train3"]]) 
L_test3_train3_list <- sapply(res, function(x) x[["L_test3_train3"]]) 

beta0_mincon  <- sapply(res, function(x) x[["beta0"]])
beta1_mincon  <- sapply(res, function(x) x[["beta1"]])
var_mincon    <- sapply(res, function(x) x[[12]])
scale_mincon  <- sapply(res, function(x) x[[13]])

beta0_clik2  <- sapply(res, function(x) x[["beta0_clik2"]])
beta1_clik2  <- sapply(res, function(x) x[["beta1_clik2"]])
var_clik2    <- sapply(res, function(x) x[[16]])
scale_clik2  <- sapply(res, function(x) x[[17]])

beta0_palm  <- sapply(res, function(x) x[["beta0_palm"]])
beta1_palm  <- sapply(res, function(x) x[["beta1_palm"]])
var_palm    <- sapply(res, function(x) x[[20]])
scale_palm  <- sapply(res, function(x) x[[21]])

beta0_adapcl  <- sapply(res, function(x) x[["beta0_adapcl"]])
beta1_adapcl  <- sapply(res, function(x) x[["beta1_adapcl"]])
var_adapcl    <- sapply(res, function(x) x[[24]])
scale_adapcl  <- sapply(res, function(x) x[[25]])

test_par <- cbind(var_test, scale_test) 


# delet Na

keep2 <- (colSums(is.na(L_test1_train1_list)) == 0) &
         (colSums(is.na(L_test2_train1_list)) == 0) &
         (colSums(is.na(L_test3_train1_list)) == 0) &
         (colSums(is.na(L_test1_train2_list)) == 0) &
         (colSums(is.na(L_test2_train2_list)) == 0) &
         (colSums(is.na(L_test3_train2_list)) == 0) &
         (colSums(is.na(L_test1_train3_list)) == 0) &
         (colSums(is.na(L_test2_train3_list)) == 0) &
         (colSums(is.na(L_test3_train3_list)) == 0)

# filter all L_lists (columns)
L_test1_train1_list <- L_test1_train1_list[, keep2, drop = FALSE]
L_test2_train1_list <- L_test2_train1_list[, keep2, drop = FALSE]
L_test3_train1_list <- L_test3_train1_list[, keep2, drop = FALSE]
L_test1_train2_list <- L_test1_train2_list[, keep2, drop = FALSE]
L_test2_train2_list <- L_test2_train2_list[, keep2, drop = FALSE]
L_test3_train2_list <- L_test3_train2_list[, keep2, drop = FALSE]
L_test1_train3_list <- L_test1_train3_list[, keep2, drop = FALSE]
L_test2_train3_list <- L_test2_train3_list[, keep2, drop = FALSE]
L_test3_train3_list <- L_test3_train3_list[, keep2, drop = FALSE]

beta0_mincon <- beta0_mincon[keep2]
beta1_mincon <- beta1_mincon[keep2]
var_mincon   <- var_mincon[keep2]
scale_mincon <- scale_mincon[keep2]

beta0_clik2  <- beta0_clik2[keep2]
beta1_clik2  <- beta1_clik2[keep2]
var_clik2    <- var_clik2[keep2]
scale_clik2  <- scale_clik2[keep2]

beta0_palm   <- beta0_palm[keep2]
beta1_palm   <- beta1_palm[keep2]
var_palm     <- var_palm[keep2]
scale_palm   <- scale_palm[keep2]

beta0_adapcl <- beta0_adapcl[keep2]
beta1_adapcl <- beta1_adapcl[keep2]
var_adapcl   <- var_adapcl[keep2]
scale_adapcl <- scale_adapcl[keep2]

var_test = var_test[keep2]
scale_test = scale_test[keep2]


test_par <- test_par[keep2, ]
Y_test <- scale(test_par, center = m_par, scale = std_par)



L_test1_train1 <- array(as.numeric(t(as.matrix(L_test1_train1_list))), dim = c(nrow(t(as.matrix(L_test1_train1_list))), ncol(t(as.matrix(L_test1_train1_list))), 1))
L_test2_train1 <- array(as.numeric(t(as.matrix(L_test2_train1_list))), dim = c(nrow(t(as.matrix(L_test2_train1_list))), ncol(t(as.matrix(L_test2_train1_list))), 1))
L_test3_train1 <- array(as.numeric(t(as.matrix(L_test3_train1_list))), dim = c(nrow(t(as.matrix(L_test3_train1_list))), ncol(t(as.matrix(L_test3_train1_list))), 1))

L_test1_train2 <- array(as.numeric(t(as.matrix(L_test1_train2_list))), dim = c(nrow(t(as.matrix(L_test1_train2_list))), ncol(t(as.matrix(L_test1_train2_list))), 1))
L_test2_train2 <- array(as.numeric(t(as.matrix(L_test2_train2_list))), dim = c(nrow(t(as.matrix(L_test2_train2_list))), ncol(t(as.matrix(L_test2_train2_list))), 1))
L_test3_train2 <- array(as.numeric(t(as.matrix(L_test3_train2_list))), dim = c(nrow(t(as.matrix(L_test3_train2_list))), ncol(t(as.matrix(L_test3_train2_list))), 1))

L_test1_train3 <- array(as.numeric(t(as.matrix(L_test1_train3_list))), dim = c(nrow(t(as.matrix(L_test1_train3_list))), ncol(t(as.matrix(L_test1_train3_list))), 1))
L_test2_train3 <- array(as.numeric(t(as.matrix(L_test2_train3_list))), dim = c(nrow(t(as.matrix(L_test2_train3_list))), ncol(t(as.matrix(L_test2_train3_list))), 1))
L_test3_train3 <- array(as.numeric(t(as.matrix(L_test3_train3_list))), dim = c(nrow(t(as.matrix(L_test3_train3_list))), ncol(t(as.matrix(L_test3_train3_list))), 1))





# ------------------------------------------------------------------------------
epochs_n1 = 20-1
py_run_string("
import random
import numpy as np
import torch
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)")
source_python("NN_M2_v3.py")


# train 1
pred_test1_train1 = NN_model_est_M2(L_train1, Y_train, L_test1_train1, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test1_train1 = t(t(pred_test1_train1[[1]])*std_par + m_par)

pred_test2_train1 = NN_model_est_M2(L_train1, Y_train, L_test2_train1, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test2_train1 = t(t(pred_test2_train1[[1]])*std_par + m_par)

pred_test3_train1 = NN_model_est_M2(L_train1, Y_train, L_test3_train1, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test3_train1 = t(t(pred_test3_train1[[1]])*std_par + m_par)



# train 2
pred_test1_train2 = NN_model_est_M2(L_train2, Y_train, L_test1_train2, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test1_train2 = t(t(pred_test1_train2[[1]])*std_par + m_par)

pred_test2_train2 = NN_model_est_M2(L_train2, Y_train, L_test2_train2, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test2_train2 = t(t(pred_test2_train2[[1]])*std_par + m_par)

pred_test3_train2 = NN_model_est_M2(L_train2, Y_train, L_test3_train2, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test3_train2 = t(t(pred_test3_train2[[1]])*std_par + m_par)



# train 3
pred_test1_train3 = NN_model_est_M2(L_train3, Y_train, L_test1_train3, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test1_train3 = t(t(pred_test1_train3[[1]])*std_par + m_par)

pred_test2_train3 = NN_model_est_M2(L_train3, Y_train, L_test2_train3, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test2_train3 = t(t(pred_test2_train3[[1]])*std_par + m_par)

pred_test3_train3 = NN_model_est_M2(L_train3, Y_train, L_test3_train3, Y_test, batch_size=100, epochs=20, lr=1e-3)
Y_pred_test3_train3 = t(t(pred_test3_train3[[1]])*std_par + m_par)





# m11 = min(c(pred_test_M2[[2]],pred_test_M2[[3]]))
# m22 = max(c(pred_test_M2[[2]],pred_test_M2[[3]]))
# plot(0:epochs_n1, pred_test_M2[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (M2)", type = "b", col = "red", xlim = c(0,epochs_n1), ylim = c(m11, m22))
# points(0:epochs_n1, pred_test_M2[[3]], col = "blue", type = "b")
# legend("topright", legend = c("Train MSE", "Validation MSE"), col = c("red", "blue"), lty = 1, pch = 1)                 

plot(var_test, Y_pred_unscaled_M2[,1], xlab = "True var", ylab = "Pred var (M2)", main = "var (M2)", xlim = c(0, 4), ylim = c(0, 5))
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 

plot(scale_test, Y_pred_unscaled_M2[,2], xlab = "True scale", ylab = "Pred scale (M2)", main = "scale (M2)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 





# ------------------------------------------------------------------------------


t1 <- Sys.time()
difftime(t1, t0, units = "mins") 

sum(keep2)
sum(keep)
mean(z) # = 1.310178e-17


# check larger predictions:
sum(var_mincon>5)
sum(scale_mincon>0.15)

sum(var_clik2>5)
sum(scale_clik2>0.15)

sum(var_palm>5)
sum(scale_palm>0.15)

sum(var_adapcl>5)
sum(scale_adapcl>0.15)

sum(Y_pred_test1_train1[,1]>5)
sum(Y_pred_test1_train1[,2]>0.15)

sum(Y_pred_test2_train1[,1]>5)
sum(Y_pred_test2_train1[,2]>0.15)

sum(Y_pred_test3_train1[,1]>5)
sum(Y_pred_test3_train1[,2]>0.15)


sum(Y_pred_test1_train2[,1]>5)
sum(Y_pred_test1_train2[,2]>0.15)

sum(Y_pred_test2_train2[,1]>5)
sum(Y_pred_test2_train2[,2]>0.15)

sum(Y_pred_test3_train2[,1]>5)
sum(Y_pred_test3_train2[,2]>0.15)


sum(Y_pred_test1_train3[,1]>5)
sum(Y_pred_test1_train3[,2]>0.15)

sum(Y_pred_test2_train3[,1]>5)
sum(Y_pred_test2_train3[,2]>0.15)

sum(Y_pred_test3_train3[,1]>5)
sum(Y_pred_test3_train3[,2]>0.15)


# remove larger
mean((var_mincon[var_mincon<5] - var_test[var_mincon<5])^2)
mean((scale_mincon[scale_mincon<0.15] - scale_test[scale_mincon<0.15])^2)

mean((var_palm[var_palm<5] - var_test[var_palm<5])^2)
mean((scale_palm[scale_palm<0.15] - scale_test[scale_palm<0.15])^2)

mean((var_clik2[var_clik2<5] - var_test[var_clik2<5])^2)
mean((scale_clik2[scale_clik2<0.15] - scale_test[scale_clik2<0.15])^2)

mean((var_adapcl[var_adapcl<5] - var_test[var_adapcl<5])^2)
mean((scale_adapcl[scale_adapcl<0.15] - scale_test[scale_adapcl<0.15])^2)


mean((Y_pred_test1_train1[,1][Y_pred_test1_train1[,1]<5]      - var_test[Y_pred_test1_train1[,1]<5])^2)
mean((Y_pred_test1_train1[,2][Y_pred_test1_train1[,2]<0.15] - scale_test[Y_pred_test1_train1[,2]<0.15])^2)

mean((Y_pred_test2_train1[,1][Y_pred_test2_train1[,1]<5]      - var_test[Y_pred_test2_train1[,1]<5])^2)
mean((Y_pred_test2_train1[,2][Y_pred_test2_train1[,2]<0.15] - scale_test[Y_pred_test2_train1[,2]<0.15])^2)

mean((Y_pred_test3_train1[,1][Y_pred_test3_train1[,1]<5]      - var_test[Y_pred_test3_train1[,1]<5])^2)
mean((Y_pred_test3_train1[,2][Y_pred_test3_train1[,2]<0.15] - scale_test[Y_pred_test3_train1[,2]<0.15])^2)


mean((Y_pred_test1_train2[,1][Y_pred_test1_train2[,1]<5]      - var_test[Y_pred_test1_train2[,1]<5])^2)
mean((Y_pred_test1_train2[,2][Y_pred_test1_train2[,2]<0.15] - scale_test[Y_pred_test1_train2[,2]<0.15])^2)

mean((Y_pred_test2_train2[,1][Y_pred_test2_train2[,1]<5]      - var_test[Y_pred_test2_train2[,1]<5])^2)
mean((Y_pred_test2_train2[,2][Y_pred_test2_train2[,2]<0.15] - scale_test[Y_pred_test2_train2[,2]<0.15])^2)

mean((Y_pred_test3_train2[,1][Y_pred_test3_train2[,1]<5]      - var_test[Y_pred_test3_train2[,1]<5])^2)
mean((Y_pred_test3_train2[,2][Y_pred_test3_train2[,2]<0.15] - scale_test[Y_pred_test3_train2[,2]<0.15])^2)


mean((Y_pred_test1_train3[,1][Y_pred_test1_train3[,1]<5]      - var_test[Y_pred_test1_train3[,1]<5])^2)
mean((Y_pred_test1_train3[,2][Y_pred_test1_train3[,2]<0.15] - scale_test[Y_pred_test1_train3[,2]<0.15])^2)

mean((Y_pred_test2_train3[,1][Y_pred_test2_train3[,1]<5]      - var_test[Y_pred_test2_train3[,1]<5])^2)
mean((Y_pred_test2_train3[,2][Y_pred_test2_train3[,2]<0.15] - scale_test[Y_pred_test2_train3[,2]<0.15])^2)

mean((Y_pred_test3_train3[,1][Y_pred_test3_train3[,1]<5]      - var_test[Y_pred_test3_train3[,1]<5])^2)
mean((Y_pred_test3_train3[,2][Y_pred_test3_train3[,2]<0.15] - scale_test[Y_pred_test3_train3[,2]<0.15])^2)








# ------------------------------------------------------------------------------
# Training:
# ------------------------------------------------------------------------------

plot.res = 600
png("data_ranges_betas_mincon_beta0.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(beta0, beta0_mincon, xlab = "True beta0", ylab = "Pred beta0", main = "beta0 (kppm)", xlim = c(4, 6), ylim = c(3, 7))
usr <- c(4, 6, 4, 6)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()

plot.res = 600
png("data_ranges_betas_mincon_beta1.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(beta1, beta1_mincon, xlab = "True beta1", ylab = "Pred beta1", main = "beta1 (kppm)", xlim = c(0, 1), ylim = c(-0.5, 1.5))
usr <- c(0, 1, 0, 1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()





# mincon:
plot.res = 600
png("data_ranges_betas_mincon_var.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(var_test, var_mincon, xlab = "True variance parameters", ylab = "Predicted variance parameters (mincon)", main = "Variance parameter (mincon)", xlim = c(0, 4), ylim = c(0, 5))
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()

plot.res = 600
png("data_ranges_betas_mincon_scale.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(scale_test, scale_mincon, xlab = "True scale parameters", ylab = "Predicted scale parameters (mincon)", main = "Scale parameter (mincon)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()
# ------------------------------------------------------------------------------
# clik2
plot.res = 600
png("data_ranges_betas_clik2_var.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(var_test, var_clik2, xlab = "True variance parameters", ylab = "Predicted variance parameters (clik2)", main = "Variance parameter (clik2)", xlim = c(0, 4), ylim = c(0, 5))
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()

plot.res = 600
png("data_ranges_betas_clik2_scale.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(scale_test, scale_clik2, xlab = "True scale parameters", ylab = "Predicted scale parameters (clik2)", main = "Scale parameter (clik2)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()
# ------------------------------------------------------------------------------
# palm
plot.res = 600
png("data_ranges_betas_palm_var.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(var_test, var_palm, xlab = "True variance parameters", ylab = "Predicted variance parameters (palm)", main = "Variance parameter (palm)", xlim = c(0, 4), ylim = c(0, 5))
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()

plot.res = 600
png("data_ranges_betas_palm_scale.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(scale_test, scale_palm, xlab = "True scale parameters", ylab = "Predicted scale parameters (palm)", main = "Scale parameter (palm)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()
# ------------------------------------------------------------------------------
# adapcl
plot.res = 600
png("data_ranges_betas_adapcl_var.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(var_test, var_adapcl, xlab = "True variance parameters", ylab = "Predicted variance parameters (adapcl)", main = "Variance parameter (adapcl)", xlim = c(0, 4), ylim = c(0, 5))
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()

plot.res = 600
png("data_ranges_betas_adapcl_scale.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(scale_test, scale_adapcl, xlab = "True scale parameters", ylab = "Predicted scale parameters (adapcl)", main = "Scale parameter (adapcl)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()




plot.res = 600
png("data_ranges_betas_DSBI_loss.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
m11 = min(c(pred_test_M2[[2]],pred_test_M2[[3]]))
m22 = max(c(pred_test_M2[[2]],pred_test_M2[[3]]))
plot(0:epochs_n1, pred_test_M2[[2]], xlab = "epochs", ylab = "MSE", main = "Train and Validation MSE (DSBI)", type = "b", col = "red", xlim = c(0,epochs_n1+1), ylim = c(m11, m22))
points(0:epochs_n1, pred_test_M2[[3]], col = "blue", type = "b")
legend("topright", legend = c("Train MSE", "Validation MSE"), col = c("red", "blue"), lty = 1, pch = 1)                 
dev.off()


plot.res = 600
png("data_ranges_betas_M2_var.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(var_test, Y_pred_unscaled_M2[,1], xlab = "True variance parameters", ylab = "Predicted variance parameters (DSBI)", main = "Variance parameter (DSBI)", xlim = c(0, 4), ylim = c(0, 5))
usr <- c(0.001, 4, 0.001, 4)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()

plot.res = 600
png("data_ranges_betas_M2_scale.png", width = 6.2 * plot.res, height = 4.5 * plot.res, res = 600)
plot(scale_test, Y_pred_unscaled_M2[,2], xlab = "True scale parameters", ylab = "Predicted scale parameters (DSBI)", main = "Scale parameter (DSBI)", xlim = c(0, 0.1), ylim = c(0, 0.15))
usr <- c(0.001, 0.1, 0.001, 0.1)
x0  <- max(usr[1], usr[3])
x1  <- min(usr[2], usr[4])
segments(x0, x0, x1, x1, lwd = 2, col = "red") 
dev.off()