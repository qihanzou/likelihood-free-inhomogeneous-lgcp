set.seed(2025)
t0 = Sys.time()
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

simulate_LGCP_point_pattern_1cov = function(params) {
  Y = simulate_grf(params$nx, params$ny, params$L, range = params$xi, var = params$alpha)
  Lambda = exp(params$gamma0 - 0.5*params$alpha + params$gamma1*params$z1 + Y)
  Lambda_im = im(mat = Lambda, xrange = c(0, params$L), yrange = c(0, params$L))
  Xi = rpoispp(Lambda_im)
  list(points = cbind(Xi$x, Xi$y), Lambda = Lambda, fields = list(z1 = params$z1, z2 = params$z2, z3 = params$z3, Y = Y))
}

simulate_LGCP_point_pattern_2cov = function(params) {
  Y = simulate_grf(params$nx, params$ny, params$L, range = params$xi, var = params$alpha)
  Lambda = exp(params$gamma0 - 0.5*params$alpha + params$gamma1*params$z1 + params$gamma2*params$z2 + Y)
  Lambda_im = im(mat = Lambda, xrange = c(0, params$L), yrange = c(0, params$L))
  Xi = rpoispp(Lambda_im)
  list(points = cbind(Xi$x, Xi$y), Lambda = Lambda, fields = list(z1 = params$z1, z2 = params$z2, z3 = params$z3, Y = Y))
}

simulate_LGCP_point_pattern_3cov = function(params) {
  Y = simulate_grf(params$nx, params$ny, params$L, range = params$xi, var = params$alpha)
  Lambda = exp(params$gamma0 - 0.5*params$alpha + params$gamma1*params$z1 + params$gamma2*params$z2 + params$gamma3*params$z3 + Y)
  Lambda_im = im(mat = Lambda, xrange = c(0, params$L), yrange = c(0, params$L))
  Xi = rpoispp(Lambda_im)
  list(points = cbind(Xi$x, Xi$y), Lambda = Lambda, fields = list(z1 = params$z1, z2 = params$z2, z3 = params$z3, Y = Y))
}


set.seed(123)
z1 = simulate_grf(256, 256, 1, range = 0.05, var = 1)
set.seed(456)
z2 = simulate_grf(256, 256, 1, range = 0.05, var = 1)
set.seed(789)
z3 = simulate_grf(256, 256, 1, range = 0.05, var = 1)
z1 = z1 - mean(z1)
z2 = z2 - mean(z2)   
z3 = z3 - mean(z3)

set.seed(2025)
W     = owin(c(0,1), c(0,1))
rmax  = rmax.rule("K", W)
r     = seq(0, rmax, length.out = 513)


# ------------------------------------------------------------------------------
# training 
# ------------------------------------------------------------------------------
ntrain = 10000
beta0_train = runif(ntrain,  4, 6) 
beta1_train = runif(ntrain,  -1, 1) 
beta2_train = runif(ntrain,  -1, 1) 
beta3_train = runif(ntrain,  -1, 1)

var_train   = runif(ntrain, 0.001, 4)
scale_train = runif(ntrain, 0.001, 0.1)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

library(progressr)
handlers(global = TRUE)           
handlers("txtprogressbar") 

with_progress({
  p = progressor(along = seq_len(ntrain))
  res_train = future_lapply(
    seq_len(ntrain),
    function(j) {
      p(sprintf("iteration %d/%d", j, ntrain))

      params_train_true = list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                           xi = scale_train[j], alpha = var_train[j],
                           gamma0 = beta0_train[j],  gamma1 = beta1_train[j], gamma2 = beta2_train[j],
                           z1 = z1, z2 = z2, z3 = z3)
      
      params_train_1more = list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                          xi = scale_train[j], alpha = var_train[j],
                          gamma0 = beta0_train[j],  gamma1 = beta1_train[j], gamma2 = beta2_train[j], gamma3 = beta3_train[j], 
                          z1 = z1, z2 = z2, z3 = z3)
      
      params_train_1less = list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                          xi = scale_train[j], alpha = var_train[j],
                          gamma0 = beta0_train[j],  gamma1 = beta1_train[j],
                          z1 = z1, z2 = z2, z3 = z3)
      
      sim_train_true = simulate_LGCP_point_pattern_2cov(params_train_true)
      sim_train_1more = simulate_LGCP_point_pattern_3cov(params_train_1more)
      sim_train_1less = simulate_LGCP_point_pattern_1cov(params_train_1less)
      
      X_train_true  = ppp(x = sim_train_true$points[,1], y = sim_train_true$points[,2], window = owin(c(0, 1), c(0, 1)))
      X_train_1more = ppp(x = sim_train_1more$points[,1], y = sim_train_1more$points[,2], window = owin(c(0, 1), c(0, 1)))
      X_train_1less = ppp(x = sim_train_1less$points[,1], y = sim_train_1less$points[,2], window = owin(c(0, 1), c(0, 1)))
      

      z_im1 = im(mat= sim_train_true$fields$z1, xcol = seq(0, 1, length.out = 256), yrow = seq(0, 1, length.out = 256))
      z_im2 = im(mat= sim_train_true$fields$z2, xcol = seq(0, 1, length.out = 256), yrow = seq(0, 1, length.out = 256))
      z_im3 = im(mat= sim_train_true$fields$z3, xcol = seq(0, 1, length.out = 256), yrow = seq(0, 1, length.out = 256))
      
      Lambda_true_train_true = exp(params_train_true$gamma0 + params_train_true$gamma1*z_im1 + params_train_true$gamma2*z_im2)
      Lambda_true_train_1more = exp(params_train_1more$gamma0 + params_train_1more$gamma1*z_im1 + params_train_1more$gamma2*z_im2 + params_train_1more$gamma3*z_im3)
      Lambda_true_train_1less = exp(params_train_1less$gamma0 + params_train_1less$gamma1*z_im1)
      
      
      Lfv_true = Linhom(X_train_true, lambda = Lambda_true_train_true, r = r, correction = "border") 
      Lfv_1more = Linhom(X_train_1more, lambda = Lambda_true_train_1more, r = r, correction = "border")
      Lfv_1less = Linhom(X_train_1less, lambda = Lambda_true_train_1less, r = r, correction = "border")
      
      
      Lvec_true = as.numeric(Lfv_true$border - Lfv_true$r)
      Lvec_1more = as.numeric(Lfv_1more$border - Lfv_1more$r)
      Lvec_1less = as.numeric(Lfv_1less$border - Lfv_1less$r)
      
      list(L_true = Lvec_true, 
           L_1more = Lvec_1more, 
           L_1less = Lvec_1less) 
    },
    future.seed = 2025,
    future.packages = c("spatstat.geom", "spatstat.explore")
  )
})
train_par = cbind(var_train, scale_train) 

L_list_true = sapply(res_train, function(x) x[["L_true"]])
keep_true = (colSums(is.na(L_list_true)) == 0) 
L_list_true = L_list_true[, keep_true, drop = FALSE]
train_par_true = train_par[keep_true, , drop = FALSE]
m_par_true  = apply(train_par_true, 2, mean)
std_par_true = apply(train_par_true, 2, sd)
Y_train_true = scale(train_par_true, center = m_par_true, scale = std_par_true)
m_L_true   = rowMeans(L_list_true, na.rm = TRUE)                 
std_L_true = apply(L_list_true, 1, sd) 
L_mat_std_true = sweep(L_list_true, 1, m_L_true, "-")
L_mat_std_true = sweep(L_mat_std_true, 1, std_L_true, "/")      
L_train_true = array(as.numeric(t(L_mat_std_true)), dim = c(ncol(L_mat_std_true), nrow(L_mat_std_true), 1))

L_list_1more = sapply(res_train, function(x) x[["L_1more"]])
keep_1more = (colSums(is.na(L_list_1more)) == 0) 
L_list_1more = L_list_1more[, keep_1more, drop = FALSE]
train_par_1more = train_par[keep_1more, , drop = FALSE]
m_par_1more  = apply(train_par_1more, 2, mean)
std_par_1more = apply(train_par_1more, 2, sd)
Y_train_1more = scale(train_par_1more, center = m_par_1more, scale = std_par_1more)
m_L_1more   = rowMeans(L_list_1more, na.rm = TRUE)                 
std_L_1more = apply(L_list_1more, 1, sd) 
L_mat_std_1more = sweep(L_list_1more, 1, m_L_1more, "-")
L_mat_std_1more = sweep(L_mat_std_1more, 1, std_L_1more, "/")      
L_train_1more = array(as.numeric(t(L_mat_std_1more)), dim = c(ncol(L_mat_std_1more), nrow(L_mat_std_1more), 1))

L_list_1less = sapply(res_train, function(x) x[["L_1less"]])
keep_1less = (colSums(is.na(L_list_1less)) == 0) 
L_list_1less = L_list_1less[, keep_1less, drop = FALSE]
train_par_1less = train_par[keep_1less, , drop = FALSE]
m_par_1less  = apply(train_par_1less, 2, mean)
std_par_1less = apply(train_par_1less, 2, sd)
Y_train_1less = scale(train_par_1less, center = m_par_1less, scale = std_par_1less)
m_L_1less   = rowMeans(L_list_1less, na.rm = TRUE)                 
std_L_1less = apply(L_list_1less, 1, sd) 
L_mat_std_1less = sweep(L_list_1less, 1, m_L_1less, "-")
L_mat_std_1less = sweep(L_mat_std_1less, 1, std_L_1less, "/")      
L_train_1less = array(as.numeric(t(L_mat_std_1less)), dim = c(ncol(L_mat_std_1less), nrow(L_mat_std_1less), 1))



# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
ntest = 5000
beta0_test = runif(ntest, 4, 6)
beta1_test = runif(ntest, -1, 1)
beta2_test = runif(ntest, -1, 1)

var_test   = runif(ntest, 0.001, 4)
scale_test = runif(ntest, 0.001, 0.1)

plan(sequential)
gc()
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

with_progress({
  p2 = progressor(along = seq_len(ntest))
  res = future_lapply(
    seq_len(ntest),
    function(j) {
      p2(sprintf("iteration %d/%d", j, ntest))

      # True model should be 2 covs.
      params2 = list(L = 1, nx = 256, ny = 256, z_scale = 0.05,
                     xi = scale_test[j], alpha = var_test[j],
                     gamma0 = beta0_test[j], gamma1 = beta1_test[j], gamma2 = beta2_test[j],
                     z1 = z1, z2 = z2, z3 = z3)
      
      sim_cov2 = simulate_LGCP_point_pattern_2cov(params2)
      

      z_im1 = im(mat= sim_cov2$fields$z1, xcol = seq(0, params2$L, length.out = params2$nx), yrow = seq(0, params2$L, length.out = params2$ny))
      z_im2 = im(mat= sim_cov2$fields$z2, xcol = seq(0, params2$L, length.out = params2$nx), yrow = seq(0, params2$L, length.out = params2$ny))
      z_im3 = im(mat= sim_cov2$fields$z3, xcol = seq(0, params2$L, length.out = params2$nx), yrow = seq(0, params2$L, length.out = params2$ny))
      
      
      X1 = ppp(x = sim_cov2$points[,1], y = sim_cov2$points[,2], window = owin(c(0, params2$L), c(0, params2$L)))

      # under true model structure:
      fits_kppm = kppm(X1 ~ z_im1 + z_im2, clusters = "LGCP", method = "mincon")
      beta_KPPM = coef(fits_kppm)
      fits_clik2 = kppm(X1 ~ z_im1 + z_im2, clusters = "LGCP", method = "clik2")
      beta_clik2 = coef(fits_clik2)
      fits_palm = kppm(X1 ~ z_im1 + z_im2, clusters = "LGCP", method = "palm")
      beta_palm = coef(fits_palm)
      fits_adapcl = kppm(X1 ~ z_im1 + z_im2, clusters = "LGCP", method = "adapcl")
      beta_adapcl = coef(fits_palm)
      

      Lambda_est_test_true = exp(beta_KPPM[1] + beta_KPPM[2]*z_im1 + beta_KPPM[3]*z_im2)
      Lambda_true_test_true = exp(params2$gamma0 + params2$gamma1*z_im1 + params2$gamma2*z_im2)
      
      
      fits_kppm_1more = kppm(X1 ~ z_im1 + z_im2 + z_im3, clusters = "LGCP")
      beta_KPPM_1more = coef(fits_kppm_1more)
      
      fits_kppm_1less = kppm(X1 ~ z_im1, clusters = "LGCP")
      beta_KPPM_1less = coef(fits_kppm_1less)
      
      Lambda_est_test_1more = exp(beta_KPPM_1more[1] + beta_KPPM_1more[2]*z_im1 + beta_KPPM_1more[3]*z_im2 + beta_KPPM_1more[4]*z_im3)
      Lambda_est_test_1less = exp(beta_KPPM_1less[1] + beta_KPPM_1less[2]*z_im1)
      
      
      Lfv_test_truelam_true = Linhom(X1, lambda = Lambda_true_test_true, r = r, correction = "border")
      Lfv_test_estlam_true  = Linhom(X1, lambda = Lambda_est_test_true,  r = r, correction = "border")
      Lfv_test_estlam_1more = Linhom(X1, lambda = Lambda_est_test_1more, r = r, correction = "border")
      Lfv_test_estlam_1less = Linhom(X1, lambda = Lambda_est_test_1less, r = r, correction = "border")
      

      L_test_truelam_true = as.numeric(Lfv_test_truelam_true$border - Lfv_test_truelam_true$r)
      L_test_estlam_true  = as.numeric(Lfv_test_estlam_true$border  - Lfv_test_estlam_true$r)
      L_test_estlam_1more = as.numeric(Lfv_test_estlam_1more$border - Lfv_test_estlam_1more$r)
      L_test_estlam_1less = as.numeric(Lfv_test_estlam_1less$border - Lfv_test_estlam_1less$r)
      
      
      
      L_test_truelam_true = (L_test_truelam_true - m_L_true) / std_L_true
      L_test_estlam_true = (L_test_estlam_true - m_L_true) / std_L_true
      L_test_estlam_1more = (L_test_estlam_1more - m_L_1more) / std_L_1more
      L_test_estlam_1less = (L_test_estlam_1less - m_L_1less) / std_L_1less
      

      
      list(
        L_test_truelam_true = L_test_truelam_true,
        L_test_estlam_true = L_test_estlam_true,
        L_test_estlam_1more = L_test_estlam_1more,
        L_test_estlam_1less = L_test_estlam_1less,
        
        beta0_mincon  = as.numeric(beta_KPPM["(Intercept)"]),
        beta1_mincon  = as.numeric(beta_KPPM["z_im1"]),
        beta2_mincon  = as.numeric(beta_KPPM["z_im2"]),
        
        var_mincon   = as.numeric(fits_kppm$par[[1]]),
        scale_mincon = as.numeric(fits_kppm$par[[2]]),
        
        var_clik2   = as.numeric(fits_clik2$par[[1]]),
        scale_clik2 = as.numeric(fits_clik2$par[[2]]),
        
        var_palm   = as.numeric(fits_palm$par[[1]]),
        scale_palm = as.numeric(fits_palm$par[[2]]),
        
        var_adapcl   = as.numeric(fits_adapcl$par[[1]]),
        scale_adapcl = as.numeric(fits_adapcl$par[[2]]),
        
        beta0_KPPM_1more = as.numeric(beta_KPPM_1more["(Intercept)"]),
        beta1_KPPM_1more = as.numeric(beta_KPPM_1more["z_im1"]),
        beta2_KPPM_1more = as.numeric(beta_KPPM_1more["z_im2"]),
        beta3_KPPM_1more = as.numeric(beta_KPPM_1more["z_im3"]),
        
        beta0_KPPM_1less = as.numeric(beta_KPPM_1less["(Intercept)"]),
        beta1_KPPM_1less = as.numeric(beta_KPPM_1less["z_im1"])
      )
    },
    future.seed = 2025,
    future.packages = c("spatstat.geom", "spatstat.explore", "spatstat.model")
  )
})

L_test_truelam_true_list = sapply(res, function(x) x[["L_test_truelam_true"]]) 
L_test_estlam_true_list = sapply(res, function(x) x[["L_test_estlam_true"]]) 
L_test_estlam_1more_list = sapply(res, function(x) x[["L_test_estlam_1more"]]) 
L_test_estlam_1less_list = sapply(res, function(x) x[["L_test_estlam_1less"]]) 

beta0_1more  = sapply(res, function(x) x[["beta0_KPPM_1more"]])
beta1_1more  = sapply(res, function(x) x[["beta1_KPPM_1more"]])
beta2_1more  = sapply(res, function(x) x[["beta2_KPPM_1more"]])
beta3_1more  = sapply(res, function(x) x[["beta3_KPPM_1more"]])

beta0_1less  = sapply(res, function(x) x[["beta0_KPPM_1less"]])
beta1_1less  = sapply(res, function(x) x[["beta1_KPPM_1less"]])


beta0_mincon  = sapply(res, function(x) x[["beta0_mincon"]])
beta1_mincon  = sapply(res, function(x) x[["beta1_mincon"]])
beta2_mincon  = sapply(res, function(x) x[["beta2_mincon"]])
var_mincon    = sapply(res, function(x) x[["var_mincon"]])
scale_mincon  = sapply(res, function(x) x[["scale_mincon"]])

var_clik2    = sapply(res, function(x) x[["var_clik2"]])
scale_clik2  = sapply(res, function(x) x[["scale_clik2"]])

var_palm    = sapply(res, function(x) x[["var_palm"]])
scale_palm  = sapply(res, function(x) x[["scale_palm"]])

var_adapcl    = sapply(res, function(x) x[["var_adapcl"]])
scale_adapcl  = sapply(res, function(x) x[["scale_adapcl"]])

test_par = cbind(var_test, scale_test) 
Y_test_truemodel <- scale(test_par, center = m_par_true, scale = std_par_true)
Y_test_1more <- scale(test_par, center = m_par_1more, scale = std_par_1more)
Y_test_1less <- scale(test_par, center = m_par_1less, scale = std_par_1less)



keep2 = (colSums(is.na(L_test_truelam_true_list)) == 0) &
  (colSums(is.na(L_test_estlam_true_list)) == 0) &
  (colSums(is.na(L_test_estlam_1more_list)) == 0) &
  (colSums(is.na(L_test_estlam_1less_list)) == 0) 

L_test_truelam_true_list = L_test_truelam_true_list[, keep2, drop = FALSE]
L_test_estlam_true_list = L_test_estlam_true_list[, keep2, drop = FALSE]
L_test_estlam_1more_list = L_test_estlam_1more_list[, keep2, drop = FALSE]
L_test_estlam_1less_list = L_test_estlam_1less_list[, keep2, drop = FALSE]


beta0_mincon = beta0_mincon[keep2]
beta1_mincon = beta1_mincon[keep2]
var_mincon   = var_mincon[keep2]
scale_mincon = scale_mincon[keep2]

var_clik2    = var_clik2[keep2]
scale_clik2  = scale_clik2[keep2]

var_palm     = var_palm[keep2]
scale_palm   = scale_palm[keep2]

var_adapcl   = var_adapcl[keep2]
scale_adapcl = scale_adapcl[keep2]

var_test = var_test[keep2]
scale_test = scale_test[keep2]

test_par = test_par[keep2, ]


L_test_truelam_true = array(as.numeric(t(as.matrix(L_test_truelam_true_list))), dim = c(nrow(t(as.matrix(L_test_truelam_true_list))), ncol(t(as.matrix(L_test_truelam_true_list))), 1))
L_test_estlam_true = array(as.numeric(t(as.matrix(L_test_estlam_true_list))), dim = c(nrow(t(as.matrix(L_test_estlam_true_list))), ncol(t(as.matrix(L_test_estlam_true_list))), 1))
L_test_estlam_1more = array(as.numeric(t(as.matrix(L_test_estlam_1more_list))), dim = c(nrow(t(as.matrix(L_test_estlam_1more_list))), ncol(t(as.matrix(L_test_estlam_1more_list))), 1))
L_test_estlam_1less = array(as.numeric(t(as.matrix(L_test_estlam_1less_list))), dim = c(nrow(t(as.matrix(L_test_estlam_1less_list))), ncol(t(as.matrix(L_test_estlam_1less_list))), 1))





# ------------------------------------------------------------------------------
py_run_string("
import random
import numpy as np
import torch
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)")
source_python("C:/Users/qihan/Desktop/LGCP/range_betas/simulation_study_range_betas_new/NN_M2_v3.py")


# Scenario 1
pred_true_model_true_beta = NN_model_est_M2(L_train_true, Y_train_true, 
                                            L_test_truelam_true, Y_test_truemodel, 
                                            batch_size=100, epochs=20, lr=1e-3)
Y_pred_true_model_true_beta = t(t(pred_true_model_true_beta[[1]])*std_par_true + m_par_true)

# Scenario 2
pred_true_model_est_beta = NN_model_est_M2(L_train_true, Y_train_true, 
                                            L_test_estlam_true, Y_test_truemodel, 
                                            batch_size=100, epochs=20, lr=1e-3)
Y_pred_true_model_est_beta = t(t(pred_true_model_est_beta[[1]])*std_par_true + m_par_true)

# Scenario 4-2
pred_1more_model_est_beta = NN_model_est_M2(L_train_1more, Y_train_1more, 
                                           L_test_estlam_1more, Y_test_1more, 
                                           batch_size=100, epochs=20, lr=1e-3)
Y_pred_1more_model_est_beta = t(t(pred_1more_model_est_beta[[1]])*std_par_1more + m_par_1more)

# Scenario 4-1
pred_1less_model_est_beta = NN_model_est_M2(L_train_1less, Y_train_1less, 
                                           L_test_estlam_1less, Y_test_1less, 
                                           batch_size=100, epochs=20, lr=1e-3)
Y_pred_1less_model_est_beta = t(t(pred_1less_model_est_beta[[1]])*std_par_1less + m_par_1less)



beta_train_true = cbind(beta0_train, beta1_train, beta2_train)
beta_true_test_true = cbind(beta0_test, beta1_test, beta2_test)
beta_est_test_true = cbind(beta0_mincon, beta1_mincon, beta2_mincon)

beta_train_1more = cbind(beta0_train, beta1_train, beta2_train, beta3_train)
beta_est_test_1more = cbind(beta0_1more, beta1_1more, beta2_1more, beta3_1more)

beta_train_1less = cbind(beta0_train, beta1_train)
beta_est_test_1less = cbind(beta0_1less, beta1_1less)


source_python("C:/Users/qihan/Desktop/LGCP/range_betas/simulation_study_range_betas_new/NN_M1_v3.py")
# +beta as input
# Scenario 1
pred_true_model_true_beta = NN_model_est_M1(L_train_true, Y_train_true, 
                                            L_test_truelam_true, Y_test_truemodel, 
                                            beta_train_true, beta_true_test_true, 
                                            batch_size=100, epochs=20, lr=1e-3)
Y_pred_true_model_true_beta = t(t(pred_true_model_true_beta[[1]])*std_par_true + m_par_true)

# Scenario 2
pred_true_model_est_beta = NN_model_est_M1(L_train_true, Y_train_true, 
                                           L_test_estlam_true, Y_test_truemodel, 
                                           beta_train_true, beta_est_test_true, 
                                           batch_size=100, epochs=20, lr=1e-3)
Y_pred_true_model_est_beta = t(t(pred_true_model_est_beta[[1]])*std_par_true + m_par_true)

# Scenario 4-2
pred_1more_model_est_beta = NN_model_est_M1(L_train_1more, Y_train_1more, 
                                            L_test_estlam_1more, Y_test_1more, 
                                            beta_train_1more, beta_est_test_1more, 
                                            batch_size=100, epochs=20, lr=1e-3)
Y_pred_1more_model_est_beta = t(t(pred_1more_model_est_beta[[1]])*std_par_1more + m_par_1more)

# Scenario 4-1
pred_1less_model_est_beta = NN_model_est_M1(L_train_1less, Y_train_1less, 
                                            L_test_estlam_1less, Y_test_1less, 
                                            beta_train_1less, beta_est_test_1less, 
                                            batch_size=100, epochs=20, lr=1e-3)
Y_pred_1less_model_est_beta = t(t(pred_1less_model_est_beta[[1]])*std_par_1less + m_par_1less)






# ------------------------------------------------------------------------------


t1 = Sys.time()
difftime(t1, t0, units = "mins") 



# check larger predictions:
sum(var_mincon>5)
sum(scale_mincon>0.15)

sum(var_clik2>5)
sum(scale_clik2>0.15)

sum(var_palm>5)
sum(scale_palm>0.15)

sum(var_adapcl>5)
sum(scale_adapcl>0.15)

sum(Y_pred_true_model_true_beta[,1]>5)
sum(Y_pred_true_model_true_beta[,2]>0.15)

sum(Y_pred_true_model_est_beta[,1]>5)
sum(Y_pred_true_model_est_beta[,2]>0.15)

sum(Y_pred_1more_model_est_beta[,1]>5)
sum(Y_pred_1more_model_est_beta[,2]>0.15)

sum(Y_pred_1less_model_est_beta[,1]>5)
sum(Y_pred_1less_model_est_beta[,2]>0.15)



# remove larger
mean((var_mincon[var_mincon<5] - var_test[var_mincon<5])^2)
mean((scale_mincon[scale_mincon<0.15] - scale_test[scale_mincon<0.15])^2)

mean((var_palm[var_palm<5] - var_test[var_palm<5])^2)
mean((scale_palm[scale_palm<0.15] - scale_test[scale_palm<0.15])^2)

mean((var_clik2[var_clik2<5] - var_test[var_clik2<5])^2)
mean((scale_clik2[scale_clik2<0.15] - scale_test[scale_clik2<0.15])^2)

mean((var_adapcl[var_adapcl<5] - var_test[var_adapcl<5])^2)
mean((scale_adapcl[scale_adapcl<0.15] - scale_test[scale_adapcl<0.15])^2)


mean((Y_pred_true_model_true_beta[,1][Y_pred_true_model_true_beta[,1]<5]      - var_test[Y_pred_true_model_true_beta[,1]<5])^2)
mean((Y_pred_true_model_true_beta[,2][Y_pred_true_model_true_beta[,2]<0.15] - scale_test[Y_pred_true_model_true_beta[,2]<0.15])^2)

mean((Y_pred_true_model_est_beta[,1][Y_pred_true_model_est_beta[,1]<5]      - var_test[Y_pred_true_model_est_beta[,1]<5])^2)
mean((Y_pred_true_model_est_beta[,2][Y_pred_true_model_est_beta[,2]<0.15] - scale_test[Y_pred_true_model_est_beta[,2]<0.15])^2)

mean((Y_pred_1more_model_est_beta[,1][Y_pred_1more_model_est_beta[,1]<5]      - var_test[Y_pred_1more_model_est_beta[,1]<5])^2)
mean((Y_pred_1more_model_est_beta[,2][Y_pred_1more_model_est_beta[,2]<0.15] - scale_test[Y_pred_1more_model_est_beta[,2]<0.15])^2)

mean((Y_pred_1less_model_est_beta[,1][Y_pred_1less_model_est_beta[,1]<5]      - var_test[Y_pred_1less_model_est_beta[,1]<5])^2)
mean((Y_pred_1less_model_est_beta[,2][Y_pred_1less_model_est_beta[,2]<0.15] - scale_test[Y_pred_1less_model_est_beta[,2]<0.15])^2)


# a bug, I also need to use index for betas...


