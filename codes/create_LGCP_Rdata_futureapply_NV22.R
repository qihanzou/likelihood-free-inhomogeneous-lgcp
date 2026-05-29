library(future.apply)
library(spatstat.geom)     
library(spatstat.explore)  
library(spatstat.random)   
library(spatstat.model)
library(tibble)
library(purrr)

ncores <- max(1, parallel::detectCores() - 2)
seed_train = 2025
seed_test = 2026



# Use future_lapply
simulate_lgcp_batch <- function(mu, var, scale, workers, seed) {
  n <- length(mu)
  plan(sequential)  
  gc()
  plan(multisession, workers = workers)
  
  future_lapply(
    seq_len(n),
    function(i) {
      spatstat.random::rLGCP(mu = mu[i], var = var[i], scale = scale[i], saveLambda = FALSE)
    },
    future.seed = seed, future.packages  = c("spatstat.random")                        
  )
}

# TRAIN
ntrain <- 1000
mu_train    <- runif(ntrain, 4, 6)
var_train   <- runif(ntrain, 0, 4)
scale_train <- runif(ntrain, 0.001, 0.1)
LGCP_sims_train <- simulate_lgcp_batch(mu_train, var_train, scale_train, workers = ncores, seed = seed_train)

# TEST  
ntest <- 500
mu_test    <- runif(ntest, 4, 6)
var_test   <- runif(ntest, 0, 4)
scale_test <- runif(ntest, 0.001, 0.1)
LGCP_sims_test <- simulate_lgcp_batch(mu_test, var_test, scale_test, workers = ncores, seed = seed_test)




# Train
plan(sequential)
gc()
plan(multisession, workers = ncores)
Data_LGCP_train <- tibble(mu = mu_train, var = var_train, scale = scale_train, pp = LGCP_sims_train)
Data_LGCP_train$N <- map_dbl(Data_LGCP_train$pp, spatstat.geom::npoints)
Data_LGCP_train$L <- future_lapply(
  seq_along(Data_LGCP_train$pp),
  function(i) {
    Li <- spatstat.explore::Lest(Data_LGCP_train$pp[[i]], correction = "best")
    Li$iso - Li$r
  },
  future.seed = seed_train, future.packages = c("spatstat.explore", "spatstat.geom") 
)
Data_LGCP_train$pp <- NULL
idx_na_L <- vapply(Data_LGCP_train$L, function(L) any(is.na(L)), logical(1))
Data_LGCP_train <- Data_LGCP_train[!idx_na_L, ]


m_L <- mean(unlist(Data_LGCP_train$L))
std_L <- sd(unlist(Data_LGCP_train$L))
train_L <- lapply(Data_LGCP_train$L, function(L){(L - m_L) / std_L})
train_L <- array_reshape(train_L, c(nrow(Data_LGCP_train), length(train_L[[1]]), 1))
train_N <- select(Data_LGCP_train, N)
m_N <- apply(train_N, 2, mean)
std_N <- apply(train_N, 2, sd)
train_N <- scale(as.matrix(train_N), center = m_N, scale = std_N)
train_par <- as.matrix(select(Data_LGCP_train,mu:scale))
m_par <- apply(train_par, 2, mean)
std_par <- apply(train_par, 2, sd)
train_par <- scale(train_par, center = m_par, scale = std_par)




# Test
plan(sequential)
gc()
plan(multisession, workers = ncores)
Data_LGCP_test <- tibble(mu = mu_test, var = var_test, scale = scale_test, pp = LGCP_sims_test)
Data_LGCP_test$L <- future_lapply(
  seq_along(Data_LGCP_test$pp),
  function(i) {
    Li <- spatstat.explore::Lest(Data_LGCP_test$pp[[i]], correction = "best")
    Li$iso - Li$r
  },
  future.seed = seed_test, future.packages = c("spatstat.explore", "spatstat.geom") 
)
Data_LGCP_test$N <- map_dbl(Data_LGCP_test$pp, spatstat.geom::npoints)
Data_LGCP_test$mincon <- future_lapply(
  seq_along(Data_LGCP_test$pp),
  function(i) {
    spatstat.model::kppm(Data_LGCP_test$pp[[i]], clusters = "LGCP")
  },
  future.seed = seed_test, future.packages = c("spatstat.model", "spatstat.geom") 
)
Data_LGCP_test$mu_mincon    <- purrr::map_dbl(Data_LGCP_test$mincon, "mu")
Data_LGCP_test$var_mincon   <- purrr::map_dbl(Data_LGCP_test$mincon, c("par","sigma2"))
Data_LGCP_test$scale_mincon <- purrr::map_dbl(Data_LGCP_test$mincon, c("par","alpha"))
Data_LGCP_test$pp     <- NULL
Data_LGCP_test$mincon <- NULL
test_L <- lapply(Data_LGCP_test$L, function(L){(L - m_L) / std_L})
test_L <- array_reshape(test_L, c(nrow(Data_LGCP_test), length(test_L[[1]]), 1))
test_N <- scale(as.matrix(select(Data_LGCP_test, N)), center = m_N, scale = std_N)
test_par <- scale(as.matrix(select(Data_LGCP_test, mu:scale)), center = m_par, scale = std_par)
test_mincon <- as.matrix(select(Data_LGCP_test, mu_mincon:scale_mincon))


# Save the training and testing data 
save(train_L, train_N, train_par, test_L,  test_N,  test_par, test_mincon,
     m_L, std_L, m_N, std_N, m_par, std_par, 
     seed_train, seed_test, ntrain, ntest,
     file = "LGCP_train10000_test5000_test.Rdata")



