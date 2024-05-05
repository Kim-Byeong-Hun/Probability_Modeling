library(ecr)
library(rminer)
library(smoof) # For the benchmark functions

# Load wine quality dataset directly from UCI repository:
file = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
d = read.table(file = file, sep = ";", header = TRUE)

# Convert the output variable into 2 classes of wine:
# "low" <- 3,4,5 or 6; "high" <- 7, 8 or 9
d$quality = cut(d$quality, c(1, 6, 10), c("low", "high"))

# To speed up the demonstration, only 25% of the data is used:
n = nrow(d) # total number of samples
ns = round(n * 0.25) # select a quarter of the samples
set.seed(12345) # for replicability
ALL = sample(1:n, ns) # contains 25% of the index samples
w = d[ALL,] # new wine quality data.frame

# Show the attribute names:
cat("attributes:", names(w), "\n")
cat("output class distribution (25% samples):\n")
print(table(w$quality)) # show distribution of classes

# Save dataset to a local CSV file:
write.table(w, "wq25.csv", col.names = TRUE, row.names = FALSE, sep = ";")

# Holdout split into training (70%) and test data (30%):
H = holdout(w$quality, ratio = 0.7)
cat("nr. training samples:", length(H$tr), "\n")
cat("nr. test samples:", length(H$ts), "\n")

# Save to file the holdout split index:
save(H, file = "wine-H.txt", ascii = TRUE)
output = ncol(w) # output target index (last column)
maxinputs = output - 1 # number of maximum inputs

# Auxiliary functions:
# Rescale x from [0,1] to [min,max] domain:
transf = function(x, min, max) return (x * (max - min) + min)

# Decode the x genome into the model hyperparameters:
decode = function(x) {
  nrounds = round(transf(x[1], 1, 220)) # [1,200]
  eta = x[2] # [0.0,1.0]
  gamma = transf(x[3], 0, 10) # [0,10]
  max_depth = round(transf(x[4], 0, 12)) # {0,...,12}
  return (c(nrounds, eta, gamma, max_depth))
}

# Evaluation function (requires some computation):
eval = function(x) {
  features = round(x[1:maxinputs]) # 0 or 1 vector
  inputs = which(features == 1) # indexes with 1 values
  if (length(inputs) == 0) inputs = 1
  J = c(inputs, output) # attributes
  k3 = c("kfold", 3, 123) # internal 3-fold validation
  hpar = decode(x[(maxinputs + 1):length(x)])
  M = suppressWarnings(try(
    mining(quality~., w[H$tr, J], method = k3,
           model = "xgboost", nrounds = hpar[1],
           eta = hpar[2], gamma = hpar[3], max_depth = hpar[4])
    , silent = TRUE))
  if (class(M) == "try-error") auc = 0.5 # worst auc
  else auc = as.numeric(mmetric(M, metric = "AUC"))
  auc1 = 1 - auc # maximization into minimization goal
  ninputs = length(inputs) # number of input features
  EVALS <<- EVALS + 1 # update evaluations
  if (EVALS == 1 || EVALS %% Psize == 0) # show current evaluation:
    cat(EVALS, " evaluations (AUC: ", round(auc, 2),
        " nr.features:", ninputs, ")\n", sep = "")
  return (c(auc1, ninputs)) # 1-auc, ninputs
}

# SMS-EMOA multi-objective optimization:
cat("SMS-EMOA optimization:\n")
m = 2 # two objectives: AUC and number of input features
hxgb = 4 # number of hyperparameters for xgboost
genome = maxinputs + hxgb # genome length
lower = rep(0, genome)
upper = rep(1, genome)
EVALS <<- 0 # global variable
PTM = proc.time() # start clock
Psize = 20 # population size

# Use smsemoa function for optimization
res = smsemoa(
  fitness.fun = eval,
  n.objectives = m,
  n.dim = genome,
  minimize = c(TRUE, TRUE),
  lower = lower,
  upper = upper,
  mu = Psize,
  ref.point = c(1, genome)
)

sec = (proc.time() - PTM)[3] # get seconds elapsed
cat("time elapsed:", sec, "\n")

# Save to file the optimized Pareto front:
save(res, file = "wine-s1-sms-emoa.txt", ascii = TRUE)

