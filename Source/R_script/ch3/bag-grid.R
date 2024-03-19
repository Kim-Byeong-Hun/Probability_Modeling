### bag-grid.R file ###
source("C:/Users/user/Desktop/2024년/[Lecture] Probability Modeling/Source/R_script/blind.R") # load the blind search methods
source("C:/Users/user/Desktop/2024년/[Lecture] Probability Modeling/Source/R_script/grid.R") # load the grid search methods
source("C:/Users/user/Desktop/2024년/[Lecture] Probability Modeling/Source/R_script/functions.R") # load the profit function
# grid search for all bag prices, step of 100$
PTM=proc.time() # start clock
S1=gsearch(profit,rep(1,5),rep(1000,5),rep(100,5),"max")
sec=(proc.time()-PTM)[3] # get seconds elapsed
cat("gsearch best s:",S1$sol,"f:",S1$eval,"time:",sec,"s\n")
# grid search 2 for all bag prices, step of 100$
PTM=proc.time() # start clock
S2=gsearch2(profit,rep(1,5),rep(1000,5),rep(100,5),"max")
sec=(proc.time()-PTM)[3] # get seconds elapsed
cat("gsearch2 best s:",S2$sol,"f:",S2$eval,"time:",sec,"s\n")
# nested grid with 3 levels and initial step of 500$
PTM=proc.time() # start clock
S3=ngsearch(profit,rep(1,5),rep(1000,5),3,rep(500,5),"max")
sec=(proc.time()-PTM)[3] # get seconds elapsed
cat("ngsearch best s:",S3$sol,"f:",S3$eval,"time:",sec,"s\n")