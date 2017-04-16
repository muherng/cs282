library(plotly)
packageVersion('plotly')
library(MASS)
library(mvtnorm)
bivn <- mvrnorm(1000, mu = c(0, 0), Sigma = matrix(c(1, .5, .5, 1), 2))

# Define the cars vector with 5 values
cars <- c(1, 3, 6, 4, 9)

library(plotly)
library(stringr)
library(reshape2)
library(e1071)
x = c(-0.86,-0.3,-0.05,0.73)
n = c(5,5,5,5)
y = c(0,1,3,5)
bio = data.frame(x,n,y)

pdf_eval <- function(row){
  a = row[1,1]
  b = row[1,2]
  #print(row)
  obs = c(a,b)
  mu = c(0,10)
  cov = matrix(c(4,10,10,100),nrow=2,ncol=2)
  pdf = dmvnorm(obs,mu,cov)
  for (i in 1:dim(bio)[1]){
    row = bio[i,]
    x = row[1,1]
    n = row[1,2]
    y = row[1,3]
    theta = a + b*x
    pdf = pdf*sigmoid(theta)^n*(1-sigmoid(theta))^(n-y)
  }
  return(pdf)
}

post <- function(data) {
  size = dim(data)[1]
  height = rep(0L, size)
  for (i in 1:size){
    row = data[i,]
    height[i] = pdf_eval(row)
  }
  data[,"post"] <- height
  return(data)
}

print("xgrid")
inc = 0.1
xgrid <-  seq(from = -5, to = 5, by = inc)
ygrid <-  seq(from = -5, to = 5, by = inc)
part = length(xgrid)
# Generate a dataframe with every possible combination of wt and hp
data.fit <-  expand.grid(alpha = xgrid, beta = ygrid)
cont = post(data.fit)
p <- plot_ly(cont, x = ~alpha, y = ~beta, z = ~post, type = "contour") %>% layout(autosize = F, width = 600, height = 500)
p

draw <- function(cont,part,xgrid,ygrid){
  mat = matrix(0,part,part)
  for (i in 1:dim(cont)[1]){
      alpha = match(cont[i,1],xgrid)
      beta = match(cont[i,2],ygrid)
      mat[alpha,beta] = cont[i,3]
  }
  
  a_post = rowSums(mat)
  a_post = a_post/sum(a_post)
  for (i in 2:length(a_post)){
    a_post[i] = a_post[i] + a_post[i-1]
  }
  #print(a_post)
  u_draws = sort(runif(1000))
  alpha_draws = rep(0,1000)
  alpha_index = rep(0,1000)
  for (i in 1:1000){
    dr = xgrid[1]
    index = 1
    for (j in 1:length(a_post)){
      if (a_post[j] < u_draws[i]){
        dr = xgrid[j]
        index = j
      } else {
        break
      }
    }
    alpha_draws[i] = dr
    alpha_index[i] = index
  }
  u_draws = runif(1000)
  beta_draws = rep(0,1000)
  for (i in 1:1000){
    b_post = mat[alpha_index[i],]
    b_post = b_post/sum(b_post)
    print("bpost")
    print(b_post)
    for (j in 2:length(b_post)){
      b_post[j] = b_post[j] + b_post[j-1]
    }
    dr = ygrid[1]
    index = 1
    for (j in 1:length(b_post)){
      if (b_post[j] < u_draws[i]){
        dr = ygrid[j]
        index = j
      } else {
        break
      }
    }
    beta_draws[i] = dr
    #beta_index[i] = index
  }
  for (i in 1:length(alpha_draws)){
    alpha_draws[i] = alpha_draws[i] + inc*runif(1)
    beta_draws[i] = beta_draws[i] + inc*runif(1)
  }
    
  return(list(alpha_draws,beta_draws))
}

points <- draw(cont,part,xgrid,ygrid)
alpha_draws = points[[1]]
beta_draws = points[[2]]

LD = rep(0,1000)
for (i in 1:length(alpha_draws)){
  LD[i] = -1.0*alpha_draws[i]/beta_draws[i]
}
LD = LD[ !LD > 5]
LD = LD[ !LD < -5]
hist(LD,breaks=seq(-5,5,0.1))

plot(points[[1]], points[[2]], main="Scatterplot", 
     xlab="alpha", ylab="beta",pch=16,xlim = c(-5,5),ylim=c(-5,5),asp=1) 

