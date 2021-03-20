#Import packages 

#install.packages('keras')
#install.packages('tensorflow')
library(keras)
library(tensorflow)
library(cowplot)
library(tidyverse)
library(glue)
library(ggplot2)

#Data 
df <- read.csv("~/Documents/Master In Analytics /MGSC 661 - Multivariate /Term Project - World Countries /SenVsSK2.csv")#import the data and name it df
View(df)


###############Data Discovery 
qplot(Year, GDP_Per_Capita, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Life_Expectancy, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Mortality_rate_infant, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Population, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Fertility_rate, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Population, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Agriculture, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Industry, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Capital_Investment, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Domestic_credit_to_private_sector, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Trade_Balance, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Aid_Per_Capita, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Perc_Urban_Population, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Adolescent_fertility_rate, data = df, colour = CountryName,geom = c("point", "line"))
qplot(Year, Household_consumption, data = df, colour = CountryName,geom = c("point", "line"))

########### PCA ANALYSIS 

#For all variables
df1 = df[,-c(1,2)] #categ to remove
df1 = df1[,-c(12,14)]#NAs te remove 
interestk = scale(df1)#scale the data  
ggpairs(interestk)
library(ggfortify)
pca = prcomp(interestk, scale = TRUE)
autoplot(pca, data = df, loadings = TRUE, loadings.label = TRUE , col = "CountryName")
pve=(pca$sdev^2)/sum(pca$sdev^2)
par(mfrow=c(1,2))
plot(pve, ylim=c(0,1))
plot(cumsum(pve), ylim=c(0,1))
pca

##################
################## LSTM MODEL
##################


############################Senegal 
Series <-  df[1:60,c(1,5)] #keep only Senegal GDP Per Capita and Years 

#Visualize data 
Series %>%
  ggplot(aes(Year, GDP_Per_Capita)) +
  geom_point(color = palette_light()[[1]], alpha = 0.5) +
  theme_tq() +
  labs(
    title = "Senegal - From 1960 to 2019 (Full Data Set)"
  )
#Examine the ACF
Seriess <- Series$GDP_Per_Capita
acf(Seriess, lag.max = 50000)

#If non stationary differenciate it 
require(ggplot2)
diffed <- diff(Seriess, difference = 1)
head(diffed)
dataa <- data.frame("Years" = 1960:2018,"GDP/Capita" = c(diffed))
qplot(Years, GDP.Capita, data = dataa, geom = c("point", "line"))

#Lagged dataset: Data in a supervised learning mode 
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}

supervised = lag_transform(diffed,1)
head(supervised)

##The dataset has been split according to results from ACF plot 
N = nrow(supervised)
n = round(N *0.7, digits = 0)
supervised = supervised[1:45,]
train = supervised[1:30,]
test  = supervised[31:45,]

#Standardize the data --> Normalize the data 


scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}


Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]



##MODEL 

# Reshape the input to 3-dim
dim(x_train) <- c(length(x_train), 1, 1) #samples, timesteps, features

# specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 15                # must be a common divisor of both the train and test samples
units = 5                   
##set seed
set.seed(1000)
#=========================================================================================

model <- keras_model_sequential() 
model%>%
  layer_lstm(units, input_shape = c( X_shape2, X_shape3),return_sequences = TRUE )%>%
  layer_lstm(units = 1, return_sequences = FALSE )%>%
  layer_dense(units = 1)
#batch_input_shape = c(batch_size, X_shape2, X_shape3)
#, stateful= TRUE
#kernel_regularizer = regularizer_l1(0.001)
##compile the model

model %>% compile(
  loss = 'mae',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('mean_squared_error')
)

summary(model)


##Fit the model
Epochs = 50
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}

set.seed(1000)


#####
#The following code will be required to revert the predicted values to the original scale.
## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}


#####
#Prediction of the training dataset 
L = length(x_train)
dim(x_train) = c(length(x_train), 1, 1)

scaler = Scaled$scaler
predictions_train = numeric(0)
for(i in 1:L){
  X = x_train[i , , ]
  dim(X) = c(1,1,1)
  # forecast
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  predictions_train <- c(predictions_train,yhat)
}
predictions_train = diffinv(predictions_train, difference = 1, xi = c(Seriess[1]))
predictions_train

##Accurate y train
L = length(y_train)
scaler = Scaled$scaler
accurate_train = numeric(0)
for(i in 1:L){
  # invert scaling
  yhat = invert_scaling(y_train[i], scaler,  c(-1, 1))
  # save accurate value
  accurate_train <- c(accurate_train,yhat)
}
accurate_train= diffinv(accurate_train, difference = 1, xi = c(Seriess[1]))
accurate_train
#visualize 
data <- data.frame("Years" = 1960:1990,"Actual" = c(accurate_train), "Predicted" = c(predictions_train))
library(ggplot2)

ggplot(data, aes(Years)) +                    
  geom_line(aes(y=Predicted), colour="red") +  
  geom_line(aes(y=Actual), colour="green")  

#Get the RMSE 
library(Metrics)
rmse(accurate_train,predictions_train) #should I keep the log for that 
mape(accurate_train,predictions_train)



##Make predictions (test)
L = length(x_test)
dim(x_test) = c(length(x_test), 1, 1)

scaler = Scaled$scaler
predictions = numeric(0)
for(i in 1:L){
  X = x_test[i , , ]
  dim(X) = c(1,1,1)
  # forecast
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  predictions <- c(predictions,yhat)
}
predictions = diffinv(predictions, difference = 1, xi = c(Seriess[30]))
predictions = predictions[2:L]
predictions


##Accurate y test
L = length(y_test)
scaler = Scaled$scaler
accurate = numeric(0)
for(i in 1:L){
  # invert scaling
  yhat = invert_scaling(y_test[i], scaler,  c(-1, 1))
  # save accurate value
  accurate <- c(accurate,yhat)
}
accurate= diffinv(accurate, difference = 1, xi = c(Seriess[30]))
accurate = accurate[2:L]
accurate
#visualize 
data <- data.frame("Years" = 1991:2004,"Actual" = c(accurate), "Predicted" = c(predictions))
library(ggplot2)

ggplot(data, aes(Years)) +                    
  geom_line(aes(y=Predicted), colour="red") +  
  geom_line(aes(y=Actual), colour="green")+
  scale_colour_manual(breaks = c("Predicted" = "red","Actual" = "green"))

#Get the RMSE 
library(Metrics)
rmse(accurate,predictions) 
mape(accurate,predictions)

##Combine
all_a <-c(accurate_train, accurate)
all_b<-c(predictions_train, predictions)
dataX <- data.frame("Years" = 1960:2004,"Actual" = c(all_a), "Predicted" = c(all_b))
ggplot(dataX, aes(Years)) +                    
  geom_line(aes(y=Predicted), colour="red") +  
  geom_line(aes(y=Actual), colour="green")  




############################
############################South Korea 
############################



Series <-  df[61:120,c(1,5)] #keep only SK
#Visualize data 
Series %>%
  ggplot(aes(Year, GDP_Per_Capita)) +
  geom_point(color = palette_light()[[1]], alpha = 0.5) +
  theme_tq() +
  labs(
    title = "South Korea - From 1960 to 2019 (Full Data Set)"
  )

#Examine the ACF
Seriess <- Series$GDP_Per_Capita
acf(Seriess, lag.max = 50000)


#If non stationary differenciate it 
require(ggplot2)

diffed <- diff(Seriess, difference = 1)
head(diffed)
dataa <- data.frame("Years" = 1960:2018,"GDP/Capita" = c(diffed))
qplot(Years, GDP.Capita, data = dataa, geom = c("point", "line"), main = 'South Korea')

#Lagged dataset: Data in a supervised learning mode 
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}

supervised = lag_transform(diffed,1)
head(supervised)

##The dataset has been split according to results from ACF plot 
supervised = supervised[1:45,]

train = supervised[1:30,]
test  = supervised[31:45,]
#Standardize the data --> Normalize the data 

scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}


Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]



##MODEL 
#Once the LSTM model is fit to the training data, it can be used to make forecasts

# Reshape the input to 3-dim
dim(x_train) <- c(length(x_train), 1, 1) #samples, timesteps, features

# specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 15                # must be a common divisor of both the train and test samples
units = 5                  
##set seed
set.seed(78910)
#=========================================================================================
###
model <- keras_model_sequential() 
model%>%
  layer_lstm(units, input_shape = c( X_shape2, X_shape3),return_sequences = TRUE )%>%
  layer_lstm(units = 1, return_sequences = FALSE )%>% #add a second layer 
  layer_dense(units = 1)


##compile the model

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.03, decay = 1e-6 ),  
  metrics = c('mean_squared_error')
)

summary(model)


##Fit the model
Epochs = 50  
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



#####
#The following code will be required to revert the predicted values to the original scale.
## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}


#####
#Prediction of the training dataset 
L = length(x_train)
dim(x_train) = c(length(x_train), 1, 1)

scaler = Scaled$scaler
predictions_train = numeric(0)
for(i in 1:L){
  X = x_train[i , , ]
  dim(X) = c(1,1,1)
  # forecast
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  predictions_train <- c(predictions_train,yhat)
}
predictions_train = diffinv(predictions_train, difference = 1, xi = c(Seriess[1]))
predictions_train

##Accurate y train
L = length(y_train)
scaler = Scaled$scaler
accurate_train = numeric(0)
for(i in 1:L){
  # invert scaling
  yhat = invert_scaling(y_train[i], scaler,  c(-1, 1))
  # save accurate value
  accurate_train <- c(accurate_train,yhat)
}
accurate_train= diffinv(accurate_train, difference = 1, xi = c(Seriess[1]))
accurate_train 

#visualize the training set
data <- data.frame("Years" = 1960:1990,"Actual" = c(accurate_train), "Predicted" = c(predictions_train))
library(ggplot2)

ggplot(data, aes(Years)) +                    
  geom_line(aes(y=Predicted), colour="red") +  
  geom_line(aes(y=Actual), colour="green")  

#Get the RMSE 
library(Metrics)
rmse(accurate_train,predictions_train)  
mape(accurate_train,predictions_train)

##Make predictions (test)
L = length(x_test)
dim(x_test) = c(length(x_test), 1, 1)

scaler = Scaled$scaler
predictions = numeric(0)
for(i in 1:L){
  X = x_test[i , , ]
  dim(X) = c(1,1,1)
  # forecast
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  predictions <- c(predictions,yhat)
}
predictions = diffinv(predictions, difference = 1, xi = c(Seriess[30]))
predictions = predictions[2:L]
predictions

##Accurate y test
L = length(y_test)
scaler = Scaled$scaler
accurate = numeric(0)
for(i in 1:L){
  # invert scaling
  yhat = invert_scaling(y_test[i], scaler,  c(-1, 1))
  # save accurate value
  accurate <- c(accurate,yhat)
}
accurate= diffinv(accurate, difference = 1, xi = c(Seriess[30]))
accurate = accurate[2:L]
accurate
#visualize 
data <- data.frame("Years" = 1990:2003,"Actual" = c(accurate), "Predicted" = c(predictions))
library(ggplot2)

ggplot(data, aes(Years)) +                    
  geom_line(aes(y=Predicted), colour="red") +  
  geom_line(aes(y=Actual), colour="green")  

#Get the RMSE 
library(Metrics)
rmse(accurate,predictions)
mape(accurate,predictions) 


##Combine the plots
all_a <-c(accurate_train, accurate)
all_b<-c(predictions_train, predictions)
dataX <- data.frame("Years" = 1960:2004,"Actual" = c(all_a), "Predicted" = c(all_b))
ggplot(dataX, aes(Years)) +                    
  geom_line(aes(y=Predicted), colour="red") +  
  geom_line(aes(y=Actual), colour="green")  