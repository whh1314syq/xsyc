library(caret)
library(forecast)
fun_arima = function(history_raw){
  # ##
  # 数据长度短于一个周期长度时，模型中仍然可以使用frequency=12
  # Note:columns "sales" & "timestamp" cannot be renamed. 
  # Given m samples, m-12 samples will be in the 1st folds,m-6 samples in the 7th folds.
  # We set m-6 = n_predictors (so we will obtain 6 cv_error at least)
  
  #n_predictors = sum(!names(history) %in% c("timestamp","sales"))
  
  if(nrow(history_raw) >= 18){ 

    cat("Arima: Executing Cross-Validation.\n")
        
    cv_index = createTimeSlices(history_raw$sales, initialWindow=nrow(history_raw)-12, horizon=1, fixedWindow=FALSE)
    cv_train = cv_index$train
    cv_test  = cv_index$test
    error_list = rep(NA, length(cv_train))
    
    for(i in 1:length(cv_train)){
      train = history_raw[cv_train[[i]], ]
      test  = history_raw[cv_test[[i]], ]
      
      result <- tryCatch({
        model <- auto.arima(ts(train$sales, frequency = 12),
                           seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)
        pred  <- forecast(model, h=1)%>%data.frame()
        error <- fun_error_measurement(pred$Point.Forecast, test$sales)
        list(error=error)
      },error=function(e){
      })
      
      error_list[[i]] <- result$error
    }
    
    cv_error <- fun_cv_error(error_list)
  }
      #xreg_train = train%>%select(-timestamp, -sales)
      #xreg_test  = test%>% select(-timestamp, -sales)
      
      #result <- tryCatch({
      #  model = auto.arima(ts(train$sales), xreg = xreg_train,lambda = BoxCox.lambda(ts(train$sales)),
      #                     seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)
      #  pred = forecast(model, xreg = xreg_test, h = 1)%>%data.frame()
      #  pred_fitted = model$fitted
      #  error = fun_error_measurement(pred$Point.Forecast,test$sales)
      #  fig = dygraph_plot_history(history, train, test, pred, pred_fitted,"Arima",i)
      #  result<-list(error=error,fig=fig)
      #},error=function(e){
      #  tryCatch({
      #    model = auto.arima(ts(train$sales), lambda = BoxCox.lambda(ts(train$sales)),
      #                       seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)
      #    pred = forecast(model, h = 1)%>%data.frame()
      #    pred_fitted = model$fitted
      #    error = fun_error_measurement(pred$Point.Forecast,test$sales)
      #    fig = dygraph_plot_history(history, train, test, pred, pred_fitted,"Arima",i)
      #  },error=function(e){
      #  })
      #  return(list(error=error,fig=fig))
      #})
      
      #error_list[[i]]<-result$error
      #plot_history[[i]]<-result$fig
   #}
  #  cv_error = fun_cv_error(error_list)
  #}
  
  if(nrow(history_raw) >= 9 & nrow(history_raw) < 18){
    
    cat("Arima: Executing Hold-Out.\n")
    
    train <- history_raw[1:(nrow(history_raw)-3), ]
    test  <- history_raw[(nrow(history_raw)-3+1):nrow(history_raw), ]
    #xreg_train = train %>% select(-timestamp, -sales)
    #xreg_test = test %>% select(-timestamp)
    
    result <- tryCatch({
      model <- auto.arima(ts(train$sales), 
                         seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)  
      pred  <- forecast(model,  h = 3)%>%data.frame()
      list(model=model, pred=pred)
    },error=function(e){
    })
    
    model = result$model
    pred  = result$pred
    error = fun_error_measurement(pred$Point.Forecast, test$sales)
    #pred_fitted = model$fitted
    #plot_history = dygraph_plot_history(history, train, test, pred, pred_fitted,"Arima",0)
    cv_error = error
  }
  
  if(nrow(history_raw) < 9){
    
    cat("Arima: Executing Defaults.\n")
    cv_error = NA
  }
  
  # Train the model on the full datasets
  result <- tryCatch({
    model <- auto.arima(ts(history_raw$sales, frequency = 12),
                        seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)  
    model_order <- model$arma
    pred <- forecast(model, h=4)%>%data.frame()
    list(model=model, pred=pred, model_order=model_order)
  },error=function(e){
  })
  model <- result[[1]]
  pred  <- result[[2]]
  model_order <- result[[3]]

  return(list(pred=pred$Point.Forecast, cv_error=cv_error, model_order=model_order))
}

fun_stl= function(history_raw){
  # ##
  # length of history_raw at least >= 25 for stl model
  # cross validation folds is 12
  
  if(nrow(history_raw) >= 37){
    
    cat("STL: Executing Cross Validation.\n")
    
    cv_index = createTimeSlices(history_raw$sales, initialWindow=nrow(history_raw)-12, horizon=1, fixedWindow=FALSE)#M3
    #cv_index = createTimeSlices(history_raw$sales, initialWindow=nrow(history_raw)-13, horizon=2, fixedWindow=FALSE)#M2
    #cv_index = createTimeSlices(history_raw$sales, initialWindow=nrow(history_raw)-12, horizon=1, fixedWindow=FALSE)#M1
    cv_train = cv_index$train
    cv_test  = cv_index$test
    error_list = rep(NA, length(cv_train))
    
    for(i in 1:length(cv_train)){
      train = history_raw[cv_train[[i]], ]
      test  = history_raw[cv_test[[i]], ]
      
      model <- stl(ts(train$sales, frequency=12), s.window="periodic", t.window=13, robust=TRUE)
      #model <- stl(ts(train$sales, frequency=12), s.window=5, t.window=13, robust=TRUE)
      pred  <- model%>%forecast(method="naive", h=1)%>%data.frame()
      error <- fun_error_measurement(pred$Point.Forecast, test$sales)
      error_list[[i]] <- error
    }
    
    cv_error <- fun_cv_error(error_list)
  }
  
  if(nrow(history_raw) >= 28 & nrow(history_raw) < 37){
    
    cat("STL: Executing Hold Out.\n")
    
    train <- history_raw[1:(nrow(history_raw)-3), ]
    test  <- history_raw[(nrow(history_raw)-2):nrow(history_raw), ]
    model <- stl(ts(train$sales, frequency=12), s.window="periodic", t.window=13, robust=TRUE)
    pred  <- model%>%forecast(method="naive", h=3)%>%data.frame()
    error <- fun_error_measurement(pred$Point.Forecast, test$sales)
    cv_error <- error
  }
  
  if(nrow(history_raw) < 28){
    cat("STL: Executing Default.\n")
    cv_error      <- NA
  }
  
  # Train model on full datasets
  result <- tryCatch({
    model <- stl(ts(history_raw$sales, frequency=12), s.window="periodic", t.window=13, robust=TRUE)
    pred  <- model%>%forecast(method="naive", h=4)%>%data.frame()
    list(model=model,pred=pred)
  },error=function(e){
    return(list(model=NA,pred=NA))
  })
  pred <- result$pred
  
  return(list(pred=pred$Point.Forecast, cv_error=cv_error))
}

fun_ets = function(history_raw){
  # ##
  # 数据长度短于一个周期长度时，模型中仍然可以使用frequency=12
  # Note:columns "sales" & "timestamp" cannot be renamed
  # Given m samples, m-12 samples will be in the 1st folds.
  # We set m-12 = 6 (each fold contains 6 samples at least)
  
  if(nrow(history_raw) >= 18){
    
    cat("ETS: Executing Cross-Validation.\n")
    
    cv_index = createTimeSlices(history_raw$sales, initialWindow = nrow(history_raw)-12, horizon = 1, fixedWindow = FALSE)
    cv_train = cv_index$train
    cv_test = cv_index$test
    error_list = rep(NA, length(cv_train))
    
    for(i in 1:length(cv_train)){
      train  = history_raw[cv_train[[i]], ]
      test   = history_raw[cv_test[[i]], ]
      tryCatch({
        model <- ets(ts(train$sales, frequency = 12))  
        pred  <- forecast(model, h = 1)%>%data.frame()
        error <- fun_error_measurement(pred$Point.Forecast,test$sales)
        error_list[[i]] <- error
      }, error = function(e){
      })
    }
    
    cv_error = fun_cv_error(error_list)
  }
  
  if(nrow(history_raw) >= 9 & nrow(history_raw) < 18){
    
    cat("ETS: Executing Hold-Out.\n")
    
    train = history_raw[1:(nrow(history_raw)-3), ]
    test  = history_raw[(nrow(history_raw)-3+1):nrow(history_raw), ]
    
    model <- ets(ts(train$sales, frequency = 12))  
    pred  <- forecast(model, h = 3)%>%data.frame()
    error <- fun_error_measurement(pred$Point.Forecast,test$sales)
    cv_error <- error
  }
  
  if(nrow(history_raw) < 9){
    
    cat("ETS: Executing Defaults.\n")
    cv_error = NA
  }  
  
  # Train the model on full datasets.
  model <- ets(ts(history_raw$sales, frequency = 12))  
  model_parameter <- model$components
  pred  <- forecast(model, h = 4)%>%data.frame()
  
  return(list(pred=pred$Point.Forecast, cv_error=cv_error, model_parameter=model_parameter))
  
}
fun_error_measurement = function(pred,true,method="smape4"){
  # Given positive pred & true:
  if(method=="mae")     return(mean(abs(pred-true)))                              #[0,Inf)  
  if(method=="mape"){
    df = data.frame(pred=pred,true=true)
    df$mape = abs((pred-true)/true)
    df$mape[df$pred==0 & df$true==0] = 0
    return(mean(df$mape))                                                         #[0,Inf)
  }    
  if(method=="smape1")  return(mean(abs(pred-true)/((pred+true)/2)))              #[0,2)  Armstrong (1985)  
  if(method=="smape2")  return(mean(abs(pred-true)/(abs(pred+true)/2)))           #[0,2)  Makridakis (1993)
  if(method=="smape3")  return(mean(abs(pred-true)/((abs(pred)+abs(true))/2)))    #[0,2)  Chen and Yang (2004)
  if(method=="smape5")  return(sum(abs(pred-true))/sum(pred+true))                #[0,1)
  if(method=="smape4"){
    df = data.frame(pred=pred,true=true)
    df$smape = abs(pred-true)/(abs(pred)+abs(true))
    df$smape[df$pred==0 & df$true==0] = 0
    return(mean(df$smape))                                                        #[0,1) 
  }   
  if(method=="lnQ")     return(sum(log(pred/true)^2))                             #[0,Inf) 
}
fun_cv_error = function(error_list,method="mean"){
  error_list_temp = error_list[!is.na(error_list) & !is.infinite(error_list)]
  if(method == "rank"){
    rank = rank(error_list_temp)
    weights = exp(1/rank)/sum(exp(1/rank))    
  }  
  if(method == "mean"){
    return(mean(error_list_temp))   
  }
  cv_error = sum(error_list_temp*weights)
  return(cv_error)
}

fun_arima2 = function(history_raw){
  # ##
  # 数据长度短于一个周期长度时，模型中仍然可以使用frequency=12
  # Note:columns "sales" & "timestamp" cannot be renamed. 
  # Given m samples, m-12 samples will be in the 1st folds,m-6 samples in the 7th folds.
  # We set m-6 = n_predictors (so we will obtain 6 cv_error at least)
  
  #n_predictors = sum(!names(history) %in% c("timestamp","sales"))
  
  if(nrow(history_raw) >= 18){ 

    cat("Arima: Executing Cross-Validation.\n")
        
    cv_index = createTimeSlices(history_raw$sales, initialWindow=nrow(history_raw)-12, horizon=1, fixedWindow=FALSE)
    cv_train = cv_index$train
    cv_test  = cv_index$test
    error_list = rep(NA, length(cv_train))
    
    for(i in 1:length(cv_train)){
      train = history_raw[cv_train[[i]], ]
      test  = history_raw[cv_test[[i]], ]
      
      
        model <- auto.arima(ts(train$sales, frequency = 12),
                           seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)
        pred  <- forecast(model, h=1)%>%data.frame()
        error <- fun_error_measurement(pred$Point.Forecast, test$sales)
       
      
      error_list[[i]] <- result$error
    }
    
    cv_error <- fun_cv_error(error_list)
  }
      #xreg_train = train%>%select(-timestamp, -sales)
      #xreg_test  = test%>% select(-timestamp, -sales)
      
      #result <- tryCatch({
      #  model = auto.arima(ts(train$sales), xreg = xreg_train,lambda = BoxCox.lambda(ts(train$sales)),
      #                     seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)
      #  pred = forecast(model, xreg = xreg_test, h = 1)%>%data.frame()
      #  pred_fitted = model$fitted
      #  error = fun_error_measurement(pred$Point.Forecast,test$sales)
      #  fig = dygraph_plot_history(history, train, test, pred, pred_fitted,"Arima",i)
      #  result<-list(error=error,fig=fig)
      #},error=function(e){
      #  tryCatch({
      #    model = auto.arima(ts(train$sales), lambda = BoxCox.lambda(ts(train$sales)),
      #                       seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)
      #    pred = forecast(model, h = 1)%>%data.frame()
      #    pred_fitted = model$fitted
      #    error = fun_error_measurement(pred$Point.Forecast,test$sales)
      #    fig = dygraph_plot_history(history, train, test, pred, pred_fitted,"Arima",i)
      #  },error=function(e){
      #  })
      #  return(list(error=error,fig=fig))
      #})
      
      #error_list[[i]]<-result$error
      #plot_history[[i]]<-result$fig
   #}
  #  cv_error = fun_cv_error(error_list)
  #}
  
  if(nrow(history_raw) >= 9 & nrow(history_raw) < 18){
    
    cat("Arima: Executing Hold-Out.\n")
    
    train <- history_raw[1:(nrow(history_raw)-3), ]
    test  <- history_raw[(nrow(history_raw)-3+1):nrow(history_raw), ]
    #xreg_train = train %>% select(-timestamp, -sales)
    #xreg_test = test %>% select(-timestamp)
    
    result <- tryCatch({
      model <- auto.arima(ts(train$sales), 
                         seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)  
      pred  <- forecast(model,  h = 3)%>%data.frame()
      list(model=model, pred=pred)
    },error=function(e){
    })
    
    model = result$model
    pred  = result$pred
    error = fun_error_measurement(pred$Point.Forecast, test$sales)
    #pred_fitted = model$fitted
    #plot_history = dygraph_plot_history(history, train, test, pred, pred_fitted,"Arima",0)
    cv_error = error
  }
  
  if(nrow(history_raw) < 9){
    
    cat("Arima: Executing Defaults.\n")
    cv_error = NA
  }
  
  # Train the model on the full datasets
  result <- tryCatch({
    model <- auto.arima(ts(history_raw$sales, frequency = 12),
                        seasonal = TRUE, stepwise = TRUE, approximation = TRUE, allowmean = TRUE, allowdrift = TRUE)  
    model_order <- model$arma
    pred <- forecast(model, h=4)%>%data.frame()
    list(model=model, pred=pred, model_order=model_order)
  },error=function(e){
  })
  model <- result[[1]]
  pred  <- result[[2]]
  model_order <- result[[3]]

  return(list(pred=pred$Point.Forecast))
}
