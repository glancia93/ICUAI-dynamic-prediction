###============== Packages ==================##
library(survminer)
library(survival)
library(mstate)
require(MASS)
require(dynpred)
library(mstate)
library(survival)
library(survminer)
library(dplyr)
library(coefplot)
library(ROCR)
library(permutations)
library(ggcorrplot)
library(colorspace)
library(ggplot2)
library(boot)
library(pec)
# for tidy
Packages <- c( "pec"  ,"survAUC") # load multiple packages



###=======================================     Functions   ========================================####
#######################################################################################################
############################################################################################################################
######  To calculate conditional CIF #####
############################################################################################################################


CIFpredict_linear<- function(linear,BH0, s_land,w)
{ #linear<- linear_pred
  pred<-rep(0,length(linear))
  
  horizon<-s_land+w
  BHsw<-evalstep(BH0$time,BH0$hazard,horizon,subst=0) #evaluate Hazard at step function
  BHs<-evalstep(BH0$time,BH0$hazard,s_land,subst=0)
  pred<-1-exp((-BHsw+BHs)*exp(linear))
  
  return(pred)           
}

#######################################################



#-----------------------------------------------------------------------------------
Fwpredict <- function(bet, sf, newdata, tt)
  {
    nt <- length(tt)
    sf <- data.frame(time=sf$time,surv=sf$surv,Haz=-log(sf$surv))
    Fw <- data.frame(time=tt,Fw=NA)
    for (i in 1:nt) {
      sfi <- sf
      tti <- tt[i]
      sfi$Haz <- sfi$Haz * exp(newdata[1]*bet[1] + newdata[2]*bet[2]*f1(tt[i]) + newdata[2]*bet[3]*f2(tt[i]) + 
                                 newdata[2]*bet[4]*f3(tt[i]) + bet[5]*g1(tt[i]) + bet[6]*g2(tt[i])) ### need to customize this model form according to the landmark PSH supermodel
      tmp <- evalstep(sfi$time,sfi$Haz,c(tti,tti+w),subst=0)
      Fw$Fw[i] <- 1-exp(-(tmp[2]-tmp[1]))
    }
    return(Fw)
  }


##################################################
##################################################
# function for AUC per landmark
##################################################
##################################################



##### AUC for validation#######
AUC_land_covariate<-function(data,w,covariate, flag= FALSE)
{wdays<-w
#LMdata<-read.csv(paste0("landmark_FINALE_06_10_21_w",wdays,".csv"),sep=",")
LMdata<-data
count<-LMdata$count
LM_PSH_super_NN_no <- coxph(as.formula(paste0("Surv(Tstart,Tstop,Failure==1)~",paste(covariate, collapse=" + "),"+cluster(id_cr)")), 
                            data=LMdata, ties="efron", method="breslow",x = TRUE)


if (flag) {
  print(summary(LM_PSH_super_NN_no))
}

# data set with predictions
stot<-unique(LMdata$s_day)[1:25]
landmark<-LMdata$s_day
horizon<-LMdata$s_day+w
id<-LMdata$id
failure<-LMdata$failure
count<-LMdata$count
linear_pred<-predict(LM_PSH_super_NN_no, type="lp")

BH0<-basehaz(LM_PSH_super_NN_no)
prediction_AI<-as.vector(CIFpredict_linear(linear_pred,BH0, landmark,w))
failure<-as.factor(ifelse(LMdata$failure==1,1,0))
data_pred_time_NN<-data.frame(id=id,landmark=landmark,horizon=horizon,failure=failure,prediction_AI=prediction_AI,count=count)
data_pred_time_NN<-data_pred_time_NN[data_pred_time_NN$count==1,]
data_pred_time_NN$prediction_AI<-as.vector(data_pred_time_NN$prediction_AI)
#data_pred_time_NN$failure<-as.factor(ifelse(data_pred_time_NN$failure==1,1,0))

AUCs<-rep(0,25)
for (i in 1:25)
{ #print(i)
  prediction_AI_i<-data_pred_time_NN$prediction[data_pred_time_NN$landmark==stot[i]]
  failure_i<-data_pred_time_NN$failure[data_pred_time_NN$landmark==stot[i]]
  pred_AU_time_NN <- prediction(prediction_AI_i, failure_i)
  auc.perf_time_NN<- performance(pred_AU_time_NN, measure = "auc")
  AUCs[i]<-as.vector(unlist(auc.perf_time_NN@y.values))
}

AUC_data<-data.frame(width=rep(wdays,25),s_day=stot,horizon=w+stot,AUC=AUCs)
return(AUC_data)
}


#####
coxph_summary<-function(data,w,ratio,covariate)
{  
  #ratio: ratio of splitting
  n_id<-length(unique(data$ReAd_pat))
  id_total<-unique(data$ReAd_pat)
  nn<-round(ratio*n_id)
  id_sampled<-sample(id_total,nn)
  id_validation<-id_total[!id_total%in%id_sampled]
  wdays<-w
  #LMdata<-read.csv(paste0("landmark_FINALE_NEWW_w",wdays,".csv"),sep=",")
  LMdata<-data[data$ReAd_pat%in%id_sampled,]
  data_validation<-data[data$ReAd_pat%in%id_validation,]
  count<-LMdata$count
  LM_PSH_super_NN_no <- coxph(as.formula(paste0("Surv(Tstart,Tstop,Failure==1)~",paste(covariate, collapse=" + "),"+cluster(id_cr)")), 
                              data=LMdata, ties="efron", method="breslow",x = TRUE)
  
  print(summary(LM_PSH_super_NN_no))
  return()
}
  


### validation routine #K fold validation
AUC_land_validation<-function(data,w,ratio,covariate, flag= FALSE)
{  
   #ratio: ratio of splitting
  n_id<-length(unique(data$ReAd_pat))
  id_total<-unique(data$ReAd_pat)
  nn<-round(ratio*n_id)
  id_sampled<-sample(id_total,nn)
  id_validation<-id_total[!id_total%in%id_sampled]
  wdays<-w
#LMdata<-read.csv(paste0("landmark_FINALE_NEWW_w",wdays,".csv"),sep=",")
LMdata<-data[data$ReAd_pat%in%id_sampled,]
data_validation<-data[data$ReAd_pat%in%id_validation,]
count<-LMdata$count
LM_PSH_super_NN_no <- coxph(as.formula(paste0("Surv(Tstart,Tstop,Failure==1)~",paste(covariate, collapse=" + "),"+cluster(id_cr)")), 
                            data=LMdata, ties="efron", method="breslow",x = TRUE)

if(flag){
  print(summary(LM_PSH_super_NN_no))
}

# only for getting the values of the linear predictors
LM_PSH_super_NN_validation<-coxph(as.formula(paste0("Surv(Tstart,Tstop,Failure==1)~",paste(covariate, collapse=" + "),"+cluster(id_cr)")), 
                                data=data_validation, ties="efron", method="breslow",x = TRUE)

# data set with predictions
stot<-unique(data_validation$s_day)[1:25]
landmark<-data_validation$s_day
horizon<-data_validation$s_day+w
id<-data_validation$id
failure<-data_validation$failure
count<-data_validation$count
linear_pred<-predict(LM_PSH_super_NN_no, type="lp", newdata=data_validation)

BH0<-basehaz(LM_PSH_super_NN_no)
prediction_AI<-as.vector(CIFpredict_linear(linear_pred,BH0, landmark,w))
failure<-as.factor(ifelse(data_validation$failure==1,1,0))
data_pred_time_NN<-data.frame(id=id,landmark=landmark,horizon=horizon,failure=failure,prediction_AI=prediction_AI,count=count)
data_pred_time_NN<-data_pred_time_NN[data_pred_time_NN$count==1,]
data_pred_time_NN$prediction_AI<-as.vector(data_pred_time_NN$prediction_AI)
#data_pred_time_NN$failure<-as.factor(ifelse(data_pred_time_NN$failure==1,1,0))

AUCs<-rep(0,25)
for (i in 1:25)
{ #print(i)
  prediction_AI_i<-data_pred_time_NN$prediction[data_pred_time_NN$landmark==stot[i]]
  #print(prediction_AI_i)
  failure_i<-data_pred_time_NN$failure[data_pred_time_NN$landmark==stot[i]]
  #print(failure_i)
  pred_AU_time_NN <- prediction(prediction_AI_i, failure_i)
  auc.perf_time_NN<- performance(pred_AU_time_NN, measure = "auc")
  AUCs[i]<-as.vector(unlist(auc.perf_time_NN@y.values))
}

AUC_data<-data.frame(width=rep(wdays,25),s_day=stot,horizon=w+stot,AUC=AUCs)
return(AUC_data)
}
