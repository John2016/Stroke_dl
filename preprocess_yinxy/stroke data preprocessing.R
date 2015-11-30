library(dplyr)
library(xlsx)
##read the stroke data
info <- read.csv("D:/data/stroke/新增中心点去0去5基本库20130606_1.csv")
info <- as.matrix(info)
##read the items of vessel examination
vessel_examine_names <- read.xlsx("H:/Stroke_Luo/From Luo/vessel_examin_names_910.xlsx",1,header=F)
vessel_examine_names <- as.matrix(vessel_examine_names)
##count the valid data for each vessel examination item
k <- 0
count <- matrix(0,1,length(vessel_examine_names))
col_names <- colnames(info)
for (i in vessel_examine_names)
{
  count[(k+1)] <- length(which(info[,which(col_names==i,T)]!="NA",T))
  k <- k+1
}
write.csv(count,"valid data number for each item.csv",row.names=F)
##find the patients whose information are relatively complete by calculating the intersection of those the valid data exceeds 50000
namesExceed58800 <- vessel_examine_names[which(count>50000),T]
interID <- info[,1]
for (i in c(2:length(namesExceed58800)))
{
  ID <- info[which(info[,which(col_names==namesExceed58800[i],T)]!="NA",T),1]
  interID <- intersect(interID,ID)
}
##data without those valid data less than 50000, i.e. with only 70 examination items
data <- info[which(info[,1]%in%interID,T),which(col_names%in%namesExceed58800,T)] 
colnames(data) <- col_names[which(col_names%in%namesExceed58800,T)]
write.csv(data,"data with 70 features.csv",row.names=F)
##data1 with all the 86 vessel examination
data1 <- info[which(info[,1]%in%interID,T),which(col_names%in%vessel_examine_names,T)]
colnames(data1) <- col_names[which(col_names%in%vessel_examine_names,T)]
write.csv(data1,"data with 86 features.csv",row.names=F)
##check the variation range in each item
vari <- matrix(0,(length(colnames(data1))-1),2)
vari[,1] <- colnames(data1)[-1]
for (i in c(2:ncol(data1)))
{
  content <- unique(data1[which(data1[,i]!="NA",T),i])
  if (length(content)==0)
  {
    vari[(i-1),2] <- NA
  }
  else if (length(content)==1)
  {
    vari[(i-1),2] <- content
  }
  else if (length(content)==2)
  {
    vari[(i-1),2] <- paste(content[1],content[2],sep=",")
  }
  else 
  {
    cont <- paste(content[1],content[2],sep=",")
    for (j in c(3:length(content)))
    {
      cont <- paste(cont,content[j],sep=",")
    }
    vari[(i-1),2] <- cont
  }
}
write.csv(vari,"range of each examination.csv",row.names=F)
##data without replicates
unidata <- unique(data[,-1])
write.csv(unidata,"unique data of the examination.csv",row.names=F)
unidata1 <- unique(data)
write.csv(unidata1,"unique data of the examination and diagnose.csv",row.names=F)
