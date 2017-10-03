library(ggplot2)
library(reshape2)
cols <- c("red", 
          "blue", "green","purple",
          "grey0","turquoise3","yellow2",
          "aquamarine4","coral4","blueviolet",
          "darkgreen","darkslateblue","gray0")
datasets <- c("australian","vehicle","wbcd","sonar", "hillvalley","musk1", "arrhythmia","madelon","multiplefeatures") 
methods <- c("Exact","Rectangle")

ltys <- c(1,2,3,4,5,6,7)
pchs <- c(0,1,2,3,4,5,6)

simpleCap <- function(x) {
  paste(toupper(substring(x, 1,1)), substring(x, 2),
        sep="", collapse=" ")
}

type="Full"
for(dataset in datasets){
  title = simpleCap(dataset)
  dataDir = paste("Data",type,sep ="/")
  dataDir = paste(dataDir,dataset,sep = "/")
  
  methodFile = paste("Exact","txt",sep=".")
  methodDir = paste(dataDir,methodFile,sep="/")
  dataExact = read.csv(methodDir,header = FALSE,sep=",")
  
  methodFile = paste("Rectangle","txt",sep=".")
  methodDir = paste(dataDir,methodFile,sep="/")
  dataRect = read.csv(methodDir,header = FALSE,sep=",")
  
  ggplot() +
    geom_point(data=dataExact, aes(x=dataExact$V1,y=dataExact$V2,color="Exact"))+
    geom_line(data=dataExact, aes(x=dataExact$V1,y=dataExact$V2,color="Exact"))+
    geom_point(data=dataRect, aes(x=dataRect$V1,y=dataRect$V2,color="Rectangle"))+
    geom_line(data=dataRect, aes(x=dataRect$V1,y=dataRect$V2,color="Rectangle"))+
    ggtitle(title)+
    xlab('Nf') +
    ylab('Err')+
    labs(color="Methods")
  
  ggsave(paste("Graphs",paste(dataset,"png",sep="."),sep="/"))
}