library("DESeq2")
library("stringr")

# Authors: Pauline Hiort
# Date: 2021_07_22
# R version: 3.6

##### Application of DESeq2 to normalize the RM scores of FlexiQuant-LF #####

##### read command line arguments
myArgs <- commandArgs(trailingOnly = TRUE)

input_file = myArgs[1]
coldata_file = myArgs[2]

##### read coldata with sample groups
coldata = read.csv(coldata_file, sep=",", header=TRUE)

##### read RM scores and reformat to one index/rownames columns
df = read.csv(input_file, sep=",", header=TRUE)
df$index = paste0(df$ProteinID,",",df$PeptideID)
rownames(df) <- df$index
df = subset(df, select = -c(ProteinID, PeptideID, index))
##### remove peptides with missing values
df = na.omit(df)
row_ids = c(rownames(df))

##### reformat RM score to integers with 6 numbers after the decimal point
cts = data.frame(apply(df, 2, as.numeric))
cts = cts*100000
cts = data.frame(apply(cts, 2, as.integer))
rownames(cts) = row_ids

##### DESeq2 normalization over the sample groups
normalized_counts2 = data.frame(dummy = integer(length(row_ids)))
rownames(normalized_counts2) = row_ids

for (elem in c(levels(coldata$Group))){
  coldata2 = coldata[coldata$Group == elem, ]
  cts2 = cts[,coldata2$Sample]

  ##### create DESeq data structure
  dds = DESeqDataSetFromMatrix(countData=cts2, colData=coldata2, design=~1)

  ##### calculate size factors for normalization
  dds = estimateSizeFactors(dds)
  
  ##### normalize the counts
  normalized_counts2 = cbind(normalized_counts2, counts(dds, normalized=T))
}

normalized_counts2 = subset(normalized_counts2, select = -c(dummy))
normalized_counts2 = normalized_counts2/100000

##### create output table
header = as.data.frame(as.matrix(t(c("ProteinID,PeptideID", str_replace_all(colnames(cts), 'X', '')))))
rownames(header) = header$V1
header = header[,2:(ncol(normalized_counts2)+1)]
colnames(header) = colnames(normalized_counts2)

output = rbind(normalized_counts2, header)
output = output[c(nrow(output), 1:nrow(output)-1),]

##### write to output file
output_file = str_replace(input_file, ".csv", "_DESeq.csv")
write.table(output, file=output_file, sep=",", quote=F, col.names=FALSE)