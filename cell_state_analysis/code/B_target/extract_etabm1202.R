#!/usr/bin/env Rscript
suppressMessages({library(Biobase); library(affy)})
f <- "E:/rms_persister/data/target/E-TABM-1202.eSet.r"
eset <- NULL
tryCatch({ eset <- readRDS(gzfile(f, "rb")) }, error=function(e) NULL)
if (is.null(eset)) { e <- new.env(); load(f, envir=e); for(o in ls(e)) if(inherits(e[[o]],"eSet")){eset<-e[[o]];break} }

expr <- exprs(eset)
pd <- pData(eset)
cat("expr dims (features x samples):", nrow(expr), "x", ncol(expr), "\n")
cat("pData dims:", nrow(pd), "x", ncol(pd), "\n")

# identify survival-related columns
keys <- c("Status", "survival_time", "Age", "Histology", "IRS", "Stage", "Clinical", "COG", "risk")
surv_cols <- colnames(pd)[grepl(paste(keys, collapse="|"), colnames(pd), ignore.case=TRUE)]
cat("survival/clinical columns:\n")
for (cc in surv_cols) cat("  ", cc, "\n")

# write expression + pData as TSV
write.table(expr, "E:/rms_persister/data/target/E-TABM-1202.expr.tsv", sep="\t", quote=FALSE, row.names=TRUE, col.names=TRUE)
write.table(pd, "E:/rms_persister/data/target/E-TABM-1202.pheno.tsv", sep="\t", quote=FALSE, row.names=TRUE, col.names=TRUE)
cat("written expr + pheno TSV\n")

# summary of survival status
if ("Factor.Value..Status" %in% colnames(pd)) {
  print(table(pd[["Factor.Value..Status"]], useNA="ifany"))
}
