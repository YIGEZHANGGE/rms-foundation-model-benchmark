#!/usr/bin/env Rscript
suppressMessages(library(Biobase))
f <- "E:/rms_persister/data/target/E-TABM-1202.eSet.r"

eset <- NULL
tryCatch({ eset <- readRDS(gzfile(f, "rb")) }, error = function(e) {})
if (is.null(eset)) {
  e <- new.env()
  load(f, envir = e)
  objs <- ls(e)
  for (o in objs) {
    if (inherits(e[[o]], "ExpressionSet") || inherits(e[[o]], "eSet")) { eset <- e[[o]]; break }
  }
}
if (is.null(eset)) stop("could not read ExpressionSet")

cat("class:", class(eset), "\n")
cat("dims (features x samples):", nrow(eset), "x", ncol(eset), "\n")
cat("phenoData cols:", paste(colnames(pData(eset)), collapse=", "), "\n")
cat("featureData cols:", paste(colnames(fData(eset)), collapse=", "), "\n")
cat("phenoData head:\n")
print(head(pData(eset), 2))
cat("featureData head:\n")
print(head(fData(eset), 2))
cat("\nannotation:", annotation(eset), "\n")
cat("n features:", length(featureNames(eset)), "\n")
cat("feature names (first 8):", paste(head(featureNames(eset), 8), collapse=", "), "\n")
cat("n samples:", ncol(eset), "\n")
cat("sample names (first 5):", paste(head(sampleNames(eset), 5), collapse=", "), "\n")
