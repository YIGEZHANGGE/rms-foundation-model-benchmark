#!/usr/bin/env Rscript
suppressMessages({library(Biobase); library(affy)})
f <- "E:/rms_persister/data/target/E-TABM-1202.eSet.r"
raw <- NULL
tryCatch({ raw <- readRDS(gzfile(f, "rb")) }, error=function(e) NULL)
if (is.null(raw)) { e <- new.env(); load(f, envir=e); for(o in ls(e)) if(inherits(e[[o]],"AffyBatch")){raw<-e[[o]];break} }
if (is.null(raw)) stop("no AffyBatch")
cat("raw dims:", nrow(raw), "x", ncol(raw), "\n")

eset <- rma(raw)   # normalize -> ExpressionSet (54675 probe sets)
expr <- exprs(eset)
cat("RMA done. expr dims:", nrow(expr), "x", ncol(expr), "\n")

write.table(expr, "E:/rms_persister/data/target/E-TABM-1202.rma_expr.tsv", sep="\t", quote=FALSE, row.names=TRUE, col.names=TRUE)
cat("written rma_expr TSV\n")
