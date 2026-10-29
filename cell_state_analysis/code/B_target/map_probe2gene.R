#!/usr/bin/env Rscript
suppressMessages({library(AnnotationDbi); library(hgu133plus2.db)})

expr <- read.delim("E:/rms_persister/data/target/E-TABM-1202.rma_expr.tsv", row.names=1, check.names=FALSE)
probes <- rownames(expr)
cat("probes:", length(probes), "| samples:", ncol(expr), "\n")

sym <- mapIds(hgu133plus2.db, keys=probes, column="SYMBOL", keytype="PROBEID", multiVals="first")
keep <- !is.na(sym)
expr <- expr[keep, ]; sym <- sym[keep]
cat("probes with gene symbol:", sum(keep), "\n")

# collapse multi-probe genes: keep the probe with highest mean expression
means <- rowMeans(expr)
ord <- order(means, decreasing=TRUE)
expr <- expr[ord, ]; sym <- sym[ord]
first_idx <- !duplicated(sym)
expr_gene <- expr[first_idx, ]
rownames(expr_gene) <- sym[first_idx]
cat("gene-level matrix:", nrow(expr_gene), "genes x", ncol(expr_gene), "samples\n")

write.table(expr_gene, "E:/rms_persister/data/target/E-TABM-1202.gene_expr.tsv", sep="\t", quote=FALSE, row.names=TRUE, col.names=TRUE)
cat("written gene_expr TSV\n")
