args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
src <- args[1]; out <- args[2]
e <- new.env()
load(file.path(src,'mat_count.RData'), envir=e)
load(file.path(src,'df_samples.RData'), envir=e)
x <- e$mat_count; m <- as.data.frame(e$df_samples)
stopifnot(is.matrix(x), all(is.finite(x)), all(x>=0), all(x==floor(x)),
          !anyDuplicated(colnames(x)), !anyDuplicated(rownames(x)),
          !anyDuplicated(m$SampleID), setequal(colnames(x),m$SampleID))
write.table(data.frame(taxon_id=rownames(x),x,check.names=FALSE),
            file.path(out,'farmm_counts.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
write.table(m, file.path(out,'farmm_samples.tsv'), sep='\t',row.names=FALSE,quote=FALSE,na='')
