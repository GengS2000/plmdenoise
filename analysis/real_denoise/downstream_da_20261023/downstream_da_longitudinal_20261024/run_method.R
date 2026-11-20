args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=4) stop('run_method.R PACKAGE INPUT_FOLDER METHOD OUTPUT_FOLDER')
package <- args[1]; input <- args[2]; method <- args[3]; output <- args[4]
source(file.path(package,'da_engines.R'))
dir.create(output,recursive=TRUE,showWarnings=FALSE)
set.seed(20261023)
meta <- read.csv(file.path(input,'metadata.csv'),check.names=FALSE,stringsAsFactors=FALSE)
tab <- read.csv(file.path(input,'proportions.csv'),check.names=FALSE,stringsAsFactors=FALSE)
stopifnot(!anyDuplicated(meta$sample_id),!anyDuplicated(tab$sample_id),identical(meta$sample_id,tab$sample_id))
p <- as.matrix(tab[,-1,drop=FALSE]); storage.mode(p) <- 'double'
rownames(p) <- tab$sample_id; rownames(meta) <- meta$sample_id
stopifnot(all(is.finite(p)),all(p>=0),all(abs(rowSums(p)-1)<1e-7))
DA_NUISANCE <- grep('^C[0-9]+$',names(meta),value=TRUE)
meta$Xcase <- as.numeric(meta$X); meta$participant <- factor(meta$participant)
design <- model.matrix(reformulate(c('Xcase',DA_NUISANCE)),meta)
stopifnot(qr(design)$rank==ncol(design),nrow(design)>ncol(design)+4)
pkg <- switch(method,linda='MicrobiomeStat',maaslin2='Maaslin2',glmmpql='MASS',zigmm='NBZIMM')
if(is.null(pkg)||!requireNamespace(pkg,quietly=TRUE)) stop('Missing package: ',pkg)
pseudocount <- 1e-6
run <- da_capture(switch(method,linda=da_linda(p,meta,output,pseudocount),maaslin2=da_maaslin2(p,meta,output,pseudocount),
    glmmpql=da_per_taxon(p,meta,'glmmpql',pseudocount),zigmm=da_per_taxon(p,meta,'zigmm',pseudocount)))
if(inherits(run$value,'error')) {
    result<-da_empty(colnames(p));result$fit_status<-'method_error';result$fit_message<-conditionMessage(run$value)
} else result<-da_apply_warnings(run$value,run$warnings,scope=if(method %in% c('linda','maaslin2')) 'run' else 'taxon')
result$zero_fraction <- colMeans(p==0)[match(result$taxon,colnames(p))]
result$response_scale <- switch(method,linda='internal_CLR_log2',maaslin2='log2_proportion_plus_1e-6',glmmpql='log2_proportion_plus_1e-6',zigmm='asin_sqrt_proportion')
write.csv(result,file.path(output,'engine_results.csv'),row.names=FALSE,na='NA')
writeLines(c(capture.output(sessionInfo()),paste('formula:',paste(c('Xcase',DA_NUISANCE),collapse=' + '),'+ (1|participant)')),file.path(output,'sessionInfo.txt'))
if(!any(is.finite(result$pvalue))) quit(status=3)
