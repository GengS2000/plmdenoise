# PLMdenoise manuscript analysis scripts

Source archive for the manuscript. Original relative paths are retained. Data, checkpoints, result tables, frozen manifests and external comparator implementations are not bundled.

## Directory guide
- code_v3_1: m4_flex core, simulation, benchmark adapters, differential abundance, PCA and decomposition.
- real_denoise: cohort preparation, frozen generalization/window source, posterior exports and differential abundance.
- 实际数据分析的结果/预测分型: ridge prediction, bootstrap ASV ranking and ablation.
- simulation_results: comparison plots and frozen decomposition source.
- Make_Figure: manuscript figure scripts.

## Manuscript entry points
- Figure 2: code_v3_1/run_validation.py and compare_methods.py; simulation_results/comparison/figure plotting scripts.
- Figure 3: code_v3_1/run_da.py, run_dr.py, run_decomposition.py; Make_Figure/第三步效应分解+应用/prepare_simulation_examples.py.
- Figure 4: Make_Figure/Casestudyfig/build_case.py, using revised FARMM/SOIL result tables.
- Tables 1 and 2: 实际数据分析的结果/预测分型/predict_full_microbiome.py.
- Figure 5: bootstrap_m2_asv_norms.py, run_bootstrap_top10_ablation.py and Make_Figure/牙龈炎应用/plot_asv_ablation.py.

## Execution requirements
Python dependency lists are under code_v3_1 and real_denoise. Some workflows also require R packages, MATLAB/Octave and external comparison implementations. Prediction requires organized window exports and RFdata.csv at paths declared in predict_phenotype.py. Figures require archived result tables. Frozen runners require matching original plan.json, PLAN.sha256 and input payloads.

Historical path conventions are preserved; review input/output roots before execution. This is a staged source archive, not a tested one-command reproduction package.

## Interpretation
Window denoising is temporally truncated and transductive. Ridge classification uses repeated nested subject-level validation. Conditional intervals bootstrap fixed out-of-fold scores. Fixed bootstrap Top-10 panels use the same cohort for selection before outer validation and are exploratory. The fixed-seven-ASV script retains prediction/ablation only.

## Provenance
SOURCE_MANIFEST.csv records workspace-relative source paths and original/staged SHA-256 hashes. Original files were not edited. Third-party authorship and notices remain applicable. A repository license has not been assigned.
