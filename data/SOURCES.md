# Input data and provenance

These files support reproduction from archived clinical and derived tables. They do not reconstruct the original raw omics analysis. The same digests appear in output/input_sha256.json.

| File | Bytes | SHA-256 |
|---|---:|---|
| brca_pam50.txt | 3168 | 61a45b2fc7cbfec3ffb37f540d2da1922fdab483143ca0bded369096360b9106 |
| causal_features_dataset.csv | 120523 | 9b2faf95523f7b5bdeade5b620ee5f9fca8026de370d8dd48545b22e1eee1be4 |
| clinical.tsv | 6989771 | 6db210fc25ed1493264c6ca6c3478477fc12c279bd2a13379183f78dcd209bba |
| follow_up.tsv | 6956758 | 78e81e84db3579474caf379e75e963eae200efad0f76134d7547854fe07d174c |
| ite_ready_dataset_v2.csv | 308233 | 34d00ee56d8619e07713aa22a4ee250db8c9208372904aacf96a8b3f71d074d1 |
| outcome.csv | 28407 | a0fabdff8aa7695ffa1f7f3a235a48ae87cc43e20458054c32b82ee5ca0631ee |
| pathology_detail.tsv | 538230 | ba60d984c13d175bce8558a1bd7ca3ec0cb5aabc626f37b403e98ad8af02633d |

clinical.tsv, follow_up.tsv and pathology_detail.tsv are open-access exports from the NCI Genomic Data Commons (TCGA-BRCA). outcome.csv is the derived survival table, reconciled against clinical.tsv by stage 03. brca_pam50.txt holds PAM50 calls. causal_features_dataset.csv and ite_ready_dataset_v2.csv are derived tables from the earlier multimodal feature pipeline; the earlier adjustment set is read from ite_ready_dataset_v2.csv. The original UCSC Xena exports and raw assay matrices are not included. See the README for scope.
