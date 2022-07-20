#!/bin/bash
# usage: ./get_gtex.sh
# Purpose: Retrieve gtex data
# 
# Exported environmental variables:
#   RAW_DATA_ROOT - location to store raw data
# 
# Benchmark: ~30 sec

RAW_DATA_ROOT=dist

# Initialize directories
mkdir -p ${RAW_DATA_ROOT}/gtex
# Retrieve public data and main model data for pre-training
pushd ${RAW_DATA_ROOT}/gtex
wget https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
wget https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
popd

