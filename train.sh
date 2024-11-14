#!/bin/bash

# Define lists of possible values for variables
dataset_names=('hwu64' 'SNIPS' 'CLINC' 'BANKING' 'FB_en' 'mpqa' 'yahoo')
enc_types=('roberta' 'actual_bert' 'distilbert' 'electra')
dom_nondoms=('dom' 'nondom')
coarse_fines=('coarse' 'fine')

# Iterate over all combinations
for dataset_name in "${dataset_names[@]}"; do
    for enc_type in "${enc_types[@]}"; do
        for dom_nondom in "${dom_nondoms[@]}"; do
            for coarse_fine in "${coarse_fines[@]}"; do
                echo "Running command for dataset_name: $dataset_name, enc_type: $enc_type, dom_nondom: $dom_nondom, coarse_fine: $coarse_fine"
                python3 train.py --gpu_id 0 --src_folder "${dataset_name}/" --trg_folder "${dataset_name}/PASTE_${enc_type}_${dom_nondom}_${coarse_fine}_intent" --bert_mode gen --gen_direct af --l2 y --enc_type "$enc_type" --coarse_fine "${coarse_fine}" --dom_nondom ${dom_nondom} > "${dataset_name}_PASTE_${enc_type}_${dom_nondom}_${coarse_fine}_intent_POINTERS.txt"
            done
        done
    done
done
