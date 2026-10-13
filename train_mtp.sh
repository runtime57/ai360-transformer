#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python3 train.py --config-name=train_mtp \
    writer.run_name=hpc_mtp \
    datasets.train.tokenization_workers=4 \
    datasets.val.tokenization_workers=4 \
    trainer.override=True \
    "$@"
