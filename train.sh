#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
!python train.py --config-name=train \
    writer=cometml \
    writer.run_name=hpc_mtp \
    datasets.train.tokenization_workers=4 \
    datasets.val.tokenization_workers=4 \
    trainer.use_amp=True \
    trainer.override=True \
    "$@"

