#!/bin/bash
python3 train.py --config-name=train_alibi writer.run_name="tiny_stories_alibi" "$@"
