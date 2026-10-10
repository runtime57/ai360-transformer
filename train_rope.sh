#!/bin/bash
python3 train.py --config-name=train_rope writer.run_name="tiny_stories_rope" "$@"
