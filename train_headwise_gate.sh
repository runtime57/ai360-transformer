#!/bin/bash
python3 train.py --config-name=train_tiny_stories writer.run_name="tiny_stories_headwise_gate" model=gpt_headwise_gate "$@"
