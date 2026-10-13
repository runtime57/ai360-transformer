#!/bin/bash
python3 train.py --config-name=train_rope writer.run_name="tiny_stories_rope_elementwise_gate" model=gpt_rope_elementwise_gate "$@"
