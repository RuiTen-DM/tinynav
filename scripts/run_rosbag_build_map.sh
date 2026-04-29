#!/bin/bash
set -euo pipefail

rosbag_path=$(uv run hf download --repo-type dataset --cache-dir /tinynav UniflexAI/rosbag2_go2_looper)
map_save_path=/tinynav/output/map_go2_looper

# PerceptionNode is now embedded in build_map_node.py — no separate source process needed.
tmux new-session \; \
  split-window -h \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tinynav/core/build_map_node.py --map_save_path $map_save_path --bag_file $rosbag_path" C-m \; \
  select-pane -t 1 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m
