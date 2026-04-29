#!/bin/bash
ground_plane_args=""
if [[ "${GROUND_PLANE_CONSTRAINT:-0}" == "1" ]]; then
  ground_plane_args="--ground_plane_constraint"
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tinynav/core/perception_node.py ${ground_plane_args}" C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys 'uv run python /tinynav/tinynav/platforms/lekiwi_control.py' C-m \; \
  select-pane -t 3 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 4 \; send-keys 'bash' C-m
