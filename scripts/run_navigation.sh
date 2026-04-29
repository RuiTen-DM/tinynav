#!/bin/bash

map_path="PATH/TO/MAP"
ground_plane_args=""
if [[ "${GROUND_PLANE_CONSTRAINT:-0}" == "1" ]]; then
  ground_plane_args="--ground_plane_constraint"
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 3 \; split-window -v \; \
  select-pane -t 4 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tinynav/core/perception_node.py ${ground_plane_args}" C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path" C-m \; \
  select-pane -t 3 \; send-keys "uv run python /tinynav/tinynav/platforms/cmd_vel_control.py" C-m \; \
  select-pane -t 4 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 5 \; send-keys "sleep 3 && uv run python /tinynav/tool/pub_pois.py --tinynav_map_path $map_path" C-m
