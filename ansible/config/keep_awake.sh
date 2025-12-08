#!/bin/bash

echo "Keeping system awake... Press Ctrl+C to stop."

# Block sleep while this script runs
systemd-inhibit --what=handle-lid-switch:sleep --who="keep-awake" --why="Prevent system sleep" bash -c "
    while true; do
        sleep 30
    done
"
