#!/bin/bash
# Disable screensaver + screen blanking + DPMS + system idle actions

# Cinnamon / GNOME settings
gsettings set org.cinnamon.desktop.screensaver lock-enabled false
gsettings set org.cinnamon.desktop.screensaver idle-activation-enabled false

# X11 DPMS and screensaver off
xset -dpms
xset s off
xset s noblank

# XFCE / power manager (if present)
if command -v xfconf-query >/dev/null 2>&1; then
    xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/blank-on-ac -s 0 2>/dev/null
    xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/blank-on-battery -s 0 2>/dev/null
fi

# Systemd idle actions off (only needs to run once, but harmless if repeated)
sudo sed -i 's/^#\?IdleAction=.*/IdleAction=ignore/' /etc/systemd/logind.conf
sudo sed -i 's/^#\?IdleActionSec=.*/IdleActionSec=0/' /etc/systemd/logind.conf
sudo systemctl restart systemd-logind
