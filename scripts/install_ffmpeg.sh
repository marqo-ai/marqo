#!/bin/bash
# This script is meant to be run at buildtime to install `ffmpeg`

dnf config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo
dnf install -y ffmpeg
