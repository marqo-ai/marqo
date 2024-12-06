#!/bin/bash
# This script is meant to be run at build time.
set -euo pipefail
set -x

mkdir -p /root/nltk_data/tokenizers
curl https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip -o /root/nltk_data/tokenizers/punkt_tab.zip
unzip /root/nltk_data/tokenizers/punkt_tab.zip -d /root/nltk_data/tokenizers/
rm /root/nltk_data/tokenizers/punkt_tab.zip
set +x