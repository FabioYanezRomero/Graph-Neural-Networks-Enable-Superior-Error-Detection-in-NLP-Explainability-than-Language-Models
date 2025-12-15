#!/usr/bin/env bash
set -euo pipefail

# Validate graphs for both SST-2 and AG News with default sizes

bash scripts/validate_sst2_graphs.sh
bash scripts/validate_ag_news_graphs.sh
