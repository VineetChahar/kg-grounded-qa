#!/usr/bin/env bash
# Full reproduction pipeline: clone sources -> build KG -> embed -> train GNN -> eval set -> run eval.
# Requires: python venv with requirements.txt installed, `ollama serve` running with
# qwen2.5:3b-instruct (or another local model) pulled.
set -euo pipefail
cd "$(dirname "$0")/.."

REPOS=(click flask itsdangerous jinja markupsafe werkzeug)
mkdir -p data/repos
for repo in "${REPOS[@]}"; do
  if [ ! -d "data/repos/$repo" ]; then
    echo "cloning $repo..."
    git clone --depth 1 -q "https://github.com/pallets/${repo}.git" "data/repos/$repo"
  fi
done

python -m kg_grounded_qa.graph_build
python -m kg_grounded_qa.embeddings
python -m kg_grounded_qa.gnn
python -m kg_grounded_qa.build_eval_set
python -m kg_grounded_qa.run_eval
