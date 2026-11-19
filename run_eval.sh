#!/usr/bin/env bash
# Full evaluation, in GPU-friendly order. Usage: bash run_eval.sh
set -u
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe
# wait for golden-set generation if it is still running
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*eval synth*' }) { exit 0 } else { exit 1 }"; do sleep 20; done
echo "== golden set";      cat eval/golden/golden_v1.card.json
echo "== gate";            $PY -m labellens.cli eval gate       > eval_gate.log 2>&1;       echo "exit $?"
echo "== generation";      $PY -m labellens.cli eval generation --n 120 > eval_generation.log 2>&1; echo "exit $?"
echo "== raw embeddings";  $PY -m labellens.cli ingest --variant raw > data_embed_raw.log 2>&1; echo "exit $?"
echo "== retrieval";       $PY -m labellens.cli eval retrieval --update-baseline > eval_retrieval.log 2>&1; echo "exit $?"
echo "== done"
