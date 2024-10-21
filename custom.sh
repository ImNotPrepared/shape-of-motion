#! /usr/bin/env python
# rm -rf ./outdir
WORK_DIR=$1
PORT=$2
python run_rendering.py --work-dir "$WORK_DIR" --port "$PORT" 