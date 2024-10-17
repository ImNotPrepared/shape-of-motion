#! /usr/bin/env python
# rm -rf ./outdir
PORT=$1
python run_rendering_pc.py --work-dir ./outdir_true_4 --port "$PORT"