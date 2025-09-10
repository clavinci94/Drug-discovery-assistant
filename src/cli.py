import os, sys, subprocess

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def _run(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH",".")
    subprocess.check_call(cmd, cwd=ROOT, shell=True, env=env)

def phase1():
    target = os.environ.get("TID","CHEMBL279")
    limit  = os.environ.get("LIMIT","600")
    _run(f'./scripts/run_phase1_preprocessing.sh {target} {limit}')

def ui():
    _run('python scripts/app_gradio.py')

def optimize():
    seeds = os.environ.get("SEEDS","CC(=O)Oc1ccccc1C(=O)O").split(",")
    seeds = " ".join([f'"{s.strip()}"' for s in seeds if s.strip()])
    _run(f'./scripts/optimize_quick.sh {seeds}')
