#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from ode_models import list_models, integrate_model, load_presets, run_preset

def parse_params(param_list: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in param_list:
        if "=" not in item:
            raise SystemExit(f"Invalid --param '{item}', expected key=value")
        k, v = item.split("=", 1)
        k = k.strip(); v = v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out

def save_csv(path: str | Path, t: np.ndarray, y: np.ndarray):
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"y{i}" for i in range(y.shape[1])])
        for ti, yi in zip(t, y):
            w.writerow([ti, *yi])

def cmd_list_models(_args):
    for m in list_models():
        print(m)

def cmd_list_presets(args):
    presets = load_presets(args.preset_file)
    for name, p in presets.items():
        print(f"{name:22s} model={p.get('model','?'):14s} tspan={p.get('tspan')} dt={p.get('dt')}")

def cmd_run(args):
    if args.preset:
        if not args.preset_file:
            raise SystemExit("Need --preset-file when using --preset")
        presets = load_presets(args.preset_file)
        if args.preset not in presets:
            raise SystemExit(f"Preset '{args.preset}' not found")
        t, y = run_preset(presets[args.preset])
        used = presets[args.preset]["model"]
    else:
        params = parse_params(args.param or [])
        y0 = [float(v) for v in args.y0]
        t, y = integrate_model(args.model, y0=y0, t0=args.t0, t1=args.t1, dt=args.dt,
                               backend=args.backend, stiff=args.stiff, rtol=args.rtol, atol=args.atol,
                               method=args.method, **params)
        used = args.model

    print(f"# model={used} steps={len(t)} dim={y.shape[1]} t0={t[0]:.6g} t1={t[-1]:.6g}")
    if args.out_csv:
        save_csv(args.out_csv, t, y)
        print(f"Saved CSV: {args.out_csv}")

def build_parser():
    p = argparse.ArgumentParser(prog="ode-lab", description="Tiny CLI for ODE Lab (v3).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("list-models"); s1.set_defaults(func=cmd_list_models)
    s2 = sub.add_parser("list-presets"); s2.add_argument("--preset-file", required=True); s2.set_defaults(func=cmd_list_presets)

    s3 = sub.add_parser("run")
    g = s3.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--preset")
    s3.add_argument("--preset-file")
    s3.add_argument("--y0", nargs="+", default=[])
    s3.add_argument("--t0", type=float, default=0.0)
    s3.add_argument("--t1", type=float, default=10.0)
    s3.add_argument("--dt", type=float, default=None)
    s3.add_argument("--backend", default="auto")
    s3.add_argument("--stiff", action="store_true")
    s3.add_argument("--rtol", type=float, default=1e-6)
    s3.add_argument("--atol", type=float, default=1e-9)
    s3.add_argument("--method", default=None)
    s3.add_argument("--param", action="append", default=[])
    s3.add_argument("--out-csv")
    s3.set_defaults(func=cmd_run)

    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
