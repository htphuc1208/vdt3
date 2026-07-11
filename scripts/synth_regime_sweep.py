"""Regime sweep: where does each RCA mechanism win on synthetic telecom incidents?

Sweeps the two decisive knobs:
  * victim_amp           : downstream victims louder than the root (>1 = symptom
                           amplification, the real-telecom alarm-flood regime).
  * prop_delay / resolution ratio : propagation delay per hop relative to the
                           sampling interval (<1 = onset ordering collapses,
                           the OpenRCA 1-min regime; >1 = onsets are separable).

For each regime point it generates N labelled cases and reports Hit@1 / Hit@3 for
the four mechanisms. The output is an operating-regime map: it predicts which
mechanism to trust on real data given its resolution and amplification, and
explains why de-collapse won on OpenRCA (concentrated signal, sub-resolution
propagation) while topology/temporal should win in alarm-flood telecom.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from telco_mas.synth.telco_rca_sim import build_topology, generate_case
from telco_mas.synth.methods import METHODS


def _hit(ranked, root, k):
    return int(root in ranked[:k])


def evaluate(*, n_cases, resolution_s, prop_delay_s, victim_amp, noise_rate, seed):
    """Return Hit@1 overall and on the propagating subset (where mechanisms differ)."""
    rng = random.Random(seed)
    topo = build_topology(rng)
    hit1 = {m: 0 for m in METHODS}
    prop_hit1 = {m: 0 for m in METHODS}
    prop_n = 0
    for _ in range(n_cases):
        case = generate_case(topo, rng, resolution_s=resolution_s, prop_delay_s=prop_delay_s,
                             victim_amp=victim_amp, distractor_rate=noise_rate)
        prop = case.propagating
        prop_n += int(prop)
        for name, fn in METHODS.items():
            h = _hit(fn(case), case.root, 1)
            hit1[name] += h
            if prop:
                prop_hit1[name] += h
    overall = {m: round(hit1[m] / n_cases, 3) for m in METHODS}
    prop = {m: round(prop_hit1[m] / prop_n, 3) if prop_n else 0.0 for m in METHODS}
    return overall, prop, round(prop_n / n_cases, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="cases per regime point")
    ap.add_argument("--resolution", type=float, default=60.0, help="sampling interval (s)")
    ap.add_argument("--amps", default="1.0,1.5,2.5")
    ap.add_argument("--ratios", default="0.2,0.5,1,2,5", help="prop_delay / resolution")
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/synth_regime_sweep.json")
    args = ap.parse_args(argv)

    amps = [float(a) for a in args.amps.split(",") if a.strip()]
    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]
    grid = []
    ms = list(METHODS)
    print(f"n={args.n}/point  onset-noise=resolution={args.resolution}s  distractor={args.noise}")
    print("Hit@1 on PROPAGATING faults (root fans out to victims — where mechanisms differ):\n")
    header = f"{'amp':>4} {'delay/noise':>11} | " + " ".join(f"{m[:11]:>11}" for m in ms) + "   %prop"
    for amp in amps:
        print(header)
        print("-" * len(header))
        for ratio in ratios:
            overall, prop, frac = evaluate(n_cases=args.n, resolution_s=args.resolution,
                                           prop_delay_s=ratio * args.resolution, victim_amp=amp,
                                           noise_rate=args.noise, seed=args.seed)
            row = f"{amp:>4} {ratio:>11} | " + " ".join(f"{prop[m]:>11}" for m in ms) + f"   {frac}"
            print(row)
            grid.append({"victim_amp": amp, "delay_over_onset_noise": ratio,
                         "prop_delay_s": ratio * args.resolution,
                         "hit1_overall": overall, "hit1_propagating": prop, "frac_propagating": frac})
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "meta": {"n_per_point": args.n, "resolution_s": args.resolution,
                 "noise": args.noise, "methods": ms}, "grid": grid}, indent=2))
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
