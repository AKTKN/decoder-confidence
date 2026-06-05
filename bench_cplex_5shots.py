"""Benchmark: CPLEX ILP logical gap on BB [[144,12,12]] d=12, phenomenological p=0.03, 5 shots."""
import time
import numpy as np
import stim

CIRCUIT_PATH = "circuits/bivariate_bicycle_code_Z/code=bivariate_bicycle_code_Z,d=12,rounds=12,noisemodel=phenomenological,p=0.03,use_both=False.stim"
NUM_SHOTS = 5
SEED = 42

circuit = stim.Circuit.from_file(CIRCUIT_PATH)
dem = circuit.detector_error_model(decompose_errors=False)

print(f"DEM: {dem.num_detectors} detectors, {dem.num_observables} observables")

sampler = circuit.compile_detector_sampler(seed=SEED)
raw = sampler.sample(NUM_SHOTS, separate_observables=True)
dets_raw, obs_raw = raw
dets = np.array(dets_raw, dtype=np.uint8)
obs  = np.array(obs_raw,  dtype=np.uint8)
print(f"Sampled {NUM_SHOTS} shots  (non-trivial syndromes: {(dets.any(axis=1)).sum()})")

import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, "Donotsync/ILP-decoder/src")

from decoder_confidence.decoding._ilp_cplex_logicalgap import _build_cplex_decoder_config, _build_cplex_decoder

config = _build_cplex_decoder_config({"threads": 1})
decoder = _build_cplex_decoder(dem, config)

print("\nDecoding 5 shots (logical gap) …")
times = []
for i in range(NUM_SHOTS):
    syndrome = dets[i].astype(int)
    t0 = time.perf_counter()
    result = decoder.decoder.decode_batch_result(
        [syndrome], get_logicalgap=True, logical_gap_flip_last_detector=False
    )
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    r = result[0]
    lg = r.metadata.get("logical_gap")
    pred = r.predicted_observables
    err = bool(np.any(np.array(pred, dtype=int) ^ obs[i]))
    print(f"  shot {i}: {elapsed:.2f}s  logical_gap={lg:.4f}  logical_error={err}")

print(f"\nTotal : {sum(times):.2f}s")
print(f"Mean  : {np.mean(times):.2f}s")
print(f"Median: {np.median(times):.2f}s")
print(f"Max   : {max(times):.2f}s")
print(f"Min   : {min(times):.2f}s")
