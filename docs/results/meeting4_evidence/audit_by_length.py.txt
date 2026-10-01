import json, glob, collections, statistics
R="/home/sk10741/kvcache-storage/runs/"
def rows(run, sub):
    out=[]
    for f in sorted(glob.glob(R+run+"/"+sub+"/**/*.jsonl",recursive=True)):
        if "token_ids" in f: continue
        for l in open(f):
            if l.strip(): out.append(json.loads(l))
    return out
fc={(r["pid"],r["decoding"]["seed"]):r for r in rows("math500-fc-powered-merged","math500")}
ro=collections.defaultdict(dict)
for run in ["readout-diss256-g0","readout-diss256-g1","readout-diss256-h2o","readout-diss256-rkv"]:
    for r in rows(run,"readout"):
        if r["readout_correct"] is None: continue
        ref=r["reference_key"]; ro[r["condition"]["press"]][(str(ref[1]),int(ref[5]))]=r["readout_correct"]
keys=sorted(set.intersection(*[set(v) for v in ro.values()]))
bins=[(0,1000),(1000,2000),(2000,4000),(4000,8000),(8000,20000)]
print("forward readout accuracy by reference trace length (tokens), budget 256; n per bin")
for lo,hi in bins:
    ks=[k for k in keys if lo<=fc[k]["n_tokens"]<hi]
    if not ks: continue
    line=f"  [{lo:5d},{hi:5d}) n={len(ks):3d} fc={statistics.fmean(fc[k]['correct'] for k in ks):.3f}"
    for p in ("random","streamingllm","h2o","rkv"):
        line+=f"  {p}={statistics.fmean(ro[p][k] for k in ks):.3f}"
    print(line)
# free-run accuracy by FULL-CACHE reference length (pre-treatment covariate)
comp=collections.defaultdict(dict)
for run in ["grid-diss256-tok","grid-diss256-h2o","grid-diss256-rkv"]:
    for r in rows(run,"math500"): comp[r["press"]][(r["pid"],r["decoding"]["seed"])]=(r["correct"],r["n_tokens"],r["truncated"])
print("free-run accuracy / mean generated tokens / truncation by full-cache reference length")
for lo,hi in bins:
    ks=[k for k in fc if lo<=fc[k]["n_tokens"]<hi and k in comp["random"]]
    if not ks: continue
    line=f"  [{lo:5d},{hi:5d}) n={len(ks):3d} fc={statistics.fmean(fc[k]['correct'] for k in ks):.3f}"
    for p in ("random","streamingllm","h2o","rkv"):
        line+=f" | {p} acc={statistics.fmean(comp[p][k][0] for k in ks):.3f} tok={statistics.fmean(comp[p][k][1] for k in ks):5.0f} tr={statistics.fmean(comp[p][k][2] for k in ks):.2f}"
    print(line)
