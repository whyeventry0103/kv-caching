import sys, json, glob, collections, statistics, yaml
sys.path.insert(0,"src")
from kvcache.datasets import load_benchmark
from kvcache.grading import is_equiv
from kvcache.canonicalize import extract_answer
from kvcache.generation import has_answer_marker
bcfg=yaml.safe_load(open("configs/benchmarks.yaml"))["benchmarks"]
probs={p.pid:p for p in load_benchmark("math500",bcfg["math500"],subset=100,allow_download=False)}
R="/home/sk10741/kvcache-storage/runs/"
def rows(run):
    out=[]
    for f in sorted(glob.glob(R+run+"/math500/**/*.jsonl",recursive=True)):
        if "token_ids" in f: continue
        for l in open(f):
            if l.strip(): out.append(json.loads(l))
    return out
fc={ (r["pid"],r["decoding"]["seed"]):r for r in rows("math500-fc-powered-merged") if r["pid"] in probs and r["decoding"]["seed"] in (11,23,47,89)}
print("FC matched rows",len(fc),"acc %.4f"%(sum(r["correct"] for r in fc.values())/len(fc)))
for run in ["grid-diss256-tok","grid-diss256-h2o","grid-diss256-rkv","grid-diss1024","grid-diss2048-tok"]:
    rs=rows(run)
    by=collections.defaultdict(list)
    for r in rs: by[(r["press"],r["budget"])].append(r)
    for (press,b),v in sorted(by.items()):
        keys=[(r["pid"],r["decoding"]["seed"]) for r in v]
        dup=len(keys)-len(set(keys))
        missing_gold=[r["pid"] for r in v if r["pid"] not in probs]
        mism=0; recorrect=0; trunc=0; fr=collections.Counter(); ntok=[]; cr=[]
        for r in v:
            g=probs[r["pid"]].gold_answer
            has=has_answer_marker(r["trace"]); tr=bool(r["hit_token_cap"]) and not has
            c=(not tr) and has and is_equiv(g, extract_answer(r["trace"]) if has else "")
            recorrect+=c; mism+=(c!=r["correct"]); trunc+=r["truncated"]; fr[r["finish_reason"]]+=1; ntok.append(r["n_tokens"]); cr.append(r["compression_ratio"])
        n=len(v); fcs=[fc[k]["correct"] for k in keys if k in fc]
        print(f"{run:18s} {press:13s} B={b:<5} n={n} dup={dup} missing_gold={len(missing_gold)} acc_stored={sum(r['correct'] for r in v)/n:.4f} acc_rescored={recorrect/n:.4f} mismatch={mism} trunc={trunc/n:.3f} finish={dict(fr)} fc_on_same_keys={sum(fcs)/len(fcs):.4f} mean_tok={statistics.fmean(ntok):.0f} median_tok={statistics.median(ntok):.0f} mean_ratio={statistics.fmean(cr):.3f}")
