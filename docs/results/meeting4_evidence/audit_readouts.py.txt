import sys, json, glob, collections, statistics, yaml
sys.path.insert(0,"src")
from kvcache.datasets import load_benchmark
from kvcache.grading import is_equiv
from kvcache.canonicalize import extract_answer
bcfg=yaml.safe_load(open("configs/benchmarks.yaml"))["benchmarks"]
probs={p.pid:p for p in load_benchmark("math500",bcfg["math500"],subset=100,allow_download=False)}
R="/home/sk10741/kvcache-storage/runs/"
def rows(run, sub="math500"):
    out=[]
    for f in sorted(glob.glob(R+run+"/"+sub+"/**/*.jsonl",recursive=True)):
        if "token_ids" in f: continue
        for l in open(f):
            if l.strip(): out.append(json.loads(l))
    return out
fc={ (r["pid"],r["decoding"]["seed"]):r for r in rows("math500-fc-powered-merged") if r["pid"] in probs and r["decoding"]["seed"] in (11,23,47,89)}
comp={}
for run in ["grid-diss256-tok","grid-diss256-h2o","grid-diss256-rkv"]:
    for r in rows(run): comp[(r["press"],r["pid"],r["decoding"]["seed"])]=r["correct"]
for name,runs in [("forward readout",["readout-diss256-g0","readout-diss256-g1","readout-diss256-h2o","readout-diss256-rkv"]),("reverse readout",["reverse-readout-diss256","reverse-readout-diss256-h2o","reverse-readout-diss256-rkv"])]:
    by=collections.defaultdict(list)
    for run in runs:
        for r in rows(run,"readout"):
            ref=r["reference_key"]; pid=str(ref[1]); seed=int(ref[5])
            cell=(r["condition"]["press"],str(r["condition"]["budget"])) if name=="forward readout" else (str(ref[3]),str(ref[4]))
            by[cell].append((pid,seed,r))
    print("==",name)
    for cell,v in sorted(by.items()):
        keys=[(p,s) for p,s,_ in v]; dup=len(keys)-len(set(keys))
        defined=[(p,s,r) for p,s,r in v if r["readout_correct"] is not None]
        mism=sum(1 for p,s,r in defined if (is_equiv(probs[p].gold_answer, extract_answer(r["generated_text"])) != r["readout_correct"]))
        acc=sum(r["readout_correct"] for _,_,r in defined)/len(defined)
        fcd=statistics.fmean(fc[(p,s)]["correct"] for p,s,_ in defined)
        press=cell[0]
        compd=statistics.fmean(comp[(press,p,s)] for p,s,_ in defined) if name=="forward readout" and press in ("random","streamingllm","h2o","rkv") else None
        print(f"  {cell} n={len(v)} dup={dup} defined={len(defined)} acc={acc:.4f} rescored_mismatch={mism} fc_on_defined={fcd:.4f} comp_on_defined={compd} -> residual(comp-readout)={None if compd is None else round(compd-acc,4)}")
