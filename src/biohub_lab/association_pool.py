"""Pool two observable association experts on the original candidate support."""
import math

def pool(original,specialist,mode):
    assert mode in ('specialist','balanced','uncertainty')
    by_target={}
    for edge in original:by_target.setdefault(edge[1],[]).append(edge)
    output={};weights=[]
    def margin(values):
        values=sorted(values,reverse=True)
        return values[0]-values[1] if len(values)>1 else 1.
    for target,edges in by_target.items():
        before=[original[e] for e in edges]
        after=[specialist.get(e,original[e]) for e in edges]
        if mode=='specialist':weight=1.
        elif mode=='balanced':weight=.5
        else:
            a,b=margin(before),margin(after)
            weight=.5*max(0.,(.2-a)/.2) if a<.2 and b>a+.02 else 0.
        weights.append(weight)
        for edge,p,q in zip(edges,before,after):
            # Missing sparse-cache values never mean zero probability.
            output[edge]=p if weight==0 or edge not in specialist else math.exp((1-weight)*math.log(max(1e-8,p))+weight*math.log(max(1e-8,q)))
    return output,dict(mode=mode,targets=len(weights),active_targets=sum(w>0 for w in weights),mean_weight=sum(weights)/max(1,len(weights)),shared_edges=len(set(original)&set(specialist)))
