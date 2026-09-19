"""Map pre-pruning image associations through the baseline's GEFF reader IDs."""
import numpy as np

def decode_cache(raw_nodes,final_nodes,frames):
    lookup={}
    for k,v in raw_nodes.items():
        key=tuple(int(round(v[a])) for a in ('t','z','y','x'))
        lookup.setdefault(key,[]).append(k)
    # The baseline exporter uses these exact reader IDs, then smooths positions.
    for k,v in final_nodes.items():
        if k in raw_nodes:assert v['t']==raw_nodes[k]['t'],'Reader/export ID alignment failed'
    visual={};stats=dict(frame_pairs=0,proposed_pairs=0,mapped_pairs=0,raw_nodes=len(raw_nodes),
        final_nodes=len(final_nodes),final_ids_in_raw=sum(k in raw_nodes for k in final_nodes))
    for frame in frames:
        def mapping(coords):
            result={}
            for i,point in enumerate(coords):
                matches=lookup.get(tuple(int(x) for x in point),[])
                if len(matches)==1 and matches[0] in final_nodes:result[i]=matches[0]
            return result
        source=mapping(frame['source_coords']);target=mapping(frame['target_coords'])
        edges=frame['edges'];probs=frame['prob'];assert len(edges)==len(probs)
        assert np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
        for (a,b),p in zip(edges,probs):
            if int(a) in source and int(b) in target:
                e=(source[int(a)],target[int(b)])
                assert final_nodes[e[1]]['t']==final_nodes[e[0]]['t']+1
                if e in visual:raise ValueError('Duplicate captured pair')
                visual[e]=float(p);stats['mapped_pairs']+=1
        stats['frame_pairs']+=1;stats['proposed_pairs']+=len(edges)
    return visual,stats
