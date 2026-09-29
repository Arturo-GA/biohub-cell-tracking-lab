"""Appended to the exact CPU7 replay; E070 uses the same fixed definitions."""
exec(__CONTEXT_SOURCE__, globals())
exec(__POSITIONS_SOURCE__, globals())
exec(__VARIANTS_SOURCE__, globals())
(OUT/'run_manifest.json').write_text(_json.dumps(RUN_MANIFEST, indent=2))
E070_START = _t.monotonic()
_original_pipeline = pipeline


def pipeline(stem, cfg, xr=None):
    ns, es = _original_pipeline(stem, cfg, xr)
    if cfg.get('swap'):
        ns, es, op = temporal_swaps(ns, es, relative_gain=cfg['swap'])
        DIAGNOSTICS[ACTIVE_NAME][stem].update(op)
    return ns, es


def replay_positions(ns, es, cfg):
    if 'fork_smooth' not in cfg and 'position_mode' not in cfg:
        return full_linefit_round(ns, es, 2, .8)
    plain = full_linefit_round(ns, es, 2, .8, raw_output=True, tie_weights=(cfg.get('fork_smooth', .8),))
    smoothed = {k: dict(t=v[0], z=v[1], y=v[2], x=v[3]) for k,v in plain.items()}
    fixed, changed = e070_positions(ns, es, smoothed, cfg)
    return {k: rounded_node(v) for k,v in fixed.items()}


CONFIGS=[]
for name in LABS[LAB_PART]:
    cfg=dict(BASE); settings=variant(name); xr=settings.pop('xr'); cfg.update(settings)
    CONFIGS.append((name,cfg,xr))

# Compare the entire coordinate adapter against upstream smoothing on both
# smallest graphs before spending time on all 199 videos.
for prefix in ('44b6','6bba'):
    stem=min((s for s in STEMS if s.startswith(prefix)),key=lambda s:len(CACHE[s][0]))
    for name,cfg,xr in CONFIGS:
        ACTIVE_NAME=name
        ns,es=pipeline(stem,cfg,xr)
        actual=replay_positions(ns,es,cfg)
        original=linefit_smooth_output_graph({k:dict(v) for k,v in ns.items()},es,collections.Counter())
        expected,_=e070_positions(ns,es,original,cfg)
        expected={k:rounded_node(v) for k,v in expected.items()}
        assert actual==expected,('E070 smoothing parity failed',stem,name)
log('E070 POSITION PARITY PASS',len(CONFIGS)*2)

results=[]; all_rows={}; metric_cache={}
for name,cfg,xr in CONFIGS:
    ACTIVE_NAME=name; tasks=[]; fingerprints={}; rows={}
    for s in STEMS:
        ns,es=pipeline(s,cfg,xr)
        plain=replay_positions(ns,es,cfg)
        pairs=[(int(e['source_id']),int(e['target_id'])) for e in es]
        ids=sorted(plain); h=_hashlib.sha256()
        h.update(np.asarray(ids,dtype=np.int64).tobytes())
        h.update(np.asarray([plain[k] for k in ids],dtype=np.int64).tobytes())
        h.update(np.asarray(pairs,dtype=np.int64).reshape(-1,2).tobytes())
        key=(s,h.hexdigest());fingerprints[s]=key
        DIAGNOSTICS[name][s].update(final_graph_hash=key[1])
        if key in metric_cache:rows[s]=metric_cache[key]
        else:tasks.append((s,plain,pairs,CACHE[s][4]))
    with Pool(4) as pool:
        for s,row,error in pool.imap_unordered(worker,tasks):
            if error:raise RuntimeError((s,error))
            rows[s]=row;metric_cache[fingerprints[s]]=row
    assert set(rows)==set(STEMS)
    all_rows[name]=rows
    (OUT/(name+'.rows.json')).write_text(_json.dumps(rows))
    if name in ('C3_public0955','C3_fork8'):
        for s,row in rows.items():
            for key,value in EXPECTED_BASELINE[name][s].items():
                assert abs(row[key]-value)<1e-10,('E070 baseline mismatch',name,s,key,row[key],value)
        log('E070 BASELINE EXACT MATCH',name,len(rows))
    sc,adj,div=score(rows,STEMS)
    base=all_rows['C3_public0955'];d,lo,hi,p,w,l=boot(base,rows,STEMS)
    control=all_rows.get('C3_fork8',base)
    fd,flo,fhi,fp,fw,fl=boot(control,rows,STEMS)
    item=dict(name=name,score=sc,delta=d,delta_vs_fork8=fd,ci90=[lo,hi],win=w,lose=l,
              fork8_win=fw,fork8_lose=fl,div=list(div),videos=len(STEMS),evaluated=len(tasks),reused=len(STEMS)-len(tasks),
              cfg=cfg,xr=xr,groups={g:score(rows,[s for s in STEMS if s.startswith(g)])[0]-score(base,[s for s in STEMS if s.startswith(g)])[0] for g in ('44b6','6bba')})
    results.append(item)
    (OUT/'results.json').write_text(_json.dumps(results,indent=2))
    (OUT/'graph_diagnostics.json').write_text(_json.dumps(DIAGNOSTICS))
    log('E070 RESULT',_json.dumps(item))
(OUT/'complete.json').write_text(_json.dumps(dict(videos=len(STEMS),configurations=len(results),seconds=_t.monotonic()-E070_START)))
log('E070 COMPLETE',LAB_PART)
