"""Appended to the exact c3 replay definitions by build_e069.py."""
import hashlib
import time

exec(__INDEPENDENT_RULES__, globals())
RUN_STARTED=time.monotonic()
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'run_manifest.json').write_text(_json.dumps(RUN_MANIFEST,indent=2))


def save_json(name, data):
    (OUT/name).write_text(_json.dumps(data,indent=1))


def choose_panel(stems):
    # Fixed seed, two embryo groups x four node-count strata x four videos.
    # Selection never reads ground-truth scores or donor performance.
    selected=[]
    for prefix in ('44b6','6bba'):
        ordered=sorted((s for s in stems if s.startswith(prefix)),key=lambda s:(len(CACHE[s][0]),s))
        for stratum,group in enumerate(np.array_split(ordered,4)):
            group=sorted(map(str,group),key=lambda s:hashlib.sha256(('e069-fixed-panel-1:'+s).encode()).hexdigest())
            assert len(group)>=4
            for i,s in enumerate(group[:4]):
                selected.append(dict(stem=s,group=prefix,density_stratum=stratum,
                                     subset='development' if i<2 else 'confirmation',nodes=len(CACHE[s][0])))
    return selected


def c3_graph(stem, fork=False):
    global ACTIVE_NAME
    ACTIVE_NAME='C3_fork8' if fork else 'C3_public0955'
    return pipeline(stem,BASE,C3_VARIANTS[ACTIVE_NAME])


if MODE in ('capture','image_cache'):
    import zarr
    import gzip
    if MODE=='capture':
        import torch
        exec(__INDEPENDENT_LINKER__, globals())
        assert torch.cuda.is_available(), 'This image-feature task requires GPU'
        model_file=next(Path('/kaggle/input').rglob('00000008.pth'))
        model,sampler,checkpoint_sha=load_independent_model(__MODEL_DEFINITIONS__,model_file)
        assert checkpoint_sha==EXPECTED_CHECKPOINT_SHA, 'Donor checkpoint version changed'
        torch.backends.cudnn.benchmark=False
        save_json('model.json',dict(checkpoint_sha256=checkpoint_sha,device=torch.cuda.get_device_name(0),
                                   coordinates='native z,y,x divided by (1,4,4)',pyramid=['d0','d1','e2']))
    panel=choose_panel(STEMS);save_json('panel.json',panel)
    cache_dir=OUT/'candidates';cache_dir.mkdir(exist_ok=True)
    bundle_dir=OUT/'bundles';bundle_dir.mkdir(exist_ok=True)
    completed=[]
    for entry in panel:
        if time.monotonic()-RUN_STARTED>5400:
            save_json('partial.json',dict(completed=completed,reason='90min extraction budget'))
            raise RuntimeError('Capture budget reached: preserve completed files, do not score a partial panel as full')
        stem=entry['stem'];started=time.monotonic()
        ns,es=c3_graph(stem);candidates=swap_candidates(ns,es)
        by_t=collections.defaultdict(list)
        for c in candidates: by_t[c['t']].append(c)
        frame_ids=collections.defaultdict(list)
        for k in sorted(ns): frame_ids[int(ns[k]['t'])].append(k)
        group=zarr.open_group(str(TRAIN_DIR/f'{stem}.zarr'),mode='r')
        quantiles=dict(group.attrs).get('image_statistics',{}).get('quantiles',{})
        lo,hi=float(quantiles['0.001']),float(quantiles['0.999'])
        assert np.isfinite([lo,hi]).all() and hi>lo
        image=group['0'];assert len(image.shape)==4 and image.shape[1:]==(64,256,256)
        if MODE=='image_cache':
            frames=sorted({f for t in by_t for f in (t,t+1)})
            volumes=np.stack([np.asarray(image[t,::1,::4,::4]) for t in frames]) if frames else np.zeros((0,64,64,64),np.uint16)
            # Annotation export is confined to this CPU lab; model inference
            # below never consults these arrays or their labels.
            truth=td.graph.IndexedRXGraph.from_geff(str(TRAIN_DIR/f'{stem}.geff'))[0]
            truth_nodes=truth.node_attrs().select(['node_id','t','z','y','x']).to_numpy()
            truth_edges=truth.edge_attrs().select(['source_id','target_id']).to_numpy()
            ids=sorted(ns)
            np.savez_compressed(bundle_dir/f'{stem}.npz',frames=np.asarray(frames),images=volumes,
                quantiles=np.asarray([lo,hi]),nid=np.asarray(ids),
                nodes=np.asarray([[ns[k][a] for a in ('t','z','y','x')] for k in ids]),
                edges=np.asarray([[e['source_id'],e['target_id']] for e in es],dtype=np.int64).reshape(-1,2),
                eprob=np.asarray([np.nan if e.get('edge_prob') is None else e['edge_prob'] for e in es]),
                truth_nodes=truth_nodes,truth_edges=truth_edges,t_true=np.asarray([CACHE[stem][4] or np.nan]))
            with gzip.open(cache_dir/f'{stem}.json.gz','wt',encoding='utf-8') as f:
                _json.dump(dict(stem=stem,candidates=candidates,graph=DIAGNOSTICS['C3_public0955'][stem]),f)
            completed.append(stem)
            log('IMAGE EXPORT',len(completed),'/',len(panel),stem,'candidates',len(candidates),
                'MB',round((bundle_dir/f'{stem}.npz').stat().st_size/1e6,1),'seconds',round(time.monotonic()-started,1))
            del volumes,truth
            continue
        cached={};scored=[]
        for t,cs in sorted(by_t.items()):
            for frame in (t,t+1):
                if frame not in cached:
                    volume=np.asarray(image[frame,::1,::4,::4],dtype=np.float32)
                    volume=np.maximum((volume-lo)/(hi-lo+1e-6),0)
                    cached[frame]=sample_frame(model,sampler,volume,frame_ids[frame],ns)
            logits=pair_logits(model,cached[t],cached[t+1],(64,64,64))
            scored.extend(annotate_candidates(cs,logits,frame_ids[t],frame_ids[t+1]))
            cached={f:v for f,v in cached.items() if f==t+1}
        assert len(scored)==len(candidates)
        record=dict(stem=stem,candidates=scored,seconds=time.monotonic()-started,
                    nodes=len(ns),edges=len(es),graph=DIAGNOSTICS['C3_public0955'][stem])
        (cache_dir/f'{stem}.json').write_text(_json.dumps(record))
        completed.append(stem)
        log('CAPTURE',len(completed),'/',len(panel),stem,'candidates',len(scored),'seconds',round(record['seconds'],1))
        del cached
    save_json('complete.json',dict(videos=len(completed),stems=completed,seconds=time.monotonic()-RUN_STARTED))
    log('E069 CAPTURE COMPLETE',MODE,len(completed))

else:
    baseline_rows={}
    if MODE in ('linker_eval','local_linker_eval'):
        if MODE=='local_linker_eval':
            import gzip
            capture_manifest=next(Path('/kaggle/input').rglob('e069_local_manifest.json'))
            assert _json.loads(capture_manifest.read_text())['image_manifest']==CAPTURE_MANIFEST
        else:
            capture_manifest=next(Path('/kaggle/input').rglob('e069_capture/run_manifest.json'))
            assert _json.loads(capture_manifest.read_text())==CAPTURE_MANIFEST
        captured=capture_manifest.parent
        panel=_json.loads((captured/'panel.json').read_text())
        assert _json.loads((captured/'complete.json').read_text())['videos']==32
        STEMS=[x['stem'] for x in panel]
        if MODE=='local_linker_eval':
            with gzip.open(captured/'e069_records.bin','rt',encoding='utf-8') as f:candidates=_json.load(f)
            assert set(candidates)==set(STEMS)
        else:
            candidates={s:_json.loads((captured/'candidates'/f'{s}.json').read_text()) for s in STEMS}
        configurations=['C3_public0955','geometry','neural_strict','neural_balanced',
                        'join_geometry','join_neural','fork_veto','C3_fork8','fork8_neural_strict','fork8_joint_neural','fork8_veto']
        save_json('panel.json',panel)
    else:
        assert MODE=='duplicates' and len(STEMS)==199
        configurations=['C3_public0955','duplicate_strict','duplicate_balanced','C3_fork8','fork8_duplicate_strict','fork8_duplicate_balanced']
    results=[];all_operations={};metric_cache={}
    for name in configurations:
        graphs={};operations={}
        for stem in STEMS:
            ns,es=c3_graph(stem,fork=name.startswith('fork8_') or name=='C3_fork8')
            if MODE in ('linker_eval','local_linker_eval') and name not in ('C3_public0955','C3_fork8'):
                source=candidates[stem]
                if not name.startswith('fork8_'):
                    for k in ('node_hash','edge_hash'):
                        assert source['graph'][k]==DIAGNOSTICS['C3_public0955'][stem][k]
                    cs=source['candidates']
                else:
                    # Revalidate all four-frame contexts against the fork8 graph.
                    current={(c['a'],c['b'],c['da'],c['db']):c for c in swap_candidates(ns,es)}
                    cs=[c for c in source['candidates'] if (c['a'],c['b'],c['da'],c['db']) in current
                        and c['context']==current[(c['a'],c['b'],c['da'],c['db'])]['context']]
                if name in ('fork_veto','fork8_veto'):
                    ns,es,op=apply_fork_veto(ns,es,source.get('forks',[]))
                elif name in ('join_geometry','join_neural'):
                    ns,es,op=apply_joins(ns,es,source.get('joins',[]),neural=name=='join_neural')
                else:
                    mode='neural_strict' if name.startswith('fork8_') else name
                    ns,es,op=apply_swaps(ns,es,cs,mode=mode)
                    if name=='fork8_joint_neural':
                        ns,es,join_op=apply_joins(ns,es,source.get('joins',[]))
                        op.update(joins=join_op['joins'],join_details=join_op)
            elif 'duplicate' in name:
                kw={} if name.endswith('strict') else dict(radius=3.,max_length=40,max_confidence=.75,confidence_gap=.05,coverage=.85)
                ns,es,op=persistent_duplicates(ns,es,**kw)
            else: op={}
            graphs[stem]=(ns,es);operations[stem]=op
        ACTIVE_NAME=name;DIAGNOSTICS.setdefault(name,{})
        for s in STEMS: DIAGNOSTICS[name].setdefault(s,{})
        if MODE in ('linker_eval','local_linker_eval'):
            fingerprints={}
            for stem,(ns,es) in graphs.items():
                h=hashlib.sha256()
                ids=sorted(ns)
                h.update(np.asarray(ids,dtype=np.int64).tobytes())
                h.update(np.asarray([[ns[k][a] for a in ('t','z','y','x')] for k in ids],dtype=np.float64).tobytes())
                h.update(np.asarray([[e['source_id'],e['target_id']] for e in es],dtype=np.int64).reshape(-1,2).tobytes())
                fingerprints[stem]=(stem,h.hexdigest())
            original_stems=STEMS
            missing=[s for s in STEMS if fingerprints[s] not in metric_cache]
            try:
                STEMS=missing
                fresh=official_rows(graphs,BASE) if missing else {}
            finally:
                STEMS=original_stems
            for s,row in fresh.items():metric_cache[fingerprints[s]]=row
            rows={s:metric_cache[fingerprints[s]] for s in STEMS}
            log('EXACT GRAPH METRIC CACHE',name,'evaluated',len(missing),'reused',len(STEMS)-len(missing))
        else:
            rows=official_rows(graphs,BASE)
        if MODE in ('linker_eval','local_linker_eval') and name=='C3_public0955':
            for stem,row in rows.items():
                for key,value in EXPECTED_BASELINE[stem].items():
                    assert abs(row[key]-value)<1e-10,('Baseline mismatch',stem,key,row[key],value)
            log('C3 PANEL BASELINE EXACT MATCH',len(rows))
        save_json(name+'.rows.json',rows)
        if not baseline_rows: baseline_rows=rows
        sc,adj,div=score(rows,STEMS)
        if MODE=='duplicates' and name=='C3_public0955': assert abs(sc-.9275133844979068)<1e-8
        d,lo,hi,p,w,l=boot(baseline_rows,rows,STEMS)
        item=dict(name=name,score=sc,delta=d,ci90=[lo,hi],win=w,lose=l,division=list(div),videos=len(STEMS))
        if MODE in ('linker_eval','local_linker_eval'):
            item['subsets']={label:score(rows,[r['stem'] for r in panel if r['subset']==label])[0]-score(baseline_rows,[r['stem'] for r in panel if r['subset']==label])[0]
                             for label in ('development','confirmation')}
        item['operations']={k:sum(op.get(k,0) for op in operations.values()) for k in ('candidates','eligible','swaps','joins','fork_vetoes','removed_tracks','removed_nodes')}
        results.append(item);all_operations[name]=operations
        save_json('results.json',results);save_json('operations.json',all_operations)
        log('E069 RESULT',_json.dumps(item))
    save_json('complete.json',dict(configurations=len(results),videos=len(STEMS),seconds=time.monotonic()-RUN_STARTED,
                                 note='Reused detector-training videos; the confirmation subset is not independent detector validation.'))
    log('E069 CPU COMPLETE',MODE)
