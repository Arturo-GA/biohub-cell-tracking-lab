"""Package sequential E041/E042/E043 experiments; never launch implicitly."""
import json,sys
from pathlib import Path
from build_three_lines import build,ROOT

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='E041_PREPARE':
        build(stage,'scripts/dense_supervision_prepare_runner.py',[],[],internet=True)
    elif stage=='E041_REGRID':
        build(stage,'scripts/dense_supervision_regrid_runner.py',['scripts/dense_supervision_prepare_runner.py','scripts/spatial_probe_features_runner.py','src/biohub_lab/spatial_probe.py'],['jarturo/biohub-e041-prepare-cpu'])
    elif stage in ['E041_TRAIN','E041_EVALUATE']:
        if stage=='E041_TRAIN':
            r=json.loads((ROOT/'results/E041_REGRID_completed.json').read_text())
            cfg=dict(seed=410920,steps=4000,deadline_seconds=3600,data_sha256=r['manifest_sha256'],biohub_data_sha256='b8df897a2b8496d1b1a37018147df03f9db25b6c55806328e68b675b6ea3454a',frames=[10,30,50,70,90],gate='budget256: dense matched3 greater than all E039 controls in each embryo; matched7 no worse than any')
            (ROOT/'baseline/e041_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
        files=['baseline/e041_protocol.json','src/biohub_lab/dense_center.py','src/biohub_lab/temporal_detector.py','scripts/spatial_probe_features_runner.py','src/biohub_lab/spatial_probe.py','scripts/temporal_detector_evaluate_runner.py','scripts/association_cpu_runner.py']
        train=stage.endswith('TRAIN')
        build(stage,'scripts/dense_supervision_'+('train' if train else 'evaluate')+'_runner.py',files,['jarturo/biohub-e038-prepare-cpu']+(['jarturo/biohub-e041-regrid-cpu'] if train else ['jarturo/biohub-e041-train','jarturo/biohub-e039-evaluate-cpu']),gpu=train)
    elif stage.startswith('E042_'):
        mode=stage.split('_')[1].lower();files=['scripts/spatial_probe_features_runner.py','src/biohub_lab/spatial_probe.py','scripts/association_cpu_runner.py','src/biohub_lab/resolution_probe.py']
        if mode=='train':
            r=json.loads((ROOT/'results/E042_PREPARE_completed.json').read_text());cfg=dict(seed=420920,steps=3000,deadline_seconds=3600,data_sha256=r['manifest_sha256'],input='two channels: current crop plus annotated consecutive-parent query; same examples in both arms',gate='native mean-per-video localization error <0.9 coarse and improves each embryo')
            (ROOT/'baseline/e042_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
        if mode!='prepare':files+=['baseline/e042_protocol.json']
        source={'prepare':'jarturo/biohub-e038-prepare-cpu','train':'jarturo/biohub-e042-prepare-cpu','evaluate':'jarturo/biohub-e042-train'}[mode]
        build(stage,'scripts/resolution_'+mode+'_runner.py',files,[source],gpu=mode=='train')
    elif stage=='E043_EVALUATE':
        build(stage,'scripts/uncertain_lineage_runner.py',['src/biohub_lab/uncertain_lineage.py','scripts/association_cpu_runner.py','scripts/spatial_probe_features_runner.py','src/biohub_lab/spatial_probe.py'],['jarturo/biohub-temporal-graph-prepare-cpu','jarturo/biohub-temporal-graph-features'])
