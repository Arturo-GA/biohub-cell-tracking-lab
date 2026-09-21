import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E047',raw_sha256='7167b59f9ecf522153c92f546fa306e3fa083745d128fc7e6d7f73285ecebcf0',dense_sha256=None,static_sha256='de5aaf41154afdf54857a56265943c551e159b6a3b11bce4f6589499f777ef3f',graph_sha256='c34d3d8daa9a50292001153ba0b8f500ca71b81f756a85cac9e02e1b5826df22',budget=512,protocol='Exploratory follow-up to E045 at512. Frozen dense+static on all100 frames of other reused16 cohort. E031 normalized images clipped to[0,1]. CPU spatial consensus unchanged; donor bridges E034 fixed geometry; local snap Harmonic<=3um one-to-one. No GT in proposals.',gate='Full graph score >=Harmonic+0.001; edgeTP gain, no edgeFP or divisionFP increase, no divisionTP loss; nonnegative score change per embryo. Then confirmation before submission.')
cfg['dense_sha256']=json.loads(Path('results/E041_TRAIN_completed.json').read_text())['manifest_sha256']
Path('baseline/e047_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E047_INFER','scripts/consensus_infer_runner.py',['baseline/e047_protocol.json','scripts/ensemble_evaluate_runner.py','src/biohub_lab/dense_center.py','src/biohub_lab/temporal_detector.py'],['jarturo/biohub-dense-detector-prepare-cpu','jarturo/biohub-e041-train','jarturo/biohub-e039-train'],gpu=True)
