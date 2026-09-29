"""Measure actual pretrained U-Net memory on the laptop; no quality claims."""
import importlib.util
import json
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'artifacts/e012_detector_check/repo/src/biohub_tracking/models/temporal_unet.py'
spec = importlib.util.spec_from_file_location('e071_temporal_unet', source)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
weights = ROOT / 'artifacts/e012_detector_check/primary/edge_predictor_best.pth'
state = torch.load(weights, map_location='cpu', weights_only=True)
model = module.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128])
model.load_state_dict({k.removeprefix('unet.'):v for k,v in state.items() if k.startswith('unet.')},strict=True)
model.eval().cuda(); torch.set_num_threads(2)
shapes = json.loads((ROOT/'results/E054_CONTROL_completed.json').read_text(encoding='utf-8'))['result']['shapes']
trials = sorted({(v[1], (v[2]+3)//4, (v[3]+3)//4) for v in shapes.values()})
results=[]
for shape in trials:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    try:
        with torch.inference_mode():
            x=torch.zeros((1,2,1,*shape),device='cuda')
            y=model(x);torch.cuda.synchronize()
        row=dict(shape=shape,batch=1,temporal_window=2,precision='float32',status='PASS',
                 seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del x,y
    except torch.cuda.OutOfMemoryError as exc:
        row=dict(shape=shape,status='OOM',message=str(exc)[:250])
        if 'x' in locals():del x
        torch.cuda.empty_cache()
    results.append(row);print(json.dumps(row),flush=True)
report=dict(device=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch_version=torch.__version__,trials=results,
            scope='Pretrained U-Net only on synthetic inputs matching visible volume dimensions; not the complete dual-model pipeline or a quality validation.')
(ROOT/'results/E071/laptop_gpu_probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
