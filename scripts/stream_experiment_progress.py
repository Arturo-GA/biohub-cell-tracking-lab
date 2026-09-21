"""Read bounded-lived Kaggle log streams; disconnects do not mean job failures."""
import json,sys
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
api=KaggleApi();api.authenticate();user,slug=sys.argv[1].split('/');after=float(sys.argv[2]) if len(sys.argv)>2 else 0
seen=set();announced=set()
with api.build_kaggle_client() as client:
    r=ApiGetKernelSessionLogsStreamRequest();r.user_name=user;r.kernel_slug=slug;r.wait_for_logs_url_seconds=5
    try:
        response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(r)
        for line in response.iter_lines(decode_unicode=True):
            if not line.startswith('data: '):continue
            try:event=json.loads(line[6:])
            except json.JSONDecodeError:continue  # SSE terminal/heartbeat markers are not log JSON.
            if not isinstance(event,dict):continue
            message=event.get('data','')
            if 'DENSE_ARCHIVE' in message:continue  # Large directory listing is not progress.
            if event.get('time',0)<after or message in seen:continue
            if 'SECONDARY_EDGE_TTA_ACTIVE' in message:
                if 'secondary_tta' in announced:continue
                announced.add('secondary_tta')
            if 'SYNTH_EDGE ' in message:
                key='synthetic_pair_'+message.split('SYNTH_EDGE ',1)[1].split()[0]
                if key in announced:continue
                announced.add(key)
            if any(k in message for k in ['Synthetic edge-only third model','SYNTH_EDGE ','Prediction completed','independent video shards','Loaded DeepCenter',' FINAL:','SECONDARY_EDGE_TTA_ACTIVE','ENSEMBLE_','DENSE_','RESOLUTION_','UNCERT_','DIVISION_DATA','DIVISION_HEAD','DIVISION_METRIC','JOINT_PROPOSALS','JOINT_METRIC','NEURAL_PREP','NEURAL_FEATURES','NEURAL_HEAD','NEURAL_METRIC','NEURAL_GRAPH','THREE_PREP','SPATIAL_','CENTER_','EVENT_','Traceback','Error:']):
                seen.add(message);print(round(event.get('time',0),1),message.rstrip(),flush=True)
    except requests.exceptions.ChunkedEncodingError:
        print('Log stream disconnected; check kernel status separately.',flush=True)
