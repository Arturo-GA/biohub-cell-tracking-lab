"""Finalize this research round only after all joint CPU metrics exist."""
import json
from datetime import datetime,timezone
from pathlib import Path
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
joint=read('results/E053_EVALUATE_completed.json')['result'];assert joint['status']=='complete'
parallel=read('results/E052_PARALLEL_completed.json')['result'];sequential=read('results/E052_REPLAY_completed.json')['result'];assert parallel['metrics']==sequential['metrics']
best=joint['selected'];score=joint['metrics'][best]['score'];improved=score>.9502832669356378+1e-8
decision=dict(status='selected' if improved else 'not_promoted',selected=best,score=score,delta_vs_prior_visual=score-.9502832669356378,joint_submission_built=Path('results/E054_SUBMISSION_launch.json').exists(),reason='New joint gain requires final inference' if improved else 'All 16 added-bridge variants fail to beat visual-only; do not resubmit the already-tested visual configuration')
save('results/E054_JOINT_decision.json',decision)
status=read('results/STATUS.json');control=read('results/E054_CONTROL_status.json');attempt=read('results/E054_CONTROL_attempt.json')
status['quality_status']='E051/E052/E053 completed. '+('Joint candidate improved locally.' if improved else 'No new joint improvement.')+' Pure public-matched control submitted; Kaggle status '+control['status']+'.'
status['E052']=dict(status='COMPLETE',control_reproduced=True,parallel_and_sequential_metrics_identical=True,promoted=False,receipts=['results/E052_PARALLEL_completed.json','results/E052_REPLAY_completed.json'],seconds_parallel=parallel['seconds'],seconds_sequential=sequential['seconds'])
status['E053']=dict(status='COMPLETE',selected=best,score=score,delta_vs_prior_visual=joint['delta_vs_prior_visual'],promoted=improved,receipt='results/E053_EVALUATE_completed.json',seconds=joint['seconds'])
status['E054_CONTROL']=dict(status='SUBMITTED',kernel=attempt['kernel'],version=attempt['version'],submission_ref=attempt['ref'],submission_status=control['status'],public_score=control['public_score'],visible_csv_equals_public_v3=True,csv_locally_validated=True,pending_notebook=False)
status['latest_submission_ref']=attempt['ref'];status['updated_at_utc']=datetime.now(timezone.utc).isoformat()
if control['status']=='COMPLETE' and control['public_score']:status['own_leaderboard_score']=max(float(status.get('own_leaderboard_score',0)),float(control['public_score']))
save('results/STATUS.json',status)
rows=['| Configuración | Control | + Denso | + Ponderado |','|---|---:|---:|---:|']
for arm in parallel['metrics']:
 rows.append('| '+arm+' | '+' | '.join(f"{joint['metrics'][arm+s]['score']:.9f}" for s in ('','_dense','_weighted'))+' |')
text='\n'.join(rows)
p=Path('docs/E051_E054_PUBLICOS_Y_ENSAMBLE.es.md');s=p.read_text(encoding='utf-8').split('## Resultados conjuntos')[0]
s+='## Resultados conjuntos\n\n'+text+'\n\n'
s+=f"E053 terminó en {joint['seconds']:.3f} segundos CPU. Selección: `{best}`; diferencia frente a la asociación visual anterior: {joint['delta_vs_prior_visual']:.9f}. "
s+=('Existe mejora local; véase el recibo para la siguiente ejecución.' if improved else 'Ninguna de las dieciséis combinaciones nuevas supera el control visual. No se construye ni lanza una submission conjunta repetida.')+'\n\n'
s+='Los puentes que mejoraron la calibración antigua de 16 videos no recuperan enlaces anotados adicionales en estos otros ocho, y reducen ligeramente el score ajustado. Este resultado no prueba que cada nodo añadido sea incorrecto: las etiquetas son parciales. Sí impide asumir que las ganancias de los componentes se suman o se trasladan al leaderboard.\n\n'
s+=f"**Submission enviada: {attempt['ref']}**, notebook [jarturo/biohub-e054-public-control](https://www.kaggle.com/code/jarturo/biohub-e054-public-control), versión {attempt['version']}. Estado consultado: **{control['status']}**"+(f", score **{control['public_score']}**" if control['public_score'] else ', todavía sin score')+'. La inferencia completa se descargó y validó localmente, con coincidencia exacta del SHA256 público. Duración del runner: 959.283 segundos; inferencia principal: 9.27 minutos. No quedan notebooks de esta ronda ejecutándose; solo el scoring de Kaggle si continúa pendiente.\n\n'
s+='Siete pruebas locales pasaron. Las métricas de E052 coinciden exactamente entre la ejecución secuencial (1740.422 s) y la distribuida (1024.196 s); ambas consumieron CPU, no GPU. Código, paquetes privados, errores recuperados y recibos quedan versionados en GitHub.\n'
p.write_text(s,encoding='utf-8')
p=Path('README.md');s=p.read_text(encoding='utf-8');block=f"**E051–E054, 21 de septiembre:** se confirmó E050 = 0.946. Tres nuevas fusiones y dieciséis combinaciones con detectores no superaron la asociación visual en los ocho videos reutilizados. Nuestro control `tight55` reproduce byte por byte el CSV público de Harmonic V3. Se envió la ablación limpia **{attempt['ref']}** ({control['status']}"+(f", {control['public_score']}" if control['public_score'] else ', sin score todavía')+"). [Auditoría, resultados y límites](docs/E051_E054_PUBLICOS_Y_ENSAMBLE.es.md).\n\n"
if '**E051–E054, 21 de septiembre:**' not in s:s=s.replace('# Biohub Cell Tracking Lab\n\n','# Biohub Cell Tracking Lab\n\n'+block,1)
p.write_text(s,encoding='utf-8');print(json.dumps(decision,indent=2))
