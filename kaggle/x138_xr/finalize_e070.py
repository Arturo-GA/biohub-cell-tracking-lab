"""Freeze five verified candidates for manual submission; never calls Kaggle."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/E070'
ORDER = ['F8_dc030', 'C3_fork8', 'F8_prune12_joint', 'F8_prune12', 'F8_joint']
RATIONALE = {
    'F8_dc030': ('Exploratorio prioritario', 'Mayor score local; 16 divisiones falsas menos, pero pierde una verdadera frente a fork8. Su intervalo incremental cruza cero.'),
    'C3_fork8': ('Referencia con evidencia local mas amplia', '100 videos mejoran y 3 empeoran frente a c3; permite medir el cambio comun a toda la tanda.'),
    'F8_prune12_joint': ('Exploratorio combinado', 'Combina poda 12 y correccion de cruces/posiciones en bifurcaciones. Segundo score local; incremento pequeno e incierto.'),
    'F8_prune12': ('Conservador; incremento exploratorio', '84 videos mejoran y 4 empeoran frente a fork8. La ganancia incremental es pequena y casi desaparece al retirar los cinco mejores aportes.'),
    'F8_joint': ('Exploratorio de baja prioridad', 'La ganancia incremental procede de un video y tres empeoran. Conserva la poda original; permite probar la correccion sin poda 12.'),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    queue_path = RES / 'submission_queue.json'
    assert not queue_path.exists(), 'Queue already frozen; inspect it instead of overwriting submission state.'
    ledger = load(RES / 'ledger.json')
    ranking = {r['name']: r for r in load(RES / 'ranking.json')['ranking']}
    inventory = load(RES / 'verified_inventory.json')
    assert inventory['distinct_visible_csv_count'] == 6
    assert not inventory['duplicate_visible_outputs']
    for lab in ('a', 'b'):
        r = load(RES / f'lab_{lab}_completed.json')
        assert r['manifest_verified'] and r['complete']['videos'] == 199
    artifacts = {}
    # Re-hash the six downloaded CSVs and frozen notebooks before selecting five.
    for r in inventory['verified_candidates']:
        name = r['candidate']
        assert r['gate'] == 'PASS' and r['csv_bounds_and_graph'] == 'PASS'
        assert r['leaderboard_submitted'] is False and r['version'] == 1
        if name == 'C3_fork8':
            output = ROOT / 'outputs/e068_review/final_c3_fork8'
            folder = ROOT / 'kaggle/x138_xr/k_c3_fork8'
            receipt = ROOT / 'results/E068/c3_fork8_verified.json'
        else:
            run = ledger['runs'][name]
            assert run['status'] == 'COMPLETE' and run['download_verified']
            assert run['kernel'] == r['kernel'] and run['version'] == r['version']
            output = ROOT / 'outputs/e070' / name
            folder = Path(run['folder'])
            receipt = RES / f'{name}_verified.json'
        csv = output / 'submission.csv'
        assert sha(csv) == r['csv_sha256']
        assert sha(folder / 'notebook.ipynb') == r['notebook_sha256']
        assert load(output / 'run_manifest.json') == load(folder / 'expected_manifest.json')
        artifacts[name] = dict(r, file_name='submission.csv', local_file=str(csv),
                               verification_receipt=str(receipt),
                               notebook_url='https://www.kaggle.com/code/' + r['kernel'])
    assert len({artifacts[n]['csv_sha256'] for n in ORDER}) == 5
    now = datetime.now(timezone.utc).isoformat()
    selected = []
    for i, name in enumerate(ORDER, 1):
        label, reason = RATIONALE[name]
        selected.append(dict(artifacts[name], order=i, evidence_label=label,
                             selection_reason=reason, local_evidence=ranking[name]))
    reserve = dict(artifacts['F8_p8'], selection_reason=(
        'Reserva fuera de los cinco: +0.000039410 sobre fork8, 23 videos mejoran y 99 empeoran; '
        'al retirar los cinco mejores aportes el incremento cae a -0.000302281.'))
    queue = dict(experiment='E070', ready_at_utc=now, status='READY',
                 reset_utc=ledger['reset_utc'], reset_lima='2026-09-27T19:00:00-05:00',
                 competition='biohub-cell-tracking-during-development',
                 auto_submit=False, leaderboard_submissions_made=0,
                 policy='Preparation heartbeat does not submit. On an authorized submission turn, check current quota, prior submissions and version before sending each candidate.',
                 score_warning='199 reused training/tuning videos. Local scores and bootstrap intervals are diagnostics, not public forecasts or private guarantees. The confirmed public reference c3 scored 0.955.',
                 candidates=selected, reserves=[reserve],
                 rejected={
                     'F8_curvature': 'Negative increment over fork8: -0.000007659.',
                     'F8_outlier': 'Negative increment over fork8: -0.000162693; incremental CI90 entirely below zero.',
                     'F8_end04': 'Tiny aggregate gain, 175 losses vs 14 wins; negative increment after removing top five contributions.'})
    save(queue_path, queue)
    ledger.update(selection_pending=False, final_selected=ORDER, final_ready_count=5,
                  technically_verified_count=6, technically_verified_candidates=list(artifacts),
                  selection_remaining='None. Five-candidate queue frozen; pause preparation heartbeat. No leaderboard sends from this follow-up.',
                  selection_completed_at_utc=now, queue_file='results/E070/submission_queue.json',
                  prepared_reserves=['F8_p8'], next_gpu_pair_if_no_negative_evidence=[],
                  next_gpu_priority=[], heartbeat_pause_required=True)
    ledger['next_gpu_decision']['status'] = 'VERIFIED_AND_SELECTED'
    save(RES / 'ledger.json', ledger)
    save(RES / 'readiness_progress.json', dict(
        checked_at_utc=now, target=5, selected_ready=5, technically_verified=6,
        distinct_visible_csv_count=6, evaluation_status='COMPLETE; queue frozen',
        selected_candidates=ORDER, pending_gpu=[], leaderboard_submissions_made=0,
        heartbeat_pause_required=True))
    table = '\n'.join(
        f"| {i} | {name} | {ranking[name]['score']:.9f} | {ranking[name]['delta']:+.9f} | {RATIONALE[name][0]} |"
        for i, name in enumerate(ORDER, 1))
    report = f'''# E070 — cinco candidatos listos

Cierre: {now}. Cinco candidatos elegidos entre seis inferencias completas, todas distintas y verificadas. Ningún envío al leaderboard realizado por E070. Reset objetivo: **28-09-2026 00:00 UTC / 27-09 19:00 Lima**.

## Orden propuesto

| Orden | Candidato | Score local | Diferencia local frente a c3 | Evidencia |
|---|---|---:|---:|---|
{table}

La cola ejecutable está en `results/E070/submission_queue.json`: kernel y versión exactos, archivo, hashes de código/CSV, recibos y justificación. El orden prioriza DC030 y luego mide fork8, el cambio común a toda la tanda. Las inferencias ya están ejecutadas; no hace falta relanzarlas para enviar su `submission.csv` v1.

## Qué encontramos

Los dos laboratorios CPU terminaron: ocho variantes y dos controles, estos últimos repetidos en ambos laboratorios. Cada configuración cubre 199 videos. Los controles coinciden por video entre laboratorios y con la referencia anterior. La caché solo reutiliza grafos finales idénticos, incluidas coordenadas redondeadas y enlaces.

DC030 sube de 0.928395837 (fork8) a 0.928908911 local: +0.000513074. Mejora en ambos grupos, con 36 videos que ganan y siete que pierden. Sus divisiones TP/FP/FN cambian de 24/49/127 a 23/33/128: elimina 16 falsas y pierde una verdadera. Su intervalo bootstrap incremental del 90% es [-0.000663242, +0.001417651], así que es una hipótesis prometedora, todavía incierta.

La descomposición de la métrica atribuye +0.000500000 de esa ganancia a divisiones y solo +0.000013074 a enlaces. DC030 mejora principalmente las decisiones de división; no demuestra una mejora amplia del seguimiento. Joint también depende de divisiones: +0.000121212 en ese componente y -0.000022061 en enlaces. Poda 12 aporta +0.000074358 en enlaces y no cambia las divisiones. Esta diferencia justifica conservar la combinación, además de las variantes individuales. Evidencia: `results/E070/increment_components.json`.

Fork8 tiene una señal local más extendida frente a c3: 100 videos mejoran y tres empeoran; conserva las 24 divisiones verdaderas y reduce falsas de 62 a 49. Poda 12 agrega una ganancia pequeña, con 84 mejoras y cuatro pérdidas sobre fork8. La combinación con joint queda segunda por score local, pero su intervalo incremental también cruza cero.

Joint solo mejora un video y empeora tres frente a fork8; queda último en prioridad. En el test visible cambia coordenadas de 125 nodos, sin cambiar enlaces. La combinación con poda 12 permite probar ese mecanismo con un filtrado distinto. No son cinco modelos independientes: comparten la misma base y sus errores estarán correlacionados.

## Descartes y reserva

- P8 está ejecutado y verificado como sexta reserva. Mejora solo +0.000039410 sobre fork8; 23 videos mejoran y 99 empeoran. Al retirar los cinco mayores aportes, el incremento cae a -0.000302281. DC030 ocupa su plaza.
- Curvatura: -0.000007659 frente a fork8. No promover.
- Corrección de posiciones atípicas: -0.000162693 frente a fork8; intervalo incremental completamente negativo. No promover.
- Corte final 0.4: mejora agregada mínima, con 175 videos que empeoran y 14 que mejoran frente a fork8; tampoco merece otra inferencia.

## Validación y límites

Los seis CSV tienen hashes diferentes. Los cinco seleccionados pasan manifiesto, identidad del notebook congelado, hash de fuente, validación estructural y límites físicos del CSV, conteos por video y GATE del log. Las métricas corrieron en CPU; las inferencias en GPU, con máximo dos sesiones simultáneas.

Estos 199 videos se reutilizaron para desarrollo y selección. Los intervalos y las pruebas retirando videos son diagnósticos, no validación independiente ni probabilidades de subir en Kaggle. El **0.955 público corresponde a c3**; ningún score E070 local debe sumarse a él para prometer un resultado público.

Antes de enviar en una acción autorizada, comprobar cuota y envíos anteriores, y usar kernel/versión/archivo de la cola. No enviar hashes duplicados. El heartbeat de preparación debe quedar desactivado al cerrar la cola y no envía automáticamente al leaderboard.

Evidencia: `results/E070/ranking.json`, `verified_inventory.json`, `submission_queue.json`, recibos `*_verified.json`, `lab_*_completed.json` y `outputs/e070/lab_*/e070_*/robustness.json`.
'''
    (ROOT / 'docs/E070_RESULTS.es.md').write_text(report, encoding='utf-8')
    (ROOT / 'docs/E070_EXECUTION.es.md').write_text(f'''# E070 — preparación completada

Actualizado: {now}. **Cinco candidatos elegidos, ejecutados y verificados**, más P8 como reserva técnica. Todos los kernels registrados terminaron; no relanzarlos. Ambos laboratorios CPU están descargados y analizados.

La fuente operativa es `results/E070/submission_queue.json`; incluye el orden DC030, C3_fork8, prune12_joint, prune12 y joint, con kernels v1, CSV y hashes. Resultados y límites: `docs/E070_RESULTS.es.md`. Registro de ejecuciones: `results/E070/ledger.json`.

Reset previsto: 28-09-2026 00:00 UTC / 27-09 19:00 Lima. **Este seguimiento prepara, no envía al leaderboard.** No se ha consumido ninguna submission en E070. En una acción posterior de envío, verificar cuota y envíos previos antes de usar la cola; no volver a ejecutar inferencias ni enviar duplicados.

El heartbeat `biohub-completar-cinco-submissions` se configuró cada 25 minutos por petición de Arturo. Con la cola cerrada corresponde desactivarlo y registrar la respuesta de la herramienta. No abrir más experimentos desde este seguimiento.

Para auditar: Python ML `C:/dev/biohub/.venv/Scripts/python.exe`; SDK `C:/dev/rsna-knee/.venv/Scripts/python.exe`. `analyze_e070.py` reproduce ranking e inventario; `finalize_e070.py` verificó todos los hashes antes de congelar la cola y rechaza sobrescribir una cola ya creada.
''', encoding='utf-8')
    print(json.dumps(dict(status='READY', selected=ORDER, reserve='F8_p8', auto_submit=False)))


if __name__ == '__main__':
    main()
