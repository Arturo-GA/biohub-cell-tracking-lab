# E051–E054: comparación pública y combinación de componentes

La submission E050 **56409893 terminó con 0.946**, igual que el control y la
asociación visual anterior. Su ganancia local no se trasladó al leaderboard.
La revisión del 21 de septiembre consulta de nuevo el listado de Kaggle y
descarga seis códigos. Fuentes y hashes: `results/E051_public_audit.json`.

## Diferencias comprobadas

Los notebooks de [zhehaoliang](https://www.kaggle.com/code/zhehaoliang/biohub-p26-o01-exact-public-0947)
y [reyhanksatria](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb)
usan umbral DeepCenter de divisiones **0.20**, frente a nuestro **0.25**.
El radio de reasignación **5.5 µm**, procedente del resultado de barrido público
de Harmonic V3/proxy, ya estaba presente en E050; no fue suficiente para subir.
La arquitectura principal, los dos modelos y buena parte de su fusión ya
coinciden con nuestra base. La diferencia de score no demuestra por sí sola
cuál ajuste la causa.

Se inspeccionan también las nuevas ideas de
[mezcla adaptativa](https://www.kaggle.com/code/yudaiyamauchi/lb-exploration-c-disagreement-adaptive),
[mejor enlace mutuo](https://www.kaggle.com/code/yudaiyamauchi/lb-exploration-e-mutual-best),
[ReID](https://www.kaggle.com/code/arnav170/biohub-reid3s) y
[fine tuning del Transformer](https://www.kaggle.com/code/noisyislands/biohub-transformer-finetune).
No se atribuye una mejora verificada a estas propuestas: ReID tiene peso
predeterminado cero, y el fine tuning requiere revisar sus negativos sintéticos
porque las anotaciones son parciales.

## E051: mezcla de asociaciones en CPU

Adaptación propia de las ideas anteriores sobre probabilidades capturadas,
no reproducción exacta de la modificación pública de logits antes del ILP.
Se combinan Transformer, movimiento local y apariencia de parches; el margen
entre los dos mejores candidatos determina pesos observables. Se conserva al
menos la mitad de peso Transformer y se bloquean las divisiones originales.

| Variante | Score local | Enlaces TP / FP / FN |
|---|---:|---:|
| Harmonic original | 0.943682890 | 4699 / 171 / 166 |
| Asociación visual anterior | **0.950283267** | **4715 / 153 / 150** |
| Mejor enlace mutuo | 0.950080404 | 4714 / 153 / 151 |
| Pesos adaptativos | 0.947761708 | 4710 / 161 / 155 |
| Adaptativo + mutuo | 0.948527657 | 4711 / 158 / 154 |

Las tres propuestas nuevas se rechazan. No recuperan divisiones; todas quedan
en 0 TP / 2 FP / 5 FN. Proceso CPU: 277.935 segundos; GPU: cero.
Estos ocho videos ya se han reutilizado; no constituyen validación independiente.

## E052: reproducir el posprocesado público

Se extraen una vez mapas float32 del DeepCenter público con sus ocho vistas
TTA, checkpoint verificado por SHA256 y carga estricta: **163.546 segundos de
proceso GPU**. CPU combina umbrales 0.25/0.20 y radios 6.0/5.5, con y sin
asociación visual. Se utiliza el grafo crudo, se vuelve a aplicar todo el
posprocesado y se remapean las probabilidades capturadas a cada grafo resultante.
No se reutilizan IDs de nodos reparados de otra variante.

La primera evaluación CPU falló por un módulo omitido en el paquete; se corrigió
sin repetir la extracción GPU. La ejecución secuencial tarda aproximadamente
90 segundos por variante y video; se preparó una reproducción con cuatro
procesos CPU. El control debe reproducir exactamente ambos scores E023 antes
de permitir promoción.

## E053: ensayo conjunto de los componentes

Se extraen campos del detector denso E041 y del detector estático E039 sobre
los mismos ocho videos. El primer intento detectó ausencia de `zarr` antes
de inferir; la recuperación lee directamente los bloques de imagen como el
código público y verifica su forma. Extracción correcta: **57.063 segundos de
proceso GPU**. La normalización y conversión float16 reproducen E047/E050.

La comparación CPU aplica puentes densos y puentes ponderados a cada grafo
E052, conservando nodos originales. La selección considera también todos los
controles sin puentes, de modo que puede rechazar el ensamble. Se informa el
resultado de cada combinación; no se suman ganancias obtenidas por separado.

Los tiempos citados son de proceso, no una medición de facturación. No se
reentrena ningún modelo ni se utiliza Colab. El lector del posprocesado público
omite instalación e inferencia principal; la optimización de búsqueda de pesos
prioriza rutas existentes y conserva la verificación de hashes.

## Resultados conjuntos

Pendientes de completar la evaluación CPU. El constructor E054 exige una
mejora frente a la asociación visual y un recibo de selección antes de crear
el notebook final. No se afirma mejora de leaderboard ni se ha enviado E054.
