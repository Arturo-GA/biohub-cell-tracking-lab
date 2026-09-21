# Procedencia

- E024 usa la arquitectura de **NucVerse3D**, Vergara, Perez-Gallardo, Velasco y colaboradores, desde [Segovia-lab/NucVerse3D](https://github.com/Segovia-lab/NucVerse3D), commit `d809a2e6cf380342708b7a9107574b259e6b34eb`. La copia sin cambios de la arquitectura y su licencia MIT están en `vendor/nucverse/`. El checkpoint generalista `resunet_combined_1000/best_model.weights.h5` procede de [Zenodo 18517324](https://doi.org/10.5281/zenodo.18517324), publicado bajo CC-BY-4.0; SHA256 `38e65c00b26413c37c8959392d96d10e3d4ed7cb8f8c21ff4c33f740969b406c`. Los pesos permanecen fuera de Git. `nucverse_instances.py` adapta su agrupación por gradiente para guardar solo la posición final; `nucverse_tiles.py` implementa inferencia por bloques con promedio uniforme en solapes y normalización percentil, sin su deconvolución. No se afirma reproducir exactamente todo el pipeline publicado.

- `visual_sequence.py`, `visual_replay.py`, `visual_capture.py` y `visual_candidates.py` son adaptadores y optimizadores propios. E021 carga en runtime la arquitectura pública `SimpleNodeTransformer` y el checkpoint secundario ya incluidos en los recursos de Harmonic, sin redistribuir pesos. E022 instrumenta la inferencia congelada para guardar alternativas antes de su filtrado. Se mantienen las atribuciones de Harmonic/Pilkwang indicadas abajo; estas pruebas no se presentan como implementaciones de Trackastra, Ultrack ni FOCUS-3D.

- `baseline/harmonic_inference.py` deriva de las celdas 0–6 de **Biohub Harmonic Fusion**, autor Igor Zharov (`flexonafft`), descargado de [Kaggle](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) el 2026-09-13. Se conservan las atribuciones del código; utiliza modelos y código de soporte públicos de Pilkwang y una regla de fusión atribuida por su autor a Yusuke Togashi. Se eliminaron las etapas posteriores de validación/sweep y se normalizaron los imports de `__future__` para poder ejecutarlo como script. Las afirmaciones de puntaje que aparezcan en logs o comentarios heredados son del autor original y pueden ser históricas.
- El archivo entregado por el usuario corresponde a **improved-metric-hack-last-call**, Aman Atar (`amanatar`), [fuente](https://www.kaggle.com/code/amanatar/improved-metric-hack-last-call). Se analizó; sus nodos hub y fake forks no se incorporan a los candidatos.
- `src/biohub_official/metrics.py` y `division_metrics.py` son copias del [repositorio oficial RoyerLab](https://github.com/royerlab/kaggle-cell-tracking-competition) en el commit indicado en `baseline/provenance.json`. Su licencia BSD-3-Clause se incluye en `src/biohub_official/LICENSE`.
- Las adiciones de `src/biohub_lab`, los generadores, las pruebas y los documentos se crearon para este proyecto. La regla Division Guard es una hipótesis propia inspirada por el problema de asociación en divisiones; **no implementa HOCT**.
- Los tres datasets de modelos permanecen en Kaggle. La API informó CC0-1.0 para el soporte de Pilkwang; las condiciones de otros modelos se revisan individualmente antes de redistribuirlos. Este repositorio no aplica una licencia general a código o pesos ajenos. La licencia específica del notebook Harmonic no fue devuelta por `kernels pull`; se conserva la atribución y el repositorio es privado.
- Referencias originales descargadas y checkout de investigación permanecen en carpetas ignoradas. `baseline/provenance.json` registra SHA256 para detectar cambios.
- `hoct_linker.py` usa el checkpoint público general_v1 de [HOCT](https://github.com/royerlab/hoct), Jordão Bragantini y colaboradores, y adapta su orden de características, estadísticas de normalización y softmax parental. Código fuente fijado en `2ccc5040823bc944ab67790abd1f56eea7cd4f05`; licencia MIT completa en `licenses/HOCT.txt`. Los pesos públicos se alojan separadamente en un dataset privado de Kaggle, sin datos ni predicciones del usuario. El adaptador, segmentación y matching son implementaciones de este proyecto; no reproducen el solver de tracklets de dos pasadas del original.
- Los módulos `temporal_*` son implementaciones propias. La red de imágenes, la atención entre candidatos, la cabeza simétrica de divisiones y el optimizador se entrenaron/desarrollaron para E004. Trackastra es una referencia conceptual; no se copiaron sus pesos ni su implementación. El generador sintético de José Freitas solo se revisó como posible fuente futura y no intervino en el entrenamiento ni en las evaluaciones de E004.
- `src/biohub_cellect/model.py` es una copia sin modificaciones de `unetext3Dn_con7.py` de [CELLECT](https://github.com/zzz333za/CELLECT), commit `3586070926f7f1fd5d8df37456861d22bdc63236`, SHA256 `3b44f7c64c4b948ab7cc14bb32d5a86f61d3f4cff0ee3425a5ec32cc6c17d3ba`. Licencia GPL-2.0 incluida en `licenses/CELLECT.txt`. `cellect_detector.py` adapta su preprocesamiento y extracción de máximos de localización descritos en `inference.py` y `recoloss.py`; conserva su procedencia y licencia aplicable. Utiliza el checkpoint público `U-ext+-x3rd-149.0-4.6540.pth`, SHA256 `f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5`. El dataset privado de Kaggle contiene solamente estos pesos públicos, licencia y procedencia. No se reproduce el tracker CELLECT completo: se sustituyen su MLP de exclusión y su asociación por NMS físico, corroboración temporal y el pipeline de Harmonic. La pertenencia de Biohub al entrenamiento original de CELLECT no se ha verificado.
- `gaussian_detector.py` y el adaptador de propuestas de E008/E009 son implementaciones de este proyecto. El ajuste de una o dos gaussianas no utiliza pesos de CELLECT, HOCT ni NucVerse. No se presenta como reproducción de Ultrack ni de otro paper.
- El empaquetado normaliza CRLF a LF sin cambiar la lógica del modelo CELLECT. Su hash canónico usado en runtime es `bb0902db6161fa8e8fb4eac5ff58d0b6cc4bcf0754cfaecdbc5128f8dc9724da`; el hash de descarga original indicado arriba corresponde a los bytes CRLF del autor.
- `joint_lineage.py` (E010) es un optimizador propio de hipótesis de centros y eventos temporales. La selección conjunta de hipótesis de segmentación y enlaces de [Ultrack, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0) es una referencia conceptual; no se copian su código ni sus pesos, ni se reproduce su implementación completa. E010 reutiliza las propuestas verificadas de E008/E009 como entradas, sin incluir en su paquete el modelo CELLECT ni sus pesos.
- `detection_dag.py` y `dag_coverage.py` (E011) son implementaciones propias de representación de parejas, flujo para dos trayectorias disjuntas y auditoría de cobertura. Reutilizan los adaptadores de detección CELLECT y gaussianas documentados arriba. E011 incluye el código y licencia GPL-2.0 de CELLECT y adjunta sus pesos públicos ya verificados; no utiliza predicciones de Harmonic ni los centros anotados de la preparación temporal como entrada del generador.
- `harmonic_centers.py` (E012) extrae en runtime las funciones de detección del código de Pilkwang con los parches de Harmonic ya atribuidos arriba, eliminando la predicción de enlaces y usando un lector de metadatos de imagen sin GEFF. Reutiliza los dos checkpoints temporales públicos de Harmonic. Su combinación de centros y comparación de cobertura son adaptadores de este proyecto. E012 consume las propuestas verificadas de E011; no redistribuye sus datos, los pesos ni las correspondencias con anotaciones.
- Los módulos `event_*` (E013) implementan un selector propio con características congeladas de los dos UNet públicos, atención local, cabezas de enlaces/divisiones y selección temporal mediante MILP. Reutilizan la detección combinada de E012 y el adaptador CELLECT con su licencia. No se presentan como reproducción de otro tracker. El paquete enviado a Kaggle incluye código, licencias, configuración y hashes; las imágenes, anotaciones y pesos se leen de fuentes adjuntas y no se incorporan al paquete ni a Git.
# Cellpose-DINO experimental dependency (E028)

E028 uses the Cellpose 4.2.1.1 package and public `cpdino-vitb` checkpoint from
[MouseLand](https://github.com/MouseLand/cellpose), with the
[DINOv3 code](https://github.com/facebookresearch/dinov3) dependency.
Authors and references: [Cellpose model documentation](https://cellpose.readthedocs.io/en/latest/models.html)
and [public model card](https://huggingface.co/mouseland/cellpose-sam).
The model card lists BSD-3-Clause; Cellpose documents CC-BY-NC for its original
training datasets. Those datasets are not redistributed by this repository.
Dependency wheels retain their own licenses; weights and wheels are outside Git.
Pinned revisions, hashes, compatibility handling, and experimental scope are
recorded in `baseline/e028_cellpose.json` and `docs/E028_CELLPOSE_DINO.es.md`.

## SpatialDINO y experimentos E038–E040

E038 utiliza el checkpoint público de [SpatialDINO](https://github.com/kirchhausenlab/spatialdino),
Kirchhausen Lab, `step=249999/backbone.pth`, servido por su bucket público AWS.
El adaptador de inferencia `spatial_probe.py` implementa las operaciones de su
ViT-S/8 3D con SDPA de PyTorch. Arquitectura contrastada con el commit
`ca3ab86b34430d963f12a3909baaeb9343c63b7d`; el repositorio anuncia licencia MIT.
Los archivos de referencia conservan sus avisos originales en una carpeta
ignorada. Los pesos no se incluyen en Git y no se infiere una licencia propia
para ellos a partir de la licencia del código. La procedencia, hash y límites
de la adaptación constan en `docs/E038_E040_TRES_LINEAS.es.md`.

E039 y E040 son modelos propios entrenados desde cero: campo 3D hacia centros
con supervisión parcial y consulta de imágenes de madre/dos hijas,
respectivamente. No se presentan como reproducciones de SpatialDINO, HOCT,
Trackastra ni de otro método publicado.

## NIS3D y experimentos E041–E043

E041 usa imágenes y anotaciones externas de **NIS3D: A Completely Annotated
Benchmark for Dense 3D Nuclei Image Segmentation**, NeurIPS 2023.
[Repositorio de los autores](https://github.com/yu-lab-vt/NIS3D),
[registro de datos 11456029](https://zenodo.org/records/11456029), licencia
CC-BY-4.0 según los metadatos del registro. Se generan recortes normalizados,
mapas de centros y máscaras de confianza, y se corrige la anisotropía usando
los `Info.txt`. Datos y pesos quedan fuera de Git; sus hashes se registran.

E042 es un control propio de información espacial. E043 es una implementación
propia de selección entera de eventos y perturb-and-MAP, inspirada en la línea
de seguimiento con incertidumbre; se revisó el repositorio público
[NabaviLab](https://github.com/NabaviLab/bayesian-transformer-cell-tracking).
No se incorpora su código ni se afirma reproducir su modelo bayesiano.

E048 includes a checkpoint-compatible implementation of the public DivNet UNet
architecture described in https://www.kaggle.com/code/canhtoanle/biohub-div-complete-v33a
and the configuration of https://www.kaggle.com/datasets/giorgosi/biohub-divnet-v2 .
The model weights remain in Kaggle/ignored local artifacts and are not committed.
The implementation uses strict state-dict validation. Public scores and training
membership are not independently established; the diagnostic is explicitly conditional.
The research audit executes only reviewed model class/preprocessing definitions;
no full external notebook or out-of-volume metric manipulation is incorporated.
