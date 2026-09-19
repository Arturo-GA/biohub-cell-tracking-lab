# E016: entrenamiento de asociación con identidad única

Lanzado el 19 de septiembre de 2026 como notebook privado `jarturo/biohub-lab-identity-parent-cpu`, versión 1, solo CPU. Reutiliza los grafos y las características visuales E013 verificadas por hash; no ejecuta de nuevo los detectores.

## Cambio estructural

E015 mostró que alternativas de una misma célula podían recibir simultáneamente crédito positivo. E016 utiliza solo detecciones de origen Harmonic ya guardadas, y establece correspondencias únicas por frame con el GT de entrenamiento dentro de 7 µm. Esta correspondencia es nuestra asignación diagnóstica, no una afirmación de equivalencia exacta con tracksdata. Los casos asignados por cercanía aún pueden contener ruido.

Para cada detección hija considera hasta 12 padres a menos de 20 µm en el frame anterior. La red comparte el codificador de apariencia y geometría de E014, pero cambia la supervisión: cross-entropy entre padres candidatos y una opción nula, en lugar de clasificación binaria independiente y ranking de pares. No utiliza pesos entrenados de E014.

Solo hijas asignadas a una célula con padre anotado generan ejemplos. Los candidatos sin correspondencia se enmascaran en la pérdida: no son negativos. La etiqueta nula significa que el padre verdadero no está representado entre esos candidatos, no que la célula sea fondo. En inferencia todos los candidatos disponibles compiten; esta diferencia entre candidatos observables en entrenamiento e inferencia es una limitación que debe medirse.

El decodificador realiza asignación bipartita global con opción de no enlazar y capacidad uno por origen y destino. Esta primera prueba estudia continuaciones; no predice divisiones. Comparte el mismo decodificador con el control geométrico.

## Entrenamiento y comparación

- 48 videos de fit, 3000 pasos, batch 128, AdamW, learning rate inicial 0.0003, semilla 20260919, dos hilos CPU.
- Se selecciona el checkpoint final fijado de antemano. No se consulta GT de calibración durante el entrenamiento ni para elegir checkpoints.
- Se generan y congelan los CSV de ambos métodos antes de evaluar los 16 videos de calibración con la métrica oficial fijada.
- Control geométrico: utilidad `3 - distancia_um / 5`, opción nula de utilidad cero. Es un comparador fijo sobre las mismas detecciones, no el sistema Harmonic completo ni una configuración optimizada por calibración.
- El CSV incluye los nodos que participan en enlaces, conforme al escritor existente. Ambos métodos parten de las mismas detecciones, pero pueden conservar números distintos de nodos activos.
- Los 48 videos de evaluación anteriores no se usan. El detector secundario subyacente tuvo entrenamiento sobre los 199 videos públicos; estos resultados siguen siendo desarrollo condicional.

Una mejora frente al control geométrico no demuestra superar Harmonic completo. Antes de reemplazar el sistema de referencia se necesita una comparación pareada de ese sistema y considerar divisiones. No se envía automáticamente al leaderboard.

## Verificaciones y entrega

Cuatro pruebas superadas: identidad única con duplicados desconocidos enmascarados, etiqueta nula ante padre ausente del conjunto candidato, asignación global con capacidad y opción nula, y entrenamiento CPU sintético con guardado y recarga del checkpoint. La primera ejecución local encontró una restricción de permisos en directorios temporales; las cuatro pruebas pasaron al ejecutarse con los permisos necesarios. Fuentes y notebook compilados; metadatos privados, GPU/TPU desactivados, internet desactivado.

Los checkpoints y las predicciones quedan en los outputs privados de Kaggle. Git almacena código, configuración y recibos, sin imágenes ni pesos. No se configuró monitoreo automático.

Archivos de salida principales: `identity_parent/result.json`, `training.json`, `fit_counts.json`, `last.pt`, `frozen_predictions.json`, `learned_metrics.json`, `geometric_metrics.json`. Al terminar, verificar progreso de 3000 pasos, hashes, scores agregados, cobertura, cantidad de nodos y mejoras por video; no decidir solo por la pérdida de entrenamiento.
