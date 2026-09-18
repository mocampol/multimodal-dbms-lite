
## Tabla resumen

| Técnica | Ventajas | Desventajas | Cuándo usarla |
|---|---|---|---|
| **B+ clusterizado** | Igualdad, rango y orden en O(log n) + secuencial; el scan ordenado es prácticamente gratis. | Solo una columna por tabla puede ser clusterizada; construir/mantener el índice implica reescribir el archivo de datos ordenado. | Columna de acceso más frecuente y con muchas consultas por rango u `ORDER BY` (típicamente la PK). |
| **B+ no clusterizado** | Igualdad, rango y orden en O(log n); se pueden tener varios por tabla sin reordenar los datos. | Cada resultado implica un salto adicional al Heap File (menos localidad que el clusterizado); espacio extra por cada índice. | Columnas secundarias con búsquedas por rango u orden frecuentes, cuando no pueden ser la columna clusterizada. |
| **Hash dinámico** | Igualdad en O(1) amortizado; se adapta al crecimiento de datos sin reorganización manual. | No soporta rango ni orden (N/A); no aprovechable si la consulta necesita `ORDER BY`/`BETWEEN`. | Columnas con búsquedas exclusivamente por igualdad exacta y alto volumen (ej. claves foráneas, lookups puntuales). |
