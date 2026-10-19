# Reporte Experimental: Indexación SPIMI, BM25 y Similitud del Coseno (TF-IDF)

Este informe resume los resultados experimentales de **tiempo de construcción de índice** y **latencia de consultas con 20 repeticiones estadísticas** para $N \in [1,000, 10,000, 100,000]$.

## 1. Tiempo de Creación del Índice (Index Build Time & SPIMI Breakdown)

| Corpus | N (Docs) | Tiempo Total (s) | RAM (s) | Spill I/O (s) | Merge Heap (s) | Bloques | Vocabulario | Rendimiento (docs/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spanish | 1,000 | **0.3473s** | 0.2585s | 0.0411s | 0.0454s | 1 | 27,060 | 2,880 docs/s |

## 2. Latencia de Consulta: BM25 y Similitud Coseno vs Escaneo Secuencial (20 Pasadas)

| Corpus | N | Tipo | Consulta | BM25 Media (ms) | Coseno Media (ms) | Secuencial (ms) | Speedup Coseno |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spanish | 1,000 | 1-term (high freq) | `gobierno` | 0.123 ms | 0.135 ms | 216.98 ms | **1603.7x** |
| spanish | 1,000 | 1-term (high freq) | `espana` | 0.151 ms | 0.173 ms | 223.32 ms | **1291.6x** |
| spanish | 1,000 | 1-term (low freq) | `submarino` | 0.006 ms | 0.003 ms | 217.56 ms | **67988.9x** |
| spanish | 1,000 | 1-term (med freq) | `tribunal` | 0.045 ms | 0.046 ms | 217.59 ms | **4679.4x** |
| spanish | 1,000 | 2-terms | `economia mercado` | 0.049 ms | 0.054 ms | 218.81 ms | **4074.8x** |
| spanish | 1,000 | 2-terms | `partido politica` | 0.120 ms | 0.116 ms | 218.93 ms | **1884.1x** |
| spanish | 1,000 | 3-terms | `crisis laboral empleo` | 0.081 ms | 0.077 ms | 223.68 ms | **2886.2x** |
| spanish | 1,000 | 3-terms | `presidente ministro parlamento` | 0.113 ms | 0.123 ms | 219.31 ms | **1777.3x** |
| spanish | 1,000 | 4-terms | `justicia juez audiencia nacional` | 0.167 ms | 0.172 ms | 220.85 ms | **1281.8x** |
| spanish | 1,000 | out of vocab | `terminoinexistente999` | 0.004 ms | 0.000 ms | 218.43 ms | **1092158.5x** |
