# Reporte Experimental: Indexación SPIMI, BM25 y Similitud del Coseno (TF-IDF)

Este informe resume los resultados experimentales de **tiempo de construcción de índice** y **latencia de consultas con 20 repeticiones estadísticas** para $N \in [1,000, 10,000, 100,000]$.

## 1. Tiempo de Creación del Índice (Index Build Time & SPIMI Breakdown)

| Corpus | N (Docs) | Tiempo Total (s) | RAM (s) | Spill I/O (s) | Merge Heap (s) | Bloques | Vocabulario | Rendimiento (docs/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spanish | 1,000 | **0.3518s** | 0.2605s | 0.0397s | 0.0489s | 1 | 27,060 | 2,842 docs/s |

## 2. Latencia de Consulta: BM25 y Similitud Coseno vs Escaneo Secuencial (20 Pasadas)

| Corpus | N | Tipo | Consulta | BM25 Media (ms) | Coseno Media (ms) | Secuencial (ms) | Speedup Coseno |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spanish | 1,000 | 1-term (high freq) | `gobierno` | 0.121 ms | 0.137 ms | 217.22 ms | **1590.2x** |
| spanish | 1,000 | 1-term (high freq) | `espana` | 0.157 ms | 0.164 ms | 217.71 ms | **1329.9x** |
| spanish | 1,000 | 1-term (low freq) | `submarino` | 0.006 ms | 0.003 ms | 218.13 ms | **64154.8x** |
| spanish | 1,000 | 1-term (med freq) | `tribunal` | 0.055 ms | 0.050 ms | 217.64 ms | **4335.5x** |
| spanish | 1,000 | 2-terms | `economia mercado` | 0.050 ms | 0.054 ms | 223.59 ms | **4133.0x** |
| spanish | 1,000 | 2-terms | `partido politica` | 0.116 ms | 0.119 ms | 218.51 ms | **1837.8x** |
| spanish | 1,000 | 3-terms | `crisis laboral empleo` | 0.086 ms | 0.076 ms | 220.16 ms | **2881.7x** |
| spanish | 1,000 | 3-terms | `presidente ministro parlamento` | 0.116 ms | 0.122 ms | 219.56 ms | **1804.1x** |
| spanish | 1,000 | 4-terms | `justicia juez audiencia nacional` | 0.176 ms | 0.165 ms | 220.96 ms | **1339.2x** |
| spanish | 1,000 | out of vocab | `terminoinexistente999` | 0.004 ms | 0.000 ms | 224.07 ms | **746891.7x** |
