# Reporte Experimental: Indexación SPIMI, BM25 y Similitud del Coseno (TF-IDF)

Este informe resume los resultados experimentales de **tiempo de construcción de índice** y **latencia de consultas con 20 repeticiones estadísticas** para $N \in [1,000, 10,000, 100,000]$.

## 1. Tiempo de Creación del Índice (Index Build Time & SPIMI Breakdown)

| Corpus | N (Docs) | Tiempo Total (s) | RAM (s) | Spill I/O (s) | Merge Heap (s) | Bloques | Vocabulario | Rendimiento (docs/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spanish | 1,000 | **0.3643s** | 0.2567s | 0.0422s | 0.0627s | 1 | 27,060 | 2,745 docs/s |

## 2. Latencia de Consulta: BM25 y Similitud Coseno vs Escaneo Secuencial (20 Pasadas)

| Corpus | N | Tipo | Consulta | BM25 Media (ms) | Coseno Media (ms) | Secuencial (ms) | Speedup Coseno |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spanish | 1,000 | 1-term (high freq) | `gobierno` | 0.126 ms | 0.149 ms | 217.49 ms | **1459.7x** |
| spanish | 1,000 | 1-term (high freq) | `espana` | 0.153 ms | 0.164 ms | 217.32 ms | **1323.5x** |
| spanish | 1,000 | 1-term (low freq) | `submarino` | 0.006 ms | 0.003 ms | 217.90 ms | **68093.5x** |
| spanish | 1,000 | 1-term (med freq) | `tribunal` | 0.044 ms | 0.046 ms | 218.57 ms | **4772.3x** |
| spanish | 1,000 | 2-terms | `economia mercado` | 0.050 ms | 0.054 ms | 219.70 ms | **4046.0x** |
| spanish | 1,000 | 2-terms | `partido politica` | 0.120 ms | 0.122 ms | 219.16 ms | **1797.9x** |
| spanish | 1,000 | 3-terms | `crisis laboral empleo` | 0.081 ms | 0.075 ms | 219.25 ms | **2927.2x** |
| spanish | 1,000 | 3-terms | `presidente ministro parlamento` | 0.112 ms | 0.119 ms | 220.00 ms | **1853.5x** |
| spanish | 1,000 | 4-terms | `justicia juez audiencia nacional` | 0.173 ms | 0.170 ms | 222.50 ms | **1306.5x** |
| spanish | 1,000 | out of vocab | `terminoinexistente999` | 0.004 ms | 0.000 ms | 224.59 ms | **1122956.5x** |
