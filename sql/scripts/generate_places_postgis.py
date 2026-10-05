#!/usr/bin/env python3
"""
Script para generar inserts masivos de puntos espaciales aleatorios en Lima Metropolitana
para la tabla places_postgis con tipo GEOMETRY.
"""

import random
from pathlib import Path

# Clusters urbanos representativos de Lima Metropolitana (evitando mar y cerros despoblados)
CLUSTERS = [
    {
        "zona": "Lima Centro (Cercado, Breña, La Victoria, Lince)",
        "center_lon": -77.035,
        "center_lat": -12.060,
        "spread_lon": 0.025,
        "spread_lat": 0.025,
        "weight": 25,
    },
    {
        "zona": "Lima Sur (Miraflores, San Isidro, Surco, Barranco, San Borja)",
        "center_lon": -77.015,
        "center_lat": -12.115,
        "spread_lon": 0.035,
        "spread_lat": 0.030,
        "weight": 35,
    },
    {
        "zona": "Lima Este (Ate, Santa Anita, El Agustino, La Molina)",
        "center_lon": -76.945,
        "center_lat": -12.050,
        "spread_lon": 0.035,
        "spread_lat": 0.030,
        "weight": 20,
    },
    {
        "zona": "Lima Norte (Los Olivos, Independencia, San Martin de Porres, Comas)",
        "center_lon": -77.060,
        "center_lat": -11.965,
        "spread_lon": 0.030,
        "spread_lat": 0.035,
        "weight": 20,
    },
]

TIPOS = [
    "Parque", "Farmacia", "Restaurante", "Tienda", "Estacion Metropolitano",
    "Banco", "Clinica", "Colegio", "Cafe", "Grifo", "Supermercado", "Gimnasio",
    "Libreria", "Panaderia", "Plaza", "Puesto de Salud", "Comisaria"
]

DISTRITOS = [
    "Miraflores", "San Isidro", "Cercado", "Ate", "Surco", "San Borja",
    "Los Olivos", "Barranco", "La Molina", "San Miguel", "Lince", "Jesus Maria",
    "Breña", "Magdalena", "Pueblo Libre", "Santa Anita", "Independencia"
]


def generate_sql(num_records: int = 500, seed: int = 42) -> str:
    random.seed(seed)
    lines = []
    lines.append("CREATE TABLE places_postgis (")
    lines.append("    id INTEGER PRIMARY KEY,")
    lines.append("    nombre VARCHAR(100),")
    lines.append("    geom GEOMETRY")
    lines.append(");")
    lines.append("")
    lines.append("CREATE INDEX idx_places_postgis_geom ON places_postgis(geom) USING RTREE;")
    lines.append("")
    lines.append("BEGIN TRANSACTION;")

    cluster_choices = []
    for c in CLUSTERS:
        cluster_choices.extend([c] * c["weight"])

    for i in range(1, num_records + 1):
        cluster = random.choice(cluster_choices)
        lon = round(random.gauss(cluster["center_lon"], cluster["spread_lon"] / 2.0), 5)
        lat = round(random.gauss(cluster["center_lat"], cluster["spread_lat"] / 2.0), 5)

        # Límites de seguridad para Lima Metropolitana
        lon = max(-77.12, min(-76.88, lon))
        lat = max(-12.22, min(-11.90, lat))

        tipo = random.choice(TIPOS)
        distrito = random.choice(DISTRITOS)
        nombre = f"{tipo} {distrito} #{i}"

        lines.append(
            f"INSERT INTO places_postgis (id, nombre, geom) VALUES ({i}, '{nombre}', POINT({lon}, {lat}));"
        )

    lines.append("END TRANSACTION;")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "generated"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "places_postgis_inserts.sql"

    sql_content = generate_sql(num_records=500)
    out_file.write_text(sql_content, encoding="utf-8")
    print(f"Generado con éxito: {out_file} ({len(sql_content.splitlines())} líneas)")
