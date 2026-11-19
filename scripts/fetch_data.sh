#!/bin/sh
# Re-download both series. Overwrites the snapshots: backtest numbers may then change.
set -eu
cd "$(dirname "$0")/.."
curl -fsS -o data/eco2mix_national_tr_snapshot.csv \
  "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/csv?select=date_heure%2Cconsommation&where=consommation%20is%20not%20null%20and%20date_heure%20%3E%3D%20%272026-09-01T00%3A00%3A00Z%27%20and%20date_heure%20%3C%20%272026-11-19T00%3A00%3A00Z%27&order_by=date_heure&delimiter=%2C&timezone=UTC"
shasum -a 256 data/eco2mix_national_tr_snapshot.csv
curl -fsS -o data/paris_bike_sebastopol_sn_snapshot.csv \
  "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/exports/csv?select=date%2Csum_counts&where=id_compteur%3D%27100057445-104057445%27%20and%20date%20%3E%3D%20%272026-09-01T00%3A00%3A00Z%27%20and%20date%20%3C%20%272026-11-19T00%3A00%3A00Z%27&order_by=date&delimiter=%2C&timezone=UTC"
shasum -a 256 data/paris_bike_sebastopol_sn_snapshot.csv
