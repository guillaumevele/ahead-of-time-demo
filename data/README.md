# Data

`eco2mix_national_tr_snapshot.csv` is a frozen extract of the dataset
"Données éCO2mix nationales temps réel" published by RTE on Open Data Réseaux Énergies
(https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/), under the
Licence Ouverte v2.0 (Etalab).

- columns kept: `date_heure` (UTC) and `consommation` (national load, MW, 15-minute step);
- range: 2026-09-01 00:00 UTC to 2026-11-18 23:45 UTC, 7580 rows;
- downloaded on 2026-11-19 with `scripts/fetch_data.sh`. Real-time values are provisional
  and RTE may revise them, so a fresh download can differ from this snapshot. The backtest
  prints the SHA-256 of the file it used.
