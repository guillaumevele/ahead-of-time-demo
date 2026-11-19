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

`paris_bike_sebastopol_sn_snapshot.csv` is a frozen extract of the dataset
"Comptage vélo - Données compteurs" published by the Ville de Paris
(https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/), made
available under the Open Database License (ODbL) v1.0. Source: Ville de Paris, Direction
de la Voirie et des Déplacements.

- one counter: "Totem 73 boulevard de Sébastopol S-N" (`100057445-104057445`), the counter
  with the highest total over September 2026;
- columns kept: `date` (UTC, hourly) and `sum_counts` (bikes counted in the hour);
- range: 2026-09-01 00:00 UTC to 2026-11-18 22:00 UTC, 1822 rows (74 hours absent from the source);
- downloaded on 2026-11-19 with `scripts/fetch_data.sh`.
