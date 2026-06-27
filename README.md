# Mundial Predictor 2026 ⚽

Modelo de Machine Learning para predecir resultados (Victoria/Empate/Derrota) de partidos de fútbol internacional, con seguimiento en vivo durante el Mundial 2026.

## Motivación

Proyecto de portfolio para practicar el rol de Data Analyst / Data Scientist: ETL de un dataset histórico de 49.000+ partidos, feature engineering futbolístico, entrenamiento y comparación de modelos de clasificación, y validación contra partidos reales del Mundial 2026 en curso.

## Dataset

- **[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)** (martj42, Kaggle) — resultados de partidos internacionales, actualizado a diario.
- **[International Football Elo Ratings](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)** — historial de ELO partido a partido.
- **[2026 FIFA World Cup Historical Elo Ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)** — ELO de respaldo para los 48 clasificados al Mundial 2026.

## Features construidas

| Feature | Descripción |
|---|---|
| `elo_diff` | Diferencia de ELO entre local y visitante (calculada con `merge_asof` para usar el ELO previo al partido, sin data leakage) |
| `home_forma` / `away_forma` | Puntos acumulados en los últimos 5 partidos de cada equipo (rolling window) |
| `neutral` | Si el partido se jugó en cancha neutral |
| `h2h_victorias_a` / `h2h_victorias_b` | Historial de victorias entre ambos equipos antes de la fecha del partido (cumsum + shift, evitando leakage) |

## Modelos y resultados

Split train/test 80/20 respetando el orden cronológico (sin mezclar fechas).

| Modelo | Accuracy | Log-loss |
|---|---|---|
| Baseline (regla simple sobre ELO) | 44.81% | — |
| Regresión logística multinomial | 67.08% | 0.718 |
| Random Forest (200 árboles) | **69.62%** | **0.672** |

**Feature importance (Random Forest):** la forma reciente de ambos equipos (37% + 33%) pesa más que la diferencia de ELO (20%) en la decisión del modelo.

**Limitación identificada:** ambos modelos subpredicen empates de forma sistemática — la regresión logística acierta empates solo 18% de las veces que ocurren; Random Forest mejora esto a 34%, pero sigue siendo la clase más difícil de predecir. Es un patrón conocido en modelos de fútbol: el empate ocupa una región angosta del espacio de features, mientras que las victorias claras ocupan regiones mucho más amplias.

## Validación en vivo — Mundial 2026

Los modelos entrenados se aplicaron a los 48 partidos de fase de grupos ya jugados al 23/06/2026, usando ELO de respaldo para equipos sin historial suficiente (Cabo Verde, Bosnia y Herzegovina, RD Congo, Nueva Zelanda).

| Modelo | Accuracy en Mundial 2026 |
|---|---|
| Regresión logística | 66.67% |
| Random Forest | 68.75% |

Los resultados son consistentes con el accuracy del set de test histórico, lo cual valida que el modelo generaliza razonablemente bien a datos completamente nuevos. La debilidad en empates identificada en el entrenamiento se confirmó en la práctica: 9 de los 13 errores de la regresión logística fueron partidos que terminaron en empate.

Las predicciones partido por partido están en [`outputs/predicciones_fase_grupos_2026-06-27.csv`](outputs/predicciones_fase_grupos_2026-06-27.csv).

## Estructura del repositorio

```
mundial-predictor/
├── data/
│   ├── raw/          # datasets originales de Kaggle
│   └── processed/     # dataset_features.csv (features ya calculadas)
├── notebooks/
│   ├── 01_EDA_y_features.ipynb    # carga, limpieza, feature engineering
│   └── 02_modelo.ipynb            # entrenamiento, evaluación, Mundial en vivo
├── outputs/            # predicciones generadas durante el torneo
└── README.md
```

## Próximos pasos

- Probar un modelo Poisson/Dixon-Coles para predecir goles esperados de cada equipo, en vez de clasificación directa V/E/D
- Ajustar `home_forma`/`away_forma` ponderando por la fuerza del rival (actualmente trata por igual una victoria contra un equipo top y contra un equipo débil)
- Incorporar `tournament` como feature (mediante encoding) para distinguir partidos competitivos de amistosos con equipos alternativos
- Re-evaluar contra fase eliminatoria del Mundial 2026 a medida que se jueguen los partidos

## Stack

Python · pandas · numpy · scikit-learn · scipy · Jupyter Notebook

## Autor

Pablo — [github.com/MaximilianoPalma03](https://github.com/MaximilianoPalma03)
