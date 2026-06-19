# Projet MLOps — Adult Income (Douaa EL KHEIR)

## Problematique

Predire si le **revenu annuel** d'une personne depasse **50 000 $** ou non,
a partir de donnees demographiques et professionnelles issues du recensement americain.

| Classe | Signification |
|--------|---------------|
| `0`    | Revenu <= 50K $ |
| `1`    | Revenu > 50K $  |

**Interet :** segmentation de profils, aide a la decision RH ou analyse socio-economique.

## Dataset

- **Source :** [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult)
- **Fichier :** `data/adult.csv` — 32 561 lignes, 15 colonnes
- **Cible :** colonne `income` (`<=50K` / `>50K`)
- **Desequilibre :** ~76 % classe 0 / ~24 % classe 1

### Colonnes

| Type | Colonnes |
|------|----------|
| Numeriques | `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week` |
| Categorielles | `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country` |

## Demarrage

```bash
cd todo
make install
make data
export PYTHONPATH=.
uv run python -m mlproject.train
```

## Fichiers du projet

| Fichier | Role |
|---------|------|
| `data/adult.csv` | Jeu de donnees brut |
| `todo/Makefile` | Commandes d'installation et pipeline |
| `todo/pyproject.toml` | Dependances Python du projet |
