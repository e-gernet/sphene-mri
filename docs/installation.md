# Installation

## Prérequis

- [Git](https://git-scm.com/) pour le versionnage
- [Pixi](https://pixi.sh/) pour la gestion de l'environnement et des dépendances

## Récupérer le projet

```bash
git clone git@github.com:e-gernet/sphene-mri.git
cd sphene-mri
```

## Installer l'environnement

Pixi lit `pixi.toml` et `pixi.lock` et recrée un environnement identique à
celui utilisé pour le développement :

```bash
pixi install
```

Aucune activation manuelle n'est nécessaire — toutes les commandes `pixi run`
s'exécutent automatiquement dans cet environnement.

### Environnements optionnels

En plus de l'environnement `default` (strict nécessaire pour faire tourner
le pipeline), deux environnements optionnels sont disponibles :

| Environnement | Contenu | Installation |
|---------------|---------|--------------|
| `dev`  | Outils de développement : lint (`ruff`), formatage, documentation (`mkdocs`), `ipython` | `pixi install -e dev` |
| `gpu`  | Accélération GPU du filtrage spatial via `cupy` (nécessite un GPU NVIDIA/CUDA, Windows ou Linux uniquement) | `pixi install -e gpu` |

Chaque environnement optionnel inclut automatiquement tout le contenu de
`default`, en plus de ses propres dépendances — pas besoin de tout
réinstaller.

## Vérifier l'installation

```bash
pixi run python -c "import nibabel, numpy, scipy, skimage; print('OK')"
```

## Lancer le pipeline

```bash
pixi run sphene
```

Avec accélération GPU pour le filtrage (nécessite `pixi install -e gpu`) :

```bash
pixi run -e gpu sphene-gpu
```

## Lancer la documentation en local

```bash
pixi run -e dev docs
```

La documentation est alors accessible sur `http://127.0.0.1:8000`.
