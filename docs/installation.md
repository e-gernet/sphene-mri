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

## Vérifier l'installation

```bash
pixi run python -c "import nibabel, numpy, scipy, skimage; print('OK')"
```

## Lancer le pipeline

```bash
pixi run python main.py
```

## Lancer la documentation en local

```bash
pixi run mkdocs serve
```

La documentation est alors accessible sur `http://127.0.0.1:8000`.