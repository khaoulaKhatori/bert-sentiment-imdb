# Analyse de sentiments avec BERT (IMDb)

Classification automatique de critiques de films (positif / négatif) avec un modèle **BERT** fine-tuné, en utilisant **Python**, **Hugging Face Transformers** et un pipeline NLP complet : exploration des données, baseline, fine-tuning, évaluation et interprétation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VOTRE-PSEUDO/bert-sentiment-imdb/blob/main/analyse_sentiments_bert_imdb.ipynb)

## Objectif

Comparer deux approches sur le jeu de données IMDb :

1. **Baseline** : `distilbert-base-uncased-finetuned-sst-2-english`, utilisé tel quel (sans entraînement de notre part) ;
2. **BERT fine-tuné** : `bert-base-uncased` entraîné sur des critiques IMDb.

Puis interpréter les résultats (analyse d'erreurs, LIME, visualisation de l'attention avec BertViz).

## Données

- Jeu de données : [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb), chargé avec la bibliothèque `datasets`.
- 25 000 critiques d'entraînement et 25 000 de test, classes parfaitement équilibrées.
- Sous-échantillon utilisé ici : **10 000 critiques pour l'entraînement** et **2 000 pour le test**.
- Longueur moyenne des critiques : environ 234 mots (textes tronqués à 256 tokens).

## Méthodologie

| Élément | Valeur |
|---|---|
| Modèle | `bert-base-uncased` + couche de classification (2 classes) |
| Longueur maximale | 256 tokens |
| Époques | 2 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Outil d'entraînement | `Trainer` (Hugging Face) |
| Matériel | GPU Google Colab |

## Résultats (2 000 critiques de test)

| Modèle | Accuracy | Précision | Rappel | F1 |
|---|---|---|---|---|
| Baseline (DistilBERT SST-2) | 0,8795 | 0,8982 | 0,8560 | 0,8766 |
| **BERT fine-tuné (IMDb)** | **0,9045** | 0,8805 | **0,9360** | **0,9074** |

Matrices de confusion (lignes : classe réelle, colonnes : classe prédite) :

| | Baseline : négatif | Baseline : positif | BERT : négatif | BERT : positif |
|---|---|---|---|---|
| **Réel négatif** | 903 | 97 | 873 | 127 |
| **Réel positif** | 144 | 856 | 64 | 936 |

Le fine-tuning améliore l'accuracy d'environ **2,5 points** et le F1 d'environ 3 points. Le modèle fine-tuné rate beaucoup moins de critiques positives (64 erreurs contre 144), au prix de quelques faux positifs en plus.

## Interprétation

- **Erreurs** : 191 erreurs sur 2 000 critiques. Les plus « confiantes » viennent de critiques ironiques, nuancées ou longues (la conclusion se trouve parfois après la troncature à 256 tokens).
- **LIME** : sur la phrase *« The plot was predictable, but the acting was brilliant and I loved the ending »*, le modèle prédit « positif » à 99 %. Les mots « brilliant » et « loved » poussent vers le positif, « predictable » vers le négatif.
- **BertViz** : permet de voir comment l'attention relie les mots, par exemple « not » et « good » dans *« The movie was not good at all »*.
- **Limite illustrée** : la phrase ironique *« Oh great, another two hours of my life I will never get back »* est classée « positive » (confiance 57 %), alors qu'elle est négative. Le modèle est trompé par « great ».

## Limites et pistes d'amélioration

- Difficulté avec l'**ironie**, les **négations** et les avis **mitigés**.
- Jeu de données limité aux critiques de films en anglais (biais de domaine).
- Pistes : plus de données et d'époques, `max_length` plus grand, modèles plus récents (RoBERTa, DeBERTa), jeu de données multi-domaines.

## Installation et utilisation

1. Ouvrir le notebook dans Google Colab (bouton ci-dessus) et activer le GPU.
2. Ou en local : `pip install -r requirements.txt`, puis ouvrir `analyse_sentiments_bert_imdb.ipynb` avec Jupyter.
3. Exécuter les cellules dans l'ordre.

## Technologies

Python, PyTorch, Hugging Face Transformers, Datasets, scikit-learn, LIME, BertViz, Matplotlib, Seaborn.
