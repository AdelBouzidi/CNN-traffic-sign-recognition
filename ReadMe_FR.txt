# Projet : Reconnaissance des panneaux de signalisation (GTSRB) ||||========================>>>
 
Ce projet a été réalisé dans le cadre du module **HPC pour l’Intelligence Artificielle** du **Master 2 CHPS (Calcul Haute Performance et Simulation)** à l’**Université de Perpignan Via Domitia (UPVD)**.



Ce dépot inclut tout le nécessaire pour créer le modèle de détection des panneaux routiers pour le Projet 2 sur la conduite autonome, basé sur le dataset GTSRB.

J'ai construit un CNN from scratch, couvrant le prétraitement et l'entrainement et l'évaluation et l'inférence. 


## Organisation et role des fichiers ||============>>>>>>

Les scripts clés d'abord, ensuite analyse, outputs, et notebook.



### Scripts principaux (pipeline du modèle) |======>>>
model.py : Définit l'architecture CNN pour les 43 classes.

data.py : Charge GTSRB, prétraite (resize, normalisation, encoding) pour la reproductibilité.

train.py : Entraine le modèle avec dropout, EarlyStopping, sauve le tout.

evaluate.py : Évalue sur validation/test avec accuracy, confusion matrix, rapport.

infer_demo.py : Inférence sur images nouvelles pour démontrer la généralisation.


### Scripts d’analyse et de visualisation |======>>>
inspect_errors.py : Examine les erreurs et confusions entre panneaux similaires

plot_training_curves.py : Crée des graphs de loss et accuracy pour checker la convergence et le sur-apprentissage.


### Fichiers de résultats et de sortie |======>>>
training_history.json : Historique de l'entrainement pour l'analyse.

classification_report.txt : Détails sur précision, rappel, F1-score.

baseline_cnn.keras : Modele final prét.

### Notebook d’exploration: |======>>>
01_explore_dataset.ipynb : Exploration initiale du dataset.


