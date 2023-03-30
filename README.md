# tp1_deepLearning
# Rapport : Prédiction du diabète chez les Indiens Pima

## Install

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install keras
pip install tensorflow
```
## 1. Introduction
Dans ce projet, j'ai utilisé un réseau de neurones profonds pour prédire si un individu souffre de diabète en fonction de diverses caractéristiques médicales. 

## 2. Préparation des données
Les données proviennent de Kaggle. 

   a. Séparation des données en caractéristiques et cibles :
      Nous avons séparé les caractéristiques (les colonnes sans 'Outcome') dans un data frame "train_data" et la cible (la colonne 'Outcome') dans la dataFrame "eval_data".

   b. Remplacement des zéros par NaN :
      Je me suis appercus plus tard que dans le csv il y'a pas mal de valeur a zero comme une préssion arterielle a 0 ce qui signifie qu'une personne est morte, j'ai donc decidé de remplacer ces valeurs par des Nan et j'ai enssuite calculé la moyenne de chaque colonne, en ignorant les valeurs NaN, et remplacé les valeurs manquantes par la moyenne correspondante.


   d. Division des données en ensembles d'entraînement et de test :
      j'ai utilisé la fonction `train_test_split` de scikit-learn pour diviser les données en ensembles d'entraînement (80 %) et de test (20 %).

   e. Normalisation des données :
      Afin d'améliorer la convergence du modèle, j'ai normalisé les caractéristiques en utilisant `StandardScaler` de scikit-learn.

## 3. Construction du modèle de réseau de neurones
j'ai utilisé Keras pour construire un modèle de réseau de neurones séquentiel avec plusieurs couches cachées et une couche de sortie. Le modèle est composé de :

   a. Quatre couches cachées de 128 neurones chacune avec une fonction d'activation ReLU.
   
   b. Des couches de normalisation par lots après chaque couche cachée pour accélérer l'entraînement et améliorer la performance du modèle.
   
   c. Des couches Dropout avec un taux de 0.5 après chaque couche cachée pour prévenir le surapprentissage.
   
   d. Une couche de sortie avec un seul neurone et une fonction d'activation sigmoïde pour prédire la probabilité qu'un individu soit diabétique.
   
Avant d'en venir a cela j'avais fais : 
  a. Quatre couches cachées de 64 neurones chacune avec une fonction d'activation sigmoid.
  
  b. Des couches Dropout avec un taux de 0.5 après chaque couche cachée pour prévenir le surapprentissage.
  
  c. Une couche de sortie avec un seul neurone et une fonction d'activation sigmoïde pour prédire la probabilité qu'un individu soit diabétique.
  
## 4. Entraînement du modèle
j'ai compilé le modèle en utilisant l'optimiseur Adam(je l'ai pris car il était connue et qu'il fonctionne bien sinon j'avais tester "Nadam"
, "Adadelta",   avec un taux d'apprentissage de 0.001,j'ai mis a 0.001 car plus rapide vue que la mise a jour des poids est moin importante

la fonction de perte 'binary_crossentropy' et j'ai aussi tester avec "mean_squared_error" ainsi que "mean_absolute_error" et la métrique 'accuracy'. 

j'ai également utilisé la technique d'early stopping pour éviter le surapprentissage. Le modèle a été entraîné avec les données d'entraînement en utilisant une taille de batch de 32 (au par avant a 150)  et un ensemble de validation de 10 % des données d'entraînement pour surveiller les performances.
