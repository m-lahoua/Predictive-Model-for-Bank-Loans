import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#### -- DATA CLEANING -- ####

#lire la base de données
df=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

#déterminer la présence des valeurs manquantes
df.info()

#voir les variables catégoriques
print(df.describe(include='object'))

#voir les variables numériques et déterminer la présence des valeurs négatives
print(df.describe(include='number'))

#séparation des variables
cat_data=[]
num_data=[]
for i,c in enumerate(df.dtypes):
    if c==object:
        cat_data.append(df.iloc[:,i])
    else:
        num_data.append(df.iloc[:,i])

#transformer les listes en format de base de données
cat_data=pd.DataFrame(cat_data).transpose()
#print(cat_data)
num_data=pd.DataFrame(num_data).transpose()
#print(num_data)

#renseignement des valeurs catégoriques manquantes
cat_data=cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
print(cat_data.isnull().sum().any())

#renseignement des valeurs numériques manquantes
num_data = num_data.bfill()
print(num_data.isnull().sum().any())

# Remplacer la colonne Loan_Status de Y et N a 1 et 0
target_value={'Y' :1, 'N' :0}
target=cat_data['Loan_Status']
cat_data.drop('Loan_Status',axis=1, inplace=True)
target=target.map(target_value)

#encodage des variables catégoriques
le=LabelEncoder()
for i in cat_data:
    cat_data[i]=le.fit_transform(cat_data[i])

#supprimer loan_id
cat_data.drop('Loan_ID',axis=1,inplace=True)
    
#concatenation des variables dans une seule base
X=pd.concat([cat_data,num_data,target],axis=1)
#verification de la base
pd.set_option('display.max_columns',X.shape[0]+1)
print(X)
X.info()


#### -- ANALYSE EXPLORATOIRE -- ####

#variable target
print(target.value_counts())

#visualisation de la variable target 'Loan_Status'
plt.figure(figsize=(8,6))
sns.countplot(x='Loan_Status', data=X)
plt.title("Diagramme des crédits accordés et non accordés")
plt.xlabel("Loan Status")
plt.ylabel("Nombre d'occurrences")
yes=(target.value_counts()[1]/len(target))*100
no=(target.value_counts()[0]/len(target))*100
print(f'le pourcentage des crédits accordés et:{yes}')
print(f'le pourcentage des crédits non accordés et:{no}')
plt.show()

## - impact des variables - ##

# Diagramme Comparaison Pour l'historique de credit
grid=sns.FacetGrid(X,col='Loan_Status',height=3.2,aspect=1.6)
grid.map(sns.countplot,'Credit_History')
plt.show()

# Diagramme Comparaison Pour les gens Marié
#Ici on peux faire le diagramme pour beaucoup de colonne (Education, Gender...)
grid=sns.FacetGrid(X,col='Loan_Status',height=3.2,aspect=1.6)
grid.map(sns.countplot,'Married')
plt.show()

# revenu du demandeur
plt.figure(figsize=(8,6))
plt.title("comparaison pour le revenue des demandeurs")
plt.xlabel("income")
plt.ylabel("loan status")
plt.scatter(X['ApplicantIncome'],X['Loan_Status'])
plt.show()

# les medians des valeurs
median=X.groupby('Loan_Status').median()
print(median)

#### -- Réalisation du modèle en ce basant sur les algo du machine learning -- ####

#séparation de la base de données en unes base de test et une base d'entrainement
y=target # ici target c'est la colonne de Loan_Status avec 1,0
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in sss.split(X,y):
    X_train,X_test=X.iloc[train],X.iloc[test]
    y_train,y_test=y.iloc[train],y.iloc[test]

print('X_train taille:', X_train.shape)
print('X_test taille:', X_test.shape)
print('Y_train taille:', y_train.shape)
print('Y_test taille:', y_test.shape)

#On va appliquer 3 algorithmes Logistic Regression, KNeighborsClassifier, DecisionTreeClassifier
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}

#la fonction de precision
def accu(y_true, y_pred, retu=False):
    acc=accuracy_score(y_true,y_pred)
    if retu:
        return acc
    else:
        print(f'la precision du modèle est: {acc}')

#Fonction pour entraîner, tester et évaluer les modèles
def train_test_eval(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        print(name, ':')
        model.fit(X_train, y_train)
        accu(y_test, model.predict(X_test))
        print('-' * 30)

# Utilisation de la fonction avec vos données
train_test_eval(models, X_train, y_train, X_test, y_test)

#2eme BDD à partir de la première mais avec 3 colonnes
X_2 = X[['Credit_History', 'Married', 'CoapplicantIncome']]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X_2, y):
    X_train, X_test = X_2.iloc[train], X_2.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
print('X_train taille: ', X_train.shape)
print('X_test taille: ', X_test.shape)
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape)
train_test_eval(models, X_train, y_train, X_test, y_test)

#### -- Déploiment du modèle -- ####

#appliquer la regression logistique sur le 2ème base de données
Classifier=LogisticRegression()
Classifier.fit(X_2,y)

#enregistrer le modèle sur un fichier pkl
pickle.dump(Classifier,open('model.pkl','wb'))