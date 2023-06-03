#!/usr/bin/env python
# coding: utf-8

# # MODELOS SVM 

# ## Modelo com 300 linhas

# In[387]:


# Importar bibliotecas

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sb


# In[319]:


# Importar dataset

df = pd.read_excel(r"C:\Users\Usuário\Desktop\dataset_deslizamentos.xlsx")
print(df.head()) # mostra as primeiras 5 linhas


# In[320]:


# Verificar o total de linhas e colunas no dataset

df.shape


# In[321]:


# Descrever dados estatísticos do dataset

df.describe()


# In[322]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

df['deslizamento'].value_counts()


# In[391]:


sb.pairplot(df, hue = "deslizamento")


# In[323]:


# Separação das variáveis

X = df.drop(columns='deslizamento', axis=1)
Y = df['deslizamento']


# In[324]:


# Variáveis independentes

print(X)


# In[325]:


# Variável dependente

print(Y)


# In[326]:


# Criação de instância e fit

scaler = StandardScaler()
scaler.fit(X)


# In[327]:


# Transformar as variáveis "x" em dados de entrada

standarized_data = scaler.transform(X) 
X = standarized_data 

print(X)
print(Y)


# In[328]:


# Dividir o dataset na proporção 70/30

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify = Y, random_state=2)

# Imprimir o tamanho do dataset original
print("Dataset original:", X.shape)

# Imprimir o dataset de teste
print("Dataset de teste:", X_test.shape)

# Imprimir o dataset de treinamento
print("Dataset de treinamento (X_train):", X_train.shape)


# ## 🦾 Criação do modelo 

# In[329]:


classifier = svm.SVC(kernel='linear')


# In[330]:


# Treinar modelo usando o dataset de teste
classifier.fit(X_train, Y_train)


# In[331]:


# Visualizar o coef

classifier.coef_[0]


# In[332]:


# Separando w1 

w1 =classifier.coef_[0][0]
print(w1)


# In[333]:


# Separando w2

w2 = classifier.coef_[0][1]
print(w2)


# In[334]:


# Utilizando o intercept como w0

w3 = classifier.intercept_[0]
print(w3)


# In[335]:


# Acurácia nos dados de treinamento
train_pred = classifier.predict(X_train)
accuracy_train = accuracy_score(train_pred, Y_train)


# In[336]:


print("Pontuação de acurácia dos dados de treinamento = {}".format(accuracy_train))


# In[337]:


#Acurácia no dados de teste
test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)


# In[338]:


print("Pontuação de acurácia dos dados de teste = {}".format(accuracy_test))


# ## 📊 Métricas

# In[339]:


import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score


# In[340]:


# Calcular a matriz de confusão
cm = confusion_matrix(Y_test, test_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


# In[341]:


train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Score de Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Score de Validação Cruzada")
plt.legend(loc="best")
plt.xlabel("Amostras de treinamento")
plt.ylabel("Score")
plt.show()


# In[342]:


# F-score nos dados de treinamento
f1_train = f1_score(Y_train, train_pred)

print("F-score nos dados de treinamento = {}".format(f1_train))


# In[343]:


# Calcular as probabilidades das classes positivas
probabilities = classifier.decision_function(X_test)

# Calcular a taxa de falsos positivos e a taxa de verdadeiros positivos
fpr, tpr, thresholds = roc_curve(Y_test, probabilities)

# Calcular a área sob a curva ROC (AUC-ROC)
auc = roc_auc_score(Y_test, probabilities)

# Plotar a curva ROC
plt.plot(fpr, tpr, color='purple', label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linha de referência para uma classificação aleatória
plt.xlabel('(1-especificidade)%')
plt.ylabel('Sensibilidade (%)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# ## 🔮 Input para predição

# In[356]:


# Exemplo de valores de precipitação, pluviômetro e umidade a serem previstos
novo_dado = np.array([[0, 0, 76]])

# Aplicar a transformação aos novos dados
novo_dado_transformado = scaler.transform(novo_dado)

# Fazer a predição dos deslizamentos
predicao = classifier.predict(novo_dado_transformado)

if predicao == 1:
    print("Desliza")
else:
    print("Não desliza")


# ## Modelo com dados originais

# In[393]:


# Importar dataset

df = pd.read_excel(r"C:\Users\Usuário\Desktop\teste_deslizamentos.xlsx")
print(df.head()) # mostra as primeiras 5 linhas


# In[394]:


# Verificar o total de linhas e colunas no dataset

df.shape


# In[395]:


# Descrever dados estatísticos do dataset

df.describe()


# In[396]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

df['deslizamento'].value_counts()


# In[401]:


sb.pairplot(df, hue = "deslizamento")


# In[397]:


# Separação das variáveis

X = df.drop(columns='deslizamento', axis=1)
Y = df['deslizamento']


# In[398]:


# Variáveis independentes

print(X)


# In[399]:


print(Y)


# In[400]:


# Criação de instância e fit

scaler = StandardScaler()
scaler.fit(X)


# In[283]:


# Transformar as variáveis "x" em dados de entrada

standarized_data = scaler.transform(X) 
Z = standarized_data 

print(X)
print(Y)


# In[284]:


# Dividir o dataset na proporção 70/30

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify = Y, random_state=2)

# Imprimir o tamanho do dataset original
print("Dataset original:", X.shape)

# Imprimir o dataset de teste
print("Dataset de teste:", X_test.shape)

# Imprimir o dataset de treinamento
print("Dataset de treinamento (X_train):", X_train.shape)


# ## 🦾 Criação do modelo 

# In[285]:


classifier = svm.SVC(kernel='linear')


# In[286]:


# Treinar modelo usando o dataset de teste
classifier.fit(X_train, Y_train)


# In[287]:


# Acurácia nos dados de treinamento
train_pred = classifier.predict(X_train)
accuracy_train = accuracy_score(train_pred, Y_train)


# In[288]:


print("Pontuação de acurácia dos dados de treinamento = {}".format(accuracy_train))


# In[289]:


#Acurácia no dados de teste
test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)


# In[290]:


print("Pontuação de acurácia dos dados de teste = {}".format(accuracy_test))


# ## 📊 Métricas

# In[291]:


# Calcular a matriz de confusão
cm = confusion_matrix(Y_test, test_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


# In[292]:


train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Score de Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Score de Validação Cruzada")
plt.legend(loc="best")
plt.xlabel("Amostras de treinamento")
plt.ylabel("Score")
plt.show()


# In[293]:


# F-score nos dados de treinamento
f1_train = f1_score(Y_train, train_pred)

print("F-score nos dados de treinamento = {}".format(f1_train))


# In[294]:


# Calcular as probabilidades das classes positivas
probabilities = classifier.decision_function(X_test)

# Calcular a taxa de falsos positivos e a taxa de verdadeiros positivos
fpr, tpr, thresholds = roc_curve(Y_test, probabilities)

# Calcular a área sob a curva ROC (AUC-ROC)
auc = roc_auc_score(Y_test, probabilities)

# Plotar a curva ROC
plt.plot(fpr, tpr, color='purple', label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linha de referência para uma classificação aleatória
plt.xlabel('(1-especificidade)%')
plt.ylabel('Sensibilidade (%)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# ## 🔮Input para predição

# In[402]:


# Exemplo de valores de precipitação, pluviômetro e umidade a serem previstos
novo_dado = np.array([[0, 0, 76]])

# Aplicar a transformação aos novos dados
novo_dado_transformado = scaler.transform(novo_dado)

# Fazer a predição dos deslizamentos
predicao = classifier.predict(novo_dado_transformado)

if predicao == 1:
    print("Desliza")
else:
    print("Não desliza")


# ## Modelo com 1000 linhas

# In[403]:


# Importar dataset

df = pd.read_excel(r"C:\Users\Usuário\Desktop\dataset_deslizamentos_1000.xlsx")
print(df.head()) # mostra as primeiras 5 linhas


# In[404]:


# Verificar o total de linhas e colunas no dataset

df.shape


# In[405]:


# Descrever dados estatísticos do dataset

df.describe()


# In[406]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

df['deslizamento'].value_counts()


# In[408]:


sb.pairplot(df, hue = "deslizamento")


# In[361]:


# Separação das variáveis

X = df.drop(columns='deslizamento', axis=1)
Y = df['deslizamento']


# In[362]:


# Variáveis independentes

print(X)


# In[363]:


# Variáveis independentes

print(Y)


# In[364]:


# Criação de instância e fit

scaler = StandardScaler()
scaler.fit(X)


# In[365]:


# Transformar as variáveis "x" em dados de entrada

standarized_data = scaler.transform(X) 
X = standarized_data 

print(X)
print(Y)


# In[366]:


# Dividir o dataset na proporção 70/30 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify = Y, random_state=2)

# Imprimir o tamanho do dataset original
print("Dataset original:", X.shape)

# Imprimir o dataset de teste
print("Dataset de teste:", X_test.shape)

# Imprimir o dataset de treinamento
print("Dataset de treinamento (X_train):", X_train.shape)


# ## 🦾 Criação do modelo 

# In[367]:


classifier = svm.SVC(kernel='linear')


# In[368]:


# Treinar modelo usando o dataset de teste
classifier.fit(X_train, Y_train)


# In[369]:


# Acurácia nos dados de treinamento
train_pred = classifier.predict(X_train)
accuracy_train = accuracy_score(train_pred, Y_train)


# In[370]:


print("Pontuação de acurácia dos dados de treinamento = {}".format(accuracy_train))


# In[371]:


#Acurácia no dados de teste
test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)


# In[372]:


print("Pontuação de acurácia dos dados de teste = {}".format(accuracy_test))


# ## 📊 Métricas

# In[373]:


# Calcular a matriz de confusão
cm = confusion_matrix(Y_test, test_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


# In[374]:


train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Score de Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Score de Validação Cruzada")
plt.legend(loc="best")
plt.xlabel("Amostras de treinamento")
plt.ylabel("Score")
plt.show()


# In[375]:


# F-score nos dados de treinamento
f1_train = f1_score(Y_train, train_pred)

print("F-score nos dados de treinamento = {}".format(f1_train))


# In[376]:


# Calcular as probabilidades das classes positivas
probabilities = classifier.decision_function(X_test)

# Calcular a taxa de falsos positivos e a taxa de verdadeiros positivos
fpr, tpr, thresholds = roc_curve(Y_test, probabilities)

# Calcular a área sob a curva ROC (AUC-ROC)
auc = roc_auc_score(Y_test, probabilities)

# Plotar a curva ROC
plt.plot(fpr, tpr, color='purple', label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linha de referência para uma classificação aleatória
plt.xlabel('(1-especificidade)%')
plt.ylabel('Sensibilidade (%)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# ## 🔮 Input para predição

# In[386]:


# Exemplo de valores de precipitação, pluviômetro e umidade a serem previstos
novo_dado = np.array([[1, 20, 75]])

# Aplicar a transformação aos novos dados
novo_dado_transformado = scaler.transform(novo_dado)

# Fazer a predição dos deslizamentos
predicao = classifier.predict(novo_dado_transformado)

if predicao == 1:
    print("Desliza")
else:
    print("Não desliza")

