#!/usr/bin/env python
# coding: utf-8

# # MODELO DE REGRESSÃO LOGÍSTICA

# ## Modelo com 300 linhas

# In[10]:


# Importar bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression


# In[11]:


# Importar dataset

df = pd.read_excel(r"C:\Users\Usuário\Desktop\dataset_deslizamentos.xlsx")
df.head() # mostra as primeiras 5 linhas


# In[12]:


# Verificar o total de linhas e colunas no dataset

df.shape


# In[38]:


# Descrever dados estatísticos do dataset
df.describe()


# In[14]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

df['deslizamento'].value_counts()


# In[15]:


# Separação das variáveis

X = df.drop(columns='deslizamento', axis=1)
Y = df['deslizamento']


# In[16]:


# Dividir o dataset na proporção 70/30

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify = Y, random_state=2)

# Imprimir o tamanho do dataset original
print("Dataset original:", X.shape)

# Imprimir o dataset de teste
print("Dataset de teste:", X_test.shape)

# Imprimir o dataset de treinamento
print("Dataset de treinamento (X_train):", X_train.shape)


# In[17]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## 🦾 Criação do modelo 

# In[18]:


m1 = 'Logistic Regression'
lr = LogisticRegression()


# In[43]:


model = lr.fit(X_train, Y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(Y_test, lr_predict)
lr_acc_score_test = accuracy_score(Y_test, lr_predict)
lr_pred_train = lr.predict(X_train)  # Predição nos dados de treinamento
lr_acc_score_train = accuracy_score(Y_train, lr_pred_train)  # Acurácia nos dados de treinamento


# In[44]:


#print("confusion matrix")
#print(lr_conf_matrix)
#print("\n")
# Acurácia nos dados de teste
print("Pontuação de acurácia dos dados de teste:", lr_acc_score_test*100, '\n')
print(classification_report(Y_test, lr_predict))

# Acurácia nos dados de treinamento
print("Pontuação de acurácia dos dados de treinamento:", lr_acc_score_train*100, '\n')
print(classification_report(Y_train, lr_pred_train))


# ## 📊 Métricas

# In[32]:


import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score


# In[64]:


# Calcular a matriz de confusão
cm = confusion_matrix(Y_test, lr_predict)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


# In[39]:


train_sizes, train_scores, test_scores = learning_curve(lr, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Score de Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Score de Validação Cruzada")
plt.legend(loc="best")
plt.xlabel("Amostras de treinamento")
plt.ylabel("Score")
plt.show()


# In[41]:


train_pred = lr.predict(X_train)
f1_train = f1_score(Y_train, train_pred)

print("F-score nos dados de treinamento = {}".format(f1_train))


# In[46]:


# Calcular as probabilidades das classes positivas
probabilities = lr.decision_function(X_test)

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


# ## Modelo com 1000 linhas 

# In[47]:


# Importar dataset

df_2 = pd.read_excel(r"C:\Users\Usuário\Desktop\dataset_deslizamentos_1000.xlsx")
df.head() # mostra as primeiras 5 linhas


# In[48]:


# Verificar o total de linhas e colunas no dataset

df_2.shape


# In[49]:


# Descrever dados estatísticos do dataset
df_2.describe()


# In[50]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

df_2['deslizamento'].value_counts()


# In[51]:


# Separação das variáveis

X = df_2.drop(columns='deslizamento', axis=1)
Y = df_2['deslizamento']


# In[52]:


# Dividir o dataset na proporção 70/30

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify = Y, random_state=2)

# Imprimir o tamanho do dataset original
print("Dataset original:", X.shape)

# Imprimir o dataset de teste
print("Dataset de teste:", X_test.shape)

# Imprimir o dataset de treinamento
print("Dataset de treinamento (X_train):", X_train.shape)


# In[53]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## 🦾 Criação do modelo 

# In[54]:


m1 = 'Logistic Regression'
lr = LogisticRegression()


# In[55]:


model = lr.fit(X_train, Y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(Y_test, lr_predict)
lr_acc_score_test = accuracy_score(Y_test, lr_predict)
lr_pred_train = lr.predict(X_train)  # Predição nos dados de treinamento
lr_acc_score_train = accuracy_score(Y_train, lr_pred_train)  # Acurácia nos dados de treinamento


# In[58]:


#print("confusion matrix")
#print(lr_conf_matrix)
#print("\n")
# Acurácia nos dados de teste
print("Pontuação de acurácia dos dados de teste:", lr_acc_score_test*100, '\n')
print(classification_report(Y_test, lr_predict))

# Acurácia nos dados de treinamento
print("Pontuação de acurácia dos dados de treinamento:", lr_acc_score_train*100, '\n')
print(classification_report(Y_train, lr_pred_train))


# ## Métricas

# In[65]:


# Calcular a matriz de confusão
cm = confusion_matrix(Y_test, lr_predict)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


# In[59]:


train_sizes, train_scores, test_scores = learning_curve(lr, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Score de Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Score de Validação Cruzada")
plt.legend(loc="best")
plt.xlabel("Amostras de treinamento")
plt.ylabel("Score")
plt.show()


# In[60]:


train_pred = lr.predict(X_train)
f1_train = f1_score(Y_train, train_pred)

print("F-score nos dados de treinamento = {}".format(f1_train))


# In[61]:


# Calcular as probabilidades das classes positivas
probabilities = lr.decision_function(X_test)

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

