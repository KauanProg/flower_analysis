# Membros da Equipe
# Kauan Deyvid Bezerra de Sousa - 510270
# Billy Grahan Alves Rodrigues - 508010
# Marcos Gabriel de Mesquita Mauricio - 509127

import math
import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(X, y, learning_rate=0.01, epochs=10000):
    X = np.insert(X, 0, 1, axis=1)

    num_features = X.shape[1]
    weights = np.zeros(num_features)

    for _ in range(epochs):
        predictions = sigmoid(np.dot(X, weights))
        errors = y - predictions

        gradient = np.dot(X.T, errors)
        weights += learning_rate * gradient

    return weights

def predict_logistic_regression(X, weights, threshold=0.5):
    X = np.insert(X, 0, 1, axis=1)
    probabilities = sigmoid(np.dot(X, weights))
    predictions = np.where(probabilities >= threshold, 1, 0)
    return predictions

def get_X_train(train_data):
    return train_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

def get_X_test(test_data):
    return test_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

def get_Y_train(train_data, specie):
    return np.where(train_data['Species'].values == specie, 1, 0)

def get_Y_test(test_data, specie):
    return np.where(test_data['Species'].values == specie, 1, 0)

def accuracy(y_test, y_prediction):
    correct_predictions = 0
    total_samples = len(y_test)
    
    for i in range(total_samples):
        if round(y_test[i]) == round(y_prediction[i]):
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    return accuracy

def precision(y_test, y_prediction):
    true_positives = 0
    predicted_positives = 0
    
    for i in range(len(y_test)):
        if round(y_test[i]) == 1 and round(y_prediction[i]) == 1:
            true_positives += 1
        if round(y_prediction[i]) == 1:
            predicted_positives += 1
    
    if predicted_positives == 0:
        precision = 0
    else:
        precision = true_positives / predicted_positives
    
    return precision

def recall(y_test, y_prediction):
    true_positives = 0
    actual_positives = 0
    
    for i in range(len(y_test)):
        if round(y_test[i]) == 1 and round(y_prediction[i]) == 1:
            true_positives += 1
        if round(y_test[i]) == 1:
            actual_positives += 1
    
    if actual_positives == 0:
        recall = 0
    else:
        recall = true_positives / actual_positives
    
    return recall

def calcula_acertos(data_treino, labels_treino, data_teste, labels_teste, k):
    verdadeiros_positivos = 0
    falsos_positivos = 0
    verdadeiros_negativos = 0
    falsos_negativos = 0

    # Itera sobre pares de rótulos reais e dados de teste usando zip
    for true_label, ponto_teste in zip(labels_teste, data_teste):
        # Faz a predição com KNN
        predicao = knn(data_treino, labels_treino, ponto_teste, k)

        # Avalia as métricas
        if true_label == 1 and predicao == 1:
            verdadeiros_positivos += 1
        elif true_label == 0 and predicao == 1:
            falsos_positivos += 1
        elif true_label == 0 and predicao == 0:
            verdadeiros_negativos += 1
        elif true_label == 1 and predicao == 0:
            falsos_negativos += 1

    # Calcula métricas
    acuracia = (verdadeiros_positivos + verdadeiros_negativos) / len(labels_teste)

    if verdadeiros_positivos + falsos_positivos == 0:
        precisao = 0
    else:
        precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)

    if verdadeiros_positivos + falsos_negativos == 0:
        recall = 0
    else:
        recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)

    return acuracia, precisao, recall

# Função para calcular a distância euclidiana entre dois pontos
def distancia_euclidiana(ponto1, ponto2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(ponto1, ponto2)))

# Função KNN para predizer a classe de um ponto de teste
def knn(treino, labels, ponto_teste, k):
    # Calcula as distâncias entre o ponto de teste e os pontos de treinamento
    distancias = [distancia_euclidiana(ponto_teste, ponto) for ponto in treino]
    
    # Obtém os índices dos k vizinhos mais próximos
    indices_vizinhos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:k]
    
    # Obtém os rótulos dos vizinhos mais próximos
    vizinhos_labels = [labels[i] for i in indices_vizinhos]
    
    # Converte os rótulos para inteiros
    classes = list(set(vizinhos_labels))
    vizinhos_labels = [classes.index(label) for label in vizinhos_labels]
    
    # Conta a ocorrência de cada rótulo
    contagem_labels = [0] * len(classes)
    for label in vizinhos_labels:
        contagem_labels[label] += 1
    
    # Determina a classe predita com base na contagem de rótulos
    classe_predita = classes[contagem_labels.index(max(contagem_labels))]
    return classe_predita

flowers_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

flowers_df = pd.read_csv("Iris.csv")

# Embaralha os dados
flowers_df = flowers_df.sample(frac=1).reset_index(drop=True)

# Separa 80% dos dados para treinamento e 20% para teste
train_size = int(0.8 * len(flowers_df))
train_data, test_data = flowers_df[:train_size], flowers_df[train_size:]

total_accuracy_logistic = 0
total_precision_logistic = 0
total_recall_logistic = 0

total_accuracy_knn = 0
total_precision_knn = 0
total_recall_knn = 0

for specie in flowers_list:
    # Eixos de treinamento para Regressão Logística
    X_train_logistic = get_X_train(train_data)
    Y_train_logistic = get_Y_train(train_data, specie)

    # Treina o modelo de Regressão Logística
    weights_logistic = train_logistic_regression(X_train_logistic, Y_train_logistic)

    # Extrai as features dos dados de teste
    X_test_logistic = get_X_test(test_data)

    # Testa o modelo de Regressão Logística
    predictions_logistic = predict_logistic_regression(X_test_logistic, weights_logistic)

    # Avaliação da Regressão Logística
    accuracy_logistic = accuracy(get_Y_test(test_data, specie), predictions_logistic)
    precision_logistic = precision(get_Y_test(test_data, specie), predictions_logistic)
    recall_logistic = recall(get_Y_test(test_data, specie), predictions_logistic)

    # Acumula as métricas para Regressão Logística
    total_accuracy_logistic += accuracy_logistic
    total_precision_logistic += precision_logistic
    total_recall_logistic += recall_logistic

    print("Regressão Logística")
    print(f"Espécie: {specie}")
    print("Acurácia:", round(accuracy_logistic, 2))
    print("Precisão:", round(precision_logistic, 2))
    print("Recall:", round(recall_logistic, 2))
    print()

    X_train_knn = get_X_train(train_data)
    Y_train_knn = get_Y_train(train_data, specie)
    
    X_test_knn = get_X_test(test_data)
    Y_test_knn = get_Y_test(test_data, specie)

    k = 3
    acuracia_knn, precisao_knn, recall_knn = calcula_acertos(X_train_knn, Y_train_knn, X_test_knn, Y_test_knn, k)

    total_accuracy_knn += acuracia_knn
    total_precision_knn += precisao_knn
    total_recall_knn += recall_knn

    print("KNN")
    print(f"Espécie: {specie}")
    print("Acurácia:", round(acuracia_knn, 2))
    print("Precisão:", round(precisao_knn, 2))
    print("Recall:", round(recall_knn, 2))
    print()

average_accuracy_logistic = total_accuracy_logistic / len(flowers_list)
average_precision_logistic = total_precision_logistic / len(flowers_list)
average_recall_logistic = total_recall_logistic / len(flowers_list)

print("Regressão Logística")
print("Média Geral:")
print("Acurácia Média:", round(average_accuracy_logistic, 2))
print("Precisão Média:", round(average_precision_logistic, 2))
print("Recall Médio:", round(average_recall_logistic, 2))
print()

average_accuracy_knn = total_accuracy_knn / len(flowers_list)
average_precision_knn = total_precision_knn / len(flowers_list)
average_recall_knn = total_recall_knn / len(flowers_list)

print("KNN")
print("Média Geral:")
print("Acurácia Média:", round(average_accuracy_knn, 2))
print("Precisão Média:", round(average_precision_knn, 2))
print("Recall Médio:", round(average_recall_knn, 2))
print()