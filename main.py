import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(X, y, learning_rate=0.01, loop=1000):
    X = np.insert(X, 0, 1, axis=1)

    num_features = X.shape[1]
    weights = np.zeros(num_features)

    for _ in range(loop):
        predictions = sigmoid(np.dot(X, weights))
        errors = y - predictions

        gradient = np.dot(X.T, errors)
        weights += learning_rate * gradient

    return weights

def predict_logistic_regression(X, weights):
    X = np.insert(X, 0, 1, axis=1)
    predictions = sigmoid(np.dot(X, weights))
    return predictions.round()

def get_X_train_logistic(train_data):
    return train_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

def get_Y_train_logistic(train_data, specie):
    return (train_data['Species'] == specie).astype(int).values

def get_X_test_logistic(test_data):
    return test_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

def get_Y_test_logistic(test_data, specie):
    return (test_data['Species'] == specie).astype(int).values

def accuracy(y_true, y_pred):
    correct_predictions = 0
    total_samples = len(y_true)
    
    for i in range(total_samples):
        if round(y_true[i]) == round(y_pred[i]):
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    return accuracy

def precision(y_true, y_pred):
    true_positives = 0
    predicted_positives = 0
    
    for i in range(len(y_true)):
        if round(y_true[i]) == 1 and round(y_pred[i]) == 1:
            true_positives += 1
        if round(y_pred[i]) == 1:
            predicted_positives += 1
    
    if predicted_positives == 0:
        precision = 0
    else:
        precision = true_positives / predicted_positives
    
    return precision

def recall(y_true, y_pred):
    true_positives = 0
    actual_positives = 0
    
    for i in range(len(y_true)):
        if round(y_true[i]) == 1 and round(y_pred[i]) == 1:
            true_positives += 1
        if round(y_true[i]) == 1:
            actual_positives += 1
    
    if actual_positives == 0:
        recall = 0
    else:
        recall = true_positives / actual_positives
    
    return recall

flowers_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Leitura do arquivo CSV usando pandas
flowers_df = pd.read_csv("Iris.csv")

# Embaralha os dados
flowers_df = flowers_df.sample(frac=1).reset_index(drop=True)

# Separa 80% dos dados para treinamento e 20% para teste
train_size = int(0.8 * len(flowers_df))
train_data, test_data = flowers_df[:train_size], flowers_df[train_size:]

# Inicializa as métricas acumuladas
total_accuracy = 0
total_precision = 0
total_recall = 0

for specie in flowers_list:
    # Eixos de treinamento
    X_train_logistic = get_X_train_logistic(train_data)
    Y_train_logistic = get_Y_train_logistic(train_data, specie)

    # Treina o modelo
    weights_logistic = train_logistic_regression(X_train_logistic, Y_train_logistic)

    # Extrai as features dos dados de teste
    X_test_logistic = get_X_test_logistic(test_data)

    # Testa o modelo
    predictions_logistic = predict_logistic_regression(X_test_logistic, weights_logistic)

    # Avaliação da Regressão Logística
    accuracy_logistic = accuracy(get_Y_test_logistic(test_data, specie), predictions_logistic)
    precision_logistic = precision(get_Y_test_logistic(test_data, specie), predictions_logistic)
    recall_logistic = recall(get_Y_test_logistic(test_data, specie), predictions_logistic)

    # Acumula as métricas
    total_accuracy += accuracy_logistic
    total_precision += precision_logistic
    total_recall += recall_logistic

    print(f"Espécie: {specie}")
    print("Acurácia:", accuracy_logistic)
    print("Precisão:", precision_logistic)
    print("Recall:", recall_logistic)
    print()

# Calcula as métricas médias
average_accuracy = total_accuracy / len(flowers_list)
average_precision = total_precision / len(flowers_list)
average_recall = total_recall / len(flowers_list)

print("Média Geral:")
print("Acurácia Média:", average_accuracy)
print("Precisão Média:", average_precision)
print("Recall Médio:", average_recall)
print()
