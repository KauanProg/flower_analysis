import random
import csv
import math

def calcula_acertos(data_treino, labels_treino, data_teste, labels_teste):
    # Inicializa uma lista contendo 1 para cada acerto e 0 para cada erro
    acertos = []

    # Itera sobre pares de rótulos reais e dados de teste usando zip
    for true_label, ponto_teste in zip(labels_teste, data_teste):
        # Verifica se a predição do KNN é igual ao rótulo real
        if (knn(data_treino, labels_treino, ponto_teste, 3) == true_label):
            acerto = 1
        else:
            acerto = 0
        
        # Adiciona o resultado (1 ou 0) à lista de acertos
        acertos.append(acerto)

    # Soma todos os elementos da lista para obter o número total de acertos
    num_acertos = sum(acertos)
    return num_acertos


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

# Carrega os dados do arquivo CSV
data = []
labels = []

with open('Iris.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Ignora o cabeçalho
    for row in reader:
        # Adiciona os atributos numéricos à lista de dados
        data.append(list(map(float, row[1:5])))
        # Adiciona o rótulo à lista de rótulos
        labels.append(row[5])

# Embaralha os dados
indices_embaralhados = list(range(len(data)))
random.shuffle(indices_embaralhados)

# Reorganiza os dados e rótulos com base nos índices embaralhados
data = [data[i] for i in indices_embaralhados]
labels = [labels[i] for i in indices_embaralhados]

# Divide em treinamento e teste (80% treinamento, 20% teste)
indice_divisao = int(0.8 * len(data))
data_treino, labels_treino = data[:indice_divisao], labels[:indice_divisao]
data_teste, labels_teste = data[indice_divisao:], labels[indice_divisao:]

# Calcula a porcentagem de acertos no conjunto de teste

porcentagem_acertos = (calcula_acertos(data_treino, labels_treino, data_teste, labels_teste) / len(labels_teste)) * 100

# Exibe a porcentagem de acertos no conjunto de teste
print(f'Porcentagem de acertos no conjunto de teste: {porcentagem_acertos:.2f}%')
