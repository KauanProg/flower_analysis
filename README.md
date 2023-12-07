# Relatório

1. Introdução:
   O modelo de regressão logística foi treinado e avaliado para prever a espécie de flores Iris com base em suas características botânicas. O conjunto de dados foi dividido em treinamento (80%) e teste (20%), e o modelo foi treinado separadamente para cada espécie: Iris-setosa, Iris-versicolor e Iris-virginica.

2. Resultados por Espécie:

   - Regressão Logística
      - Iris-setosa:
         - Acurácia: 1.0
         - Precisão: 1.0
         - Recall: 1.0

      - Iris-versicolor:
         - Acurácia: 0.6
         - Precisão: 0.43
         - Recall: 1.0

      - Iris-virginica:
         - Acurácia: 1.0
         - Precisão: 1.0
         - Recall: 1.0

   - KNN
      - Iris-setosa:
         - Acurácia: 1.0
         - Precisão: 1.0
         - Recall: 1.0

      - Iris-versicolor:
         - Acurácia: 0.97
         - Precisão: 0.9
         - Recall: 1.0

      - Iris-virginica:
         - Acurácia: 0.97
         - Precisão: 1.0
         - Recall: 0.89

3. Médias Gerais:
   - Regressão Logística
     - Acurácia Média: 0.87
     - Precisão Média: 0.81
     - Recall Médio: 1
  
   - KNN
     - Acurácia Média: 0.98
     - Precisão Média: 0.97
     - Recall Médio: 0.96

4. Conclusões:
   - Regressão Logística
     - A regressão logística apresentou um desempenho perfeito para a espécie Iris-setosa, com acurácia, precisão e recall todos igual a 1.0. Isso indica que o modelo conseguiu prever corretamente todas as instâncias desta espécie.
     - Para a espécie Iris-versicolor, o desempenho da regressão logística foi mais modesto, com uma acurácia de 0.6. Isso sugere que o modelo teve dificuldade em distinguir corretamente esta espécie. A precisão foi relativamente baixa (0.43), indicando um número significativo de falsos positivos, enquanto o recall foi alto (1.0), indicando que o modelo identificou corretamente todos os verdadeiros positivos desta espécie.
     - Assim como para Iris-setosa, a regressão logística teve um desempenho excelente para a espécie Iris-virginica, com acurácia, precisão e recall todos iguais a 1.0. Isso sugere que o modelo foi capaz de prever com precisão todas as instâncias dessa espécie.
  
   - KNN
     - O algoritmo KNN também teve um desempenho perfeito para a espécie Iris-setosa, com acurácia, precisão e recall todos iguais a 1.0. Isso indica que ambos os modelos foram bem-sucedidos na identificação desta espécie.
     - O KNN apresentou um desempenho notável para a espécie Iris-versicolor, com acurácia elevada (0.97), precisão robusta (0.9) e recall máximo (1.0). Isso sugere uma capacidade eficaz de distinguir esta espécie, com uma precisão mais elevada em comparação com a regressão logística.
     - Para a espécie Iris-virginica, o KNN também demonstrou um bom desempenho, com acurácia elevada (0.97), precisão perfeita (1.0) e recall forte (0.89). Isso indica uma capacidade geral de identificar corretamente instâncias desta espécie, com uma ênfase na minimização de falsos positivos.
  