# Modelo de Previsão para Lotofácil com Deep Learning Híbrido (Transformer-LSTM)

Este projeto apresenta um modelo avançado de Deep Learning para prever números de sorteios da Lotofácil. A abordagem combina o poder de redes Long Short-Term Memory (LSTM) para capturar dependências temporais em sequências com a capacidade inovadora de Transformers (com atenção multi-cabeça) para identificar relações complexas e de longo alcance nos dados. O objetivo é demonstrar a aplicação de arquiteturas híbridas e de última geração em problemas de previsão de séries temporais.

## Tecnologias Utilizadas
* **Linguagem de Programação:** Python
* **Frameworks/Bibliotecas:**
    * TensorFlow & Keras (para construção e treinamento do modelo de Deep Learning, incluindo a camada MultiHeadAttention e LayerNormalization)
    * Pandas (para manipulação e pré-processamento de dados)
    * NumPy (para operações numéricas eficientes)
    * Scikit-learn (para pré-processamento de dados (e.g., MinMaxScaler))
    * Matplotlib (para visualização)

## Arquitetura do Modelo

O coração deste projeto é um modelo de rede neural híbrido que integra:
Transformer Encoder Block;
Utiliza camadas de Atenção Multi-Cabeça (MultiHeadAttention) para ponderar a importância de diferentes partes da sequência de entrada, permitindo ao modelo focar nas informações mais relevantes, independentemente de sua posição.
Inclui redes feed-forward e normalização de camada para estabilizar e aprimorar o aprendizado.
Rede Long Short-Term Memory (LSTM);
Especialmente eficaz para processar e aprender padrões em sequências de dados temporais, capturando dependências de curto e médio prazo.
Combinação e Camadas Densas;
As saídas do Transformer (após GlobalAveragePooling1D) e da LSTM são concatenadas.
Essa representação combinada é então passada por camadas densas (Fully Connected) para realizar a previsão final dos números do sorteio.
Esta arquitetura tira proveito das strengths de ambas as abordagens para uma análise mais robusta das sequências de sorteios.

## Estrutura do Projeto

* `Transformer_LSTM_Lotofacil.py`: O notebook principal contendo todo o código-fonte do modelo, treinamento, avaliação e simulações.
* `requirements.txt`: Lista das bibliotecas Python e suas versões necessárias para a execução do projeto.
* `Lotofacil-original100.xlsx: Conjunto de dados de exemplo (subconjunto de 100 sorteios) utilizado para o treinamento e teste do modelo.
* `images/`: Pasta contendo gráficos de desempenho gerados.

## Como Executar o Projeto

Este projeto foi desenvolvido e otimizado para ser executado no [Google Colab](https://colab.research.google.com/) devido às suas dependências específicas e acesso a GPUs para treinamento eficiente.

1.  **Clone o Repositório** (ou baixe o ZIP):
    ```bash
    git clone [https://github.com/sergiotabarez/hybrid-lottery-prediction-model.git](https://github.com/sergiotabarez/hybrid-lottery-prediction-model.git)
    cd lottery-prediction-model
    ```

2.  **Abra o Notebook no Google Colab:**
    * Vá para [https://colab.research.google.com/](https://colab.research.google.com/)
    * Clique em `File` (Arquivo) > `Upload notebook` (Carregar notebook) e selecione o arquivo `Transformer_LSTM_lotofacil.py` que você baixou do repositório.
    * *(Alternativamente, se você já clonou o repositório para seu Google Drive ou localmente e quer abrir direto no Colab pelo GitHub, você pode ir em `File` > `Open notebook` > `GitHub` e colar o link direto para o seu notebook no GitHub: `https://github.com/sergiotabarez/lottery-prediction-model/blob/main/Deep_Learning_LSTM.py`)*

3.  **Carregue o Arquivo de Dados (`Lotofacil-original100.xlsx`):**
    * No ambiente do Colab, utilize a opção de upload de arquivos (ícone de pasta na barra lateral esquerda).
    * Faça upload do arquivo `Lotofacil-original100.xlsx` para o ambiente de execução do Colab.
    * **Alternativamente, no próprio notebook, você pode incluir um código para montar o Google Drive e carregar o arquivo de lá, ou usar `files.upload()` do Colab Utilities.** *(Decida qual método de carregamento de dados você usou no seu notebook e instrua o usuário de acordo.)*

4.  **Instale as Dependências:**
    * Dentro do Colab, crie uma nova célula de código e execute o seguinte comando para instalar todas as bibliotecas necessárias:
        ```python
        !pip install -r requirements.txt
        ```

5.  **Execute a Célula**

## Resultados e Análise

* **Gráfico de Perda (Loss History):** Demonstra a convergência do modelo durante o treinamento.
    ![Gráfico de Perda](images/grafico_perda.png) 
   
* **Exemplo do Modelo** 
    ![Exemplo de Código](images/modelo.png)

* **Bibliotecas**
    ![Resultados da Simulação](images/bibliotecas.png)

## Observações

Este modelo híbrido Transformer-LSTM representa um avanço significativo na tentativa de modelar a Lotofácil. Embora o projeto utilize um subconjunto de 100 sorteios para demonstração e otimização do tempo de treinamento, a arquitetura foi projetada para escalar.

## Autor

**Sergio Tabarez**

[www.linkedin.com/in/sergiotabarez]

[sergio.tabarez@gmail.com]

[https://github.com/sergiotabarez/lottery-prediction-model]

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
