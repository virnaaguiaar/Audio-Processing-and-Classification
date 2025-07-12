# Audio Processing and Classification

Este projeto é um sistema completo de processamento e classificação de áudio que também serve como guia educativo sobre o funcionamento de redes neurais convolucionais. 

## 📋 Visão Geral
O projeto está organizado em 4 etapas principais:
1. 🎙️ Gravação de áudio
2. 📊 Geração de espectrogramas
3. 🧠 Classificação com rede neural
4. 📈 Análise de métricas


## 🎙️ Gravação 

O código é um sistema de gravação de áudio que:
- Organiza os áudios em categorias (como "left" e "right");
- Cria uma pasta principal para armazenar os arquivos;
- Grava áudio usando o navegador e converte para formato WAV;
- Numera automaticamente cada nova gravação;
- Salva os arquivos em pastas específicas para cada comando;
- Possui um botão para iniciar o processo de gravação.


Primeiramente, é necessário criar um mecanismo de gravação de áudio caso você ainda não tenha os arquivos de áudio. Utilizei para esse mecanismo o GoogleColaboratoy que faz a ponte direta com o GoogleDrive para salvar os aquivos em pastas específicas.

    from google.colab import drive
    drive.mount('/content/drive')

    %cd /content/drive/MyDrive/audio2025/audios

Para a pasta de áudios gravados:

    output_dir = "/content/drive/MyDrive/audio2025/audios/gravados"  
    os.makedirs(output_dir, exist_ok=True)

Para a pasta dos espectogramas gerados através do processamento do áudio:

    espectrograma_dir = "/content/drive/MyDrive/audio2025/audios/espectrogramas"

Para salvar o modelo:
    
    with open('/content/drive/MyDrive/audio2025/audios/model.tflite', 'wb') as f:
      f.write(tflite_model)

    dado = np.load('/content/drive/MyDrive/audio2025/audios/dados_teste_validacao.npz')

    modelo = tf.keras.models.load_model('/content/drive/MyDrive/audio2025/audios/modelo.keras')


##  📊 Spectograma  

Este código realiza o pré-processamento de áudios e gera espectrogramas. 

#### 🔍 O que são espectogramas?

São representações visuais de frequências ao longo do tempo, onde:

- Eixo X: Tempo;
- Eixo Y: frequência (escala logarítmica);
- Cores: Intensidade (dB).


O código desta seção:
- Monta o Google Drive para acessar arquivos;
- Instala e importa bibliotecas necessárias (librosa, OpenCV, matplotlib, scipy);
- Aplica um filtro passa-banda para isolar frequências entre 100 Hz e 10.000 Hz, removendo ruídos indesejados, usados também para remover silêncios do início e fim do áudio;
- Converte o áudio em espectrogramas;
- Salva os espectrogramas em pastas separadas pelas categoria desejadas;
- Armazena os espectrogramas e seus rótulos em listas para uso futuro.

### → Filtragem de Sinais (Filtro Passa-Banda)
 #### 🔍 O que é?

 Um filtro passa-banda permite a passagem de frequências dentro de uma faixa específica (entre fcorte_inf e fcorte_sup), enquanto atenua frequências fora dessa faixa.
- Remove ruídos e frequências indesejadas (ex.: 50/60 Hz de interferência elétrica).
- Melhora a qualidade do áudio antes da análise espectral.

#### Como funciona no código?

|Parâmetros| Função | 
|--------|--------|
|`butter()`| Projeta um filtro Butterworth (resposta suave na banda de passagem) |
|`butter()` → `order=5` | Quanto maior a ordem, mais "íngreme" é a filtragem |
|`butter()` → `btype='band'` | Define um filtro passa-banda|
| `butter()` → `inf_normalz, sup_normalz` |Frequências normalizadas para evitar aliasing|
| `filtfilt()` |Aplica o filtro duas vezes (ida e volta) para evitar atraso de fase (distorção temporal)|
   

### → Pré-Processamento de Áudio (Librosa)
 #### Operações principais:

|Parâmetros| Função | Saída | Motivação |
|--------|--------|--------|--------|
|`librosa.stft()`| Divide o sinal em pequenos segmentos e calcula a Transformada de Fourier para cada um| Matriz complexa representando magnitudes e fases em diferentes frequências ao longo do tempo| |
|`librosa.amplitude_to_db()` |Converte amplitudes em decibéis (dB) (escala logarítmica) | |O ouvido humano percebe sons em escala logarítmica; Melhora o contraste em espectrogramas|

### → Tópicos Extra

- Short-Time Fourier Transform (stft):
    - Analisa sinais que variam com o tempo;
    - Divide o sinal em pequenos segmentos de tempo (frames) e aplica a Transformada de Fourier à cada um dos segmentos;
    - Produz um espectrograma.
  
- Taxa de Amostragem (sr) e Nyquist:
  
    Teorema de Nyquist:
    - Para reconstruir um sinal, a taxa de amostragem deve ser pelo menos o dobro da frequência máxima presente no sinal.
    - Ex.: Se sr=44100 Hz, a maior frequência detectável é 22050 Hz.

- Normalização de Frequências:

    Antes de aplicar o filtro, as frequências são normalizadas pela frequência de Nyquist para evitar distorções no filtro digital.

- Pré-requisito para Machine Learning:
  
    - Dados devem estar em formato numérico (imagens como arrays)
    - Rótulos devem ser codificados (ex.: 0, 1)

- Leitura com OpenCV (cv2.imread()):
  - Converte a imagem em um array NumPy para processamento posterior (ex.: redes neurais);
  - Permite pós-processamento de imagens (redimensionamento, equalização de histograma);
  - Compatível com frameworks de deep learning (ex.: TensorFlow, PyTorch).

##  📊 Classificação  

### → Camadas da Rede Neural

Uma rede neural é composta por: `CAMADA DE ENTRADA + CAMADA OCULTA + CAMADA DE SAÍDA` 

    
A camada de entrada representa todos os dados que damos para que o modelo seja trenidado. A seguinte é a camada oculta, que faz o treinamneto. É possível possuir mais de uma camada oculta, dependendo da finalidade do usuário. A última camada é a de saída, que nos dá o resultado do treinamento.

A camada oculta da rede neural é composta por outras camadas, como descrito abaixo:

| Camada Oculta| Função | Descrição |
|--------|--------|-----------------|
| `Conv2D` | Camada convolucional bidimensional | A principal camada |
| `MaxPooling2D` | Reduz pela metade (/2) as dimensões da imagem| Ao diminuir a resolução, diminui-se a complexidade computacional e evita overfitting |
| `Flatten` | Achata a entrada que é matriz multidimensional e o transforma em um vetor de uma única dimensão|Passo necessário antes de adicionar camadas densas|
| `Dense` |Cada neurônio está conectado a todos os neurônios da camada anterior |-|
| `Dropout` | Queima aleatoriamente alguns neurônios durante o treinamento |Reduzir o overfitting |
| `Input` | Define a forma da entrada da rede |Número de pixels da imagem e o número de canais de cor|



  ![Sem título](https://github.com/user-attachments/assets/4ee69baf-af77-44a8-bea3-e9004dd7fbf4)


### → Keras

O KERAS nos permite fazer um classificação de multiclasse.

No Keras podemos importar uma biblioteca que permite modelar de acordo com o padrão de uma rede neural, o `sequential`:  pilha linear de camadas, onde cada camada recebe a saída da camada anterior

    from tensorflow_keras import Sequential

### → Configuração de Treinamento em Keras

#### Otimizador Adam (Adaptive Moment Estimation)
- Ajusta as taxas de aprendizado individualmente para cada parâmetro da rede neural
- Combina os benefícios do **momentum** e da **adaptação de taxas de aprendizado**
- Melhora a eficiência e estabilidade do treinamento

#### Função de Perda: SparseCategoricalCrossentropy
- Utilizada para problemas de **classificação multiclasse**
- Versão "esparsa" significa que os labels são fornecidos como **inteiros** (ex: [0, 2, 1]) em vez de **one-hot encoding** (ex: [[1,0,0], [0,0,1], [0,1,0]])

##### Parâmetro `from_logits`:
| Valor  | Comportamento |
|--------|---------------|
| `False` | A saída da rede já está normalizada (probabilidades via softmax) |
| `True`  | A rede retorna logits (valores brutos, não normalizados). Keras aplicará softmax automaticamente |

#### Exemplo de Código

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )


##  📊 Métricas  

### → Análise de Desempenho do Modelo

#### Métricas de Classificação

| Valor  | Comportamento | Equação |
|--------|---------------|---------------|
| `Precision` | Mede a proporção de exemplos positivos corretamente classificados entre todos os classificados como positivos| $\frac{VP}{VP + FP}$|
| `Recall/Sensitivity` |  Mede a proporção de exemplos positivos corretamente classificados entre todos os que realmente são positivos|$\frac{VP}{VP + FN}$ |
| `F1-Score` | Média harmônica entre Precisão e Sensibilidade| $2 \times \frac{\text{Precisão} \times \text{Sensibilidade}}{\text{Precisão} + \text{Sensibilidade}}$|
| `Especificidade`  | Mede a proporção de negativos corretamente identificados: |$\frac{VN}{VN + FP}$ |
| ` Accuracy` | Monitora a **porcentagem de previsões corretas** durante o treinamento e validação |$\frac{\text{Previsões Corretas}}{\text{Total de Exemplos}}$|

#### Legenda:
- **VP (Verdadeiros Positivos):** Casos positivos corretamente classificados
- **VN (Verdadeiros Negativos):** Casos negativos corretamente classificados
- **FP (Falsos Positivos):** Casos negativos erroneamente classificados como positivos
- **FN (Falsos Negativos):** Casos positivos erroneamente classificados como negativos


### → Matriz de Confusão

#### 🔍 O que é?
 
É uma tabela que compara as previsões do modelo com os valores reais (rótulos verdadeiros). Cada célula mostra quantas vezes uma combinação específica de previsão e valor real ocorreu.

- Diagonal principal: Acertos do modelo.
- Demais células: Erros (confusões entre classes).

![Sem título(1)](https://github.com/user-attachments/assets/73d15459-57a1-4856-8234-40fc84953edb)

Por que usar?

- Identifica padrões de erro: Quais classes são mais confundidas.
- Vai além da acurácia: Mostra trade-offs entre FP e FN.
- Essencial para classes desbalanceadas.

A partir da matriz, calculamos:

- Acurácia (Accuracy)
- Precisão (Precision)
- Recall/Sensibilidade (Recall)
