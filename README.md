# Audio Processing and Classification

Este é um projeto de processamento e classificacao de audio que também tem o objetio de ser uma exposição de como funciona uma rede neural convolucional.

O projeto é dividido em quatro partes que serão expostas nesse readme. Cada parte do código está marcada com comentários que explicam sua função.

## 🎙️ Gravação 

O código é um sistema de gravação de áudio que:
- Organiza os áudios em categorias (como "left" e "right");
- Cria uma pasta principal para armazenar os arquivos;
- Grava áudio usando o navegador e converte para formato WAV;
- Numera automaticamente cada nova gravação;
- Salva os arquivos em pastas específicas para cada comando;
- Possui um botão para iniciar o processo de gravação.


Primeiramente é necessário criar um mecanismo de gravação de áudio caso você ainda nao tenha os arquivos de áudio. Utilizei para esse mecanismo o googleColaboratoy que faz a ponte direta com o googleDrive para salvar os aquivos em pastas específicas.

    from google.colab import drive
    drive.mount('/content/drive')

    %cd /content/drive/MyDrive/audio2025/audios


    -Para a pasta de áudios gravados:

    output_dir = "/content/drive/MyDrive/audio2025/audios/gravados"  
    os.makedirs(output_dir, exist_ok=True)


    - Para a pasta dos espectogramas gerados através do processamento do áudio:

    espectrograma_dir = "/content/drive/MyDrive/audio2025/audios/espectrogramas"


    - Para salvar o modelo:
    
    with open('/content/drive/MyDrive/audio2025/audios/model.tflite', 'wb') as f:
      f.write(tflite_model)

    dado = np.load('/content/drive/MyDrive/audio2025/audios/dados_teste_validacao.npz')

    modelo = tf.keras.models.load_model('/content/drive/MyDrive/audio2025/audios/modelo.keras')

##  📊 Spectograma  
Este código realiza o pré-processamento de áudios e gera espectrogramas. 

#### O que são espectogramas?

São representações visuais de frequências ao longo do tempo, onde:

- Eixo X: Tempo;
- Eixo Y: frequência (escala logarítmica);
- Cores: Intensidade (dB).


O códio desta seção:
- Monta o Google Drive para acessar arquivos;
- Instala e importa bibliotecas necessárias (librosa, OpenCV, matplotlib, scipy);
- Aplica um filtro passa-banda para isolar frequências entre 100 Hz e 10.000 Hz, removendo ruídos indesejados, usados também para remover silêncios do início e fim do áudio;
- Converte o áudio em espectrogramas;
- Salva os espectrogramas em pastas separadas pelas categoria desejadas;
- Armazena os espectrogramas e seus rótulos em listas para uso futuro.

### → Filtragem de Sinais (Filtro Passa-Banda)
 #### O que é?

 Um filtro passa-banda permite a passagem de frequências dentro de uma faixa específica (entre fcorte_inf e fcorte_sup), enquanto atenua frequências fora dessa faixa.
- Remove ruídos e frequências indesejadas (ex.: 50/60 Hz de interferência elétrica).
- Melhora a qualidade do áudio antes da análise espectral.

#### Como funciona no código?

- butter():

    Projeta um filtro Butterworth (resposta suave na banda de passagem).

    Parâmetros:
  
        order=5 → Quanto maior a ordem, mais "íngreme" é a filtragem.
        btype='band' → Define um filtro passa-banda.
        Frequências normalizadas (inf_normalz, sup_normalz) para evitar aliasing.

- filtfilt():
  
    Aplica o filtro duas vezes (ida e volta) para evitar atraso de fase (distorção temporal).

### → Pré-Processamento de Áudio (Librosa)
 #### Operações principais:
 - librosa.stft() (Short-Time Fourier Transform)

    - Divide o sinal em pequenos segmentos e calcula a Transformada de Fourier para cada um.

    - Saída: Matriz complexa representando magnitudes e fases em diferentes frequências ao longo do tempo.

- librosa.amplitude_to_db()

    - Converte amplitudes em decibéis (dB) (escala logarítmica).

    Motivação:
    - O ouvido humano percebe sons em escala logarítmica.

    - Melhora o contraste em espectrogramas.


### → Tópicos Extra

- Taxa de Amostragem (sr) e Nyquist
    Teorema de Nyquist:
    - Para reconstruir um sinal, a taxa de amostragem deve ser pelo menos o dobro da frequência máxima presente no sinal.
    Ex.: Se sr=44100 Hz, a maior frequência detectável é 22050 Hz.

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

#keras: classificação de multiclasse / sequential: pilha linear de camadas, onde cada camada recebe a saída da camada anterior

from tensorflow_keras import Sequential

'''
Conv2D: camada convolucional bidimensional, a principal camada

MaxPooling2D: Camada de pooling, que reduz(/2) as dimensões da imagem (diminuindo a resolução) = diminuir a complexidade computacional/evitar overfitting

Flatten: "achata" a entrada q é matriz multidimensional -> vetor de uma única dimensão / (antes de adicionar camadas densas

Dense: Camada densa - cada neurônio está conectado a todos os neurônios da camada anterior

Dropout(reguarização): Queima aleatoriamente alguns neurônios durante o treinamento (reduzir o overfitting)

Input: Define a forma da entrada da rede (número de pixels da imagem e o número de canais de cor)'''
 






#Analisar o desempenho do modelo em termos de falsos positivos e falsos negativos

#Precisão: exemplos positivos corretamente classificados olhando os classificados positivos.
#Precisão = (Verdadeiros Positivos) / (Verdadeiros Positivos + Falsos Positivos)
#Sensibilidade: exemplos positivos corretamente classificados olhando os que realmente são positivos
#Sensibilidade = (Verdadeiros Positivos) / (Verdadeiros Positivos + Falsos Negativos)






#Adam (Adaptive Moment Estimation): ajusta as taxas de aprendizado para cada parâmetro da rede, melhorando a eficiência e estabilidade do treinamento
#Sparse...: Função de perda utilizada em classificação multiclasse. Com (labels) são números inteiros (em vez de vetores one-hot)
#from_logits=False: indica que a saída da rede neural (logits) já é normalizada pelo softmax. // =True: indica que a rede retorna logits não normalizados // Keras irá aplicar a softmax automaticamente
#Keras monitorar a porcentagem de previsões corretas do modelo

    model.compile(optimizar='adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])






#classification_report: resumo completo das principais métricas de desempenho de um modelo de classificação - precisão, recall, F1-score, acurácia, para cada classe
#confusion_matrix: cria uma matriz de confusão - comparação entre as previsões do modelo e os rótulos reais (cada célula da matriz: número de ocorrências de cada combinação de rótulos previstos e reais)
#accuracy_score: (previsões corretas)/(total)

