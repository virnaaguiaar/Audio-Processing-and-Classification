# Audio Processing and Classification

This project is a complete audio processing and classification system that also serves as an educational guide on how convolutional neural networks work. 

## 📋 Overview
The project is organized into 4 main stages:
1. 🎙️ Audio recording
2. 📊 Spectrogram generation
3. 🧠 Neural network classification
4. 📈 Metric analysis

## 🎙️ Recording

The code is an audio recording system that:
- Organizes audio files into categories (such as "left" and "right");
- Creates a main folder to store the files;
- Records audio using the browser and converts it to WAV format;
- Automatically numbers each new recording;
- Saves files into specific folders for each command;
- Includes a button to start the recording process.

First, you need to create an audio recording mechanism if you don't already have the audio files. For this, I used Google Colaboratory, which connects directly to Google Drive to save the files into specific folders.

```python

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/audio2025/audios
```
For recorded audio files:

```python
output_dir = "/content/drive/MyDrive/audio2025/audios/recordings" 
os.makedirs(output_dir, exist_ok=True)
```


For the spectrograms folder (generated from audio processing):
```python
spectrogram_dir = "/content/drive/MyDrive/audio2025/audios/spectrograms"
```

For saving the trained model:
```python
with open('/content/drive/MyDrive/audio2025/audios/model.tflite', 'wb') as f:
  f.write(tflite_model)

dado = np.load('/content/drive/MyDrive/audio2025/audios/dados_teste_validacao.npz')

model = tf.keras.models.load_model('/content/drive/MyDrive/audio2025/audios/modelo.keras')
```

##  📊 Spectrogram

This code performs audio pre-processing and generates spectrograms.

#### 🔍 What are spectrograms?

Spectrograms are visual representations of frequency content over time, where:

- X-axis: Time
- Y-axis: Frequency (logarithmic scale)
- Colors: Intensity (dB)

🛠️ What this code does:

- Google Drive integration:
    - Mounts Google Drive to access audio files
- Dependencies:
    - Installs and imports required libraries (librosa, OpenCV, matplotlib, scipy)
- Audio processing:
    - Applies bandpass filter (100 Hz to 10,000 Hz) to isolate target frequencies and remove noise
    - Automatically trims silence from beginning/end of audio clips
- Spectrogram conversion:
    - Converts filtered audio to spectrogram images
    - Saves spectrograms in category-specific folders (e.g., "left", "right")
    - Stores spectrograms and their labels in arrays for model training


### → Signal Filtering (Bandpass Filter)
 #### 🔍 What is it?

 A bandpass filter allows frequencies within a specific range (between low_cut and high_cut) to pass through, while attenuating frequencies outside this range:
- Removes noise and unwanted frequencies (e.g., 50/60 Hz electrical interference)
- Enhances audio quality before spectral analysis

#### How does it work in the code?

|Parameter| Function | 
|--------|--------|
|`butter()`| Designs a Butterworth filter (maximally flat frequency response in passband) |
|`butter()` → `order=5` | Higher order = steeper roll-off (more aggressive filtering) |
|`butter()` → `btype='band'` | Specifies bandpass filter type |
| `butter()` → `inf_normalz, sup_normalz` | Normalized frequencies (prevents aliasing) |
| `filtfilt()` | Zero-phase filtering (applies filter forward+backward to eliminate phase delay) |
   

### → Audio Preprocessing (Librosa)
 #### Core Operations:

|Parameter| Purpose | Output | Motivation |
|--------|--------|--------|--------|
|`librosa.stft()`| Splits signal into short segments and computes Fourier Transform for each| Complex matrix representing magnitude and phase across frequencies over time| Time-frequency decomposition for spectral analysis|
|`librosa.amplitude_to_db()` |	Converts amplitudes to decibel (dB) scale logarithmic | dB-scaled spectrogram|Human hearing perceives sound logarithmically; Enhances spectrogram contrast|

### → Additional Topics

- **Short-Time Fourier Transform (stft):**
    - Analyzes time-varying signals
    - Splits signal into short time segments (frames) and applies Fourier Transform to each segment
    - Produces a spectrogram (time-frequency representation)
  
- **Sampling Rate (sr) and Nyquist Theorem:**
  
    Nyquist Theorem:
    - Sampling rate must be ≥ 2 × maximum signal frequency
    - Ex.: For sr=44100 Hz, the max detectable frequancy is 22050 Hz

- **Frequency Normalization:**

    Before applying the filter, frequencies are normalized by the Nyquist frequency to prevent distortion in digital filtering.

- **Machine Learning Prerequisites:**
  
    - Data must be in numerical format (e.g., images as NumPy arrays)
    - Labels must be encoded (e.g., `0`, `1`) 

- **Reading with OpenCV (`cv2.imread()`):**
  - Converts an image into a NumPy array for further processing (e.g., neural networks)  
  - Enables post-processing such as resizing or histogram equalization  
  - Compatible with deep learning frameworks (e.g., TensorFlow, PyTorch)  

##  🧠 Classificação  

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

No Keras podemos importar uma biblioteca que permite modelar de acordo com o padrão de uma rede neural, o `Sequential`:  pilha linear de camadas, onde cada camada recebe a saída da camada anterior
```python
from tensorflow_keras import Sequential
```

### → Configuração de Treinamento em Keras

#### Otimizador Adam (Adaptive Moment Estimation)
- Ajusta as taxas de aprendizado individualmente para cada parâmetro da rede neural;
- Combina os benefícios do **momentum** e da **adaptação de taxas de aprendizado**;
- Melhora a eficiência e estabilidade do treinamento.

#### Função de Perda: SparseCategoricalCrossentropy
- Utilizada para problemas de **classificação multiclasse**;
- Versão "esparsa" significa que os labels são fornecidos como **inteiros** (ex: [0, 2, 1]) em vez de **one-hot encoding** (ex: [[1,0,0], [0,0,1], [0,1,0]]).

##### Parâmetro `from_logits`:
| Valor  | Comportamento |
|--------|---------------|
| `False` | A saída da rede já está normalizada (probabilidades via softmax) |
| `True`  | A rede retorna logits (valores brutos, não normalizados). Keras aplicará softmax automaticamente |

#### Exemplo de Código
```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

##  📈 Métricas  

### → Análise de Desempenho do Modelo

#### Métricas de Classificação

| Valor  | Comportamento | Equação |
|--------|---------------|---------------|
| `Precision` | Mede a proporção de exemplos positivos corretamente classificados entre todos os classificados como positivos| $\frac{VP}{VP + FP}$|
| `Recall` |  Mede a proporção de exemplos positivos corretamente classificados entre todos os que realmente são positivos|$\frac{VP}{VP + FN}$ |
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

- Identifica padrões de erro: quais classes são mais confundidas;
- Vai além da acurácia: mostra trade-offs entre FP e FN;
- Essencial para classes desbalanceadas.

A partir da matriz, calculamos:

- Accuracy;
- Precision;
- Recall.
