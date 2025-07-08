# Audio-Processing-and-Classification

Este é um projeto de processamento e classificacao de audio que também tem o objetio de ser uma xposição e como funciona uma rede neural convolucional.

O projeto é dividido em quatro partes que serão expostas nesse readme. Cada parte do código está marcada com comentários que explicam sua função.

## Gravação

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

## Spectograma
Este código realiza o pré-processamento de áudios e gera espectrogramas. 
Espectogramas são representações visuais de frequências ao longo do tempo.

- Monta o Google Drive para acessar arquivos;
- Instala e importa bibliotecas necessárias (librosa, OpenCV, matplotlib, scipy);
- Aplica um filtro passa-banda para isolar frequências entre 100 Hz e 10.000 Hz, removendo ruídos indesejados, usados também para remover silêncios do início e fim do áudio;
- Converte o áudio em espectrogramas;
- Salva os espectrogramas em pastas separadas pelas categoria desejadas;
- Armazena os espectrogramas e seus rótulos em listas para uso futuro.

  ### Filtragem de Sinais (Filtro Passa-Banda)
#### O que é?

Um filtro passa-banda permite a passagem de frequências dentro de uma faixa específica (entre fcorte_inf e fcorte_sup), enquanto atenua frequências fora dessa faixa.
- Remove ruídos e frequências indesejadas (ex.: 50/60 Hz de interferência elétrica).
- Melhora a qualidade do áudio antes da análise espectral.

#### Como funciona no código?

- butter() (da biblioteca scipy.signal):

    -Projeta um filtro Butterworth (resposta suave na banda de passagem).
    -Parâmetros:
  
        order=5 → Quanto maior a ordem, mais "íngreme" é a filtragem.
        btype='band' → Define um filtro passa-banda.
        Frequências normalizadas (inf_normalz, sup_normalz) para evitar aliasing.

- filtfilt():
  
    Aplica o filtro duas vezes (ida e volta) para evitar atraso de fase (distorção temporal).
