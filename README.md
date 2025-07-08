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


