# Audio-Processing-and-Classification

Este é um projeto de processamento e classificacao de audio que também tem o objetio de ser uma xposição e como funciona uma rede neural convolucional.

O projeto é dividido em quatro partes que serão expostas nesse readme.

## Gravação
Primeiramente é necessário creiar um mecanismo de gravação de áudio caso você ainda nao tenha os arquivos de áudio. Utilizei para esse mecanismo o googleColaboratoy que faz a ponte direta com o googleDrive para salvar os aquivos em pastas específicas.

    # Categories (Commands)

    # commands

    # main output directory

    # Function to record audio and return the WAV data

    # Save the audio to a temporary file

    # Convert audio to WAV using FFmpeg

    # Reading WAV File Data

    # Remove temporary files

    # function to process audio files and save them in their respective folders

    # Create a category folder if it doesn't exist,

    # Get the next recording number

    # Generate the next available WAV filename with automatic numbering

    # Record audio and automatically save

    # Function to start recording with a button

Explicação simples:
O código é um sistema de gravação de áudio que:

    Organiza os áudios em categorias (como "left" e "right")

    Cria uma pasta principal para armazenar os arquivos

    Grava áudio usando o navegador e converte para formato WAV

    Numera automaticamente cada nova gravação

    Salva os arquivos em pastas específicas para cada comando

    Possui um botão para iniciar o processo de gravação

Cada parte do código está marcada com comentários que explicam sua função, desde a criação de diretórios até o processamento e salvamento dos arquivos de áudio.
