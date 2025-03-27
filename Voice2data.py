import os
import math
import argparse
import numpy as np
import librosa
from pydub import AudioSegment
import csv


def convert_to_wav(input_path, output_path):
    """
    Converte o arquivo de áudio para formato WAV.
    """
    # Detecta a extensão do arquivo
    ext = os.path.splitext(input_path)[1].lower()
    # Caso já seja WAV, copia o arquivo
    if ext == ".wav":
        AudioSegment.from_wav(input_path).export(output_path, format="wav")
    else:
        # Carrega com pydub (suporta mp3, flac, wma, etc.)
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")


def extract_features(frame, sr):
    """
    Extrai os recursos do frame de áudio.
    Cada função do librosa é configurada para operar com a janela inteira.
    Retorna:
        sc  : Centróide Espectral (float)
        sb  : Largura de Banda Espectral (float)
        sr  : Rolloff Espectral (float)
        zcr : Taxa de Cruzamento por Zero (float)
        rms : Energia RMS (float)
        chroma: array de 7 valores (chromagram)
        mfcc: array de 13 valores
    """
    n_fft = len(frame)
    hop_length = n_fft  # Para obter um único valor por frame

    # Caso o frame seja totalmente silêncio ou nulo, forçamos zeros
    if np.all(frame == 0):
        sc = sb = sr_roll = zcr_val = rms_val = 0.0
        chroma = np.zeros(7)
        mfcc = np.zeros(13)
    else:
        sc = librosa.feature.spectral_centroid(y=frame, sr=sr, n_fft=n_fft, hop_length=hop_length)[0, 0]
        sb = librosa.feature.spectral_bandwidth(y=frame, sr=sr, n_fft=n_fft, hop_length=hop_length)[0, 0]
        sr_roll = librosa.feature.spectral_rolloff(y=frame, sr=sr, n_fft=n_fft, hop_length=hop_length)[0, 0]
        zcr_val = librosa.feature.zero_crossing_rate(y=frame, frame_length=n_fft, hop_length=hop_length)[0, 0]
        rms_val = librosa.feature.rms(y=frame, frame_length=n_fft, hop_length=hop_length)[0, 0]
        # Chromagram com 7 bins
        chroma = librosa.feature.chroma_stft(y=frame, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=7).flatten()
        # MFCC com 13 coeficientes
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).flatten()
    return sc, sb, sr_roll, zcr_val, rms_val, chroma, mfcc


def min_max_normalize(data):
    """
    Normaliza cada coluna (exceto a primeira, que contém o timestep) utilizando min-max.
    Se max == min, os valores serão todos zero.
    """
    data_norm = data.copy()
    # Itera pelas colunas a partir da segunda (índice 1)
    for j in range(1, data.shape[1]):
        col = data[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        if max_val - min_val != 0:
            data_norm[:, j] = (col - min_val) / (max_val - min_val)
        else:
            data_norm[:, j] = 0.0
    return data_norm


def process_audio(input_file, output_csv):
    # Cria um arquivo temporário WAV
    temp_wav = "temp_audio.wav"
    convert_to_wav(input_file, temp_wav)

    # Carrega o áudio preservando a taxa de amostragem original
    y, sr = librosa.load(temp_wav, sr=None)

    # Normaliza a amplitude (escala entre -1 e 1)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Remove silêncios do início e fim
    y, _ = librosa.effects.trim(y, top_db=20)

    # Define o tamanho do frame: 20ms
    frame_length = int(sr * 0.02)
    total_samples = len(y)
    n_frames = math.ceil(total_samples / frame_length)

    # Lista para armazenar os dados: cada linha será [timestep, sc, sb, sr, zcr, rms, chroma (7), mfcc (13)]
    features_list = []

    for i in range(n_frames):
        start = i * frame_length
        end = start + frame_length
        # Se o frame estiver incompleto, faz padding com zeros
        frame = y[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
        # Calcula o timestamp (em segundos) do início do frame
        timestamp = start / sr
        sc, sb, sr_roll, zcr_val, rms_val, chroma, mfcc = extract_features(frame, sr)
        # Junta os recursos em uma única lista
        row = [timestamp, sc, sb, sr_roll, zcr_val, rms_val]
        row.extend(chroma.tolist())
        row.extend(mfcc.tolist())
        features_list.append(row)

    # Converte para array numpy para normalização
    data_array = np.array(features_list)
    data_norm = min_max_normalize(data_array)

    # Formata os valores com 16 casas decimais
    formatted_data = [[f"{val:.16f}" for val in row] for row in data_norm]

    # Define cabeçalhos para o CSV
    headers = ['timestamp', 'sc', 'sb', 'sr', 'zcr', 'rms']
    headers += [f'chroma_{i + 1}' for i in range(7)]
    headers += [f'mfcc_{i + 1}' for i in range(13)]

    # Escreve o CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(formatted_data)

    # Remove o arquivo temporário
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    print(f"Processamento concluído. CSV salvo em: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processamento de áudio: conversão, normalização, remoção de silêncio, extração de features e salvamento em CSV.")
    parser.add_argument("input_file", help="C:/Users/thier/OneDrive/Documentos/amostra.mp3")
    parser.add_argument("output_csv", help="C:/Users/thier/OneDrive/Documentos/audio2data")
    args = parser.parse_args()

    process_audio(args.input_file, args.output_csv)
