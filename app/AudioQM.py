# AudioQM.py
from __future__ import annotations

import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from typing import Dict, Any, Tuple, Optional
import logging
import os

# Configura logger para AudioQM
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [AudioQM] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================ FUNÇÕES DE CÁLCULO DE MÉTRICAS DE ÁUDIO ============================

def _load_audio_data(audio_path: str, sr: Optional[int] = None) -> Optional[Tuple[np.ndarray, int]]:
    """
    Carrega um arquivo de áudio usando librosa.
    Retorna (dados_audio, sample_rate) ou None em caso de erro.
    """
    try:
        # Tenta carregar com pydub primeiro para obter o sample_rate original de forma mais robusta
        # e depois usa librosa para carregar com o sr desejado ou original.
        audio_segment = AudioSegment.from_file(audio_path)
        original_sr = audio_segment.frame_rate
        audio_data, loaded_sr = librosa.load(audio_path, sr=sr if sr else original_sr, mono=True)
        return audio_data, loaded_sr
    except CouldntDecodeError:
        logger.warning(
            f"Não foi possível decodificar o arquivo de áudio: {audio_path}. Formato inválido ou corrompido.")
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar áudio {audio_path}: {e}")
        return None


def calculate_clipping(audio_data: np.ndarray) -> float:
    """
    Calcula a proporção de amostras que estão no limite máximo (clipping).
    Retorna uma porcentagem.
    """
    if audio_data.size == 0:
        return 0.0
    # Define um pequeno delta para considerar clipping próximo ao limite
    threshold = 0.999 * np.max(np.abs(audio_data))
    clipped_samples = np.sum(np.abs(audio_data) >= threshold)
    clipping_ratio = (clipped_samples / audio_data.size) * 100
    return clipping_ratio


def calculate_silence_ratio(audio_data: np.ndarray, sr: int, top_db: float = 20) -> float:
    """
    Calcula a proporção de silêncio no áudio usando detecção de voz/som baseada em energia.
    Retorna uma porcentagem.
    """
    if audio_data.size == 0:
        return 100.0  # 100% de silêncio se não há dados
    # Usa librosa.effects.split para identificar segmentos não-silenciosos
    # top_db: o limiar em dB abaixo do pico para considerar silêncio
    non_silent_intervals = librosa.effects.split(audio_data, top_db=top_db)
    total_non_silent_duration = np.sum([interval[1] - interval[0] for interval in non_silent_intervals])
    total_duration_samples = len(audio_data)

    if total_duration_samples == 0:
        return 100.0

    silence_ratio = (1 - (total_non_silent_duration / total_duration_samples)) * 100
    return silence_ratio


def calculate_dynamic_range(audio_data: np.ndarray) -> float:
    """
    Calcula a faixa dinâmica do áudio em dB.
    Diferença entre o pico e o RMS (Root Mean Square) médio.
    """
    if audio_data.size == 0:
        return 0.0
    peak_amplitude = np.max(np.abs(audio_data))
    rms = np.sqrt(np.mean(audio_data ** 2))

    if rms == 0 or peak_amplitude == 0:  # Evita log de zero
        return 0.0

    # Adiciona um pequeno valor para evitar log de zero se rms for muito pequeno
    dynamic_range_db = 20 * np.log10(peak_amplitude / (rms + 1e-10))
    return dynamic_range_db


def calculate_bandwidth(audio_data: np.ndarray, sr: int) -> float:
    """
    Estima a largura de banda (bandwidth) do áudio em kHz.
    Usa o centróide espectral e a dispersão espectral como indicadores.
    Retorna o valor em kHz.
    """
    if audio_data.size == 0:
        return 0.0
    # Calcula o espectrograma STFT
    stft = np.abs(librosa.stft(audio_data))

    # Calcula o centróide espectral (média ponderada das frequências)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, S=stft)[0]
    # Calcula a dispersão espectral (desvio padrão das frequências)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, S=stft)[0]

    # Uma estimativa simples de largura de banda pode ser Centroid + Dispersão
    # Ou apenas o centróide como um indicador da "frequência central"
    # Para uma estimativa mais robusta, pode-se usar a frequência onde a energia cai para um certo percentual.
    # Aqui, vamos usar o centróide médio como um proxy para a "frequência principal"
    # e adicionar a dispersão para ter uma ideia da "largura" do espectro.
    # Isso é uma simplificação, a largura de banda real é mais complexa.
    avg_centroid = np.mean(spectral_centroid)
    avg_bandwidth_spread = np.mean(spectral_bandwidth)

    # Uma heurística simples: centróide + 2*desvio padrão (aproximação de 95% da energia)
    # ou apenas o centróide como um indicador primário.
    # Para evitar valores muito altos, vamos usar o centróide como o principal indicador de "frequência ativa"
    # e garantir que não exceda o Nyquist.
    estimated_bandwidth = min(avg_centroid + avg_bandwidth_spread, sr / 2)  # Limita à frequência de Nyquist

    return estimated_bandwidth / 1000.0  # Retorna em kHz


def calculate_snr_simple(audio_data: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512,
                         top_db: float = 20) -> float:
    """
    Calcula uma estimativa simples de SNR (Signal-to-Noise Ratio) em dB.
    Compara a energia de segmentos de fala com a energia de segmentos de silêncio.
    Isso é uma aproximação e não um SNR verdadeiro que requer ruído puro.
    """
    if audio_data.size == 0:
        return 0.0

    # Identifica segmentos não-silenciosos (provavelmente fala)
    non_silent_intervals = librosa.effects.split(audio_data, top_db=top_db, frame_length=frame_length,
                                                 hop_length=hop_length)

    speech_energy = 0.0
    noise_energy = 0.0
    total_samples = len(audio_data)

    # Coleta energia de segmentos de fala
    for start, end in non_silent_intervals:
        speech_energy += np.sum(audio_data[start:end] ** 2)

    # Coleta energia de segmentos de "ruído" (partes que não são fala)
    last_end = 0
    for start, end in non_silent_intervals:
        noise_energy += np.sum(audio_data[last_end:start] ** 2)
        last_end = end
    noise_energy += np.sum(audio_data[last_end:total_samples] ** 2)  # Adiciona o final

    # Evita divisão por zero
    if noise_energy == 0:
        return 99.0  # Retorna um valor alto para indicar ruído muito baixo
    if speech_energy == 0:
        return 0.0  # Se não há fala, SNR é 0 ou negativo

    snr_db = 10 * np.log10(speech_energy / noise_energy)
    return snr_db


def get_audio_metrics(audio_path: str, sr: Optional[int] = None) -> Dict[str, Any]:
    """
    Calcula um conjunto abrangente de métricas de qualidade para um arquivo de áudio.
    """
    metrics: Dict[str, Any] = {
        'clipping': 'N/A',
        'snr_avg': 'N/A',
        'silence_ratio_avg': 'N/A',
        'dynamic_range_avg': 'N/A',
        'bandwidth_avg': 'N/A',
        'pesq_avg': 'N/A',  # Placeholder, requer referência
        'stoi_avg': 'N/A',  # Placeholder, requer referência
        'si_sdr_avg': 'N/A'  # Placeholder, requer referência
    }

    audio_data_sr = _load_audio_data(audio_path, sr=sr)
    if audio_data_sr is None:
        return metrics  # Retorna N/A se o áudio não puder ser carregado

    audio_data, loaded_sr = audio_data_sr

    try:
        metrics['clipping'] = calculate_clipping(audio_data)
        metrics['silence_ratio_avg'] = calculate_silence_ratio(audio_data, loaded_sr)
        metrics['dynamic_range_avg'] = calculate_dynamic_range(audio_data)
        metrics['bandwidth_avg'] = calculate_bandwidth(audio_data, loaded_sr)
        metrics['snr_avg'] = calculate_snr_simple(audio_data, loaded_sr)
    except Exception as e:
        logger.error(f"Erro ao calcular métricas para {audio_path}: {e}")
        # As métricas permanecerão 'N/A' conforme inicializado

    # PESQ, STOI, SI-SDR são complexos e geralmente exigem um sinal de referência limpo.
    # Para um dataset genérico sem referências, eles não podem ser calculados de forma significativa.
    # Mantenha como 'N/A' ou implemente com bibliotecas externas se o requisito for estrito
    # e você tiver os dados de referência.
    # Exemplo (requer instalação de pypesq, pystoi, pyssi):
    # from pypesq import pesq
    # from pystoi.stoi import stoi
    # from pyss_sdr import si_sdr
    # metrics['pesq_avg'] = pesq(ref_audio, deg_audio, sr)
    # metrics['stoi_avg'] = stoi(ref_audio, deg_audio, sr)
    # metrics['si_sdr_avg'] = si_sdr(ref_audio, deg_audio)

    return metrics


# Exemplo de uso (apenas para teste direto do arquivo)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando teste de AudioQM.py...")

    # Crie um arquivo de áudio dummy para teste
    # Você pode substituir por um caminho para um arquivo .wav ou .mp3 real
    dummy_audio_path = "dummy_audio.wav"
    try:
        from scipy.io.wavfile import write

        samplerate = 44100
        duration = 1.0  # seconds
        frequency = 440  # Hz
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        write(dummy_audio_path, samplerate, data.astype(np.float32))
        logger.info(f"Arquivo de áudio dummy criado em {dummy_audio_path}")

        # Teste as métricas
        metrics = get_audio_metrics(dummy_audio_path)
        print("\nMétricas de Áudio para o arquivo dummy:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.2f}")
            else:
                print(f"- {key}: {value}")

    except ImportError:
        logger.error("scipy não está instalado. Não é possível criar arquivo de áudio dummy para teste.")
        logger.info(
            "Por favor, instale scipy (`pip install scipy`) ou forneça um arquivo de áudio existente para testar.")
    except Exception as e:
        logger.error(f"Erro durante o teste de AudioQM: {e}")
    finally:
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
            logger.info(f"Arquivo dummy removido: {dummy_audio_path}")