import os
import warnings
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import librosa
from librosa.util.exceptions import ParameterError
from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Pipeline modular de pré-processamento de áudio para extração de features.

    Suporta:
      - Múltiplos formatos (wav, mp3, flac, ogg, wma) usando pydub.
      - Normalização de amplitude.
      - Remoção de silêncio por energia (VAD).
      - Extração de Log-Mel Spectrograms ou MFCCs.
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            frame_length_ms: float = 25.0,
            frame_shift_ms: float = 10.0,
            n_mfcc: int = 13,
            n_mels: int = 40,
            n_fft: Optional[int] = None,
            vad_energy_thresh: float = -40,  # dBFS, mais intuitivo para pydub
            min_segment_duration: float = 0.5,  # Mínimo em segundos
            verbose: bool = False
    ):
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        # n_fft padrão se não for fornecido, baseado na prática comum
        self.n_fft = n_fft if n_fft is not None else int(self.sample_rate * self.frame_length_ms / 1000)
        # Garante que n_fft é uma potência de 2 ou um valor adequado para FFT
        if not (self.n_fft & (self.n_fft - 1) == 0) and self.n_fft < 2048:  # Verifica se é potência de 2
            # Se não for potência de 2, arredonda para a próxima potência de 2
            self.n_fft = int(2 ** np.ceil(np.log2(self.sample_rate * self.frame_length_ms / 1000)))
            logger.warning(f"N_FFT ajustado para a próxima potência de 2: {self.n_fft}")

        self.hop_length = int(self.sample_rate * self.frame_shift_ms / 1000)
        self.vad_energy_thresh = vad_energy_thresh
        self.min_segment_duration = min_segment_duration
        self.verbose = verbose

    def _load_audio(self, audio_path: str) -> Optional[AudioSegment]:
        """Carrega um arquivo de áudio usando pydub."""
        try:
            audio = AudioSegment.from_file(audio_path)
            # Converte para mono se não for
            if audio.channels > 1:
                audio = audio.set_channels(1)
            # Converte para a sample_rate desejada
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            return audio
        except CouldntDecodeError:
            logger.error(
                f"Não foi possível decodificar o arquivo de áudio: {audio_path}. Formato não suportado ou corrompido.")
            return None
        except Exception as e:
            logger.error(f"Erro ao carregar o arquivo de áudio {audio_path}: {e}")
            return None

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normaliza a amplitude do áudio para -20 dBFS."""
        # Se o áudio estiver muito silencioso, normalize para um nível razoável
        if audio.dBFS > -20.0:
            normalized_audio = audio.normalize()  # Normaliza para 0 dBFS e depois para -20
            # Ou, se preferir uma normalização mais controlada, use `apply_gain`
            # `gain = -20 - audio.dBFS` # Calcule o ganho necessário para atingir -20 dBFS
            # normalized_audio = audio.apply_gain(gain)
        else:
            normalized_audio = audio.set_frame_rate(self.sample_rate)
        return normalized_audio

    def _remove_silence(self, audio: AudioSegment) -> AudioSegment:
        """
        Remove segmentos de silêncio do áudio usando pydub.silence.split_on_silence.
        """
        min_silence_len = 100  # ms, pode ajustar
        # Aumentar o threshold aqui para ser mais agressivo na remoção de silêncio.
        # noise_reduction=True pode ajudar, mas é mais complexo.

        # pydub.silence.split_on_silence retorna uma lista de AudioSegmentos.
        # Ajuste o 'silence_thresh' para ser o vad_energy_thresh que está em dBFS.
        # 'keep_silence' pode ser 0 para remover completamente o silêncio entre os segmentos.

        audio_chunks = silence.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=self.vad_energy_thresh,
            keep_silence=0  # Remove todo o silêncio entre os chunks
        )

        if not audio_chunks:
            if self.verbose:
                logger.warning("Nenhum segmento de não-silêncio encontrado. Retornando áudio vazio.")
            return AudioSegment.empty()

        # Concatena os segmentos de volta em um único AudioSegment
        combined_audio = AudioSegment.empty()
        for chunk in audio_chunks:
            # Filtra chunks muito curtos que podem ser ruído ou irrelevantes
            if len(chunk) / 1000.0 >= self.min_segment_duration:
                combined_audio += chunk

        if combined_audio.duration_seconds < self.min_segment_duration:
            if self.verbose:
                logger.warning(
                    f"Áudio combinado muito curto após remoção de silêncio ({combined_audio.duration_seconds:.2f}s). Retornando áudio vazio.")
            return AudioSegment.empty()

        return combined_audio

    def _audio_segment_to_numpy(self, audio_segment: AudioSegment) -> np.ndarray:
        """Converte um AudioSegment para um array NumPy."""
        # Converte para array de amostras float (librosa espera float)
        # pydub retorna amostras como inteiros; precisamos normalizá-las para float.
        # Usa np.float32 para compatibilidade com TensorFlow.
        audio_np = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        # Normaliza as amostras para o intervalo [-1.0, 1.0]
        # Depende da profundidade de bits do áudio (ex: 16-bit, 24-bit)
        max_val = 2 ** (audio_segment.sample_width * 8 - 1)
        audio_np /= max_val
        return audio_np

    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extrai MFCCs de um array NumPy de áudio."""
        # y: áudio (array numpy)
        # sr: sample rate
        # n_mfcc: número de coeficientes (ex: 13, 40)
        # n_fft: tamanho da janela FFT (normalmente 2048)
        # hop_length: número de amostras entre quadros sucessivos (normalmente 512)

        # librosa.feature.mfcc já calcula o MFCC a partir do espectrograma de Mel.
        # É comum usar n_mels na criação do Mel Spectrogram antes do MFCC.

        # 1. Compute Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # 2. Convert to log scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 3. Compute MFCCs from log-Mel Spectrogram
        mfccs = librosa.feature.mfcc(
            S=log_mel_spectrogram,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            dct_type=2  # Tipo de DCT discreta
        )

        # Transpose para ter (frames, n_mfcc)
        return mfccs.T

    def _extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extrai Log-Mel Spectrograms de um array NumPy de áudio."""
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram.T  # Transpose para ter (frames, n_mels)

    def _pad_or_truncate(self, features: np.ndarray, expected_frames: int) -> np.ndarray:
        """
        Pad ou trunca as features para ter um número fixo de frames.
        Espera features com shape (frames, feature_dim).
        """
        current_frames = features.shape[0]
        feature_dim = features.shape[1]

        if current_frames == expected_frames:
            return features
        elif current_frames < expected_frames:
            # Pad com zeros no final
            padding_needed = expected_frames - current_frames
            # Garante que o padding_shape é (padding_needed, feature_dim)
            padded_features = np.pad(features, ((0, padding_needed), (0, 0)), mode='constant')
            return padded_features
        else:  # current_frames > expected_frames
            # Truncar aleatoriamente ou do início/fim. Aqui, truncando do início.
            # Pode-se implementar um truncamento aleatório para mais variabilidade
            # start_frame = np.random.randint(0, current_frames - expected_frames + 1)
            # return features[start_frame:start_frame + expected_frames, :]
            return features[:expected_frames, :]

    def extract_features(self, audio_path: str, feature_type: str = "mfcc", expected_frames: Optional[int] = None) -> \
    Optional[np.ndarray]:
        """
        Extrai features de áudio completas com pré-processamento.

        Args:
            audio_path (str): Caminho para o arquivo de áudio.
            feature_type (str): Tipo de feature a extrair ('mfcc' ou 'mel_spectrogram').
            expected_frames (Optional[int]): Se especificado, pad ou trunca as features
                                             para este número de frames. Caso contrário,
                                             retorna o número de frames natural do áudio.
        Returns:
            Optional[np.ndarray]: Array NumPy das features com shape (frames, feature_dim).
                                  Retorna None se houver erro ou áudio muito curto.
        """
        audio_segment = self._load_audio(audio_path)
        if audio_segment is None:
            return None

        if audio_segment.duration_seconds < self.min_segment_duration:
            if self.verbose:
                logger.warning(
                    f"Áudio '{os.path.basename(audio_path)}' muito curto ({audio_segment.duration_seconds:.2f}s). Pulando.")
            return None

        # Normaliza o áudio antes da remoção de silêncio e conversão para numpy
        normalized_audio = self._normalize_audio(audio_segment)

        # Remove silêncio
        processed_audio = self._remove_silence(normalized_audio)

        if processed_audio.duration_seconds < self.min_segment_duration:
            if self.verbose:
                logger.warning(
                    f"Áudio '{os.path.basename(audio_path)}' muito curto após remoção de silêncio ({processed_audio.duration_seconds:.2f}s). Pulando.")
            return None

        y = self._audio_segment_to_numpy(processed_audio)

        features: Optional[np.ndarray] = None
        try:
            if feature_type == "mfcc":
                features = self._extract_mfcc(y)
            elif feature_type == "mel_spectrogram":
                features = self._extract_mel_spectrogram(y)
            else:
                raise ValueError(f"Tipo de feature '{feature_type}' não suportado.")
        except ParameterError as e:
            logger.error(f"Erro de parâmetro ao extrair features de {os.path.basename(audio_path)}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado ao extrair features de {os.path.basename(audio_path)}: {e}")
            return None

        if features is None or features.shape[0] == 0:
            if self.verbose:
                logger.warning(f"Nenhuma feature extraída de {os.path.basename(audio_path)}.")
            return None

        if expected_frames is not None:
            features = self._pad_or_truncate(features, expected_frames)
            if features.shape[0] != expected_frames:
                logger.error(
                    f"Erro no padding/truncamento: features com {features.shape[0]} frames, esperado {expected_frames}.")
                return None

        return features


# Bloco de teste
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s [Voice2data] %(message)s')

    test_audio_path = "test_audio.wav"
    sample_rate = 16000
    duration_seconds = 5  # seconds

    # Gerar um arquivo de áudio WAV simples para teste
    try:
        from pydub.playback import play  # Para reproduzir se necessário

        # Cria um áudio de silêncio
        audio = AudioSegment.silent(duration=duration_seconds * 1000, frame_rate=sample_rate)
        # Adiciona um tom (ex: 440 Hz) no meio para simular "voz"
        tone_duration = 1000  # 1 segundo de tom
        tone = AudioSegment.from_tone(440, duration=tone_duration, frame_rate=sample_rate)
        audio = audio.overlay(tone, position=(duration_seconds * 1000 / 2) - (tone_duration / 2))

        audio.export(test_audio_path, format="wav")
        logger.info(f"Arquivo de teste '{test_audio_path}' criado.")
    except Exception as e:
        logger.error(f"Não foi possível criar arquivo de teste de áudio: {e}")
        logger.info("Por favor, crie manualmente um arquivo 'test_audio.wav' para executar os testes.")
        test_audio_path = None  # Impede a execução dos testes se o arquivo não puder ser criado

    if test_audio_path and os.path.exists(test_audio_path):
        preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            n_mfcc=40,
            n_mels=40,
            vad_energy_thresh=-50,  # Um pouco mais agressivo para teste
            min_segment_duration=0.1,  # Permite segmentos curtos para teste
            verbose=True
        )

        # Definir expected_frames com base nos parâmetros do preprocessor
        # (duration_seconds * sample_rate - n_fft) / hop_length + 1
        # No entanto, como VAD altera a duração, é melhor não depender de uma duração fixa para o teste
        # mas sim testar a capacidade de _pad_or_truncate.

        # Testando extração de MFCC
        expected_frames_for_test = 100  # Definir um número de frames esperado para o teste

        print(
            f"\n--- Testando extração de MFCC para {test_audio_path} (Expected Frames: {expected_frames_for_test}) ---")
        mfcc_features = preprocessor.extract_features(test_audio_path, feature_type="mfcc",
                                                      expected_frames=expected_frames_for_test)
        if mfcc_features is not None:
            print(f"Shape das features MFCC: {mfcc_features.shape}")
            print(f"Tipo de dado das features MFCC: {mfcc_features.dtype}")
            if mfcc_features.shape[0] == expected_frames_for_test:
                print("Número de frames MFCC corresponde ao esperado.")
            else:
                print(
                    f"ATENÇÃO: Número de frames MFCC ({mfcc_features.shape[0]}) não corresponde ao esperado ({expected_frames_for_test}).")
        else:
            print("Não foi possível extrair features MFCC.")

        print(
            f"\n--- Testando extração de Log-Mel Spectrogram para {test_audio_path} (Expected Frames: {expected_frames_for_test}) ---")
        mel_features = preprocessor.extract_features(test_audio_path, feature_type="mel_spectrogram",
                                                     expected_frames=expected_frames_for_test)
        if mel_features is not None:
            print(f"Shape das features Log-Mel Spectrogram: {mel_features.shape}")
            print(f"Tipo de dado das features Log-Mel Spectrogram: {mel_features.dtype}")
            if mel_features.shape[0] == expected_frames_for_test:
                print("Número de frames Log-Mel Spectrogram corresponde ao esperado.")
            else:
                print(
                    f"ATENÇÃO: Número de frames Log-Mel Spectrogram ({mel_features.shape[0]}) não corresponde ao esperado ({expected_frames_for_test}).")
        else:
            print("Não foi possível extrair features Log-Mel Spectrogram.")

    if os.path.exists(test_audio_path):
        os.remove(test_audio_path)
        logger.info(f"Arquivo de teste '{test_audio_path}' removido.")