# Predictor.py
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
import pickle
from sklearn.preprocessing import LabelEncoder

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Importa as camadas customizadas do TrainModel
# É crucial que essas camadas sejam definidas ou importadas aqui para que o Keras possa carregá-las
from TrainModel import AudioFeatureNormalization, AttentionLayer  # Garante que as custom layers sejam carregadas

# Configura logger para Predictor
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [Predictor] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelPredictor:
    """
    Classe para carregar um modelo treinado e realizar predições de detecção de deepfake.
    """

    def __init__(self, model_path: Union[str, Path]):
        """
        Inicializa o ModelPredictor.

        Args:
            model_path (Union[str, Path]): Caminho para o diretório onde o modelo .h5 e o encoder .pkl estão salvos.
        """
        self.model_dir = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model_loaded = False

        # Dicionário para camadas customizadas. É crucial para o Keras carregar modelos com elas.
        self.custom_objects = {
            'AudioFeatureNormalization': AudioFeatureNormalization,
            'AttentionLayer': AttentionLayer
        }

        # Estas serão extraídas do input_shape do modelo carregado
        self.expected_frames: Optional[int] = None
        self.feature_dim: Optional[int] = None
        self.model_input_rank: Optional[int] = None  # Rank (number of dimensions) expected by the model

        self.load_model()  # Tenta carregar o modelo e o encoder na inicialização

    def load_model(self):
        """
        Carrega o modelo Keras mais recente e o LabelEncoder do diretório especificado.
        Extrai expected_frames e feature_dim do input_shape do modelo.
        """
        # Limpa o estado atual do modelo e encoder
        self.model = None
        self.label_encoder = None
        self.expected_frames = None
        self.feature_dim = None
        self.model_input_rank = None
        self.model_loaded = False

        # Procura pelo arquivo .h5 mais recente
        model_files = sorted(self.model_dir.glob('*.h5'), key=os.path.getmtime, reverse=True)
        if not model_files:
            logger.warning(f"Nenhum arquivo de modelo (.h5) encontrado em {self.model_dir}")
            return

        latest_model_path = model_files[0]
        encoder_path = self.model_dir / "label_encoder.pkl"

        try:
            # Carrega o modelo Keras
            self.model = load_model(latest_model_path, custom_objects=self.custom_objects)
            self.model_loaded = True
            logger.info(
                f"Modelo '{latest_model_path.name}' carregado com sucesso. Input shape: {self.model.input_shape}")

            # Extrair expected_frames e feature_dim do input_shape do modelo
            # model.input_shape é (None, dim1, dim2, ...) onde None é o batch size
            input_shape_no_batch = self.model.input_shape[1:]
            self.model_input_rank = len(input_shape_no_batch)

            if self.model_input_rank == 2:  # (frames, feature_dim)
                self.expected_frames = input_shape_no_batch[0]
                self.feature_dim = input_shape_no_batch[1]
            elif self.model_input_rank == 3:  # (frames, feature_dim, channels) - commonly (frames, feature_dim, 1)
                self.expected_frames = input_shape_no_batch[0]
                self.feature_dim = input_shape_no_batch[1]
                # Note: The last dimension (channels) is handled by reshape if needed.
            else:
                logger.warning(f"Input shape do modelo ({self.model.input_shape}) inesperado. "
                               f"Não foi possível extrair expected_frames e feature_dim automaticamente.")
                # Fallback to manual setting or error handling if shape is crucial

            logger.info(
                f"Dimensões do modelo extraídas: Expected Frames={self.expected_frames}, Feature Dim={self.feature_dim}")

            # Carrega o LabelEncoder
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"LabelEncoder carregado com sucesso. Classes: {self.label_encoder.classes_}")
            else:
                logger.warning(f"LabelEncoder não encontrado em {encoder_path}. "
                               f"As predições decodificadas podem ser imprecisas. Certifique-se de que 'FAKE' e 'REAL' estejam entre as classes 0 e 1, respectivamente.")

        except Exception as e:
            logger.error(f"Erro ao carregar o modelo ou LabelEncoder de {self.model_dir}: {e}")
            self.model_loaded = False  # Garante que o estado seja false em caso de falha

    def set_label_encoder(self, encoder: Optional[LabelEncoder]):
        """Define o LabelEncoder a ser usado pelo preditor. Usado principalmente para testes ou injeção."""
        self.label_encoder = encoder
        if self.label_encoder:
            logger.info(f"LabelEncoder definido. Classes: {self.label_encoder.classes_}")
        else:
            logger.info("LabelEncoder resetado (nenhum encoder definido).")

    def _prepare_features_for_prediction(self, features: np.ndarray) -> np.ndarray:
        """
        Prepara as features para o formato esperado pelo modelo (padding/truncating e adição de dimensão de canal/batch).
        Adapta-se a modelos que esperam entrada 2D (sequência) ou 3D (imagem/espectrograma com canal).

        Args:
            features (np.ndarray): Array de features. Pode ser 2D (frames, feature_dim)
                                   ou 3D (frames, feature_dim, 1), ou 4D (batch, frames, feature_dim, 1).
                                   A dimensão do batch será adicionada ou ajustada no final.

        Returns:
            np.ndarray: Features prontas para predição, com dimensão de lote.
        """
        if not self.model_loaded or self.model is None or self.expected_frames is None or self.feature_dim is None:
            raise RuntimeError(
                "Modelo não carregado ou dimensões de entrada não definidas. Não é possível preparar features.")

        # Remove a dimensão do batch se já estiver presente (útil para sliding_window)
        if features.ndim == self.model_input_rank + 1:
            if features.shape[0] != 1:
                logger.warning(f"Features de entrada com batch_size > 1 para _prepare_features_for_prediction. "
                               f"Processando apenas a primeira amostra do batch. Shape: {features.shape}")
            features = features[0]  # Pega a primeira amostra do batch

        # Ajusta para o rank esperado pelo modelo (2D ou 3D sem batch)
        if self.model_input_rank == 2:  # Modelo espera (frames, feature_dim)
            if features.ndim == 3 and features.shape[-1] == 1:
                features = features.squeeze(axis=-1)  # Remove dimensão de canal: (frames, feature_dim)
            elif features.ndim != 2:
                raise ValueError(
                    f"Features de entrada com shape {features.shape} não compatível com o input 2D do modelo.")
        elif self.model_input_rank == 3:  # Modelo espera (frames, feature_dim, channels)
            if features.ndim == 2:
                features = features[..., np.newaxis]  # Adiciona dimensão de canal: (frames, feature_dim, 1)
            elif features.ndim == 3 and features.shape[-1] != self.model.input_shape[
                -1]:  # Se o número de canais não for 1
                logger.warning(
                    f"Features de entrada 3D com {features.shape[-1]} canais para modelo que espera {self.model.input_shape[-1]} canais. "
                    f"Ajustando para o número de canais esperado do modelo.")
                # Isso é um ponto que pode precisar de lógica mais complexa se o modelo usar mais de 1 canal.
                # Por simplicidade, se o modelo espera N canais, e a entrada tem mais, ela é truncada para N.
                features = features[..., :self.model.input_shape[-1]]
            elif features.ndim != 3:
                raise ValueError(
                    f"Features de entrada com shape {features.shape} não compatível com o input 3D do modelo.")
        else:
            raise ValueError(
                f"Rank de input do modelo ({self.model_input_rank}) não suportado para preparação de features.")

        # Pad ou truncate para expected_frames (sempre necessário se o modelo tiver um tamanho fixo)
        num_frames = features.shape[0]
        if num_frames < self.expected_frames:
            padding_needed = self.expected_frames - num_frames
            # Cria a tupla de padding dinamicamente com base nas dimensões da feature
            padding_tuple = ((0, padding_needed),) + ((0, 0),) * (features.ndim - 1)
            features = np.pad(features, padding_tuple, mode='constant')
            logger.debug(f"Padding aplicado. Novo shape: {features.shape}")
        elif num_frames > self.expected_frames:
            features = features[:self.expected_frames, ...]
            logger.debug(f"Truncamento aplicado. Novo shape: {features.shape}")

        # Adiciona a dimensão de lote (batch size de 1)
        features = np.expand_dims(features, axis=0)
        return features

    def predict_single_audio(self, features: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Realiza a predição para um único áudio.

        Args:
            features (np.ndarray): Features de áudio (frames, feature_dim) ou (frames, feature_dim, 1).
                                   Será preparado internamente.

        Returns:
            Tuple[str, float, np.ndarray]: A label prevista ('REAL' ou 'FAKE'), a confiança
                                          e o array de probabilidades de todas as classes.
        """
        if not self.model_loaded or self.model is None:
            logger.error("Modelo não carregado para predição.")
            return "Erro: Modelo não carregado", 0.0, np.array([])

        try:
            prepared_features = self._prepare_features_for_prediction(features)
        except Exception as e:
            logger.error(f"Erro ao preparar features para predição única: {e}")
            return "Erro: Preparação de features", 0.0, np.array([])

        predictions = self.model.predict(prepared_features, verbose=0)  # verbose=0 para não poluir logs
        raw_probabilities = predictions[0]  # Pega o array de probabilidades para a única amostra

        predicted_class_index = np.argmax(raw_probabilities)
        confidence = float(raw_probabilities[predicted_class_index])

        predicted_label: str
        if self.label_encoder:
            try:
                predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
            except ValueError:
                logger.warning(
                    f"Índice de classe {predicted_class_index} fora do range do LabelEncoder. Retornando índice como string.")
                predicted_label = str(predicted_class_index)
            except Exception as e:
                logger.warning(f"Erro ao decodificar label com LabelEncoder: {e}. Retornando índice como string.")
                predicted_label = str(predicted_class_index)
        else:
            # Fallback se não houver LabelEncoder
            if raw_probabilities.shape[0] == 2:  # Se for binário (0 ou 1)
                predicted_label = "FAKE" if predicted_class_index == 0 else "REAL"  # Assumindo mapeamento 0=FAKE, 1=REAL
                logger.warning("LabelEncoder não carregado. Inferindo labels FAKE/REAL com base nos índices 0/1.")
            else:
                predicted_label = str(predicted_class_index)
                logger.warning(f"LabelEncoder não carregado. Retornando rótulo numérico: {predicted_label}.")

        logger.info(f"Predição: {predicted_label.upper()} (Confiança: {confidence:.4f})")
        return predicted_label.upper(), confidence, raw_probabilities

    def predict_batch(self, list_of_features: List[np.ndarray]) -> List[Tuple[str, float, np.ndarray]]:
        """
        Realiza a predição para um lote de áudios.

        Args:
            list_of_features (List[np.ndarray]): Lista de arrays de features (frames, feature_dim) ou (frames, feature_dim, 1).

        Returns:
            List[Tuple[str, float, np.ndarray]]: Lista de tuplas (label, confiança, array de probabilidades) para cada áudio.
        """
        if not self.model_loaded or self.model is None:
            logger.error("Modelo não carregado para predição em lote.")
            return []

        if not list_of_features:
            return []

        try:
            # Prepara todas as features no lote
            prepared_batch = np.concatenate([
                self._prepare_features_for_prediction(f) for f in list_of_features
            ], axis=0)
        except Exception as e:
            logger.error(f"Erro ao preparar features para predição em lote: {e}")
            return []

        predictions = self.model.predict(prepared_batch, verbose=0)

        results: List[Tuple[str, float, np.ndarray]] = []
        for raw_probabilities in predictions:  # Cada 'pred' aqui é um array de probabilidades para uma amostra
            predicted_class_index = np.argmax(raw_probabilities)
            confidence = float(raw_probabilities[predicted_class_index])

            predicted_label: str
            if self.label_encoder:
                try:
                    predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
                except ValueError:
                    logger.warning(
                        f"Índice de classe {predicted_class_index} fora do range do LabelEncoder. Retornando índice como string.")
                    predicted_label = str(predicted_class_index)
                except Exception as e:
                    logger.warning(f"Erro ao decodificar label com LabelEncoder: {e}. Retornando índice como string.")
                    predicted_label = str(predicted_class_index)
            else:
                if raw_probabilities.shape[0] == 2:
                    predicted_label = "FAKE" if predicted_class_index == 0 else "REAL"
                else:
                    predicted_label = str(predicted_class_index)

            results.append((predicted_label.upper(), confidence, raw_probabilities))

        logger.info(f"Predição em lote concluída para {len(results)} amostras.")
        return results


# --- Função de Predição com Janela Deslizante (externa à classe para flexibilidade) ---

def sliding_window_predict(
        predictor: ModelPredictor,
        full_audio_features: np.ndarray,  # (frames, feature_dim) ou (frames, feature_dim, 1)
        window_frames: int,
        hop_frames: int,
        threshold: float = 0.5,
        target_fake_label: str = 'FAKE'  # Nova feature para definir a label 'fake'
) -> Tuple[bool, float, float, Dict[str, float]]:  # Retorna is_fake, avg_prob_fake, max_prob_fake, all_avg_probs
    """
    Realiza a predição em um áudio longo usando uma janela deslizante.

    Args:
        predictor (ModelPredictor): Instância do ModelPredictor com o modelo carregado.
        full_audio_features (np.ndarray): Features de áudio completo. Pode ser 2D (frames, feature_dim)
                                           ou 3D (frames, feature_dim, 1).
        window_frames (int): Número de frames por janela.
        hop_frames (int): Número de frames para "pular" entre as janelas.
        threshold (float): Limiar de probabilidade para classificar como "FAKE".
        target_fake_label (str): A string exata da label que corresponde à classe "FAKE" no LabelEncoder (ex: 'FAKE', 'fake').

    Returns:
        Tuple[bool, float, float, Dict[str, float]]:
            - True se o áudio é classificado como FAKE, False se REAL.
            - A probabilidade média da classe "FAKE" nas janelas.
            - A probabilidade máxima da classe "FAKE" em qualquer janela.
            - Um dicionário com as probabilidades médias de todas as classes.
    """
    if not predictor.model_loaded or predictor.model is None:
        logger.error("Modelo não carregado para predição com janela deslizante.")
        return False, 0.0, 0.0, {}

    all_window_raw_predictions: List[np.ndarray] = []  # Para armazenar os arrays de probabilidades de cada janela
    num_total_frames = full_audio_features.shape[0]

    # Verifica se window_frames é compatível com o expected_frames do modelo
    if predictor.expected_frames is None:
        logger.error("Expected frames do modelo não definido. Não é possível realizar sliding window.")
        return False, 0.0, 0.0, {}

    if window_frames != predictor.expected_frames:
        logger.warning(
            f"window_frames ({window_frames}) é diferente de predictor.expected_frames ({predictor.expected_frames}). "
            f"As janelas serão ajustadas pelo _prepare_features_for_prediction, mas isso pode indicar um uso incorreto ou sub-ótimo.")

    start_frame = 0
    while start_frame + window_frames <= num_total_frames:
        end_frame = start_frame + window_frames
        window_features = full_audio_features[start_frame:end_frame, ...]

        try:
            # Chama predict_single_audio para obter as probabilidades cruas
            _, _, raw_probs = predictor.predict_single_audio(window_features)
            if raw_probs.size > 0:
                all_window_raw_predictions.append(raw_probs)
        except Exception as e:
            logger.error(f"Erro ao processar janela de áudio de {start_frame} a {end_frame}: {e}")
            # Pular a janela com erro
            pass

        start_frame += hop_frames

    if not all_window_raw_predictions:
        logger.warning("Nenhuma janela processada para predição deslizante. Áudio pode ser muito curto ou houve erros.")
        return False, 0.0, 0.0, {}

    all_window_raw_predictions_np = np.array(all_window_raw_predictions)  # Shape: (num_windows, num_classes)

    # Calcular probabilidades médias para todas as classes
    avg_probs_per_class: Dict[str, float] = {}
    if predictor.label_encoder:
        for i, class_name in enumerate(predictor.label_encoder.classes_):
            if all_window_raw_predictions_np.shape[1] > i:
                avg_probs_per_class[str(class_name).upper()] = np.mean(all_window_raw_predictions_np[:, i])
            else:
                logger.warning(
                    f"Índice de classe {i} fora do range das predições. Classes disponíveis: {all_window_raw_predictions_np.shape[1]}")
    else:
        # Fallback se LabelEncoder não estiver disponível, usa índices numéricos
        for i in range(all_window_raw_predictions_np.shape[1]):
            avg_probs_per_class[f"CLASS_{i}"] = np.mean(all_window_raw_predictions_np[:, i])
        logger.warning("LabelEncoder não carregado. Probabilidades médias usando índices numéricos das classes.")

    # Determinar a probabilidade média e máxima para a classe FAKE
    fake_class_index: Optional[int] = None
    if predictor.label_encoder:
        try:
            # Garante que 'target_fake_label' é exatamente como foi ajustado no LabelEncoder
            if target_fake_label in predictor.label_encoder.classes_:
                fake_class_index = predictor.label_encoder.transform([target_fake_label])[0]
            elif target_fake_label.lower() in [c.lower() for c in predictor.label_encoder.classes_]:
                # Tenta encontrar correspondência case-insensitive se não houver exata
                idx = [c.lower() for c in predictor.label_encoder.classes_].index(target_fake_label.lower())
                fake_class_index = predictor.label_encoder.transform([predictor.label_encoder.classes_[idx]])[0]
            else:
                logger.warning(
                    f"Classe '{target_fake_label}' não encontrada no LabelEncoder. Tentando inferir índice 0.")
                fake_class_index = 0  # Default para 0, comum para 'FAKE'
        except Exception as e:
            logger.warning(
                f"Erro ao obter índice de '{target_fake_label}' do LabelEncoder: {e}. Tentando inferir índice 0.")
            fake_class_index = 0  # Fallback padrão
    else:
        # Se LabelEncoder não está carregado, assuma a classe 0 como FAKE para binário
        if all_window_raw_predictions_np.shape[1] == 2:
            fake_class_index = 0  # 'FAKE' é geralmente 0 em sistemas binários se não houver encoder
        else:
            logger.error(
                "Não foi possível determinar o índice da classe 'FAKE' sem LabelEncoder ou com múltiplas classes.")
            return False, 0.0, 0.0, avg_probs_per_class

    avg_prob_fake = 0.0
    max_prob_fake = 0.0
    is_fake_overall = False

    if fake_class_index is not None and all_window_raw_predictions_np.shape[1] > fake_class_index:
        avg_prob_fake = np.mean(all_window_raw_predictions_np[:, fake_class_index])
        max_prob_fake = np.max(all_window_raw_predictions_np[:, fake_class_index])
        is_fake_overall = avg_prob_fake >= threshold  # Usando a média para a decisão final
        logger.info(
            f"Probabilidade FAKE (Média nas janelas): {avg_prob_fake:.4f}, (Máxima na janela): {max_prob_fake:.4f}")
    else:
        logger.error(
            f"Índice da classe 'FAKE' ({fake_class_index}) inválido ou fora do range de predições. {all_window_raw_predictions_np.shape[1]} classes.")

    return is_fake_overall, avg_prob_fake, max_prob_fake, avg_probs_per_class


# ============================ TESTES (somente para desenvolvimento) ============================
if __name__ == "__main__":
    # Configurações dummy para teste
    DUMMY_MODEL_DIR = Path("dummy_model_artifacts_predictor")
    DUMMY_MODEL_DIR.mkdir(exist_ok=True)
    DUMMY_MODEL_PATH = DUMMY_MODEL_DIR / "dummy_detector_model.h5"
    DUMMY_ENCODER_PATH = DUMMY_MODEL_DIR / "label_encoder.pkl"
    # Essas dimensões virão do modelo carregado agora
    DUMMY_EXPECTED_FRAMES = 100
    DUMMY_FEATURE_DIM = 40
    DUMMY_NUM_CLASSES = 2  # Ex: 'fake', 'real'

    logger.info("\n--- Iniciando testes do Predictor.py ---")

    # 1. Cria um modelo dummy para teste que simula o output do TrainModel
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
            Reshape, GRU, Bidirectional
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras import layers  # Importa layers para o Transformer
        from TrainModel import AudioFeatureNormalization, AttentionLayer


        # Função auxiliar para criar um modelo compatível com diferentes architectures
        def create_dummy_model_for_predictor_test(input_shape_for_model, num_classes, architecture_type="cnn_gru"):
            inputs = Input(shape=input_shape_for_model)
            x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(inputs)

            if architecture_type == "cnn_gru":
                # Ensure 4D for CNN
                if len(input_shape_for_model) == 2:
                    x = Reshape((input_shape_for_model[0], input_shape_for_model[1], 1), name="reshape_cnn_test")(x)
                elif len(input_shape_for_model) == 3 and input_shape_for_model[-1] != 1:
                    x = x[..., :1]  # Truncate to 1 channel for simplicity

                x = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_test")(x)
                x = BatchNormalization(name="bn1_test")(x)
                x = MaxPooling2D((2, 2), name="pool1_test")(x)
                x = Dropout(0.2, name="dropout1_test")(x)

                x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2_test")(x)
                x = BatchNormalization(name="bn2_test")(x)
                x = MaxPooling2D((2, 2), name="pool2_test")(x)
                x = Dropout(0.2, name="dropout2_test")(x)

                shape_before_gru = x.shape
                x = Reshape((shape_before_gru[1], shape_before_gru[2] * shape_before_gru[3]),
                            name="reshape_for_gru_test")(x)

                x = GRU(64, return_sequences=True, name="gru_test")(x)
                x = AttentionLayer(name="attention_layer_test")(x)

            elif architecture_type == "bidirectional_gru":
                # Ensure 3D for Bi-GRU (remove channel dim if present)
                if len(input_shape_for_model) == 3 and input_shape_for_model[-1] == 1:
                    x = Reshape((input_shape_for_model[0], input_shape_for_model[1]), name="reshape_bi_gru_test")(x)
                elif len(input_shape_for_model) == 2:
                    pass  # Already 2D

                x = Bidirectional(GRU(64, return_sequences=True), name="bi_gru_test")(x)
                x = AttentionLayer(name="attention_layer_test")(x)

            elif architecture_type == "transformer":
                # Assume input_shape_for_model is already (frames, feature_dim) for transformer
                # If it's (frames, feature_dim, 1), remove the channel dim
                if len(input_shape_for_model) == 3 and input_shape_for_model[-1] == 1:
                    x = Reshape((input_shape_for_model[0], input_shape_for_model[1]), name="reshape_transformer_test")(
                        x)
                elif len(input_shape_for_model) == 2:
                    pass

                # Positional Encoding (simplificado)
                seq_len_model = input_shape_for_model[0]
                feature_dim_model = input_shape_for_model[1]
                pos_encoding = layers.Embedding(seq_len_model, feature_dim_model)(tf.range(seq_len_model))
                x = x + pos_encoding

                # Transformer Encoder Block
                num_heads = 4
                ff_dim = 64
                attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim_model)(x, x)
                attn_output = layers.Dropout(0.2)(attn_output)
                x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

                ff_output = layers.Dense(ff_dim, activation="relu")(x)
                ff_output = layers.Dense(feature_dim_model)(ff_output)
                ff_output = layers.Dropout(0.2)(ff_output)
                x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

                x = layers.GlobalAveragePooling1D(name="transformer_avg_pool_test")(x)

            else:
                raise ValueError("Arquitetura de teste não reconhecida.")

            x = Dense(64, activation='relu', name="dense1_test")(x)
            x = BatchNormalization(name="bn_dense_test")(x)
            x = Dropout(0.3, name="dropout_final_test")(x)
            outputs = Dense(num_classes, activation='softmax', name="output_layer_test")(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            return model


        # Testar com diferentes arquiteturas
        test_architectures = {
            "cnn_gru": (DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1),  # 4D input for CNN
            "bidirectional_gru": (DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1),
            # 3D input for Bi-GRU (channel will be squeezed)
            "transformer": (DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM)  # 2D input for Transformer
        }

        for arch_name, arch_input_shape in test_architectures.items():
            print(f"\n--- Criando e testando modelo dummy para arquitetura: {arch_name} ---")

            # Limpa o diretório de modelo dummy antes de cada teste de arquitetura
            if DUMMY_MODEL_DIR.exists():
                shutil.rmtree(DUMMY_MODEL_DIR)
            DUMMY_MODEL_DIR.mkdir(exist_ok=True)  # Recria

            dummy_model = create_dummy_model_for_predictor_test(arch_input_shape, DUMMY_NUM_CLASSES, arch_name)
            logger.info(f"Resumo do Modelo Dummy para {arch_name}:")
            dummy_model.summary(print_fn=lambda x: logger.info(x))

            # Adaptação da AudioFeatureNormalization para o modelo dummy
            norm_layer_in_dummy_model = None
            for layer in dummy_model.layers:
                if isinstance(layer, AudioFeatureNormalization):
                    norm_layer_in_dummy_model = layer
                    break
            if norm_layer_in_dummy_model:
                # Criar dados dummy para adaptação com o shape esperado pelo layer
                # O input_shape_for_model do create_dummy_model já inclui o canal se for 3D
                if len(arch_input_shape) == 3:  # (frames, features, 1)
                    dummy_adapt_data = np.random.rand(50, arch_input_shape[0], arch_input_shape[1],
                                                      arch_input_shape[2]).astype(np.float32)
                else:  # (frames, features)
                    dummy_adapt_data = np.random.rand(50, arch_input_shape[0], arch_input_shape[1]).astype(np.float32)
                norm_layer_in_dummy_model.adapt(dummy_adapt_data)
                logger.info("Camada AudioFeatureNormalization do modelo dummy adaptada.")
            else:
                logger.warning(
                    "Camada AudioFeatureNormalization não encontrada no modelo dummy. Pode causar problemas de carregamento.")

            dummy_model.save(DUMMY_MODEL_PATH)
            logger.info(f"Modelo dummy para {arch_name} salvo em: {DUMMY_MODEL_PATH}")

            dummy_encoder = LabelEncoder()
            dummy_encoder.fit(['fake', 'real'])
            with open(DUMMY_ENCODER_PATH, 'wb') as f:
                pickle.dump(dummy_encoder, f)
            logger.info(f"LabelEncoder dummy salvo em: {DUMMY_ENCODER_PATH}")

            # Agora, testa o ModelPredictor com este modelo dummy
            predictor_test = ModelPredictor(model_path=DUMMY_MODEL_DIR)

            if predictor_test.model_loaded:
                logger.info(
                    f"Predictor carregado com sucesso para {arch_name}. Extraído: frames={predictor_test.expected_frames}, dim={predictor_test.feature_dim}")

                # Gerar features dummy para predição, sempre em 3D para ser flexível (o _prepare_features_for_prediction vai ajustar)
                single_features = np.random.rand(DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1).astype(np.float32)
                long_features = np.random.rand(DUMMY_EXPECTED_FRAMES * 2, DUMMY_FEATURE_DIM, 1).astype(np.float32)

                print(f"--- Testando predição única para {arch_name} ---")
                try:
                    label, confidence, probs = predictor_test.predict_single_audio(single_features)
                    print(f"Resultado Predição Única: {label} (Confiança: {confidence:.4f}) Probabilidades: {probs}")
                except Exception as e:
                    logger.exception(f"Erro durante a predição única para {arch_name}.")

                print(f"--- Testando predição com janela deslizante para {arch_name} ---")
                try:
                    is_fake, avg_prob, max_prob, all_avg_probs = sliding_window_predict(
                        predictor_test,
                        long_features,
                        window_frames=DUMMY_EXPECTED_FRAMES,
                        hop_frames=DUMMY_EXPECTED_FRAMES // 2,
                        threshold=0.5
                    )
                    print(f"Resultado Janela Deslizante: {'FAKE' if is_fake else 'REAL'} "
                          f"(Média Prob Fake: {avg_prob:.4f}, Máxima Prob Fake: {max_prob:.4f}) "
                          f"Médias de todas as classes: {all_avg_probs}")
                except Exception as e:
                    logger.exception(f"Erro durante a predição com janela deslizante para {arch_name}.")
            else:
                logger.error(f"Falha ao carregar o modelo para teste da arquitetura {arch_name}.")

    except Exception as e:
        logger.error(f"Erro fatal durante a criação ou teste de modelos dummy: {e}")
    finally:
        # Limpa o diretório dummy após todos os testes
        if DUMMY_MODEL_DIR.exists():
            logger.info(f"\nRemovendo diretório dummy: {DUMMY_MODEL_DIR}")
            shutil.rmtree(DUMMY_MODEL_DIR)