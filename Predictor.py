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
# Certifique-se de que AudioFeatureNormalization e AttentionLayer são classes e não funções
# create_model NÃO é necessário aqui para o uso da classe
from TrainModel import AudioFeatureNormalization, AttentionLayer

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

    def __init__(self, model_path: Union[str, Path], expected_frames: int, feature_dim: int):
        """
        Inicializa o ModelPredictor.

        Args:
            model_path (Union[str, Path]): Caminho para o diretório onde o modelo .h5 e o encoder .pkl estão salvos.
            expected_frames (int): O número de frames esperado para o input do modelo.
            feature_dim (int): A dimensão das features (e.g., n_mfcc, n_mels).
        """
        self.model_dir = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.expected_frames = expected_frames
        self.feature_dim = feature_dim
        self.model_loaded = False
        self.audio_preprocessor = None  # Será injetado pelo app.py, necessário para sliding_window_predict

        # Dicionário para camadas customizadas. É crucial para o Keras carregar modelos com elas.
        self.custom_objects = {
            'AudioFeatureNormalization': AudioFeatureNormalization,
            'AttentionLayer': AttentionLayer
        }

        self.load_model()  # Tenta carregar o modelo e o encoder na inicialização

    def load_model(self):
        """
        Carrega o modelo Keras mais recente e o LabelEncoder do diretório especificado.
        """
        # Limpa o estado atual do modelo e encoder
        self.model = None
        self.label_encoder = None
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

            # Carrega o LabelEncoder
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"LabelEncoder carregado com sucesso. Classes: {self.label_encoder.classes_}")
            else:
                logger.warning(f"LabelEncoder não encontrado em {encoder_path}. As predições serão numéricas.")

        except Exception as e:
            logger.error(f"Erro ao carregar o modelo ou LabelEncoder de {self.model_dir}: {e}")
            self.model_loaded = False  # Garante que o estado seja false em caso de falha

    def set_label_encoder(self, encoder: Optional[LabelEncoder]):
        """Define o LabelEncoder a ser usado pelo preditor."""
        self.label_encoder = encoder
        if self.label_encoder:
            logger.info(f"LabelEncoder definido. Classes: {self.label_encoder.classes_}")
        else:
            logger.info("LabelEncoder resetado (nenhum encoder definido).")

    def _prepare_features_for_prediction(self, features: np.ndarray) -> np.ndarray:
        """
        Prepara as features para o formato esperado pelo modelo (padding/truncating e adição de dimensão de canal).

        Args:
            features (np.ndarray): Array de features 2D (frames, feature_dim) ou 3D (frames, feature_dim, 1).

        Returns:
            np.ndarray: Features 3D (1, expected_frames, feature_dim, 1) prontas para predição.
        """
        if features.ndim == 2:
            # Se for (frames, feature_dim), adiciona a dimensão de canal
            features = features[..., np.newaxis]  # (frames, feature_dim, 1)

        # Pad ou truncate para expected_frames
        num_frames = features.shape[0]
        if num_frames < self.expected_frames:
            padding_needed = self.expected_frames - num_frames
            features = np.pad(features, ((0, padding_needed), (0, 0), (0, 0)), mode='constant')
            logger.debug(f"Padding aplicado. Novo shape: {features.shape}")
        elif num_frames > self.expected_frames:
            features = features[:self.expected_frames, :, :]
            logger.debug(f"Truncamento aplicado. Novo shape: {features.shape}")

        # Adiciona a dimensão de lote (batch size de 1)
        # Transforma de (expected_frames, feature_dim, 1) para (1, expected_frames, feature_dim, 1)
        features = np.expand_dims(features, axis=0)
        return features

    def predict_single_audio(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Realiza a predição para um único áudio.

        Args:
            features (np.ndarray): Features de áudio 3D (frames, feature_dim, 1)
                                   ou 4D (1, frames, feature_dim, 1).
                                   Será preparado internamente.

        Returns:
            Tuple[str, float]: A label prevista ('REAL' ou 'FAKE') e a confiança.
        """
        if not self.model_loaded or self.model is None:
            logger.error("Modelo não carregado para predição.")
            return "Erro: Modelo não carregado", 0.0

        if self.label_encoder is None:
            logger.warning("LabelEncoder não carregado. Retornando rótulo numérico.")
            # Tentativa de inferir labels se classes_ for acessível e tiver 2 elementos
            if hasattr(self.model.output_shape, '__len__') and len(self.model.output_shape) > 0 and \
                    self.model.output_shape[-1] == 2:
                # Se o modelo tem 2 saídas (provavelmente 0 e 1), podemos tentar mapear
                # Assumindo que a classe 0 é 'fake' e 1 é 'real' ou vice-versa,
                # dependendo de como foi treinado. Idealmente, o LabelEncoder resolve isso.
                logger.info("Modelo com 2 classes de saída, tentando inferir rótulos (REAL/FAKE).")
                # Vamos manter o comportamento padrão de retorno de números se não houver LE
                # ou usar um mapeamento padrão se soubermos as classes, mas isso é arriscado sem LE.
                # Para robustez, é melhor ter o LabelEncoder.
                pass
            else:
                logger.warning("Modelo não tem 2 classes de saída previsíveis sem LabelEncoder. Retornando 'N/A'.")
                return "N/A", 0.0

        # Prepara as features para o input do modelo
        # A função _prepare_features_for_prediction já trata a dimensão de canal e batch
        prepared_features = self._prepare_features_for_prediction(features)

        # Realiza a predição
        predictions = self.model.predict(prepared_features)

        # Assume que o modelo retorna probabilidades para cada classe
        # Para um modelo binário (REAL/FAKE), predictions[0] será uma lista de 2 elementos [prob_fake, prob_real]
        # ou [prob_real, prob_fake] dependendo da ordem do LabelEncoder.
        # Precisamos da classe com a maior probabilidade.

        # Pega o índice da classe com a maior probabilidade
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        if self.label_encoder:
            # Decodifica o índice de volta para o nome da classe
            predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
        else:
            # Se não houver LabelEncoder, retorna o índice como string
            predicted_label = str(predicted_class_index)
            # Adiciona uma lógica básica para inferir "FAKE" ou "REAL" se classes forem 0 ou 1
            if predicted_label == "0" and len(predictions[0]) == 2:
                # A ordem típica em muitos LEs é alfabética: FAKE=0, REAL=1. Mas não é garantido.
                # Para evitar confusão, se o LE não existir, vamos retornar 'Class 0' ou 'Class 1'
                # ou usar o threshold para "FAKE" vs "REAL" se for uma saída binária de sigmoid.
                # No momento, a saída é softmax.
                pass

        logger.info(f"Predição: {predicted_label} (Confiança: {confidence:.4f})")
        return predicted_label.upper(), confidence  # Retorna em maiúsculas

    def predict_batch(self, list_of_features: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Realiza a predição para um lote de áudios.

        Args:
            list_of_features (List[np.ndarray]): Lista de arrays de features 3D (frames, feature_dim, 1).

        Returns:
            List[Tuple[str, float]]: Lista de tuplas (label, confiança) para cada áudio.
        """
        if not self.model_loaded or self.model is None:
            logger.error("Modelo não carregado para predição em lote.")
            return []

        if not list_of_features:
            return []

        # Prepara todas as features no lote
        # Cada item na lista deve ser (frames, feature_dim, 1) e _prepare_features_for_prediction
        # o transformará em (1, expected_frames, feature_dim, 1).
        # Precisamos concatená-los no eixo 0 para formar um único tensor de batch.
        prepared_batch = np.concatenate([
            self._prepare_features_for_prediction(f) for f in list_of_features
        ], axis=0)

        predictions = self.model.predict(prepared_batch)

        results: List[Tuple[str, float]] = []
        for pred in predictions:
            predicted_class_index = np.argmax(pred)
            confidence = float(pred[predicted_class_index])

            if self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
            else:
                predicted_label = str(predicted_class_index)  # Fallback se não houver LabelEncoder

            results.append((predicted_label.upper(), confidence))

        logger.info(f"Predição em lote concluída para {len(results)} amostras.")
        return results


# --- Função de Predição com Janela Deslizante (externa à classe para flexibilidade) ---

def sliding_window_predict(
        predictor: ModelPredictor,
        full_audio_features: np.ndarray,  # (frames, feature_dim, 1)
        window_frames: int,
        hop_frames: int,
        threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Realiza a predição em um áudio longo usando uma janela deslizante.

    Args:
        predictor (ModelPredictor): Instância do ModelPredictor com o modelo e preprocessor carregados.
        full_audio_features (np.ndarray): Features 3D de áudio completo (frames, feature_dim, 1).
        window_frames (int): Número de frames por janela.
        hop_frames (int): Número de frames para "pular" entre as janelas.
        threshold (float): Limiar de probabilidade para classificar como "FAKE".

    Returns:
        Tuple[bool, float]: True se o áudio é classificado como FAKE, False se REAL.
                            Retorna a probabilidade média da classe "FAKE" ou a média da classe com maior probabilidade.
    """
    if not predictor.model_loaded or predictor.model is None:
        logger.error("Modelo não carregado para predição com janela deslizante.")
        return False, 0.0

    if predictor.audio_preprocessor is None:
        logger.error("AudioPreprocessor não foi injetado no ModelPredictor. Necessário para janela deslizante.")
        return False, 0.0

    all_window_predictions: List[np.ndarray] = []  # Para armazenar as saídas de probabilidade de cada janela
    num_total_frames = full_audio_features.shape[0]
    feature_dim = full_audio_features.shape[1]

    # Iterar sobre o áudio com a janela deslizante
    start_frame = 0
    while start_frame + window_frames <= num_total_frames:
        end_frame = start_frame + window_frames
        window_features = full_audio_features[start_frame:end_frame, :, :]

        # Prepara a janela para o modelo (batch size de 1)
        # Note: _prepare_features_for_prediction já pad/trunca e adiciona dimensão de batch.
        # Aqui, a janela já deve ter o window_frames, então _pad_or_truncate não seria necessário
        # se window_frames == expected_frames do modelo.
        # Vamos assumir que window_frames é o self.expected_frames do predictor para simplicidade,
        # ou que _prepare_features_for_prediction dentro de predict_single_audio irá lidar com isso.
        # Vamos chamar predict_single_audio que já faz o pré-processamento.

        # _prepare_features_for_prediction espera (frames, feature_dim, 1)
        # e retorna (1, expected_frames, feature_dim, 1)
        prepared_window = predictor._prepare_features_for_prediction(window_features)

        # Realiza a predição na janela
        window_pred = predictor.model.predict(prepared_window)[0]  # Pega o array de probabilidades

        all_window_predictions.append(window_pred)
        start_frame += hop_frames

    if not all_window_predictions:
        logger.warning("Nenhuma janela processada para predição deslizante. Áudio pode ser muito curto.")
        return False, 0.0

    # Convertendo a lista de previsões para um array numpy para fácil manipulação
    all_window_predictions_np = np.array(all_window_predictions)  # Shape: (num_windows, num_classes)

    # Obter o índice da classe 'FAKE' e 'REAL' do LabelEncoder, se disponível
    fake_class_index: Optional[int] = None
    real_class_index: Optional[int] = None

    if predictor.label_encoder:
        try:
            # Tenta encontrar o índice da classe 'FAKE' e 'REAL'
            # Isso assume que 'fake' e 'real' são as classes esperadas.
            if 'fake' in predictor.label_encoder.classes_:
                fake_class_index = predictor.label_encoder.transform(['fake'])[0]
            if 'real' in predictor.label_encoder.classes_:
                real_class_index = predictor.label_encoder.transform(['real'])[0]

            logger.debug(f"Índices das classes: FAKE={fake_class_index}, REAL={real_class_index}")
        except Exception as e:
            logger.warning(f"Não foi possível encontrar índices de 'fake' ou 'real' no LabelEncoder: {e}")
            fake_class_index = None
            real_class_index = None

    avg_prob_fake = 0.0
    is_fake_overall = False

    if fake_class_index is not None and all_window_predictions_np.shape[1] > fake_class_index:
        # Se temos o índice 'FAKE', calculamos a probabilidade média de ser FAKE
        avg_prob_fake = np.mean(all_window_predictions_np[:, fake_class_index])
        is_fake_overall = avg_prob_fake >= threshold
        logger.info(f"Média de probabilidade FAKE da janela: {avg_prob_fake:.4f}")
    else:
        # Fallback: Se não puder determinar a classe 'FAKE' (sem LabelEncoder ou classe ausente),
        # classificamos pela classe majoritária ou pela probabilidade média da classe 1 (assumindo binário 0/1)
        # Para um modelo com duas saídas (ex: [prob_classe_0, prob_classe_1]),
        # podemos pegar a média da probabilidade da classe com o índice 1 como heurística.
        if all_window_predictions_np.shape[1] == 2:  # Se o modelo tem 2 classes de saída
            # Assumimos que a classe com maior índice é a "positiva" ou "deepfake"
            # No entanto, isso é uma suposição. O LabelEncoder é crucial.
            avg_prob_class1 = np.mean(all_window_predictions_np[:, 1])
            is_fake_overall = avg_prob_class1 >= threshold
            avg_prob_fake = avg_prob_class1  # Usamos a prob da classe 1 como proxy para "fake"
            logger.warning(
                f"LabelEncoder não disponível ou classes 'real'/'fake' não encontradas. Usando prob média da classe 1 ({avg_prob_class1:.4f}) como 'fake' para limiar.")
        else:
            # Último recurso: Apenas retorna False e 0 se não conseguir decidir
            logger.error(
                "Não foi possível determinar a classificação com janela deslizante sem LabelEncoder ou com número de classes inesperado.")
            return False, 0.0

    return is_fake_overall, avg_prob_fake


# ============================ TESTES (somente para desenvolvimento) ============================
if __name__ == "__main__":
    # Configurações dummy para teste
    DUMMY_MODEL_DIR = Path("dummy_model_artifacts")
    DUMMY_MODEL_DIR.mkdir(exist_ok=True)
    DUMMY_MODEL_PATH = DUMMY_MODEL_DIR / "dummy_detector_model.h5"
    DUMMY_ENCODER_PATH = DUMMY_MODEL_DIR / "label_encoder.pkl"
    DUMMY_EXPECTED_FRAMES = 100
    DUMMY_FEATURE_DIM = 40
    DUMMY_NUM_CLASSES = 2  # Ex: 'fake', 'real'

    logger.info("\n--- Iniciando testes do Predictor.py ---")

    # 1. Cria um modelo dummy para teste
    try:
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam


        def create_dummy_model(input_shape, num_classes):
            """Cria um modelo CNN simples para fins de teste."""
            inputs = Input(shape=input_shape)
            # Aplicar AudioFeatureNormalization como primeira camada
            x = AudioFeatureNormalization(epsilon=1e-6)(inputs)
            x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.25)(x)

            x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.25)(x)

            # Exemplo de AttentionLayer (se necessário)
            # x = AttentionLayer()(x) # Descomentar se a atenção for realmente usada na arquitetura final

            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(num_classes, activation='softmax')(x)  # Softmax para classificação multiclasse

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            return model


        # Ajuste o input_shape para o dummy model: (frames, feature_dim, 1)
        dummy_input_shape = (DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1)
        dummy_model = create_dummy_model(dummy_input_shape, DUMMY_NUM_CLASSES)
        dummy_model.summary()
        dummy_model.save(DUMMY_MODEL_PATH)
        logger.info(f"Modelo dummy salvo em: {DUMMY_MODEL_PATH}")

        # Cria um LabelEncoder dummy
        dummy_encoder = LabelEncoder()
        # Assumindo classes 'fake' e 'real'
        dummy_encoder.fit(['fake', 'real'])
        with open(DUMMY_ENCODER_PATH, 'wb') as f:
            pickle.dump(dummy_encoder, f)
        logger.info(f"LabelEncoder dummy salvo em: {DUMMY_ENCODER_PATH}")

    except Exception as e:
        logger.error(f"Erro ao criar ou salvar modelo/encoder dummy para teste: {e}")
        logger.info("Pulando testes do preditor devido à falha na criação do modelo dummy.")
        # Limpa o diretório dummy se houver arquivos parciais
        if DUMMY_MODEL_DIR.exists():
            shutil.rmtree(DUMMY_MODEL_DIR)
        exit()  # Sai se o dummy model não puder ser criado

    # 2. Inicializa o ModelPredictor
    predictor_test = ModelPredictor(
        model_path=DUMMY_MODEL_DIR,
        expected_frames=DUMMY_EXPECTED_FRAMES,
        feature_dim=DUMMY_FEATURE_DIM
    )

    # 3. Testa o carregamento do modelo
    if predictor_test.model_loaded:
        logger.info("Modelo e LabelEncoder carregados com sucesso para teste.")

        # 4. Cria um AudioPreprocessor dummy para o teste de sliding_window
        from Voice2data import AudioPreprocessor  # Importa aqui para evitar circular

        dummy_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            n_mfcc=DUMMY_FEATURE_DIM,  # Garante que n_mfcc seja igual a FEATURE_DIM
            n_mels=DUMMY_FEATURE_DIM,  # Também para mel_spectrogram
            n_fft=512,
            vad_energy_thresh=-40.0,
            min_segment_duration=0.5,
            verbose=False
        )
        predictor_test.audio_preprocessor = dummy_preprocessor  # Injeta o preprocessor

        print("\n--- Testando predição de áudio único ---")
        # Criando uma amostra dummy com o shape (frames, feature_dim, 1)
        dummy_features_single = np.random.rand(DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1).astype(np.float32)
        try:
            label, confidence = predictor_test.predict_single_audio(dummy_features_single)
            print(f"Resultado Predição Única: {label} (Confiança: {confidence:.4f})")
        except Exception as e:
            logger.exception("Erro durante a predição de áudio único.")

        print("\n--- Testando predição com janela deslizante ---")
        # Criando features de áudio longo (ex: 200 frames > 100 expected_frames)
        # full_audio_features deve ser (frames, feature_dim, 1)
        long_audio_features_dummy = np.random.rand(DUMMY_EXPECTED_FRAMES * 2, DUMMY_FEATURE_DIM, 1).astype(np.float32)
        try:
            is_fake, avg_prob = sliding_window_predict(
                predictor_test,
                long_audio_features_dummy,
                window_frames=DUMMY_EXPECTED_FRAMES,  # Janela do mesmo tamanho do input do modelo
                hop_frames=DUMMY_EXPECTED_FRAMES // 2,  # Pulo de meia janela
                threshold=0.5
            )
            print(
                f"Resultado Janela Deslizante: {'FAKE' if is_fake else 'REAL'} (Probabilidade Média Fake: {avg_prob:.4f})")
        except Exception as e:
            logger.exception("Erro durante a predição com sliding window.")

        print("\n--- Testando predição em lote ---")
        # Criando várias amostras com o shape esperado para a função _prepare_features_for_prediction
        dummy_features_batch = [
            np.random.rand(DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1).astype(np.float32) * 0.1,  # Tendendo a REAL
            np.random.rand(DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1).astype(np.float32) * 100,  # Tendendo a FAKE
            np.random.rand(DUMMY_EXPECTED_FRAMES, DUMMY_FEATURE_DIM, 1).astype(np.float32) * 0.5  # Mediano
        ]
        try:
            batch_results = predictor_test.predict_batch(dummy_features_batch)
            for i, (label, confidence) in enumerate(batch_results):
                print(
                    f"Amostra {i + 1}: Resultado: {label} (Confiança: {confidence:.4f})")
        except Exception as e:
            logger.exception("Erro durante a predição em lote.")


    else:
        logger.error(
            "Falha ao carregar o modelo para teste. Verifique o caminho ou se o modelo foi salvo corretamente.")

    # Limpa o diretório dummy se ele foi criado para o teste
    if DUMMY_MODEL_DIR.exists():
        logger.info(f"Removendo diretório dummy: {DUMMY_MODEL_DIR}")
        shutil.rmtree(DUMMY_MODEL_DIR)