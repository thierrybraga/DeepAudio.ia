# TrainModel.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional, Any, Dict, Callable
import logging
import os
from datetime import datetime
import shutil

# Configura logger para TrainModel
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [TrainModel] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================ CAMADAS CUSTOMIZADAS ============================
# Estas camadas devem ser importadas em Predictor.py também.

class AudioFeatureNormalization(layers.Layer):
    """
    Camada de normalização adaptativa para features de áudio.
    Normaliza a média e a variância dos recursos de entrada.
    A camada aprende a média e a variância dos dados de treinamento durante a fase de adaptação.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super(AudioFeatureNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = 1e-6

    def build(self, input_shape):
        feature_dim = input_shape[self.axis] if self.axis != -1 else input_shape[-1]
        shape_for_stats = [1] * len(input_shape)
        actual_axis_index = len(input_shape) + self.axis if self.axis < 0 else self.axis
        shape_for_stats[actual_axis_index] = feature_dim

        self.mean = self.add_weight(
            name='mean',
            shape=shape_for_stats,
            initializer='zeros',
            trainable=False
        )
        self.variance = self.add_weight(
            name='variance',
            shape=shape_for_stats,
            initializer='ones',
            trainable=False
        )
        super(AudioFeatureNormalization, self).build(input_shape)

    def adapt(self, data: tf.Tensor):
        """
        Calcula e define a média e a variância dos dados de entrada.
        Esta função deve ser chamada APENAS UMA VEZ no dataset de treinamento para aprender as estatísticas.
        """
        if not tf.is_tensor(data):
            data = tf.constant(data, dtype=tf.float32)

        axes_to_reduce = [i for i in range(len(data.shape)) if
                          i != (len(data.shape) + self.axis if self.axis < 0 else self.axis)]
        mean = tf.reduce_mean(data, axis=axes_to_reduce, keepdims=True)
        variance = tf.math.reduce_variance(data, axis=axes_to_reduce, keepdims=True)

        self.mean.assign(mean)
        self.variance.assign(variance)
        logger.info(
            f"Camada de Normalização adaptada. Média shape: {self.mean.shape}, Variância shape: {self.variance.shape}")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return (inputs - self.mean) / tf.sqrt(self.variance + self.epsilon)

    def get_config(self) -> Dict[str, Any]:
        config = super(AudioFeatureNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


class AttentionLayer(layers.Layer):
    """
    Camada de atenção personalizada.
    Permite que o modelo foque nas partes mais relevantes da sequência de entrada.
    Implementa uma atenção do tipo Bahdanau (additive attention).
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) != 3:
            raise ValueError(
                f"AttentionLayer espera entrada 3D (batch, seq_len, features_dim), mas recebeu {input_shape}")

        features_dim = input_shape[-1]
        self.W = self.add_weight(name="att_weight", shape=(features_dim, features_dim),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(features_dim,),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_context", shape=(features_dim,),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        uit = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        ait = tf.matmul(uit, tf.expand_dims(self.u, axis=-1))
        ait = tf.squeeze(ait, axis=-1)
        alphas = tf.nn.softmax(ait)
        output = inputs * tf.expand_dims(alphas, axis=-1)
        output = tf.reduce_sum(output, axis=1)
        return output

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape((input_shape[0], input_shape[-1]))

    def get_config(self) -> Dict[str, Any]:
        return super(AttentionLayer, self).get_config()


# ============================ FUNÇÃO PARA CODIFICAÇÃO POSICIONAL ============================
def get_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """
    Gera codificação posicional sinusoidal para Transformer.

    Args:
        seq_len: Comprimento da sequência (número de frames).
        d_model: Dimensão das features (deve corresponder à dimensão do modelo).

    Returns:
        Tensor com codificação posicional de forma (1, seq_len, d_model).
    """

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    positions = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = get_angles(positions, i, d_model)

    # Aplica sin para índices pares e cos para índices ímpares
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_encoding = angles[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# ============================ BLOCO TRANSFORMER ============================
class TransformerEncoderBlock(layers.Layer):
    """
    Bloco Transformer com Multi-Head Self-Attention e Feed-Forward.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        # Multi-Head Self-Attention
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self) -> Dict[str, Any]:
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ============================

def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "transformer",
                 dropout_rate: float = 0.3, l2_reg_strength: float = 0.001) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim) para Transformer).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura do modelo a ser construído (atualizado apenas para 'transformer').
        dropout_rate: Taxa de dropout a ser aplicada em algumas camadas.
        l2_reg_strength: Força da regularização L2 para camadas densas.

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    if architecture == "transformer":
        # Adaptação para Transformer: Entrada (batch, sequence_length, features)
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape((input_shape[0], input_shape[1]), name="flatten_channel_for_transformer")(x)
        elif len(input_shape) == 2:
            pass  # Already in correct shape
        elif len(input_shape) == 3 and input_shape[-1] != 1:
            logger.warning(f"Input shape {input_shape} for Transformer expects 3D or 4D with last dim 1. "
                           f"Using as is, assuming last dim is feature.")
            pass
        else:
            raise ValueError(f"Input shape {input_shape} not suitable for 'transformer' architecture.")

        # Definir parâmetros do Transformer
        seq_len = input_shape[0]  # Comprimento da sequência (frames)
        d_model = input_shape[1] if len(input_shape) == 2 else input_shape[1] * input_shape[2] if len(
            input_shape) == 3 else input_shape[1]
        num_heads = 8  # Aumentado para melhor captura de dependências
        ff_dim = 256  # Aumentado para maior capacidade
        num_layers = 4  # Número de blocos Transformer

        # Ajustar dimensões se necessário
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # (batch, 1, features_dim)
            seq_len = 1

        # Projeção inicial para ajustar d_model
        x = layers.Dense(d_model, activation="relu", name="input_projection")(x)

        # Adicionar codificação posicional
        pos_encoding = get_positional_encoding(seq_len, d_model)
        x = x + pos_encoding

        # Múltiplos blocos Transformer
        for i in range(num_layers):
            x = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f"transformer_block_{i + 1}"
            )(x)

        # Pooling global para classificação
        x = layers.GlobalAveragePooling1D(name="transformer_avg_pool")(x)

    else:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. Apenas 'transformer' é suportado nesta versão.")

    # Camadas densas comuns
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg_strength), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.5, name="dropout_final")(x)

    # Camada de saída
    output_tensor = layers.Dense(num_classes, activation='softmax', name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


class ModelTrainer:
    """
    Classe para encapsular o processo de treinamento do modelo de Deep Learning.
    """

    def __init__(self,
                 model_dir: str,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 epochs: int = 50,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 patience: int = 10,
                 use_plateau: bool = True,
                 architecture: str = "transformer",
                 log_base_dir: str = "logs",
                 dropout_rate: float = 0.3,
                 l2_reg_strength: float = 0.001,
                 learning_rate: float = 0.001,
                 data_augmenter: Optional[Callable[[tf.Tensor, tf.Tensor], tf.data.Dataset]] = None,
                 class_weights: Optional[Dict[int, float]] = None
                 ):
        """
        Inicializa o ModelTrainer.

        Args:
            model_dir: Diretório onde o modelo será salvo.
            input_shape: A forma dos dados de entrada esperada pelo modelo.
            num_classes: Número de classes de saída.
            epochs: Número máximo de épocas de treinamento.
            batch_size: Tamanho do lote para treinamento.
            validation_split: Proporção dos dados para validação.
            patience: Paciência para EarlyStopping.
            use_plateau: Se True, usa ReduceLROnPlateau callback.
            architecture: Arquitetura do modelo (atualizado para 'transformer').
            log_base_dir: Diretório base para salvar logs do TensorBoard e CSV.
            dropout_rate: Taxa de dropout para as camadas do modelo.
            l2_reg_strength: Força da regularização L2.
            learning_rate: Taxa de aprendizado inicial para o otimizador.
            data_augmenter: Uma função que recebe X_train e y_train e retorna um tf.data.Dataset aumentado.
            class_weights: Um dicionário de pesos de classe para lidar com dados desbalanceados.
        """
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_filename = f"deepfake_detector_model_{self.timestamp}.h5"
        self.model_path = os.path.join(model_dir, self.model_filename)
        logger.info(f"O modelo será salvo em: {self.model_path}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.use_plateau = use_plateau
        self.architecture = architecture
        self.log_base_dir = log_base_dir
        self.dropout_rate = dropout_rate
        self.l2_reg_strength = l2_reg_strength
        self.learning_rate = learning_rate
        self.data_augmenter = data_augmenter
        self.class_weights = class_weights
        self.model: Optional[models.Model] = None
        self.label_encoder: Optional[LabelEncoder] = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_base_dir, exist_ok=True)

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Optional[tf.keras.callbacks.History]:
        """
        Treina o modelo com os dados fornecidos.

        Args:
            X: Features de entrada (numpy array de forma (num_samples, frames, features_dim) ou 4D para CNN).
            y: Labels de saída (numpy array ou lista de strings).

        Returns:
            Um objeto History contendo os dados de treinamento ou None em caso de erro.
        """
        logger.info(f"Iniciando treinamento do modelo com arquitetura: {self.architecture}")
        logger.info(f"Shape dos dados de entrada (X): {X.shape}, Shape das labels (y): {y.shape}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Dados de treinamento ou labels estão vazios. Treinamento abortado.")
            return None

        # Encode labels if they are strings
        if y.dtype == object:
            self.label_encoder = LabelEncoder()
            try:
                y_encoded = self.label_encoder.fit_transform(y)
                if len(self.label_encoder.classes_) < 2:
                    logger.error(
                        "Menos de duas classes únicas encontradas nas labels. Não é possível treinar um modelo de classificação.")
                    return None
            except Exception as e:
                logger.error(f"Erro ao codificar labels: {e}")
                return None
            logger.info(f"Labels originais: {self.label_encoder.classes_}")
            logger.info(f"Labels codificadas (primeiros 5): {y_encoded[:5]}")
        else:
            y_encoded = y
            if len(np.unique(y_encoded)) < 2:
                logger.error(
                    "Menos de duas classes únicas encontradas nas labels. Não é possível treinar um modelo de classificação.")
                return None

        # Split data
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split, random_state=42, stratify=y_encoded
            )
        except ValueError as e:
            logger.error(
                f"Erro ao dividir dados (train_test_split) com estratificação: {e}. Isso pode ocorrer se uma classe tiver pouquíssimos exemplos.")
            logger.warning(
                "Tentando dividir dados sem estratificação (random_state fixo). Isso pode impactar a distribuição das classes se o dataset for pequeno ou desbalanceado.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split, random_state=42
            )
            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                logger.error(
                    "A divisão dos dados resultou em conjuntos de treinamento ou validação vazios. Treinamento abortado.")
                return None

        logger.info(f"Dados divididos: Treino={X_train.shape}, Validação={X_val.shape}")

        # Create model
        self.model = create_model(self.input_shape, self.num_classes, self.architecture,
                                  self.dropout_rate, self.l2_reg_strength)
        logger.info("Resumo do Modelo:")
        self.model.summary(print_fn=lambda x: logger.info(x))

        # Adapt normalization layer
        norm_layer = None
        for layer in self.model.layers:
            if isinstance(layer, AudioFeatureNormalization):
                norm_layer = layer
                break

        if norm_layer:
            logger.info("Adaptando a camada AudioFeatureNormalization com dados de treinamento...")
            norm_layer.adapt(X_train)
            logger.info("Camada AudioFeatureNormalization adaptada com sucesso.")
        else:
            logger.warning(
                "Camada AudioFeatureNormalization não encontrada no modelo. Certifique-se de que ela está incluída na função create_model.")

        # Configure optimizer with configurable learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks: List[tf.keras.callbacks.Callback] = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        ]

        if self.use_plateau:
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.patience / 2), min_lr=0.00001,
                                  verbose=1))

        log_dir_tensorboard = os.path.join(self.log_base_dir, f"tensorboard_{self.timestamp}")
        callbacks.append(TensorBoard(log_dir=log_dir_tensorboard, histogram_freq=1))

        csv_log_path = os.path.join(self.log_base_dir, f"training_log_{self.timestamp}.csv")
        callbacks.append(CSVLogger(csv_log_path, append=True))

        # Prepare data for training
        if self.data_augmenter:
            logger.info("Aplicando aumento de dados (data augmentation) ao dataset de treinamento.")
            train_dataset = self.data_augmenter(X_train, y_train).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.batch_size).prefetch(
                tf.data.AUTOTUNE)
            fit_kwargs = {"x": train_dataset, "validation_data": val_dataset}
        else:
            fit_kwargs = {"x": X_train, "y": y_train, "validation_data": (X_val, y_val)}

        # Add class weights if provided
        if self.class_weights:
            logger.info(f"Usando pesos de classe para treinamento: {self.class_weights}")
            fit_kwargs["class_weight"] = self.class_weights

        logger.info(f"Iniciando o treinamento do modelo por {self.epochs} épocas com batch_size={self.batch_size}...")
        try:
            history = self.model.fit(
                **fit_kwargs,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )
            logger.info("Treinamento do modelo concluído.")
            logger.info(f"Modelo salvo em: {self.model_path}")
            return history
        except Exception as e:
            logger.exception(f"Erro durante o treinamento do modelo: {e}")
            return None

    def get_label_encoder(self) -> Optional[LabelEncoder]:
        """Retorna o LabelEncoder utilizado, se houver."""
        return self.label_encoder


# ============================ FUNÇÕES DE AUMENTO DE DADOS (PLACEHOLDER) ============================

def simple_audio_augmenter(X_train: np.ndarray, y_train: np.ndarray) -> tf.data.Dataset:
    """
    Função de placeholder para aumento de dados de áudio.
    Adicione técnicas de aumento de dados reais aqui (e.g., ruído, pitch shift, time stretch).
    """

    def _augment(audio_features, label):
        # Exemplo: Adicionar ruído aleatório (muito simplificado)
        noise = tf.random.normal(shape=tf.shape(audio_features), mean=0.0, stddev=0.01, dtype=tf.float32)
        augmented_audio_features = audio_features + noise
        return augmented_audio_features, label

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# Exemplo de uso (apenas para teste direto do arquivo)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando teste de TrainModel.py (Transformer)...")

    dummy_input_shape = (100, 40)  # (frames, features) for Transformer
    num_samples = 200

    # Criar um dataset desbalanceado para testar class_weights
    dummy_X = np.random.rand(num_samples, dummy_input_shape[0], dummy_input_shape[1]).astype(np.float32)
    dummy_y = np.array(['REAL'] * int(num_samples * 0.75) + ['FAKE'] * int(num_samples * 0.25))
    np.random.shuffle(dummy_y)

    print(f"Dummy X shape: {dummy_X.shape}")
    print(f"Dummy y shape: {dummy_y.shape}")
    print(f"Distribuição das classes no dummy_y: {np.unique(dummy_y, return_counts=True)}")

    # Definir um diretório de teste específico para o Main
    test_model_dir = "test_models_transformer"
    test_log_dir = os.path.join(test_model_dir, "logs")

    # Limpar diretório de teste antes de iniciar
    if os.path.exists(test_model_dir):
        logger.info(f"Limpando diretório de teste existente '{test_model_dir}'...")
        shutil.rmtree(test_model_dir)

    # Calcular pesos de classe para o dataset desbalanceado
    num_fake = np.sum(dummy_y == 'FAKE')
    num_real = np.sum(dummy_y == 'REAL')
    total_samples_for_weights = num_fake + num_real
    calculated_class_weights = {
        0: total_samples_for_weights / (2.0 * num_fake),  # Weight for 'FAKE'
        1: total_samples_for_weights / (2.0 * num_real)  # Weight for 'REAL'
    }
    logger.info(f"Pesos de classe calculados: {calculated_class_weights}")

    # Testar a arquitetura Transformer
    trainer = ModelTrainer(
        model_dir=test_model_dir,
        input_shape=dummy_input_shape,
        num_classes=len(np.unique(dummy_y)),
        epochs=3,  # Reduced epochs for faster testing
        batch_size=32,
        patience=2,
        use_plateau=True,
        architecture="transformer",
        log_base_dir=test_log_dir,
        dropout_rate=0.4,
        l2_reg_strength=0.0005,
        learning_rate=0.0005,
        data_augmenter=simple_audio_augmenter,
        class_weights=calculated_class_weights
    )

    try:
        history = trainer.train_model(dummy_X, dummy_y)
        if history:
            print(f"\nTreinamento concluído.")
            if 'val_loss' in history.history:
                print(f"Melhor perda de validação: {min(history.history['val_loss']):.4f}")
            if 'val_accuracy' in history.history:
                print(f"Melhor acurácia de validação: {max(history.history['val_accuracy']):.4f}")
            model_saved_path = trainer.model_path
            if os.path.exists(model_saved_path):
                print(f"Modelo de teste salvo em {model_saved_path}")
            else:
                print(f"Modelo de teste NÃO foi salvo em {model_saved_path}.")
        else:
            print(f"Treinamento não foi concluído com sucesso.")
    except Exception as e:
        logger.exception(f"Erro durante o treinamento de teste: {e}")

    # Limpeza final após todos os testes
    if os.path.exists(test_model_dir):
        logger.info(f"\nLimpando diretório de teste '{test_model_dir}'...")
        shutil.rmtree(test_model_dir)
        logger.info(f"'{test_model_dir}' removido.")