
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
    Camada de atenção personalizada (Bahdanau).
    Permite que o modelo foque nas partes mais relevantes da sequência de entrada.
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

class GraphAttentionLayer(layers.Layer):
    """
    Camada de Atenção em Grafos (GAT) simplificada para modelar relações espectro-temporais.
    Inspirada em https://keras.io/examples/graph/gat_node_classification/
    """
    def __init__(self, output_dim: int, num_heads: int = 4, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]):
        feature_shape, adj_shape = input_shape
        self.input_dim = feature_shape[-1]

        # Weights for each attention head
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.input_dim, self.output_dim * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )
        self.attention_kernel = self.add_weight(
            name="attention_kernel",
            shape=(2 * self.output_dim, self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim * self.num_heads,),
            initializer="zeros",
            trainable=True
        )
        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        features, adj_matrix = inputs
        # Linear transformation
        Wh = tf.matmul(features, self.kernel)  # (batch, nodes, output_dim * num_heads)
        Wh = tf.reshape(Wh, (-1, features.shape[1], self.num_heads, self.output_dim))  # (batch, nodes, heads, out_dim)

        # Attention mechanism
        Wh1 = Wh[:, :, :, None, :]  # (batch, nodes, heads, 1, out_dim)
        Wh2 = Wh[:, :, :, :, None]  # (batch, nodes, heads, out_dim, 1)
        e = tf.matmul(Wh1, Wh2, transpose_b=True)  # (batch, nodes, heads, nodes, 1)
        e = tf.squeeze(e, axis=-1)  # (batch, nodes, heads, nodes)

        # Apply attention coefficients
        attention = tf.nn.softmax(e, axis=1)  # (batch, nodes, heads, nodes)
        attention = tf.reduce_mean(attention, axis=2)  # (batch, nodes, nodes)
        attention = attention * adj_matrix  # Apply adjacency matrix

        # Aggregate features
        output = tf.matmul(attention, Wh)  # (batch, nodes, heads, out_dim)
        output = tf.reshape(output, (-1, features.shape[1], self.output_dim * self.num_heads))  # (batch, nodes, out_dim * heads)
        output = output + self.bias
        return tf.nn.relu(output)

    def get_config(self) -> Dict[str, Any]:
        config = super(GraphAttentionLayer, self).get_config()
        config.update({"output_dim": self.output_dim, "num_heads": self.num_heads})
        return config

# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ============================

def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "default",
                 dropout_rate: float = 0.3, l2_reg_strength: float = 0.001) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim, 1) para CNN).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura do modelo a ser construído ("default", "cnn_baseline", "bidirectional_gru", "resnet_gru", "transformer", "aasist").
        dropout_rate: Taxa de dropout a ser aplicada em algumas camadas.
        l2_reg_strength: Força da regularização L2 para camadas densas.

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    # Helper function for reshaping for CNN
    def _reshape_for_cnn(tensor: tf.Tensor, target_shape: Tuple[int, ...]) -> tf.Tensor:
        """Reshapes the input tensor to 4D for CNNs, handling different input_shape scenarios."""
        if len(tensor.shape) == 3:  # (batch, frames, features_dim)
            return layers.Reshape((target_shape[0], target_shape[1], 1))(tensor)
        elif len(tensor.shape) == 4 and tensor.shape[-1] != 1:
            logger.warning(f"Input tensor shape {tensor.shape} for CNN expects last dim to be 1. Slicing to 1 channel.")
            return layers.Lambda(lambda y: y[..., :1])(tensor)
        elif len(tensor.shape) == 4 and tensor.shape[-1] == 1:
            return tensor
        else:
            raise ValueError(f"Unexpected input tensor shape for CNN: {tensor.shape}")

    # Helper function for Residual Block
    def residual_block(x_in: tf.Tensor, filters: int, kernel_size: Tuple[int, int], stage: str) -> tf.Tensor:
        shortcut = x_in

        x = layers.Conv2D(filters, kernel_size, activation='relu', padding='same', name=f"res{stage}_conv1")(x_in)
        x = layers.BatchNormalization(name=f"res{stage}_bn1")(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', name=f"res{stage}_conv2")(x)
        x = layers.BatchNormalization(name=f"res{stage}_bn2")(x)

        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=f"res{stage}_shortcut")(shortcut)

        x = layers.add([x, shortcut], name=f"res{stage}_add")
        x = layers.Activation('relu', name=f"res{stage}_relu")(x)
        return x

    if architecture == "default":
        x = _reshape_for_cnn(x, input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Dropout(dropout_rate, name="dropout1")(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = layers.Dropout(dropout_rate, name="dropout2")(x)
        shape_before_gru = x.shape
        x = layers.Reshape((shape_before_gru[1], shape_before_gru[2] * shape_before_gru[3]), name="reshape_for_gru")(x)
        if tf.config.list_physical_devices('GPU'):
            logger.info("Usando CuDNNGRU para otimização de GPU.")
            x = layers.CuDNNGRU(128, return_sequences=True, name="gru1")(x)
        else:
            logger.info("Usando GRU (CPU/compatível com GPU sem CuDNN).")
            x = layers.GRU(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate, name="gru1")(x)
            x = layers.GRU(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate, name="gru2")(x)
        x = AttentionLayer(name="attention_layer")(x)

    elif architecture == "cnn_baseline":
        x = _reshape_for_cnn(x, input_shape)
        x = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name="conv_b1")(x)
        x = layers.BatchNormalization(name="bn_b1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool_b1")(x)
        x = layers.Dropout(dropout_rate, name="dropout_b1")(x)
        x = layers.Conv2D(64, (5, 5), activation='relu', padding='same', name="conv_b2")(x)
        x = layers.BatchNormalization(name="bn_b2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool_b2")(x)
        x = layers.Dropout(dropout_rate, name="dropout_b2")(x)
        x = layers.Flatten(name="flatten")(x)

    elif architecture == "bidirectional_gru":
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape((input_shape[0], input_shape[1]), name="flatten_channel_for_gru")(x)
        elif len(input_shape) == 2:
            pass
        elif len(input_shape) == 3 and input_shape[-1] != 1:
            logger.warning(f"Input shape {input_shape} for Bidirectional GRU expects 3D or 4D with last dim 1. "
                           f"Using as is, assuming last dim is feature.")
            pass
        else:
            raise ValueError(f"Input shape {input_shape} not suitable for 'bidirectional_gru' architecture.")
        if tf.config.list_physical_devices('GPU'):
            logger.info("Usando CuDNNGRU Bidirecional para otimização de GPU.")
            x = layers.Bidirectional(layers.CuDNNGRU(128, return_sequences=True), name="bi_gru1")(x)
            x = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True), name="bi_gru2")(x)
        else:
            logger.info("Usando GRU Bidirecional (CPU/compatível com GPU sem CuDNN).")
            x = layers.Bidirectional(
                layers.GRU(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate),
                name="bi_gru1")(x)
            x = layers.Bidirectional(
                layers.GRU(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate),
                name="bi_gru2")(x)
        x = AttentionLayer(name="attention_layer")(x)

    elif architecture == "resnet_gru":
        x = _reshape_for_cnn(x, input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="resnet_conv_init")(x)
        x = layers.BatchNormalization(name="resnet_bn_init")(x)
        x = layers.MaxPooling2D((2, 2), name="resnet_pool_init")(x)
        x = residual_block(x, 64, (3, 3), stage='a')
        x = layers.MaxPooling2D((2, 2), name="resnet_pool_a")(x)
        x = layers.Dropout(dropout_rate, name="resnet_dropout_a")(x)
        x = residual_block(x, 128, (3, 3), stage='b')
        x = layers.MaxPooling2D((2, 2), name="resnet_pool_b")(x)
        x = layers.Dropout(dropout_rate, name="resnet_dropout_b")(x)
        shape_before_gru = x.shape
        x = layers.Reshape((shape_before_gru[1], shape_before_gru[2] * shape_before_gru[3]), name="resnet_reshape_for_gru")(x)
        if tf.config.list_physical_devices('GPU'):
            x = layers.CuDNNGRU(128, return_sequences=True, name="resnet_gru1")(x)
        else:
            x = layers.GRU(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate, name="resnet_gru1")(x)
        x = AttentionLayer(name="attention_layer_resnet")(x)

    elif architecture == "transformer":
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape((input_shape[0], input_shape[1]), name="flatten_channel_for_transformer")(x)
        elif len(input_shape) == 2:
            pass
        elif len(input_shape) == 3 and input_shape[-1] != 1:
            logger.warning(f"Input shape {input_shape} for Transformer expects 3D or 4D with last dim 1. "
                           f"Using as is, assuming last dim is feature.")
            pass
        else:
            raise ValueError(f"Input shape {input_shape} not suitable for 'transformer' architecture.")
        seq_len = input_shape[0] if len(input_shape) >= 2 else input_shape[0]
        feature_dim = input_shape[1] if len(input_shape) == 2 else input_shape[1] * input_shape[2] if len(input_shape) == 3 else input_shape[1]
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
            seq_len = 1
        pos_encoding = layers.Embedding(seq_len, feature_dim)(tf.range(seq_len))
        x = x + pos_encoding
        num_heads = 4
        ff_dim = 64
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ff_output = layers.Dense(ff_dim, activation="relu")(x)
        ff_output = layers.Dense(feature_dim)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        x = layers.GlobalAveragePooling1D(name="transformer_avg_pool")(x)

    elif architecture == "aasist":
        # AASIST: Spectro-Temporal Graph Attention Network
        # Expects input as spectrogram (batch, frames, features, 1)
        x = _reshape_for_cnn(x, input_shape)

        # Initial convolutional layers to extract spectro-temporal features
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="aasist_conv1")(x)
        x = layers.BatchNormalization(name="aasist_bn1")(x)
        x = layers.MaxPooling2D((2, 2), name="aasist_pool1")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout1")(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="aasist_conv2")(x)
        x = layers.BatchNormalization(name="aasist_bn2")(x)
        x = layers.MaxPooling2D((2, 2), name="aasist_pool2")(x)
        x = layers.Dropout(dropout_rate, name="aasist_dropout2")(x)

        # Reshape for graph attention: (batch, nodes, features)
        shape_before_gat = x.shape
        num_nodes = shape_before_gat[1] * shape_before_gat[2]
        x = layers.Reshape((num_nodes, shape_before_gat[3]), name="aasist_reshape_for_gat")(x)

        # Create adjacency matrix (simplified, fully connected graph)
        adj_matrix = tf.ones((x.shape[0], num_nodes, num_nodes), dtype=tf.float32)

        # Graph Attention Layer
        x = GraphAttentionLayer(output_dim=64, num_heads=4, name="aasist_gat1")([x, adj_matrix])
        x = layers.Dropout(dropout_rate, name="aasist_dropout_gat1")(x)

        # Heterogeneous stack node (simplified as a dense layer to aggregate)
        x = layers.Dense(128, activation='relu', name="aasist_stack_node")(x)
        x = layers.GlobalAveragePooling1D(name="aasist_global_pool")(x)

    else:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. Escolha 'default', 'cnn_baseline', 'bidirectional_gru', 'resnet_gru', 'transformer', ou 'aasist'.")

    # Camadas densas comuns a todas as arquiteturas
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
                 architecture: str = "default",
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
            input_shape: A forma dos dados de entrada esperada pelo modelo (e.g., (frames, features_dim, 1)).
            num_classes: Número de classes de saída.
            epochs: Número máximo de épocas de treinamento.
            batch_size: Tamanho do lote para treinamento.
            validation_split: Proporção dos dados para validação.
            patience: Paciência para EarlyStopping.
            use_plateau: Se True, usa ReduceLROnPlateau callback.
            architecture: Arquitetura do modelo ("default", "cnn_baseline", "bidirectional_gru", "resnet_gru", "transformer", "aasist").
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
        noise = tf.random.normal(shape=tf.shape(audio_features), mean=0.0, stddev=0.01, dtype=tf.float32)
        augmented_audio_features = audio_features + noise
        return augmented_audio_features, label

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Exemplo de uso (apenas para teste direto do arquivo)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando teste de TrainModel.py (Refatorado com AASIST)...")

    dummy_input_shape = (100, 40, 1)  # (frames, features, channels)
    num_samples = 200

    # Criar um dataset desbalanceado para testar class_weights
    dummy_X = np.random.rand(num_samples, dummy_input_shape[0], dummy_input_shape[1], dummy_input_shape[2]).astype(
        np.float32)
    dummy_y = np.array(['REAL'] * int(num_samples * 0.75) + ['FAKE'] * int(num_samples * 0.25))
    np.random.shuffle(dummy_y)

    print(f"Dummy X shape: {dummy_X.shape}")
    print(f"Dummy y shape: {dummy_y.shape}")
    print(f"Distribuição das classes no dummy_y: {np.unique(dummy_y, return_counts=True)}")

    # Definir um diretório de teste específico para o Main
    test_model_dir = "test_models_refactored"
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

    # Testando todas as arquiteturas, incluindo a nova "aasist"
    architectures_to_test = ["default", "cnn_baseline", "bidirectional_gru", "resnet_gru", "transformer", "aasist"]

    for arch in architectures_to_test:
        logger.info(f"\n--- Testando arquitetura: {arch} ---")
        current_dummy_input_shape = dummy_input_shape
        if arch == "transformer" or arch == "aasist":
            current_dummy_input_shape = (dummy_input_shape[0], dummy_input_shape[1])
            if len(dummy_X.shape) == 4 and dummy_X.shape[-1] == 1:
                temp_dummy_X = dummy_X.reshape(dummy_X.shape[0], dummy_X.shape[1], dummy_X.shape[2])
            else:
                temp_dummy_X = dummy_X
            logger.info(
                f"Ajustando dummy_X para arquitetura '{arch}'. Novo shape: {temp_dummy_X.shape}. Input_shape para modelo: {current_dummy_input_shape}")
        else:
            temp_dummy_X = dummy_X
            logger.info(
                f"Usando dummy_X original para arquitetura '{arch}'. Shape: {temp_dummy_X.shape}. Input_shape para modelo: {current_dummy_input_shape}")

        trainer = ModelTrainer(
            model_dir=os.path.join(test_model_dir, arch),
            input_shape=current_dummy_input_shape,
            num_classes=len(np.unique(dummy_y)),
            epochs=3,
            batch_size=32,
            patience=2,
            use_plateau=True,
            architecture=arch,
            log_base_dir=os.path.join(test_log_dir, arch),
            dropout_rate=0.4,
            l2_reg_strength=0.0005,
            learning_rate=0.0005,
            data_augmenter=simple_audio_augmenter,
            class_weights=calculated_class_weights
        )

        try:
            history = trainer.train_model(temp_dummy_X, dummy_y)
            if history:
                print(f"\nTreinamento para {arch} concluído.")
                if 'val_loss' in history.history:
                    print(f"Melhor perda de validação: {min(history.history['val_loss']):.4f}")
                if 'val_accuracy' in history.history:
                    print(f"Melhor acurácia de validação: {max(history.history['val_accuracy']):.4f}")
                model_saved_path = trainer.model_path
                if os.path.exists(model_saved_path):
                    print(f"Modelo de teste para {arch} salvo em {model_saved_path}")
                else:
                    print(f"Modelo de teste para {arch} NÃO foi salvo em {model_saved_path}.")
            else:
                print(f"Treinamento para {arch} não foi concluído com sucesso.")
        except Exception as e:
            logger.exception(f"Erro durante o treinamento de teste para arquitetura {arch}.")

    # Limpeza final após todos os testes
    if os.path.exists(test_model_dir):
        logger.info(f"\nLimpando diretório de teste '{test_model_dir}'...")
        shutil.rmtree(test_model_dir)
        logger.info(f"'{test_model_dir}' removido.")