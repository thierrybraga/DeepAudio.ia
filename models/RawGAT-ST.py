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

class AudioFeatureNormalization(layers.Layer):
    """
    Camada de normalização adaptativa para features de áudio.
    Normaliza a média e a variância dos recursos de entrada.
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
        if not tf.is_tensor(data):
            data = tf.constant(data, dtype=tf.float32)
        axes_to_reduce = [i for i in range(len(data.shape)) if
                          i != (len(data.shape) + self.axis if self.axis < 0 else self.axis)]
        mean = tf.reduce_mean(data, axis=axes_to_reduce, keepdims=True)
        variance = tf.math.reduce_variance(data, axis=axes_to_reduce, keepdims=True)
        self.mean.assign(mean)
        self.variance.assign(variance)
        logger.info(f"Camada de Normalização adaptada. Média shape: {self.mean.shape}, Variância shape: {self.variance.shape}")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return (inputs - self.mean) / tf.sqrt(self.variance + self.epsilon)

    def get_config(self) -> Dict[str, Any]:
        config = super(AudioFeatureNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config

class AttentionLayer(layers.Layer):
    """
    Camada de atenção personalizada (Bahdanau).
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) != 3:
            raise ValueError(f"AttentionLayer espera entrada 3D (batch, seq_len, features_dim), mas recebeu {input_shape}")
        features_dim = input_shape[-1]
        self.W = self.add_weight(name="att_weight", shape=(features_dim, features_dim), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(features_dim,), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_context", shape=(features_dim,), initializer="glorot_uniform", trainable=True)
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
    Camada de Atenção em Grafos (GAT) para modelagem espectro-temporal.
    Adaptada de implementações Keras para RawGAT-ST.
    """
    def __init__(self, output_dim: int, num_heads: int = 4, dropout_rate: float = 0.1, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.kernel = None
        self.attention_kernel = None
        self.bias = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(feature_dim, self.output_dim * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )
        self.attention_kernel = self.add_weight(
            name="attention_kernel",
            shape=(2 * self.output_dim, 1),
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

    def call(self, inputs: tf.Tensor, adjacency_matrix: tf.Tensor = None) -> tf.Tensor:
        # inputs: (batch, nodes, features)
        batch_size, num_nodes, feature_dim = inputs.shape

        # Transform features
        features = tf.matmul(inputs, self.kernel)  # (batch, nodes, output_dim * num_heads)
        features = tf.reshape(features, (batch_size, num_nodes, self.num_heads, self.output_dim))
        features = tf.transpose(features, [0, 2, 1, 3])  # (batch, heads, nodes, output_dim)

        # Compute attention coefficients
        f_1 = tf.matmul(features, self.attention_kernel[:self.output_dim, :])  # (batch, heads, nodes, 1)
        f_2 = tf.matmul(features, self.attention_kernel[self.output_dim:, :])  # (batch, heads, nodes, 1)
        logits = f_1 + tf.transpose(f_2, [0, 1, 3, 2])  # (batch, heads, nodes, nodes)

        # Apply adjacency matrix (if provided) or assume fully connected
        if adjacency_matrix is None:
            adjacency_matrix = tf.ones((num_nodes, num_nodes))  # Fully connected graph
        logits = tf.where(adjacency_matrix > 0, logits, tf.fill(tf.shape(logits), -1e9))
        attention_coefs = tf.nn.softmax(logits, axis=-1)
        attention_coefs = tf.nn.dropout(attention_coefs, rate=self.dropout_rate)

        # Aggregate features
        outputs = tf.matmul(attention_coefs, features)  # (batch, heads, nodes, output_dim)
        outputs = tf.transpose(outputs, [0, 2, 1, 3])  # (batch, nodes, heads, output_dim)
        outputs = tf.reshape(outputs, (batch_size, num_nodes, self.output_dim * self.num_heads))
        outputs = outputs + self.bias
        return tf.nn.relu(outputs)

    def get_config(self) -> Dict[str, Any]:
        config = super(GraphAttentionLayer, self).get_config()
        config.update({"output_dim": self.output_dim, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate})
        return config

# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ============================

def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "default",
                 dropout_rate: float = 0.3, l2_reg_strength: float = 0.001) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim, 1) para CNN).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura ("default", "cnn_baseline", "bidirectional_gru", "resnet_gru", "transformer", "rawgat_st").
        dropout_rate: Taxa de dropout.
        l2_reg_strength: Força da regularização L2.

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    def _reshape_for_cnn(tensor: tf.Tensor, target_shape: Tuple[int, ...]) -> tf.Tensor:
        if len(tensor.shape) == 3:
            return layers.Reshape((target_shape[0], target_shape[1], 1))(tensor)
        elif len(tensor.shape) == 4 and tensor.shape[-1] != 1:
            logger.warning(f"Input tensor shape {tensor.shape} for CNN expects last dim to be 1. Slicing to 1 channel.")
            return layers.Lambda(lambda y: y[..., :1])(tensor)
        elif len(tensor.shape) == 4 and tensor.shape[-1] == 1:
            return tensor
        else:
            raise ValueError(f"Unexpected input tensor shape for CNN: {tensor.shape}")

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
            logger.warning(f"Input shape {input_shape} for Bidirectional GRU expects 3D or 4D with last dim 1.")
            pass
        else:
            raise ValueError(f"Input shape {input_shape} not suitable for 'bidirectional_gru' architecture.")
        if tf.config.list_physical_devices('GPU'):
            logger.info("Usando CuDNNGRU Bidirecional para otimização de GPU.")
            x = layers.Bidirectional(layers.CuDNNGRU(128, return_sequences=True), name="bi_gru1")(x)
            x = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True), name="bi_gru2")(x)
        else:
            logger.info("Usando GRU Bidirecional (CPU/compatível com GPU sem CuDNN).")
            x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate), name="bi_gru1")(x)
            x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate), name="bi_gru2")(x)
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
            logger.warning(f"Input shape {input_shape} for Transformer expects 3D or 4D with last dim 1.")
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

    elif architecture == "rawgat_st":
        # RawGAT-ST: Spectro-Temporal Graph Attention Network
        # Input: Assumes spectrogram input (batch, frames, features, 1) or (batch, frames, features)
        if len(input_shape) == 4 and input_shape[-1] == 1:
            x = layers.Reshape((input_shape[0], input_shape[1]), name="flatten_channel_for_rawgat")(x)
        elif len(input_shape) == 2:
            pass
        else:
            raise ValueError(f"Input shape {input_shape} not suitable for 'rawgat_st' architecture. Expected 2D (frames, features) or 4D with last dim 1.")

        # Simplified SincNet-inspired layer (using 1D Conv as proxy)
        x = layers.Conv1D(filters=64, kernel_size=51, strides=1, padding='same', activation='relu', name="sincnet_proxy")(x)
        x = layers.BatchNormalization(name="sincnet_bn")(x)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="sincnet_pool")(x)
        x = layers.Dropout(dropout_rate, name="sincnet_dropout")(x)

        # Prepare for graph attention: treat frames and features as nodes
        seq_len, feature_dim = x.shape[1], x.shape[2]
        num_nodes = seq_len  # Treat each time frame as a node
        x = layers.Reshape((num_nodes, feature_dim), name="reshape_for_gat")(x)

        # Create a simple adjacency matrix (fully connected graph for simplicity)
        # In a full implementation, this could be computed based on spectro-temporal proximity
        adjacency_matrix = tf.ones((num_nodes, num_nodes))

        # Apply Graph Attention Layers
        x = GraphAttentionLayer(output_dim=64, num_heads=4, dropout_rate=dropout_rate, name="gat1")(x, adjacency_matrix)
        x = layers.LayerNormalization(epsilon=1e-6, name="gat_ln1")(x)
        x = GraphAttentionLayer(output_dim=32, num_heads=4, dropout_rate=dropout_rate, name="gat2")(x, adjacency_matrix)
        x = layers.LayerNormalization(epsilon=1e-6, name="gat_ln2")(x)

        # Aggregate node features
        x = layers.GlobalAveragePooling1D(name="gat_avg_pool")(x)

    else:
        raise ValueError(
            f"Arquitetura '{architecture}' não reconhecida. Escolha 'default', 'cnn_baseline', 'bidirectional_gru', 'resnet_gru', 'transformer', ou 'rawgat_st'.")

    # Camadas densas comuns
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg_strength), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.5, name="dropout_final")(x)
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
                 class_weights: Optional[Dict[int, float]] = None):
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
        logger.info(f"Iniciando treinamento do modelo com arquitetura: {self.architecture}")
        logger.info(f"Shape dos dados de entrada (X): {X.shape}, Shape das labels (y): {y.shape}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Dados de treinamento ou labels estão vazios. Treinamento abortado.")
            return None

        if y.dtype == object:
            self.label_encoder = LabelEncoder()
            try:
                y_encoded = self.label_encoder.fit_transform(y)
                if len(self.label_encoder.classes_) < 2:
                    logger.error("Menos de duas classes únicas encontradas nas labels.")
                    return None
            except Exception as e:
                logger.error(f"Erro ao codificar labels: {e}")
                return None
            logger.info(f"Labels originais: {self.label_encoder.classes_}")
            logger.info(f"Labels codificadas (primeiros 5): {y_encoded[:5]}")
        else:
            y_encoded = y
            if len(np.unique(y_encoded)) < 2:
                logger.error("Menos de duas classes únicas encontradas nas labels.")
                return None

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split, random_state=42, stratify=y_encoded
            )
        except ValueError as e:
            logger.error(f"Erro ao dividir dados com estratificação: {e}")
            logger.warning("Tentando dividir dados sem estratificação.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split, random_state=42
            )
            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                logger.error("A divisão dos dados resultou em conjuntos vazios.")
                return None

        logger.info(f"Dados divididos: Treino={X_train.shape}, Validação={X_val.shape}")

        self.model = create_model(self.input_shape, self.num_classes, self.architecture,
                                  self.dropout_rate, self.l2_reg_strength)
        logger.info("Resumo do Modelo:")
        self.model.summary(print_fn=lambda x: logger.info(x))

        norm_layer = None
        for layer in self.model.layers:
            if isinstance(layer, AudioFeatureNormalization):
                norm_layer = layer
                break
        if norm_layer:
            logger.info("Adaptando a camada AudioFeatureNormalization...")
            norm_layer.adapt(X_train)
            logger.info("Camada AudioFeatureNormalization adaptada com sucesso.")
        else:
            logger.warning("Camada AudioFeatureNormalization não encontrada no modelo.")

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks: List[tf.keras.callbacks.Callback] = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        ]
        if self.use_plateau:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.patience / 2), min_lr=0.00001, verbose=1))
        log_dir_tensorboard = os.path.join(self.log_base_dir, f"tensorboard_{self.timestamp}")
        callbacks.append(TensorBoard(log_dir=log_dir_tensorboard, histogram_freq=1))
        csv_log_path = os.path.join(self.log_base_dir, f"training_log_{self.timestamp}.csv")
        callbacks.append(CSVLogger(csv_log_path, append=True))

        if self.data_augmenter:
            logger.info("Aplicando aumento de dados ao dataset de treinamento.")
            train_dataset = self.data_augmenter(X_train, y_train).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            fit_kwargs = {"x": train_dataset, "validation_data": val_dataset}
        else:
            fit_kwargs = {"x": X_train, "y": y_train, "validation_data": (X_val, y_val)}

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
        return self.label_encoder

def simple_audio_augmenter(X_train: np.ndarray, y_train: np.ndarray) -> tf.data.Dataset:
    def _augment(audio_features, label):
        noise = tf.random.normal(shape=tf.shape(audio_features), mean=0.0, stddev=0.01, dtype=tf.float32)
        augmented_audio_features = audio_features + noise
        return augmented_audio_features, label
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando teste de TrainModel.py (Refatorado)...")
    dummy_input_shape = (100, 40, 1)
    num_samples = 200
    dummy_X = np.random.rand(num_samples, dummy_input_shape[0], dummy_input_shape[1], dummy_input_shape[2]).astype(np.float32)
    dummy_y = np.array(['REAL'] * int(num_samples * 0.75) + ['FAKE'] * int(num_samples * 0.25))
    np.random.shuffle(dummy_y)
    print(f"Dummy X shape: {dummy_X.shape}")
    print(f"Dummy y shape: {dummy_y.shape}")
    print(f"Distribuição das classes no dummy_y: {np.unique(dummy_y, return_counts=True)}")
    test_model_dir = "test_models_refactored"
    test_log_dir = os.path.join(test_model_dir, "logs")
    if os.path.exists(test_model_dir):
        logger.info(f"Limpando diretório de teste existente '{test_model_dir}'...")
        shutil.rmtree(test_model_dir)
    num_fake = np.sum(dummy_y == 'FAKE')
    num_real = np.sum(dummy_y == 'REAL')
    total_samples_for_weights = num_fake + num_real
    calculated_class_weights = {
        0: total_samples_for_weights / (2.0 * num_fake),
        1: total_samples_for_weights / (2.0 * num_real)
    }
    logger.info(f"Pesos de classe calculados: {calculated_class_weights}")
    architectures_to_test = ["default", "cnn_baseline", "bidirectional_gru", "resnet_gru", "transformer", "rawgat_st"]
    for arch in architectures_to_test:
        logger.info(f"\n--- Testando arquitetura: {arch} ---")
        current_dummy_input_shape = dummy_input_shape
        if arch in ["transformer", "rawgat_st"]:
            current_dummy_input_shape = (dummy_input_shape[0], dummy_input_shape[1])
            if len(dummy_X.shape) == 4 and dummy_X.shape[-1] == 1:
                temp_dummy_X = dummy_X.reshape(dummy_X.shape[0], dummy_X.shape[1], dummy_X.shape[2])
            else:
                temp_dummy_X = dummy_X
            logger.info(f"Ajustando dummy_X para arquitetura '{arch}'. Novo shape: {temp_dummy_X.shape}. Input_shape para modelo: {current_dummy_input_shape}")
        else:
            temp_dummy_X = dummy_X
            logger.info(f"Usando dummy_X original para arquitetura '{arch}'. Shape: {temp_dummy_X.shape}. Input_shape para modelo: {current_dummy_input_shape}")
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
    if os.path.exists(test_model_dir):
        logger.info(f"\nLimpando diretório de teste '{test_model_dir}'...")
        shutil.rmtree(test_model_dir)
        logger.info(f"'{test_model_dir}' removido.")