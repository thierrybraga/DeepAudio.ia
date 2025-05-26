# TrainModel.py - Versão Otimizada
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Tuple, Optional, Any, Dict, Callable, Union
import logging
import os
import math
from datetime import datetime
import shutil
import pickle
import json

# Configura logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [TrainModel] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================ CAMADAS CUSTOMIZADAS OTIMIZADAS ============================

class AudioFeatureNormalization(layers.Layer):
    """Camada de normalização adaptativa otimizada para features de áudio."""

    def __init__(self, axis: int = -1, momentum: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = 1e-6

    def build(self, input_shape):
        feature_dim = input_shape[self.axis] if self.axis != -1 else input_shape[-1]
        shape_for_stats = [1] * len(input_shape)
        actual_axis_index = len(input_shape) + self.axis if self.axis < 0 else self.axis
        shape_for_stats[actual_axis_index] = feature_dim

        self.moving_mean = self.add_weight(
            name='moving_mean', shape=shape_for_stats,
            initializer='zeros', trainable=False
        )
        self.moving_variance = self.add_weight(
            name='moving_variance', shape=shape_for_stats,
            initializer='ones', trainable=False
        )
        self.scale = self.add_weight(
            name='scale', shape=shape_for_stats,
            initializer='ones', trainable=True
        )
        self.offset = self.add_weight(
            name='offset', shape=shape_for_stats,
            initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def adapt(self, data: tf.Tensor):
        """Adapta as estatísticas com os dados de treinamento."""
        if not tf.is_tensor(data):
            data = tf.constant(data, dtype=tf.float32)

        axes_to_reduce = [i for i in range(len(data.shape)) if
                          i != (len(data.shape) + self.axis if self.axis < 0 else self.axis)]
        mean = tf.reduce_mean(data, axis=axes_to_reduce, keepdims=True)
        variance = tf.math.reduce_variance(data, axis=axes_to_reduce, keepdims=True)

        self.moving_mean.assign(mean)
        self.moving_variance.assign(variance)
        logger.info(f"Normalização adaptada - Média: {mean.shape}, Variância: {variance.shape}")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if training:
            # Durante treinamento, calcular estatísticas do batch e atualizar média móvel
            axes_to_reduce = [i for i in range(len(inputs.shape)) if
                              i != (len(inputs.shape) + self.axis if self.axis < 0 else self.axis)]
            batch_mean = tf.reduce_mean(inputs, axis=axes_to_reduce, keepdims=True)
            batch_variance = tf.math.reduce_variance(inputs, axis=axes_to_reduce, keepdims=True)

            # Atualizar média móvel
            self.moving_mean.assign(
                self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean
            )
            self.moving_variance.assign(
                self.momentum * self.moving_variance + (1 - self.momentum) * batch_variance
            )

            normalized = (inputs - batch_mean) / tf.sqrt(batch_variance + self.epsilon)
        else:
            # Durante inferência, usar média móvel
            normalized = (inputs - self.moving_mean) / tf.sqrt(self.moving_variance + self.epsilon)

        return normalized * self.scale + self.offset

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"axis": self.axis, "momentum": self.momentum})
        return config


class MultiHeadSelfAttention(layers.Layer):
    """Camada de Multi-Head Self-Attention otimizada para sequências de áudio."""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model, use_bias=False)
        self.wk = layers.Dense(d_model, use_bias=False)
        self.wv = layers.Dense(d_model, use_bias=False)
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def split_heads(self, x, batch_size):
        """Divide a última dimensão em (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calcula a atenção dot-product escalada."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]

        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        # Add & Norm (conexão residual + normalização)
        return self.layer_norm(inputs + self.dropout(output, training=training))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class TemporalAttention(layers.Layer):
    """Atenção temporal especializada para sequências de áudio."""

    def __init__(self, units: int, return_attention: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention

        self.W_temporal = layers.Dense(units, activation='tanh')
        self.W_context = layers.Dense(1, use_bias=False)
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs, training=None):
        # inputs: (batch, time_steps, features)
        attention_scores = self.W_context(self.W_temporal(inputs))
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Weighted sum with attention
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)

        if self.return_attention:
            return context_vector, attention_weights
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "return_attention": self.return_attention
        })
        return config


class ChannelSpatialAttention(layers.Layer):
    """Atenção canal-espacial para features de áudio 2D."""

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        # Channel attention
        self.channel_attention = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(channels // self.reduction_ratio, activation='relu'),
            layers.Dense(channels, activation='sigmoid'),
            layers.Reshape((1, 1, channels))
        ])

        # Spatial attention
        self.spatial_attention = tf.keras.Sequential([
            layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        ])

        super().build(input_shape)

    def call(self, inputs):
        # Channel attention
        channel_att = self.channel_attention(inputs)
        x = inputs * channel_att

        # Spatial attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_input = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_att = self.spatial_attention(spatial_input)

        return x * spatial_att

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


# ============================ ARQUITETURAS OTIMIZADAS ============================

def create_efficientnet_temporal_model(input_shape: Tuple[int, ...],
                                       num_classes: int = 2,
                                       dropout_rate: float = 0.3) -> models.Model:
    """Cria modelo baseado em EfficientNet com modelagem temporal."""

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Normalização adaptativa
    x = AudioFeatureNormalization(axis=-1, name="audio_norm")(x)

    # Se entrada é 3D (frames, height, width), aplicar EfficientNet frame a frame
    if len(input_shape) == 3:
        # Assumir que é (time_steps, height, width) ou similar
        # Converter para (time_steps, height, width, 1) se necessário
        if input_shape[-1] < 3:  # Não é RGB
            x = tf.expand_dims(x, axis=-1)

        # Aplicar base pré-treinada
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=x.shape[2:]  # Remove batch e time dimensions
        )
        base_model.trainable = False  # Freeze initially

        x = layers.TimeDistributed(base_model)(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    else:
        # Para outras formas, usar camadas convolucionais customizadas
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = tf.expand_dims(x, axis=1)  # Add time dimension

    # Modelagem temporal
    x = layers.LSTM(256, return_sequences=True, dropout=dropout_rate)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=dropout_rate)(x)

    # Multi-head attention
    x = MultiHeadSelfAttention(d_model=128, num_heads=8, dropout_rate=dropout_rate)(x)

    # Final classification
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def create_advanced_cnn_lstm_model(input_shape: Tuple[int, ...],
                                   num_classes: int = 2,
                                   dropout_rate: float = 0.3) -> models.Model:
    """Cria modelo CNN-LSTM avançado com atenção."""

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Normalização adaptativa
    x = AudioFeatureNormalization(axis=-1)(x)

    # Garantir formato 4D para CNN
    if len(input_shape) == 2:
        x = tf.expand_dims(x, axis=-1)
    elif len(input_shape) == 3 and input_shape[-1] > 1:
        x = tf.expand_dims(x, axis=-1)

    # Blocos CNN com atenção
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = ChannelSpatialAttention()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = ChannelSpatialAttention()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = ChannelSpatialAttention()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Preparar para LSTM
    shape_before_lstm = x.shape
    x = layers.Reshape((shape_before_lstm[1],
                        shape_before_lstm[2] * shape_before_lstm[3]))(x)

    # Camadas LSTM bidirecionais
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True,
                                         dropout=dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                         dropout=dropout_rate))(x)

    # Atenção temporal
    x = TemporalAttention(units=256)(x)

    # Classificação final
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def create_transformer_model(input_shape: Tuple[int, ...],
                             num_classes: int = 2,
                             d_model: int = 256,
                             num_heads: int = 8,
                             num_layers: int = 6,
                             dropout_rate: float = 0.1) -> models.Model:
    """Cria modelo Transformer otimizado para áudio."""

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Normalização
    x = AudioFeatureNormalization(axis=-1)(x)

    # Projeção para d_model se necessário
    if input_shape[-1] != d_model:
        x = layers.Dense(d_model)(x)

    # Positional encoding
    seq_len = input_shape[0] if len(input_shape) >= 2 else input_shape[0]
    positions = tf.range(start=0, limit=seq_len, dtype=tf.float32)
    pos_encoding = positional_encoding(seq_len, d_model)
    x += pos_encoding[:seq_len, :]

    x = layers.Dropout(dropout_rate)(x)

    # Transformer encoder layers
    for i in range(num_layers):
        # Multi-head self-attention
        x = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{i}"
        )(x)

        # Feed forward
        ff_output = layers.Dense(d_model * 4, activation="relu")(x)
        ff_output = layers.Dense(d_model)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    """Cria codificação posicional sin/cos."""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    """Calcula os ângulos para codificação posicional."""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# ============================ MELHORIAS NO PIPELINE ============================

class AdvancedAudioAugmenter:
    """Augmentador de dados avançado para áudio."""

    def __init__(self,
                 noise_factor: float = 0.005,
                 time_shift_range: int = 5,
                 freq_mask_prob: float = 0.3,
                 time_mask_prob: float = 0.3):
        self.noise_factor = noise_factor
        self.time_shift_range = time_shift_range
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob

    def __call__(self, X: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        """Aplica augmentação aos dados."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.map(self._augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def _augment_fn(self, audio_features, label):
        """Função de augmentação."""
        # Adicionar ruído gaussiano
        if tf.random.uniform([]) < 0.5:
            noise = tf.random.normal(
                shape=tf.shape(audio_features),
                mean=0.0,
                stddev=self.noise_factor,
                dtype=tf.float32
            )
            audio_features = audio_features + noise

        # Time shifting
        if tf.random.uniform([]) < 0.3:
            shift = tf.random.uniform([],
                                      -self.time_shift_range,
                                      self.time_shift_range,
                                      dtype=tf.int32)
            audio_features = tf.roll(audio_features, shift, axis=0)

        # SpecAugment-like frequency masking
        if tf.random.uniform([]) < self.freq_mask_prob:
            audio_features = self._frequency_mask(audio_features)

        # SpecAugment-like time masking
        if tf.random.uniform([]) < self.time_mask_prob:
            audio_features = self._time_mask(audio_features)

        return audio_features, label

    def _frequency_mask(self, spectrogram):
        """Aplica máscara de frequência."""
        freq_max = tf.shape(spectrogram)[1]
        f = tf.random.uniform([], 0, min(8, freq_max // 4), dtype=tf.int32)
        f0 = tf.random.uniform([], 0, freq_max - f, dtype=tf.int32)

        # Create mask
        mask = tf.ones_like(spectrogram)
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.stack([
                tf.range(tf.shape(spectrogram)[0])[:, None],
                tf.range(f0, f0 + f)[None, :]
            ], axis=-1),
            tf.zeros((tf.shape(spectrogram)[0], f))
        )

        return spectrogram * mask

    def _time_mask(self, spectrogram):
        """Aplica máscara temporal."""
        time_max = tf.shape(spectrogram)[0]
        t = tf.random.uniform([], 0, min(10, time_max // 4), dtype=tf.int32)
        t0 = tf.random.uniform([], 0, time_max - t, dtype=tf.int32)

        # Create mask
        mask = tf.ones_like(spectrogram)
        indices = tf.range(t0, t0 + t)
        updates = tf.zeros((t, tf.shape(spectrogram)[1]))
        mask = tf.tensor_scatter_nd_update(mask, indices[:, None], updates)

        return spectrogram * mask


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Scheduler de learning rate com annealing cosseno."""

    def __init__(self, T_max: int, eta_min: float = 0, verbose: int = 0):
        super().__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.initial_lr = None

    def on_train_begin(self, logs=None):
        self.initial_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        if self.verbose > 0:
            print(f'Epoch {epoch + 1}: CosineAnnealingScheduler setting learning rate to {lr:.6f}.')


class UncertaintyEstimator:
    """Estimador de incerteza usando Monte Carlo Dropout."""

    def __init__(self, model: tf.keras.Model, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediz com estimativa de incerteza."""
        predictions = []

        for _ in range(self.num_samples):
            # Enable dropout during inference
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Epistemic uncertainty (mean of standard deviations)
        epistemic_uncertainty = np.mean(std_pred, axis=-1)

        return mean_pred, epistemic_uncertainty


# ============================ CLASSE PRINCIPAL OTIMIZADA ============================

class OptimizedModelTrainer:
    """Classe otimizada para treinamento de modelos de detecção de deepfake."""

    def __init__(self,
                 model_dir: str,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 patience: int = 15,
                 architecture: str = "advanced_cnn_lstm",
                 log_base_dir: str = "logs",
                 dropout_rate: float = 0.3,
                 l2_reg_strength: float = 0.001,
                 learning_rate: float = 0.001,
                 use_mixed_precision: bool = True,
                 use_cosine_annealing: bool = True,
                 enable_augmentation: bool = True,
                 label_encoder: Optional[LabelEncoder] = None,
                 cross_validation_folds: int = 0):

        self.model_dir = model_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.architecture = architecture
        self.log_base_dir = log_base_dir
        self.dropout_rate = dropout_rate
        self.l2_reg_strength = l2_reg_strength
        self.learning_rate = learning_rate
        self.cross_validation_folds = cross_validation_folds

        # Advanced options
        self.use_mixed_precision = use_mixed_precision
        self.use_cosine_annealing = use_cosine_annealing
        self.enable_augmentation = enable_augmentation
        self.label_encoder = label_encoder

        # Initialize components
        self.model: Optional[tf.keras.Model] = None
        self.uncertainty_estimator: Optional[UncertaintyEstimator] = None
        self.augmenter = AdvancedAudioAugmenter() if enable_augmentation else None

        # Setup directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_base_dir, exist_ok=True)

        # Setup mixed precision
        if self.use_mixed_precision and tf.config.list_physical_devices('GPU'):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")

    def create_model(self) -> tf.keras.Model:
        """Cria o modelo baseado na arquitetura especificada."""

        if self.architecture == "efficientnet_temporal":
            model = create_efficientnet_temporal_model(
                self.input_shape, self.num_classes, self.dropout_rate
            )
        elif self.architecture == "advanced_cnn_lstm":
            model = create_advanced_cnn_lstm_model(
                self.input_shape, self.num_classes, self.dropout_rate
            )
        elif self.architecture == "transformer":
            model = create_transformer_model(
                self.input_shape, self.num_classes, dropout_rate=self.dropout_rate
            )
        else:
            raise ValueError(f"Arquitetura '{self.architecture}' não reconhecida")

        # Compile with optimized settings
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.l2_reg_strength
        )

        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def prepare_callbacks(self, model_path: str) -> List[tf.keras.callbacks.Callback]:
        """Prepara os callbacks otimizados."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                model_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_weights_only=False
            )
        ]

        # Cosine annealing scheduler
        if self.use_cosine_annealing:
            callbacks.append(
                CosineAnnealingScheduler(
                    T_max=self.epochs,
                    eta_min=self.learning_rate * 0.01,
                    verbose=1
                )
            )
        else:
            # Fallback to ReduceLROnPlateau
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.patience // 2,
                    min_lr=1e-7,
                    verbose=1
                )
            )

        # TensorBoard
        log_dir_tensorboard = os.path.join(
            self.log_base_dir, f"tensorboard_{self.timestamp}"
        )
        callbacks.append(
            TensorBoard(
                log_dir=log_dir_tensorboard,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                profile_batch='500,520'
            )
        )

        # CSV Logger
        csv_log_path = os.path.join(
            self.log_base_dir, f"training_log_{self.timestamp}.csv"
        )
        callbacks.append(CSVLogger(csv_log_path, append=True))

        return callbacks

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calcula pesos de classe para dados desbalanceados."""
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        return dict(zip(unique_classes, class_weights))

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Executa validação cruzada estratificada."""
        if self.cross_validation_folds <= 1:
            raise ValueError("cross_validation_folds deve ser > 1 para validação cruzada")

        skf = StratifiedKFold(
            n_splits=self.cross_validation_folds,
            shuffle=True,
            random_state=42
        )

        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'val_loss': []
        }

        logger.info(f"Iniciando validação cruzada com {self.cross_validation_folds} folds")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Treinando fold {fold + 1}/{self.cross_validation_folds}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Criar modelo para este fold
            fold_model = self.create_model()

            # Callbacks para este fold
            fold_model_path = os.path.join(
                self.model_dir,
                f"model_fold_{fold}_{self.timestamp}.h5"
            )
            fold_callbacks = self.prepare_callbacks(fold_model_path)

            # Preparar dados
            if self.augmenter:
                train_dataset = self.augmenter(X_train_fold, y_train_fold)
                train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold))
                val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
                fit_data = {"x": train_dataset, "validation_data": val_dataset}
            else:
                fit_data = {
                    "x": X_train_fold,
                    "y": y_train_fold,
                    "validation_data": (X_val_fold, y_val_fold)
                }

            # Pesos de classe
            class_weights = self.compute_class_weights(y_train_fold)
            fit_data["class_weight"] = class_weights

            # Treinar
            history = fold_model.fit(
                **fit_data,
                epochs=self.epochs,
                callbacks=fold_callbacks,
                verbose=0
            )

            # Avaliar
            val_metrics = fold_model.evaluate(X_val_fold, y_val_fold, verbose=0)
            metric_names = fold_model.metrics_names

            for i, metric_name in enumerate(metric_names):
                if metric_name in fold_scores:
                    fold_scores[metric_name].append(val_metrics[i])

        # Calcular estatísticas finais
        cv_results = {}
        for metric, scores in fold_scores.items():
            cv_results[f"{metric}_mean"] = np.mean(scores)
            cv_results[f"{metric}_std"] = np.std(scores)

        logger.info("Resultados da validação cruzada:")
        for metric, value in cv_results.items():
            logger.info(f"  {metric}: {value:.4f}")

        return cv_results

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Optional[tf.keras.callbacks.History]:
        """Treina o modelo com todas as otimizações."""
        logger.info(f"Iniciando treinamento otimizado - Arquitetura: {self.architecture}")
        logger.info(f"Shape dos dados: X={X.shape}, y={y.shape}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Augmentação: {self.enable_augmentation}")

        # Validações básicas
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Dados vazios fornecidos")
            return None

        # Processar labels
        y_processed = self._process_labels(y)
        if y_processed is None:
            return None

        # Validação cruzada se solicitada
        if self.cross_validation_folds > 1:
            cv_results = self.cross_validate(X, y_processed)
            self._save_cv_results(cv_results)

        # Dividir dados
        X_train, X_val, y_train, y_val = self._split_data(X, y_processed)
        if X_train is None:
            return None

        # Criar modelo
        self.model = self.create_model()
        logger.info("Resumo do modelo:")
        self.model.summary(print_fn=lambda x: logger.info(x))

        # Adaptar normalização
        self._adapt_normalization_layer(X_train)

        # Preparar callbacks
        model_path = os.path.join(
            self.model_dir,
            f"deepfake_detector_{self.architecture}_{self.timestamp}.h5"
        )
        callbacks = self.prepare_callbacks(model_path)

        # Preparar dados de treinamento
        fit_kwargs = self._prepare_training_data(X_train, y_train, X_val, y_val)

        # Treinar
        try:
            logger.info(f"Iniciando treinamento por {self.epochs} épocas...")
            history = self.model.fit(
                **fit_kwargs,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Treinamento concluído com sucesso")
            logger.info(f"Modelo salvo em: {model_path}")

            # Salvar componentes adicionais
            self._save_additional_components(model_path)

            # Inicializar estimador de incerteza
            self.uncertainty_estimator = UncertaintyEstimator(self.model)

            return history

        except Exception as e:
            logger.exception(f"Erro durante o treinamento: {e}")
            return None

    def _process_labels(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Processa e codifica labels."""
        if y.dtype == object:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                try:
                    y_encoded = self.label_encoder.fit_transform(y)
                except Exception as e:
                    logger.error(f"Erro ao codificar labels: {e}")
                    return None
            else:
                try:
                    y_encoded = self.label_encoder.transform(y)
                except Exception as e:
                    logger.error(f"Erro ao transformar labels: {e}")
                    return None

            if len(self.label_encoder.classes_) < 2:
                logger.error("Menos de 2 classes encontradas")
                return None

            logger.info(f"Classes: {self.label_encoder.classes_}")
            return y_encoded
        else:
            if len(np.unique(y)) < 2:
                logger.error("Menos de 2 classes numéricas encontradas")
                return None
            return y

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        """Divide os dados em treino e validação."""
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.validation_split,
                random_state=42,
                stratify=y
            )
            logger.info(f"Dados divididos - Treino: {X_train.shape}, Validação: {X_val.shape}")
            return X_train, X_val, y_train, y_val
        except ValueError as e:
            logger.error(f"Erro na divisão dos dados: {e}")
            try:
                # Fallback sem estratificação
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.validation_split, random_state=42
                )
                logger.warning("Divisão sem estratificação aplicada")
                return X_train, X_val, y_train, y_val
            except Exception as e2:
                logger.error(f"Erro na divisão sem estratificação: {e2}")
                return None, None, None, None

    def _adapt_normalization_layer(self, X_train: np.ndarray):
        """Adapta a camada de normalização."""
        for layer in self.model.layers:
            if isinstance(layer, AudioFeatureNormalization):
                logger.info("Adaptando camada de normalização...")
                layer.adapt(X_train)
                logger.info("Normalização adaptada com sucesso")
                break
        else:
            logger.warning("Camada AudioFeatureNormalization não encontrada")

    def _prepare_training_data(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Prepara os dados para treinamento."""
        if self.augmenter:
            logger.info("Aplicando augmentação de dados...")
            train_dataset = self.augmenter(X_train, y_train)
            train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            fit_kwargs = {"x": train_dataset, "validation_data": val_dataset}
        else:
            fit_kwargs = {
                "x": X_train,
                "y": y_train,
                "validation_data": (X_val, y_val)
            }

        # Adicionar pesos de classe
        class_weights = self.compute_class_weights(y_train)
        fit_kwargs["class_weight"] = class_weights
        logger.info(f"Pesos de classe: {class_weights}")

        return fit_kwargs

    def _save_additional_components(self, model_path: str):
        """Salva componentes adicionais (label encoder, config, etc.)."""
        base_path = model_path.replace('.h5', '')

        # Salvar label encoder
        if self.label_encoder is not None:
            encoder_path = f"{base_path}_label_encoder.pkl"
            try:
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                logger.info(f"Label encoder salvo em: {encoder_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar label encoder: {e}")

        # Salvar configuração do modelo
        config = {
            'architecture': self.architecture,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'use_mixed_precision': self.use_mixed_precision,
            'timestamp': self.timestamp
        }

        config_path = f"{base_path}_config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuração salva em: {config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")

    def _save_cv_results(self, cv_results: Dict[str, float]):
        """Salva resultados da validação cruzada."""
        cv_path = os.path.join(
            self.log_base_dir,
            f"cv_results_{self.timestamp}.json"
        )
        try:
            with open(cv_path, 'w') as f:
                json.dump(cv_results, f, indent=2)
            logger.info(f"Resultados CV salvos em: {cv_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar resultados CV: {e}")

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediz com estimativa de incerteza."""
        if self.uncertainty_estimator is None:
            logger.warning("Estimador de incerteza não inicializado. Usando predição padrão.")
            predictions = self.model.predict(X)
            uncertainty = np.zeros(predictions.shape[0])
            return predictions, uncertainty

        return self.uncertainty_estimator.predict_with_uncertainty(X)

    def get_label_encoder(self) -> Optional[LabelEncoder]:
        """Retorna o label encoder."""
        return self.label_encoder


# ============================ ENSEMBLE DE MODELOS ============================

class ModelEnsemble:
    """Ensemble de múltiplos modelos para maior robustez."""

    def __init__(self, models: List[tf.keras.Model], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Número de pesos deve ser igual ao número de modelos")

        # Normalizar pesos
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predição por ensemble com voto ponderado."""
        predictions = []

        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)

        # Média ponderada
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predição com medida de confiança baseada no acordo entre modelos."""
        predictions = []

        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Média ponderada
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        # Confiança baseada no desvio padrão entre modelos
        agreement = np.std(predictions, axis=0)
        confidence = 1 - np.mean(agreement, axis=-1)

        return ensemble_pred, confidence


# ============================ FUNÇÃO DE TESTE OTIMIZADA ============================

def run_optimized_test():
    """Executa teste das funcionalidades otimizadas."""
    logger.info("=== TESTE DAS OTIMIZAÇÕES ===")

    # Dados dummy mais realistas
    dummy_input_shape = (128, 64, 1)  # (time_steps, features, channels)
    num_samples = 500

    # Criar dados sintéticos
    np.random.seed(42)
    dummy_X = np.random.randn(num_samples, *dummy_input_shape).astype(np.float32)

    # Labels desbalanceadas (70% REAL, 30% FAKE)
    dummy_y = np.array(['REAL'] * int(num_samples * 0.7) +
                       ['FAKE'] * int(num_samples * 0.3))
    np.random.shuffle(dummy_y)

    logger.info(f"Dados de teste criados: X={dummy_X.shape}, y={dummy_y.shape}")
    logger.info(f"Distribuição: {np.unique(dummy_y, return_counts=True)}")

    # Preparar label encoder
    label_encoder = LabelEncoder()
    dummy_y_encoded = label_encoder.fit_transform(dummy_y)
    num_classes = len(label_encoder.classes_)

    # Diretório de teste
    test_dir = "test_optimized_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Testar arquiteturas otimizadas
    architectures = ["advanced_cnn_lstm", "transformer"]

    for arch in architectures:
        logger.info(f"\n--- Testando arquitetura otimizada: {arch} ---")

        try:
            trainer = OptimizedModelTrainer(
                model_dir=test_dir,
                input_shape=dummy_input_shape,
                num_classes=num_classes,
                epochs=3,  # Reduzido para teste
                batch_size=32,
                architecture=arch,
                use_mixed_precision=True,
                use_cosine_annealing=True,
                enable_augmentation=True,
                label_encoder=label_encoder,
                cross_validation_folds=3  # Teste de CV
            )

            # Executar validação cruzada
            cv_results = trainer.cross_validate(dummy_X, dummy_y_encoded)
            logger.info(f"CV concluída para {arch}: {cv_results}")

            # Treinar modelo final
            history = trainer.train_model(dummy_X, dummy_y_encoded)

            if history:
                logger.info(f"✅ {arch} treinado com sucesso")

                # Testar predição com incerteza
                test_sample = dummy_X[:5]
                predictions, uncertainty = trainer.predict_with_uncertainty(test_sample)
                logger.info(f"Predições: {predictions.shape}, Incerteza: {uncertainty.shape}")

            else:
                logger.error(f"❌ Falha no treinamento de {arch}")

        except Exception as e:
            logger.exception(f"❌ Erro em {arch}: {e}")

    # Limpeza
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    logger.info("=== TESTE CONCLUÍDO ===")


# ============================ CAMADAS DE COMPATIBILIDADE ============================
# Mantidas para compatibilidade com Predictor.py e código existente

class AttentionLayer(layers.Layer):
    """
    Camada de atenção legada para compatibilidade com Predictor.py.
    Implementa uma atenção do tipo Bahdanau (additive attention).
    NOTA: Esta é mantida apenas para compatibilidade. Use MultiHeadSelfAttention para novos projetos.
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


def create_model(input_shape, num_classes=2, architecture="default",
                 dropout_rate=0.3, l2_reg_strength=0.001):
    """
    Função create_model legada para compatibilidade.
    NOTA: Use OptimizedModelTrainer.create_model() para novos projetos.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    # Helper function for reshaping for CNN
    def _reshape_for_cnn(tensor, target_shape):
        if len(tensor.shape) == 3:  # (batch, frames, features_dim)
            return layers.Reshape((target_shape[0], target_shape[1], 1))(tensor)
        elif len(tensor.shape) == 4 and tensor.shape[-1] != 1:
            return layers.Lambda(lambda y: y[..., :1])(tensor)
        elif len(tensor.shape) == 4 and tensor.shape[-1] == 1:
            return tensor
        else:
            raise ValueError(f"Unexpected input tensor shape for CNN: {tensor.shape}")

    if architecture == "default":
        # Ensure 4D input for CNN
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
        x = layers.Reshape((shape_before_gru[1],
                            shape_before_gru[2] * shape_before_gru[3]),
                           name="reshape_for_gru")(x)

        if tf.config.list_physical_devices('GPU'):
            x = layers.GRU(128, return_sequences=True, name="gru1")(x)
        else:
            x = layers.GRU(128, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate, name="gru1")(x)
            x = layers.GRU(64, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate, name="gru2")(x)

        x = AttentionLayer(name="attention_layer")(x)

    # Outras arquiteturas podem ser adicionadas aqui conforme necessário
    else:
        # Para compatibilidade, usar a arquitetura avançada como fallback
        logger.warning(f"Arquitetura '{architecture}' mapeada para 'advanced_cnn_lstm'")
        return create_advanced_cnn_lstm_model(input_shape, num_classes, dropout_rate)

    # Camadas densas finais
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_strength), name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.5, name="dropout_final")(x)

    output_tensor = layers.Dense(num_classes, activation='softmax', name="output_layer")(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


class ModelTrainer:
    """
    Classe legada para compatibilidade com código existente.
    NOTA: Use OptimizedModelTrainer para novos projetos.
    """

    def __init__(self, model_dir, input_shape, num_classes, epochs=50, batch_size=32,
                 validation_split=0.2, patience=10, use_plateau=True, architecture="default",
                 log_base_dir="logs", dropout_rate=0.3, l2_reg_strength=0.001,
                 learning_rate=0.001, data_augmenter=None, class_weights=None,
                 label_encoder=None):

        self.model_dir = model_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_filename = f"deepfake_detector_model_{self.timestamp}.h5"
        self.model_path = os.path.join(model_dir, self.model_filename)

        # Parâmetros de treinamento
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
        self.model = None
        self.label_encoder = label_encoder

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_base_dir, exist_ok=True)

    def train_model(self, X, y):
        """Treina o modelo usando a implementação legada."""
        logger.info(f"Iniciando treinamento do modelo com arquitetura: {self.architecture}")
        logger.info(f"Shape dos dados de entrada (X): {X.shape}, Shape das labels (y): {y.shape}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Dados de treinamento ou labels estão vazios. Treinamento abortado.")
            return None

        # Encode labels if they are strings
        if y.dtype == object:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                try:
                    y_encoded = self.label_encoder.fit_transform(y)
                except Exception as e:
                    logger.error(f"Erro ao codificar labels: {e}")
                    return None
            else:
                try:
                    y_encoded = self.label_encoder.transform(y)
                except Exception as e:
                    logger.error(f"Erro ao transformar labels: {e}")
                    return None

            if len(self.label_encoder.classes_) < 2:
                logger.error("Menos de duas classes únicas encontradas")
                return None
            logger.info(f"Labels originais: {self.label_encoder.classes_}")
        else:
            y_encoded = y
            if len(np.unique(y_encoded)) < 2:
                logger.error("Menos de duas classes únicas encontradas")
                return None

        # Split data
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split, random_state=42, stratify=y_encoded
            )
        except ValueError as e:
            logger.warning("Tentando divisão sem estratificação")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split, random_state=42
            )

        logger.info(f"Dados divididos: Treino={X_train.shape}, Validação={X_val.shape}")

        # Create model
        self.model = create_model(self.input_shape, self.num_classes, self.architecture,
                                  self.dropout_rate, self.l2_reg_strength)

        # Adapt normalization layer
        for layer in self.model.layers:
            if isinstance(layer, AudioFeatureNormalization):
                logger.info("Adaptando a camada AudioFeatureNormalization...")
                layer.adapt(X_train)
                break

        # Configure optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        ]

        if self.use_plateau:
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.patience / 2),
                                  min_lr=0.00001, verbose=1)
            )

        # Prepare data for training
        if self.data_augmenter:
            logger.info("Aplicando aumento de dados...")
            train_dataset = self.data_augmenter(X_train, y_train).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.batch_size).prefetch(
                tf.data.AUTOTUNE)
            fit_kwargs = {"x": train_dataset, "validation_data": val_dataset}
        else:
            fit_kwargs = {"x": X_train, "y": y_train, "validation_data": (X_val, y_val)}

        # Add class weights if provided
        if self.class_weights:
            fit_kwargs["class_weight"] = self.class_weights

        # Train
        try:
            history = self.model.fit(
                **fit_kwargs,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Treinamento concluído com sucesso")
            logger.info(f"Modelo salvo em: {self.model_path}")

            # Salvar label encoder
            if self.label_encoder is not None:
                encoder_filename = f"label_encoder_{self.timestamp}.pkl"
                encoder_path = os.path.join(self.model_dir, encoder_filename)
                try:
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(self.label_encoder, f)
                    logger.info(f"LabelEncoder salvo em: {encoder_path}")
                except Exception as e:
                    logger.error(f"Erro ao salvar LabelEncoder: {e}")

            return history

        except Exception as e:
            logger.exception(f"Erro durante o treinamento: {e}")
            return None

    def get_label_encoder(self):
        """Retorna o label encoder."""
        return self.label_encoder


def simple_audio_augmenter(X_train: np.ndarray, y_train: np.ndarray) -> tf.data.Dataset:
    """
    Função de augmentação legada para compatibilidade.
    """

    def _augment(audio_features, label):
        noise = tf.random.normal(shape=tf.shape(audio_features), mean=0.0, stddev=0.01, dtype=tf.float32)
        augmented_audio_features = audio_features + noise
        return augmented_audio_features, label

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# ============================ EXECUÇÃO PRINCIPAL ============================

def run_compatibility_test():
    """Testa a compatibilidade com o código legado."""
    logger.info("=== TESTE DE COMPATIBILIDADE ===")

    # Dados de teste
    dummy_input_shape = (100, 40, 1)
    num_samples = 100

    dummy_X = np.random.rand(num_samples, *dummy_input_shape).astype(np.float32)
    dummy_y = np.array(['REAL'] * 50 + ['FAKE'] * 50)

    # Testar classe legada
    trainer = ModelTrainer(
        model_dir="test_legacy",
        input_shape=dummy_input_shape,
        num_classes=2,
        epochs=2,
        batch_size=16,
        architecture="default"
    )

    try:
        history = trainer.train_model(dummy_X, dummy_y)
        if history:
            logger.info("✅ Compatibilidade com ModelTrainer legada OK")
        else:
            logger.error("❌ Falha na compatibilidade")
    except Exception as e:
        logger.exception(f"❌ Erro na compatibilidade: {e}")

    # Limpeza
    if os.path.exists("test_legacy"):
        shutil.rmtree("test_legacy")


def run_optimized_test():
    """Executa teste das funcionalidades otimizadas."""
    logger.info("=== TESTE DAS OTIMIZAÇÕES ===")

    # Dados dummy mais realistas
    dummy_input_shape = (128, 64, 1)  # (time_steps, features, channels)
    num_samples = 500

    # Criar dados sintéticos
    np.random.seed(42)
    dummy_X = np.random.randn(num_samples, *dummy_input_shape).astype(np.float32)

    # Labels desbalanceadas (70% REAL, 30% FAKE)
    dummy_y = np.array(['REAL'] * int(num_samples * 0.7) +
                       ['FAKE'] * int(num_samples * 0.3))
    np.random.shuffle(dummy_y)

    logger.info(f"Dados de teste criados: X={dummy_X.shape}, y={dummy_y.shape}")
    logger.info(f"Distribuição: {np.unique(dummy_y, return_counts=True)}")

    # Preparar label encoder
    label_encoder = LabelEncoder()
    dummy_y_encoded = label_encoder.fit_transform(dummy_y)
    num_classes = len(label_encoder.classes_)

    # Diretório de teste
    test_dir = "test_optimized_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Testar arquiteturas otimizadas
    architectures = ["advanced_cnn_lstm", "transformer"]

    for arch in architectures:
        logger.info(f"\n--- Testando arquitetura otimizada: {arch} ---")

        try:
            trainer = OptimizedModelTrainer(
                model_dir=test_dir,
                input_shape=dummy_input_shape,
                num_classes=num_classes,
                epochs=3,  # Reduzido para teste
                batch_size=32,
                architecture=arch,
                use_mixed_precision=True,
                use_cosine_annealing=True,
                enable_augmentation=True,
                label_encoder=label_encoder,
                cross_validation_folds=3  # Teste de CV
            )

            # Executar validação cruzada
            cv_results = trainer.cross_validate(dummy_X, dummy_y_encoded)
            logger.info(f"CV concluída para {arch}: {cv_results}")

            # Treinar modelo final
            history = trainer.train_model(dummy_X, dummy_y_encoded)

            if history:
                logger.info(f"✅ {arch} treinado com sucesso")

                # Testar predição com incerteza
                test_sample = dummy_X[:5]
                predictions, uncertainty = trainer.predict_with_uncertainty(test_sample)
                logger.info(f"Predições: {predictions.shape}, Incerteza: {uncertainty.shape}")

            else:
                logger.error(f"❌ Falha no treinamento de {arch}")

        except Exception as e:
            logger.exception(f"❌ Erro em {arch}: {e}")

    # Limpeza
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    logger.info("=== TESTE CONCLUÍDO ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Testar compatibilidade primeiro
    run_compatibility_test()

    # Depois testar otimizações
    run_optimized_test()

# ============================ EXPORTS PARA COMPATIBILIDADE ============================

# Garantir que todas as classes necessárias estejam disponíveis para importação
__all__ = [
    'AudioFeatureNormalization',
    'AttentionLayer',  # Camada legada
    'MultiHeadSelfAttention',  # Nova camada otimizada
    'TemporalAttention',
    'ChannelSpatialAttention',
    'create_model',  # Função legada
    'ModelTrainer',  # Classe legada
    'OptimizedModelTrainer',  # Classe otimizada
    'simple_audio_augmenter',
    'ModelEnsemble',
    'UncertaintyEstimator',
    'AdvancedAudioAugmenter',
    'CosineAnnealingScheduler'
]