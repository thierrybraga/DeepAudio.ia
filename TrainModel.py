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
from typing import List, Tuple, Optional, Any, Dict
import logging
import os
from datetime import datetime
import shutil # Importado para o bloco de limpeza do __main__

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
        self.epsilon = 1e-6  # Para evitar divisão por zero
        # self.mean e self.variance serão criados em `build`

    def build(self, input_shape):
        # input_shape pode ser (None, frames, features_dim, 1) ou (None, frames, features_dim)
        # O último eixo é sempre o feature_dim para normalização.
        feature_dim = input_shape[self.axis] if self.axis != -1 else input_shape[-1]

        # Cria a forma para broadcasting. Por exemplo, para input_shape=(None, 100, 40, 1)
        # e axis=-1, shape_for_stats será (1, 1, 1, 40)
        shape_for_stats = [1] * len(input_shape)
        # Determina a posição real do eixo, lidando com eixos negativos
        actual_axis_index = len(input_shape) + self.axis if self.axis < 0 else self.axis
        shape_for_stats[actual_axis_index] = feature_dim

        self.mean = self.add_weight(
            name='mean',
            shape=shape_for_stats,
            initializer='zeros',  # Inicializa com zeros
            trainable=False
        )
        self.variance = self.add_weight(
            name='variance',
            shape=shape_for_stats,
            initializer='ones',   # Inicializa com uns
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

        # Calcula a média e variância sobre todos os eixos, exceto o eixo de feature.
        # Por exemplo, para (batch, frames, features, 1), reduz sobre (batch, frames, 1)
        # para obter estatísticas por feature para cada canal (se houver).
        axes_to_reduce = [i for i in range(len(data.shape)) if
                          i != (len(data.shape) + self.axis if self.axis < 0 else self.axis)]
        mean = tf.reduce_mean(data, axis=axes_to_reduce, keepdims=True)
        variance = tf.math.reduce_variance(data, axis=axes_to_reduce, keepdims=True)

        # Atribui os valores calculados aos pesos da camada
        self.mean.assign(mean)
        self.variance.assign(variance)
        logger.info(
            f"Camada de Normalização adaptada. Média shape: {self.mean.shape}, Variância shape: {self.variance.shape}")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Normaliza os inputs usando a média e variância aprendidas.
        Assume que a camada já foi adaptada (os pesos `self.mean` e `self.variance`
        já contêm os valores calculados ou carregados de um modelo salvo).
        """
        # A normalização será feita usando os valores atuais de self.mean e self.variance.
        # Estes serão os valores iniciais (0, 1) antes da adaptação, e os valores aprendidos
        # depois que `adapt` é chamado no `ModelTrainer`.
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


# ============================ FUNÇÕES DE CONSTRUÇÃO DE MODELOS ============================
def create_model(input_shape: Tuple[int, ...], num_classes: int = 2, architecture: str = "default",
                 dropout_rate: float = 0.3) -> models.Model:
    """
    Cria e compila um modelo Keras baseado na arquitetura especificada.

    Args:
        input_shape: A forma dos dados de entrada (e.g., (frames, features_dim, 1) para CNN).
        num_classes: Número de classes de saída (padrão é 2 para REAL/FAKE).
        architecture: O tipo de arquitetura do modelo a ser construído ("default" ou "cnn_baseline").
        dropout_rate: Taxa de dropout a ser aplicada em algumas camadas.

    Returns:
        Um modelo Keras compilado.
    """
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    # Normalização dos recursos de áudio
    # A camada é instanciada e chamada aqui, mas seus pesos `mean` e `variance`
    # serão adaptados (calculados) depois, no ModelTrainer.train_model, antes do `model.fit`.
    x = AudioFeatureNormalization(axis=-1, name="audio_norm_layer")(x)

    if architecture == "default":
        # Garante que a entrada para a CNN seja 4D (batch, height, width, channels)
        if len(input_shape) == 2: # Se a entrada é (frames, features_dim)
            x = layers.Reshape((input_shape[0], input_shape[1], 1), name="reshape_for_cnn")(x)
        elif len(input_shape) == 3 and input_shape[-1] != 1: # Se a entrada é (frames, features_dim, channels) mas channels > 1
            logger.warning(f"Input shape inesperado para 'default' architecture: {input_shape}. Esperado (frames, features_dim, 1). ")
            # Reduz para um único canal se houver mais de um, ou apenas continua se for o caso
            x = layers.Lambda(lambda y: y[..., :1])(x)
        elif len(input_shape) != 3 or input_shape[-1] != 1: # Outros casos inesperados
            logger.warning(f"Input shape inesperado para 'default' architecture: {input_shape}. Não é possível garantir o formato da CNN.")


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
            x = layers.GRU(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate,
                           name="gru1")(x)

        x = AttentionLayer(name="attention_layer")(x)

    elif architecture == "cnn_baseline":
        # Garante que a entrada para a CNN seja 4D (batch, height, width, channels)
        if len(input_shape) == 2: # Se a entrada é (frames, features_dim)
            x = layers.Reshape((input_shape[0], input_shape[1], 1), name="reshape_for_cnn_baseline")(x)
        elif len(input_shape) == 3 and input_shape[-1] != 1: # Se a entrada é (frames, features_dim, channels) mas channels > 1
            logger.warning(f"Input shape inesperado para 'cnn_baseline' architecture: {input_shape}. Esperado (frames, features_dim, 1).")
            x = layers.Lambda(lambda y: y[..., :1])(x)
        elif len(input_shape) != 3 or input_shape[-1] != 1: # Outros casos inesperados
            logger.warning(f"Input shape inesperado para 'cnn_baseline' architecture: {input_shape}. Não é possível garantir o formato da CNN.")

        x = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name="conv_b1")(x)
        x = layers.BatchNormalization(name="bn_b1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool_b1")(x)
        x = layers.Dropout(dropout_rate, name="dropout_b1")(x)

        x = layers.Conv2D(64, (5, 5), activation='relu', padding='same', name="conv_b2")(x)
        x = layers.BatchNormalization(name="bn_b2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool_b2")(x)
        x = layers.Dropout(dropout_rate, name="dropout_b2")(x)

        x = layers.Flatten(name="flatten")(x)

    else:
        raise ValueError(f"Arquitetura '{architecture}' não reconhecida. Escolha 'default' ou 'cnn_baseline'.")

    # Camadas densas comuns a ambas as arquiteturas
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="dense1")(x)
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
                 log_base_dir: str = "logs", # Alterado para log_base_dir
                 dropout_rate: float = 0.3):
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
            architecture: Arquitetura do modelo ("default" ou "cnn_baseline").
            log_base_dir: Diretório base para salvar logs do TensorBoard e CSV.
            dropout_rate: Taxa de dropout para as camadas do modelo.
        """
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Gerar timestamp uma única vez na inicialização
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
        self.log_base_dir = log_base_dir # Usar log_base_dir aqui
        self.dropout_rate = dropout_rate
        self.model: Optional[models.Model] = None
        self.label_encoder: Optional[LabelEncoder] = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_base_dir, exist_ok=True) # Criar o diretório base de logs

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Optional[tf.keras.callbacks.History]:
        """
        Treina o modelo com os dados fornecidos.

        Args:
            X: Features de entrada (numpy array de forma (num_samples, frames, features_dim)).
            y: Labels de saída (numpy array ou lista de strings).

        Returns:
            Um objeto History contendo os dados de treinamento ou None em caso de erro.
        """
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
                logger.error("A divisão dos dados resultou em conjuntos de treinamento ou validação vazios. Treinamento abortado.")
                return None

        logger.info(f"Dados divididos: Treino={X_train.shape}, Validação={X_val.shape}")

        self.model = create_model(self.input_shape, self.num_classes, self.architecture, self.dropout_rate)
        logger.info("Resumo do Modelo:")
        self.model.summary(print_fn=lambda x: logger.info(x))

        norm_layer = None
        for layer in self.model.layers:
            if isinstance(layer, AudioFeatureNormalization):
                norm_layer = layer
                break

        if norm_layer:
            logger.info("Adaptando a camada AudioFeatureNormalization com dados de treinamento...")
            # A adaptação deve ser feita AQUI, antes do model.fit()
            norm_layer.adapt(X_train)
            logger.info("Camada AudioFeatureNormalization adaptada com sucesso.")
        else:
            logger.warning(
                "Camada AudioFeatureNormalization não encontrada no modelo. Certifique-se de que ela está incluída na função create_model.")

        callbacks: List[tf.keras.callbacks.Callback] = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        ]

        if self.use_plateau:
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.patience / 2), min_lr=0.00001,
                                  verbose=1))

        # Usando self.timestamp para o diretório de logs específicos desta execução
        log_dir_tensorboard = os.path.join(self.log_base_dir, f"tensorboard_{self.timestamp}")
        callbacks.append(TensorBoard(log_dir=log_dir_tensorboard, histogram_freq=1))

        csv_log_path = os.path.join(self.log_base_dir, f"training_log_{self.timestamp}.csv")
        callbacks.append(CSVLogger(csv_log_path, append=True))

        logger.info(f"Iniciando o treinamento do modelo por {self.epochs} épocas com batch_size={self.batch_size}...")
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
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


# Exemplo de uso (apenas para teste direto do arquivo)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando teste de TrainModel.py...")

    dummy_input_shape = (100, 40, 1)  # (frames, features, channels)
    num_samples = 200

    dummy_X = np.random.rand(num_samples, dummy_input_shape[0], dummy_input_shape[1], dummy_input_shape[2]).astype(np.float32)

    dummy_y = np.array(['REAL'] * (num_samples // 2) + ['FAKE'] * (num_samples // 2))
    np.random.shuffle(dummy_y)

    print(f"Dummy X shape: {dummy_X.shape}")
    print(f"Dummy y shape: {dummy_y.shape}")

    # Definir um diretório de teste específico para o Main
    test_model_dir = "test_models_refactored"
    test_log_dir = os.path.join(test_model_dir, "logs") # Logs dentro do diretório de teste

    # Limpar diretório de teste antes de iniciar, para garantir um ambiente limpo
    if os.path.exists(test_model_dir):
        logger.info(f"Limpando diretório de teste existente '{test_model_dir}'...")
        shutil.rmtree(test_model_dir)

    trainer = ModelTrainer(
        model_dir=test_model_dir,
        input_shape=dummy_input_shape,
        num_classes=len(np.unique(dummy_y)),
        epochs=5,
        batch_size=32,
        patience=2,
        use_plateau=True,
        architecture="default",
        log_base_dir=test_log_dir, # Passar o diretório de logs para o Trainer
        dropout_rate=0.3
    )

    try:
        history = trainer.train_model(dummy_X, dummy_y)
        if history:
            print("\nTreinamento de teste concluído.")
            if 'val_loss' in history.history:
                print(f"Melhor perda de validação: {min(history.history['val_loss']):.4f}")
            if 'val_accuracy' in history.history:
                print(f"Melhor acurácia de validação: {max(history.history['val_accuracy']):.4f}")

            model_saved_path = trainer.model_path
            if os.path.exists(model_saved_path):
                print(f"Modelo de teste salvo em {model_saved_path}")
            else:
                print(f"Modelo de teste NÃO foi salvo em {model_saved_path} (pode ser devido ao EarlyStopping ou erro).")
        else:
            print("Treinamento de teste não foi concluído com sucesso.")

    except Exception as e:
        logger.exception("Erro durante o treinamento de teste.")
    finally:
        # Limpa o diretório de teste após o término, independentemente do sucesso
        if os.path.exists(test_model_dir):
            logger.info(f"Limpando diretório de teste '{test_model_dir}'...")
            shutil.rmtree(test_model_dir)
            logger.info(f"'{test_model_dir}' removido.")