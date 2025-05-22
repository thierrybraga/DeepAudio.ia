from __future__ import annotations

import logging
import os
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
import pickle  # Para salvar/carregar LabelEncoder

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory, session
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename

# Importa as classes refatoradas
from Predictor import ModelPredictor, sliding_window_predict
from TrainModel import ModelTrainer
from Voice2data import AudioPreprocessor

# ============================ CONFIGURAÇÕES GLOBAIS ============================
CONFIG: Dict[str, Any] = {
    "MAX_UPLOAD_SIZE": int(os.getenv("MAX_UPLOAD_SIZE", 500 * 1024 * 1024)),  # 500 MB
    "ALLOWED_DATASET_EXT": {"zip"},
    "ALLOWED_AUDIO_EXT": {"wav", "mp3", "flac", "wma", "ogg"},
    "MIN_AUDIO_DURATION": float(os.getenv("MIN_AUDIO_DURATION", 0.5)),
    "EXPECTED_FRAMES": int(os.getenv("EXPECTED_FRAMES", 100)),  # Definir aqui o número de frames fixo
    "SAMPLE_RATE": int(os.getenv("SAMPLE_RATE", 16000)),
    "FRAME_LENGTH_MS": float(os.getenv("FRAME_LENGTH_MS", 25.0)),
    "FRAME_SHIFT_MS": float(os.getenv("FRAME_SHIFT_MS", 10.0)),
    "N_MFCC": int(os.getenv("N_MFCC", 40)),  # N_MFCC ou N_MELS definem a feature_dim
    "N_MELS": int(os.getenv("N_MELS", 40)),
    "N_FFT": int(os.getenv("N_FFT", 512)),
    "VAD_ENERGY_THRESH": float(os.getenv("VAD_ENERGY_THRESH", -40.0)),
    "MIN_SEGMENT_DURATION": float(os.getenv("MIN_SEGMENT_DURATION", 0.5)),
    "FEATURE_TYPE": os.getenv("FEATURE_TYPE", "mfcc"),  # 'mfcc' ou 'mel_spectrogram'
    "TRAINING_EPOCHS": int(os.getenv("TRAINING_EPOCHS", 50)),
    "TRAINING_BATCH_SIZE": int(os.getenv("TRAINING_BATCH_SIZE", 32)),
    "TRAINING_PATIENCE": int(os.getenv("TRAINING_PATIENCE", 10)),
    "TRAINING_USE_PLATEAU": os.getenv("TRAINING_USE_PLATEAU", "True").lower() in ('true', '1', 't'),
    "TRAINING_ARCHITECTURE": os.getenv("TRAINING_ARCHITECTURE", "default"),
    "PREDICTION_THRESHOLD": float(os.getenv("PREDICTION_THRESHOLD", 0.5)),
    "PREDICTION_WINDOW_FRAMES": int(os.getenv("PREDICTION_WINDOW_FRAMES", 100)),
    "PREDICTION_HOP_FRAMES": int(os.getenv("PREDICTION_HOP_FRAMES", 50)),
    "USE_SLIDING_WINDOW_FOR_LONG_AUDIO": os.getenv("USE_SLIDING_WINDOW_FOR_LONG_AUDIO", "True").lower() in (
        'true', '1', 't'),
    "VERBOSE": os.getenv("VERBOSE", "False").lower() in ('true', '1', 't')
}

# Caminhos de diretórios e arquivos
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
DATASET_FOLDER = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "model_artifacts"  # Diretório para salvar modelos e encoders
MODEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# ============================ FUNÇÕES AUXILIARES (HELPERS) ============================

def allowed_file(filename: str, allowed_extensions: Set[str]) -> bool:
    """Verifica se a extensão do arquivo é permitida."""
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in allowed_extensions


def clean_directory(directory_path: Path, remove_subfolders: bool = False):
    """Limpa o conteúdo de um diretório, opcionalmente removendo subpastas."""
    if not directory_path.exists():
        logger.warning(f"Diretório não encontrado para limpeza: {directory_path}")
        return

    logger.info(f"Iniciando limpeza do diretório: {directory_path} (Remover subfolders: {remove_subfolders})")
    for item in directory_path.iterdir():
        try:
            if item.is_file():
                os.remove(item)
                logger.debug(f"Arquivo removido: {item}")
            elif item.is_dir() and remove_subfolders:
                shutil.rmtree(item)
                logger.debug(f"Subpasta removida: {item}")
            elif item.is_dir() and not remove_subfolders:
                logger.debug(f"Subpasta mantida (configuração remove_subfolders=False): {item}")
        except OSError as e:
            logger.error(f"Erro ao remover {item}: {e}")
            # Flash message no Flask só faz sentido se estiver em um contexto de requisição
            # flash(f"Erro ao limpar {item}: {e}", "danger") # Removido para não usar fora do contexto de app
    logger.info(f"Limpeza do diretório {directory_path} concluída.")


def process_uploaded_dataset(file_path: Path) -> Tuple[bool, str]:
    """Processa um arquivo ZIP de dataset, descompactando-o."""
    if not zipfile.is_zipfile(file_path):
        return False, "O arquivo não é um arquivo ZIP válido."

    extract_to_path = DATASET_FOLDER
    clean_directory(extract_to_path, remove_subfolders=True)  # Limpa antes de extrair

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                member_path = (extract_to_path / member).resolve()
                # Garante que não há "zip slip" - extração fora do diretório de destino
                if not str(member_path).startswith(str(extract_to_path.resolve())):
                    raise ValueError(f"Tentativa de extração fora do diretório permitido: {member}")
            zip_ref.extractall(extract_to_path)
        return True, "Dataset descompactado com sucesso."
    except zipfile.BadZipFile:
        return False, "Arquivo ZIP corrompido ou inválido."
    except ValueError as e:
        logger.error(f"Erro de segurança ao descompactar: {e}")
        return False, f"Erro de segurança ao descompactar o arquivo: {e}"
    except Exception as e:
        logger.exception(f"Erro inesperado ao descompactar o dataset: {e}")
        return False, f"Erro inesperado ao descompactar o dataset: {e}"
    finally:
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"Arquivo ZIP temporário removido: {file_path}")


def prepare_training_data(preprocessor: AudioPreprocessor) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
    """
    Prepara os dados de treinamento a partir do diretório de datasets.
    Retorna X (features), y (rótulos numéricos) e o LabelEncoder.
    """
    all_audio_paths: List[Path] = []
    all_labels: List[str] = []

    # Percorre as subpastas (que representam as classes/labels)
    # Ex: datasets/REAL/, datasets/FAKE/
    for class_dir in DATASET_FOLDER.iterdir():
        if class_dir.is_dir():
            label = class_dir.name.lower()  # A label é o nome da subpasta (ex: 'real', 'fake')
            # Itera sobre os arquivos dentro da subpasta de classe
            for audio_file_path in class_dir.iterdir():  # Use iterdir() para arquivos diretos
                if allowed_file(audio_file_path.name, CONFIG["ALLOWED_AUDIO_EXT"]):
                    all_audio_paths.append(audio_file_path)
                    all_labels.append(label)

    if not all_audio_paths:
        logger.warning(
            "Nenhum arquivo de áudio encontrado nos subdiretórios do dataset. Verifique a estrutura (ex: dataset/real/, dataset/fake/).")
        return None, None, None

    # Garante que as classes esperadas ('real', 'fake') estão no LabelEncoder
    le = LabelEncoder()
    # Usar classes conhecidas para garantir que 'real' e 'fake' sempre tenham o mesmo mapeamento
    # Mesmo se um dataset inicial não contiver ambas as classes.
    # No entanto, se o dataset NÃO contiver real ou fake, o fit_transform pode falhar ou resultar em menos classes.
    # O train_test_split com stratify também exige pelo menos 2 samples por classe.
    # Para simplicidade, vamos permitir que o LabelEncoder se ajuste ao que for encontrado,
    # mas o treinamento irá falhar se houver menos de 2 classes.
    le.fit(sorted(list(set(all_labels))))  # Fit nas classes que realmente foram encontradas

    if len(le.classes_) < 2:
        logger.error(
            f"Apenas {len(le.classes_)} classe(s) encontrada(s) no dataset: {le.classes_}. São necessárias pelo menos 2 classes (real e fake) para treinamento.")
        return None, None, None

    encoded_labels = le.transform(all_labels)

    features_list: List[np.ndarray] = []
    processed_labels_list: List[int] = []
    failed_audios_count = 0

    # Defina a dimensão das features que o modelo espera
    feature_dim = CONFIG["N_MFCC"] if CONFIG["FEATURE_TYPE"] == "mfcc" else CONFIG["N_MELS"]
    expected_frames = CONFIG["EXPECTED_FRAMES"]

    for i, audio_path in enumerate(all_audio_paths):
        try:
            current_features = preprocessor.extract_features(
                str(audio_path),
                feature_type=CONFIG["FEATURE_TYPE"],
                expected_frames=expected_frames  # Garante que as features extraídas já têm o tamanho correto
            )
            if current_features is not None and current_features.shape[0] == expected_frames:
                features_list.append(current_features)
                processed_labels_list.append(encoded_labels[i])
            else:
                failed_audios_count += 1
                current_frames = current_features.shape[0] if current_features is not None else 'None'
                logger.warning(
                    f"Pulando {audio_path.name}: features não extraídas ou número de frames incorreto ({current_frames} vs {expected_frames}).")
        except Exception as e:
            failed_audios_count += 1
            logger.error(f"Erro ao processar áudio {audio_path.name} para treinamento: {e}")

    if not features_list:
        flash(
            "Nenhum áudio válido processado para treinamento. Verifique o dataset e as configurações de pré-processamento.",
            "danger")
        return None, None, None

    X = np.array(features_list).astype(np.float32)  # (num_samples, frames, features_dim)

    # Redimensiona para (num_samples, frames, features_dim, 1) para CNNs
    # Apenas se a dimensão de canal ainda não estiver presente.
    if X.ndim == 3:  # Se já é (num_samples, frames, features_dim)
        X = X[..., np.newaxis]  # Adiciona a dimensão de canal (última dimensão)
    elif X.ndim == 2:  # Caso inesperado se extract_features retorna (frames, features_dim)
        logger.error(
            "Features 2D inesperadas após empilhamento em prepare_training_data. Verifique a saída do AudioPreprocessor.")
        return None, None, None

    y = np.array(processed_labels_list)

    logger.info(f"Total de áudios processados para treinamento: {len(X)}")
    logger.info(f"Áudios falhos durante o processamento: {failed_audios_count}")
    logger.info(f"Shape final do dataset X: {X.shape}")
    logger.info(f"Shape final do dataset y: {y.shape}")
    logger.info(f"Classes do LabelEncoder: {le.classes_}")

    return X, y, le


# Funções para salvar/carregar LabelEncoder
def save_label_encoder(encoder: LabelEncoder, path: Path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(encoder, f)
        logger.info(f"LabelEncoder salvo em {path}")
    except Exception as e:
        logger.error(f"Erro ao salvar LabelEncoder: {e}")


def load_label_encoder(path: Path) -> Optional[LabelEncoder]:
    if not path.exists():
        logger.warning(f"LabelEncoder não encontrado em {path}")
        return None
    try:
        with open(path, 'rb') as f:
            encoder = pickle.load(f)
        logger.info(f"LabelEncoder carregado de {path}")
        return encoder
    except Exception as e:
        logger.error(f"Erro ao carregar LabelEncoder: {e}")
        return None


# ============================ FUNÇÃO PRINCIPAL DA APLICAÇÃO FLASK ============================

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['MAX_CONTENT_LENGTH'] = CONFIG["MAX_UPLOAD_SIZE"]
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "uma_chave_secreta_muito_segura_para_dev")
    app.permanent_session_lifetime = timedelta(minutes=60)

    # Cria diretórios na inicialização
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Garante que o diretório 'model_artifacts' existe

    # Inicializa o AudioPreprocessor
    audio_preprocessor = AudioPreprocessor(
        sample_rate=CONFIG["SAMPLE_RATE"],
        frame_length_ms=CONFIG["FRAME_LENGTH_MS"],
        frame_shift_ms=CONFIG["FRAME_SHIFT_MS"],
        n_mfcc=CONFIG["N_MFCC"] if CONFIG["FEATURE_TYPE"] == "mfcc" else CONFIG["N_MELS"],  # Usar a dimensão correta
        n_mels=CONFIG["N_MELS"],  # n_mels ainda é necessário para a função de mel
        n_fft=CONFIG["N_FFT"],
        vad_energy_thresh=CONFIG["VAD_ENERGY_THRESH"],
        min_segment_duration=CONFIG["MIN_SEGMENT_DURATION"],
        verbose=CONFIG["VERBOSE"]
    )

    # Inicializa o ModelPredictor
    # O feature_dim deve ser N_MFCC ou N_MELS dependendo do FEATURE_TYPE
    feature_dim_for_predictor = CONFIG["N_MFCC"] if CONFIG["FEATURE_TYPE"] == "mfcc" else CONFIG["N_MELS"]
    model_predictor = ModelPredictor(
        model_path=MODEL_DIR,  # Passar o diretório, não o nome do arquivo específico
        expected_frames=CONFIG["EXPECTED_FRAMES"],
        feature_dim=feature_dim_for_predictor
    )
    # Adiciona o preprocessor ao predictor, pois sliding_window_predict pode precisar dele
    model_predictor.audio_preprocessor = audio_preprocessor  # Adicionando esta linha

    # Tenta carregar o LabelEncoder associado ao modelo (se existir)
    loaded_encoder = load_label_encoder(MODEL_ENCODER_PATH)
    if loaded_encoder:
        model_predictor.set_label_encoder(loaded_encoder)

    # Context processor para injetar variáveis em todos os templates
    @app.context_processor
    def inject_global_data():
        return dict(
            model_exists=model_predictor.model_loaded,
            config=CONFIG,
            uploaded_audio_files=os.listdir(UPLOAD_FOLDER) if UPLOAD_FOLDER.exists() else []
        )

    # ============================ ROTAS DA APLICAÇÃO ============================

    @app.route("/")
    def index():
        logger.info("Acessando a página inicial.")
        return render_template("index.html")

    @app.route("/about")
    def about():
        logger.info("Acessando a página 'Sobre'.")
        return render_template("about.html")

    @app.route("/upload", methods=["POST"])
    def upload_file():
        logger.info("Recebendo solicitação de upload.")
        if "file" not in request.files:
            flash("Nenhum arquivo enviado.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("Nenhum arquivo selecionado.", "danger")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit(".", 1)[1].lower()
            file_path = UPLOAD_FOLDER / filename

            # Limpa o diretório de uploads antes de salvar o novo arquivo
            clean_directory(UPLOAD_FOLDER)

            if file_extension in CONFIG["ALLOWED_AUDIO_EXT"]:
                try:
                    file.save(file_path)
                    flash(f"Arquivo '{filename}' enviado com sucesso. Pronto para predição.", "success")
                except Exception as e:
                    logger.exception(f"Erro ao salvar arquivo de áudio: {e}")
                    flash(f"Erro ao salvar o arquivo de áudio: {e}", "danger")
                return redirect(url_for("index"))  # Redireciona para a página inicial com o arquivo carregado
            elif file_extension in CONFIG["ALLOWED_DATASET_EXT"]:
                try:
                    temp_zip_path = UPLOAD_FOLDER / filename
                    file.save(temp_zip_path)
                    success, message = process_uploaded_dataset(temp_zip_path)
                    if success:
                        flash(message, "success")
                    else:
                        flash(f"Erro ao processar dataset: {message}", "danger")
                except Exception as e:
                    logger.exception(f"Erro ao salvar ou processar arquivo ZIP: {e}")
                    flash(f"Erro ao salvar ou processar o arquivo ZIP: {e}", "danger")
                return redirect(url_for("train"))
            else:
                flash(
                    f"Tipo de arquivo não permitido: .{file_extension}. Apenas {', '.join(CONFIG['ALLOWED_AUDIO_EXT'])} para áudio ou {', '.join(CONFIG['ALLOWED_DATASET_EXT'])} para datasets.",
                    "danger")
                return redirect(url_for("index"))

        return redirect(url_for("index"))

    # A rota /predict será removida
    # @app.route("/predict", methods=["POST"])
    # def predict_audio():
    #     ... (código movido para fake_detector POST)

    @app.route("/train", methods=["GET", "POST"])
    def train():
        logger.info("Acessando a página de treinamento.")
        if request.method == "POST":
            logger.info("Iniciando processo de treinamento.")
            flash("Iniciando treinamento do modelo. Isso pode levar algum tempo...", "info")

            X, y, le = prepare_training_data(audio_preprocessor)
            if X is None or y is None or le is None:
                # prepare_training_data já deve ter flasheado a mensagem de erro
                return redirect(url_for("train"))

            try:
                # Define o input_shape para o ModelTrainer com base nos dados preparados
                # X.shape[1:] deve ser (frames, features_dim, 1)
                input_shape_for_trainer = X.shape[1:]

                trainer = ModelTrainer(
                    model_dir=str(MODEL_DIR),  # Passa o diretório
                    epochs=CONFIG["TRAINING_EPOCHS"],
                    batch_size=CONFIG["TRAINING_BATCH_SIZE"],
                    patience=CONFIG["TRAINING_PATIENCE"],
                    use_plateau=CONFIG["TRAINING_USE_PLATEAU"],
                    architecture=CONFIG["TRAINING_ARCHITECTURE"],
                    input_shape=input_shape_for_trainer,  # Passa o shape correto para o trainer
                    num_classes=len(le.classes_)  # Número de classes
                )
                history = trainer.train_model(X, y)

                if history:
                    # Verifica se 'val_loss' e 'val_accuracy' estão no histórico antes de acessar
                    final_val_loss = min(history.history['val_loss']) if 'val_loss' in history.history else float('inf')
                    final_val_accuracy = max(
                        history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0

                    if final_val_loss != float('inf'):  # Se o treinamento realmente aconteceu e teve validação
                        flash(
                            f"Treinamento concluído! Melhor V_Loss: {final_val_loss:.4f}, Melhor V_Acc: {final_val_accuracy:.4f}",
                            "success")
                    else:
                        flash(
                            "Treinamento concluído, mas sem dados de validação disponíveis no histórico (pode ser devido a dataset pequeno ou early stopping muito cedo).",
                            "warning")

                    # Salva o LabelEncoder treinado pelo trainer
                    if trainer.get_label_encoder():
                        save_label_encoder(trainer.get_label_encoder(), MODEL_ENCODER_PATH)
                        logger.info(f"LabelEncoder treinado salvo em {MODEL_ENCODER_PATH}")

                    # Recarrega o modelo mais recente e o LabelEncoder no predictor
                    model_predictor.load_model()  # Vai carregar o modelo mais recente no MODEL_DIR
                    if trainer.get_label_encoder():
                        model_predictor.set_label_encoder(trainer.get_label_encoder())  # Usa o LE do trainer
                    logger.info("Modelo e LabelEncoder recarregados no preditor após treinamento.")
                else:
                    flash("Treinamento não foi concluído com sucesso ou foi interrompido.", "warning")

            except Exception as e:
                logger.exception(f"Erro durante o treinamento do modelo: {e}")
                flash(f"Erro durante o treinamento do modelo: {e}. Verifique os logs para mais detalhes.", "danger")

            return redirect(url_for("train"))

        dataset_exists = DATASET_FOLDER.exists() and any(DATASET_FOLDER.iterdir())
        return render_template("train.html", dataset_exists=dataset_exists)

    @app.route("/clean_data", methods=["POST"])
    def clean_data():
        logger.info("Solicitação para limpar dados de upload e dataset.")
        clean_directory(UPLOAD_FOLDER, remove_subfolders=True)
        clean_directory(DATASET_FOLDER, remove_subfolders=True)
        flash("Dados de upload e dataset limpos.", "success")
        return redirect(url_for("index"))

    @app.route("/delete_model", methods=["POST"])
    def delete_model():
        logger.info("Solicitação para excluir o modelo.")
        # Exclui todos os arquivos .h5 no diretório do modelo
        deleted_count = 0
        if MODEL_DIR.exists():
            for f in MODEL_DIR.glob("*.h5"):
                try:
                    os.remove(f)
                    deleted_count += 1
                    logger.info(f"Modelo excluído: {f}")
                except OSError as e:
                    logger.error(f"Erro ao excluir modelo {f}: {e}")

            # Exclui o LabelEncoder também
            if MODEL_ENCODER_PATH.exists():
                try:
                    os.remove(MODEL_ENCODER_PATH)
                    logger.info(f"LabelEncoder excluído: {MODEL_ENCODER_PATH}")
                except OSError as e:
                    logger.error(f"Erro ao excluir LabelEncoder {MODEL_ENCODER_PATH}: {e}")
                    flash(f"Erro ao excluir LabelEncoder: {e}", "danger")  # Flash para o usuário

        if deleted_count > 0:
            flash(f"Total de {deleted_count} modelo(s) e LabelEncoder(s) excluídos.", "success")
        else:
            flash("Nenhum modelo ou LabelEncoder para excluir.", "warning")
            logger.warning("Tentativa de excluir modelo, mas nenhum modelo encontrado.")

        # Reseta o estado do predictor e o encoder após a exclusão
        model_predictor.load_model()
        model_predictor.set_label_encoder(None)
        return redirect(url_for("index"))

    @app.route('/static/<path:filename>')
    def static_files(filename):
        return send_from_directory(app.root_path + '/static', filename)

    # Rota para exibir o formulário do detector de deepfake e processar a predição
    @app.route("/fake_detector", methods=["GET", "POST"])
    def fake_detector():
        logger.info("Acessando a página do detector de deepfake (GET/POST request).")
        prediction_result = None  # Inicializa a variável para o template

        if request.method == "POST":
            logger.info("Recebendo solicitação de predição via /fake_detector (POST).")

            if not model_predictor.model_loaded:
                flash("Modelo de detecção não carregado. Treine um modelo primeiro.", "danger")
                # Renderiza a página novamente com a mensagem de erro
                return render_template("fake_detector.html", model_exists=model_predictor.model_loaded,
                                       prediction_result=None)

            if "voice_sample" not in request.files:
                flash("Nenhum arquivo enviado.", "danger")
                return render_template("fake_detector.html", model_exists=model_predictor.model_loaded,
                                       prediction_result=None)

            file = request.files["voice_sample"]
            if file.filename == "":
                flash("Nenhum arquivo selecionado.", "danger")
                return render_template("fake_detector.html", model_exists=model_predictor.model_loaded,
                                       prediction_result=None)

            if file and allowed_file(file.filename, CONFIG["ALLOWED_AUDIO_EXT"]):
                filename = secure_filename(file.filename)
                audio_file_path = UPLOAD_FOLDER / filename

                clean_directory(UPLOAD_FOLDER)  # Limpa o diretório de uploads antes de salvar
                logger.info(f"Arquivo '{filename}' salvo para predição.")
                try:
                    file.save(audio_file_path)

                    # --- Lógica de Predição (movida de /predict) ---
                    full_features_2d = audio_preprocessor.extract_features(
                        str(audio_file_path),
                        feature_type=CONFIG["FEATURE_TYPE"],
                        expected_frames=None
                    )

                    if full_features_2d is None or full_features_2d.shape[0] == 0:
                        flash("Não foi possível extrair características do áudio ou áudio muito curto.", "danger")
                        return render_template("fake_detector.html", model_exists=model_predictor.model_loaded,
                                               prediction_result=None)

                    full_features_3d = full_features_2d[..., np.newaxis]

                    result_label = "N/A"
                    result_confidence = 0.0
                    method_used = "N/A"

                    if CONFIG["USE_SLIDING_WINDOW_FOR_LONG_AUDIO"] and \
                            full_features_3d.shape[0] > CONFIG["PREDICTION_WINDOW_FRAMES"]:
                        logger.info(
                            f"Áudio longo detectado ({full_features_3d.shape[0]} frames). Usando predição com janela deslizante.")
                        is_fake_bool, avg_prob_fake = sliding_window_predict(
                            predictor=model_predictor,
                            full_audio_features=full_features_3d,
                            window_frames=CONFIG["PREDICTION_WINDOW_FRAMES"],
                            hop_frames=CONFIG["PREDICTION_HOP_FRAMES"],
                            threshold=CONFIG["PREDICTION_THRESHOLD"]
                        )
                        result_label = "FAKE" if is_fake_bool else "REAL"
                        result_confidence = avg_prob_fake
                        method_used = "Janela Deslizante"
                    else:
                        prepared_features_2d = audio_preprocessor._pad_or_truncate(
                            full_features_3d.squeeze(axis=-1), CONFIG["EXPECTED_FRAMES"]
                        )
                        prepared_features_3d = prepared_features_2d[..., np.newaxis]

                        logger.info(f"Áudio processado como um todo. Shape para predição: {prepared_features_3d.shape}")

                        predicted_label, confidence_value = model_predictor.predict_single_audio(prepared_features_3d)
                        result_label = predicted_label
                        result_confidence = confidence_value
                        method_used = "Áudio Completo"

                    # Prepara o resultado para passar ao template
                    prediction_result = {
                        "label": result_label,
                        "confidence": f"{(result_confidence * 100):.2f}%",
                        "method": method_used,
                        "filename": filename  # Adiciona o nome do arquivo para melhor feedback
                    }
                    flash(f"Análise de '{filename}' concluída.", "success")  # Feedback de sucesso via flash

                except CouldntDecodeError as e:
                    flash(
                        f"Erro ao decodificar o arquivo de áudio: {e}. Certifique-se de que é um formato de áudio válido e não está corrompido.",
                        "danger")
                    logger.error(f"Erro de decodificação para {filename}: {e}")
                except Exception as e:
                    flash(f"Ocorreu um erro durante a predição: {e}", "danger")
                    logger.exception(f"Erro inesperado durante a predição para {filename}.")
                finally:
                    if audio_file_path.exists():
                        os.remove(audio_file_path)  # Limpa o arquivo após o processamento
            else:
                flash(f"Tipo de arquivo não permitido. Apenas {', '.join(CONFIG['ALLOWED_AUDIO_EXT'])}.", "danger")

        # Renderiza a página do detector, passando o resultado da predição (se houver)
        return render_template("fake_detector.html", model_exists=model_predictor.model_loaded,
                               result=prediction_result)

    return app


# ============================ PONTO DE ENTRADA ============================
if __name__ == "__main__":
    app = create_app()
    logger.info("DeepAudio Flask App iniciado.")

    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ('true', '1', 't')
    app.run(debug=debug_mode, host=host, port=port)