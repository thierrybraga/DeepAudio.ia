import os
import zipfile
import glob
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash

# Importa funções dos módulos existentes
from Voice2data import process_audio
from TrainModel import create_cnn_model
from Predictor import load_csv_features, preprocess_features, load_trained_model, predict_speech_match

# Configurações de pastas
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
FEATURES_FOLDER = os.path.join(BASE_DIR, "features")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# Cria as pastas, se não existirem
for folder in [UPLOAD_FOLDER, DATASET_FOLDER, FEATURES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.secret_key = "segredo"  # necessário para usar flash messages

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if "dataset_zip" not in request.files:
        flash("Nenhum arquivo enviado!", "warning")
        return redirect(url_for("index"))
    file = request.files["dataset_zip"]
    if file.filename == "":
        flash("Nenhum arquivo selecionado!", "warning")
        return redirect(url_for("index"))
    if file and file.filename.lower().endswith(".zip"):
        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(zip_path)
        # Extrai o zip para o diretório DATASET_FOLDER
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATASET_FOLDER)
        flash("Dataset enviado e extraído com sucesso!", "success")
    else:
        flash("Por favor, envie um arquivo .zip", "danger")
    return redirect(url_for("index"))

@app.route("/pitch_extract", methods=["POST"])
def pitch_extract():
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.wma", "*.ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(DATASET_FOLDER, "**", ext), recursive=True))

    if not audio_files:
        flash("Nenhum arquivo de áudio encontrado no dataset.", "warning")
        return redirect(url_for("index"))

    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_csv = os.path.join(FEATURES_FOLDER, base_name + ".csv")
        try:
            process_audio(audio_file, output_csv)
        except Exception as e:
            flash(f"Erro ao processar {audio_file}: {str(e)}", "danger")
            continue
    flash("Extração de features (Pitch Extract) concluída!", "success")
    return redirect(url_for("index"))

@app.route("/train_model", methods=["POST"])
def train_model():
    csv_files = glob.glob(os.path.join(FEATURES_FOLDER, "*.csv"))
    if not csv_files:
        flash("Nenhum arquivo CSV com features encontrado. Execute o Pitch Extract primeiro.", "warning")
        return redirect(url_for("index"))

    X_list = []
    y_list = []
    for csv_file in csv_files:
        try:
            features = load_csv_features(csv_file)  # remove timestamp
            features = preprocess_features(features, expected_frames=100)
            X_list.append(features[0])
            # Para exemplo, atribui rótulos aleatórios (0 ou 1)
            y_list.append(np.random.randint(0, 2))
        except Exception as e:
            flash(f"Erro ao carregar {csv_file}: {str(e)}", "danger")
            continue

    if not X_list:
        flash("Não foi possível carregar nenhum dado para treinamento.", "warning")
        return redirect(url_for("index"))

    X = np.array(X_list)
    y = np.array(y_list)

    input_shape = (100, X.shape[2])
    model = create_cnn_model(input_shape)

    model.fit(X, y, epochs=5, batch_size=4, verbose=1)

    model.save(MODEL_PATH)
    flash("Treinamento concluído e modelo salvo!", "success")
    return redirect(url_for("index"))

@app.route("/fake_detector", methods=["GET", "POST"])
def fake_detector():
    if request.method == "GET":
        return render_template("fake_detector.html")
    else:
        if "voice_sample" not in request.files:
            flash("Nenhum arquivo enviado!", "warning")
            return redirect(url_for("fake_detector"))
        file = request.files["voice_sample"]
        if file.filename == "":
            flash("Nenhum arquivo selecionado!", "warning")
            return redirect(url_for("fake_detector"))
        sample_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(sample_path)
        sample_csv = os.path.join(UPLOAD_FOLDER, "sample_features.csv")
        try:
            process_audio(sample_path, sample_csv)
        except Exception as e:
            flash(f"Erro na extração de features: {str(e)}", "danger")
            return redirect(url_for("fake_detector"))

        try:
            features = load_csv_features(sample_csv)
            features = preprocess_features(features, expected_frames=100)
            model = load_trained_model(MODEL_PATH)
            match, confidence = predict_speech_match(model, features)
            if match:
                result = f"Voz CORRESPONDE ao modelo treinado (confiança: {confidence:.4f})."
            else:
                result = f"Voz NÃO corresponde ao modelo treinado (confiança: {confidence:.4f})."
        except Exception as e:
            flash(f"Erro durante a predição: {str(e)}", "danger")
            return redirect(url_for("fake_detector"))
        return render_template("fake_detector.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
