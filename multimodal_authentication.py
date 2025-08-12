import os
from pathlib import Path
import torchaudio
import pandas as pd
import numpy as np
from pydub import AudioSegment
from speechbrain.inference import SpeakerRecognition
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, DetCurveDisplay
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# === SAFE PATH SETUP ===
SAFE_LOCAL_PATH = Path("C:/BiometricsCache")
SAFE_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
os.environ["SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY"] = "copy"
os.environ["SPEECHBRAIN_CACHE"] = str(SAFE_LOCAL_PATH / "tmpdir")
MODEL_DIR = SAFE_LOCAL_PATH / "pretrained_model"

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parent
VOICE_TSV_PATH = BASE_DIR / "data" / "validated.tsv"
VOICE_CLIP_DIR = BASE_DIR / "data" / "clips"
FACE_EMB_PATH = BASE_DIR / "data" / "X-5-SoF.npy"

# === MODEL LOAD ===
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"local_strategy": "copy"}
)

# === CONVERT AUDIO ===
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.with_suffix(".wav")
    sound = AudioSegment.from_mp3(str(mp3_path))
    sound.export(wav_path, format="wav")
    return wav_path

# === EMBEDDING EXTRACTION ===
def extract_speaker_embeddings(df):
    embeddings = []
    for path in df["path"]:
        full_path = VOICE_CLIP_DIR / path.strip()
        if not full_path.exists():
            continue
        if full_path.suffix == ".mp3":
            full_path = convert_mp3_to_wav(full_path)
        signal, fs = torchaudio.load(str(full_path))
        emb = speaker_model.encode_batch(signal)
        emb_np = emb.squeeze().detach().cpu().numpy()
        embeddings.append(emb_np)
    return np.array(embeddings)

# === FUSION ===
def feature_level_fusion(voice, face):
    scaler = StandardScaler()
    voice_norm = scaler.fit_transform(voice)
    face_norm = scaler.fit_transform(face)
    return np.concatenate([voice_norm, face_norm], axis=1)

# === METRIC FUNCTIONS ===
def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute(fnr - fpr))]

def calculate_dprime(fpr, tpr):
    return np.abs(norm.ppf(1 - fpr.mean()) - norm.ppf(tpr.mean()))

def plot_combined_curves(y_true, score_dict):
    # Binarize labels for ROC/AUC (One-vs-Rest style)
    from sklearn.preprocessing import label_binarize
    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)

    # === ROC Curve ===
    plt.figure()
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
        auc_score = auc(fpr, tpr)
        eer = calculate_eer(fpr, tpr)
        dprime = calculate_dprime(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc_score:.2f} EER={eer:.2f} d′={dprime:.2f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.title("ROC Curve (Combined)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.show()

    # === DET Curve ===
    plt.figure()
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
        DetCurveDisplay(fpr=fpr, fnr=1 - tpr, estimator_name=name).plot()
    plt.title("DET Curve (Combined)")
    plt.grid(True)
    plt.show()

    # === Score Distribution ===
    plt.figure()
    for name, scores in score_dict.items():
        sns.kdeplot(scores.ravel(), label=f"{name}")
    plt.title("Score Distribution (Combined)")
    plt.xlabel("Classifier Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

# === EVALUATION ===
def evaluate_embeddings(X, y, name):
    clf = SVC(kernel="linear", probability=True)
    skf = StratifiedKFold(n_splits=5)
    accs, probs, labels = [], [], []
    for train, test in skf.split(X, y):
        clf.fit(X[train], y[train])
        accs.append(clf.score(X[test], y[test]))
        probs.extend(clf.predict_proba(X[test]))
        labels.extend(y[test])
    avg = np.mean(accs)
    print(f"{name} accuracy: {avg:.4f}")
    return avg, np.array(probs), np.array(labels)

# === MAIN ===
def main():
    df = pd.read_csv(VOICE_TSV_PATH, sep="\t")
    filtered_df = df[df["client_id"].isin(df["client_id"].value_counts()[lambda x: x >= 5].index)]
    sampled_df = filtered_df.groupby("client_id").head(5).sort_values("client_id").reset_index(drop=True)

    print(f"Processing {len(sampled_df)} filtered audio samples...")
    voice_emb = extract_speaker_embeddings(sampled_df)
    print("Voice embeddings shape:", voice_emb.shape)

    face_emb = np.load(FACE_EMB_PATH)
    if face_emb.ndim == 3:
        face_emb = face_emb.reshape(face_emb.shape[0], -1)
    print("Face embeddings shape:", face_emb.shape)

    sim_len = min(len(voice_emb), len(face_emb))
    labels = np.array(sorted([i for i in range(sim_len // 5) for _ in range(5)]))

    voice_acc, voice_probs, voice_y = evaluate_embeddings(voice_emb[:sim_len], labels, "Voice-only")
    face_acc, face_probs, _ = evaluate_embeddings(face_emb[:sim_len], labels, "Face-only")
    fused = feature_level_fusion(voice_emb[:sim_len], face_emb[:sim_len])
    fusion_acc, fusion_probs, _ = evaluate_embeddings(fused, labels, "Fusion")

    print("\n=== Summary Table ===")
    print("Model\t\tAccuracy")
    print(f"Voice-only\t{voice_acc:.4f}")
    print(f"Face-only\t{face_acc:.4f}")
    print(f"Fusion\t\t{fusion_acc:.4f}")

    score_dict = {
    "Voice": voice_probs,
    "Face": face_probs,
    "Fusion": fusion_probs
    }
    plot_combined_curves(labels, score_dict)


if __name__ == "__main__":
    main()
