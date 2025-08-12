# Analyzing Multimodal Biometric Data with Occluded Facial Conditions

## 📌 Overview
This project explores how **feature-level fusion** of face and voice embeddings can improve biometric authentication in scenarios where facial features are partially hidden (e.g., scarves, glasses, masks).  
Using **precomputed facial landmarks** from the SOF dataset and **ECAPA-TDNN voice embeddings** from Mozilla Common Voice, we demonstrate that multimodal fusion greatly boosts performance compared to unimodal systems.

For a detailed explanation of the research, see the presentation file in:  
`presentation/Analyzing_Multimodal_Biometric_Data.pptx`

---

## 🎯 Goals
- Evaluate the impact of facial occlusion on biometric accuracy  
- Compare **face-only**, **voice-only**, and **fused** authentication  
- Investigate the effectiveness of **feature-level fusion**  
- Highlight ethical considerations in biometric data collection and usage  

---

## 📂 Datasets
**Face Data (SOF Dataset)**  
- Surveillance Occluded Faces dataset  
- Precomputed 68-landmark facial embeddings  
- Occlusion examples: scarves, glasses  

**Voice Data (Mozilla Common Voice)**  
- Clean MP3 recordings converted to WAV  
- Embeddings generated using ECAPA-TDNN model  

⚠️ **Note:** Raw datasets are not included in this repository due to size and licensing.

---

## 🛠️ Methods
1. **Data Preprocessing**  
   - Converted audio to WAV  
   - Generated voice embeddings using ECAPA-TDNN  
   - Extracted precomputed 68-point facial landmarks  

2. **Synthetic User Mapping**  
   - Linked face and voice samples for simulated multimodal profiles  

3. **Feature-Level Fusion**  
   - Combined embeddings before classification  

4. **Evaluation**  
   - Metric: Accuracy  

---

## 📊 Results

| System     | Accuracy |
|------------|----------|
| Face-only  | 15%      |
| Voice-only | 95%      |
| **Fusion** | **97.5%**|

- Facial recognition suffers heavily in occlusion scenarios.  
- Voice recognition remains highly robust.  
- **Feature-level fusion** effectively combines both modalities to achieve the best performance.

*Future work:* Add ROC and DET curve evaluations for more detailed analysis.

---

## ⚖️ Ethical Considerations
- **Privacy & Consent:** Biometric data is highly sensitive — ensure informed consent.  
- **Bias & Fairness:** Potential underperformance for certain demographics.  
- **Data Security:** Avoid repurposing biometric data without explicit permission.

---

## 📂 Repository Structure
├── scripts/
│ └── multimodal_authentication.py
├── presentation/
│ └── Analyzing_Multimodal_Biometric_Data.pptx
├── README.md
├── LICENSE
└── .gitignore
