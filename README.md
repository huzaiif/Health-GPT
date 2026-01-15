# HealthGPT - Intelligent Disease Prediction & Health Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B)
![Gemini API](https://img.shields.io/badge/Google%20Gemini-AI-purple)
![License](https://img.shields.io/badge/License-MIT-green)

**HealthGPT** is a comprehensive health platform that combines Machine Learning and Generative AI to provide early disease prediction and personalized health advice. It features a specialized AI chatbot and predictive models for Diabetes, Heart Disease, and Parkinson's.

## 🚀 Features

*   **🤖 AI Health Assistant:** A conversational AI powered by Google's Gemini model that answers your health-related queries and provides actionable advice.
*   **🩺 Disease Prediction Models:**
    *   **Diabetes Prediction:** Estimates the likelihood of diabetes based on diagnostic measures like Glucose, BMI, and Insulin levels.
    *   **Heart Disease Prediction:** Analyzes cardiovascular health metrics to predict potential heart disease risks.
    *   **Parkinson's Disease Prediction:** Uses voice measurement data (MDVP, Jitter, Shimmer, etc.) to detect early signs of Parkinson's.
*   **💡 Personalized Health Tips:** Automatically generates specialized health advice based on the prediction results using Generative AI.

## 🛠️ Technology Stack

*   **Frontend:** Streamlit
*   **AI/LLM:** Google Gemini API (`gemini-flash-lite-latest`)
*   **Machine Learning:** Scikit-learn, Numpy
*   **Model Storage:** Pickle

## 📂 Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── saved_models/               # Pre-trained ML models (.sav)
├── dataset/                    # Datasets used for training
├── colab_files_to_train_models/# Jupyter notebooks for model training
└── README.md                   # Project documentation
```

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/huzaiif/Health-GPT.git
    cd Health-GPT
    ```

2.  **Install Dependencies**
    Ensure you have Python installed. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**
    *   Create a `.env` file in the root directory.
    *   Add your Google Gemini API key:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 🧠 How It Works

1.  **Select a Service:** Use the sidebar to navigate between the "Health Assistant" chatbot or one of the three disease prediction tools.
2.  **Input Data:** For disease prediction, enter the required medical parameters (e.g., Age, BMI, Blood Pressure).
3.  **Get Results:** Click the "Test Result" button to get the prediction.
4.  **AI Advice:** If configured, the AI will provide context-aware health tips based on your result.

## ⚠️ Medical Disclaimer

**HealthGPT is an informational tool and NOT a substitute for professional medical advice, diagnosis, or treatment.**
*   The predictions are based on machine learning models and may not always be 100% accurate.
*   Always consult with a qualified healthcare provider for any medical concerns.
*   In case of a medical emergency, contact your local emergency services immediately.

## 👨‍💻 Author

**Huzaif**
