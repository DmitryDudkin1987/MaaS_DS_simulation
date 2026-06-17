# 🚀 MaaS_DS_simulation (Model as a Service)

Реализация подхода **Model as a Service (MaaS)** для задачи машинного обучения. Проект демонстрирует, как обученную ML-модель обернуть в REST API с использованием **FastAPI**.

---

## 📁 Структура проекта

```text
MaaS_DS_simulation/
├── models/
│   ├── .gitkeep
│   └── model.joblib
├── src/
│   ├── app.py
│   ├── inference.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── train.py
├── db/
├── requirements.txt
├── README.md
└── .gitignore

🛠️ Установка и запуск

git clone https://github.com/DmitryDudkin1987/MaaS_DS_simulation.git
cd MaaS_DS_simulation
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
python -m src.pipeline
uvicorn src.app:app --reload --port 8890