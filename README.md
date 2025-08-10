# TOD Play MVP (local)

- Backend: FastAPI (simple overlay demo)
- Frontend: vanilla HTML/JS uploader

## Run backend (Windows PowerShell)
```
cd backend
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs

## Frontend
Open `frontend/index.html` in your browser and set API URL to `http://127.0.0.1:8000`.
