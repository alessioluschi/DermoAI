@echo off
:: DermoAI — KB RAG Environment Setup (Windows)
:: Crea un venv leggero solo per costruire la knowledge base.
:: Nessuna GPU richiesta.
:: Uso: setup_env_kb.bat

echo ============================================================
echo DermoAI - KB RAG Environment Setup
echo ============================================================

python -m venv venv_kb
call venv_kb\Scripts\activate

echo [1/2] Installing KB dependencies...
pip install --upgrade pip --quiet
pip install -r requirements_kb.txt

echo [2/2] Verifying installation...
python -c "import chromadb, sentence_transformers, Bio, requests, bs4; print('  All KB dependencies OK')"

echo.
echo ============================================================
echo Setup complete!
echo.
echo Per attivare l'ambiente:
echo   call venv_kb\Scripts\activate
echo.
echo Per costruire la KB RAG:
echo   python rag_kb\build_kb.py --email tua@email.com
echo.
echo Per reindicizzare da cache locale (senza internet):
echo   python rag_kb\build_kb.py --rebuild-from-cache
echo ============================================================
pause
