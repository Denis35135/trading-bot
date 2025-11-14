@echo off
REM ============================================
REM AUTOBOT ULTIMATE - Installation Script
REM ============================================

echo.
echo ========================================
echo   AUTOBOT ULTIMATE - Installation
echo ========================================
echo.

REM Check Python
echo [1/5] Verification Python...
python --version
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Telecharger Python 3.9+ sur: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo [2/5] Creation environnement virtuel...
python -m venv venv
if errorlevel 1 (
    echo ERREUR: Impossible de creer l'environnement virtuel
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo [3/5] Activation environnement virtuel...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/5] Mise a jour pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo [5/5] Installation des dependances...
pip install -r requirements.txt --break-system-packages
if errorlevel 1 (
    echo ERREUR: Installation des dependances echouee
    pause
    exit /b 1
)

REM Create .env if not exists
if not exist .env (
    echo.
    echo Creation du fichier .env...
    copy .env.example .env
    echo.
    echo IMPORTANT: Editer le fichier .env avec vos cles API Binance!
)

REM Run requirements check
echo.
echo Verification des dependances...
python check_requirements.py

echo.
echo ========================================
echo   Installation terminee avec succes!
echo ========================================
echo.
echo Prochaines etapes:
echo 1. Editer .env avec vos cles API Binance
echo 2. Lancer le bot: python main.py
echo.
pause
