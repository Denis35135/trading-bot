@echo off
REM ===================================================================
REM THE BOT - Script d'installation automatique (Windows)
REM ===================================================================

setlocal enabledelayedexpansion

REM Couleurs (via ANSI si supportÃƒÂ©)
set "BLUE=[94m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM ===================================================================
REM Fonctions
REM ===================================================================

:print_step
echo.
echo %BLUE%===^> %~1%NC%
echo.
goto :eof

:print_success
echo %GREEN%[OK] %~1%NC%
goto :eof

:print_error
echo %RED%[ERREUR] %~1%NC%
goto :eof

:print_warning
echo %YELLOW%[ATTENTION] %~1%NC%
goto :eof

REM ===================================================================
REM VÃƒÂ©rifications
REM ===================================================================

:check_python
call :print_step "Verification de Python..."

where python >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Python non trouve!"
    echo.
    echo Installez Python 3.9+ depuis: https://www.python.org/downloads/
    echo IMPORTANT: Cochez "Add Python to PATH" lors de l'installation
    echo.
    pause
    exit /b 1
)

REM VÃƒÂ©rifier la version
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Python version: %PYTHON_VERSION%

REM Extraire major et minor
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% lss 3 (
    call :print_error "Python 3.9+ requis"
    pause
    exit /b 1
)

if %PYTHON_MAJOR% equ 3 if %PYTHON_MINOR% lss 9 (
    call :print_error "Python 3.9+ requis (trouve: %PYTHON_VERSION%)"
    pause
    exit /b 1
)

call :print_success "Python %PYTHON_VERSION% OK"
goto :eof

:check_pip
call :print_step "Verification de pip..."

python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "pip non trouve!"
    echo Installation de pip...
    python -m ensurepip --upgrade
)

call :print_success "pip OK"
goto :eof

:install_talib
call :print_step "Installation de TA-Lib..."

REM Tester si dÃƒÂ©jÃƒÂ  installÃƒÂ©
python -c "import talib" >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "TA-Lib deja installe"
    goto :eof
)

echo.
echo TA-Lib necessite un wheel precompile pour Windows.
echo.
echo 1. Telechargez le wheel correspondant a votre version Python:
echo    https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo.
echo Exemples:
echo    - Python 3.9 (64-bit): TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
echo    - Python 3.10 (64-bit): TA_Lib-0.4.24-cp310-cp310-win_amd64.whl
echo    - Python 3.11 (64-bit): TA_Lib-0.4.24-cp311-cp311-win_amd64.whl
echo.
echo 2. Placez le fichier .whl dans le dossier courant
echo.
echo 3. Installez avec: pip install TA_Lib-0.4.24-cpXX-cpXX-win_amd64.whl
echo.

call :print_warning "Installez TA-Lib manuellement puis relancez ce script"
echo.
set /p continue="Continuer sans TA-Lib? (O/N): "
if /i "%continue%" neq "O" (
    echo Installation annulee.
    pause
    exit /b 1
)

goto :eof

:create_venv
call :print_step "Creation de l'environnement virtuel..."

if exist venv (
    call :print_warning "L'environnement virtuel existe deja"
    set /p recreate="Recreer? (O/N): "
    if /i "!recreate!" equ "O" (
        echo Suppression de l'ancien environnement...
        rmdir /s /q venv
    ) else (
        call :print_success "Utilisation de l'environnement existant"
        goto :eof
    )
)

python -m venv venv
if %errorlevel% neq 0 (
    call :print_error "Impossible de creer l'environnement virtuel"
    pause
    exit /b 1
)

call :print_success "Environnement virtuel cree"
goto :eof

:install_dependencies
call :print_step "Installation des dependances Python..."

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Installer requirements
if not exist requirements.txt (
    call :print_error "requirements.txt non trouve!"
    pause
    exit /b 1
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    call :print_error "Echec installation des dependances"
    pause
    exit /b 1
)

call :print_success "Dependances installees"
goto :eof

:setup_config
call :print_step "Configuration..."

REM CrÃƒÂ©er .env
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        call :print_success "Fichier .env cree"
        call :print_warning "IMPORTANT: Editez .env avec vos cles API Binance!"
    ) else (
        call :print_error ".env.example non trouve"
    )
) else (
    call :print_warning ".env existe deja (non modifie)"
)

REM CrÃƒÂ©er config.py
if not exist config.py (
    if exist config.example.py (
        copy config.example.py config.py >nul
        call :print_success "Fichier config.py cree"
    ) else (
        call :print_error "config.example.py non trouve"
    )
) else (
    call :print_warning "config.py existe deja (non modifie)"
)

goto :eof

:create_directories
call :print_step "Creation de la structure de dossiers..."

REM CrÃƒÂ©er les dossiers
if not exist data\logs mkdir data\logs
if not exist data\models mkdir data\models
if not exist data\cache mkdir data\cache
if not exist data\backups mkdir data\backups

REM CrÃƒÂ©er .gitkeep
type nul > data\logs\.gitkeep
type nul > data\models\.gitkeep
type nul > data\cache\.gitkeep
type nul > data\backups\.gitkeep

call :print_success "Structure de dossiers creee"
goto :eof

:install_redis_optional
call :print_step "Installation de Redis (optionnel)..."

echo.
echo Redis est optionnel mais ameliore les performances.
echo.
echo Pour Windows, installez Memurai (Redis pour Windows):
echo https://www.memurai.com/get-memurai
echo.
call :print_warning "Redis non installe automatiquement sur Windows"
echo.

goto :eof

:test_connection
call :print_step "Test de connexion (optionnel)..."

echo.
set /p test_now="Tester la connexion maintenant? (O/N): "
if /i "%test_now%" equ "O" (
    call venv\Scripts\activate.bat
    if exist test_connection.py (
        python test_connection.py
    ) else (
        call :print_error "test_connection.py non trouve"
    )
) else (
    call :print_warning "Test ignore (lancez 'python test_connection.py' plus tard)"
)

goto :eof

:show_next_steps
echo.
echo %GREEN%=========================================================%NC%
echo.
echo            INSTALLATION TERMINEE!
echo.
echo %GREEN%=========================================================%NC%
echo.
echo.
echo %YELLOW%PROCHAINES ETAPES:%NC%
echo.
echo 1. Configurez vos cles API Binance:
echo    %BLUE%notepad .env%NC%
echo.
echo 2. (Optionnel) Ajustez la configuration:
echo    %BLUE%notepad config.py%NC%
echo.
echo 3. Activez l'environnement virtuel:
echo    %BLUE%venv\Scripts\activate%NC%
echo.
echo 4. Testez la connexion:
echo    %BLUE%python test_connection.py%NC%
echo.
echo 5. Lancez le bot en mode Paper Trading:
echo    %BLUE%python main.py --mode paper%NC%
echo.
echo 6. Une fois teste, passez en mode Live:
echo    %BLUE%python main.py --mode live%NC%
echo.
echo.
echo %YELLOW%DOCUMENTATION:%NC%
echo    - Guide d'installation: docs\INSTALLATION.md
echo    - Configuration: docs\CONFIGURATION.md
echo    - FAQ: docs\FAQ.md
echo.
echo.
echo %GREEN%Bon trading!%NC%
echo.
goto :eof

REM ===================================================================
REM MAIN
REM ===================================================================

:main
cls
call :print_header

REM VÃƒÂ©rifier qu'on est dans le bon rÃƒÂ©pertoire
if not exist main.py (
    call :print_error "main.py non trouve!"
    echo.
    echo Lancez ce script depuis le repertoire racine de The Bot
    echo.
    pause
    exit /b 1
)

REM Ãƒâ€°tapes d'installation
call :check_python
if %errorlevel% neq 0 exit /b 1

call :check_pip
call :install_talib
call :create_venv
if %errorlevel% neq 0 exit /b 1

call :install_dependencies
if %errorlevel% neq 0 exit /b 1

call :setup_config
call :create_directories
call :install_redis_optional
call :test_connection
call :show_next_steps

echo.
pause
exit /b 0

REM ===================================================================
REM Lancer l'installation
REM ===================================================================

call :main
_header
echo.
echo %BLUE%=========================================================%NC%
echo.
echo              THE BOT - INSTALLATION
echo.
echo      Bot de Trading Algorithmique pour Binance
echo.
echo %BLUE%=========================================================%NC%
echo.
goto :eof

:print