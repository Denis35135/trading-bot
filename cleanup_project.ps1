# 🧹 Script de nettoyage du projet The Bot
# À exécuter dans PowerShell depuis la racine du projet

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "🧹 NETTOYAGE DU PROJET THE BOT" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Fonction pour supprimer un fichier
function Remove-FileIfExists {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        try {
            Remove-Item $FilePath -Force
            Write-Host "✅ Supprimé: $FilePath" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "❌ Erreur suppression: $FilePath - $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "⚠️  Fichier introuvable: $FilePath" -ForegroundColor Yellow
        return $false
    }
}

# Fonction pour renommer un fichier
function Rename-FileIfExists {
    param([string]$OldName, [string]$NewName)
    
    if (Test-Path $OldName) {
        if (Test-Path $NewName) {
            Write-Host "⚠️  $NewName existe déjà, suppression de $OldName" -ForegroundColor Yellow
            Remove-Item $OldName -Force
        }
        else {
            try {
                Rename-Item $OldName $NewName -Force
                Write-Host "✅ Renommé: $OldName → $NewName" -ForegroundColor Green
                return $true
            }
            catch {
                Write-Host "❌ Erreur renommage: $OldName - $($_.Exception.Message)" -ForegroundColor Red
                return $false
            }
        }
    }
    else {
        Write-Host "⚠️  Fichier introuvable: $OldName" -ForegroundColor Yellow
        return $false
    }
}

Write-Host "📋 ÉTAPE 1: Suppression des fichiers temporaires" -ForegroundColor Yellow
Write-Host ""

$filesToDelete = @(
    "arborescence.txt",
    "projet_complet.txt",
    "STRUCTURE.txt",
    "VERIFICATION.txt",
    "VERIFICATION_COMPLETE.txt",
    "_env",
    "_env.example",
    "_gitkeep",
    "default_config.json"
)

$deletedCount = 0
foreach ($file in $filesToDelete) {
    if (Remove-FileIfExists $file) {
        $deletedCount++
    }
}

Write-Host ""
Write-Host "📋 ÉTAPE 2: Renommage des fichiers" -ForegroundColor Yellow
Write-Host ""

$renamedCount = 0
if (Rename-FileIfExists "_gitignore" ".gitignore") {
    $renamedCount++
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "✅ NETTOYAGE TERMINÉ" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Résumé:" -ForegroundColor White
Write-Host "  • Fichiers supprimés: $deletedCount / $($filesToDelete.Count)" -ForegroundColor White
Write-Host "  • Fichiers renommés: $renamedCount / 1" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Prochaines étapes:" -ForegroundColor Yellow
Write-Host "  1. Vérifier que tout est OK: dir" -ForegroundColor White
Write-Host "  2. Appliquer les corrections de code (voir corrections_code.txt)" -ForegroundColor White
Write-Host "  3. Lancer les tests: python check_requirements.py" -ForegroundColor White
Write-Host ""
