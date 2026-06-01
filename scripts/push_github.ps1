# Run after creating an empty GitHub repo (no README).
# Example: https://github.com/YOUR_USERNAME/telecom-churn-prediction

param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUsername,

    [string]$RepoName = "telecom-churn-prediction"
)

$remote = "https://github.com/$GitHubUsername/$RepoName.git"
Set-Location (Split-Path $PSScriptRoot -Parent)

git remote remove origin 2>$null
git remote add origin $remote
git push -u origin main
Write-Host "Pushed to $remote"
