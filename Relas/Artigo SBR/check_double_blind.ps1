<#
.SYNOPSIS
  Varre .tex/.bib (e opcionalmente o PDF) procurando vazamentos que quebrem
  a revisao double-blind do SBR 2026.

.EXEMPLO
  powershell -ExecutionPolicy Bypass -File .\check_double_blind.ps1 -Path . -Pdf .\sbr2026_paper.pdf
#>

param(
  [string]$Path = ".",
  [string]$Pdf  = ""
)

$padroes = @(
  'universidade', 'university', 'universidad', 'instituto', 'institute',
  'federal', 'laborat', 'departamento', 'department of',
  'acknowledg', 'agradecim', '\\thanks',
  'cnpq', 'capes', 'fapesp', 'fapes', 'finep', 'fapemig', 'faperj',
  'grant\s*(no|number|#)', 'processo\s*n',
  'github\.com', 'gitlab\.com', 'lattes', 'orcid', 'linkedin',
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
  'our (previous|earlier|prior) (work|paper|approach|method|system)',
  'we (previously|have previously|earlier)', 'our (lab|laboratory|group|team|university)',
  'natalnet', 'ufrn', 'ufpb', 'usp', 'unicamp', 'ufmg', 'ita\b', 'ufam'
)

$arquivos = Get-ChildItem -Path $Path -Include *.tex, *.bib -Recurse -File
if (-not $arquivos) { Write-Host "Nenhum .tex/.bib encontrado em $Path" -ForegroundColor Yellow }

$total = 0
foreach ($f in $arquivos) {
  $linhas = Get-Content -LiteralPath $f.FullName
  for ($i = 0; $i -lt $linhas.Count; $i++) {
    foreach ($p in $padroes) {
      if ($linhas[$i] -match $p) {
        Write-Host ("[{0}:{1}] {2}" -f $f.Name, ($i + 1), $linhas[$i].Trim()) -ForegroundColor Red
        $total++
        break
      }
    }
  }
}

Write-Host ""
if ($total -eq 0) {
  Write-Host "OK: nenhum padrao suspeito nos fontes." -ForegroundColor Green
} else {
  Write-Host ("ATENCAO: {0} linha(s) suspeita(s). Revise cada uma antes de submeter." -f $total) -ForegroundColor Yellow
  Write-Host "Falsos positivos sao comuns (ex.: nome de universidade em uma referencia legitima de terceiros)."
}

# ---- Metadados do PDF -------------------------------------------------------
if ($Pdf -and (Test-Path $Pdf)) {
  Write-Host "`n--- Metadados do PDF ---" -ForegroundColor Cyan
  $exif = Get-Command exiftool -ErrorAction SilentlyContinue
  $pdfinfo = Get-Command pdfinfo -ErrorAction SilentlyContinue
  if ($exif)        { & exiftool -Author -Title -Subject -Keywords -Creator -Producer $Pdf }
  elseif ($pdfinfo) { & pdfinfo $Pdf }
  else {
    Write-Host "exiftool/pdfinfo nao encontrados. Verifique manualmente em:" -ForegroundColor Yellow
    Write-Host "  Adobe Reader > Arquivo > Propriedades > Descricao"
  }
  Write-Host "`nLembre-se de dar Ctrl+F no PDF pelo seu sobrenome e pela sigla da instituicao."
}
