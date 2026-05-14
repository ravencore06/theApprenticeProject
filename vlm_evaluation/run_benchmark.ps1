param(
    [string]$DataPath = "sample_dataset.json",
    [string]$ModelName = "llava-hf/llava-1.5-7b-hf",
    [switch]$NoQuantize = $false,
    [string]$OutputPath = "results.json",
    [int]$MaxNewTokens = 256
)

$QuantizeFlag = if ($NoQuantize) { "--no_quantize" } else { "" }

Write-Host "=== VLM Evaluation Benchmark ===" -ForegroundColor Cyan
Write-Host "Dataset : $DataPath"
Write-Host "Model   : $ModelName"
Write-Host "Quantize: $(-not $NoQuantize)"
Write-Host "Output  : $OutputPath"
Write-Host ""

python evaluate.py `
    --data_path $DataPath `
    --model_name $ModelName `
    $QuantizeFlag `
    --output_path $OutputPath `
    --max_new_tokens $MaxNewTokens

if ($LASTEXITCODE -eq 0) {
    Write-Host "Benchmark completed successfully." -ForegroundColor Green
} else {
    Write-Host "Benchmark failed with exit code $LASTEXITCODE." -ForegroundColor Red
}
