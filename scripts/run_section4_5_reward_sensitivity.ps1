param(
    [int]$MaxParallel = 2,
    [string[]]$OnlyLabels = @()
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = 'D:\Program Files\anaconda\python.exe'
$entryScript = Join-Path $PSScriptRoot 'run_group_receptiveness_maml.py'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}
if ($MaxParallel -lt 1) {
    throw 'MaxParallel must be positive.'
}

$session = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = Join-Path $projectRoot "outputs\section4_5_reward_sensitivity_logs\$session"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$commonArguments = @(
    $entryScript,
    (Join-Path $projectRoot 'unused.pt'),
    '--task-mode', 'elasticity',
    '--guidance-mode', 'direct',
    '--direct-action-low', '0.01',
    '--direct-action-high', '0.99',
    '--direct-initial-recommendation', '0.5',
    '--direct-state-signal', 'adjustment_distance',
    '--response-interpolation', 'linear',
    '--task-split-mode', 'range_ood',
    '--elasticity-range-profile', 'wide',
    '--balanced-elasticity-batches',
    '--meta-iterations', '200',
    '--meta-batch-size', '4',
    '--support-episodes', '5',
    '--query-episodes', '10',
    '--validation-query-episodes', '48',
    '--test-query-episodes', '64',
    '--validation-interval', '5',
    '--inner-learning-rate', '0.8',
    '--meta-learning-rate', '0.00002',
    '--outer-update-epochs', '3',
    '--calibration-coefficient', '1.0',
    '--policy-gradient-coefficient', '0.05',
    '--fresh-meta-actor',
    '--meta-actor-initialization', 'static',
    '--residual-head-gain', '0.15',
    '--reward-mode', 'deficit',
    '--deficit-progress-weight', '1.0',
    '--deficit-round-cost', '0.01',
    '--deficit-success-bonus', '0.25',
    '--deficit-timeout-penalty', '0.25',
    '--deficit-epsilon', '0.00000001',
    '--recommendation-cost-weight', '0.01',
    '--remaining-deficit-cost-weight', '0.05',
    '--seed', '8203',
    '--task-split-seed', '2026',
    '--validation-case-seed', '41001',
    '--test-case-seed', '51001'
)

$variants = @(
    [pscustomobject]@{ Label = 'mod_1p0'; Modification = '1.0'; Unexecuted = '0.15' },
    [pscustomobject]@{ Label = 'mod_2p0'; Modification = '2.0'; Unexecuted = '0.15' },
    [pscustomobject]@{ Label = 'unexec_0p10'; Modification = '1.5'; Unexecuted = '0.10' },
    [pscustomobject]@{ Label = 'unexec_0p20'; Modification = '1.5'; Unexecuted = '0.20' }
)
if ($OnlyLabels.Count -gt 0) {
    $variants = @($variants | Where-Object { $_.Label -in $OnlyLabels })
    if ($variants.Count -eq 0) {
        throw 'None of the requested labels is defined.'
    }
}

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$running = @()
$completed = @()

foreach ($variant in $variants) {
    while ($running.Count -ge $MaxParallel) {
        $finished = $running | Where-Object { $_.Process.HasExited }
        if (-not $finished) {
            Start-Sleep -Seconds 2
            continue
        }
        foreach ($item in $finished) {
            $item.Process.WaitForExit()
            $item.Process.Refresh()
            $exitCode = $item.Process.ExitCode
            if ($null -ne $exitCode -and $exitCode -ne 0) {
                throw "Variant $($item.Label) failed. See $($item.ErrorLog)"
            }
            $completed += $item
        }
        $running = @($running | Where-Object { -not $_.Process.HasExited })
    }

    $stdout = Join-Path $logDir "$($variant.Label).out.log"
    $stderr = Join-Path $logDir "$($variant.Label).err.log"
    $arguments = $commonArguments + @(
        '--deficit-modification-cost', $variant.Modification,
        '--unexecuted-recommendation-cost-weight', $variant.Unexecuted
    )
    $startParameters = @{
        FilePath = $pythonExe
        ArgumentList = $arguments
        WorkingDirectory = $projectRoot
        RedirectStandardOutput = $stdout
        RedirectStandardError = $stderr
        WindowStyle = 'Hidden'
        PassThru = $true
    }
    $process = Start-Process @startParameters
    $running += [pscustomobject]@{
        Label = $variant.Label
        Process = $process
        OutputLog = $stdout
        ErrorLog = $stderr
    }
    Write-Output "Started $($variant.Label) as PID $($process.Id)"
}

foreach ($item in $running) {
    $item.Process.WaitForExit()
    $item.Process.Refresh()
    $exitCode = $item.Process.ExitCode
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw "Variant $($item.Label) failed. See $($item.ErrorLog)"
    }
    $completed += $item
}

Write-Output "All sensitivity variants completed. Logs: $logDir"
foreach ($item in $completed) {
    Write-Output "$($item.Label): $($item.OutputLog)"
}
