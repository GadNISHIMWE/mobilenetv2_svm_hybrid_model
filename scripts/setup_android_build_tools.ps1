$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ToolsDir = Join-Path $Root ".android-build-tools"
$DownloadsDir = Join-Path $ToolsDir "downloads"
$JdkDir = Join-Path $ToolsDir "jdk"
$GradleDir = Join-Path $ToolsDir "gradle"
$SdkDir = Join-Path $ToolsDir "android-sdk"

$JdkUrl = "https://aka.ms/download-jdk/microsoft-jdk-17.0.18-windows-x64.zip"
$GradleUrl = "https://services.gradle.org/distributions/gradle-8.14.5-bin.zip"
$CmdlineToolsUrl = "https://dl.google.com/android/repository/commandlinetools-win-14742923_latest.zip"

New-Item -ItemType Directory -Force -Path $DownloadsDir, $JdkDir, $GradleDir, $SdkDir | Out-Null

function Download-IfMissing($Url, $Target) {
    if (Test-Path $Target) {
        Write-Host "Already downloaded: $Target"
        return
    }

    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Target
}

function Expand-Clean($Archive, $Destination) {
    if (Test-Path $Destination) {
        Write-Host "Already extracted: $Destination"
        return
    }

    $Temp = "$Destination.tmp"
    if (Test-Path $Temp) {
        Remove-Item -Recurse -Force $Temp
    }

    New-Item -ItemType Directory -Force -Path $Temp | Out-Null
    Expand-Archive -Path $Archive -DestinationPath $Temp
    Move-Item -Path $Temp -Destination $Destination
}

$JdkZip = Join-Path $DownloadsDir "microsoft-jdk-17.zip"
$GradleZip = Join-Path $DownloadsDir "gradle-bin.zip"
$CmdlineToolsZip = Join-Path $DownloadsDir "android-commandline-tools.zip"

Download-IfMissing $JdkUrl $JdkZip
Download-IfMissing $GradleUrl $GradleZip
Download-IfMissing $CmdlineToolsUrl $CmdlineToolsZip

Expand-Clean $JdkZip (Join-Path $JdkDir "expanded")
Expand-Clean $GradleZip (Join-Path $GradleDir "expanded")

$CmdlineLatest = Join-Path $SdkDir "cmdline-tools\latest"
if (-not (Test-Path $CmdlineLatest)) {
    $CmdlineTemp = Join-Path $ToolsDir "cmdline-tools-expanded"
    if (Test-Path $CmdlineTemp) {
        Remove-Item -Recurse -Force $CmdlineTemp
    }
    New-Item -ItemType Directory -Force -Path $CmdlineTemp | Out-Null
    Expand-Archive -Path $CmdlineToolsZip -DestinationPath $CmdlineTemp
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CmdlineLatest) | Out-Null
    Move-Item -Path (Join-Path $CmdlineTemp "cmdline-tools") -Destination $CmdlineLatest
    Remove-Item -Recurse -Force $CmdlineTemp
}

$JavaHome = Get-ChildItem -Path (Join-Path $JdkDir "expanded") -Directory | Select-Object -First 1
$GradleHome = Get-ChildItem -Path (Join-Path $GradleDir "expanded") -Directory | Select-Object -First 1

if (-not $JavaHome) {
    throw "Could not find extracted JDK folder."
}

if (-not $GradleHome) {
    throw "Could not find extracted Gradle folder."
}

$env:JAVA_HOME = $JavaHome.FullName
$env:ANDROID_HOME = $SdkDir
$env:ANDROID_SDK_ROOT = $SdkDir
$env:Path = "$($JavaHome.FullName)\bin;$($GradleHome.FullName)\bin;$SdkDir\cmdline-tools\latest\bin;$SdkDir\platform-tools;$env:Path"

Write-Host "Installing Android SDK packages ..."
sdkmanager.bat --sdk_root="$SdkDir" "platform-tools" "platforms;android-35" "build-tools;35.0.0"

Write-Host "Accepting Android SDK licenses ..."
"y`ny`ny`ny`ny`ny`ny`ny`ny`ny`n" | sdkmanager.bat --sdk_root="$SdkDir" --licenses

Write-Host ""
Write-Host "Android build tools are ready."
Write-Host "JAVA_HOME=$($JavaHome.FullName)"
Write-Host "ANDROID_HOME=$SdkDir"
Write-Host "GRADLE_HOME=$($GradleHome.FullName)"
