$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ToolsDir = Join-Path $Root ".android-build-tools"
$JdkExpanded = Join-Path $ToolsDir "jdk\expanded"
$GradleExpanded = Join-Path $ToolsDir "gradle\expanded"
$SdkDir = Join-Path $ToolsDir "android-sdk"
$NativeProject = Join-Path $Root "android-native"

$JavaHome = Get-ChildItem -Path $JdkExpanded -Directory | Select-Object -First 1
$GradleHome = Get-ChildItem -Path $GradleExpanded -Directory | Select-Object -First 1

if (-not $JavaHome) {
    throw "JDK not found. Run scripts\setup_android_build_tools.ps1 first."
}

if (-not $GradleHome) {
    throw "Gradle not found. Run scripts\setup_android_build_tools.ps1 first."
}

if (-not (Test-Path $SdkDir)) {
    throw "Android SDK not found. Run scripts\setup_android_build_tools.ps1 first."
}

$env:JAVA_HOME = $JavaHome.FullName
$env:ANDROID_HOME = $SdkDir
$env:ANDROID_SDK_ROOT = $SdkDir
$env:Path = "$($JavaHome.FullName)\bin;$($GradleHome.FullName)\bin;$SdkDir\cmdline-tools\latest\bin;$SdkDir\platform-tools;$env:Path"

Push-Location $NativeProject
try {
    gradle.bat assembleDebug
}
finally {
    Pop-Location
}

$Apk = Join-Path $NativeProject "app\build\outputs\apk\debug\app-debug.apk"
if (-not (Test-Path $Apk)) {
    throw "APK build finished but app-debug.apk was not found."
}

Write-Host ""
Write-Host "APK built successfully:"
Write-Host $Apk
