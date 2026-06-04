# Leaf Classifier Native Android

This is a plain Android app with no Expo dependency. It uploads a selected or captured image to the Flask backend at `/predict`.

## Build Requirements

Install Android Studio first. It includes:

- JDK
- Android SDK
- Gradle support

## Build APK

Open this `android-native` folder in Android Studio, let Gradle sync, then choose:

```text
Build > Build Bundle(s) / APK(s) > Build APK(s)
```

The APK will be created under:

```text
android-native/app/build/outputs/apk/debug/
```

## Backend

Run the Flask backend from the project root:

```powershell
venv\Scripts\python.exe app.py
```

The app defaults to:

```text
http://172.31.50.226:5000
```

You can edit that URL in the app if your laptop IP changes.
