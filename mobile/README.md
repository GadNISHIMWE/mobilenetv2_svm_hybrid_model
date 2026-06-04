# Leaf Classifier Mobile

React Native mobile client for the Flask + MobileNetV2/SVM backend.

## Run the Backend

From the project root:

```bash
python app.py
```

The backend now listens on `0.0.0.0:5000`, so phones on the same Wi-Fi can reach it.

## Run the Mobile App

From this folder:

```bash
npm install
npm start
```

Open the app with Expo Go or an emulator.

## Backend URL

The app currently defaults to:

```text
http://172.31.50.226:5000
```

If your computer's IP changes, enter your current local network address in the app:

```text
http://YOUR_COMPUTER_IP:5000
```

On Windows, you can find it with:

```powershell
ipconfig
```

Look for the IPv4 address under your active Wi-Fi adapter.
