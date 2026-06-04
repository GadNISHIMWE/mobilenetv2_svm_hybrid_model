import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import { Ionicons } from "@expo/vector-icons";
import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";

const DEFAULT_API_URL = "http://172.31.50.226:5000";

const STATUS_COLORS = {
  leaf: {
    background: "#dff3df",
    border: "#7ab87b",
    text: "#1f6f3b"
  },
  not_leaf: {
    background: "#ffe5e1",
    border: "#e17867",
    text: "#9d2f23"
  },
  uncertain: {
    background: "#fff1cb",
    border: "#e1a936",
    text: "#7a5208"
  }
};

export default function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState(false);

  const resultTheme = useMemo(() => {
    return STATUS_COLORS[result?.status] || STATUS_COLORS.uncertain;
  }, [result]);

  useEffect(() => {
    testBackend();
  }, []);

  async function pickImage(source) {
    setError("");
    setResult(null);
    setStatusMessage(source === "camera" ? "Opening camera ..." : "Opening gallery ...");

    try {
      const permission =
        source === "camera"
          ? await ImagePicker.requestCameraPermissionsAsync()
          : await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permission.granted) {
        setError("Permission was not granted. Allow photos/camera access in phone settings.");
        setStatusMessage("");
        return;
      }

      const picker =
        source === "camera"
          ? ImagePicker.launchCameraAsync
          : ImagePicker.launchImageLibraryAsync;

      const response = await picker({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.85
      });

      if (!response.canceled && response.assets?.length) {
        setImage(response.assets[0]);
        setStatusMessage("Image selected. Ready to classify.");
      } else {
        setStatusMessage("");
      }
    } catch (err) {
      setStatusMessage("");
      setError(err.message || "Could not open the image picker.");
    }
  }

  async function testBackend() {
    setError("");
    setStatusMessage("Testing backend connection ...");

    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/health`);
      const data = await response.json();

      if (!response.ok || data.status !== "healthy") {
        throw new Error(data.message || "Backend did not return healthy.");
      }

      setBackendHealthy(true);
      setStatusMessage("Backend is connected.");
    } catch (err) {
      setBackendHealthy(false);
      setStatusMessage("");
      setError(
        `Cannot reach backend at ${apiUrl}. Check that phone and laptop are on the same network.`
      );
    }
  }

  async function classifyImage() {
    if (!image) {
      setError("Select an image first.");
      return;
    }

    if (!backendHealthy) {
      setError("Please verify the backend URL before classifying.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setStatusMessage("Sending image to backend ...");

    const form = new FormData();
    form.append("image", {
      uri: image.uri,
      name: image.fileName || image.uri.split("/").pop() || "leaf.jpg",
      type: image.mimeType || image.type || "image/jpeg"
    });

    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        body: form,
        headers: {
          Accept: "application/json"
        }
      });

      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);
      setStatusMessage("Classification complete.");
    } catch (err) {
      setStatusMessage("");
      setError(err.message || "Could not reach the backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView
        style={styles.keyboard}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView contentContainerStyle={styles.container}>
          <View style={styles.header}>
            <Text style={styles.title}>Leaf Classifier</Text>
            <Text style={styles.subtitle}>MobileNetV2 + SVM</Text>
          </View>

          <View style={styles.panel}>
            <Text style={styles.label}>Backend URL</Text>
            <TextInput
              value={apiUrl}
              onChangeText={text => {
                setApiUrl(text);
                setBackendHealthy(false);
              }}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              placeholder="http://192.168.1.20:5000"
              style={styles.input}
            />
            <Pressable style={styles.testButton} onPress={testBackend}>
              <Ionicons name="wifi-outline" size={18} color="#1f6f3b" />
              <Text style={styles.testButtonText}>Test backend</Text>
            </Pressable>
            {!!backendHealthy && (
              <Text style={styles.backendStatus}>Backend reachable</Text>
            )}
          </View>

          <View style={styles.imageFrame}>
            {image ? (
              <Image source={{ uri: image.uri }} style={styles.preview} />
            ) : (
              <View style={styles.emptyPreview}>
                <Ionicons name="image-outline" size={54} color="#8aa093" />
                <Text style={styles.emptyText}>No image selected</Text>
              </View>
            )}
          </View>

          <View style={styles.actions}>
            <Pressable style={styles.secondaryButton} onPress={() => pickImage("library")}>
              <Ionicons name="images-outline" size={20} color="#1f6f3b" />
              <Text style={styles.secondaryButtonText}>Gallery</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={() => pickImage("camera")}>
              <Ionicons name="camera-outline" size={20} color="#1f6f3b" />
              <Text style={styles.secondaryButtonText}>Camera</Text>
            </Pressable>
          </View>

          <Pressable
            style={[styles.primaryButton, (!image || loading || !backendHealthy) && styles.disabledButton]}
            disabled={!image || loading || !backendHealthy}
            onPress={classifyImage}
          >
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <>
                <Ionicons name="scan-outline" size={20} color="#ffffff" />
                <Text style={styles.primaryButtonText}>Classify</Text>
              </>
            )}
          </Pressable>

          {!!statusMessage && (
            <View style={styles.statusBox}>
              <Text style={styles.statusText}>{statusMessage}</Text>
            </View>
          )}

          {!!error && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {result && (
            <View
              style={[
                styles.resultBox,
                { backgroundColor: resultTheme.background, borderColor: resultTheme.border }
              ]}
            >
              <Text style={[styles.resultClass, { color: resultTheme.text }]}>
                {result.class}
              </Text>
              <Text style={styles.resultMeta}>Confidence: {result.confidence}%</Text>
              <View style={styles.meterTrack}>
                <View
                  style={[
                    styles.meterFill,
                    {
                      width: `${Math.min(Math.max(result.confidence, 0), 100)}%`,
                      backgroundColor: resultTheme.text
                    }
                  ]}
                />
              </View>
              <View style={styles.metaRow}>
                <View>
                  <Text style={styles.metaLabel}>Image Size</Text>
                  <Text style={styles.metaValue}>{result.image_size}</Text>
                </View>
                <View>
                  <Text style={styles.metaLabel}>Features</Text>
                  <Text style={styles.metaValue}>{result.features}</Text>
                </View>
              </View>
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#f4f7f3"
  },
  keyboard: {
    flex: 1
  },
  container: {
    padding: 20,
    gap: 16
  },
  header: {
    paddingTop: 12,
    paddingBottom: 4
  },
  title: {
    fontSize: 30,
    fontWeight: "800",
    color: "#163d2b"
  },
  subtitle: {
    marginTop: 4,
    color: "#6c7c71",
    fontSize: 15
  },
  panel: {
    gap: 8
  },
  label: {
    color: "#35463a",
    fontWeight: "700"
  },
  input: {
    backgroundColor: "#ffffff",
    borderColor: "#cbd8cf",
    borderRadius: 8,
    borderWidth: 1,
    color: "#17251c",
    fontSize: 15,
    paddingHorizontal: 14,
    paddingVertical: 12
  },
  testButton: {
    alignItems: "center",
    alignSelf: "flex-start",
    flexDirection: "row",
    gap: 6,
    minHeight: 36,
    paddingRight: 8
  },
  testButtonText: {
    color: "#1f6f3b",
    fontSize: 14,
    fontWeight: "800"
  },
  imageFrame: {
    backgroundColor: "#ffffff",
    borderColor: "#d5e0d8",
    borderRadius: 8,
    borderWidth: 1,
    height: 330,
    overflow: "hidden"
  },
  preview: {
    height: "100%",
    width: "100%",
    resizeMode: "cover"
  },
  emptyPreview: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    gap: 10
  },
  emptyText: {
    color: "#6c7c71",
    fontSize: 15,
    fontWeight: "600"
  },
  actions: {
    flexDirection: "row",
    gap: 12
  },
  secondaryButton: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#bfd4c6",
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    flexDirection: "row",
    gap: 8,
    justifyContent: "center",
    minHeight: 48
  },
  secondaryButtonText: {
    color: "#1f6f3b",
    fontSize: 16,
    fontWeight: "700"
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: "#1f6f3b",
    borderRadius: 8,
    flexDirection: "row",
    gap: 8,
    justifyContent: "center",
    minHeight: 54
  },
  disabledButton: {
    backgroundColor: "#98aa9e"
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 17,
    fontWeight: "800"
  },
  errorBox: {
    backgroundColor: "#ffe5e1",
    borderColor: "#e17867",
    borderRadius: 8,
    borderWidth: 1,
    padding: 14
  },
  statusBox: {
    backgroundColor: "#e7f3ea",
    borderColor: "#9ac7a8",
    borderRadius: 8,
    borderWidth: 1,
    padding: 14
  },
  statusText: {
    color: "#1f6f3b",
    fontWeight: "700"
  },
  backendStatus: {
    color: "#1f6f3b",
    fontSize: 13,
    marginTop: 8,
    fontWeight: "700"
  },
  errorText: {
    color: "#9d2f23",
    fontWeight: "700"
  },
  resultBox: {
    borderRadius: 8,
    borderWidth: 1,
    padding: 18,
    gap: 10
  },
  resultClass: {
    fontSize: 26,
    fontWeight: "900"
  },
  resultMeta: {
    color: "#35463a",
    fontSize: 16,
    fontWeight: "700"
  },
  meterTrack: {
    backgroundColor: "rgba(255,255,255,0.75)",
    borderRadius: 999,
    height: 12,
    overflow: "hidden"
  },
  meterFill: {
    borderRadius: 999,
    height: "100%"
  },
  metaRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 4
  },
  metaLabel: {
    color: "#607066",
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase"
  },
  metaValue: {
    color: "#17251c",
    fontSize: 16,
    fontWeight: "800",
    marginTop: 2
  }
});
