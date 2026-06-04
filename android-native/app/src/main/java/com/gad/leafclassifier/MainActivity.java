package com.gad.leafclassifier;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    private static final int PICK_IMAGE = 10;
    private static final int TAKE_PHOTO = 11;

    private EditText backendInput;
    private ImageView preview;
    private TextView status;
    private TextView result;
    private ProgressBar progress;
    private byte[] selectedImageBytes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestBasicPermissions();
        buildUi();
    }

    private void requestBasicPermissions() {
        if (android.os.Build.VERSION.SDK_INT >= 23) {
            requestPermissions(
                    new String[] {
                            Manifest.permission.CAMERA,
                            Manifest.permission.READ_EXTERNAL_STORAGE,
                            "android.permission.READ_MEDIA_IMAGES"
                    },
                    1
            );
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(40, 48, 40, 32);
        root.setBackgroundColor(0xFFF4F7F3);

        TextView title = new TextView(this);
        title.setText("Leaf Classifier");
        title.setTextColor(0xFF163D2B);
        title.setTextSize(30);
        title.setGravity(Gravity.START);
        title.setTypeface(null, 1);
        root.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Native Android + Flask backend");
        subtitle.setTextColor(0xFF6C7C71);
        subtitle.setTextSize(15);
        root.addView(subtitle);

        backendInput = new EditText(this);
        backendInput.setText("http://172.31.50.226:5000");
        backendInput.setSingleLine(true);
        backendInput.setTextSize(14);
        backendInput.setHint("Backend URL");
        root.addView(backendInput, blockParams());

        preview = new ImageView(this);
        preview.setBackgroundColor(0xFFFFFFFF);
        preview.setScaleType(ImageView.ScaleType.CENTER_CROP);
        root.addView(preview, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(280)
        ));

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER);
        row.setPadding(0, dp(12), 0, 0);
        root.addView(row);

        Button galleryButton = new Button(this);
        galleryButton.setText("Gallery");
        galleryButton.setOnClickListener(v -> openGallery());
        row.addView(galleryButton, rowButtonParams());

        Button cameraButton = new Button(this);
        cameraButton.setText("Camera");
        cameraButton.setOnClickListener(v -> openCamera());
        row.addView(cameraButton, rowButtonParams());

        Button classifyButton = new Button(this);
        classifyButton.setText("Classify");
        classifyButton.setTextColor(0xFFFFFFFF);
        classifyButton.setBackgroundColor(0xFF1F6F3B);
        classifyButton.setOnClickListener(v -> classifyImage());
        root.addView(classifyButton, blockParams());

        progress = new ProgressBar(this);
        progress.setVisibility(View.GONE);
        root.addView(progress);

        status = new TextView(this);
        status.setTextColor(0xFF35463A);
        status.setTextSize(15);
        root.addView(status, blockParams());

        result = new TextView(this);
        result.setTextColor(0xFF163D2B);
        result.setTextSize(24);
        result.setTypeface(null, 1);
        root.addView(result, blockParams());

        setContentView(root);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE);
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, TAKE_PHOTO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) {
            return;
        }

        try {
            Bitmap bitmap;
            if (requestCode == PICK_IMAGE) {
                Uri imageUri = data.getData();
                InputStream input = getContentResolver().openInputStream(imageUri);
                bitmap = BitmapFactory.decodeStream(input);
            } else if (requestCode == TAKE_PHOTO) {
                bitmap = (Bitmap) data.getExtras().get("data");
            } else {
                return;
            }

            preview.setImageBitmap(bitmap);
            selectedImageBytes = bitmapToJpeg(bitmap);
            result.setText("");
            status.setText("Image selected. Ready to classify.");
        } catch (Exception ex) {
            status.setText("Could not load image: " + ex.getMessage());
        }
    }

    private void classifyImage() {
        if (selectedImageBytes == null) {
            status.setText("Choose an image first.");
            return;
        }

        progress.setVisibility(View.VISIBLE);
        status.setText("Sending image to backend ...");
        result.setText("");

        new Thread(() -> {
            try {
                String response = postImage(backendInput.getText().toString().trim() + "/predict", selectedImageBytes);
                JSONObject json = new JSONObject(response);

                runOnUiThread(() -> {
                    progress.setVisibility(View.GONE);
                    status.setText("Classification complete.");
                    result.setText(
                            json.optString("class", "Unknown")
                                    + "\nConfidence: "
                                    + json.optString("confidence", "-")
                                    + "%"
                    );
                });
            } catch (Exception ex) {
                runOnUiThread(() -> {
                    progress.setVisibility(View.GONE);
                    status.setText("Failed: " + ex.getMessage());
                });
            }
        }).start();
    }

    private String postImage(String endpoint, byte[] imageBytes) throws Exception {
        String boundary = "LeafClassifierBoundary";
        HttpURLConnection conn = (HttpURLConnection) new URL(endpoint).openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        conn.setRequestProperty("Accept", "application/json");

        DataOutputStream output = new DataOutputStream(conn.getOutputStream());
        output.writeBytes("--" + boundary + "\r\n");
        output.writeBytes("Content-Disposition: form-data; name=\"image\"; filename=\"leaf.jpg\"\r\n");
        output.writeBytes("Content-Type: image/jpeg\r\n\r\n");
        output.write(imageBytes);
        output.writeBytes("\r\n--" + boundary + "--\r\n");
        output.flush();
        output.close();

        InputStream input = conn.getResponseCode() >= 400 ? conn.getErrorStream() : conn.getInputStream();
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[4096];
        int read;
        while ((read = input.read(chunk)) != -1) {
            buffer.write(chunk, 0, read);
        }
        return buffer.toString(StandardCharsets.UTF_8.name());
    }

    private byte[] bitmapToJpeg(Bitmap bitmap) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 88, output);
        return output.toByteArray();
    }

    private LinearLayout.LayoutParams blockParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        params.setMargins(0, dp(14), 0, 0);
        return params;
    }

    private LinearLayout.LayoutParams rowButtonParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(52), 1);
        params.setMargins(dp(4), 0, dp(4), 0);
        return params;
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density);
    }
}
