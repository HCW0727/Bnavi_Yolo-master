/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.w3c.dom.Text;

import static android.speech.tts.TextToSpeech.ERROR;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private YoloV5Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    private TextToSpeech tts;

    public long time1 = System.currentTimeMillis();

    private static final String TAG = "DetectorActivity";




    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != ERROR) {
                    // 언어를 선택한다.
                    tts.setLanguage(Locale.KOREAN);
                }
            }
        });



        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        final int modelIndex = modelView.getCheckedItemPosition();
        final String modelString = modelStrings.get(modelIndex);

        try {
            detector = DetectorFactory.getDetector(getAssets(), modelString);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        int cropSize = detector.getInputSize();

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        LOGGER.i("DisplaySize" + previewWidth+'/'+ previewHeight);

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    protected void updateActiveModel() {
        // Get UI information before delegating to background
        final int modelIndex = modelView.getCheckedItemPosition();
        final int deviceIndex = deviceView.getCheckedItemPosition();
        String threads = threadsTextView.getText().toString().trim();
        final int numThreads = Integer.parseInt(threads);

        handler.post(() -> {
            if (modelIndex == currentModel && deviceIndex == currentDevice
                    && numThreads == currentNumThreads) {
                return;
            }
            currentModel = modelIndex;
            currentDevice = deviceIndex;
            currentNumThreads = numThreads;

            // Disable classifier while updating
            if (detector != null) {
                detector.close();
                detector = null;
            }

            // Lookup names of parameters.
            String modelString = modelStrings.get(modelIndex);
            String device = deviceStrings.get(deviceIndex);

            LOGGER.i("Changing model to " + modelString + " device " + device);

            // Try to load model.

            try {
                detector = DetectorFactory.getDetector(getAssets(), modelString);
                // Customize the interpreter to the type of device we want to use.
                if (detector == null) {
                    return;
                }
            }
            catch(IOException e) {
                e.printStackTrace();
                LOGGER.e(e, "Exception in updateActiveModel()");
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }


            if (device.equals("CPU")) {
                detector.useCPU();
            } else if (device.equals("GPU")) {
                detector.useGpu();
            } else if (device.equals("NNAPI")) {
                detector.useNNAPI();
            }
            detector.setNumThreads(numThreads);

            int cropSize = detector.getInputSize();
            croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            previewWidth, previewHeight,
                            cropSize, cropSize,
                            sensorOrientation, MAINTAIN_ASPECT);

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
        });
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        String[] classes = {"바리케이드가","벤치가","자전거가","볼라드가","버스가","자동차가","손수레가","고양이가","의자가","개가","소화전이","무인 단말기가",
        "오토바이가","안내판이","주차 요금 정산기가","사람이","가로등이","화분이","스쿠터가","정류장이","유모차가","책상이","신호등이","교통표지판이","가로수가","트럭이","휠체어가"};

        String[] classes_org = {"barricade","bench","bicycle","bollard","bus","car","carrier","cat","chair","dog","fire_hydrant","kiosk","motorcycle",
        "movable_signage","parking_meter","person","pole","potted_plant","scooter","stop","stroller","table","traffic_light","traffic_sign","tree_trunk",
        "truck","wheelchair"};


        Integer[] classes_threshold_5 = {0,0,0,1000,0,20000,0,0,0,0,0,0,0,3500,0,4500,0,0,0,0,0,0,0,0,0,0,0};
        Integer[] classes_threshold_20 = {0,0,0,100,0,700,0,0,0,0,0,0,0,3500,0,630,0,0,0,0,0,0,0,0,0,0,0};

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        Log.e("CHECK", "run: " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();

                            String[] str_result = result.toString().split(" ");
                            int int_result = Integer.parseInt(str_result[2].replaceAll("[^0-9]", ""));
                            LOGGER.i("Drawing Rect in " + int_result);

                            if (location != null && result.getConfidence() >= minimumConfidence && int_result >= 650) {
                                //canvas.drawRect(location, paint);

                                Log.d(TAG,"Entered canvas.drawRect");
                                LOGGER.i("Drawing Rect in " + result);




                                String[] test1 = location.toString().split(",");
                                String[] test2 = new String[4];

                                Log.d(TAG, String.valueOf(test2));

                                Float[] location_center = new Float[2];
                                Float[] boundary_size = new Float[2];



                                for (int i=0; i<=3; i++){
                                    test2[i] = test1[i].replaceAll("[^0-9,^.]", "");
                                }


                                location_center[0] = (Float.parseFloat(test2[0]) + Float.parseFloat(test2[2])) / 2;
                                location_center[1] = (Float.parseFloat(test2[1]) + Float.parseFloat(test2[3])) / 2;


                                boundary_size[0] = (Float.parseFloat(test2[2]) - Float.parseFloat(test2[0]));
                                boundary_size[1] = (Float.parseFloat(test2[3]) - Float.parseFloat(test2[1]));

                                Float b_size = boundary_size[0] * boundary_size[1];

                                int index = Arrays.binarySearch(classes_org,str_result[1]);


                                double distance;
                                if(b_size > classes_threshold_5[index]){
                                    distance = -1.0;
                                }else if(b_size < classes_threshold_20[index]){
                                    distance = -2.0;
                                }else{
                                    distance = (b_size - classes_threshold_5[index]) / (classes_threshold_20[index] - classes_threshold_5[index]) * 15.0 + 5.0;
                                    distance = Math.round(distance*10)/10.0;
                                }


                                //Log.d(TAG,"str_result : " + str_result[1] + " |  b_size : " + b_size + " | distance : " + distance);
                                Log.d(TAG, "distance"+distance);

                                result.distance = distance;
                                Log.d(TAG,"result = " + result);


                                //LOGGER.i("Drawing Rect in " + location_center[0].toString() + "  " + location_center[1].toString());



                                long time2 = System.currentTimeMillis();
                                if (time2 != -1) {
                                    long Differtime = time2 - time1;

                                    LOGGER.i("DiffrerTime" + Differtime);

                                    if (Differtime > 2000){
                                        time1 = System.currentTimeMillis();
                                        //int int_result = Integer.parseInt(str_result[0].replaceAll("[^0-9]", ""));



                                        if (b_size > classes_threshold_5[index]){
                                            if (location_center[0] < 100){
                                                tts.speak(classes[index] + " 왼쪽에 있습니다.", TextToSpeech.QUEUE_FLUSH,null);
                                            }
                                            else if (location_center[0] > 250){
                                                tts.speak(classes[index] + " 오른쪽에 있습니다.", TextToSpeech.QUEUE_FLUSH,null);
                                            }
                                            else{
                                                tts.speak(classes[index] + " 전방에 있습니다.", TextToSpeech.QUEUE_FLUSH,null);
                                            }
                                        }


                                    }
                                }


                                LOGGER.i("result : " + result);



                                //paint 뿌리는 위치 수정해주는 메소드
                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);

                                mappedRecognitions.add(result );

                                Log.d(TAG, "mappedRecognitions = " + String.valueOf(mappedRecognitions));


                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
