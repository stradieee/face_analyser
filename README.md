# FaceScope — Real-Time Facial Analysis

## Important: run it through a local web server

Do **not** double-click `index.html`. Use a local HTTP server so webcam permissions and model loading work correctly.

### Windows

If Python is installed:

```bat
cd face_analyzer_web
python -m http.server 8000
```

Then open:

`http://localhost:8000`

If you have Node.js:

```bat
cd face_analyzer_web
npx serve . -l 8000
```

Then open:

`http://localhost:8000`

## If it says “Models unavailable”

1. Make sure the computer has internet access the first time.
2. Click **Reload Models**.
3. Keep the browser DevTools Console open (`F12`) to see the exact network error.
4. If your school/work network blocks GitHub or jsDelivr, use another network or host the model files locally.
5. Allow camera access when prompted.

The app tries two public model sources automatically:
- face-api.js GitHub Pages
- face-api.js GitHub repository weights

## What is actually measured

- Face detection: pretrained Tiny Face Detector.
- Facial landmarks: pretrained 68-point landmark model.
- Age: model estimate.
- Gender: model classification (`Male`/`Female`), not gender identity.
- Mood: expression classifier; this is an estimated facial expression, not a reliable measurement of someone's internal emotional state.
- Face shape: heuristic based on landmarks and proportions.
- Glasses/beard: simple image heuristics and therefore only hints.

All webcam processing occurs in the browser after the models have loaded. No camera frames are sent to an application server by this project.
