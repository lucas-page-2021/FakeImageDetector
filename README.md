# Fake Human Face Image Detector (Website Skeleton)

Static website skeleton for a fake-human-face detector product.

## Included in this skeleton
- Upload input (`file`) and image link input (`URL`)
- Major image format validation (`jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`)
- Max file size validation (`20MB`) for uploads
- Single-face policy messaging (rejects images with multiple faces in demo flow)
- Placeholder "no face detected" flow
- Placeholder prediction output:
  - `label` (`Real` or `AI-generated`)
  - `confidence` (`0-100%`)
  - `reasoning` list
- Threshold rule in demo flow: `AI-generated` when AI confidence is `>= 50%`
- Mobile-friendly UI

## Not included yet
- Advanced deep model inference (current Step 4 uses an archive-trained logistic baseline)

## Step 2 backend API (implemented)
This repo now includes a Node/Express backend skeleton with validation and safety checks:

- `POST /api/predict/file` (`multipart/form-data`, field: `image`)
- `POST /api/predict/url` (`application/json`, body: `{ "imageUrl": "https://..." }`)
- `GET /api/health`

Rules enforced:
- Supported formats: `jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`
- Max file size: `20MB`
- URL safety checks:
  - only `http/https`
  - blocks localhost and private/loopback IP URL targets
  - verifies remote content-type and size limit

## Step 3 face gate (implemented)
- Backend now runs real face detection (BlazeFace via TensorFlow.js Node) for both:
  - file uploads
  - URL-fetched images
- Enforced policy:
  - `0 faces` -> ask user to reupload a clear human face image
  - `>1 face` -> reject and ask for exactly one face
  - `1 face` -> continue to classification response

Step 4 classification is active when `server/model/classifier.json` exists.

## Step 4 classifier (implemented with archive dataset)
- Added a trainable classifier pipeline using your `archive/rvf10k` dataset.
- Trainer script:
  - `npm run train:classifier`
  - reads `archive/train.csv` and `archive/valid.csv`
  - extracts richer forensic features (color histograms, local binary patterns, edge/laplacian texture, blockiness)
  - trains regularized logistic regression with nonlinear (`zscore + squared`) feature expansion
  - writes model artifact to `server/model/classifier.json`
- Backend inference uses this artifact when present.
- If no artifact is found, backend falls back to mock classification.
- Current baseline (latest training artifact): validation accuracy `~67.2%` on a balanced 1,000-sample valid split.

## Step 5 reasoning (implemented)
- Reasoning is now generated dynamically per image from feature contribution analysis.
- The backend explains the strongest contributing forensic groups (for example: color distribution, micro-texture, edge structure, blockiness, lighting balance, channel correlation).
- API responses still return up to 3 short reasoning bullets per prediction.

Current behavior is mocked in `app.js`.

## Local run
Install dependencies:

```bash
npm install
```

Train classifier model (once, or whenever you retrain):

```bash
npm run train:classifier
```

Run backend API:

```bash
npm start
```

Serve frontend:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

Optional frontend API wiring:
- In browser devtools console, set:
  - `localStorage.setItem("apiBaseUrl", "http://localhost:8787")`
- Refresh the page.
- To return to pure mock mode:
  - `localStorage.removeItem("apiBaseUrl")`

## Make GitHub Pages use the real backend (Step 3 live)
GitHub Pages cannot run Node APIs. Deploy `server/index.js` separately, then point the frontend to that backend URL.

### Option A: Render (recommended)
1. Open [one-click deploy](https://render.com/deploy?repo=https://github.com/UrbanIntelligence/FakeProject).
2. Create the Node API service from `render.yaml`:
   - `fake-image-detector-api` (Node API)
3. Ensure environment variables include:
   - `ENABLE_TRANSFER_MODEL=1`
   - `TRANSFER_SERVICE_URL=https://scienceclub-mlmodel.hf.space`
   - `AI_THRESHOLD=60` (improves real-image accuracy while keeping fake accuracy high)
4. Wait for deploy, then copy your Node API URL (example: `https://fake-image-detector-api.onrender.com`).
5. Edit `config.js`:

```js
window.APP_CONFIG = {
  apiBaseUrl: "https://your-backend-url.onrender.com",
};
```

6. Commit and push `config.js` to `main`.
7. GitHub Pages will redeploy and start calling your live backend.

## Deploy to GitHub Pages
1. Create a GitHub repository.
2. Push this project.
3. In GitHub repo settings, enable Pages from the default branch root.
4. Your site will be available at `https://<your-username>.github.io/<repo-name>/`.

## Suggested next step
Upgrade from this logistic baseline to a stronger deep model and calibrate confidence on a held-out test set.

## Transfer-learning track (in progress)
- Added a PyTorch transfer-learning trainer at `ml/train_transfer.py` (ResNet18).
- Python deps are listed in `ml/requirements.txt`.
- Latest local checkpoint metric in `ml/artifacts/summary.json`:
  - `best_valid_acc: 0.794` (79.4%)
- Run command:
  - `python3 ml/train_transfer.py --epochs 15 --batch-size 32 --lr 2e-4 --weight-decay 1e-4 --train-per-class 1800 --valid-per-class 700 --num-workers 0 --freeze-backbone`
- Serving integration:
  - Backend can now use the transfer model first, then fallback to JS classifier.
  - Enable with environment variable: `ENABLE_TRANSFER_MODEL=1`
  - Configure transfer endpoint with: `TRANSFER_SERVICE_URL=https://scienceclub-mlmodel.hf.space`
  - Decision threshold is configurable with: `AI_THRESHOLD` (0-100, default `60`)
  - Python transfer service entrypoint: `transfer_service/app.py`
