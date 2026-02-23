# Fake Human Face Image Detector (Website Skeleton)

Static website skeleton for a fake-human-face detector product.

## Included in this skeleton
- Upload input (`file`) and image link input (`URL`)
- Major image format validation (`jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`)
- Max file size validation (`100MB`) for uploads
- Single-face policy messaging (rejects images with multiple faces in demo flow)
- Placeholder "no face detected" flow
- Placeholder prediction output:
  - `label` (`Real` or `AI-generated`)
  - `confidence` (`0-100%`)
  - `reasoning` list
- Threshold rule in demo flow: `AI-generated` when AI confidence is `>= 50%`
- Mobile-friendly UI

## Not included yet
- Real face detection
- Real AI-vs-real model inference

## Step 2 backend API (implemented)
This repo now includes a Node/Express backend skeleton with validation and safety checks:

- `POST /api/predict/file` (`multipart/form-data`, field: `image`)
- `POST /api/predict/url` (`application/json`, body: `{ "imageUrl": "https://..." }`)
- `GET /api/health`

Rules enforced:
- Supported formats: `jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`
- Max file size: `100MB`
- URL safety checks:
  - only `http/https`
  - blocks localhost and private/loopback IP URL targets
  - verifies remote content-type and size limit

Current backend responses are mocked for face count/prediction logic (same as frontend mock behavior).

Current behavior is mocked in `app.js`.

## Local run
Install dependencies:

```bash
npm install
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

## Deploy to GitHub Pages
1. Create a GitHub repository.
2. Push this project.
3. In GitHub repo settings, enable Pages from the default branch root.
4. Your site will be available at `https://<your-username>.github.io/<repo-name>/`.

## Suggested next step
Replace backend mock inference in `server/index.js` with real face detection and model inference.
