# Deploy Guide — Get Your URL in 15 Minutes

You have a deadline. Here is the fastest path to a working public endpoint.

## Recommended: Render (free tier, no card required)

### Step 1 — Get the full SHL catalog (one-time, ~2 min)

The included `data/catalog.json` is a 32-item smoke-test subset. You need the full ~500-item catalog before submitting, otherwise Recall@10 will be artificially low.

**Option A** — if you already have the JSON file from the assignment ZIP:
```bash
cp /path/to/full_catalog.json data/catalog.json
```

**Option B** — scrape it fresh (~3 minutes):
```bash
pip install -r requirements.txt
python scripts/scrape_catalog.py
```

Verify:
```bash
python -c "import json; d=json.load(open('data/catalog.json')); print(f'{len(d)} items')"
```
You should see somewhere around 480–520 items.

### Step 2 — Push to GitHub (~3 min)

```bash
cd shl-agent
git init
git add -A
git commit -m "SHL recommender: initial submission"
gh repo create shl-recommender --public --source=. --push
# OR manually:
# git remote add origin https://github.com/<you>/shl-recommender.git
# git branch -M main
# git push -u origin main
```

### Step 3 — Deploy on Render (~5 min)

1. Go to https://render.com → sign up with GitHub.
2. **New +** → **Blueprint**.
3. Connect your `shl-recommender` repo. Render will detect `render.yaml`.
4. Click **Apply**.
5. On the service page, **Environment** tab → add:
   - `GEMINI_API_KEY` = `AIzaSyDIQUQQ7uLzm1ABMVD23YpZ6S8nitfft0A` (the key from the assignment email)
6. Click **Save Changes**. The service redeploys with the key set.
7. Wait ~3 min for build + first deploy. The URL will be something like `https://shl-recommender-XXXX.onrender.com`.

### Step 4 — Verify it works (~2 min)

```bash
python scripts/smoke.py https://shl-recommender-XXXX.onrender.com
```

You should see `✓` on all five checks. The first request after a cold start takes up to 2 minutes (the spec allows this); subsequent requests are fast.

If `llm_available` is `false` in `/info`, your `GEMINI_API_KEY` env var didn't apply — go back to step 3.6 and verify, then click **Manual Deploy** → **Deploy latest commit**.

### Step 5 — Submit

The submission form asks for two things:

1. **Public API endpoint URL**: `https://shl-recommender-XXXX.onrender.com`
2. **Approach document (PDF, max 2 pages)**: convert `APPROACH.md` to PDF — easiest is paste into Google Docs and File → Download → PDF.

Submit at the form linked in the assignment email.

---

## Alternative: Railway (if Render is slow)

```bash
npm i -g @railway/cli
railway login
railway init
railway up
railway variables set GEMINI_API_KEY=AIzaSyDIQUQQ7uLzm1ABMVD23YpZ6S8nitfft0A
railway domain   # generates a public URL
```

---

## Troubleshooting

**"My endpoint times out on cold start."** Free Render dynos sleep after inactivity. The spec allows 2 minutes for the first `/health`. The harness will handle this. To stay warm during evaluation, you can hit `/health` from a cron-job.org schedule every 14 minutes.

**"Recall@10 is low."** Make sure you replaced the catalog (Step 1) and that `GEMINI_API_KEY` is set (so embeddings are used). Re-deploy after fixing.

**"Embeddings build is slow on first request."** Expected — ~30 seconds for ~500 items. They cache to disk after that. If you want to pre-build, add `python -c "from app.main import app"` to your `buildCommand`.

**"I want to test locally first."** `cp .env.example .env`, edit, then `uvicorn app.main:app --reload`. Open http://127.0.0.1:8000.
