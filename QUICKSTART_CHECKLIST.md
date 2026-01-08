# Quick Start Checklist

Follow these steps in order to get your electricity price prediction system running with daily automation.

**What you'll build:** A complete ML system that automatically collects electricity and weather data, trains a prediction model, generates daily forecasts, and displays them in a web interface - all running on free cloud services!

**Time required:** 1-2 hours (mostly waiting for data to download)

---

## ✅ Step 1: Hopsworks Setup (10 minutes)

**What is Hopsworks?** Hopsworks is a feature store - a database designed specifically for machine learning. It stores your processed data (features) and trained models in the cloud, making them accessible from anywhere.

**Why do we need it?** Instead of storing data locally on your computer, Hopsworks allows GitHub Actions (our automation system) to access the same data and models in the cloud. This enables daily automated predictions.

### 1.1 Create Account & Get API Key

**What you're doing:** Creating a free Hopsworks account and getting a secret key (API key) that allows your code to access it.

```bash
# 1. Go to: https://app.hopsworks.ai/
# 2. Click "Sign up" (free tier available)
# 3. Verify your email address
# 4. After logging in, click "Create New Project"
# 5. Project name: "electricity_price_predictor" (must match exactly)
# 6. Click on your profile icon (top right) → Settings → API Keys
# 7. Click "Generate New API Key"
# 8. IMPORTANT: Copy the key immediately - you won't see it again!
```

**Expected result:** You should have a long string starting with something like `hW3k...` - this is your API key.

### 1.2 Configure Locally

**What you're doing:** Saving your API key in a local file (`.env`) so your scripts can access Hopsworks. The `.env` file is gitignored, keeping your key private.

```bash
cd ~/electricity-price-predictor

# Create .env file with your API key (replace 'your-key-here' with your actual key)
echo "HOPSWORKS_API_KEY=your-key-here" > .env
echo "HOPSWORKS_PROJECT_NAME=electricity_price_predictor" >> .env

# Verify the file was created correctly
cat .env
# Should show:
# HOPSWORKS_API_KEY=hW3k... (your key)
# HOPSWORKS_PROJECT_NAME=electricity_price_predictor

# Install dependencies (including Hopsworks client library)
pip install -r requirements.txt
# This takes ~30 seconds and installs all required Python packages
```

**Expected result:** No errors during installation. You may see a harmless urllib3 warning - ignore it.

---

## ✅ Step 2: Load Data to Hopsworks (20-30 minutes)

**What you're doing:** Collecting historical electricity prices and weather data from public APIs, processing them into machine learning features, and uploading them to Hopsworks. This is called "backfilling" - filling in past data.

**Why this takes time:** We're downloading years of hourly data from multiple sources. The more data we have, the better our predictions will be.

### 2.1 Test Connection (Small Dataset)

**What you're doing:** Testing that everything works by collecting just 7 days of data. This is a quick sanity check before committing to a long download.

```bash
# Test with last 7 days (takes ~1 minute)
python pipelines/feature_backfill.py --mode production --start-date 2024-12-22 --end-date 2024-12-28

# You should see output like:
# "Fetching weather data... ✅ Retrieved 7 weather records"
# "Fetching electricity prices... ✅ Retrieved 8 price records"
# "Saved 6 records to feature group 'electricity_price'"
```

**Verify it worked:**
1. Go to https://app.hopsworks.ai/
2. Click on your project: "electricity_price_predictor"
3. In the left sidebar, click "Feature Groups"
4. You should see "electricity_price_v1" with 6-7 rows

**Troubleshooting:** If you see "Invalid API key", double-check your `.env` file has the correct key.

### 2.2 Backfill Historical Data

**What you're doing:** Now that the test worked, collect a full year of data. Machine learning models need lots of examples to learn patterns - more data = better predictions!

**How much data do you need?**
- Minimum: 3 months (~90 samples) - Model will work but not great
- Recommended: 1 year (~365 samples) - Good accuracy
- Ideal: 2+ years (~730 samples) - Best results

```bash
# Load 1 year of data (recommended)
python pipelines/feature_backfill.py --mode production --start-date 2024-01-01 --end-date 2024-12-28

# This takes 10-30 minutes depending on your internet speed
# You'll see progress messages as it downloads month by month
# Go get coffee! ☕ The script will keep running.
```

**What's happening behind the scenes:**
1. Downloads daily weather data from OpenMeteo (free API)
2. Downloads hourly electricity prices from Swedish API
3. Merges and processes the data (creates features like temperature trends, price patterns)
4. Uploads processed data to Hopsworks
5. Shows progress: "Saved 365 records to feature group 'electricity_price'"

**Expected result:** You should see "✅ BACKFILL COMPLETE!" with 300+ records saved.

---

## ✅ Step 3: Train Model (5 minutes)

**What you're doing:** Training a machine learning model (XGBoost) using the data you just uploaded. The model learns patterns like "when temperature drops and wind increases, electricity prices usually rise."

**What is XGBoost?** A powerful machine learning algorithm that's excellent for time series prediction. It's used by many companies for forecasting.

**What happens during training:**
1. Loads your data from Hopsworks (365 rows of features + prices)
2. Splits it: 80% for training, 20% for testing
3. Trains the model to predict prices based on weather and time patterns
4. Tests accuracy on data it hasn't seen before
5. Saves the trained model to Hopsworks

```bash
python pipelines/training_pipeline.py --mode production

# You'll see output like:
# "Loading data... ✅ Loaded 365 records"
# "Training XGBoost model..."
# "📊 Model Performance:"
# "  Test R²: 0.91" (closer to 1.0 = better)
# "  Test RMSE: 0.12 SEK/kWh" (lower = better)
```

**Understanding the metrics:**
- **R² Score (0-1):** How well the model fits. 0.9+ is excellent, 0.7-0.9 is good
- **RMSE (Root Mean Square Error):** Average prediction error in SEK/kWh. Lower is better
- **MAE (Mean Absolute Error):** Average mistake size. For context, prices typically range 0.5-2.0 SEK/kWh

**Verify it worked:**
1. Go to https://app.hopsworks.ai/
2. Click "Model Registry" in the left sidebar
3. You should see "electricity_price_xgboost_default" with metrics

**Expected result:** Model trains successfully with R² > 0.85 (if you have 1 year of data).

---

## ✅ Step 4: Generate First Forecast (2 minutes)

**What you're doing:** Using your trained model to predict electricity prices for the next 7 days. The model looks at weather forecasts and recent price trends to make predictions.

**How it works:**
1. Loads your trained model from Hopsworks
2. Fetches latest weather forecast (next 7 days)
3. Prepares the data in the same format the model was trained on
4. Generates price predictions for each day
5. Creates visualizations and saves results

```bash
python pipelines/inference_pipeline.py --mode production --days 7

# You'll see output like:
# "Fetching weather forecast... ✅ Retrieved 7 days of forecast"
# "Generating predictions... ✅ Generated 7 predictions"
# "📊 Forecast Summary:"
# "  Date range: 2024-12-30 to 2025-01-05"
# "  Avg price: 0.85 SEK/kWh"
# "  Min price: 0.52 SEK/kWh"
# "  Max price: 1.24 SEK/kWh"
```

**Check your outputs:**
```bash
ls -lh outputs/
# You should see THREE new files:
# - forecast_20241229.png (graph showing 7-day predictions)
# - forecast_20241229.csv (data file with predictions)
# - prediction_tracking.csv (log of all predictions for comparison)
```

**What's in the files:**
- **PNG file:** Beautiful chart showing predicted prices for next 7 days
- **CSV file:** Predictions in spreadsheet format (can open in Excel)
- **Tracking file:** History of all predictions (used to compare predictions vs actual prices later)

**Expected result:** Three files created in `outputs/` folder, PNG shows a clear price forecast chart.

---

## ✅ Step 5: Test Gradio UI Locally (2 minutes)

**What you're doing:** Running a web interface (built with Gradio) that displays your forecasts in an interactive dashboard. This is what users will see when you deploy to the cloud.

**What is Gradio?** A Python library that turns your ML models into shareable web apps with just a few lines of code. It creates the beautiful interface automatically.

```bash
python app.py

# You'll see output like:
# "Running on local URL:  http://127.0.0.1:7860"
# "Running on public URL: https://xxxxx.gradio.live" (temporary link)
```

**Open in your browser:** Go to http://localhost:7860

**What you should see:**
- **Tab 1: "📈 7-Day Forecast"**
  - Interactive chart showing next 7 days of predictions
  - Data table with exact prices per day
  - You can hover over the chart to see details

- **Tab 2: "📊 Predicted vs Actual"**
  - Will say "No tracking data available yet"
  - This fills in over time as predictions are made daily

- **Tab 3: "ℹ️ About"**
  - Project information and how it works

**Try clicking around:**
- Click the "🔄 Refresh Data" button - updates with latest predictions
- Navigate between tabs
- The UI should be responsive and smooth

**Stop the app:** Press `Ctrl+C` in the terminal when done testing

**Expected result:** Clean, professional-looking web interface with your forecast displayed. No errors in browser console.

---

## ✅ Step 6: Enable GitHub Actions (5 minutes)

**What you're doing:** Setting up automation so your forecasts update daily without you having to run anything manually. GitHub Actions is like a robot that runs your code in the cloud on a schedule.

**What is GitHub Actions?** Free automation service from GitHub. It's like having a computer in the cloud that wakes up every day, runs your Python scripts, and saves the results - completely automatically!

**Why do we need this?** So the system keeps making predictions every day without you having to remember to run it manually. Set it and forget it!

### 6.1 Add Hopsworks Secret to GitHub

**What you're doing:** Giving GitHub Actions access to your Hopsworks account by securely storing your API key on GitHub. This allows the automation to download data and upload predictions.

**Why "secrets"?** API keys should never be visible in your code (security risk!). GitHub Secrets encrypts and hides them - only the automation can access them.

```bash
# 1. Go to your GitHub repository:
#    https://github.com/maxdougly/sml_project/settings/secrets/actions
#
# 2. Click the green "New repository secret" button
#
# 3. Fill in the form:
#    Name: HOPSWORKS_API_KEY (must be EXACTLY this - case sensitive!)
#    Value: [paste your Hopsworks API key from Step 1]
#
# 4. Click "Add secret"
#
# 5. You should now see "HOPSWORKS_API_KEY" in your secrets list
#    (the value will be hidden for security)
```

**Security note:** Never commit API keys to your code! Always use GitHub Secrets for sensitive information.

### 6.2 Enable Actions

**What you're doing:** Activating the automation workflow that's already in your repository (`.github/workflows/electricity-price-daily.yml`). GitHub disables workflows by default for security.

```bash
# 1. Go to your repository's Actions tab:
#    https://github.com/maxdougly/sml_project/actions
#
# 2. You'll see a message: "Workflows aren't being run on this repository"
#
# 3. Click the green button: "I understand my workflows, go ahead and enable them"
#
# 4. The page will refresh and you'll see "Electricity Price Daily Pipeline"
```

**What this enables:** A workflow that runs every day at 06:00 UTC, automatically:
- Collects yesterday's data
- Generates new 7-day forecast
- Updates your predictions

### 6.3 Test Manual Run

**What you're doing:** Running the automation manually (instead of waiting for the scheduled time) to verify everything works.

**This is important!** Better to find issues now than wait 24 hours to discover something's broken.

```bash
# 1. Go to Actions tab: https://github.com/maxdougly/sml_project/actions
#
# 2. In the left sidebar, click "Electricity Price Daily Pipeline"
#
# 3. You'll see a blue button "Run workflow" - click it
#
# 4. A dropdown appears:
#    - Branch: main (should be pre-selected)
#    - Mode: production (select this)
#    Click the green "Run workflow" button
#
# 5. Wait 5-10 minutes (refresh the page occasionally)
#
# 6. The workflow will show:
#    - ⏳ Orange circle = Running
#    - ✅ Green checkmark = Success!
#    - ❌ Red X = Failed (click to see logs)
```

**If it succeeds:**
- Click on the workflow run to see detailed logs
- Check that new forecast files were created
- You'll see each step: "Collect data", "Train model", "Generate forecast"

**If it fails:**
- Click on the failed step to see the error
- Common issues: Incorrect API key, network timeout
- Fix the issue and try again

**Expected result:** Green checkmark, workflow completes in 5-10 minutes with no errors.

---

## ✅ Step 7: Deploy to HuggingFace Spaces (15 minutes)

### 7.1 Create HuggingFace Account
```bash
# Go to: https://huggingface.co/join
```

### 7.2 Create New Space
```bash
# 1. Go to: https://huggingface.co/new-space
# 2. Space name: electricity-price-predictor
# 3. SDK: Gradio
# 4. Hardware: CPU (Basic, free)
# 5. Click "Create Space"
```

### 7.3 Push to HuggingFace
```bash
cd ~/electricity-price-predictor

# Add HuggingFace remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/electricity-price-predictor

# First, commit recent changes
git add .
git commit -m "Add Gradio UI and deployment configs"
git push origin main

# Push to HuggingFace
git push hf main

# ✓ Wait 2-5 minutes for build
# ✓ Visit: https://huggingface.co/spaces/YOUR_USERNAME/electricity-price-predictor
```

---

## ✅ Step 8: Auto-Update HuggingFace from GitHub Actions (10 minutes)

### 8.1 Get HuggingFace Token
```bash
# 1. Go to: https://huggingface.co/settings/tokens
# 2. Click "New token"
# 3. Name: GitHub Actions
# 4. Role: Write
# 5. Copy the token
```

### 8.2 Add Token to GitHub Secrets
```bash
# 1. Go to: https://github.com/maxdougly/sml_project/settings/secrets/actions
# 2. Click "New repository secret"
# 3. Name: HF_TOKEN
# 4. Value: [paste HuggingFace token]
# 5. Click "Add secret"
```

### 8.3 Update Workflow (I'll help you with this)
```bash
# Edit .github/workflows/electricity-price-daily.yml
# Add HuggingFace upload step
```

---

## 🎉 Done! Your System is Live

### What Happens Now?

**Every day at 06:00 UTC (automatically):**
1. 📊 GitHub Actions collects yesterday's data
2. 💾 Saves to Hopsworks Feature Store
3. 🤖 Generates new 7-day forecast
4. 📈 Creates visualization charts
5. ☁️ Pushes to HuggingFace Space
6. 🌐 Gradio UI updates automatically

---

## 📊 Monitoring Your System

### Check Health
- **GitHub Actions**: https://github.com/maxdougly/sml_project/actions
- **Hopsworks**: https://app.hopsworks.ai/
- **Gradio UI**: https://huggingface.co/spaces/YOUR_USERNAME/electricity-price-predictor

### View Logs
```bash
# GitHub Actions logs: Click on any workflow run
# HuggingFace logs: Settings → Logs
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module 'hopsworks' not found" | Run: `pip install -r requirements.txt` |
| "Invalid API key" | Check `.env` file has correct key |
| "Feature Group not found" | Run backfill first (Step 2) |
| "Model not found" | Run training first (Step 3) |
| Gradio shows "No data" | Run inference first (Step 4) |
| GitHub Actions fails | Check HOPSWORKS_API_KEY secret is set |

---

## 📖 Full Documentation

- **Detailed Guide**: `docs/DEPLOYMENT_GUIDE.md`
- **Architecture**: `docs/UNIFIED_APPROACH.md`
- **Visualizations**: `docs/VISUALIZATION_GUIDE.md`

---

**Total Setup Time**: ~1-2 hours (mostly waiting for data backfill)

**Recurring Cost**: $0 (all free tiers)

**Maintenance**: Fully automated! ✨
