# Setup Guide for Predictors Framework

## ✅ One-Time Setup (5 minutes)

### 1. Install Modal
```bash
pip install modal
```

### 2. Create Modal Account & Login
```bash
modal setup
```
Follow the prompts to create account and authenticate.

### 3. Create Secrets in Modal Dashboard

Go to https://modal.com/secrets and create:

**anthropic-secret**:
```
ANTHROPIC_API_KEY=your_claude_api_key_here
```

**kaggle** (recommended for dataset access):
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```
Get your Kaggle API credentials from: https://www.kaggle.com/settings/account

### 4. Deploy the App
```bash
git clone https://github.com/avocardio/predictors.git
cd predictors
modal deploy modal_app.py
```

## 🎉 That's It!

The framework will now automatically:
- ⏰ **Run every Sunday at midnight UTC**
- 🔍 **Discover novel prediction tasks**
- 🏗️ **Generate experiments automatically**  
- 🚀 **Train on A100 GPUs**
- 💾 **Save results to persistent storage**
- 💰 **Only charge for actual usage (~$3-10/week)**

## 🛠️ Management Commands

### Check Status
```bash
modal run modal_app.py::check_status
```

### Manual Test Run
```bash
modal run modal_app.py::manual_run
```

### View Logs
```bash
modal logs predictors-framework
```

### Pause Scheduling
```bash
modal app stop predictors-framework
```

### Resume Scheduling  
```bash
modal deploy modal_app.py
```

## 📊 Access Results

Results are automatically saved to Modal's persistent volume. Use `check_status` to see:
- Total tasks discovered
- Latest experiment results
- Run summaries with timestamps
- Error logs if any issues

The framework is completely hands-off once deployed!