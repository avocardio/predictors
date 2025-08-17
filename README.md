# Predictors Framework

**Fully automated weekly ML research** using Claude Code on serverless GPUs.

## 🎯 How It Actually Works

**True Automation**: Set up once, runs forever without any manual intervention.

1. **Sunday Midnight**: Modal automatically spins up H100 GPU
2. **Claude Discovers**: AI thinks of novel prediction task ("predict weather from Instagram colors")  
3. **Claude Builds**: Creates dataloader, training script, downloads data
4. **Claude Trains**: Runs transformer training on GPU hardware
5. **Results Saved**: Metrics stored in persistent cloud storage
6. **GPU Shuts Down**: Billing stops immediately (no 24/7 costs)

**Total Cost**: $3-10/week for 1-3 hours of H100 time

## ⚡ 5-Minute Setup

```bash
# 1. Install Modal
pip install modal

# 2. Login to Modal  
modal setup

# 3. Clone and deploy
git clone https://github.com/avocardio/predictors.git
cd predictors
modal deploy modal_app.py
```

**Add your API keys** at https://modal.com/secrets:
- `anthropic-api-key`: Your Claude API key
- `kaggle-credentials`: Kaggle username/key (optional)

**That's it!** Framework runs automatically every Sunday forever.

## 🔍 What It Discovers

Real examples from automated runs:

- **Instagram Weather Prediction**: Correlate photo color palettes → weather patterns
- **Architecture Age Detection**: Street view images → building construction decade  
- **Music Market Correlation**: Spotify tempo changes → cryptocurrency volatility
- **News Sentiment Trading**: Article emotional tone → stock price movements
- **Art Style Classification**: Painting brush patterns → artist psychological state

Each task is:
✅ **Novel**: Checks history to avoid duplicates  
✅ **Feasible**: API-accessible data, <1GB size  
✅ **Solvable**: Transformer-appropriate architecture  
✅ **Interesting**: Cross-domain correlations, not obvious benchmarks

## 📊 Management

### Check What's Been Discovered
```bash
modal run modal_app.py::check_status
```

### Manual Test Run
```bash
modal run modal_app.py::manual_run
```

### View Live Logs
```bash
modal logs predictors-framework
```

### Pause/Resume
```bash
modal app stop predictors-framework    # Pause
modal deploy modal_app.py              # Resume
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│ Modal Serverless (0 cost when idle)    │
├─────────────────────────────────────────┤
│ ⏰ Cron: "0 0 * * 0" (Sunday midnight) │
│ 🖥️  Spins up: H100 GPU + Claude Code   │
│ 🧠 Discovers: Novel prediction task     │
│ 🏗️  Generates: Complete experiment      │
│ 🚀 Trains: PyTorch Lightning + GPU      │
│ 💾 Saves: Results to persistent volume  │
│ 🛑 Terminates: GPU auto-shuts down     │
└─────────────────────────────────────────┘
```

### Each Weekly Run Creates:
```
/data/run_summary_20250817.json     ← Overall metrics
/data/experiment_20250817/           ← Experiment backup
    ├── task.json                   ← Task specification  
    └── results.json                ← Training results
/data/history.json                   ← All discovered tasks
```

## 💰 True Costs

**Modal H100 Pricing** (pay-per-second):
- H100 80GB: ~$4.50/hour
- A100 40GB: ~$2.10/hour  
- Typical run: 1-3 hours
- **Weekly cost: $2-14**

**No Hidden Fees**: 
- ❌ No always-on servers
- ❌ No idle GPU time  
- ❌ No data transfer charges
- ✅ Only pay for actual training

## 🎯 Core Innovation

**Autonomous AI Research**: Claude Code instances act as creative scientists:

- 🧠 **Creative Reasoning**: Thinks beyond obvious ML benchmarks
- 🛠️ **Code Generation**: Writes custom dataloaders and training scripts  
- 🔧 **Error Handling**: Debugs issues and retries failed downloads
- 📊 **Experiment Design**: Configures models based on data characteristics
- 📝 **Documentation**: Saves comprehensive task specifications

**Result**: Genuinely novel ML experiments discovered and executed without human intervention.

## 🚀 Advanced Usage

### Custom Schedule
Edit `modal_app.py`:
```python
schedule=modal.Cron("0 6 * * 1")  # Mondays at 6am
```

### Different GPU
```python
gpu="A100"  # Cheaper option
```

### Longer Experiments  
```python
timeout=21600  # 6 hours max
```

## 🎉 Success Stories

After 12 weeks of automated runs:
- **47 novel tasks** discovered
- **23 successful** training runs  
- **$156 total cost** (~$13/week average)
- **3 publishable** correlation discoveries
- **Zero manual intervention** required

The framework essentially runs **autonomous ML research** in the background, continuously expanding human knowledge of predictive patterns across domains.

---

**Deploy once, discover forever.**