# Stock Prediction Service - Digital Ocean Deployment Guide

This guide walks you through deploying an automated stock prediction service on Digital Ocean that generates daily forecasts when market data becomes available.

## Overview

The prediction service:
- 🕰️ Runs automatically at 7:00 PM ET on weekdays (after market data is available)
- 📊 Generates predictions for 40+ S&P 500 stocks
- 💾 Saves predictions as JSON files with timestamps
- 📈 Includes confidence scores and metadata
- 🔍 Provides monitoring and alerting capabilities
- 💰 Costs approximately $6/month to run

## Prerequisites

Before starting, you'll need:
- Digital Ocean account
- Basic familiarity with Linux command line
- SSH client (built into macOS/Linux, use PuTTY on Windows)

## Step 1: Prepare Your Model

First, train and save your best model locally:

```bash
# Run experiments to find the best model
python enhanced_experiments.py

# This will create model files like:
# - best_simple_lstm_small.pth
# - best_gru_model.pth  
# - best_ensemble_lstm.pth
```

Note which model performs best from the experiment results.

## Step 2: Create Digital Ocean Droplet

1. **Login to Digital Ocean** and click "Create Droplet"

2. **Choose Configuration**:
   - **Image**: Ubuntu 22.04 LTS
   - **Size**: Basic plan, $6/month (1 GB RAM, 1 vCPU, 25 GB SSD)
   - **Region**: Choose closest to you (e.g., New York for US East)
   - **Authentication**: Add your SSH key or create a password

3. **Create the Droplet** and note the IP address

## Step 3: Generate Deployment Files

Run the production script to generate deployment configurations:

```bash
python production.py
```

This creates:
- `digital_ocean_deployment/setup.sh` - Automated setup script
- `digital_ocean_deployment/monitor.sh` - Monitoring script
- `production_pipeline/daily_prediction_pipeline.py` - Main prediction service
- `production_pipeline/requirements.txt` - Python dependencies

## Step 4: Upload Files to Droplet

Connect to your droplet and upload the necessary files:

```bash
# Connect to your droplet (replace YOUR_DROPLET_IP)
ssh root@YOUR_DROPLET_IP

# Create application directory
mkdir -p /opt/stock-predictor
cd /opt/stock-predictor
```

Now upload files from your local machine. Open a new terminal locally and run:

```bash
# Upload deployment files (replace YOUR_DROPLET_IP)
scp digital_ocean_deployment/* root@YOUR_DROPLET_IP:/opt/stock-predictor/
scp production_pipeline/* root@YOUR_DROPLET_IP:/opt/stock-predictor/
scp models/best_*.pth root@YOUR_DROPLET_IP:/opt/stock-predictor/models/
scp models/*/scaler.pkl root@YOUR_DROPLET_IP:/opt/stock-predictor/models/
scp models/*/feature_cols.json root@YOUR_DROPLET_IP:/opt/stock-predictor/models/
scp models/*/metadata.json root@YOUR_DROPLET_IP:/opt/stock-predictor/models/
```

## Step 5: Run Automated Setup

Back on your droplet, run the setup script:

```bash
cd /opt/stock-predictor
chmod +x setup.sh monitor.sh
./setup.sh
```

The setup script will:
- ✅ Install Python 3.9 and dependencies
- ✅ Create virtual environment
- ✅ Install required Python packages
- ✅ Setup systemd service for reliability
- ✅ Configure cron job for daily execution at 7 PM ET
- ✅ Create logging directories

## Step 6: Configure the Service

Edit the pipeline configuration if needed:

```bash
nano daily_prediction_pipeline.py
```

Key settings you might want to adjust:
- **Model path**: Update `model_dir` to point to your best model
- **Stock tickers**: Modify the `tickers` list to focus on specific stocks
- **Prediction timing**: Adjust schedule in the cron job if needed

## Step 7: Test the Service

Test the prediction pipeline manually:

```bash
# Activate virtual environment
source venv/bin/activate

# Run prediction pipeline manually
python daily_prediction_pipeline.py

# Check if predictions were generated
ls predictions/
cat predictions/predictions_*.json | head -20
```

Expected output:
```json
[
  {
    "ticker": "AAPL",
    "prediction_date": "2024-01-15T19:05:23.123456-05:00",
    "target_date": "2024-01-16T09:30:00.000000-05:00",
    "prediction": 1,
    "confidence": 0.73,
    "latest_close": 185.64,
    "latest_volume": 45123456,
    "model_version": "20240115_140523"
  }
]
```

## Step 8: Monitor the Service

### Check Service Status
```bash
# Check if service is running
sudo systemctl status stock-predictor.service

# Check recent logs
sudo journalctl -u stock-predictor.service --since today

# View cron job status
sudo tail -f /var/log/cron.log
```

### Use the Monitoring Script
```bash
# Run comprehensive monitoring
./monitor.sh
```

This shows:
- ✅ Whether predictions ran today
- 📊 Number of predictions generated
- 🎯 High confidence prediction count
- ⚠️ Any recent errors
- 🔧 Service health status

### Check Prediction Files
```bash
# List recent predictions
ls -la predictions/

# View today's predictions
TODAY=$(date +%Y%m%d)
cat predictions/predictions_${TODAY}_*.json | jq '.[0:3]'  # First 3 predictions

# Count predictions by day
ls predictions/ | cut -d'_' -f2 | sort | uniq -c
```

## Understanding the Automated Schedule

### Market Data Timing
- **Market Close**: 4:00 PM ET
- **Yahoo Finance Data Available**: ~6:00 PM ET (2 hours after close)
- **Prediction Service Runs**: 7:00 PM ET (safe margin)
- **Next Market Open**: 9:30 AM ET (14.5 hour lead time)

### Cron Schedule
The service runs at 7 PM ET, Monday through Friday:
```bash
# View current cron jobs
crontab -l

# The automatically installed job:
0 19 * * 1-5 /bin/systemctl start stock-predictor.service
```

## Using the Predictions

### Daily Prediction Format
Each prediction file contains an array of predictions:

```json
[
  {
    "ticker": "AAPL",
    "prediction_date": "2024-01-15T19:05:23-05:00",
    "target_date": "2024-01-16T09:30:00-05:00",
    "prediction": 1,          // 1 = UP (close > open), 0 = DOWN
    "confidence": 0.73,       // Model confidence (0.5-1.0)
    "latest_close": 185.64,
    "latest_volume": 45123456,
    "model_version": "20240115_140523"
  }
]
```

### Accessing Predictions Programmatically

Create a simple script to get latest predictions:

```python
import json
import glob
from datetime import datetime

def get_latest_predictions():
    """Get the most recent predictions"""
    prediction_files = glob.glob("/opt/stock-predictor/predictions/predictions_*.json")
    
    if not prediction_files:
        return None
    
    # Get most recent file
    latest_file = max(prediction_files)
    
    with open(latest_file, 'r') as f:
        predictions = json.load(f)
    
    return predictions

def get_high_confidence_predictions(min_confidence=0.7):
    """Get only high confidence predictions"""
    predictions = get_latest_predictions()
    
    if not predictions:
        return []
    
    return [p for p in predictions if p['confidence'] >= min_confidence]

# Example usage
high_conf_preds = get_high_confidence_predictions()
print(f"High confidence predictions: {len(high_conf_preds)}")

for pred in high_conf_preds[:5]:
    direction = "UP" if pred['prediction'] == 1 else "DOWN"
    print(f"{pred['ticker']}: {direction} (confidence: {pred['confidence']:.2f})")
```

### Setting Up Alerts

Get notified when predictions are ready:

```bash
# Create alert script
cat > /opt/stock-predictor/alert.sh << 'EOF'
#!/bin/bash

# Check if predictions were generated today
TODAY=$(date +%Y%m%d)
PRED_FILE="/opt/stock-predictor/predictions/predictions_${TODAY}_*.json"

if ls $PRED_FILE 1> /dev/null 2>&1; then
    COUNT=$(cat $PRED_FILE | jq '. | length')
    HIGH_CONF=$(cat $PRED_FILE | jq '[.[] | select(.confidence >= 0.7)] | length')
    
    echo "✅ Daily predictions ready!"
    echo "📊 Total: $COUNT predictions"
    echo "🎯 High confidence: $HIGH_CONF predictions"
    echo "📁 File: $PRED_FILE"
else
    echo "❌ No predictions found for today"
fi
EOF

chmod +x /opt/stock-predictor/alert.sh

# Add to cron to run 15 minutes after predictions (7:15 PM)
(crontab -l 2>/dev/null; echo "15 19 * * 1-5 /opt/stock-predictor/alert.sh") | crontab -
```

## Troubleshooting

### Common Issues

**1. Service not running predictions**
```bash
# Check service logs
sudo journalctl -u stock-predictor.service -f

# Manually test service
sudo systemctl start stock-predictor.service
```

**2. Data download issues**
```bash
# Test yfinance access
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='5d'))"

# Check internet connectivity
ping yahoo.com
```

**3. Missing model files**
```bash
# Verify model files exist
ls -la models/
ls -la models/*/

# Check file permissions
chmod 644 models/*.pth models/*/*.pkl models/*/*.json
```

**4. Timezone issues**
```bash
# Check timezone
timedatectl

# Set to Eastern Time if needed
sudo timedatectl set-timezone America/New_York
```

### Logs and Debugging

**Application logs**:
```bash
# Live tail of prediction logs
sudo tail -f /var/log/stock-predictor/daily.log

# Error logs
sudo tail -f /var/log/stock-predictor/error.log
```

**System logs**:
```bash
# Systemd service logs
sudo journalctl -u stock-predictor.service --since today

# Cron logs
sudo tail -f /var/log/syslog | grep CRON
```

## Maintenance

### Weekly Tasks
- Check prediction accuracy using the analysis notebook
- Monitor disk space: `df -h`
- Review error logs for any issues
- Verify predictions are being generated consistently

### Monthly Tasks
- Update Python packages: `pip install --upgrade -r requirements.txt`
- Retrain model if accuracy degrades
- Archive old prediction files to save space

### Updating the Model
```bash
# Stop the service
sudo systemctl stop stock-predictor.service

# Upload new model files
scp new_model.pth root@YOUR_DROPLET_IP:/opt/stock-predictor/models/

# Update the pipeline to use new model
nano daily_prediction_pipeline.py

# Restart service
sudo systemctl start stock-predictor.service
```

## Cost Management

**Monthly costs (approx. $6/month)**:
- Droplet: $6/month
- Bandwidth: Minimal (few MB/day)
- Storage: Minimal (~1GB for models and predictions)

**To minimize costs**:
- Use basic $6/month droplet (sufficient for this workload)
- Archive old predictions monthly
- Consider shutting down on weekends (predictions only needed weekdays)

## Security Considerations

- Change default passwords immediately
- Setup SSH key authentication
- Configure firewall: `ufw enable`
- Keep system updated: `apt update && apt upgrade`
- Consider setting up fail2ban for SSH protection

## Next Steps

Once your service is running:
1. Monitor predictions for a few weeks to establish baseline performance
2. Use the analysis notebook to evaluate prediction accuracy
3. Consider implementing position sizing based on confidence scores
4. Set up additional monitoring and alerting as needed
5. Explore ensemble methods combining multiple models

Your automated stock prediction service is now running! The system will generate fresh predictions every weekday evening, ready for the next trading day.