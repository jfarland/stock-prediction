import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SP500DataCollector:
    """Collect and process S&P 500 stock data"""
    
    def __init__(self):
        self.sp500_tickers = self._get_sp500_tickers()
    
    def _get_sp500_tickers(self):
        """Get S&P 500 ticker symbols"""
        # Extended list of S&P 500 stocks for better training data
        sample_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
            'ABBV', 'PFE', 'AVGO', 'KO', 'MRK', 'COST', 'PEP', 'TMO', 'WMT',
            'DHR', 'MCD', 'ABT', 'ACN', 'VZ', 'ADBE', 'NEE', 'LIN', 'TXN',
            'NKE', 'RTX', 'CRM', 'PM', 'UPS', 'LOW', 'QCOM', 'T', 'HON',
            'INTC', 'AMD', 'IBM', 'ORCL', 'NFLX', 'DIS', 'PYPL', 'CMCSA',
            'WFC', 'GE', 'F', 'GM', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'PINS',
            'SQ', 'SHOP', 'ZM', 'CRM', 'SNOW', 'PLTR', 'RBLX', 'COIN',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'USO', 'TLT', 'HYG'
        ]
        return sample_tickers
    
    def download_data(self, period="2y"):
        """Download historical data for all S&P 500 stocks"""
        print(f"Downloading data for {len(self.sp500_tickers)} stocks...")
        
        all_data = {}
        failed_tickers = []
        
        for ticker in self.sp500_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty and len(hist) > 50:
                    all_data[ticker] = hist
                    print(f"✓ {ticker}: {len(hist)} days")
                else:
                    failed_tickers.append(ticker)
                    print(f"✗ {ticker}: Insufficient data")
                    
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"✗ {ticker}: Error - {str(e)}")
        
        print(f"\nSuccessfully downloaded: {len(all_data)} stocks")
        print(f"Failed: {len(failed_tickers)} stocks")
        
        return all_data
    
    def process_data(self, raw_data):
        """Process raw data into features and labels with enhanced feature set"""
        processed_data = []
        
        for ticker, df in raw_data.items():
            df = df.copy()
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']
            df['high_close_pct'] = (df['High'] - df['Close']) / df['Close']
            df['low_close_pct'] = (df['Close'] - df['Low']) / df['Close']
            
            # Volume features
            for window in [5, 10, 20]:
                df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_ma_{window}']
            
            # Price moving averages and ratios
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']
                
            # Exponential moving averages
            for window in [12, 26]:
                df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
                df[f'price_to_ema_{window}'] = df['Close'] / df[f'ema_{window}']
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            df['bb_std'] = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility measures
            for window in [10, 20, 50]:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
            # RSI with multiple periods
            for period in [14, 21]:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
            
            # Price momentum
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            
            # Average True Range
            df['tr1'] = df['High'] - df['Low']
            df['tr2'] = abs(df['High'] - df['Close'].shift())
            df['tr3'] = abs(df['Low'] - df['Close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # On Balance Volume
            df['obv'] = (np.sign(df['returns']) * df['Volume']).cumsum()
            df['obv_ma'] = df['obv'].rolling(window=20).mean()
            
            # Target: Next day direction
            df['next_open'] = df['Open'].shift(-1)
            df['next_close'] = df['Close'].shift(-1)
            df['target'] = (df['next_close'] > df['next_open']).astype(int)
            
            # Select enhanced features
            features = [
                'returns', 'log_returns', 'high_low_pct', 'open_close_pct', 
                'high_close_pct', 'low_close_pct',
                'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
                'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
                'price_to_ema_12', 'price_to_ema_26',
                'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'volatility_10', 'volatility_20', 'volatility_50',
                'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'williams_r',
                'momentum_5', 'momentum_10', 'momentum_20', 'atr', 'obv_ma'
            ]
            
            # Clean data
            df = df.dropna()
            
            if len(df) > 60:  # Ensure enough data for larger sequences
                df_clean = df[features + ['target']].copy()
                df_clean['ticker'] = ticker
                df_clean['date'] = df.index
                processed_data.append(df_clean)
        
        return pd.concat(processed_data, ignore_index=True)

class StockDataset(Dataset):
    """Enhanced PyTorch Dataset for stock time series data"""
    
    def __init__(self, data, sequence_length=30):
        self.sequence_length = sequence_length
        self.data = self._prepare_sequences(data)
    
    def _prepare_sequences(self, data):
        """Prepare sequential data for LSTM"""
        sequences = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            features = ticker_data.drop(['target', 'ticker', 'date'], axis=1).values
            targets = ticker_data['target'].values
            
            # Create sequences
            for i in range(len(features) - self.sequence_length):
                seq_features = features[i:i+self.sequence_length]
                seq_target = targets[i+self.sequence_length]
                sequences.append((seq_features, seq_target))
        
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, target = self.data[idx]
        return torch.FloatTensor(features), torch.LongTensor([target])

class LargeStockLSTM(nn.Module):
    """Significantly larger LSTM model for stock direction prediction"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.3):
        super(LargeStockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Larger LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for more capacity
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Multiple fully connected layers
        fc_hidden = hidden_size * 2
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size * 2, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.BatchNorm1d(fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_hidden // 2, fc_hidden // 4),
            nn.BatchNorm1d(fc_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_hidden // 4, 2)  # Binary classification
        ])
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output after attention
        last_output = attn_out[:, -1, :]
        
        # Pass through FC layers
        output = last_output
        for i, layer in enumerate(self.fc_layers):
            if isinstance(layer, nn.BatchNorm1d):
                output = layer(output)
            elif isinstance(layer, (nn.ReLU, nn.Dropout)):
                output = layer(output)
            else:  # Linear layers
                output = layer(output)
        
        return output

class BacktestEngine:
    """Backtesting engine for model predictions"""
    
    def __init__(self, model, scaler, sequence_length=30):
        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        
    def prepare_backtest_data(self, processed_data, backtest_days=30):
        """Prepare data for backtesting - split last N days"""
        # Sort by date
        processed_data = processed_data.sort_values('date')
        
        backtest_data = {}
        training_data = {}
        
        for ticker in processed_data['ticker'].unique():
            ticker_data = processed_data[processed_data['ticker'] == ticker].copy()
            
            if len(ticker_data) > backtest_days + self.sequence_length:
                # Split: last backtest_days for testing, rest for training
                split_idx = len(ticker_data) - backtest_days
                
                training_data[ticker] = ticker_data.iloc[:split_idx].copy()
                backtest_data[ticker] = ticker_data.iloc[split_idx-self.sequence_length:].copy()
        
        return training_data, backtest_data
    
    def run_backtest(self, backtest_data, feature_cols):
        """Run backtesting with 1-step ahead predictions"""
        results = []
        
        for ticker, data in backtest_data.items():
            data = data.reset_index(drop=True)
            ticker_results = []
            
            # Prepare features (excluding the first sequence_length days used for context)
            features = data[feature_cols].values
            
            for i in range(self.sequence_length, len(data)):
                # Get sequence for prediction
                sequence = features[i-self.sequence_length:i]
                sequence = sequence.reshape(1, self.sequence_length, -1)
                sequence_tensor = torch.FloatTensor(sequence).to(device)
                
                # Make prediction
                self.model.eval()
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    probs = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = probs.max().item()
                
                # Get actual result
                if i < len(data):
                    actual = data.iloc[i]['target']
                    date = data.iloc[i]['date']
                    
                    ticker_results.append({
                        'ticker': ticker,
                        'date': date,
                        'prediction': prediction,
                        'actual': actual,
                        'confidence': confidence,
                        'correct': prediction == actual
                    })
            
            results.extend(ticker_results)
        
        return pd.DataFrame(results)
    
    def analyze_backtest_results(self, results_df):
        """Analyze and report backtest results"""
        print("\n" + "="*60)
        print("BACKTESTING RESULTS ANALYSIS")
        print("="*60)
        
        # Overall accuracy
        overall_accuracy = results_df['correct'].mean()
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Predictions: {len(results_df):,}")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        print(f"   Baseline (Random): 50.0%")
        print(f"   Improvement: {(overall_accuracy - 0.5) * 100:+.1f} percentage points")
        
        # Accuracy by confidence levels
        print(f"\n🎯 ACCURACY BY CONFIDENCE LEVEL:")
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(confidence_bins)-1):
            low, high = confidence_bins[i], confidence_bins[i+1]
            mask = (results_df['confidence'] >= low) & (results_df['confidence'] < high)
            subset = results_df[mask]
            if len(subset) > 0:
                acc = subset['correct'].mean()
                print(f"   {low:.1f}-{high:.1f}: {acc:.1%} ({len(subset):,} predictions)")
        
        # High confidence predictions
        high_conf_threshold = 0.7
        high_conf_mask = results_df['confidence'] >= high_conf_threshold
        high_conf_results = results_df[high_conf_mask]
        
        if len(high_conf_results) > 0:
            high_conf_acc = high_conf_results['correct'].mean()
            print(f"\n🔥 HIGH CONFIDENCE PREDICTIONS (≥{high_conf_threshold:.0%}):")
            print(f"   Count: {len(high_conf_results):,} ({len(high_conf_results)/len(results_df):.1%} of total)")
            print(f"   Accuracy: {high_conf_acc:.1%}")
            print(f"   Improvement: {(high_conf_acc - 0.5) * 100:+.1f} percentage points")
        
        # Performance by stock
        print(f"\n📈 TOP 10 BEST PERFORMING STOCKS:")
        stock_performance = results_df.groupby('ticker').agg({
            'correct': ['count', 'mean']
        }).round(3)
        stock_performance.columns = ['predictions', 'accuracy']
        stock_performance = stock_performance[stock_performance['predictions'] >= 10]  # Min 10 predictions
        top_stocks = stock_performance.sort_values('accuracy', ascending=False).head(10)
        
        for ticker, row in top_stocks.iterrows():
            print(f"   {ticker}: {row['accuracy']:.1%} ({int(row['predictions'])} predictions)")
        
        print(f"\n📉 BOTTOM 5 WORST PERFORMING STOCKS:")
        bottom_stocks = stock_performance.sort_values('accuracy', ascending=True).head(5)
        for ticker, row in bottom_stocks.iterrows():
            print(f"   {ticker}: {row['accuracy']:.1%} ({int(row['predictions'])} predictions)")
        
        # Daily performance
        print(f"\n📅 PERFORMANCE OVER TIME:")
        daily_results = results_df.groupby('date')['correct'].agg(['count', 'mean']).round(3)
        daily_results.columns = ['predictions', 'accuracy']
        
        print(f"   Best Day: {daily_results['accuracy'].max():.1%}")
        print(f"   Worst Day: {daily_results['accuracy'].min():.1%}")
        print(f"   Average Daily Predictions: {daily_results['predictions'].mean():.0f}")
        
        # Prediction distribution
        pred_dist = results_df['prediction'].value_counts()
        print(f"\n🔮 PREDICTION DISTRIBUTION:")
        print(f"   Predicted Up (1): {pred_dist.get(1, 0):,} ({pred_dist.get(1, 0)/len(results_df):.1%})")
        print(f"   Predicted Down (0): {pred_dist.get(0, 0):,} ({pred_dist.get(0, 0)/len(results_df):.1%})")
        
        actual_dist = results_df['actual'].value_counts()
        print(f"\n📊 ACTUAL DISTRIBUTION:")
        print(f"   Actual Up (1): {actual_dist.get(1, 0):,} ({actual_dist.get(1, 0)/len(results_df):.1%})")
        print(f"   Actual Down (0): {actual_dist.get(0, 0):,} ({actual_dist.get(0, 0)/len(results_df):.1%})")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results_df['actual'], results_df['prediction'])
        
        print(f"\n📋 CONFUSION MATRIX:")
        print("           Predicted")
        print("         Down   Up")
        print(f"Down   {cm[0,0]:6d} {cm[0,1]:5d}")
        print(f"Up     {cm[1,0]:6d} {cm[1,1]:5d}")
        
        # Classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(results_df['actual'], results_df['prediction'])
        recall = recall_score(results_df['actual'], results_df['prediction'])
        f1 = f1_score(results_df['actual'], results_df['prediction'])
        
        print(f"\n📊 CLASSIFICATION METRICS:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        # Trading simulation (simple strategy)
        print(f"\n💰 SIMPLE TRADING STRATEGY SIMULATION:")
        print("   Strategy: Buy when predicted UP, Sell when predicted DOWN")
        
        correct_up = len(results_df[(results_df['prediction'] == 1) & (results_df['actual'] == 1)])
        total_up_predictions = len(results_df[results_df['prediction'] == 1])
        
        if total_up_predictions > 0:
            up_accuracy = correct_up / total_up_predictions
            print(f"   'Buy' Signal Accuracy: {up_accuracy:.1%} ({correct_up}/{total_up_predictions})")
            
            # Simulate returns (simplified)
            simulated_return = (correct_up * 0.01 - (total_up_predictions - correct_up) * 0.01)  # 1% per correct prediction
            print(f"   Simulated Return: {simulated_return:.2%} (simplified calculation)")
        
        return {
            'overall_accuracy': overall_accuracy,
            'high_confidence_accuracy': high_conf_acc if len(high_conf_results) > 0 else overall_accuracy,
            'total_predictions': len(results_df),
            'results_df': results_df
        }

def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.0005):
    """Train the large LSTM model with enhanced training"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.squeeze().to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets_list.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets_list, val_predictions)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {val_acc:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_large_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses, val_accuracies

def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', alpha=0.8)
    ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green', alpha=0.8)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function with backtesting"""
    print("=== Large S&P 500 Stock Direction Prediction with Backtesting ===\n")
    
    # 1. Data Collection
    collector = SP500DataCollector()
    raw_data = collector.download_data()
    
    # 2. Data Processing
    print("\nProcessing data with enhanced features...")
    processed_data = collector.process_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Features count: {processed_data.shape[1] - 3}")  # -3 for target, ticker, date
    print(f"Target distribution:\n{processed_data['target'].value_counts()}")
    
    # 3. Prepare data for backtesting
    backtest_engine = BacktestEngine(None, None, sequence_length=30)
    training_data_dict, backtest_data_dict = backtest_engine.prepare_backtest_data(processed_data, backtest_days=30)
    
    # Combine training data
    training_data = pd.concat(training_data_dict.values(), ignore_index=True)
    
    print(f"\nBacktesting setup:")
    print(f"Training data: {len(training_data)} samples")
    print(f"Backtest stocks: {len(backtest_data_dict)} stocks")
    print(f"Backtest period: Last 30 trading days")
    
    # 4. Feature scaling
    feature_cols = [col for col in training_data.columns if col not in ['target', 'ticker', 'date']]
    scaler = StandardScaler()
    training_data[feature_cols] = scaler.fit_transform(training_data[feature_cols])
    
    # Scale backtest data
    for ticker in backtest_data_dict:
        backtest_data_dict[ticker][feature_cols] = scaler.transform(backtest_data_dict[ticker][feature_cols])
    
    # 5. Train/Test Split
    train_data, test_data = train_test_split(training_data, test_size=0.15, 
                                           stratify=training_data['target'],
                                           random_state=42)
    
    train_data, val_data = train_test_split(train_data, test_size=0.15,
                                          stratify=train_data['target'],
                                          random_state=42)
    
    print(f"\nData splits:")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 6. Create Datasets and DataLoaders
    sequence_length = 30
    
    train_dataset = StockDataset(train_data, sequence_length)
    val_dataset = StockDataset(val_data, sequence_length)
    test_dataset = StockDataset(test_data, sequence_length)
    
    batch_size = 32  # Smaller batch size for larger model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # 7. Initialize Large Model
    input_size = len(feature_cols)
    model = LargeStockLSTM(input_size=input_size, hidden_size=256, num_layers=4, dropout=0.3)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLarge Model Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    
    # 8. Train Model
    print("\nStarting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=150, learning_rate=0.0005
    )
    
    # 9. Load best model for evaluation and backtesting
    model.load_state_dict(torch.load('best_large_model.pth'))
    
    # 10. Quick evaluation on test set
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.squeeze()
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(batch_targets.numpy())
    
    test_accuracy = accuracy_score(test_targets, test_predictions)
    print(f"\n=== Test Set Evaluation ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 11. Run Backtesting
    print("\n" + "="*60)
    print("STARTING BACKTESTING PHASE")
    print("="*60)
    
    backtest_engine.model = model
    backtest_engine.scaler = scaler
    
    backtest_results = backtest_engine.run_backtest(backtest_data_dict, feature_cols)
    
    # 12. Analyze Backtest Results
    analysis_results = backtest_engine.analyze_backtest_results(backtest_results)
    
    # 13. Plot results
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # 14. Additional visualizations for backtest results
    plot_backtest_results(backtest_results)
    
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✅ Model Training Complete")
    print(f"   - Training Accuracy: {max(val_accuracies):.1%}")
    print(f"   - Test Set Accuracy: {test_accuracy:.1%}")
    print(f"   - Model Parameters: {total_params:,}")
    
    print(f"\n✅ Backtesting Complete")
    print(f"   - Backtest Accuracy: {analysis_results['overall_accuracy']:.1%}")
    print(f"   - High Confidence Accuracy: {analysis_results['high_confidence_accuracy']:.1%}")
    print(f"   - Total Predictions: {analysis_results['total_predictions']:,}")
    print(f"   - Period: Last 30 trading days")
    
    if analysis_results['overall_accuracy'] > 0.55:
        print(f"\n🎉 Model shows promising predictive ability!")
    elif analysis_results['overall_accuracy'] > 0.52:
        print(f"\n👍 Model shows modest improvement over random.")
    else:
        print(f"\n📊 Model performance similar to random - consider feature engineering.")
    
    return model, backtest_results, analysis_results

def plot_backtest_results(results_df):
    """Plot comprehensive backtest results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Daily accuracy over time
    daily_acc = results_df.groupby('date')['correct'].mean()
    ax1.plot(daily_acc.index, daily_acc.values, marker='o', alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax1.set_title('Daily Prediction Accuracy Over Time')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy by confidence level
    confidence_bins = np.arange(0.5, 1.01, 0.05)
    conf_acc = []
    conf_counts = []
    
    for i in range(len(confidence_bins)-1):
        mask = (results_df['confidence'] >= confidence_bins[i]) & (results_df['confidence'] < confidence_bins[i+1])
        subset = results_df[mask]
        if len(subset) > 0:
            conf_acc.append(subset['correct'].mean())
            conf_counts.append(len(subset))
        else:
            conf_acc.append(0)
            conf_counts.append(0)
    
    ax2.bar(confidence_bins[:-1], conf_acc, width=0.04, alpha=0.7, color='skyblue')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax2.set_title('Accuracy by Confidence Level')
    ax2.set_xlabel('Confidence Level')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top performing stocks
    stock_perf = results_df.groupby('ticker').agg({
        'correct': ['count', 'mean']
    })
    stock_perf.columns = ['count', 'accuracy']
    stock_perf = stock_perf[stock_perf['count'] >= 5]  # At least 5 predictions
    top_stocks = stock_perf.sort_values('accuracy', ascending=False).head(15)
    
    ax3.barh(range(len(top_stocks)), top_stocks['accuracy'], alpha=0.7, color='lightgreen')
    ax3.set_yticks(range(len(top_stocks)))
    ax3.set_yticklabels(top_stocks.index)
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax3.set_title('Top 15 Stock Prediction Accuracy')
    ax3.set_xlabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Prediction confidence distribution
    ax4.hist(results_df['confidence'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(x=results_df['confidence'].mean(), color='blue', linestyle='--', 
                label=f'Mean: {results_df["confidence"].mean():.3f}')
    ax4.set_title('Distribution of Prediction Confidence')
    ax4.set_xlabel('Confidence Level')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Confusion matrix heatmap
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['actual'], results_df['prediction'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Down', 'Predicted Up'], 
                yticklabels=['Actual Down', 'Actual Up'])
    plt.title('Backtesting Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, backtest_results, analysis = main()
