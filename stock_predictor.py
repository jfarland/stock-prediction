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
        # Using a subset for demonstration - in practice you'd get the full list
        # You can get the full list from Wikipedia or other sources
        sample_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
            'ABBV', 'PFE', 'AVGO', 'KO', 'MRK', 'COST', 'PEP', 'TMO', 'WMT',
            'DHR', 'MCD', 'ABT', 'ACN', 'VZ', 'ADBE', 'NEE', 'LIN', 'TXN',
            'NKE', 'RTX', 'CRM', 'PM', 'UPS', 'LOW', 'QCOM', 'T', 'HON'
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
                
                if not hist.empty and len(hist) > 50:  # Ensure sufficient data
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
        """Process raw data into features and labels"""
        processed_data = []
        
        for ticker, df in raw_data.items():
            # Calculate features
            df = df.copy()
            
            # Price-based features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']
            
            # Volume features
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # Technical indicators
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['price_to_sma5'] = df['Close'] / df['sma_5']
            df['price_to_sma20'] = df['Close'] / df['sma_20']
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Target: Next day direction (1 if close > open, 0 otherwise)
            df['next_open'] = df['Open'].shift(-1)
            df['next_close'] = df['Close'].shift(-1)
            df['target'] = (df['next_close'] > df['next_open']).astype(int)
            
            # Select features
            features = [
                'returns', 'log_returns', 'high_low_pct', 'open_close_pct',
                'volume_ratio', 'price_to_sma5', 'price_to_sma20', 
                'volatility', 'rsi'
            ]
            
            # Clean data
            df = df.dropna()
            
            if len(df) > 30:  # Ensure enough data
                df_clean = df[features + ['target']].copy()
                df_clean['ticker'] = ticker
                processed_data.append(df_clean)
        
        return pd.concat(processed_data, ignore_index=True)

class StockDataset(Dataset):
    """PyTorch Dataset for stock time series data"""
    
    def __init__(self, data, sequence_length=20):
        self.sequence_length = sequence_length
        self.data = self._prepare_sequences(data)
    
    def _prepare_sequences(self, data):
        """Prepare sequential data for LSTM"""
        sequences = []
        
        # Group by ticker
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_index()
            
            features = ticker_data.drop(['target', 'ticker'], axis=1).values
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

class StockLSTM(nn.Module):
    """LSTM model for stock direction prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """Train the LSTM model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
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
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {val_acc:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.squeeze()
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(batch_targets.numpy())
    
    accuracy = accuracy_score(targets, predictions)
    
    print("\n=== Model Evaluation ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=['Down', 'Up']))
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return accuracy

def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("=== S&P 500 Stock Direction Prediction ===\n")
    
    # 1. Data Collection
    collector = SP500DataCollector()
    raw_data = collector.download_data()
    
    # 2. Data Processing
    print("\nProcessing data...")
    processed_data = collector.process_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Target distribution:\n{processed_data['target'].value_counts()}")
    
    # 3. Feature scaling
    feature_cols = processed_data.columns[:-2]  # Exclude target and ticker
    scaler = StandardScaler()
    processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])
    
    # 4. Train/Test Split
    train_data, test_data = train_test_split(processed_data, test_size=0.2, 
                                           stratify=processed_data['target'],
                                           random_state=42)
    
    train_data, val_data = train_test_split(train_data, test_size=0.2,
                                          stratify=train_data['target'],
                                          random_state=42)
    
    print(f"\nData splits:")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 5. Create Datasets and DataLoaders
    sequence_length = 20
    
    train_dataset = StockDataset(train_data, sequence_length)
    val_dataset = StockDataset(val_data, sequence_length)
    test_dataset = StockDataset(test_data, sequence_length)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # 6. Initialize Model
    input_size = len(feature_cols)
    model = StockLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3)
    model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. Train Model
    print("\nStarting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=100, learning_rate=0.001
    )
    
    # 8. Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy = evaluate_model(model, test_loader)
    
    # 9. Plot results
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    print(f"\n=== Final Results ===")
    print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
