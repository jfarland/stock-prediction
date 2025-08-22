import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import neptune
import os
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ExperimentTracker:
    """Neptune experiment tracking wrapper with better error handling"""
    
    def __init__(self, project_name="jfarland/stock-prediction"):
        self.neptune_available = False
        self.run = None
        
        # Check if Neptune token is available
        api_token = os.getenv("NEPTUNE_API_TOKEN")
        
        if not api_token or api_token == "ANONYMOUS":
            print("⚠️  Neptune API token not found. Experiment tracking disabled.")
            print("   To enable Neptune tracking:")
            print("   1. Go to https://app.neptune.ai/")
            print("   2. Create account and project")
            print("   3. Get API token from profile settings")
            print("   4. Set: export NEPTUNE_API_TOKEN='your_token_here'")
            return
        
        try:
            # Try to initialize Neptune with better error handling
            print(f"🌊 Initializing Neptune project: {project_name}")
            
            self.run = neptune.init_run(
                project=project_name,
                api_token=api_token,
                mode="async",
                capture_stdout=False,
                capture_stderr=False,
                capture_hardware_metrics=False
            )
            self.neptune_available = True
            print("✅ Neptune tracking initialized successfully")
            
        except neptune.exceptions.ProjectNotFound as e:
            print(f"❌ Neptune project not found: {project_name}")
            print("   Please verify the project exists in Neptune UI:")
            print(f"   1. Go to https://app.neptune.ai/")
            print(f"   2. Check project exists: jfarland/stock-prediction")
            print(f"   3. Verify you have access to the project")
            
        except neptune.exceptions.NeptuneAuthTokenException as e:
            print(f"❌ Neptune authentication failed")
            print("   Check your API token is correct")
            
        except Exception as e:
            print(f"⚠️  Neptune initialization failed: {e}")
            print("   Continuing without experiment tracking...")
            
        # Fallback: disable Neptune but continue execution
        if not self.neptune_available:
            print("📊 Experiments will run without Neptune tracking")
    
    def log_hyperparameters(self, params: Dict):
        """Log hyperparameters"""
        if self.neptune_available:
            self.run["hyperparameters"] = params
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics"""
        if self.neptune_available:
            for key, value in metrics.items():
                if step is not None:
                    self.run[f"metrics/{key}"].log(value, step=step)
                else:
                    self.run[f"metrics/{key}"] = value
    
    def log_backtest_results(self, results: Dict):
        """Log backtest results"""
        if self.neptune_available:
            for key, value in results.items():
                self.run[f"backtest/{key}"] = value
    
    def finish(self):
        """Finish the run"""
        if self.neptune_available:
            self.run.stop()

class SP500DataCollector:
    """Enhanced data collector with better feature engineering"""
    
    def __init__(self):
        self.sp500_tickers = self._get_sp500_tickers()
    
    def _get_sp500_tickers(self):
        """Get a robust set of S&P 500 stocks"""
        # Focus on liquid, stable stocks for better predictions
        liquid_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
            'ABBV', 'PFE', 'AVGO', 'KO', 'MRK', 'COST', 'PEP', 'TMO', 'WMT',
            'DHR', 'MCD', 'ABT', 'ACN', 'VZ', 'ADBE', 'NEE', 'LIN', 'TXN',
            'NKE', 'RTX', 'CRM', 'PM', 'UPS', 'LOW', 'QCOM', 'T', 'HON',
            'INTC', 'AMD', 'IBM', 'ORCL', 'NFLX', 'DIS', 'CMCSA', 'WFC'
        ]
        return liquid_stocks
    
    def download_data(self, period="2y"):
        """Download with better error handling"""
        print(f"Downloading data for {len(self.sp500_tickers)} stocks...")
        
        all_data = {}
        failed_tickers = []
        
        for ticker in self.sp500_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval="1d")
                
                if not hist.empty and len(hist) > 100:  # Need more data
                    # Clean data
                    hist = hist.dropna()
                    if len(hist) > 100:
                        all_data[ticker] = hist
                        print(f"✓ {ticker}: {len(hist)} days")
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"✗ {ticker}: Error - {str(e)}")
        
        print(f"\nSuccessfully downloaded: {len(all_data)} stocks")
        return all_data
    
    def engineer_features(self, df, ticker):
        """Enhanced feature engineering with market regime awareness"""
        df = df.copy()
        
        # Basic price features with different lookbacks
        for lookback in [1, 2, 3, 5]:
            df[f'returns_{lookback}d'] = df['Close'].pct_change(lookback)
            df[f'log_returns_{lookback}d'] = np.log(df['Close'] / df['Close'].shift(lookback))
        
        # Intraday features
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['oc_ratio'] = (df['Close'] - df['Open']) / df['Open']
        df['ho_ratio'] = (df['High'] - df['Open']) / df['Open']
        df['lo_ratio'] = (df['Low'] - df['Open']) / df['Open']
        
        # Volume features with normalization
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_momentum'] = df['Volume'] / df['Volume'].shift(1)
        
        # Price momentum and mean reversion
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['Close'].rolling(window).mean()
            df[f'price_sma_ratio_{window}'] = df['Close'] / df[f'sma_{window}']
            df[f'sma_slope_{window}'] = df[f'sma_{window}'].diff(5) / df[f'sma_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['returns_1d'].rolling(window).std()
            df[f'volatility_ratio_{window}d'] = df[f'volatility_{window}d'] / df[f'volatility_{window}d'].shift(window)
        
        # Technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_momentum'] = df['rsi'].diff(5)
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = df['bb_std'] / df['bb_middle']  # Volatility squeeze indicator
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Market microstructure
        df['price_acceleration'] = df['returns_1d'].diff()
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_end'] = (df.index.day > 25).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        
        # Target engineering - multiple targets to test
        df['next_open'] = df['Open'].shift(-1)
        df['next_close'] = df['Close'].shift(-1)
        df['next_high'] = df['High'].shift(-1)
        df['next_low'] = df['Low'].shift(-1)
        
        # Primary target: direction
        df['target'] = (df['next_close'] > df['next_open']).astype(int)
        
        # Alternative targets for experimentation
        df['target_strong'] = ((df['next_close'] - df['next_open']) / df['next_open'] > 0.005).astype(int)  # >0.5% move
        df['target_intraday_high'] = (df['next_high'] > df['next_open'] * 1.01).astype(int)  # Intraday high >1%
        
        return df
    
    def process_data(self, raw_data):
        """Process with enhanced feature engineering"""
        processed_data = []
        
        for ticker, df in raw_data.items():
            try:
                df_features = self.engineer_features(df, ticker)
                
                # Select features (excluding target columns)
                feature_cols = [col for col in df_features.columns 
                              if not col.startswith(('target', 'next_')) and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                
                df_clean = df_features[feature_cols + ['target', 'target_strong', 'target_intraday_high']].copy()
                df_clean = df_clean.dropna()
                
                if len(df_clean) > 50:
                    df_clean['ticker'] = ticker
                    df_clean['date'] = df_features.index[:len(df_clean)]
                    processed_data.append(df_clean)
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        return pd.concat(processed_data, ignore_index=True) if processed_data else pd.DataFrame()

class ModelExperiments:
    """Collection of different model architectures to experiment with"""
    
    @staticmethod
    def create_simple_lstm(input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """Experiment 1: Simple LSTM - back to basics"""
        class SimpleLSTM(nn.Module):
            def __init__(self):
                super(SimpleLSTM, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 2)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.dropout(lstm_out[:, -1, :])
                return self.fc(output)
        
        return SimpleLSTM()
    
    @staticmethod
    def create_gru_model(input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """Experiment 2: GRU instead of LSTM"""
        class GRUModel(nn.Module):
            def __init__(self):
                super(GRUModel, self).__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 2)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                gru_out, _ = self.gru(x)
                output = self.dropout(gru_out[:, -1, :])
                return self.fc(output)
        
        return GRUModel()
    
    @staticmethod
    def create_cnn_lstm(input_size, hidden_size=64, dropout=0.2):
        """Experiment 3: CNN + LSTM hybrid"""
        class CNNLSTM(nn.Module):
            def __init__(self):
                super(CNNLSTM, self).__init__()
                # 1D CNN for local pattern detection
                self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                
                # LSTM for temporal modeling
                self.lstm = nn.LSTM(32, hidden_size, 2, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 2)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # CNN path
                x_cnn = x.transpose(1, 2)  # (batch, features, time)
                x_cnn = torch.relu(self.conv1(x_cnn))
                x_cnn = torch.relu(self.conv2(x_cnn))
                x_cnn = x_cnn.transpose(1, 2)  # Back to (batch, time, features)
                
                # LSTM path
                lstm_out, _ = self.lstm(x_cnn)
                output = self.dropout(lstm_out[:, -1, :])
                return self.fc(output)
        
        return CNNLSTM()
    
    @staticmethod
    def create_transformer_model(input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        """Experiment 4: Transformer model"""
        class TransformerModel(nn.Module):
            def __init__(self):
                super(TransformerModel, self).__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(100, d_model))  # Max sequence length 100
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                self.fc = nn.Linear(d_model, 2)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                seq_len = x.size(1)
                x = self.input_projection(x)
                x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
                x = self.transformer(x)
                output = self.dropout(x[:, -1, :])
                return self.fc(output)
        
        return TransformerModel()
    
    @staticmethod
    def create_ensemble_lstm(input_size, hidden_size=64, dropout=0.2):
        """Experiment 5: Ensemble of LSTMs with different configurations"""
        class EnsembleLSTM(nn.Module):
            def __init__(self):
                super(EnsembleLSTM, self).__init__()
                # Multiple LSTM branches
                self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
                self.lstm2 = nn.LSTM(input_size, hidden_size//2, 2, batch_first=True, dropout=dropout)
                self.lstm3 = nn.LSTM(input_size, hidden_size*2, 1, batch_first=True)
                
                # Combine outputs
                total_hidden = hidden_size + hidden_size//2 + hidden_size*2
                self.fc = nn.Linear(total_hidden, 2)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                out1, _ = self.lstm1(x)
                out2, _ = self.lstm2(x)
                out3, _ = self.lstm3(x)
                
                # Take last outputs and concatenate
                combined = torch.cat([
                    out1[:, -1, :],
                    out2[:, -1, :],
                    out3[:, -1, :]
                ], dim=1)
                
                output = self.dropout(combined)
                return self.fc(output)
        
        return EnsembleLSTM()
    
    @staticmethod
    def create_random_forest():
        """Experiment 6: Random Forest baseline"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def create_logistic_regression():
        """Experiment 7: Logistic Regression baseline"""
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

class StockDataset(Dataset):
    """Enhanced Dataset with flexible sequence length"""
    
    def __init__(self, data, sequence_length=20, target_col='target'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.data = self._prepare_sequences(data)
    
    def _prepare_sequences(self, data):
        sequences = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            features = ticker_data.drop(['target', 'target_strong', 'target_intraday_high', 'ticker', 'date'], axis=1).values
            targets = ticker_data[self.target_col].values
            
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

class ExperimentRunner:
    """Runs multiple experiments with different configurations"""
    
    def __init__(self, processed_data, tracker):
        self.processed_data = processed_data
        self.tracker = tracker
        self.experiments = []
        
    def setup_experiments(self):
        """Define all experiments to run"""
        
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['target', 'target_strong', 'target_intraday_high', 'ticker', 'date']]
        input_size = len(feature_cols)
        
        experiments = [
            # Experiment 1: Simple LSTM with different hyperparameters
            {
                'name': 'simple_lstm_small',
                'model_fn': lambda: ModelExperiments.create_simple_lstm(input_size, 32, 1, 0.1),
                'type': 'neural',
                'sequence_length': 10,
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 50,
                'target': 'target'
            },
            
            # Experiment 2: GRU Model
            {
                'name': 'gru_model',
                'model_fn': lambda: ModelExperiments.create_gru_model(input_size, 64, 2, 0.2),
                'type': 'neural',
                'sequence_length': 15,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'epochs': 60,
                'target': 'target'
            },
            
            # Experiment 3: CNN-LSTM Hybrid
            {
                'name': 'cnn_lstm',
                'model_fn': lambda: ModelExperiments.create_cnn_lstm(input_size, 64, 0.3),
                'type': 'neural',
                'sequence_length': 20,
                'batch_size': 32,
                'learning_rate': 0.0003,
                'epochs': 70,
                'target': 'target'
            },
            
            # Experiment 4: Transformer
            {
                'name': 'transformer',
                'model_fn': lambda: ModelExperiments.create_transformer_model(input_size, 64, 4, 3, 0.2),
                'type': 'neural',
                'sequence_length': 25,
                'batch_size': 16,
                'learning_rate': 0.0001,
                'epochs': 80,
                'target': 'target'
            },
            
            # Experiment 5: Ensemble LSTM
            {
                'name': 'ensemble_lstm',
                'model_fn': lambda: ModelExperiments.create_ensemble_lstm(input_size, 48, 0.25),
                'type': 'neural',
                'sequence_length': 30,
                'batch_size': 24,
                'learning_rate': 0.0002,
                'epochs': 90,
                'target': 'target'
            },
            
            # Experiment 6: Random Forest
            {
                'name': 'random_forest',
                'model_fn': ModelExperiments.create_random_forest,
                'type': 'sklearn',
                'sequence_length': 20,
                'target': 'target'
            },
            
            # Experiment 7: Logistic Regression
            {
                'name': 'logistic_regression',
                'model_fn': ModelExperiments.create_logistic_regression,
                'type': 'sklearn',
                'sequence_length': 20,
                'target': 'target'
            },
            
            # Experiment 8: Strong move prediction
            {
                'name': 'simple_lstm_strong_moves',
                'model_fn': lambda: ModelExperiments.create_simple_lstm(input_size, 64, 2, 0.2),
                'type': 'neural',
                'sequence_length': 15,
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 60,
                'target': 'target_strong'
            },
            
            # Experiment 9: Different scaling
            {
                'name': 'simple_lstm_robust_scaling',
                'model_fn': lambda: ModelExperiments.create_simple_lstm(input_size, 64, 2, 0.2),
                'type': 'neural',
                'sequence_length': 20,
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 60,
                'target': 'target',
                'scaler': 'robust'
            },
            
            # Experiment 10: Shorter sequences
            {
                'name': 'simple_lstm_short_seq',
                'model_fn': lambda: ModelExperiments.create_simple_lstm(input_size, 64, 2, 0.2),
                'type': 'neural',
                'sequence_length': 5,
                'batch_size': 64,
                'learning_rate': 0.002,
                'epochs': 40,
                'target': 'target'
            }
        ]
        
        self.experiments = experiments
        return experiments
    
    def run_all_experiments(self):
        """Run all experiments and track results"""
        results = []
        
        for i, exp_config in enumerate(self.experiments):
            print(f"\n" + "="*60)
            print(f"RUNNING EXPERIMENT {i+1}/{len(self.experiments)}: {exp_config['name'].upper()}")
            print("="*60)
            
            try:
                # Initialize new Neptune run for this experiment (if available)
                exp_tracker = ExperimentTracker()
                if exp_tracker.neptune_available:
                    exp_tracker.run["experiment/name"] = exp_config['name']
                    exp_tracker.log_hyperparameters(exp_config)
                
                # Run the experiment
                if exp_config['type'] == 'neural':
                    result = self._run_neural_experiment(exp_config, exp_tracker)
                else:
                    result = self._run_sklearn_experiment(exp_config, exp_tracker)
                
                result['experiment_name'] = exp_config['name']
                results.append(result)
                
                # Log results to Neptune (if available)
                if exp_tracker.neptune_available:
                    exp_tracker.log_backtest_results(result)
                    exp_tracker.finish()
                
                # Print experiment summary
                print(f"\n📊 EXPERIMENT RESULTS:")
                print(f"   Overall Accuracy: {result['overall_accuracy']:.1%}")
                print(f"   High Conf Accuracy: {result['high_confidence_accuracy']:.1%}")
                print(f"   Total Predictions: {result['total_predictions']:,}")
                
            except Exception as e:
                print(f"❌ Experiment {exp_config['name']} failed: {e}")
                import traceback
                print(f"Error details: {traceback.format_exc()}")
                continue
        
        # Compare all results
        if results:
            self._compare_experiments(results)
        else:
            print("❌ No experiments completed successfully")
            
        return results
    
    def _run_neural_experiment(self, config, tracker):
        """Run a neural network experiment"""
        
        # Prepare data with specific scaling
        scaler_type = config.get('scaler', 'standard')
        if scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # Split data for backtesting
        training_data, backtest_data = self._prepare_backtest_data(self.processed_data, 30)
        
        # Scale features
        feature_cols = [col for col in training_data.columns 
                       if col not in ['target', 'target_strong', 'target_intraday_high', 'ticker', 'date']]
        
        training_data[feature_cols] = scaler.fit_transform(training_data[feature_cols])
        
        # Scale backtest data
        for ticker in backtest_data:
            backtest_data[ticker][feature_cols] = scaler.transform(backtest_data[ticker][feature_cols])
        
        # Train/val split
        train_data, val_data = train_test_split(
            training_data, test_size=0.2, stratify=training_data[config['target']], random_state=42
        )
        
        # Create datasets
        train_dataset = StockDataset(train_data, config['sequence_length'], config['target'])
        val_dataset = StockDataset(val_data, config['sequence_length'], config['target'])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        model = config['model_fn']()
        model.to(device)
        
        # Track model info
        total_params = sum(p.numel() for p in model.parameters())
        tracker.log_metrics({'model_parameters': total_params})
        
        # Train model
        train_losses, val_losses, val_accuracies = self._train_neural_model(
            model, train_loader, val_loader, config, tracker
        )
        
        # Backtest
        backtest_results = self._run_backtest(model, backtest_data, feature_cols, config, scaler)
        
        return backtest_results
    
    def _run_sklearn_experiment(self, config, tracker):
        """Run a sklearn model experiment"""
        
        # Prepare data
        training_data, backtest_data = self._prepare_backtest_data(self.processed_data, 30)
        
        # For sklearn models, we flatten the sequences
        feature_cols = [col for col in training_data.columns 
                       if col not in ['target', 'target_strong', 'target_intraday_high', 'ticker', 'date']]
        
        # Scale features
        scaler = StandardScaler()
        training_data[feature_cols] = scaler.fit_transform(training_data[feature_cols])
        
        # Prepare training data
        X_train = training_data[feature_cols].values
        y_train = training_data[config['target']].values
        
        # Train model
        model = config['model_fn']()
        model.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        tracker.log_metrics({'train_accuracy': train_acc})
        
        # Backtest
        backtest_results = self._run_sklearn_backtest(model, backtest_data, feature_cols, config, scaler)
        
        return backtest_results
    
    def _train_neural_model(self, model, train_loader, val_loader, config, tracker):
        """Train neural network with tracking"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(config['epochs']):
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
            
            # Log to Neptune
            tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{config["epochs"]}: Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{config["name"]}.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{config["name"]}.pth'))
        
        return train_losses, val_losses, val_accuracies
    
    def _prepare_backtest_data(self, processed_data, backtest_days=30):
        """Split data for backtesting"""
        processed_data = processed_data.sort_values('date')
        
        backtest_data = {}
        training_data = {}
        
        for ticker in processed_data['ticker'].unique():
            ticker_data = processed_data[processed_data['ticker'] == ticker].copy()
            
            if len(ticker_data) > backtest_days + 50:
                split_idx = len(ticker_data) - backtest_days
                training_data[ticker] = ticker_data.iloc[:split_idx].copy()
                backtest_data[ticker] = ticker_data.iloc[split_idx-30:].copy()  # Include context
        
        # Combine training data
        training_combined = pd.concat(training_data.values(), ignore_index=True) if training_data else pd.DataFrame()
        
        return training_combined, backtest_data
    
    def _run_backtest(self, model, backtest_data, feature_cols, config, scaler):
        """Run neural network backtesting"""
        results = []
        
        for ticker, data in backtest_data.items():
            data = data.reset_index(drop=True)
            features = data[feature_cols].values
            
            for i in range(config['sequence_length'], len(data)):
                # Prepare sequence
                sequence = features[i-config['sequence_length']:i]
                sequence = sequence.reshape(1, config['sequence_length'], -1)
                sequence_tensor = torch.FloatTensor(sequence).to(device)
                
                # Predict
                model.eval()
                with torch.no_grad():
                    output = model(sequence_tensor)
                    probs = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = probs.max().item()
                
                # Get actual
                if i < len(data):
                    actual = data.iloc[i][config['target']]
                    date = data.iloc[i]['date']
                    
                    results.append({
                        'ticker': ticker,
                        'date': date,
                        'prediction': prediction,
                        'actual': actual,
                        'confidence': confidence,
                        'correct': prediction == actual
                    })
        
        return self._analyze_backtest_results(pd.DataFrame(results))
    
    def _run_sklearn_backtest(self, model, backtest_data, feature_cols, config, scaler):
        """Run sklearn model backtesting"""
        results = []
        
        for ticker, data in backtest_data.items():
            data = data.reset_index(drop=True)
            
            # For sklearn, we use the last 30 days directly (no sequences)
            for i in range(30, len(data)):  # Skip first 30 for context
                # Prepare features
                features = data.iloc[i][feature_cols].values.reshape(1, -1)
                
                # Predict
                prediction = model.predict(features)[0]
                
                # Get prediction probability if available
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features)[0]
                    confidence = probs.max()
                else:
                    confidence = 0.6  # Default for models without probability
                
                # Get actual
                actual = data.iloc[i][config['target']]
                date = data.iloc[i]['date']
                
                results.append({
                    'ticker': ticker,
                    'date': date,
                    'prediction': prediction,
                    'actual': actual,
                    'confidence': confidence,
                    'correct': prediction == actual
                })
        
        return self._analyze_backtest_results(pd.DataFrame(results))
    
    def _analyze_backtest_results(self, results_df):
        """Analyze backtest results and return metrics"""
        if len(results_df) == 0:
            return {'overall_accuracy': 0, 'total_predictions': 0}
        
        overall_accuracy = results_df['correct'].mean()
        
        # High confidence predictions
        high_conf_mask = results_df['confidence'] >= 0.7
        high_conf_results = results_df[high_conf_mask]
        high_conf_acc = high_conf_results['correct'].mean() if len(high_conf_results) > 0 else overall_accuracy
        
        # Performance by stock
        stock_performance = results_df.groupby('ticker')['correct'].agg(['count', 'mean'])
        stock_performance = stock_performance[stock_performance['count'] >= 5]
        
        # Daily performance
        daily_performance = results_df.groupby('date')['correct'].mean()
        
        return {
            'overall_accuracy': overall_accuracy,
            'high_confidence_accuracy': high_conf_acc,
            'total_predictions': len(results_df),
            'high_confidence_predictions': len(high_conf_results),
            'best_stock_accuracy': stock_performance['mean'].max() if len(stock_performance) > 0 else 0,
            'worst_stock_accuracy': stock_performance['mean'].min() if len(stock_performance) > 0 else 0,
            'best_day_accuracy': daily_performance.max() if len(daily_performance) > 0 else 0,
            'worst_day_accuracy': daily_performance.min() if len(daily_performance) > 0 else 0,
            'std_daily_accuracy': daily_performance.std() if len(daily_performance) > 0 else 0,
            'results_df': results_df
        }
    
    def _compare_experiments(self, results):
        """Compare all experiment results"""
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Experiment': result['experiment_name'],
                'Overall Accuracy': f"{result['overall_accuracy']:.1%}",
                'High Conf Accuracy': f"{result['high_confidence_accuracy']:.1%}",
                'Total Predictions': result['total_predictions'],
                'High Conf Count': result['high_confidence_predictions'],
                'Best Stock': f"{result['best_stock_accuracy']:.1%}",
                'Worst Stock': f"{result['worst_stock_accuracy']:.1%}",
                'Daily Std': f"{result['std_daily_accuracy']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Overall Accuracy', ascending=False)
        
        print("\n📊 RANKING BY OVERALL ACCURACY:")
        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            print(f"{i:2d}. {row['Experiment']:25s} - {row['Overall Accuracy']:>6s} "
                  f"(HC: {row['High Conf Accuracy']:>6s}, Pred: {row['Total Predictions']:>4d})")
        
        # Best performing experiment
        best_exp = comparison_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING EXPERIMENT:")
        print(f"   Name: {best_exp['Experiment']}")
        print(f"   Overall Accuracy: {best_exp['Overall Accuracy']}")
        print(f"   High Confidence Accuracy: {best_exp['High Conf Accuracy']}")
        print(f"   Total Predictions: {best_exp['Total Predictions']}")
        
        # Insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Check if any model beats random significantly
        best_accuracy = float(best_exp['Overall Accuracy'].strip('%')) / 100
        if best_accuracy > 0.55:
            print(f"   ✅ Best model significantly outperforms random (>55%)")
        elif best_accuracy > 0.52:
            print(f"   ⚠️  Best model shows modest improvement over random")
        else:
            print(f"   ❌ No model significantly beats random - major feature engineering needed")
        
        # Compare neural vs non-neural
        neural_results = [r for r in results if 'lstm' in r['experiment_name'].lower() or 
                         'gru' in r['experiment_name'].lower() or 
                         'transformer' in r['experiment_name'].lower() or
                         'cnn' in r['experiment_name'].lower()]
        
        sklearn_results = [r for r in results if 'forest' in r['experiment_name'].lower() or 
                          'logistic' in r['experiment_name'].lower()]
        
        if neural_results and sklearn_results:
            best_neural = max(neural_results, key=lambda x: x['overall_accuracy'])
            best_sklearn = max(sklearn_results, key=lambda x: x['overall_accuracy'])
            
            print(f"   📈 Best Neural Network: {best_neural['experiment_name']} ({best_neural['overall_accuracy']:.1%})")
            print(f"   📊 Best Traditional ML: {best_sklearn['experiment_name']} ({best_sklearn['overall_accuracy']:.1%})")
            
            if best_neural['overall_accuracy'] > best_sklearn['overall_accuracy']:
                print(f"   🧠 Neural networks perform better")
            else:
                print(f"   📊 Traditional ML performs better")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if best_accuracy < 0.52:
            print(f"   1. 🔄 Major feature engineering needed")
            print(f"   2. 📊 Try different target definitions (volatility, relative performance)")
            print(f"   3. 🏢 Consider market regime awareness")
            print(f"   4. 📈 Add macro-economic features")
            print(f"   5. ⚡ Try different prediction horizons")
        else:
            print(f"   1. 🎯 Focus on high-confidence predictions")
            print(f"   2. 📈 Optimize for precision over recall")
            print(f"   3. 🔄 Ensemble the best performing models")
            print(f"   4. 📊 Deploy with proper risk management")
        
        return comparison_df

def main():
    """Main execution with comprehensive experimentation"""
    print("=== S&P 500 Multi-Experiment Stock Prediction ===\n")
    
    # Initialize Neptune tracking
    main_tracker = ExperimentTracker("jfarland/stock-prediction")
    
    try:
        # 1. Data Collection
        collector = SP500DataCollector()
        raw_data = collector.download_data()
        
        if not raw_data:
            print("❌ No data collected. Exiting.")
            return
        
        # 2. Data Processing
        print("\nProcessing data with enhanced feature engineering...")
        processed_data = collector.process_data(raw_data)
        
        if processed_data.empty:
            print("❌ No processed data. Exiting.")
            return
        
        print(f"✅ Processed data shape: {processed_data.shape}")
        print(f"✅ Features: {processed_data.shape[1] - 6}")  # Excluding targets, ticker, date
        print(f"✅ Target distribution:\n{processed_data['target'].value_counts()}")
        
        # Log data info
        main_tracker.log_metrics({
            'total_samples': len(processed_data),
            'num_stocks': processed_data['ticker'].nunique(),
            'num_features': processed_data.shape[1] - 6,
            'target_balance': processed_data['target'].mean()
        })
        
        # 3. Setup and run experiments
        experiment_runner = ExperimentRunner(processed_data, main_tracker)
        experiments = experiment_runner.setup_experiments()
        
        print(f"\n🚀 Running {len(experiments)} experiments...")
        
        # Run all experiments
        results = experiment_runner.run_all_experiments()
        
        # 4. Overall summary
        print(f"\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        if results:
            best_result = max(results, key=lambda x: x['overall_accuracy'])
            print(f"🏆 Best Overall Performance:")
            print(f"   Experiment: {best_result['experiment_name']}")
            print(f"   Accuracy: {best_result['overall_accuracy']:.1%}")
            print(f"   Predictions: {best_result['total_predictions']:,}")
            
            # Log best result
            main_tracker.log_metrics({
                'best_experiment': best_result['experiment_name'],
                'best_accuracy': best_result['overall_accuracy'],
                'total_experiments': len(results)
            })
        else:
            print("❌ No successful experiments completed")
        
        print(f"\n✅ Experimentation complete!")
        print(f"   📊 {len(results)} experiments completed")
        print(f"   🔬 Results tracked in Neptune")
        
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        
    finally:
        main_tracker.finish()

if __name__ == "__main__":
    main()
