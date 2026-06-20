"""
NFL Game Momentum Model

This module implements multiple time series models to predict game momentum
based on scores, possessions, sacks, and turnovers.

Models implemented:
- Prophet (Facebook's forecasting tool)
- ARIMA/SARIMAX (statsmodels)
- LSTM (Long Short-Term Memory neural network)

Author: César Adrián Delgado Díaz
Portfolio: https://tu-portfolio.com
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
GitHub: https://github.com/cesar530
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import warnings
from datetime import datetime, timedelta

# Time series models
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras import optimizers
    TF_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠️ TensorFlow/Keras no disponible: {e}")
    print("   El modelo LSTM no estará disponible. Prophet y ARIMA funcionarán normalmente.")
    TF_AVAILABLE = False
    
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class GameMomentumModel:
    """
    Main class for NFL game momentum prediction.
    
    This class provides methods to train and predict using multiple
    time series models: Prophet, ARIMA/SARIMAX, and LSTM.
    """
    
    def __init__(self, model_type: str = 'prophet'):
        """
        Initialize the momentum model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use: 'prophet', 'arima', or 'lstm'
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
        if self.model_type not in ['prophet', 'arima', 'lstm']:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                "Choose from 'prophet', 'arima', or 'lstm'"
            )
        
        # Check if LSTM is requested but TensorFlow is not available
        if self.model_type == 'lstm' and not TF_AVAILABLE:
            raise RuntimeError(
                "LSTM model requires TensorFlow/Keras but they are not available. "
                "Please install tensorflow or use 'prophet' or 'arima' models instead."
            )
    
    def _prepare_prophet_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        y : pd.Series
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            Data formatted for Prophet (ds, y columns)
        """
        df = pd.DataFrame()
        
        # Prophet requires 'ds' (datetime) and 'y' (target) columns
        if 'play_number' in X.columns:
            # Create synthetic timestamps
            base_time = datetime(2024, 1, 1, 13, 0, 0)  # Game start time
            df['ds'] = [
                base_time + timedelta(seconds=i*30)
                for i in range(len(X))
            ]
        else:
            df['ds'] = pd.date_range(
                start='2024-01-01',
                periods=len(X),
                freq='30S'
            )
        
        df['y'] = y.values
        
        # Add regressors
        for col in X.columns:
            if col not in ['ds', 'y']:
                df[col] = X[col].values
        
        return df
    
    def fit_prophet(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'GameMomentumModel':
        """
        Fit Prophet model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target variable
        **kwargs : dict
            Additional arguments for Prophet
            
        Returns:
        --------
        self : GameMomentumModel
        """
        df = self._prepare_prophet_data(X, y)
        
        # Initialize Prophet model
        self.model = Prophet(
            changepoint_prior_scale=kwargs.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=kwargs.get('seasonality_prior_scale', 10),
            seasonality_mode=kwargs.get('seasonality_mode', 'additive'),
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        # Add custom regressors
        for col in X.columns:
            if col not in ['play_number', 'ds', 'y']:
                self.model.add_regressor(col)
        
        # Fit model
        self.model.fit(df)
        self.is_fitted = True
        
        return self
    
    def fit_arima(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
        **kwargs
    ) -> 'GameMomentumModel':
        """
        Fit ARIMA/SARIMAX model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features (exogenous variables)
        y : pd.Series
            Target variable
        order : Tuple[int, int, int]
            ARIMA order (p, d, q)
        **kwargs : dict
            Additional arguments for SARIMAX
            
        Returns:
        --------
        self : GameMomentumModel
        """
        # Use SARIMAX for exogenous variables support
        self.model = SARIMAX(
            y,
            exog=X,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.model = self.model.fit(disp=False)
        self.is_fitted = True
        
        return self
    
    def fit_lstm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sequence_length: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        **kwargs
    ) -> 'GameMomentumModel':
        """
        Fit LSTM model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target variable
        sequence_length : int
            Number of time steps to look back
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        **kwargs : dict
            Additional arguments for LSTM
            
        Returns:
        --------
        self : GameMomentumModel
        """
        # Scale data
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(
            X_scaled,
            y_scaled,
            sequence_length
        )
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(
                kwargs.get('lstm_units', 64),
                activation='tanh',
                return_sequences=True,
                input_shape=(sequence_length, X.shape[1])
            ),
            Dropout(kwargs.get('dropout', 0.2)),
            LSTM(kwargs.get('lstm_units_2', 32), activation='tanh'),
            Dropout(kwargs.get('dropout', 0.2)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=optimizers.Adam(
                learning_rate=kwargs.get('learning_rate', 0.001)
            ),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=kwargs.get('verbose', 0),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        self.sequence_length = sequence_length
        self.is_fitted = True
        
        return self
    
    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature array
        y : np.ndarray
            Target array
        sequence_length : int
            Length of sequences
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Sequenced features and targets
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'GameMomentumModel':
        """
        Fit the model based on the specified model type.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target variable
        **kwargs : dict
            Model-specific parameters
            
        Returns:
        --------
        self : GameMomentumModel
        """
        if self.model_type == 'prophet':
            return self.fit_prophet(X, y, **kwargs)
        elif self.model_type == 'arima':
            return self.fit_arima(X, y, **kwargs)
        elif self.model_type == 'lstm':
            return self.fit_lstm(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(
        self,
        X: pd.DataFrame,
        periods: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
        periods : int, optional
            Number of future periods to forecast (Prophet/ARIMA)
            
        Returns:
        --------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type == 'prophet':
            return self._predict_prophet(X, periods)
        elif self.model_type == 'arima':
            return self._predict_arima(X, periods)
        elif self.model_type == 'lstm':
            return self._predict_lstm(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _predict_prophet(
        self,
        X: pd.DataFrame,
        periods: Optional[int] = None
    ) -> np.ndarray:
        """Predict using Prophet model."""
        if periods is None:
            periods = len(X)
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='30S')
        
        # Add regressors
        for col in X.columns:
            if col in future.columns:
                continue
            if len(X[col]) >= len(future):
                future[col] = X[col].iloc[:len(future)].values
            else:
                # Pad with last value if needed
                padded = np.pad(
                    X[col].values,
                    (0, len(future) - len(X[col])),
                    mode='edge'
                )
                future[col] = padded
        
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-periods:]
    
    def _predict_arima(
        self,
        X: pd.DataFrame,
        periods: Optional[int] = None
    ) -> np.ndarray:
        """Predict using ARIMA model."""
        if periods is None:
            periods = len(X)
        
        forecast = self.model.forecast(steps=periods, exog=X.iloc[:periods])
        return forecast
    
    def _predict_lstm(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using LSTM model."""
        X_scaled = self.scaler_X.transform(X)
        
        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i + self.sequence_length])
        
        if len(X_seq) == 0:
            raise ValueError(
                f"Not enough data for prediction. "
                f"Need at least {self.sequence_length} samples."
            )
        
        X_seq = np.array(X_seq)
        
        # Predict and inverse transform
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
        --------
        Dict[str, Any]
            Model information
        """
        info = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        
        if self.model_type == 'lstm' and self.is_fitted:
            info['history'] = self.history.history
            info['sequence_length'] = self.sequence_length
        
        return info


# Convenience functions for quick model training
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, GameMomentumModel]:
    """
    Train all three model types and return them.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    Dict[str, GameMomentumModel]
        Dictionary of trained models
    """
    models = {}
    
    if verbose:
        print("Training Prophet model...")
    models['prophet'] = GameMomentumModel('prophet')
    models['prophet'].fit(X_train, y_train)
    
    if verbose:
        print("Training ARIMA model...")
    models['arima'] = GameMomentumModel('arima')
    models['arima'].fit(X_train, y_train, order=(2, 1, 2))
    
    if TF_AVAILABLE:
        if verbose:
            print("Training LSTM model...")
        try:
            models['lstm'] = GameMomentumModel('lstm')
            models['lstm'].fit(
                X_train,
                y_train,
                sequence_length=10,
                epochs=30,
                batch_size=16,
                verbose=0
            )
        except Exception as e:
            if verbose:
                print(f"⚠️ LSTM training failed: {e}")
                print("   Continuing with Prophet and ARIMA only...")
    else:
        if verbose:
            print("⚠️ Skipping LSTM model (TensorFlow not available)")
    
    if verbose:
        print("All available models trained successfully!")
    
    return models
