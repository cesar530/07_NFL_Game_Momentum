"""
Utility functions for NFL Game Momentum Model

Author: César Adrián Delgado Díaz
Portfolio: https://tu-portfolio.com
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
GitHub: https://github.com/cesar530
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def generate_synthetic_game_data(
    n_plays: int = 150,
    home_advantage: float = 0.55,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic NFL game data for demonstration purposes.
    
    Parameters:
    -----------
    n_plays : int
        Number of plays to simulate
    home_advantage : float
        Probability advantage for home team (0.5 = neutral)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Simulated game data with time series features
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize game state
    data = {
        'play_number': range(1, n_plays + 1),
        'quarter': [],
        'time_remaining': [],
        'home_score': [],
        'away_score': [],
        'possession': [],  # 1 for home, 0 for away
        'field_position': [],  # yards from own goal line
        'down': [],
        'yards_to_go': [],
    }
    
    # Game state variables
    home_score = 0
    away_score = 0
    quarter = 1
    time_remaining = 15 * 60  # 15 minutes per quarter
    possession = 1  # Home starts
    field_pos = 25  # Starting field position
    down = 1
    yards_to_go = 10
    
    for play in range(n_plays):
        # Update quarter and time
        if time_remaining <= 0 and quarter < 4:
            quarter += 1
            time_remaining = 15 * 60
        
        # Record current state
        data['quarter'].append(quarter)
        data['time_remaining'].append(max(0, time_remaining))
        data['home_score'].append(home_score)
        data['away_score'].append(away_score)
        data['possession'].append(possession)
        data['field_position'].append(field_pos)
        data['down'].append(down)
        data['yards_to_go'].append(yards_to_go)
        
        # Simulate play outcome
        play_time = np.random.randint(20, 45)
        time_remaining -= play_time
        
        # Determine if scoring play
        scoring_prob = 0.15 if field_pos > 80 else 0.05
        is_scoring = np.random.random() < scoring_prob
        
        # Apply home advantage
        success_prob = home_advantage if possession == 1 else (1 - home_advantage)
        
        if is_scoring:
            # Touchdown or field goal
            is_td = np.random.random() < 0.7
            points = 7 if is_td else 3
            
            if possession == 1:
                home_score += points
            else:
                away_score += points
            
            # Change possession after score
            possession = 1 - possession
            field_pos = 25
            down = 1
            yards_to_go = 10
        else:
            # Regular play
            if np.random.random() < success_prob:
                # Successful play
                yards_gained = max(0, int(np.random.normal(6, 4)))
                field_pos = min(100, field_pos + yards_gained)
                
                if yards_gained >= yards_to_go:
                    # First down
                    down = 1
                    yards_to_go = min(10, 100 - field_pos)
                else:
                    down += 1
                    yards_to_go -= yards_gained
            else:
                # Unsuccessful play or turnover
                if down >= 4 or np.random.random() < 0.1:
                    # Turnover or punt
                    possession = 1 - possession
                    field_pos = 100 - field_pos + np.random.randint(-20, 10)
                    field_pos = max(1, min(99, field_pos))
                    down = 1
                    yards_to_go = 10
                else:
                    down += 1
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['score_differential'] = df['home_score'] - df['away_score']
    df['total_score'] = df['home_score'] + df['away_score']
    df['game_time_elapsed'] = (df['quarter'] - 1) * 15 + (15 - df['time_remaining'] / 60)
    df['is_home_possession'] = df['possession']
    
    return df


def calculate_momentum_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Calculate momentum-related features from game data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Game data with scores and events
    window : int
        Rolling window size for momentum calculation
        
    Returns:
    --------
    pd.DataFrame
        Data with additional momentum features
    """
    df = df.copy()
    
    # Score momentum (rate of change)
    df['home_score_momentum'] = df['home_score'].diff().rolling(window).sum()
    df['away_score_momentum'] = df['away_score'].diff().rolling(window).sum()
    df['score_momentum_diff'] = df['home_score_momentum'] - df['away_score_momentum']
    
    # Possession changes
    df['possession_change'] = df['possession'].diff().abs()
    df['possession_momentum'] = df['possession_change'].rolling(window).sum()
    
    # Field position momentum
    df['field_position_change'] = df['field_position'].diff()
    df['field_position_momentum'] = df['field_position_change'].rolling(window).mean()
    
    # Overall momentum indicator (composite score)
    df['momentum_score'] = (
        df['score_momentum_diff'] * 0.5 +
        df['field_position_momentum'] * 0.3 +
        df['possession_momentum'] * 0.2
    )
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df


def plot_game_progression(
    df: pd.DataFrame,
    title: str = "NFL Game Progression Analysis"
) -> go.Figure:
    """
    Create interactive plot of game progression with plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Game data with scores and momentum
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Score Progression',
            'Momentum Indicator',
            'Field Position & Possession'
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Score progression
    fig.add_trace(
        go.Scatter(
            x=df['play_number'],
            y=df['home_score'],
            name='Home Score',
            line=dict(color='#013369', width=3),
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['play_number'],
            y=df['away_score'],
            name='Away Score',
            line=dict(color='#D50A0A', width=3),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Momentum indicator
    if 'momentum_score' in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df['momentum_score']]
        fig.add_trace(
            go.Bar(
                x=df['play_number'],
                y=df['momentum_score'],
                name='Momentum',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Field position
    fig.add_trace(
        go.Scatter(
            x=df['play_number'],
            y=df['field_position'],
            name='Field Position',
            line=dict(color='#FFB81C', width=2),
            fill='tozeroy',
            opacity=0.5
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Play Number", row=3, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)
    fig.update_yaxes(title_text="Yards", row=3, col=1)
    
    fig.update_layout(
        title=title,
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_momentum_heatmap(
    df: pd.DataFrame,
    feature: str = 'momentum_score',
    title: str = "Game Momentum Heatmap"
) -> plt.Figure:
    """
    Create heatmap visualization of momentum throughout the game.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Game data with momentum features
    feature : str
        Feature to visualize
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Reshape data for heatmap
    quarters = df['quarter'].max()
    plays_per_quarter = len(df) // quarters
    
    data_matrix = []
    for q in range(1, quarters + 1):
        quarter_data = df[df['quarter'] == q][feature].values
        # Pad if necessary
        if len(quarter_data) < plays_per_quarter:
            quarter_data = np.pad(
                quarter_data,
                (0, plays_per_quarter - len(quarter_data)),
                mode='constant'
            )
        data_matrix.append(quarter_data[:plays_per_quarter])
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    sns.heatmap(
        data_matrix,
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Momentum Score'},
        yticklabels=[f'Q{i+1}' for i in range(quarters)],
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Play Progression', fontsize=12)
    ax.set_ylabel('Quarter', fontsize=12)
    
    plt.tight_layout()
    return fig


def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate performance metrics for momentum predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of metric names and values
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }
    
    return metrics


def print_model_summary(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print formatted model performance summary.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Dictionary of metrics
    model_name : str
        Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:10.4f}")
    
    print(f"{'='*50}\n")
