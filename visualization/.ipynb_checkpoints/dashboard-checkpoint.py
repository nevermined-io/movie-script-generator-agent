# %% [markdown]
# # Payment Data Analysis Dashboard
# 
# This notebook analyzes payment data from specific sessions and creates visualizations using Plotly.
# 
# ## Configuration
# Session IDs to analyze:

# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from datetime import datetime
import os

# Session IDs to analyze
SESSION_IDS = [
    "c2687b82-4249-5b22-8e60-abae67edb2fb",
    "b24607ff-4934-5ced-b578-c57a69afb660"
]

def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = "src/visualization/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_payment_data():
    """Load and combine all payment data CSV files."""
    all_data = []
    for file in glob.glob("src/data/payment_data_*.csv"):
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        raise ValueError("No payment data files found in src/data/")
    
    # Combine all dataframes
    df = pd.concat(all_data, ignore_index=True)
    
    # Convert timestamp to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Remove duplicates based on request_id
    df = df.drop_duplicates(subset=['request_id'])
    
    return df

def create_summary_statistics(df, output_dir):
    """Create summary statistics dashboard."""
    # Filter for our specific sessions
    session_df = df[df['session_id'].isin(SESSION_IDS)]
    
    # Calculate summary metrics
    total_requests = len(session_df)
    total_cost = session_df['total_cost'].sum()
    total_tokens = session_df['total_tokens'].sum()
    avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0
    avg_tokens_per_request = total_tokens / total_requests if total_requests > 0 else 0
    
    # Create summary table
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Metric', 'Value'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[
            ['Total Requests', 'Total Cost ($)', 'Total Tokens', 'Avg Cost/Request ($)', 'Avg Tokens/Request'],
            [f"{total_requests:,}", f"${total_cost:.2f}", f"{total_tokens:,}", f"${avg_cost_per_request:.4f}", f"{avg_tokens_per_request:.1f}"]
        ],
        fill_color='lavender',
        align='left'))
    ])
    
    fig.update_layout(title='Payment Data Summary Statistics (Selected Sessions)')
    fig.write_html(os.path.join(output_dir, 'summary_statistics.html'))

def create_time_series_analysis(df, output_dir):
    """Create time series analysis of costs and token usage."""
    # Filter for our specific sessions
    session_df = df[df['session_id'].isin(SESSION_IDS)]
    
    # Resample data to daily frequency
    daily_data = session_df.set_index('created_at').resample('D').agg({
        'total_cost': 'sum',
        'total_tokens': 'sum',
        'request_id': 'count'
    }).reset_index()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add cost line
    fig.add_trace(
        go.Scatter(x=daily_data['created_at'], y=daily_data['total_cost'],
                  name="Daily Cost ($)", line=dict(color='blue')),
        secondary_y=False
    )
    
    # Add token count line
    fig.add_trace(
        go.Scatter(x=daily_data['created_at'], y=daily_data['total_tokens'],
                  name="Daily Tokens", line=dict(color='red')),
        secondary_y=True
    )
    
    # Add request count as bar chart
    fig.add_trace(
        go.Bar(x=daily_data['created_at'], y=daily_data['request_id'],
               name="Daily Requests", marker_color='green', opacity=0.3),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title='Daily Usage Analysis (Selected Sessions)',
        xaxis_title='Date',
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Cost ($)", secondary_y=False)
    fig.update_yaxes(title_text="Token Count", secondary_y=True)
    
    fig.write_html(os.path.join(output_dir, 'time_series_analysis.html'))

def create_session_analysis(df, output_dir):
    """Create analysis of session-specific data."""
    # Filter for our specific sessions
    session_df = df[df['session_id'].isin(SESSION_IDS)]
    
    # Group by session and calculate metrics
    session_metrics = session_df.groupby('session_id').agg({
        'total_cost': 'sum',
        'total_tokens': 'sum',
        'request_id': 'count'
    }).reset_index()
    
    # Calculate average cost per request
    session_metrics['avg_cost_per_request'] = session_metrics['total_cost'] / session_metrics['request_id']
    
    # Create subplot for session analysis
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Cost by Session', 'Token Usage by Session',
                                     'Request Count by Session', 'Average Cost per Request'))
    
    # Cost by session
    fig.add_trace(
        go.Bar(x=session_metrics['session_id'], y=session_metrics['total_cost'],
               name='Total Cost', marker_color='blue'),
        row=1, col=1
    )
    
    # Token usage by session
    fig.add_trace(
        go.Bar(x=session_metrics['session_id'], y=session_metrics['total_tokens'],
               name='Total Tokens', marker_color='red'),
        row=1, col=2
    )
    
    # Request count by session
    fig.add_trace(
        go.Bar(x=session_metrics['session_id'], y=session_metrics['request_id'],
               name='Request Count', marker_color='green'),
        row=2, col=1
    )
    
    # Average cost per request by session
    fig.add_trace(
        go.Bar(x=session_metrics['session_id'], y=session_metrics['avg_cost_per_request'],
               name='Avg Cost/Request', marker_color='purple'),
        row=2, col=2
    )
    
    # Update layout with better formatting
    fig.update_layout(
        height=800,
        title_text="Session Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    # Update x-axis labels to be more readable
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    fig.write_html(os.path.join(output_dir, 'session_analysis.html'))

def create_agent_analysis(df, output_dir):
    """Create analysis of agent usage and costs."""
    # Filter for our specific sessions
    session_df = df[df['session_id'].isin(SESSION_IDS)]
    
    if len(session_df) == 0:
        print("No data found for the specified sessions")
        return
    
    # Create subplot for agent analysis
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Cost by Agent', 'Token Usage by Agent',
                                     'Request Count by Agent', 'Average Cost per Request'))
    
    # Cost by agent
    agent_costs = session_df.groupby('agent_ids')['total_cost'].sum()
    fig.add_trace(
        go.Bar(x=agent_costs.index, y=agent_costs.values,
               name='Total Cost', marker_color='blue'),
        row=1, col=1
    )
    
    # Token usage by agent
    agent_tokens = session_df.groupby('agent_ids')['total_tokens'].sum()
    fig.add_trace(
        go.Bar(x=agent_tokens.index, y=agent_tokens.values,
               name='Total Tokens', marker_color='red'),
        row=1, col=2
    )
    
    # Request count by agent
    agent_requests = session_df.groupby('agent_ids')['request_id'].count()
    fig.add_trace(
        go.Bar(x=agent_requests.index, y=agent_requests.values,
               name='Request Count', marker_color='green'),
        row=2, col=1
    )
    
    # Average cost per request by agent
    agent_avg_cost = session_df.groupby('agent_ids').apply(
        lambda x: x['total_cost'].sum() / len(x)
    )
    fig.add_trace(
        go.Bar(x=agent_avg_cost.index, y=agent_avg_cost.values,
               name='Avg Cost/Request', marker_color='purple'),
        row=2, col=2
    )
    
    # Update layout with better formatting
    fig.update_layout(
        height=800,
        title_text="Agent Usage Analysis (Selected Sessions)",
        showlegend=False,
        template="plotly_white"
    )
    
    # Update x-axis labels to be more readable
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    fig.write_html(os.path.join(output_dir, 'agent_analysis.html'))

def main():
    """Main function to run all visualizations."""
    try:
        # Create output directory
        output_dir = ensure_output_dir()
        
        # Load data
        df = load_payment_data()
        
        # Create visualizations
        create_summary_statistics(df, output_dir)
        create_time_series_analysis(df, output_dir)
        create_agent_analysis(df, output_dir)
        create_session_analysis(df, output_dir)
        
        print(f"\nVisualizations have been saved to: {output_dir}")
        print("Open the HTML files in your web browser to view the interactive charts.")
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")

if __name__ == "__main__":
    main()
