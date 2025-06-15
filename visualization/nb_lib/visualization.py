import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from datetime import datetime
import os

import pandas as pd
import plotly.express as px

def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = "src/visualization/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_payment_data():
    """Load and combine all payment data CSV files."""
    all_data = []
    files = glob.glob("src/data/payment_data_*.csv")
    # Get absolute path to data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "src", "data"))
    files = glob.glob(os.path.join(data_dir, "payment_data_*.csv"))
    print(f"Looking for payment data files in: {data_dir}")
    print(f"Found files: {files}")
    for file in files:
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

def plot_individual_and_cumulative_costs(df):
    """
    Plots individual request costs and cumulative costs as separate plots in both USD and credits.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'created_at', 'total_cost', and 'credit_cost' columns.

    Returns:
    None
    """
    # Convert 'created_at' to datetime if it's not already
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Sort the DataFrame by 'created_at'
    df = df.sort_values(by='created_at')

    # Calculate cumulative costs for both USD and credits
    df['cumulative_cost'] = df['total_cost'].cumsum()
    df['cumulative_credit_cost'] = df['credit_cost'].cumsum()

    # Plot individual request costs in USD
    fig_individual_usd = px.scatter(df, x='created_at', y='total_cost', 
                                title='Individual Request Costs Over Time (USD)',
                                labels={'created_at': 'Request Time', 'total_cost': 'Request Cost (USD)'})
    fig_individual_usd.update_traces(mode='lines+markers')
    fig_individual_usd.show()

    # Plot cumulative costs in USD
    fig_cumulative_usd = px.line(df, x='created_at', y='cumulative_cost', 
                             title='Cumulative Costs Over Time (USD)',
                             labels={'created_at': 'Request Time', 'cumulative_cost': 'Cumulative Cost (USD)'})
    fig_cumulative_usd.show()

    # Plot individual request costs in Credits
    fig_individual_credits = px.scatter(df, x='created_at', y='credit_cost', 
                                title='Individual Request Costs Over Time (Credits)',
                                labels={'created_at': 'Request Time', 'credit_cost': 'Request Cost (Credits)'})
    fig_individual_credits.update_traces(mode='lines+markers')
    fig_individual_credits.show()

    # Plot cumulative costs in Credits
    fig_cumulative_credits = px.line(df, x='created_at', y='cumulative_credit_cost', 
                             title='Cumulative Costs Over Time (Credits)',
                             labels={'created_at': 'Request Time', 'cumulative_credit_cost': 'Cumulative Cost (Credits)'})
    fig_cumulative_credits.show()

def cost_summary(df):
    """
    Provides and prints a cost summary globally and by agent ID, including total tokens and credit cost.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'total_cost', 'total_tokens', 'credit_cost', and 'agent_ids' columns.

    Returns:
    None
    """
    # Global summary
    global_total_cost = df['total_cost'].sum()
    global_average_cost = df['total_cost'].mean()
    global_total_tokens = df['total_tokens'].sum()
    global_credit_cost = df['credit_cost'].sum()

    print("Global Summary:")
    print(f"Total Cost: {global_total_cost}")
    print(f"Average Cost: {global_average_cost}")
    print(f"Total Tokens: {global_total_tokens}")
    print(f"Total Credit Cost: {global_credit_cost}\n")

    # Agent-specific summary
    agent_summary = {}
    for index, row in df.iterrows():
        # Split the agent_ids
        agents = row['agent_ids'].split(',')
        for agent_id in agents:
            if agent_id not in agent_summary:
                agent_summary[agent_id] = {'total_cost': 0, 'total_tokens': 0, 'credit_cost': 0, 'count': 0}
            agent_summary[agent_id]['total_cost'] += row['total_cost']
            agent_summary[agent_id]['total_tokens'] += row['total_tokens']
            agent_summary[agent_id]['credit_cost'] += row['credit_cost']
            agent_summary[agent_id]['count'] += 1

    print("Agent-Specific Summary:")
    for agent_id, summary in agent_summary.items():
        average_cost = summary['total_cost'] / summary['count']
        print(f"Agent ID: {agent_id}")
        print(f"  Total Cost: {summary['total_cost']}")
        print(f"  Average Cost: {average_cost}")
        print(f"  Total Tokens: {summary['total_tokens']}")
        print(f"  Total Credit Cost: {summary['credit_cost']}\n")

# # Example usage
# # Assuming 'data' is your DataFrame
# # cost_summary(data)

# def create_summary_statistics(df, output_dir):
#     """Create summary statistics dashboard."""
#     # Filter for our specific agents
#     agent_df = df[df['agent_ids'].str.contains('|'.join(AGENT_IDS), na=False)]
    
#     # Calculate summary metrics
#     total_requests = len(agent_df)
#     total_cost = agent_df['total_cost'].sum()
#     total_tokens = agent_df['total_tokens'].sum()
#     avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0
#     avg_tokens_per_request = total_tokens / total_requests if total_requests > 0 else 0
    
#     # Create summary table
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=['Metric', 'Value'],
#                    fill_color='paleturquoise',
#                    align='left'),
#         cells=dict(values=[
#             ['Total Requests', 'Total Cost ($)', 'Total Tokens', 'Avg Cost/Request ($)', 'Avg Tokens/Request'],
#             [f"{total_requests:,}", f"${total_cost:.2f}", f"{total_tokens:,}", f"${avg_cost_per_request:.4f}", f"{avg_tokens_per_request:.1f}"]
#         ],
#         fill_color='lavender',
#         align='left'))
#     ])
    
#     fig.update_layout(title='Payment Data Summary Statistics (Selected Agents)')
#     fig.show()

# def create_time_series_analysis(df, output_dir):
#     """Create time series analysis of costs and token usage."""
#     # Filter for our specific agents
#     agent_df = df[df['agent_ids'].str.contains('|'.join(AGENT_IDS), na=False)]
    
#     # Resample data to daily frequency
#     daily_data = agent_df.set_index('created_at').resample('D').agg({
#         'total_cost': 'sum',
#         'total_tokens': 'sum',
#         'request_id': 'count'
#     }).reset_index()
    
#     # Create subplot with secondary y-axis
#     fig = make_subplots(specs=[[{"secondary_y": True}]])
    
#     # Add cost line
#     fig.add_trace(
#         go.Scatter(x=daily_data['created_at'], y=daily_data['total_cost'],
#                   name="Daily Cost ($)", line=dict(color='blue')),
#         secondary_y=False
#     )
    
#     # Add token count line
#     fig.add_trace(
#         go.Scatter(x=daily_data['created_at'], y=daily_data['total_tokens'],
#                   name="Daily Tokens", line=dict(color='red')),
#         secondary_y=True
#     )
    
#     # Add request count as bar chart
#     fig.add_trace(
#         go.Bar(x=daily_data['created_at'], y=daily_data['request_id'],
#                name="Daily Requests", marker_color='green', opacity=0.3),
#         secondary_y=False
#     )
    
#     # Update layout
#     fig.update_layout(
#         title='Daily Usage Analysis (Selected Agents)',
#         xaxis_title='Date',
#         hovermode='x unified'
#     )
    
#     # Update y-axes labels
#     fig.update_yaxes(title_text="Cost ($)", secondary_y=False)
#     fig.update_yaxes(title_text="Token Count", secondary_y=True)
    
#     fig.show()

# def create_session_analysis(df, output_dir):
#     """Create analysis of session-specific data."""
#     # Filter for our specific agents
#     agent_df = df[df['agent_ids'].str.contains('|'.join(AGENT_IDS), na=False)]
    
#     # Group by session and calculate metrics
#     session_metrics = agent_df.groupby('session_id').agg({
#         'total_cost': 'sum',
#         'total_tokens': 'sum',
#         'request_id': 'count'
#     }).reset_index()
    
#     # Calculate average cost per request
#     session_metrics['avg_cost_per_request'] = session_metrics['total_cost'] / session_metrics['request_id']
    
#     # Create subplot for session analysis
#     fig = make_subplots(rows=2, cols=2,
#                        subplot_titles=('Cost by Session', 'Token Usage by Session',
#                                      'Request Count by Session', 'Average Cost per Request'))
    
#     # Cost by session
#     fig.add_trace(
#         go.Bar(x=session_metrics['session_id'], y=session_metrics['total_cost'],
#                name='Total Cost', marker_color='blue'),
#         row=1, col=1
#     )
    
#     # Token usage by session
#     fig.add_trace(
#         go.Bar(x=session_metrics['session_id'], y=session_metrics['total_tokens'],
#                name='Total Tokens', marker_color='red'),
#         row=1, col=2
#     )
    
#     # Request count by session
#     fig.add_trace(
#         go.Bar(x=session_metrics['session_id'], y=session_metrics['request_id'],
#                name='Request Count', marker_color='green'),
#         row=2, col=1
#     )
    
#     # Average cost per request by session
#     fig.add_trace(
#         go.Bar(x=session_metrics['session_id'], y=session_metrics['avg_cost_per_request'],
#                name='Avg Cost/Request', marker_color='purple'),
#         row=2, col=2
#     )
    
#     # Update layout with better formatting
#     fig.update_layout(
#         height=800,
#         title_text="Session Analysis (Selected Agents)",
#         showlegend=False,
#         template="plotly_white"
#     )
    
#     # Update x-axis labels to be more readable
#     for i in range(1, 3):
#         for j in range(1, 3):
#             fig.update_xaxes(tickangle=45, row=i, col=j)
    
#     fig.show()

# def create_agent_analysis(df, output_dir):
#     """Create analysis of agent usage and costs."""
#     # Filter for our specific agents
#     agent_df = df[df['agent_ids'].str.contains('|'.join(AGENT_IDS), na=False)]
    
#     if len(agent_df) == 0:
#         print("No data found for the specified agents")
#         return
    
#     # Create subplot for agent analysis
#     fig = make_subplots(rows=2, cols=2,
#                        subplot_titles=('Cost by Agent', 'Token Usage by Agent',
#                                      'Request Count by Agent', 'Average Cost per Request'))
    
#     # Cost by agent
#     agent_costs = agent_df.groupby('agent_ids')['total_cost'].sum()
#     fig.add_trace(
#         go.Bar(x=agent_costs.index, y=agent_costs.values,
#                name='Total Cost', marker_color='blue'),
#         row=1, col=1
#     )
    
#     # Token usage by agent
#     agent_tokens = agent_df.groupby('agent_ids')['total_tokens'].sum()
#     fig.add_trace(
#         go.Bar(x=agent_tokens.index, y=agent_tokens.values,
#                name='Total Tokens', marker_color='red'),
#         row=1, col=2
#     )
    
#     # Request count by agent
#     agent_requests = agent_df.groupby('agent_ids')['request_id'].count()
#     fig.add_trace(
#         go.Bar(x=agent_requests.index, y=agent_requests.values,
#                name='Request Count', marker_color='green'),
#         row=2, col=1
#     )
    
#     # Average cost per request by agent
#     agent_avg_cost = agent_df.groupby('agent_ids').apply(
#         lambda x: x['total_cost'].sum() / len(x)
#     )
#     fig.add_trace(
#         go.Bar(x=agent_avg_cost.index, y=agent_avg_cost.values,
#                name='Avg Cost/Request', marker_color='purple'),
#         row=2, col=2
#     )
    
#     # Update layout with better formatting
#     fig.update_layout(
#         height=800,
#         title_text="Agent Usage Analysis (Selected Agents)",
#         showlegend=False,
#         template="plotly_white"
#     )
    
#     # Update x-axis labels to be more readable
#     for i in range(1, 3):
#         for j in range(1, 3):
#             fig.update_xaxes(tickangle=45, row=i, col=j)
    
#     fig.show()


# #################### START HERE ########################