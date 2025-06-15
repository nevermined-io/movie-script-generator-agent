from lib.helicone_client import HeliconeClient
import json
from datetime import datetime
import csv
import os
from collections import defaultdict
import glob

def get_latest_session_info():
    """
    Get the session ID and agent IDs from the latest session log file.
    
    Returns:
        tuple: (session_id, list of agent_ids) or (None, []) if no log file found
    """
    # Get all session log files
    log_files = glob.glob("src/logs/session_*.txt")
    
    if not log_files:
        return None, []
    
    # Get the most recent log file
    latest_file = max(log_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            content = f.read()
            
        # Extract session ID
        session_id = None
        agent_ids = []
        
        for line in content.split('\n'):
            if line.startswith('Session ID:'):
                session_id = line.split('Session ID:')[1].strip()
            elif ':' in line and not line.startswith('Session ID:'):
                # Extract agent ID from lines like "AgentName: agent_id"
                agent_id = line.split(':')[1].strip()
                if agent_id:  # Only add non-empty agent IDs
                    agent_ids.append(agent_id)
        
        return session_id, agent_ids
    except Exception as e:
        print(f"Error reading session log file: {e}")
        return None, []

def create_agent_filter(agent_ids, session_id=None):
    """
    Create a custom filter for multiple agent IDs and session ID.
    Uses OR for agent IDs and AND for session ID to get requests for specific agents in a specific session.
    
    Args:
        agent_ids (list): List of agent IDs to filter by
        session_id (str, optional): Session ID to filter by
        
    Returns:
        dict: Custom filter configuration for Helicone query
    """
    # If no filters are provided, return a filter that matches all requests
    if not agent_ids and not session_id:
        return "all"
    
    # If only session ID is provided
    if not agent_ids and session_id:
        return {
            "request_response_rmt": {
                "properties": {
                    "sessionid": {"equals": session_id}
                }
            }
        }
    
    # Start with the first agent ID
    agent_filter = {
        "request_response_rmt": {
            "properties": {
                "agentid": {"equals": agent_ids[0]}
            }
        }
    }
    
    # Add remaining agent IDs using OR operator
    for agent_id in agent_ids[1:]:
        agent_filter = {
            "operator": "or",
            "left": agent_filter,
            "right": {
                "request_response_rmt": {
                    "properties": {
                        "agentid": {"equals": agent_id}
                    }
                }
            }
        }
    
    # If no session ID, return just the agent filter
    if not session_id:
        return agent_filter
    
    # Add session ID using AND operator
    agent_filter = {
        "operator": "and",
        "left": agent_filter,
        "right": {
            "request_response_rmt": {
                "properties": {
                    "sessionid": {"equals": session_id}
                }
            }
        }
    }

    return agent_filter

def process_payments(margin=0.05, usd_to_credit_rate=1000.0, agent_ids=None, session_id=None, limit=10, offset=0):
    """
    Process payment statistics and print summary, including a margin for pricing and credit conversion.
    Also saves the data to a CSV file for further analysis.
    :param margin: The margin to apply to the total cost (default 5%)
    :param usd_to_credit_rate: Exchange rate from USD to credits (default 1.0)
    :param agent_ids: List of agent IDs to filter by (default None)
    :param session_id: Session ID to filter by (default None)
    :param limit: Maximum number of requests to process (default 10)
    :param offset: Number of requests to skip (default 0)
    """
    # Initialize the client with your API key
    client = HeliconeClient()

    # Create custom filter based on agent IDs and session ID
    custom_filter = create_agent_filter(agent_ids, session_id)

    # Or customize the query parameters
    response = client.query_clickhouse(
        custom_filter=custom_filter,
        is_cached=False,
        limit=limit,
        offset=offset,
        sort_by="created_at",
        sort_order="desc",
        include_inputs=True,
        is_scored=False,
        is_part_of_experiment=False
    )

    # Initialize aggregation variables
    total_requests = 0
    total_input_cost = 0
    total_output_cost = 0
    total_cost = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time_to_first_token = 0
    total_delay = 0
    model_usage = defaultdict(int)
    provider_usage = defaultdict(int)

    # Prepare CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"src/data/payment_data_{timestamp}.csv"
    
    # Define CSV headers
    csv_headers = [
        'request_id', 'created_at', 'model', 'provider', 'total_tokens',
        'prompt_tokens', 'completion_tokens', 'time_to_first_token', 'delay_ms',
        'input_cost', 'output_cost', 'total_cost', 'credit_cost',
        'margin', 'usd_to_credit_rate', 'price_with_margin_usd', 'price_with_margin_credits',
        'session_id', 'agent_ids'
    ]

    print("length of response", len(response["data"]))

    # Process all requests
    if response["data"] and len(response["data"]) > 0:
        print(f"\nProcessing {len(response['data'])} requests...")
        
        # Get existing request IDs from all payment data files
        existing_request_ids = set()
        for existing_file in glob.glob("src/data/payment_data_*.csv"):
            try:
                with open(existing_file, 'r') as f:
                    reader = csv.DictReader(f)
                    existing_request_ids.update(row['request_id'] for row in reader if row['request_id'])
            except Exception as e:
                print(f"Warning: Could not read existing file {existing_file}: {e}")
        
        # Create and open CSV file
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            
            for request in response["data"]:
                request_id = request.get("request_id")
                
                # Skip if no request ID or if already exists in any payment data file
                if not request_id or request_id in existing_request_ids:
                    continue
                
                total_requests += 1
                
                # Update model and provider usage statistics
                model = request.get('model')
                provider = request.get('provider')
                if model:
                    model_usage[model] += 1
                if provider:
                    provider_usage[provider] += 1
                
                # Accumulate token counts
                total_tokens += int(request.get('total_tokens', 0))
                total_prompt_tokens += int(request.get('prompt_tokens', 0))
                total_completion_tokens += int(request.get('completion_tokens', 0))
                total_time_to_first_token += int(request.get('time_to_first_token', 0))
                total_delay += int(request.get('delay_ms', 0))
                
                # Calculate and accumulate costs
                cost_breakdown = client.calculate_request_cost(request)
                if cost_breakdown:
                    total_input_cost += cost_breakdown['input_cost']
                    total_output_cost += cost_breakdown['output_cost']
                    total_cost += cost_breakdown['total_cost']
                
                # Prepare row data for CSV
                row_data = {
                    'request_id': request_id,
                    'created_at': request.get('request_created_at'),
                    'model': model,
                    'provider': provider,
                    'total_tokens': request.get('total_tokens'),
                    'prompt_tokens': request.get('prompt_tokens'),
                    'completion_tokens': request.get('completion_tokens'),
                    'time_to_first_token': request.get('time_to_first_token'),
                    'delay_ms': request.get('delay_ms'),
                    'input_cost': cost_breakdown['input_cost'] if cost_breakdown else None,
                    'output_cost': cost_breakdown['output_cost'] if cost_breakdown else None,
                    'total_cost': cost_breakdown['total_cost'] if cost_breakdown else None,
                    'credit_cost': cost_breakdown['total_cost'] * usd_to_credit_rate if cost_breakdown else None,
                    'margin': margin,
                    'usd_to_credit_rate': usd_to_credit_rate,
                    'price_with_margin_usd': cost_breakdown['total_cost'] * (1 + margin) if cost_breakdown else None,
                    'price_with_margin_credits': cost_breakdown['total_cost'] * (1 + margin) * usd_to_credit_rate if cost_breakdown else None,
                    'session_id': session_id,
                    'agent_ids': ','.join(agent_ids) if agent_ids else None
                }
                
                # Write row to CSV
                writer.writerow(row_data)
                
                # Print individual request details
                print(f"\nRequest {total_requests} Details:")
                print("-" * 50)
                print(f"Request ID: {request_id}")
                print(f"Created At: {request.get('request_created_at')}")
                print(f"Model: {model}")
                print(f"Provider: {provider}")
                print(f"Total Tokens: {request.get('total_tokens')}")
                print(f"Prompt Tokens: {request.get('prompt_tokens')}")
                print(f"Completion Tokens: {request.get('completion_tokens')}")
                print(f"Time to First Token: {request.get('time_to_first_token')}ms")
                print(f"Delay: {request.get('delay_ms')}ms")
                
                if cost_breakdown:
                    print(f"\nCost Breakdown:")
                    print(f"Input Cost: ${cost_breakdown['input_cost']:.6f}")
                    print(f"Output Cost: ${cost_breakdown['output_cost']:.6f}")
                    print(f"Total Cost: ${cost_breakdown['total_cost']:.6f}")
                    credit_cost = cost_breakdown['total_cost'] * usd_to_credit_rate
                    print(f"Credit Cost: {credit_cost:.6f} credits")
                    
                    # Calculate and display price with margin
                    price_with_margin_usd = cost_breakdown['total_cost'] * (1 + margin)
                    price_with_margin_credits = price_with_margin_usd * usd_to_credit_rate
                    print(f"\nPricing with {margin*100:.1f}% margin:")
                    print(f"Price with Margin (USD): ${price_with_margin_usd:.6f}")
                    print(f"Price with Margin (Credits): {price_with_margin_credits:.6f} credits")
                else:
                    print(f"\nNo pricing information found for model {model} from provider {provider}")

        print(f"\nData has been saved to: {csv_filename}")
        print(f"Total new unique requests processed: {total_requests}")
        print(f"Total existing requests skipped: {len(existing_request_ids)}")

        # Calculate total price with margin
        total_price = total_cost * (1 + margin)
        avg_price_per_request = total_price / total_requests if total_requests else 0
        
        # Calculate credit prices
        total_credits = total_price * usd_to_credit_rate
        avg_credits_per_request = avg_price_per_request * usd_to_credit_rate
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        print(f"Total Requests Processed: {total_requests}")
        
        if total_requests > 0:
            print(f"\nCost Summary:")
            print(f"Total Input Cost: ${total_input_cost:.6f}")
            print(f"Total Output Cost: ${total_output_cost:.6f}")
            print(f"Total Cost: ${total_cost:.6f}")
            print(f"Average Cost per Request: ${total_cost/total_requests:.6f}")
            print(f"\nMargin: {margin*100:.1f}%")
            print(f"Total Price (with margin): ${total_price:.6f}")
            print(f"Average Price per Request (with margin): ${avg_price_per_request:.6f}")
            print(f"\nCredit Summary (Exchange Rate: {usd_to_credit_rate:.2f} credits/USD):")
            print(f"Total Credits Required: {total_credits:.6f}")
            print(f"Average Credits per Request: {avg_credits_per_request:.6f}")
            
            print(f"\nToken Usage Summary:")
            print(f"Total Tokens: {total_tokens}")
            print(f"Total Prompt Tokens: {total_prompt_tokens}")
            print(f"Total Completion Tokens: {total_completion_tokens}")
            print(f"Average Tokens per Request: {total_tokens/total_requests:.2f}")
            
            print(f"\nPerformance Metrics:")
            print(f"Average Time to First Token: {total_time_to_first_token/total_requests:.2f}ms")
            print(f"Average Delay: {total_delay/total_requests:.2f}ms")
            
            print(f"\nModel Usage Distribution:")
            for model, count in model_usage.items():
                print(f"{model}: {count} requests ({count/total_requests*100:.1f}%)")
            
            print(f"\nProvider Usage Distribution:")
            for provider, count in provider_usage.items():
                print(f"{provider}: {count} requests ({count/total_requests*100:.1f}%)")
        else:
            print("\nNo new requests were processed. All requests were duplicates of existing data.")

    else:
        print("No requests found in the response")
        
    # Print filter information at the end
    print("\nFilter Information:")
    print(f"Session ID: {session_id}")
    print(f"Agent IDs ({len(agent_ids) if agent_ids else 0}):")
    if agent_ids:
        for agent_id in agent_ids:
            print(f"- {agent_id}")

if __name__ == "__main__":
    # Get agent IDs and session ID from the latest session log file
    session_id, agent_ids = get_latest_session_info()
    
    if agent_ids:
        print(f"\nProcessing payments for session: {session_id}")
        print(f"Found {len(agent_ids)} agents:")
        for agent_id in agent_ids:
            print(f"- {agent_id}")
        process_payments(
            agent_ids=agent_ids,
            session_id=session_id,
            limit=50,
            offset=0
        )  # Default margin is 5% and default exchange rate is 1.0 credits/USD
    else:
        print("No agent IDs found in the latest session log file")

# custom_filter = {
# "request_response_rmt": {
# "latency": {
#     "gte": 0
# }
# }
# }