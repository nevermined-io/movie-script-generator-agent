import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os
import csv

# Load environment variables from .env file
load_dotenv(override=True)

class HeliconeClient:
    """Client for interacting with the Helicone API."""
    
    BASE_URL = "https://api.helicone.ai/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Helicone client with an API key.
        
        Args:
            api_key (str, optional): Your Helicone API key. If not provided, will try to load from HELICONE_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("HELICONE_API_KEY")
        if not self.api_key:
            raise ValueError("Helicone API key not found. Please provide it either as an argument or set HELICONE_API_KEY environment variable.")
            
        self.headers = {
            "authorization": self.api_key,
            "Content-Type": "application/json"
        }
    
    def get_model_pricing(self, model_name: str, provider: str = "OpenAI") -> Optional[Dict[str, float]]:
        """Get pricing information for a specific model.
        
        Args:
            model_name (str): The name of the model
            provider (str): The provider of the model (default: "OpenAI")
            
        Returns:
            Optional[Dict[str, float]]: Dictionary containing input_cost and output_cost, or None if not found
        """
        pricing_file = os.path.join(os.path.dirname(__file__), "..", "src", "pricing", "helicone_pricing.csv")
        
        try:
            with open(pricing_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    current_provider = row.get('Provider', '').strip()
                    current_model = row.get('Model', '').strip()
                    
                    if current_provider.upper() == provider.upper() and current_model == model_name:
                        # Remove $ and convert to float
                        # NOTE: unclear why we need to multiply by 10 here
                        input_cost = 10 * float(row['Input Cost'].replace('$', ''))
                        output_cost = 10 * float(row['Output Cost'].replace('$', ''))
                        return {
                            'input_cost': input_cost,
                            'output_cost': output_cost
                        }
        except Exception as e:
            print(f"Error reading pricing file: {e}")
            return None
            
        return None
    
    def calculate_request_cost(self, request_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate the cost of a request based on token usage.
        
        Args:
            request_data (Dict[str, Any]): The request data containing model, provider, and token information
            
        Returns:
            Optional[Dict[str, float]]: Dictionary containing input_cost, output_cost, and total_cost, or None if pricing not found
        """
        model = request_data.get('model')
        provider = request_data.get('provider')
        # Convert token counts to integers
        prompt_tokens = int(request_data.get('prompt_tokens', 0))
        completion_tokens = int(request_data.get('completion_tokens', 0))
        
        pricing = self.get_model_pricing(model, provider)
        if pricing:

            input_cost_per_1k = pricing['input_cost']
            output_cost_per_1k = pricing['output_cost']
            
            # Calculate costs (pricing is per 1K tokens)
            input_cost = (prompt_tokens / 1000.0) * input_cost_per_1k
            output_cost = (completion_tokens / 1000.0) * output_cost_per_1k
            total_cost = input_cost + output_cost
            
            print("\nCost Calculation Inputs:")
            print(f"Model: {model}")
            print(f"Provider: {provider}")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Input Cost per 1K tokens: ${pricing['input_cost']}")
            print(f"Output Cost per 1K tokens: ${pricing['output_cost']}")
            print(f"Input Cost: ${input_cost:.5f}")
            print(f"Output Cost: ${output_cost:.5f}")
            print(f"Total Cost: ${total_cost:.5f}")
            
            return {
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }
        return None
    
    def query_clickhouse(
        self,
        is_cached: bool = False,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        include_inputs: bool = False,
        is_scored: bool = False,
        is_part_of_experiment: bool = False,
        custom_filter: Optional[Dict[str, Any]] = "all"
    ) -> Dict[str, Any]:
        """Query the Helicone Clickhouse database.
        
        Args:
            filter_type (str): Filter type for the query
            is_cached (bool): Whether to include cached results
            limit (int): Maximum number of results to return
            offset (int): Number of results to skip
            sort_by (str): Field to sort by
            sort_order (str): Sort order ('asc' or 'desc')
            include_inputs (bool): Whether to include input data
            is_scored (bool): Whether to include scoring data
            is_part_of_experiment (bool): Whether to include experiment data
            custom_filter (Dict[str, Any], optional): Custom filter structure for complex queries
            
        Returns:
            Dict[str, Any]: The API response as a dictionary
        """
        url = f"{self.BASE_URL}/request/query-clickhouse"
        
        # Default filter example if no custom filter is provided
        default_filter = "all"
        
        payload = {
            "filter": custom_filter if custom_filter is not None else default_filter,
            "isCached": is_cached,
            "limit": limit,
            "offset": offset,
            "sort": {sort_by: sort_order},
            "includeInputs": include_inputs,
            "isScored": is_scored,
            "isPartOfExperiment": is_part_of_experiment
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()

    def get_request(self, request_id: str) -> Dict[str, Any]:
        """Get details for a specific request by ID.
        
        Args:
            request_id (str): The ID of the request to fetch
            
        Returns:
            Dict[str, Any]: The request details
        """
        url = f"{self.BASE_URL}/request/{request_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        # Handle signed URLs for request and response bodies
        if "request_body" in data and isinstance(data["request_body"], dict) and "signed_body_url" in data:
            try:
                request_body = self._fetch_signed_url(data["signed_body_url"])
                data["request_body"] = request_body
            except Exception as e:
                print(f"Warning: Could not fetch request body from signed URL: {e}")
                
        if "response_body" in data and isinstance(data["response_body"], dict) and "signed_body_url" in data:
            try:
                response_body = self._fetch_signed_url(data["signed_body_url"])
                data["response_body"] = response_body
            except Exception as e:
                print(f"Warning: Could not fetch response body from signed URL: {e}")
        
        return data

    def _fetch_signed_url(self, signed_url: str) -> Dict[str, Any]:
        """Fetch content from a signed URL.
        
        Args:
            signed_url (str): The signed URL to fetch content from
            
        Returns:
            Dict[str, Any]: The content from the signed URL
        """
        response = requests.get(signed_url)
        response.raise_for_status()
        return response.json() 