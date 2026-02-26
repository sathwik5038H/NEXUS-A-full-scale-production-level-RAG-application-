# nexus/app/utils/costing.py

import logging

from typing import Dict, Any



# Define pricing per 1,000,000 tokens (approximate market rates)

MODEL_PRICING = {

    "gemini-2.5-flash": {

        "input": 0.10,   # $0.10 per 1M input tokens

        "output": 0.40   # $0.40 per 1M output tokens

    },

    "gpt-4o": {

        "input": 5.00,

        "output": 15.00

    },

    "gemini-2.5-pro": {

        "input": 0.20,

        "output": 0.80

    }

}



class CostEstimator:

    @staticmethod

    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:

        """Calculates the estimated cost of a request."""

        

        # Fallback to a default if model not found (or use highest price)

        pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gemini-2.5-flash"])

        

        input_cost = (input_tokens / 1_000_000) * pricing["input"]

        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        

        total_cost = input_cost + output_cost

        return round(total_cost, 6) # Precision to 6 decimal places



    @staticmethod

    def log_usage(session_id: str, model: str, usage: Dict[str, int]):

        """Logs usage and cost for auditing."""

        input_tok = usage.get("prompt_tokens", 0)

        output_tok = usage.get("completion_tokens", 0)

        

        cost = CostEstimator.calculate_cost(model, input_tok, output_tok)

        

        # In production, you would save this to a 'billing_ledger' table

        print(f"💰 BILLING [{session_id}]: {input_tok} in + {output_tok} out = ${cost:.6f} ({model})")