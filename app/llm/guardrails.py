# nexus/app/llm/guardrails.py

import re

from typing import List, Optional



class Guardrails:

    

    # 1. Pre-LLM: Block malicious inputs

    @staticmethod

    def validate_input(message: str) -> bool:

        """

        Returns False if injection/jailbreak attempt detected.

        """

        risky_patterns = [

            r"ignore previous instructions",

            r"system prompt",

            r"delete all data",

            r"you are a hacked",

            r"pwned"

        ]

        

        message_lower = message.lower()

        for pattern in risky_patterns:

            if re.search(pattern, message_lower):

                return False

        return True



    # 2. Post-LLM: Sanitize Output

    @staticmethod

    def validate_output(response_text: str) -> str:

        """

        Cleans response of dangerous leaks.

        """

        # Block API Key Leaks (Basic Regex for sk-...)

        if "sk-" in response_text and len(response_text) > 20:

             return "[REDACTED: API KEY DETECTED]"

             

        # Block Hallucinated Citations (Simple check for now)

        # In a real RAG, you'd check if the URL exists in your DB

        if "[doc_id]" in response_text:

             # Logic to verify doc_id would go here

             pass

             

        return response_text