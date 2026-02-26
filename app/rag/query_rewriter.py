import json
from app.llm.gateway import LLMGateway

class QueryRewriter:
    def __init__(self):
        self.gateway = LLMGateway()

    async def rewrite_query(self, user_query: str) -> str:
        """
        Takes a raw user query and optimizes it for vector/hybrid retrieval.
        """
        prompt = (
            "You are a search optimization expert for an enterprise internal knowledge base.\n"
            "Your job is to rewrite the user's question to maximize document retrieval for corporate policies, HR guidelines, or IT support.\n"
            "1. Resolve acronyms if obvious.\n"
            "2. Expand synonyms (e.g., 'sick' -> 'unwell, medical leave, sick time').\n"
            "3. DO NOT turn the query into a general web search (e.g., never ask for 'medical advice').\n"
            "4. Keep specific IT error codes exactly as they are.\n\n"
            f"Original Query: {user_query}\n\n"
            "Return ONLY a JSON object in this format:\n"
            '{ "rewritten_query": "The highly optimized search string" }'
        )

        try:
            # We can borrow the structured output logic from your gateway
            # Or just do a zero-temp standard call and parse it
            response = await self.gateway.model.generate_content_async(
                prompt,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            return result.get("rewritten_query", user_query)
        except Exception as e:
            print(f"⚠️ Query Rewrite failed, falling back to original: {e}")
            return user_query