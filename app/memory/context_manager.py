import tiktoken

import logging

from typing import List, Dict, Any, Callable, Awaitable



# Setup logging

logger = logging.getLogger("uvicorn")



class ContextManager:

    def __init__(self, max_tokens: int = 1000000):

        """

        Initialize the Context Manager.

        

        Args:

            max_tokens: The limit at which compression triggers. 

                        Default is 1M (Gemini 1.5 Flash limit).

                        Set lower (e.g., 2000) for stricter cost control or testing.

        """

        self.max_tokens = max_tokens

        

        # We use 'cl100k_base' (GPT-4 tokenizer) as a fast, local estimator.

        # It's not 100% identical to Gemini's tokenizer, but it's close enough 

        # for safety margins and runs locally without API calls.

        try:

            self.encoder = tiktoken.get_encoding("cl100k_base")

        except Exception as e:

            logger.warning(f"Failed to load tiktoken: {e}. Fallback to character estimation.")

            self.encoder = None



    def count_tokens(self, text: str) -> int:

        """Fast local token counting."""

        if not text:

            return 0

        

        if self.encoder:

            return len(self.encoder.encode(text))

        else:

            # Fallback: Rough estimate (4 chars ~= 1 token)

            return len(text) // 4



    def _estimate_conversation_tokens(self, messages: List[dict]) -> int:

        """Sum tokens of all messages in the history."""

        count = 0

        for msg in messages:

            content = msg.get("content", "")

            # Add a small buffer (3 tokens) for protocol overhead (role names, etc.)

            count += self.count_tokens(str(content)) + 3

        return count



    async def compress_history(

        self, 

        history: List[dict], 

        gateway_func: Callable[[List[dict], float], Awaitable[str]]

    ) -> List[dict]:

        """

        Smart Context Compression:

        1. Checks total tokens.

        2. If > Limit, triggers summarization of the oldest messages.

        3. Returns [Summary + Recent Messages].

        

        Args:

            history: List of message dicts [{'role': 'user', 'content': '...'}, ...]

            gateway_func: Async function to call the LLM for summarization.

        """

        total_tokens = self._estimate_conversation_tokens(history)

        # 🔊 LOUD DEBUG: Print this to confirm the function runs

        print(f"👀 DEBUG: Current History Tokens: {total_tokens} / Limit: {self.max_tokens}")

        # ✅ CASE 1: Under Limit (Do nothing)

        if total_tokens < self.max_tokens:

            return history



        # ✅ CASE 2: Over Limit (Trigger Compression)

        logger.info(f"🧹 GC Triggered: History size {total_tokens} > Limit {self.max_tokens}. Compressing...")



        # Strategy: Summarize the oldest 50% of the conversation

        # We keep the most recent messages intact for context continuity.

        cut_off = len(history) // 2

        

        # Safety check: If history is tiny (e.g. 1 massive message), don't slice empty lists

        if cut_off == 0:

            return history



        to_summarize = history[:cut_off]

        recent_memory = history[cut_off:]

        

        # Call the LLM to summarize the old block

        # We use the provided gateway_func to avoid circular imports

        try:

            summary_text = await gateway_func(to_summarize, 0.0)

            

            logger.info(f"📉 Compression Success. Summarized {len(to_summarize)} messages.")

            

            # Construct new compressed history

            # We inject the summary as a 'system' or 'assistant' message at the top

            compressed_message = {

                "role": "assistant", 

                "content": f"**System Note:** The beginning of this conversation was summarized to save space.\nSummary: {summary_text}"

            }

            

            # Return new history: [Summary] + [Recent Messages]

            return [compressed_message] + recent_memory



        except Exception as e:

            logger.error(f"⚠️ Compression Failed: {e}")

            # If summarization fails, we fall back to just returning the recent memory

            # (Better to lose context than to crash the app)

            return recent_memory