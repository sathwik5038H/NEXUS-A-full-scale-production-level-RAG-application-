import os

import json

import asyncio

import google.generativeai as genai

from typing import AsyncGenerator, List, Dict, Any, Optional



from app.llm.prompts import ENTERPRISE_SYSTEM_PROMPT

from app.schemas.chat import StructuredResponse

from app.llm.guardrails import Guardrails

from app.utils.costing import CostEstimator

from app.schemas.chat import RAGResponse



class LLMGateway:

    def __init__(self):

        # 1. Setup Google Gemini

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:

            raise ValueError("GEMINI_API_KEY is missing in .env file")

        

        genai.configure(api_key=api_key)

        

        # We use the standard stable model

        self.default_model = 'gemini-2.5-flash' 

        

        # Initialize the model with the System Prompt

        self.model = genai.GenerativeModel(

            model_name=self.default_model,

            system_instruction=ENTERPRISE_SYSTEM_PROMPT

        )

        

    async def chat_rag(self, user_message: str, context: str, model: str = "gemini-2.5-flash") -> RAGResponse:

        """Forces the LLM to answer using ONLY provided context and return JSON."""

        

        system_prompt = (

            "You are an enterprise knowledge assistant. Answer the user's question using ONLY the provided CONTEXT.\n"

            "You MUST cite your sources by including the chunk numbers in a list (e.g., [1, 2]).\n"

            "If the answer is NOT in the CONTEXT, you must say 'I don't know' and return an empty citations list [].\n"

            "IMPORTANT: Output MUST strictly follow this JSON schema:\n"

            '{ "answer": string, "citations": list[int], "confidence": float }'

        )

        

        final_prompt = f"{system_prompt}\n\n{context}\n\nUSER QUESTION:\n{user_message}"

        

        # Enforce Determinism & JSON

        generation_config = genai.types.GenerationConfig(

            temperature=0.0, # Zero creativity allowed for RAG

            response_mime_type="application/json"

        )

        

        # Use a fresh chat to avoid history pollution

        chat = self.model.start_chat(history=[])

        response = await chat.send_message_async(final_prompt, generation_config=generation_config)

        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count
            }
            CostEstimator.log_usage("rag-session", model, usage)

        try:

            # Parse the JSON and validate it against our Pydantic schema

            json_data = json.loads(response.text)

            return RAGResponse(**json_data)

        except Exception as e:

            raise ValueError(f"LLM failed to follow JSON schema: {e}")



    def _format_messages(self, messages: List[dict]) -> tuple[List[dict], str]:

        """

        Helper to convert OpenAI-style messages to Gemini format.

        Returns: (history, last_user_message)

        """

        chat_history = []

        last_user_message = ""



        for msg in messages:

            if msg["role"] == "user":

                last_user_message = msg["content"]

                chat_history.append({"role": "user", "parts": [msg["content"]]})

            elif msg["role"] == "assistant":

                chat_history.append({"role": "model", "parts": [msg["content"]]})

            # System prompt is handled in __init__, so we skip it here

        

        # Return history excluding the very last message (which is the new prompt)

        return chat_history[:-1], last_user_message



    async def stream_chat(

        self, 

        messages: List[dict], 

        temperature: float = 0.7,

        model: str = "gemini-2.5-flash"

    ) -> AsyncGenerator[str, None]:

        """

        Standard Chat Mode: Streams tokens for a natural conversation feel.

        Also logs cost usage at the end of the stream.

        """

        

        # 1. Prepare History

        history, last_message = self._format_messages(messages)



        # 2. Start Chat Session

        chat = self.model.start_chat(history=history)



        # 3. Send Request (Streaming)

        response_stream = await chat.send_message_async(

            last_message, 

            generation_config=genai.types.GenerationConfig(

                temperature=temperature

            ),

            stream=True

        )



        # 4. Yield Tokens & Capture Usage

        async for chunk in response_stream:

            # 🛡️ FIX: Safely access text. 

            # The final chunk often contains only metadata (finish_reason=1) and no text.

            # Accessing .text raises a ValueError in that specific case.

            try:

                if chunk.text:

                    yield chunk.text

            except ValueError:

                # This explicitly handles the "finish_reason is 1" error

                # We ignore the missing text and proceed to check metadata

                pass

            

            # 📊 Capture Token Usage (Gemini sends this in the final chunk)

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:

                usage = {

                    "prompt_tokens": chunk.usage_metadata.prompt_token_count,

                    "completion_tokens": chunk.usage_metadata.candidates_token_count

                }

                # Log usage to our Cost Estimator

                CostEstimator.log_usage("stream-session", model, usage)



    async def chat_structured(

        self, 

        messages: List[dict], 

        model: str = "gemini-2.5-flash"

    ) -> StructuredResponse:

        """

        Structured Mode: Deterministic, Validated, and Retries on failure.

        Returns a Pydantic object (StructuredResponse), NOT a string.

        """



        # 1. Prepare History

        history, last_message = self._format_messages(messages)

        

        # 2. Configure for Determinism (JSON Mode)

        # Temperature 0 = strict. MIME type = JSON.

        generation_config = genai.types.GenerationConfig(

            temperature=0.0,

            top_p=1.0,

            response_mime_type="application/json"

        )



        # 3. Retry Loop (The "Validation Pattern")

        max_retries = 2

        attempt = 0

        

        # Append instruction to ensure model knows the schema expectation

        current_prompt = (

            f"{last_message}\n\n"

            "IMPORTANT: Output must strictly follow this JSON schema:\n"

            "{ \"answer\": string, \"confidence\": float (0-1), \"citations\": list[string] }"

        )



        while attempt <= max_retries:

            try:

                # Start fresh chat session each retry to avoid polluting history with bad JSON

                chat = self.model.start_chat(history=history)

                

                # Send (Non-streaming)

                response = await chat.send_message_async(

                    current_prompt,

                    generation_config=generation_config

                )

                

                # 📊 Capture Token Usage

                if response.usage_metadata:

                     usage = {

                        "prompt_tokens": response.usage_metadata.prompt_token_count,

                        "completion_tokens": response.usage_metadata.candidates_token_count

                    }

                     CostEstimator.log_usage("struct-session", model, usage)



                raw_text = response.text

                

                # 4. Post-LLM Guardrail (Sanitize)

                cleaned_text = Guardrails.validate_output(raw_text)

                

                # 5. Parse & Validate

                json_data = json.loads(cleaned_text)

                

                # Pydantic validation (throws error if schema is wrong)

                validated_obj = StructuredResponse(**json_data)

                

                return validated_obj



            except (json.JSONDecodeError, ValueError, Exception) as e:

                print(f"⚠️ Structured Mode Failed (Attempt {attempt+1}/{max_retries+1}): {e}")

                attempt += 1

                

                # Feedback Loop: Tell the LLM what went wrong to fix it

                current_prompt = (

                    f"{last_message}\n\n"

                    f"SYSTEM ALERT: Your previous response was invalid JSON. Error: {str(e)}.\n"

                    "Please correct the format and try again."

                )

                

        # If we fail after retries

        raise ValueError("LLM failed to generate valid structured JSON after multiple attempts.")



    async def generate_summary(self, messages: List[dict], temperature: float = 0.0) -> str:

        """

        Helper for the ContextManager to run summarization tasks.

        Uses a separate, non-streaming call to avoid disrupting the main flow.

        """

        history, last_message = self._format_messages(messages)

        

        # We start a new chat with no history to keep the summary focused on the input text

        # (or passing the history if you want it to summarize the history itself)

        chat = self.model.start_chat(history=[]) 

        

        # We usually pass the text to be summarized as the 'last_message'

        response = await chat.send_message_async(

            last_message,

            generation_config=genai.types.GenerationConfig(

                temperature=temperature

            )

        )

        

        # Log cost for summarization tasks too!

        if response.usage_metadata:

             usage = {

                "prompt_tokens": response.usage_metadata.prompt_token_count,

                "completion_tokens": response.usage_metadata.candidates_token_count

            }

             CostEstimator.log_usage("system-summary", self.default_model, usage)

             

        return response.text