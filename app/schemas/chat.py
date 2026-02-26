# nexus/app/schemas/chat.py

from pydantic import BaseModel, Field, field_validator

from typing import List, Literal, Optional



class Message(BaseModel):

    role: Literal["system", "user", "assistant"]

    content: str



class ChatRequest(BaseModel):

    session_id: str = Field(..., description="Unique identifier for the user session")

    message: str = Field(..., min_length=1, description="The user's input message")

    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Creativity control")

    model: str = Field("gemini-2.5-flash", description="Target model ID")

    mode: str = Field("standard", description="'standard' or 'rag'")

    

    # ✅ New: Mode selection

    mode: Literal["chat", "structured", "rag"] = Field("chat", description="Output mode")

    rag_debug: bool = Field(False, description="Returns internal retrieval routing data")



class RAGResponse(BaseModel):

    answer: str = Field(..., description="The answer to the user's question")

    citations: List[int] = Field(..., description="List of chunk IDs used as sources")

    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")



class ChatResponse(BaseModel):

    response: str

    token_usage: dict



# ✅ New: The Strict Contract

class StructuredResponse(BaseModel):

    answer: str = Field(..., description="The direct answer to the user's question")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

    citations: List[str] = Field(default_factory=list, description="List of sources used")



    # Guardrail: Prevent Markdown in strict answers (optional but good for APIs)

    @field_validator('answer')

    def no_markdown_headers(cls, v):

        if "##" in v or "```" in v:

            # We cleanse it rather than crashing

            return v.replace("##", "").replace("```", "")

        return v