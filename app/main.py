from fastapi import UploadFile, File

from app.rag.ingestion import IngestionService

from fastapi import FastAPI, HTTPException

from fastapi.responses import StreamingResponse, JSONResponse

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from typing import List, AsyncGenerator

# from app.rag.retriever import RetrieverService
from app.rag.hybrid_retriever import HybridRetriever



# Import internal modules

from app.schemas.chat import ChatRequest, Message, StructuredResponse

from app.llm.gateway import LLMGateway

from app.memory.session_store import SessionStore

from app.llm.guardrails import Guardrails

from app.utils.costing import CostEstimator

from app.memory.context_manager import ContextManager



# Load environment variables

load_dotenv()



# Initialize core services

gateway = LLMGateway()

session_store = SessionStore()

ingestion_service = IngestionService()

# retriever_service = RetrieverService()
retriever_service = HybridRetriever()



# Initialize Context Manager

# In production, set max_tokens=1_000_000 for Gemini 1.5.

# We set it to 2000 here so you can easily test the compression logic.

context_manager = ContextManager(max_tokens=1_000_000)



@asynccontextmanager

async def lifespan(app: FastAPI):

    """

    Lifespan events: Code that runs when the server starts/stops.

    """

    print("🚀 Nexus Gateway Starting...")

    print(f"✅ Active Model: {gateway.default_model}")

    yield

    print("🛑 Nexus Gateway Shutting Down...")



app = FastAPI(title="Nexus AI Gateway", version="1.0.0", lifespan=lifespan)



@app.post("/chat")

async def chat_endpoint(request: ChatRequest):



    # === RAG MODE (Intercepts immediately) ===

    if request.mode == "rag":

        # 1. Retrieve Context (Hybrid + Reranked)
        context_str, raw_chunks, debug_data = await retriever_service.retrieve_context(
            request.message, 
            top_k=3, 
            debug=request.rag_debug
        )

       

        # 2. Generate Answer

        rag_response = await gateway.chat_rag(request.message, context_str, request.model)

       

       

        # 3. HALLUCINATION GUARDRAIL: Verify Citations

        valid_ids = [chunk["id"] for chunk in raw_chunks]

        verified_citations = []

        for cited_id in rag_response.citations:

            if cited_id in valid_ids:

                verified_citations.append(cited_id)

            else:

                print(f"🚨 Hallucination Blocked: LLM cited chunk [{cited_id}] which does not exist!")

       

        rag_response.citations = verified_citations

        final_response = rag_response.model_dump()
        if request.rag_debug:
            final_response["debug_info"] = debug_data
            
        return final_response





    # === STANDARD STREAMING & STRUCTURED MODES ===

    # 🛡️ 1. Security Guardrail: Pre-LLM Input Validation

    if not Guardrails.validate_input(request.message):

        print(f"🚨 Security Alert: Blocked input from session {request.session_id}")

        raise HTTPException(status_code=400, detail="Security Alert: Malicious input detected.")



    # 🧠 2. Retrieve Conversation History

    history_dicts = await session_store.get_history(request.session_id)

   

    # 🧹 3. Context Management (Memory Compression)

    async def summarizer_func(msgs: List[dict], temp: float = 0.0) -> str:

        summary_prompt = [

            {"role": "user", "content": "Summarize the following conversation history into a concise paragraph to retain context:"},

            {"role": "user", "content": str(msgs)}

        ]

        return await gateway.generate_summary(summary_prompt, temperature=temp)



    optimized_history = await context_manager.compress_history(

        history_dicts,

        summarizer_func

    )

   

    if len(optimized_history) < len(history_dicts):

        print(f"📉 Context Compressed: {len(history_dicts)} -> {len(optimized_history)} messages")



    # 📦 4. Construct Final Payload

    current_user_msg = Message(role="user", content=request.message)

    messages_payload = optimized_history.copy()

    messages_payload.append(current_user_msg.model_dump())



    # 🔀 5. Processing Mode Selection

    if request.mode == "structured":

        try:

            result: StructuredResponse = await gateway.chat_structured(

                messages=messages_payload,

                model=request.model

            )

            await session_store.add_message(request.session_id, current_user_msg)

            assistant_msg = Message(role="assistant", content=result.model_dump_json())

            await session_store.add_message(request.session_id, assistant_msg)

            return result

        except ValueError as e:

             raise HTTPException(status_code=500, detail=str(e))



    elif request.mode == "chat":

        async def response_generator():

            full_assistant_response = []

            async for token in gateway.stream_chat(

                messages=messages_payload,

                temperature=request.temperature,

                model=request.model

            ):

                full_assistant_response.append(token)

                yield token

           

            complete_response_text = "".join(full_assistant_response)

            await session_store.add_message(request.session_id, current_user_msg)

            assistant_msg = Message(role="assistant", content=complete_response_text)

            await session_store.add_message(request.session_id, assistant_msg)



        return StreamingResponse(response_generator(), media_type="text/plain")

   

   





@app.post("/documents/upload")

async def upload_document(file: UploadFile = File(...)):

    """Uploads a PDF, chunks it, and adds it to the RAG Knowledge Base."""

    if not file.filename.endswith(".pdf"):

        raise HTTPException(status_code=400, detail="Only PDF files are currently supported.")

   

    file_bytes = await file.read()

    result = await ingestion_service.process_pdf(file_bytes, file.filename)

    return result
  






if __name__ == "__main__":

    import uvicorn

    # Use 0.0.0.0 to allow access from other machines (if needed)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)