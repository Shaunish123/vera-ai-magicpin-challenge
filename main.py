import re
import time
import uuid
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Literal

from dotenv import load_dotenv
load_dotenv()  # Load GEMINI_API_KEY, GROQ_API_KEY from .env

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import litellm

app = FastAPI(title="Vera - Merchant AI Assistant (Refactored)")

# --- Configuration ---
DEATH_CLOCK_SECONDS = 20      # Stop calling LLMs after this many seconds into a /v1/tick request
PER_LLM_TIMEOUT = 8           # Each individual LLM call gets at most 8 seconds
PRIMARY_MODEL = "gemini/gemini-3.8-flash"
FALLBACK_PRIMARY_MODEL = "gemini/gemini-3.5-flash-lite"  # Used when primary returns 503 high-demand
SECONDARY_MODEL = "groq/qwen/qwen3.8-27b"

# --- State Management (In-Memory) ---
START_TIME = time.time()

# Keyed by (scope, context_id). Value: {"version": int, "payload": dict}
contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}

# Keyed by conversation_id. Value: list of message dicts [{"from": str, "msg": str}]
conversations: Dict[str, List[Dict[str, str]]] = {}

# --- Pydantic Models for Endpoints ---

# POST /v1/context
class ContextPayload(BaseModel):
    scope: str = Field(..., description="E.g., 'merchant', 'category', 'trigger', 'customer'")
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

class ContextSuccessResponse(BaseModel):
    accepted: bool
    ack_id: str
    stored_at: str

class ContextErrorResponse(BaseModel):
    accepted: bool
    reason: str
    current_version: int

# POST /v1/tick
class TickRequest(BaseModel):
    now: str
    available_triggers: List[str]

class ActionObject(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str = "vera"
    trigger_id: str
    template_name: str = "vera_v1"
    template_params: List[Any] = Field(default_factory=list)
    body: str
    cta: str
    suppression_key: str
    rationale: str

class TickResponse(BaseModel):
    actions: List[ActionObject]

# POST /v1/reply
class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: str
    cta: str
    rationale: str

# --- LLM Structured Output Models ---

class TickLLMResponse(BaseModel):
    body: str = Field(..., description="The message body to send.")
    cta: str = Field(..., description="A single, specific yes/no call to action based on the trigger.")
    rationale: str = Field(..., description="Why this message was chosen based on the trigger and category voice.")

class ReplyLLMResponse(BaseModel):
    action: Literal["send", "wait", "end"] = Field(
        ..., 
        description="'send' to reply, 'wait' if more info is expected from another system, 'end' if auto-reply or hostile."
    )
    body: str = Field(..., description="The reply message to the merchant, or empty if ending.")
    cta: str = Field(..., description="A single, specific yes/no call to action, or empty if ending.")
    rationale: str = Field(..., description="Why this action and message were chosen based on the conversation history.")


# --- Helper Functions ---

def get_current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_tick_prompt(trigger_payload: dict, merchant_payload: dict, category_payload: dict, customer_payload: Optional[dict]) -> str:
    prompt = "Generate a proactive message based on the following contexts.\n\n"
    prompt += f"**TRIGGER CONTEXT**:\n{trigger_payload}\n\n"
    prompt += f"**MERCHANT CONTEXT**:\n{merchant_payload}\n\n"
    prompt += f"**CATEGORY CONTEXT (Tone & Allowed Vocabulary)**:\n{category_payload}\n\n"
    if customer_payload:
        prompt += f"**CUSTOMER CONTEXT**:\n{customer_payload}\n\n"
    return prompt

def build_reply_prompt(history: List[Dict[str, str]], merchant_payload: dict, category_payload: dict) -> str:
    prompt = "Evaluate the conversation and decide the next action.\n\n"
    prompt += f"**MERCHANT CONTEXT**:\n{merchant_payload}\n\n"
    prompt += f"**CATEGORY CONTEXT (Tone & Allowed Vocabulary)**:\n{category_payload}\n\n"
    prompt += "**CONVERSATION HISTORY**:\n"
    for turn in history:
        prompt += f"[{turn['from']}]: {turn['msg']}\n"
    return prompt

SYSTEM_PROMPT = """
You are Vera, a Senior AI Assistant for magicpin merchants.
You help local businesses (salons, dentists, restaurants) improve performance and run campaigns.

STRICT CONSTRAINTS:
1. No Hallucinations: Rely 100% on the provided JSON payloads. Do NOT invent numbers, dates, features, or metrics.
2. Voice Match: Strictly adopt the tone and vocabulary guidelines specified in the Category Context (e.g., clinical for dentists, vibrant for salons).
3. High Specificity: Quote exact metrics from the Merchant/Trigger payloads (e.g., "views dropped by 5%"). Do NOT use generic marketing copy.
4. Engagement Compulsion: Messages MUST end with a single, highly specific yes/no Call to Action based entirely on the Trigger payload.
5. Auto-Reply Detection: If evaluating a merchant's reply, check if it looks like an automated WhatsApp response (e.g., "Thank you for contacting..."), or if the merchant is hostile/uninterested. If so, set the action to "end" to gracefully stop.

Output EXACTLY matching the required JSON schema.
"""


# =============================================================================
# Layer 4: Deterministic Fallback (No LLM)
# =============================================================================

def fallback_tick_response(trigger_payload: dict) -> TickLLMResponse:
    """Produces a valid TickLLMResponse without calling any LLM.
    
    Extracts whatever context it can from the trigger payload to produce
    a somewhat-relevant message. This is the safety net.
    """
    trigger_type = trigger_payload.get("type", trigger_payload.get("trigger_type", "update"))
    merchant_name = trigger_payload.get("merchant_name", "your business")

    body = (
        f"Hi! We noticed some activity regarding {trigger_type} for {merchant_name}. "
        f"Would you like to take a look and see how we can help you improve your performance?"
    )
    return TickLLMResponse(
        body=body,
        cta="Would you like to review the details? Reply Yes or No.",
        rationale=f"Deterministic fallback — LLM layers exhausted or time-capped. Trigger type: {trigger_type}."
    )


def fallback_reply_response() -> ReplyLLMResponse:
    """Produces a valid ReplyLLMResponse without calling any LLM."""
    return ReplyLLMResponse(
        action="send",
        body="Thank you for your response! I'm here to help you get the most out of magicpin. Is there anything specific you'd like assistance with?",
        cta="Would you like to explore ways to boost your visibility? Reply Yes or No.",
        rationale="Deterministic fallback — LLM layers exhausted or time-capped."
    )


# =============================================================================
# Markdown / JSON Cleaner (for Groq hallucinated wrappers)
# =============================================================================

_MD_JSON_BLOCK = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

def clean_llm_json(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output.
    
    Groq models frequently wrap valid JSON in ```json ... ``` blocks.
    This function extracts the inner content so Pydantic can parse it.
    """
    if raw is None:
        return ""
    raw = raw.strip()
    match = _MD_JSON_BLOCK.search(raw)
    if match:
        return match.group(1).strip()
    return raw


# =============================================================================
# Resilient LLM Caller — 4-Layer Cascade
# =============================================================================

def call_llm_with_failover(
    prompt: str,
    response_model: type[BaseModel],
    endpoint_start_time: float,
) -> BaseModel:
    """Attempts to get a structured LLM response using a cascading failover:
    
    Layer 1:   Gemini 3.8 Flash (primary)    — 8s timeout, 0 retries
    Layer 1.5: Gemini 3.5 Flash Lite (lite)   — 8s timeout, only if L1 hit 503/overload
    Layer 2:   Groq llama-3.1-8b (secondary)  — 8s timeout, 0 retries, markdown cleaning
    Layer 3:   Deterministic fallback (instant, no network)
    
    The caller is responsible for the Layer 0 (death-clock) check *before*
    calling this function, but we also guard internally just in case.
    """

    # --- Guard: if we're already past the death clock, skip straight to fallback ---
    elapsed = time.time() - endpoint_start_time
    if elapsed > DEATH_CLOCK_SECONDS:
        print(f"[FAILOVER] Death-clock exceeded ({elapsed:.1f}s) — skipping to deterministic fallback.")
        return _deterministic_fallback(response_model, {})

    # --- Layer 1: Gemini ---
    try:
        response = litellm.completion(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
            temperature=0.1,
            timeout=PER_LLM_TIMEOUT,
            num_retries=0,
        )
        content = response.choices[0].message.content
        result = response_model.model_validate_json(clean_llm_json(content))
        print(f"[LLM] Layer 1 (Gemini) succeeded in {time.time() - endpoint_start_time:.1f}s")
        return result
    except Exception as e:
        layer1_error = e
        print(f"[FAILOVER] Layer 1 (Gemini) failed: {type(e).__name__}: {e}")

    # --- Check time budget before Layer 1.5 ---
    elapsed = time.time() - endpoint_start_time
    if elapsed > DEATH_CLOCK_SECONDS:
        print(f"[FAILOVER] Death-clock exceeded after Layer 1 ({elapsed:.1f}s) — falling back.")
        return _deterministic_fallback(response_model, {})

    # --- Layer 1.5: Gemini Lite (only if Layer 1 failed with 503 / overload / ServiceUnavailable) ---
    l1_err_str = str(layer1_error).lower() if 'layer1_error' in dir() else ""
    is_overload = any(kw in l1_err_str for kw in ["503", "overload", "high demand", "service unavailable", "resource exhausted"])
    if is_overload:
        try:
            print(f"[LLM] Layer 1.5 — Primary overloaded, trying Gemini Lite...")
            response = litellm.completion(
                model=FALLBACK_PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_model,
                temperature=0.1,
                timeout=PER_LLM_TIMEOUT,
                num_retries=0,
            )
            content = response.choices[0].message.content
            result = response_model.model_validate_json(clean_llm_json(content))
            print(f"[LLM] Layer 1.5 (Gemini Lite) succeeded in {time.time() - endpoint_start_time:.1f}s")
            return result
        except Exception as e:
            print(f"[FAILOVER] Layer 1.5 (Gemini Lite) failed: {type(e).__name__}: {e}")

    # --- Check time budget before Layer 2 ---
    elapsed = time.time() - endpoint_start_time
    if elapsed > DEATH_CLOCK_SECONDS:
        print(f"[FAILOVER] Death-clock exceeded after Layer 1.5 ({elapsed:.1f}s) — falling back.")
        return _deterministic_fallback(response_model, {})

    # --- Layer 2: Groq ---
    try:
        response = litellm.completion(
            model=SECONDARY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
            temperature=0.1,
            timeout=PER_LLM_TIMEOUT,
            num_retries=0,
        )
        content = response.choices[0].message.content
        cleaned = clean_llm_json(content)
        result = response_model.model_validate_json(cleaned)
        print(f"[LLM] Layer 2 (Groq) succeeded in {time.time() - endpoint_start_time:.1f}s")
        return result
    except ValidationError as ve:
        print(f"[FAILOVER] Layer 2 (Groq) validation failed: {ve}")
    except Exception as e:
        print(f"[FAILOVER] Layer 2 (Groq) failed: {type(e).__name__}: {e}")

    # --- Layer 3: Deterministic Fallback ---
    print(f"[FAILOVER] All LLM layers exhausted — using deterministic fallback.")
    return _deterministic_fallback(response_model, {})


def _deterministic_fallback(
    response_model: type[BaseModel],
    trigger_payload: dict,
) -> BaseModel:
    """Routes to the correct fallback based on the response model type."""
    if response_model is TickLLMResponse:
        return fallback_tick_response(trigger_payload)
    elif response_model is ReplyLLMResponse:
        return fallback_reply_response()
    else:
        # Should never happen, but just in case — return TickLLMResponse as safe default
        return fallback_tick_response(trigger_payload)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/v1/healthz")
async def healthz():
    uptime = int(time.time() - START_TIME)
    
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _data in contexts.items():
        if scope in counts:
            counts[scope] += 1
            
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Antigravity",
        "model": f"{PRIMARY_MODEL} -> {SECONDARY_MODEL} -> deterministic",
        "approach": "fastapi+litellm, 3-layer cascading failover with 20s death-clock"
    }


@app.post("/v1/context")
async def update_context(data: ContextPayload):
    key = (data.scope, data.context_id)
    current_state = contexts.get(key)
    
    if current_state is None or data.version > current_state["version"]:
        contexts[key] = {
            "version": data.version,
            "payload": data.payload
        }
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "accepted": True,
                "ack_id": f"ack_{data.context_id}_v{data.version}",
                "stored_at": get_current_timestamp()
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "accepted": False,
            "reason": "stale_version",
            "current_version": current_state["version"]
        }
    )


@app.post("/v1/tick", response_model=TickResponse)
async def process_tick(req: TickRequest):
    """Process a batch of triggers with a 20-second death clock.
    
    The judge gives us 30 seconds total. We stop calling LLMs at 20 seconds
    and route everything remaining through the deterministic fallback to
    guarantee we return well before the deadline.
    """
    endpoint_start = time.time()
    actions = []
    
    for trigger_id in req.available_triggers:
        try:
            # ──────────────────────────────────────────────────
            # DEATH CLOCK CHECK — before processing each trigger
            # ──────────────────────────────────────────────────
            elapsed = time.time() - endpoint_start
            use_fallback_only = elapsed > DEATH_CLOCK_SECONDS

            if use_fallback_only:
                print(f"[DEATH-CLOCK] {elapsed:.1f}s elapsed — routing trigger {trigger_id} to fallback.")

            # 1. Fetch Trigger
            trigger_state = contexts.get(("trigger", trigger_id))
            if not trigger_state:
                continue
            trigger_payload = trigger_state["payload"]
            
            # 2. Extract merchant_id & Fetch Merchant
            merchant_id = trigger_payload.get("merchant_id")
            if not merchant_id:
                continue
                
            merchant_state = contexts.get(("merchant", merchant_id))
            if not merchant_state:
                continue
            merchant_payload = merchant_state["payload"]
            
            # 3. Extract category_slug & Fetch Category
            category_slug = merchant_payload.get("category_slug")
            if not category_slug:
                category_slug = merchant_payload.get("identity", {}).get("category_slug")
                
            category_payload = {}
            if category_slug:
                category_state = contexts.get(("category", category_slug))
                if category_state:
                    category_payload = category_state["payload"]
                    
            # 4. Fetch optional Customer
            customer_id = trigger_payload.get("customer_id")
            customer_payload = None
            if customer_id:
                customer_state = contexts.get(("customer", customer_id))
                if customer_state:
                    customer_payload = customer_state["payload"]

            # 5. Get LLM response (cascading or fallback-only)
            if use_fallback_only:
                llm_result = fallback_tick_response(trigger_payload)
            else:
                prompt = build_tick_prompt(trigger_payload, merchant_payload, category_payload, customer_payload)
                llm_result = call_llm_with_failover(prompt, TickLLMResponse, endpoint_start)
            
            # 6. Construct ActionObject
            conv_id = f"conv_{uuid.uuid4().hex[:12]}"
            
            action = ActionObject(
                conversation_id=conv_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                send_as="vera",
                trigger_id=trigger_id,
                template_name="vera_v1",
                template_params=[],
                body=llm_result.body,
                cta=llm_result.cta,
                suppression_key=trigger_payload.get("suppression_key", f"sup_{trigger_id}"),
                rationale=llm_result.rationale
            )
            actions.append(action)

        except Exception as e:
            # Catch-all: NEVER let a single trigger crash the batch.
            print(f"[ERROR] Unhandled exception for trigger {trigger_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    total_time = time.time() - endpoint_start
    print(f"[TICK] Processed {len(actions)}/{len(req.available_triggers)} triggers in {total_time:.1f}s")
    return TickResponse(actions=actions)


@app.post("/v1/reply", response_model=ReplyResponse)
async def process_reply(req: ReplyRequest):
    """Process a reply with the same cascading failover.
    
    Individual replies are not batched, but we still apply time-safety
    and full failover to avoid 500 errors.
    """
    reply_start = time.time()

    try:
        # Update Conversation History
        if req.conversation_id not in conversations:
            conversations[req.conversation_id] = []
            
        conversations[req.conversation_id].append({
            "from": req.from_role,
            "msg": req.message
        })
        
        # Fetch Contexts
        merchant_state = contexts.get(("merchant", req.merchant_id))
        merchant_payload = merchant_state["payload"] if merchant_state else {}
        
        category_slug = merchant_payload.get("category_slug") or merchant_payload.get("identity", {}).get("category_slug")
        category_state = contexts.get(("category", category_slug)) if category_slug else None
        category_payload = category_state["payload"] if category_state else {}
        
        # Call LLM with failover
        prompt = build_reply_prompt(
            history=conversations[req.conversation_id],
            merchant_payload=merchant_payload,
            category_payload=category_payload
        )
        
        llm_result: ReplyLLMResponse = call_llm_with_failover(prompt, ReplyLLMResponse, reply_start)
        
        # If the LLM decides to send a message, also append Vera's response to the history
        if llm_result.action == "send" and llm_result.body:
            full_reply = f"{llm_result.body} {llm_result.cta}".strip()
            conversations[req.conversation_id].append({
                "from": "vera",
                "msg": full_reply
            })
            
        return ReplyResponse(
            action=llm_result.action,
            body=llm_result.body,
            cta=llm_result.cta,
            rationale=llm_result.rationale
        )
    except Exception as e:
        # NEVER return a 500. Fall back to a safe deterministic reply.
        print(f"[ERROR] Unhandled exception in /v1/reply: {type(e).__name__}: {e}")
        traceback.print_exc()
        fb = fallback_reply_response()
        return ReplyResponse(
            action=fb.action,
            body=fb.body,
            cta=fb.cta,
            rationale=fb.rationale
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
