# Vera — magicpin AI Challenge Submission

**Team:** Team Antigravity (Shaunish Sharma)  
**Architecture:** 4-Layer Cascading Failover with 20-Second Death Clock  
**Tech Stack:** Python · FastAPI · LiteLLM · Pydantic  

---

## Overview

A stateful, production-resilient REST API built for the **magicpin Vera AI Challenge**. The bot acts as an intelligent assistant for retail merchants, synthesizing four distinct contexts (Category, Merchant, Trigger, and Customer) to generate highly specific, actionable, and engaging WhatsApp messages.

---

## Core Architecture & Tradeoffs

To survive the strict **30-second judge timeout** while operating on free-tier LLM APIs, this application implements a **4-layer cascading failover architecture**:

### Layer 0 — The 20-Second Death Clock (Global Safety Net)
A strict timing mechanism tracks elapsed execution time during batch processing in `/v1/tick`. If processing exceeds **20 seconds**, the system instantly halts all network-based LLM calls and routes remaining triggers to the Deterministic Fallback. This completely eliminates the risk of scoring a **0** due to timeout penalties.

### Layer 1 — Primary: `gemini/gemini-3.8-flash`
Configured with an **8-second timeout** and **0 retries**. Chosen for its strong reasoning capabilities on dense JSON payloads.

### Layer 1.5 — Overload Fallback: `gemini/gemini-3.5-flash-lite`
Activated **only** when Layer 1 returns a `503 Service Unavailable` / high-demand error. This lighter Gemini model absorbs traffic spikes without falling all the way to a non-Gemini provider.

### Layer 2 — Secondary: `groq/qwen3.8-27b`
If Gemini layers fail with non-overload errors (timeouts, rate limits), the router falls back to Groq. A custom **regex-based JSON cleaner** strips hallucinated markdown fences (`` ```json ... ``` ``) before Pydantic validation.

### Layer 3 — Deterministic Fallback (No Network)
If all LLMs fail or the death clock expires, the system returns safe, pre-written Pydantic responses.  
*Tradeoff:* We sacrifice specificity scores (~3/10) to **guarantee** we avoid the 0/10 timeout penalty, keeping the system 100% available.

```
Request ──▶ [Death Clock Check] ──▶ Gemini 3.8 Flash ──✓──▶ Response
                                         │ (503)
                                         ▼
                                   Gemini 3.5 Lite ──✓──▶ Response
                                         │ (fail)
                                         ▼
                                   Groq Qwen 27B ──✓──▶ Response
                                         │ (fail)
                                         ▼
                                   Deterministic Fallback ──▶ Response
```

---

## Advanced Conversation Handling

- **Pre-LLM Intent Filtering:** The `/v1/reply` endpoint intercepts incoming messages for hostility (e.g., "stop", "spam") and auto-replies (e.g., "Thank you for contacting") *before* invoking the LLM. It instantly returns a deterministic "end" action, saving compute and guaranteeing a perfect pass on edge-case scenarios.
- **Strict Adherence to Rubric:** The system prompt rigidly enforces single, low-friction binary CTAs (Yes/No) and forbids hallucinating metrics not present in the injected state.
- **Markdown-Resistant Parsing:** A regex cleaner handles Groq's tendency to wrap JSON output in markdown code fences.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/healthz` | GET | Health check with uptime and context counts |
| `/v1/metadata` | GET | Team info and model configuration |
| `/v1/context` | POST | Upsert context payloads (merchant, category, trigger, customer) |
| `/v1/tick` | POST | Batch-process triggers → generate proactive messages |
| `/v1/reply` | POST | Handle merchant replies with conversation history |

---

## Local Setup & Testing

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example and add your keys:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### 3. Run the Server

```bash
python main.py
# Server runs on http://0.0.0.0:8000
```

### 4. Run the Judge Simulator

```bash
python judge_simulator.py
```

---

## Project Structure

```
magicpin_project/
├── main.py                 # FastAPI app with cascading failover
├── judge_simulator.py      # LLM-powered judge (provided by magicpin)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (git-ignored)
├── .env.example            # Template for collaborators
├── .gitignore
├── dataset/                # Seed data (merchants, triggers, customers, categories)
├── examples/               # API call examples and case studies
├── challenge-brief.md      # Challenge specification
├── challenge-testing-brief.md
├── engagement-design.md
└── engagement-research.md
```

---

## Worst-Case Timing Analysis

| Scenario | Max Time | Under 30s? |
|----------|----------|------------|
| All triggers → Gemini succeeds | ~8s per trigger | ✅ |
| Gemini 503 → Lite succeeds | ~16s (8+8) | ✅ |
| Gemini timeout → Groq succeeds | ~16s (8+8) | ✅ |
| Both timeout → Fallback | ~16s + ~0ms | ✅ |
| Death clock fires at 20s | Remaining: ~0ms each | ✅ |

**Guaranteed maximum:** < 25 seconds (well under the 30-second deadline).

---

## License

Built for the magicpin AI Challenge 2025.
