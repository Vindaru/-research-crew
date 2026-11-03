# Research Crew

A small team of AI agents that answers research questions with **every claim backed by a quoted source**,
and a test suite that proves it.

```
Your question
     │
     ▼
Researcher ──► searches the web, writes notes: fact + link + exact quote      ◄── M1 (this milestone)
     │
     ▼
Critic ──────► checks each fact against its quote. Weak? ──► back to Researcher   ◄── M2
     │ (approved)
     ▼
Writer ──────► short report, every sentence cited                                  ◄── M2
```

Around it: golden set, code checks, LLM judge (M3), Langfuse traces, and a CI quality gate (M4).

---

## 1. Explained like you're 5 🧸

| Thing | The kid version | The grown-up version |
|---|---|---|
| **Researcher** | A kid sent to the library with a notebook. For every fact, they must write the book's name and copy the sentence. | A LangChain agent with one tool, `web_search`, that must return `ResearchNotes`. |
| **Tool** (`web_search`) | The library. The kid can only know what's in the books they opened. | A function the LLM may call. Here it calls Tavily, a search API built for agents. |
| **Structured output** | A form with boxes: *fact*, *book*, *copied sentence*. No box can be empty. | `response_format=ResearchNotes` (Pydantic). Malformed answers fail immediately. |
| **Gaps** | "I couldn't find this in any book" instead of making it up. | The `gaps` list. Honest "don't know" beats a confident guess. |
| **Fake mode** | Practising with three pretend books at home before going to the real library. | `research_fake` + `tests/fixtures`: no API key, no cost, same result every run. |
| **Langfuse** | A camera recording every book the kid opened and how long it took. | Traces of every LLM call and tool call, with cost and latency. |

---

## 2. What's in the repo

| File | What it is |
|---|---|
| `research_crew/models.py` | The forms the agents pass around: `Source`, `Finding`, `ResearchNotes` |
| `research_crew/search.py` | Real search (Tavily) and fake search (fixtures) with the same shape |
| `research_crew/researcher.py` | The Researcher: system prompt, `web_search` tool, `research()` and `research_fake()` |
| `research_crew/tracing.py` | Turns on Langfuse only if keys are set |
| `main.py` | Ask a question from your terminal |
| `tests/test_researcher.py` | 5 offline tests + 1 live test (skips without keys) |
| `.github/workflows/tests.yml` | Runs the offline tests on every push |

---

## 3. How to use it (on your Mac)

### Step A: run it offline, free, in 2 minutes

```bash
git clone https://github.com/Vindaru/research-crew.git
cd research-crew
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py --fake "What is retrieval-augmented generation?"
pytest -q          # expect: 5 passed, 1 skipped
```

You'll see numbered findings, each with a source URL and the exact quote. Try a question the fixtures
can't answer (`python main.py --fake "Who won the 1987 chess olympiad?"`) and you get a **GAP** instead of a guess.

### Step B: the real thing (Claude + live web)

1. Get a free Tavily key at https://app.tavily.com.
2. `cp .env.example .env` and fill in `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`.
3. Run:

```bash
set -a && source .env && set +a
python main.py "What changed in the EU AI Act in 2026?"
pytest -q          # now the live test runs too: 6 passed
```

### Step C: watch it think (Langfuse)

Start your local Langfuse (the same one ClaimStream uses), put its keys in `.env`, and run Step B again.
Open Langfuse → **Traces** → the newest `researcher` trace. You'll see the loop:
LLM decides a search → `web_search` call and its results → maybe another search → final structured notes,
with tokens, cost and seconds for each step.

**What to look for:** how many searches it used, whether each quote really appears in the tool output,
and which step was slowest. That's how you debug an agent.

### Where this pattern is used in real life

- **Analyst and market research:** "summarise what competitors launched this quarter, with sources".
- **Due diligence / compliance:** every statement must be traceable to a document an auditor can open.
- **Literature reviews:** healthcare, legal and academic teams collecting cited evidence fast.
- **Internal knowledge:** swap `web_search` for a search over your company's docs and it becomes a cited RAG researcher.

---

## 4. Milestones

- [x] **M1** Researcher agent with web search, structured cited output, Langfuse tracing, offline tests
- [ ] **M2** LangGraph graph: researcher → critic → writer, with a retry loop and a "not enough evidence" exit
- [ ] **M3** Golden set (~15 questions), code checks (citation faithfulness) + LLM judge
- [ ] **M4** CI eval gate, demo GIF, write-up
