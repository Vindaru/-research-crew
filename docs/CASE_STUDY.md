# Case study: a research agent that has to prove its citations

## The problem

"Deep research" agents (the feature OpenAI, Google, Perplexity and Anthropic all ship) turn a question into a
cited report in minutes. Their most damaging failure is a **confident report with a citation that doesn't say
what the report claims**. A human reader trusts the footnote and never checks it.

So the question for this project was not "can I build a research agent?" but
**"can I build one whose citations are verifiably real, and catch it automatically when they stop being real?"**

## Design

```
question ─► Researcher ─► Critic ─┬─► Writer ─► cited report
              ▲                   │
              └── feedback ───────┤ (rejected findings, max 3 attempts)
                                  └─► give_up ─► "not enough evidence"
```

| Decision | Why |
|---|---|
| **Separate agents for finding, checking, writing** | One model that researches and writes will grade its own homework. Splitting roles lets the critic say no. |
| **Structured output** (`Finding` = claim + URL + exact quote) | A citation you can't check mechanically isn't a citation. Pydantic rejects malformed output at the boundary. |
| **The search tool records every page it returns** | The critic checks quotes against what the search *actually* returned, not against the researcher's word. |
| **Critic = code check first, LLM second** | "Is this quote on that page?" is a string match: free, exact, never flaky. Only "does the quote *support* the claim?" needs a model. |
| **Writer never sees the web or rejected findings** | It has nothing to hallucinate from. |
| **An explicit give-up path** | "Not enough evidence" is a correct answer. Systems without that exit invent one. |
| **LangGraph** | The retry loop is a backwards edge in a graph with shared state, easy to test with fake agents. |

## How it's tested (the part most agent demos skip)

| Layer | What | Runs |
|---|---|---|
| Unit tests | Critic catches invented quotes and invented pages; tool output format | Every push, free |
| Behaviour tests | Deliberately bad researchers: one lies then fixes it, one always lies, one is half wrong. Checks retry, give-up, and that a rejected claim never reaches the report | Every push, free |
| Grader tests | Each grader is fed an output with one known defect and must catch it | Every push, free |
| **Sabotage test** | Replace the critic with a rubber stamp; the independent faithfulness grader and the gate must fail | Every push, free |
| Golden set, offline | 13 cases × 5 code checks through a fake crew, with a quality gate | Every push (`eval-gate` check) |
| Golden set, live | Real Claude + Tavily + LLM judge (DeepEval GEval, Claude Sonnet), traced in Langfuse | On demand + weekly |

**Golden set categories:** happy path, multi-source, unanswerable, false premise, prompt injection, vague.
Web answers change, so cases store *properties* of a good answer (status, minimum citations, must / must-not
phrases) instead of exact text.

**Headline metric: citation faithfulness** = share of cited findings whose quote appears word for word in the
page the search returned. It is re-measured by the eval independently of the critic, so weakening the critic
can't hide a regression. The gate requires **100%**.

## Results

| Run | Cases passed | Citation faithfulness | Notes |
|---|---|---|---|
| Offline (fake crew) | 12 / 13 | 100% | The one failure is intentional: the fake researcher only reads first sentences |
| Live (Claude Haiku agents, Sonnet judge) | _fill in after first live run_ | _fill in_ | _cost per run, slowest node from Langfuse_ |

## Trade-offs and what I'd do next

- **Exact-quote matching is strict.** Paraphrased-but-true findings get rejected and cost a retry. Next: allow a
  fuzzy match threshold and measure how many retries it saves vs how many bad quotes it lets through.
- **Search snippets, not full pages.** Quotes are checked against Tavily's snippet. Next: fetch full pages.
- **One critic model.** Next: compare Haiku vs Sonnet as critic on the golden set (Langfuse experiments make
  this a one-flag change) and pick on evidence: faithfulness, cost and latency together.
- **Grow the golden set from production.** Every bad trace in Langfuse becomes a new case.

## Stack

Python 3.12 · LangGraph · LangChain `create_agent` · Claude (Haiku 5.5 agents, Sonnet 5.5 judge) · Tavily ·
Pydantic · Langfuse (traces, datasets, experiments) · DeepEval GEval · pytest · GitHub Actions
