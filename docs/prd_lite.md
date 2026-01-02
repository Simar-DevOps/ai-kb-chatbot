# PRD-lite — Support KB Chatbot (v2 rollout-ready)

## 1) Summary
We are building an internal Support KB Chatbot that answers repeat operational/product questions using approved KB markdown docs. The chatbot must show citations for every answer and must refuse (“I don’t know”) when the KB does not support the answer. v2 adds feedback, analytics, and lightweight admin controls so the tool can be piloted and iterated safely.

## 2) Goals (What success looks like)
- Reduce time spent searching for internal answers by providing fast, consistent, source-backed responses.
- Improve trust by always showing citations and refusing unsupported questions.
- Create an iteration loop using feedback + logs so KB gaps get fixed quickly.
- Produce rollout-ready artifacts (docs + pilot plan + launch criteria) suitable for AI Product Ops / Solutions work.

## 3) Users
Primary:
- Support / Ops: quick troubleshooting and process answers with sources
- Implementation / CS (internal): consistent guidance with links to exact sections

Secondary:
- New hires: self-serve “how do I…?” answers with citations

## 4) Scope (In-scope for v2)
Product behavior:
- Chat UI with conversation history + reset
- Retrieval over local markdown KB docs (top-k)
- Answer grounded in retrieved chunks only
- Always show sources/citations for answers
- Strong “I don’t know” rule + escalation guidance when unsupported

Ops / iteration:
- Feedback capture (👍/👎 + optional comment) stored to CSV/JSON with metadata
- Analytics v1: log questions, timestamps, session id, top queries, feedback rate
- Admin controls in sidebar (persisted): model toggle, temperature, top-k, max tokens, LLM on/off
- Basic error handling (missing docs, empty retrieval results, file read issues)

## 5) Non-goals (Out of scope for this phase)
- External customer-facing deployment
- Taking actions (account changes, refunds, approvals, password resets)
- Live integrations (Jira/ServiceNow/Confluence/Slack search) — future phase only
- Personal data lookup or storing PII
- “Perfect answers” to anything outside the KB

## 6) Requirements
Functional requirements:
- R1: The system must retrieve relevant KB chunks for each question (top-k configurable).
- R2: The system must generate answers using only retrieved text; no free-form guessing.
- R3: The UI must display citations (document + section/chunk) with each answer.
- R4: If retrieval confidence/support is insufficient, the bot must respond with “I don’t know” + escalation steps.
- R5: The user must be able to provide 👍/👎 feedback and an optional note.
- R6: The system must log question/answer metadata for analytics (session, time, feedback, top queries).
- R7: Admin settings must persist across reruns (session/state).

Non-functional requirements:
- N1: Fast: target median response under 30 seconds locally
- N2: Reliability: app should not crash on missing/empty files; show clear message instead
- N3: Safety: avoid logging sensitive content; keep logs local for pilot
- N4: Transparency: citations are always visible; refusal is explicit when unsupported

## 7) KPIs (Pilot metrics)
Adoption:
- Weekly active users: 10+
- Questions per week: 100+

Quality:
- Helpful rate (👍 / total feedback): 70%+
- Unsupported-answer safety: 90%+ of unsupported questions trigger “I don’t know”
- Citation coverage: 95%+ of answers show at least 1 source

Efficiency:
- Median time-to-answer: under 30 seconds
- Self-serve resolution estimate: 50%+ (measured via feedback + drop in repeat questions)

Ops:
- Weekly report of top 10 queries + top 10 “unanswered topics”

## 8) Risks + Mitigations
Hallucinations / wrong guidance:
- Mitigation: strict grounding to retrieved chunks, always cite, refuse when unsupported

Outdated/incomplete KB:
- Mitigation: weekly review of negative feedback + unanswered topics; monthly refresh cadence

Sensitive info exposure:
- Mitigation: keep KB docs scoped/sanitized; no PII in logs; pilot access control

Low trust/adoption:
- Mitigation: seed high-value FAQs, example questions, and show citations by default

## 9) Launch criteria (Definition of “ready to pilot”)
- Citations visible for answers and match retrieved content
- “I don’t know” triggers reliably when retrieval returns weak/no support
- Feedback logging works (file created, rows appended, metadata present)
- Analytics log captures questions and can produce top queries
- Admin settings persist and do not break the app
- Basic errors are handled with readable messages (no stack traces in UI)

## 10) Pilot plan (Simple rollout)
- Audience: 10–20 internal users (Support/Ops/Implementation)
- Duration: 1–2 weeks
- Process:
  - Share quick-start: what to ask + expectations (“answers come from KB only”)
  - Review logs weekly: top queries, 👍 rate, unanswered topics
  - Update KB docs for top gaps, then re-test those questions

## 11) Decision summary (So we don’t drift)
- We prioritize trust (citations + refusal) over broader coverage.
- We ship v2 as an internal pilot tool first, not a public product.
- We iterate using feedback + analytics rather than adding integrations immediately.
