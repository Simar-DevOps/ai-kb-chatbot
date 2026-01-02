# Use Case Brief — Support KB Chatbot (v1)

## Problem
Internal teams lose time searching for answers across scattered docs, Slack messages, and tribal knowledge. This causes repeated questions, inconsistent guidance, slower onboarding, and avoidable tickets. We need a fast, reliable way to answer common support questions using approved knowledge base content with citations.

## Users
Primary users:
- Support / Operations staff — need quick, consistent answers and steps they can trust
- Implementation / Customer Success (internal) — need accurate product/process guidance with sources to reduce back-and-forth

Secondary users:
- New hires — need self-serve answers to common “how do I…?” questions with links to the exact policy/procedure

Out of scope (v1):
- External customer-facing chatbot
- Anything requiring account actions, approvals, payments, or personal data lookup

## Workflow
1. User opens the chatbot and types a question (or clicks an example question).
2. The app retrieves the top relevant KB chunks from local markdown docs (top-k).
3. The model generates an answer grounded only in retrieved chunks.
4. The app displays the answer plus cited sources (doc + section).
5. If the docs do not support an answer, the bot says “I don’t know” and provides escalation guidance.
6. User submits 👍/👎 feedback (optional comment), and the app logs the question + outcome for weekly review.

## KPIs
Adoption:
- Weekly active users (internal pilot): 10+
- Questions answered per week: 100+

Quality:
- Helpful rate (👍 / total feedback): 70%+
- Unsupported-answer safety: 90%+ of unsupported questions correctly trigger “I don’t know” (no guessing)
- Citation coverage: 95%+ of answers show at least 1 source

Efficiency:
- Median time to get an answer: under 30 seconds
- Self-serve resolution (no escalation after answer): 50%+ (estimated from feedback + repeat question drop)

Ops / Improvement:
- Top 10 question themes tracked weekly
- Top 10 “unanswered / weak-doc” topics tracked weekly for KB doc updates

## Risks + Mitigations
Risk: Hallucinations or incorrect guidance  
Mitigation: answer only from retrieved sources, always show citations, strict “I don’t know” fallback + escalation message

Risk: Outdated or incomplete KB content  
Mitigation: weekly review of negative feedback + top unanswered topics; monthly doc refresh cadence; add “last updated” notes in KB docs

Risk: Sensitive information or secrets included in docs/logs  
Mitigation: keep docs scoped to safe content; sanitize docs; avoid logging PII; store logs locally for pilot and review before wider rollout

Risk: Low trust or low adoption  
Mitigation: seed high-value FAQs; add example questions; share quick-start guide; measure helpful rate and iterate based on feedback
