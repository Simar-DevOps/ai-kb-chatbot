# Implementation Playbook v1 — Support KB Chatbot Pilot

## 1) Purpose
This playbook explains how to run a safe internal pilot of the Support KB Chatbot, onboard users, communicate expectations, collect feedback, and iterate weekly. The goal is to validate usefulness and trust (citations + correct “I don’t know”) before any broader rollout.

## 2) Pilot scope (What we’re launching)
What users get:
- Ask questions in a simple chat UI
- Answers grounded in internal KB markdown docs only
- Citations shown for every supported answer
- “I don’t know” + escalation guidance when the KB does not support the request
- Feedback buttons (👍/👎) + optional comment
- Logs captured for analytics (top questions, feedback rate)

What users should NOT expect:
- Actions taken on their behalf (no account changes, approvals, resets)
- Coverage beyond what exists in the KB docs
- Perfect accuracy if the KB content is incomplete

## 3) Target audience + duration
Audience:
- 10–20 internal users from Support/Ops and Implementation/CS (internal)

Duration:
- 1–2 weeks initial pilot

Success definition (pilot):
- Helpful rate ≥ 70% (👍 / total feedback)
- Correct refusals ≥ 90% on unsupported questions
- Consistent citation coverage (answers show sources)

## 4) Roles & responsibilities
Owner (you):
- Maintain KB docs, review logs weekly, triage issues, ship improvements

Pilot users:
- Ask real questions, give 👍/👎 feedback, add a short comment when 👎

Optional reviewer (buddy/manager):
- Spot-check citations, validate refusal behavior, sanity-check weekly metrics

## 5) Onboarding checklist (15 minutes)
1. Share quick-start message (see Comms below).
2. Confirm user can run/open the app (local pilot) and load the KB.
3. Explain the trust model:
   - If it cites sources, treat it as “supported by KB”
   - If it says “I don’t know,” that is expected and correct when KB is missing
4. Ask each pilot user to submit at least:
   - 3 normal questions
   - 2 edge questions (tricky / unclear)
   - 1 question that SHOULD trigger “I don’t know”
5. Confirm feedback logging is working (👍/👎 produces a row in the feedback file).

## 6) Communications plan (copy/paste messages)

### 6.1 Launch message (send to pilot group)
Subject: Internal Pilot — Support KB Chatbot (source-backed answers)

Hi team — we’re piloting a small internal Support KB Chatbot to answer repeat questions faster.

How it works:
- It answers ONLY from our KB docs and shows citations.
- If the KB doesn’t support an answer, it will say “I don’t know” and suggest escalation.
- Please use 👍/👎 after answers (add a short note when 👎 so we can fix the KB).

What to try:
- Common “how do I…?” questions
- Troubleshooting questions you see repeatedly
- At least one question you think the KB does NOT cover (to test refusals)

Pilot window: 1–2 weeks.
Goal: improve speed + consistency, and identify KB gaps.

Thanks!

### 6.2 Reminder message (mid-pilot)
Quick reminder: if you use the chatbot this week, please tap 👍/👎 after answers. A short note on 👎 helps us fix the KB quickly.

### 6.3 Closeout message (end of pilot)
Thanks for participating in the pilot. We’re reviewing top questions, helpful rate, and “unanswered topics” to update the KB and improve the tool. If you have any last feedback, reply with the most common questions you wish it handled better.

## 7) Weekly ops cadence (repeat every week)
1. Pull metrics from logs:
   - Top 10 questions
   - Helpful rate (👍 vs total feedback)
   - % “I don’t know” responses
   - Top “unanswered topics” (questions that got 👎 or IDK)
2. Bucket issues (keep it simple):
   - KB missing content
   - Retrieval didn’t fetch the right chunk
   - Answer formatting unclear / too long
   - Wrong
