# Failure Analysis Summary

- Total rows analyzed: **30**
- Pass rate (bucket or score>= 4.0): **3.3%**
- Bucket column used: **fix_bucket**

## Bucket Counts

| bucket    |   count |   pct |
|:----------|--------:|------:|
| format    |      18 |  60   |
| guardrail |      11 |  36.7 |
| PASS      |       1 |   3.3 |

## Top 3 Failure Buckets + Patterns


### format — 18 cases (60.0%)

**Common keywords in notes:** doc(18), steps(16), needs(14), answer(13), mostly(13), excerpts(13), synthesized(13), appropriate(3), refusal(3), limitation(3), add(3), based(3)
- Saved **5 examples** to: `examples_format.csv`

### guardrail — 11 cases (36.7%)

**Common keywords in notes:** answerable(6), guardrail(6), blocked(6), system(5), refuse(3), sensitive(3), out(3), scope(3), allowed(3), answering(3), idk(2), case(2)
- Saved **5 examples** to: `examples_guardrail.csv`