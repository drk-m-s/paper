---
name: research-skill
description: "Deep research skill for systematically surveying academic papers on a technical topic. Use when asked to: survey papers, find related work, do a literature review, research a topic, find all papers about X, compile papers list, summarize research trends. Searches arXiv, Google Scholar, conference proceedings (ICLR, NeurIPS, ICML, OSDI, SysML, SOSP, DAC, ISCA, ASP-DAC, EuroSys), GitHub, and social media. Extracts metadata, categorizes by methodology, ranks by importance, and outputs structured markdown reports and summary lists."
argument-hint: "research topic or goal (e.g. 'MoE offloading techniques since 2024')"
---

# Research Skill

## Purpose

Conduct a systematic, exhaustive literature survey on a technical topic. Outputs:
1. A per-paper detail report (metadata + analysis + pros/cons)
2. A categorized trend summary
3. A comprehensive flat paper list ranked by importance

## When to Use
- "Find all papers about X"
- "Do a literature review on X"
- "Survey related work for X"
- "Compile papers list about X"
- "Summarize research trends in X"

---

## Procedure

### Phase 1 — Scope Definition

Before searching, load the research context file from the workspace:
- Read [background.md](../background.md) to understand the project goal, hardware constraints, performance requirements, and design principles
- Extract: target use case, key bottlenecks, relevant model families, and implementation constraints
- This context will drive importance ranking and the pros/cons analysis

### Phase 2 — Paper Discovery

Search exhaustively across all available sources. Cast a **wide net first**, then filter. Do not stop after the first few results.

**Primary academic sources (search these first):**
- arXiv (arxiv.org) — preprints; search by keyword + date filter ≥ 2024
- Semantic Scholar — for citation counts and related papers
- Google Scholar — for conference papers and broader coverage
- ACM Digital Library — for SOSP, DAC, ASPLOS, EuroSys, ISCA, ASP-DAC
- OpenReview — for ICLR, NeurIPS, ICML submissions and reviews

**Conference proceedings to sweep (2024, 2025, 2026):**
| Conference | Domain |
|---|---|
| ICLR | ML Systems |
| NeurIPS | ML Systems |
| ICML | ML Algorithms |
| OSDI / SOSP / EuroSys | Systems |
| MLSys / SysML | ML-Systems Co-design |
| ISCA / HPCA / MICRO | Computer Architecture |
| DAC / ASP-DAC | Design Automation |
| AAAI / IJCAI | AI |

**Secondary sources:**
- GitHub — search for repos with keywords; check README and linked papers
- HuggingFace Papers (huggingface.co/papers) — trending ML papers
- Twitter/X, Reddit r/MachineLearning — community discussions linking papers
- Paper-aggregator sites (paperswithcode.com, themoonlight.io)

**Search keywords to use (combine and vary):**
```
MoE offloading, Mixture of Experts inference, expert offloading,
expert caching, expert prefetching, sparse model inference,
MoE serving, hybrid CPU-GPU MoE, speculative decoding MoE,
MoE quantization, MoE compression, edge MoE inference,
resource-constrained LLM inference, MoE scheduling
```

**Minimum coverage target:** ≥ 30 unique papers before proceeding.

### Phase 3 — Per-Paper Extraction

For every discovered paper, extract the following fields:

| Field | Instructions |
|---|---|
| **Full Title** | Exact title as published |
| **Authors** | First author + et al. + institution |
| **Venue** | Conference/journal/arXiv; include year |
| **Date** | Exact upload/publication date (YYYY-MM-DD if available) |
| **Citations** | Approximate count from Semantic Scholar or Google Scholar |
| **Core Technique** | 1–3 sentence precise technical description |
| **Hardware Assumed** | GPU/CPU/edge specs the method targets |
| **Models Evaluated** | Specific model names and sizes |
| **Benchmarks** | Task names and metrics reported |
| **Key Results** | Headline numbers (speedup, memory reduction, accuracy) |
| **URL** | DOI, arXiv ID, or direct link |

### Phase 4 — Categorization

Group papers into methodology categories. Suggested categories (adjust if the topic differs):

1. **Software-Based Offloading & Cache Management** — LRU/sparsity-aware caches, peer-GPU caching
2. **Predictive Prefetching & Scheduling** — pre-gating, learnable predictors, cross-layer scheduling
3. **Hybrid CPU-GPU Compute** — CPU-side expert execution, AMX/AVX kernels, NDP architectures
4. **Compression, Quantization & Merging** — mixed-precision, expert pruning, subspace merging
5. **High-Throughput Serving & Speculative Decoding** — speculative decoding, semantic parallelism, module batching

If a paper spans categories, assign to the **primary** contribution category. Note secondary categories in the summary.

### Phase 5 — Importance Ranking

Rank each paper **1–5** (5 = most important, 1 = least important) using the following weighted criteria:

| Criterion | Weight | Guidance |
|---|---|---|
| **Relevance to project goal** | 40% | How directly does it address constraints in `background.md`? (NPU, llama.cpp, no model modification, PCIe 4.0, 10 GB HBM) |
| **Technical novelty** | 30% | Does it introduce a genuinely new mechanism vs. incremental tuning? |
| **Venue prestige** | 20% | SOSP/OSDI/ICLR/NeurIPS = top tier; arXiv-only = lower; workshop = lowest |
| **Citation impact** | 10% | >100 citations = significant; >30 = notable; <5 = emerging |

**Tie-breaking:** prefer papers with public code (GitHub), reproducible results, or explicit llama.cpp integration.

### Phase 6 — Pros/Cons Analysis (per paper)

For each paper, write 2–4 bullet pros and 2–4 bullet cons **specifically with respect to the project goal** from `background.md`:

- **Pros**: What aspects of the approach are beneficial for the NPU-based edge offloading engine? (e.g., "works without model modification", "targets PCIe-constrained scenarios", "llama.cpp compatible")
- **Cons**: What makes it hard to adopt? (e.g., "requires retraining", "assumes multi-GPU", "CPU-only, not NPU-aware", "batch-size > 1 assumed")

### Phase 7 — Output: Per-Paper Detail Report

Write a markdown file named `deep-research-report-<date>.md` (e.g. `deep-research-report-2026-05.md`) in the `papers/` folder.

Structure:
```markdown
# Research Report: <Topic> (<Date Range>)

## Executive Summary
<2–3 paragraph overview of the landscape, dominant trends, and key gaps>

## Paper Details

### <Paper Name>
**Authors:** ... | **Venue:** ... | **Date:** ... | **Citations:** ... | **Importance:** ⭐⭐⭐⭐⭐ (5/5)

**Core Technique:** ...

**Hardware:** ... | **Models:** ... | **Benchmarks:** ...

**Key Results:** ...

**Pros (for project goal):**
- ...

**Cons (for project goal):**
- ...

**URL:** [link](url)

---
```

### Phase 8 — Output: Trend Summary

Write a markdown section (or separate file `deep-research-trends-<date>.md`) that:
- Groups papers into the 5 methodology categories
- For each category, writes 1–2 paragraphs summarizing the dominant approach, key papers, and evolution of the technique
- Ends with a **"Gaps and Future Directions"** section noting what is not yet well-solved for the project's specific constraints

### Phase 9 — Output: Comprehensive Paper List

Write a markdown file named `deep-research-list-papers-<date>.md` in the `papers/` folder.

Structure: one table per category, sorted by importance (descending), with columns:

| Paper / System Name | Date | Importance | Key Contribution |
|---|---|---|---|

After all tables, include an overall flat list of **all papers** sorted by date.

---

## Quality Checks

Before finishing, verify:
- [ ] At least 40 papers discovered (more is better)
- [ ] Every paper has all Phase 3 fields filled (no blanks)
- [ ] Importance rankings are calibrated: not all 5s; roughly bell-shaped distribution
- [ ] Pros/cons are specific to `background.md` constraints, not generic
- [ ] URLs are real and correctly formatted
- [ ] Three output files have been written: detail report, trend summary (can be section in report), and paper list
- [ ] Paper list tables are sorted by importance descending within each category

## Output Files Summary

| File | Location | Contents |
|---|---|---|
| `deep-research-report-YYYY-MM.md` | `papers/` | Per-paper details with pros/cons |
| `deep-research-list-papers-YYYY-MM.md` | `papers/` | Categorized tables + full flat list |

Both files should be self-contained and readable without cross-referencing each other.