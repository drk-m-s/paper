---
name: moe-paper-analysis
description: "Deep-dive analysis of a single MoE offloading paper. Use when asked to: analyze a paper about MoE offloading, understand a MoE inference paper, review an expert offloading framework, evaluate a paper's relevance for edge MoE inference, produce a structured paper summary. Reads PDF from local folder, searches online for metadata (authors, venue, citations), evaluates across 10 dimensions (MoE inference vs training, edge vs cloud, hardware, offload storage, models used, model modification, offline profiling, open-source code, performance, techniques), and outputs a structured markdown summary in /paper_analysis/."
argument-hint: "path to the PDF paper file (e.g. 'papers/cachegen-arxiv-2410.pdf')"
---

# MoE Paper Analysis

## Purpose

Perform a thorough, structured analysis of a single academic paper about Mixture-of-Experts (MoE) offloading for inference. The skill extracts paper content from PDF, gathers metadata from online sources, evaluates the paper against 10 specific dimensions relevant to edge MoE inference, and produces a standardized markdown summary.

## When to Use

- "Analyze this MoE paper for me"
- "Deep-dive into this paper about MoE offloading"
- "Is this paper relevant for edge MoE inference?"
- "Summarize this paper's approach to expert offloading"
- "Evaluate this paper against our project criteria"

## Scope

This skill focuses specifically on **MoE model inference offloading** for **edge/batch-size-1 scenarios**. It is NOT designed for:
- MoE training frameworks
- Cloud-scale multi-GPU serving systems (unless you explicitly ask to evaluate those)

---
## Procedure

### Step 1 — Locate and Read the PDF

If the user does not provide a PDF path, ask them which paper to analyze.

1. Determine the PDF file path (absolute or relative to workspace root)
2. If the PDF is not already in the workspace, ask the user to place it in a local folder first
3. Use Python with `PyPDF2` or `pdfplumber` to extract the full text:

```python
import pdfplumber
with pdfplumber.open("<pdf_path>") as pdf:
    full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
```

If `pdfplumber` is not installed, install it via pip first.

4. Read and store the full extracted text — you will reference it throughout the analysis.

### Step 2 — Extract Paper Metadata

Extract from the PDF text itself first, then verify/correct with online searches.

**From PDF text:**
- Paper title (usually on first page)
- Author names and affiliations
- Publication venue and date
- Abstract

**Online search (verify and supplement):**
- Search Semantic Scholar (`api.semanticscholar.org`) by paper title for citation count, official venue, and publication date
- Search Google Scholar for citation count as cross-reference
- Search arXiv if the paper is a preprint

Use this Python snippet for Semantic Scholar:
```python
import requests
# Search by title
resp = requests.get(
    "https://api.semanticscholar.org/graph/v1/paper/search",
    params={"query": "<paper_title>", "limit": 3}
)
data = resp.json()
# Then fetch details with paper ID for citationCount, venue, year, authors
```

**Collect:**
| Field | Source |
|-------|--------|
| **Paper Name** | PDF text + verified online |
| **Author's Organization(s)** | PDF affiliations + Semantic Scholar |
| **Publication Date** | PDF + Semantic Scholar/arXiv |
| **Citation Count** | Semantic Scholar + Google Scholar |
| **Venue** | Conference/journal name |
| **arXiv ID** | If applicable |

### Step 3 — 10-Dimension Analysis

For each dimension, read the relevant sections of the PDF text carefully. Cite specific sentences or paragraphs. If the paper does not address a dimension, mark it as **"Not addressed"** or **"N/A"** with a brief note.

#### Dimension 1: MoE Inference vs. Training
- Is this framework for **MoE model inference** (serving/deployment)?
- Or is it for **MoE training** (distributed training, load balancing during training)?
- **Our interest:** Inference only. If it's a training framework, flag it and explain what it does.

#### Dimension 2: Edge vs. Cloud
- Is the target scenario **edge** (batch-size = 1, single user, resource-constrained device)?
- Or is it **cloud** (batch-size > 1, high-throughput serving, multi-tenant)?
- Check: What batch size does the paper use in experiments? What hardware constraints does it mention?
- **Our interest:** Edge scenarios with batch-size = 1.

#### Dimension 3: Hardware Setup
- **Compute hardware:** GPU only / GPU + CPU / CPU only / NPU / other?
- **Specific hardware used in experiments:** e.g., "NVIDIA A6000 48GB", "Intel Xeon + RTX 4090", "Apple M2 Ultra"
- List the exact GPU/CPU models and their memory specifications.

#### Dimension 4: Offloaded Weight Storage
- Where are offloaded (cold) expert weights stored?
- **Options:** CPU DRAM (main memory) / CPU SSD (NVMe/SATA) / remote storage / other?
- What is the storage capacity and bandwidth assumed?

#### Dimension 5: MoE Models Used
- Which specific MoE models are evaluated?
  - e.g., Mixtral 8x7B, DeepSeek-V2, Qwen-MoE, Switch Transformer, ST-MoE, etc.
- What are the model sizes (total parameters, active parameters, number of experts)?

#### Dimension 6: Model Modification / Retraining Required?
- Does the framework require **adding components** to the model architecture?
- Does it require **retraining** or **fine-tuning** the model?
- Or is it **plug-and-play** with off-the-shelf model weights?
- **Our interest:** Frameworks that work without model modification.

#### Dimension 7: Offline Profiling Required?
- Does the framework require an **offline profiling phase** to identify hot vs. cold experts?
- If yes, what does it profile? (activation patterns, routing statistics, expert affinity?)
- How long does profiling take and on what dataset?
- **Our interest:** Understanding the profiling cost and whether it's practical.

#### Dimension 8: Open-Source Code
- Has the paper released implementation code?
- Search GitHub for the paper name or authors' repositories.
- Provide the GitHub URL if found. Note the license if available.

#### Dimension 9: Performance Results
- **Speedup vs. baselines:** What is the claimed speedup? Over what baseline?
- **Prefill throughput:** Tokens/second during prefill phase.
- **Decode throughput:** Tokens/second during decode/autoregressive generation.
- **Time-to-first-token (TTFT):** Latency before first token is generated.
- **Time-per-output-token (TPOT):** Average latency per generated token.
- **Memory usage:** GPU memory, CPU memory, and any offloaded storage.
- **End-to-end latency:** For a given prompt/output length.

#### Dimension 10: Techniques & Innovation
- What is the **core technical innovation**?
- Describe the mechanism in 2-4 sentences.
- What existing techniques does it build upon or combine?
- Is it a novel algorithm, a systems optimization, or a combination?

### Step 4 — Relevance Assessment

Based on the 10-dimension analysis, provide a concise relevance assessment:

- **Overall relevance to edge MoE inference:** ⭐⭐⭐⭐⭐ (1-5 scale)
- **Key strengths:** 2-4 bullet points
- **Key limitations / gaps:** 2-4 bullet points
- **Actionable for our project?** Yes / Partially / No — with reasoning

### Step 5 — Output Summary

Use the template at [assets/summary-template.md](./assets/summary-template.md) to produce the final output.

1. Create the output directory `paper_analysis/` if it doesn't already exist
2. Write the summary to `paper_analysis/<paper-name-slug>.md` (use lowercase, hyphens for spaces)
3. The summary must include all 10 dimensions and the relevance assessment

---
## Resources

- **Output template:** [assets/summary-template.md](./assets/summary-template.md) — Use this exact structure for the final markdown output.
- **Workspace background:** [background.md](../../background.md) — Project context for relevance assessment.
