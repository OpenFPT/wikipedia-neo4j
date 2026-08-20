const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, PageBreak,
  TableOfContents
} = require('docx');
const fs = require('fs');

const PAGE_W = 11906;
const PAGE_H = 16838;
const MARGIN = 1134;
const CONTENT_W = PAGE_W - MARGIN * 2;

const BLUE = "2E5DA8";
const HEADER_BG = "2E5DA8";
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 36, color: BLUE, font: "Arial" })],
    spacing: { before: 300, after: 120 },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 28, color: BLUE, font: "Arial" })],
    spacing: { before: 240, after: 80 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial" })],
    spacing: { before: 160, after: 60 },
  });
}
function para(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 24, font: "Times New Roman" })],
    spacing: { before: 60, after: 60 },
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, size: 24, font: "Times New Roman" })],
    spacing: { before: 40, after: 40 },
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function makeTable(headers, rows, widths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((text, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 22, font: "Arial" })],
      })],
    })),
  });
  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((text, ci) => new TableCell({
      borders, width: { size: widths[ci], type: WidthType.DXA },
      shading: { fill: ri % 2 === 1 ? "F5F8FC" : "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 120, right: 120 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        children: [new TextRun({ text: text || "", size: 22, font: "Times New Roman" })],
      })],
    })),
  }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...dataRows],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({ text: "Vietnamese GraphRAG — Capstone Review 1", size: 20, color: "666666", font: "Arial" }),
            new TextRun({ text: "\tFPT University  |  Week 4  |  2026", size: 20, color: "666666", font: "Arial" }),
          ],
          tabStops: [{ type: "right", position: CONTENT_W }],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } },
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          children: [
            new TextRun({ text: "ViWiki-GraphRAG  |  QE180038 · QE180005 · QE180110 · QE180018", size: 18, color: "888888", font: "Arial" }),
            new TextRun({ text: "\tPage ", size: 18, color: "888888", font: "Arial" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888", font: "Arial" }),
          ],
          tabStops: [{ type: "right", position: CONTENT_W }],
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } },
        })],
      }),
    },
    children: [
      // COVER
      new Paragraph({
        children: [new TextRun({ text: "CAPSTONE REVIEW 1 REPORT", bold: true, size: 48, font: "Arial", color: BLUE })],
        alignment: AlignmentType.CENTER, spacing: { before: 720, after: 240 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Vietnamese GraphRAG", bold: true, size: 40, font: "Arial" })],
        alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Multi-hop Question Answering over Vietnamese Wikipedia", size: 28, font: "Arial", color: "555555" })],
        alignment: AlignmentType.CENTER, spacing: { before: 0, after: 480 },
      }),
      makeTable(["Role", "Member", "Student ID"], [
        ["KG & NER", "Huynh Quoc Trung", "QE180038"],
        ["Retrieval & Eval", "Nhu Quang Anh", "QE180005"],
        ["Agent & LLM", "Pham Ngo Dinh Khoi", "QE180110"],
        ["API & Deploy", "Truong Dinh Thien", "QE180018"],
      ], [2500, 4500, 2638]),
      new Paragraph({ children: [], spacing: { before: 200 } }),
      makeTable(["", ""], [
        ["University", "FPT University — SE+AI Capstone"],
        ["Semester", "Summer 2026"],
        ["Week", "4 (June 2026)"],
        ["Supervisor", ""],
      ], [2500, 7138]),
      pageBreak(),

      // TOC
      h1("Table of Contents"),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
      pageBreak(),

      // 1
      h1("1. Problem Statement & Objectives"),
      h2("1.1 Background"),
      para("Vietnamese students and researchers frequently need to answer complex questions spanning multiple Wikipedia articles. Standard RAG systems only retrieve the nearest chunk and lose multi-hop links. Vanilla LLMs hallucinate Vietnamese facts and provide no verifiable citations."),
      h2("1.2 Objectives"),
      para("Build a fully local Vietnamese KGQA system that answers multi-hop questions with validated citations, without sending data to external services."),
      h2("1.3 Scope"),
      makeTable(["In Scope", "Out of Scope"], [
        ["Local single-process pipeline", "Web search / external retrieval"],
        ["Vietnamese Wikipedia (~590K articles)", "Gated datasets (ViMQA)"],
        ["WRRF hybrid retrieval + reranking", "Multi-agent across multiple LLMs"],
        ["Dataset generation + QC pipeline", ""],
      ], [4819, 4819]),

      // 2
      h1("2. Practical Applicability & Users"),
      makeTable(["User", "Pain Point", "Gain Point"], [
        ["University students", "Must read 5-10 articles to synthesize an answer", "Immediate answer with verified citations"],
        ["Researchers", "No reliable Vietnamese QA tool available", "Accurate academic lookup tool"],
        ["AI developers", "Lack of open-source Vietnamese RAG pipeline", "Ready-to-integrate pipeline"],
      ], [2500, 3500, 3638]),
      h3("Stakeholders"),
      bullet("End users: FPT University students (real-world testing)"),
      bullet("Academic advisors: Supervisor and capstone committee"),
      bullet("Open-source community: ViWiki-MHR dataset published under CC-BY-SA"),

      // 3
      h1("3. Innovation & New Technology"),
      makeTable(["Innovation", "Description"], [
        ["WRRF Hybrid Retrieval", "BM25 + Vector + Graph + Community — no Vietnamese QA system uses all 4 signals"],
        ["Multi-trajectory Agent", "N parallel trajectories + majority voting (arXiv:2506.19967)"],
        ["Local-first sovereignty", "Entire pipeline runs on local 8GB VRAM, no external data transfer"],
        ["ViWiki-MHR Dataset", "First Vietnamese multi-hop dataset from KG walk + 5-layer QC pipeline"],
        ["Typed Vietnamese KG", "Neo4j with 4 entity types, 6 relation types from Vietnamese Wikipedia 2026"],
      ], [3000, 6638]),

      // 4
      h1("4. System Architecture"),
      makeTable(["Layer", "Component", "Description"], [
        ["API", "FastAPI", "Auth, rate-limit, background job management"],
        ["Agent", "agent.py + agent_tools.py", "ReAct loop (max 6 iter), complexity detection, multi-trajectory"],
        ["Retrieval", "retrieve.py + hybrid.py", "WRRF: BM25(0.4)+Vector(0.4)+Graph(0.2)+Community(0.15)"],
        ["Rerank", "reranker.py", "BAAI/bge-reranker-v2-m3, top-20 → top-5"],
        ["KG", "Neo4j", "Page, Chunk, Person, Org, Location, Work nodes"],
        ["LLM", "local_llm.py", "Vi-Qwen2-7B-RAG, 4-bit NF4"],
      ], [1800, 3200, 4638]),
      h3("Graph Schema"),
      bullet("(:Page)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS_*]->(:Entity)"),
      bullet("(:Page)-[:LINKS_TO]->(:Page)"),
      bullet("Indexes: fulltext(chunk.text), vector(chunk.embedding 1024-dim), btree(entity.name)"),

      // 5
      h1("5. Requirements Specification (SRS)"),
      h2("5.1 Functional Requirements (MoSCoW)"),
      makeTable(["ID", "Requirement", "Priority", "Verifiable By"], [
        ["FR-01", "Ingest Vietnamese Wikipedia articles into Neo4j", "Must", "Unit test ingestion pipeline"],
        ["FR-02", "Answer single-hop factual questions", "Must", "EM/F1 >= 30% on ViQuAD2"],
        ["FR-03", "Answer multi-hop questions with citations", "Must", "Hit Rate >= 70% on ViWiki-MHR"],
        ["FR-04", "Bulk ingest from HuggingFace dataset", "Must", "Load test: 10K articles"],
        ["FR-05", "Generate MHQA evaluation dataset", "Must", "QC pipeline pass rate >= 80%"],
        ["FR-06", "ReAct agent with 6 tools, max 6 iterations", "Must", "Integration test 50 queries"],
        ["FR-07", "Gradio demo interface", "Should", "Demo works with 3 test cases"],
        ["FR-08", "Fine-tune Text2Cypher (QLoRA)", "Should", "Cypher execution rate >= 95%"],
      ], [1200, 3800, 1400, 3238]),
      h2("5.2 Non-Functional Requirements"),
      makeTable(["ID", "Metric", "Target", "Achieved (Week 4)"], [
        ["NFR-01", "P50 latency (simple)", "< 3s", "51ms retrieval ✓"],
        ["NFR-02", "Context Hit Rate@5", "> 70%", "64.67% (needs improvement)"],
        ["NFR-03", "MRR@5", "> 0.50", "0.4497 (needs improvement)"],
        ["NFR-04", "Test coverage", ">= 75%", "85% ✓"],
        ["NFR-05", "Hardware target", "8GB VRAM", "RTX 3060/4060 ✓"],
      ], [1200, 3000, 2000, 3438]),

      // 6
      h1("6. AI Specification"),
      h2("6.1 Problem Framing"),
      makeTable(["Layer", "Content"], [
        ["Business Problem", "Complex Vietnamese questions require synthesis across multiple Wikipedia articles"],
        ["AI Task", "Multi-hop KGQA: Graph-augmented RAG with typed knowledge graph"],
        ["Formal", "Q → retrieve {P1...Pn} from KG G → generate A + citations"],
        ["Taxonomy", "Simple (1-hop), Bridge (2-hop), Comparison (2+ hop), Fan-out (3+ hop)"],
      ], [2500, 7138]),
      h2("6.2 Dataset"),
      makeTable(["Dataset", "Size", "Source", "Purpose"], [
        ["Vietnamese Wikipedia", "~590K articles", "wikimedia/wikipedia 20231101.vi", "KG construction"],
        ["ViWiki-MHR", "~200 pairs", "KG walk + manual", "Multi-hop evaluation"],
        ["UIT-ViQuAD 2.0", "36,457 pairs", "UIT-NLP (VLSP 2021)", "Benchmark"],
        ["MHQA (generated)", "700-1000 target", "KG walks + Claude API", "Expanded evaluation"],
      ], [2500, 1800, 2800, 2538]),
      h2("6.3 Related Works (3 papers)"),
      h3("[1] Microsoft GraphRAG — Edge et al. (2024), arXiv:2404.16130"),
      para("Leiden community detection → pre-generate community summaries → map-reduce. Win rate 70-80% vs naive RAG."),
      para("Applied: Louvain community signal → WRRF weight=0.15 for thematic query recall."),
      h3("[2] Inference-Scaled GraphRAG — Thompson et al. (2025), arXiv:2506.19967"),
      para("N trajectories + temperature scaling + majority voting. +64.7% vs traditional GraphRAG on hard multi-hop."),
      para("Ap dung: AGENT_N_TRAJECTORIES, majority voting cho complex queries."),
      h3("[3] UIT-ViQuAD 2.0 — Nguyen et al. (2020), arXiv:2009.14725"),
      para("36,457 QA pairs. XLM-R achieves 77.8% EM. Applied: primary benchmark. Achieved 64.67% context hit rate (top-5)."),
      h2("6.4 AI Pipeline"),
      makeTable(["Stage", "Processing Steps"], [
        ["Ingestion", "Wikipedia → NFC normalize → chunk(500/50) → NER(PER/ORG/LOC/WORK) → embed(1024-dim) → Neo4j UNWIND"],
        ["Retrieval", "Q → BM25(0.4)+Vector(0.4)+Graph(0.2)+Community(0.15) → WRRF(k=60) → cross-encoder rerank → top-K"],
        ["Answer (Simple)", "ReAct loop max 6 iter: kg_schema, kg_query, text_search, get_passage, entity_neighborhood, path_search"],
        ["Answer (Complex)", "Decompose → N trajectories (temp scaling) → majority vote → Answer + citations"],
      ], [2500, 7138]),
      h2("6.5 Model Stack"),
      makeTable(["Component", "Model", "Quantization"], [
        ["Local LLM", "AITeamVN/Vi-Qwen2-7B-RAG", "4-bit NF4"],
        ["Embeddings (local)", "GreenNode-Embedding-Large-VN-Mixed-V1", "FP16, 1024-dim"],
        ["Embeddings (API)", "Gemini embedding-001", "N/A"],
        ["Reranker", "BAAI/bge-reranker-v2-m3", "FP16"],
        ["NER", "NlpHUST/ner-vietnamese-electra-base", "FP32"],
      ], [2500, 4500, 2638]),
      h2("6.6 Model Selection Justification"),
      makeTable(["Candidate", "Decision"], [
        ["AITeamVN/Vi-Qwen2-7B-RAG  [SELECTED]", "Purpose-built for Vietnamese RAG. Fine-tuned on Vietnamese QA pairs. Fits 8GB GPU with NF4 quantization. Best extractive QA in pilot tests."],
        ["Vistral-7B-Chat", "Rejected — general Vietnamese chat model, not optimized for extraction tasks."],
        ["Sailor-7B", "Rejected — multilingual SEA model, weaker Vietnamese factuality."],
        ["GPT-4 / Gemini API", "Rejected — rate limits conflict with multi-trajectory voting, no offline evaluation."],
        ["Models under 3B", "Rejected — insufficient reasoning for multi-hop questions."],
      ], [3500, 6138]),
      para("Embedding model: GreenNode-Embedding-Large-VN-Mixed-V1 — only production-quality Vietnamese-specific 1024-dim embedding model. Benchmarked highest on Vietnamese STS tasks among open models."),
      h2("6.7 Evaluation Metrics Justification"),
      makeTable(["Metric", "Target", "Justification"], [
        ["Context Hit Rate", "> 70%", "Best baseline (BM25+Vector, no graph) achieves 66%. +4% proves graph signal value; consistent with GraphRAG literature (Microsoft 2024: +5-15% over dense-only)."],
        ["MRR", "> 0.50", "Baseline MRR ~0.48. Modest but verifiable; aligned with BEIR benchmark norms for domain-specific retrieval."],
        ["Exact Match (EM)", "> 30%", "UIT-ViQuAD 2.0 leaderboard: top XLM-R systems achieve ~35% EM. Target is conservative and achievable."],
        ["Token-F1", "> 50%", "UIT-ViQuAD 2.0 leaderboard: top systems achieve ~65% F1. Accounts for Vietnamese tokenization variance."],
      ], [2000, 1500, 6138]),
      h2("6.8 Security & Legal Compliance"),
      h3("Security Requirements (NFR-10 to NFR-12)"),
      makeTable(["ID", "Requirement", "Verification"], [
        ["NFR-10", "API key authentication — protected endpoints require bearer token", "Request without key returns HTTP 401"],
        ["NFR-11", "Cypher injection prevention — DELETE/MERGE/CREATE/DROP keywords blocklisted", "Inject DELETE in question, verify blocked"],
        ["NFR-12", "Rate limiting — 120 req/min per client (RATE_LIMIT_PER_MINUTE env var)", "Exceed limit, verify HTTP 429"],
      ], [1200, 4500, 3938]),
      h3("Data Privacy & Legal Compliance"),
      makeTable(["Category", "Detail"], [
        ["Source license", "Vietnamese Wikipedia — CC-BY-SA 3.0. Attribution: page_title + page_url stored with every chunk and returned in API responses."],
        ["Personal data", "No PII collected or processed. System ingests publicly available encyclopedia articles only."],
        ["Derived datasets", "Generated QA datasets (ViWiki-MHR, MHQA) inherit CC-BY-SA 3.0 from source content."],
        ["Model licenses", "Vi-Qwen2-7B-RAG, GreenNode-Embedding, bge-reranker-v2-m3 — Apache 2.0 or MIT."],
      ], [2500, 7138]),
      h2("6.9 Team AI Workstream Assignment"),
      makeTable(["Member", "Student ID", "Responsibility", "Key Deliverables"], [
        ["Huynh Quoc Trung", "QE180038", "KG + NER", "6 NER backends, entity resolution, bulk export scripts"],
        ["Nhu Quang Anh", "QE180005", "Retrieval + Eval", "WRRF, reranker, evaluation pipeline, baselines"],
        ["Pham Ngo Dinh Khoi", "QE180110", "Agent + LLM + Dataset", "ReAct agent, local LLM, MHQA generation"],
        ["Truong Dinh Thien", "QE180018", "API + Deploy", "FastAPI, job management, Gradio demo"],
      ], [2500, 1500, 2200, 3438]),

      // 7
      h1("7. Results & Baselines"),
      makeTable(["Method", "MRR", "Hit Rate@5", "Latency"], [
        ["BM25 only", "~0.35", "~55%", "0.8s"],
        ["Dense only", "~0.40", "~60%", "1.2s"],
        ["Naive RAG", "~0.42", "~62%", "2.5s"],
        ["BM25 + Vector", "~0.48", "~66%", "1.5s"],
        ["Ours (WRRF+Rerank)", "0.4497", "64.67%", "51ms"],
      ], [3500, 1800, 2200, 2138]),
      para("Benchmark: ViQuAD2 (2,652 queries). Hit Rate below 70% target due to insufficient KG entity coverage — full ingestion needed."),

      // 8
      h1("8. System Scale (UCP = 96.5)"),
      makeTable(["Factor", "Value", "Notes"], [
        ["UAW", "9", "End User(1) + Admin(2) + Neo4j(3) + Gemini API(3)"],
        ["UUCW", "125", "10 use cases: 6 Complex(15) + 3 Average(10) + 1 Simple(5)"],
        ["UUCP", "134", "UAW + UUCW"],
        ["TCF", "0.9", "Complex AI/ML, concurrency, security, third-party APIs"],
        ["ECF", "0.8", "Neo4j + AI/ML learning curve, part-time team"],
        ["UCP", "96.5", "134 × 0.9 × 0.8 — Feasible within 15 weeks"],
      ], [2000, 2000, 5638]),
      para("Effort: 96.5 × 20 = 1,930 person-hours. Available: 3 × 15 × 40 = 1,800h. Existing codebase ~30% head-start → net ~1,350h. Feasible."),

      // 9
      h1("9. Timeline & Plan"),
      makeTable(["Phase", "Week", "Milestone", "Status"], [
        ["Phase 1: Foundation", "1-4", "Core pipeline + SRS + Architecture refactor", "Done"],
        ["Phase 2: Eval & Data", "5-7", "Full ingestion + Ablation + MHQA expansion", "In progress"],
        ["Phase 3: Fine-tune", "8-10", "QLoRA Text2Cypher + DPO alignment", "Pending"],
        ["Phase 4: Optimization", "11-13", "WRRF tuning + final evaluation", "Pending"],
        ["Phase 5: Defense", "14-15", "Final report + demo + capstone defense", "Pending"],
      ], [2500, 1200, 4300, 1638]),
      h2("Plan for W24 (Jun 09-15)"),
      bullet("Run full ingestion ~50K articles to get real KG size numbers"),
      bullet("Ablation study: toggle each WRRF component on/off"),
      bullet("Evaluate MHQA dataset on ViWiki-MHR (200 queries)"),
      bullet("Prepare live demo for Review 1"),
      bullet("Fine-tune WRRF weights based on ablation results"),

      pageBreak(),
      new Paragraph({
        children: [new TextRun({ text: "ViWiki-GraphRAG  |  FPT University Capstone SE+AI  |  2026  |  Corpus: Keithsel/viwiki-20260523", size: 18, color: "888888", font: "Arial", italics: true })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 480 },
      }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("review1-report-v2.docx", buf);
  console.log("Done: review1-report-v2.docx");
});
