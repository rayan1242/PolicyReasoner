import gradio as gr
import json
import time
import threading
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from tools.policy_ingestor import build_index
from agent import graph, PolicyStateInput

# ---------------------------------------------------------------------------
# Logging buffer
# ---------------------------------------------------------------------------
LOG_BUFFER = []
LOG_BUFFER_LOCK = threading.Lock()


class BufferLogHandler(logging.Handler):
    def emit(self, record):
        with LOG_BUFFER_LOCK:
            LOG_BUFFER.append(self.format(record))


root_logger = logging.getLogger()
if not any(isinstance(h, BufferLogHandler) for h in root_logger.handlers):
    handler = BufferLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s — %(message)s"))
    root_logger.addHandler(handler)

build_index()

# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

WORKFLOW_DIAGRAM = """
<div style="font-family:monospace;font-size:13px;line-height:1.8;background:#f8f9fa;
            border-radius:10px;padding:20px 32px;display:inline-block">
  <div style="text-align:center;font-weight:bold;font-size:15px;margin-bottom:8px">
    PolicyReasoner — Agent Workflow
  </div>
  <div style="color:#555">User Query</div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#2a7ae2">① Query Expansion <span style="color:#888;font-size:11px">(LLM → search tags)</span></div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#2a7ae2">② Hybrid Retrieval
    <span style="color:#888;font-size:11px">(Dense FAISS + BM25 → top 20 chunks)</span></div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#2a7ae2">③ Cross-Encoder Re-ranking
    <span style="color:#888;font-size:11px">(MiniLM → top 8 chunks)</span></div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#2a7ae2">④ Policy Analysis
    <span style="color:#888;font-size:11px">(LLM → findings + evidence + conflicts)</span></div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#2a7ae2">⑤ Summarization
    <span style="color:#888;font-size:11px">(LLM → plain English + confidence score)</span></div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#2a7ae2">⑥ Policy → Code
    <span style="color:#888;font-size:11px">(LLM → JSON rules + Python + ML features, validated)</span></div>
  <div style="color:#aaa;margin-left:16px">↓</div>
  <div style="color:#555">Final Answer</div>
</div>
"""


def conf_badge(score: float) -> str:
    color = "#2e7d32" if score >= 0.8 else "#f57c00" if score >= 0.5 else "#c62828"
    return (f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:bold">'
            f'Confidence {score:.0%}</span>')


def render_summary(summary: str, confidence: float, reason: str) -> str:
    if not summary:
        return ""
    badge = conf_badge(confidence)
    return f"""
    <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:20px">
      <div style="margin-bottom:12px">{badge}
        <span style="color:#888;font-size:12px;margin-left:10px">{reason}</span>
      </div>
      <p style="font-size:15px;line-height:1.7;margin:0">{summary}</p>
    </div>"""


def render_policies_table(chunks: list) -> str:
    if not chunks:
        return "<p style='color:#888'>No policies retrieved.</p>"
    rows = ""
    for i, c in enumerate(chunks, 1):
        score = round(c.get("cross_encoder_score", 0), 2)
        keywords = c.get("match_keywords", [])
        kw_html = " ".join(
            f'<span style="background:#e3f2fd;color:#1565c0;padding:1px 7px;'
            f'border-radius:10px;font-size:11px;margin:1px">{k}</span>'
            for k in keywords
        ) if keywords else '<span style="color:#aaa;font-size:11px">—</span>'
        rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="padding:10px 8px;width:28px;text-align:center;font-weight:bold;color:#666">{i}</td>
          <td style="padding:10px 8px">
            <b>{c['name']}</b><br/>
            <span style="font-size:11px;color:#888">{c['policy_id']} · {c['section_title']}</span>
          </td>
          <td style="padding:10px 8px;width:110px">
            <span style="background:#e8f4f8;padding:2px 8px;border-radius:10px;font-size:11px">{c['category']}</span>
          </td>
          <td style="padding:10px 8px;width:70px;text-align:center;color:#2a7ae2;font-weight:bold">{score}</td>
          <td style="padding:10px 8px">{kw_html}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead>
        <tr style="background:#f4f4f4;font-size:12px;color:#555">
          <th style="padding:8px">#</th>
          <th style="padding:8px;text-align:left">Policy</th>
          <th style="padding:8px">Category</th>
          <th style="padding:8px">CE Score</th>
          <th style="padding:8px;text-align:left">Matched Because</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def render_grounding(grounding: list) -> str:
    if not grounding:
        return "<p style='color:#888'>No evidence extracted.</p>"
    items = ""
    for g in grounding:
        conf = g.get("confidence", 0.0)
        items += f"""
        <div style="border:1px solid #e0e0e0;border-radius:8px;padding:14px 16px;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-weight:bold;color:#333">{g.get('policy_id','')}</span>
            {conf_badge(conf)}
          </div>
          <div style="font-size:12px;color:#888;margin-bottom:6px">
            Section: {g.get('section','')}
          </div>
          <blockquote style="border-left:3px solid #2a7ae2;padding-left:12px;
                             margin:0;color:#444;font-size:13px;font-style:italic">
            "{g.get('quote','')}"
          </blockquote>
        </div>"""
    return items or "<p style='color:#888'>No evidence items parsed.</p>"


def render_conflicts(conflicts: list) -> str:
    if not conflicts:
        return "<p style='color:#2e7d32;font-size:14px'>✅ No conflicts detected between retrieved policies.</p>"

    items = ""
    for c in conflicts:
        pa = c.get("policy_a", "")
        pb = c.get("policy_b", "")
        desc = c.get("description", "")
        priority = c.get("priority", "Review required")
        chain = c.get("priority_chain", [])
        chain_html = " → ".join(
            f'<b style="color:#c62828">{s}</b>' if s.lower() in priority.lower() else s
            for s in chain
        ) if chain else priority

        items += f"""
        <div style="background:#fff3e0;border-left:4px solid #ff9800;
                    border-radius:4px;padding:14px 16px;margin-bottom:10px">
          <div style="font-weight:bold;color:#e65100;margin-bottom:6px">⚠️ Conflict Detected</div>
          <div style="font-size:13px;margin-bottom:6px">
            <code style="background:#ffe0b2;padding:1px 6px;border-radius:4px">{pa}</code>
            &nbsp;vs&nbsp;
            <code style="background:#ffe0b2;padding:1px 6px;border-radius:4px">{pb}</code>
          </div>
          <div style="font-size:13px;color:#444;margin-bottom:8px">{desc}</div>
          <div style="font-size:12px;color:#888">
            Priority chain: {chain_html}
          </div>
        </div>"""
    return items


def render_json_rules(rules: dict, confidence: float) -> str:
    if not rules or "error" in rules:
        return "<p style='color:#888'>No JSON rules generated.</p>"
    pretty = json.dumps(rules, indent=2)
    badge = conf_badge(confidence)
    return f"""
    <div style="margin-bottom:8px">{badge}</div>
    <pre style="background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;
                overflow:auto;font-size:12px;line-height:1.5">{pretty}</pre>"""


def render_python(code: str, valid: bool, error: str, test_out, confidence: float) -> str:
    if not code:
        return "<p style='color:#888'>No Python code generated.</p>"

    badge = conf_badge(confidence)
    valid_badge = (
        '<span style="background:#2e7d32;color:white;padding:2px 10px;border-radius:12px;font-size:12px">✓ Compiled &amp; tested</span>'
        if valid and test_out else
        '<span style="background:#2e7d32;color:white;padding:2px 10px;border-radius:12px;font-size:12px">✓ Compiled</span>'
        if valid else
        f'<span style="background:#c62828;color:white;padding:2px 10px;border-radius:12px;font-size:12px">✗ {error or "Validation failed"}</span>'
    )

    test_html = ""
    if test_out and isinstance(test_out, dict):
        test_html = f"""
        <div style="background:#f1f8e9;border:1px solid #aed581;border-radius:6px;
                    padding:10px 14px;margin-bottom:10px;font-size:12px">
          <b>Test run output:</b> {json.dumps(test_out)}
        </div>"""
    elif error:
        test_html = f"""
        <div style="background:#fce4ec;border:1px solid #e57373;border-radius:6px;
                    padding:10px 14px;margin-bottom:10px;font-size:12px;color:#c62828">
          ⚠️ {error}
        </div>"""

    note = """<div style="background:#fff8e1;border:1px solid #ffe082;border-radius:6px;
                          padding:8px 14px;margin-bottom:10px;font-size:12px;color:#6d4c41">
      ⚠️ Generated code is a draft. Validate against the source policy before use in production.
    </div>"""

    return f"""
    <div style="margin-bottom:8px">{badge} &nbsp; {valid_badge}</div>
    {note}{test_html}
    <pre style="background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;
                overflow:auto;font-size:12px;line-height:1.5">{code}</pre>"""


def render_features(features: dict, confidence: float) -> str:
    if not features or "error" in features:
        return "<p style='color:#888'>No ML features generated.</p>"
    pretty = json.dumps(features, indent=2)
    badge = conf_badge(confidence)
    return f"""
    <div style="margin-bottom:8px">{badge}</div>
    <pre style="background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;
                overflow:auto;font-size:12px;line-height:1.5">{pretty}</pre>"""


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_agent(query, result_box):
    try:
        result_box["result"] = graph.invoke(PolicyStateInput(user_query=query))
    except Exception as e:
        result_box["error"] = str(e)


EXAMPLES = [
    "Do I need preauthorization for emergency surgery and how much will it cost?",
    "Find all billing policies covering out-of-network emergency care and detect conflicts",
    "What mental health therapy is covered and are there visit limits?",
    "What are the prior authorization requirements for buprenorphine?",
    "Compare preauthorization requirements for MRI vs routine lab tests",
]


def stream_analysis(query, output_format):
    if not query.strip():
        yield ("⚠️ Please enter a question.",) + ("",) * 8
        return

    with LOG_BUFFER_LOCK:
        LOG_BUFFER.clear()

    result_box = {}
    thread = threading.Thread(target=run_agent, args=(query, result_box))
    thread.start()

    last_idx = 0
    while thread.is_alive() or last_idx < len(LOG_BUFFER):
        with LOG_BUFFER_LOCK:
            new_logs = LOG_BUFFER[last_idx:]
            last_idx = len(LOG_BUFFER)
        if new_logs:
            status = "⏳ " + new_logs[-1].split("—")[-1].strip()
            yield (status,) + ("",) * 8
        time.sleep(0.4)

    thread.join()

    if result_box.get("error"):
        yield (f"❌ Error: {result_box['error']}",) + ("",) * 8
        return

    r = result_box["result"]
    conv = r.get("converted_output", {})

    yield (
        "✅ Done",
        render_summary(
            r.get("summary", ""),
            r.get("summary_confidence", 0.0),
            r.get("summary_confidence_reason", ""),
        ),
        render_policies_table(r.get("reranked_policies", [])),
        render_grounding(r.get("grounding", [])),
        render_conflicts(r.get("conflicts", [])),
        render_json_rules(conv.get("json_rules", {}), conv.get("json_confidence", 0.0)),
        render_python(
            conv.get("python_code", ""),
            conv.get("python_valid", False),
            conv.get("python_validation_error"),
            conv.get("python_test_output"),
            conv.get("python_confidence", 0.0),
        ),
        render_features(conv.get("ml_features", {}), conv.get("features_confidence", 0.0)),
        r.get("analysis", ""),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

TITLE = """
<div style="text-align:center;padding:24px 0 8px">
  <h1 style="font-size:36px;margin:0;font-weight:700">🏥 PolicyReasoner</h1>
  <p style="font-size:15px;color:#666;margin-top:8px">
    Agentic AI for healthcare policy discovery, conflict detection &amp; code generation
  </p>
</div>"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
        #main { max-width: 980px; margin: auto; }
        pre { white-space: pre-wrap; word-break: break-word; }
        .gr-tab-nav button { font-size: 13px; }
    """
) as demo:

    gr.HTML(TITLE)

    with gr.Column(elem_id="main"):

        with gr.Accordion("How it works", open=False):
            gr.HTML(WORKFLOW_DIAGRAM)

        with gr.Row():
            query_box = gr.Textbox(
                label="Policy Question",
                placeholder="e.g. Do I need preauthorization for emergency surgery?",
                lines=2, scale=5,
            )
            format_dd = gr.Dropdown(
                choices=["all", "json", "python", "features"],
                value="all", label="Conversion Format", scale=1,
            )

        run_btn = gr.Button("🔍 Analyze Policies", variant="primary", size="lg")

        gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[query_box], label="Examples")

        status_md = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("📋 Summary"):
                summary_html = gr.HTML("")
            with gr.Tab("📄 Retrieved Policies"):
                policies_html = gr.HTML("")
            with gr.Tab("🔎 Evidence & Grounding"):
                grounding_html = gr.HTML("")
            with gr.Tab("⚠️ Conflicts"):
                conflicts_html = gr.HTML("")
            with gr.Tab("🔧 JSON Rules"):
                json_html = gr.HTML("")
            with gr.Tab("🐍 Python Code"):
                python_html = gr.HTML("")
            with gr.Tab("📊 ML Features"):
                features_html = gr.HTML("")
            with gr.Tab("🔬 Raw Analysis"):
                raw_md = gr.Markdown("")

    gr.HTML("""
    <div style="text-align:center;margin-top:32px;font-size:12px;color:#aaa">
      Powered by Groq + LangGraph · Built for Cotiviti Intern Assessment
    </div>""")

    outputs = [status_md, summary_html, policies_html, grounding_html,
               conflicts_html, json_html, python_html, features_html, raw_md]

    run_btn.click(fn=stream_analysis, inputs=[query_box, format_dd], outputs=outputs)
    query_box.submit(fn=stream_analysis, inputs=[query_box, format_dd], outputs=outputs)

if __name__ == "__main__":
    demo.queue(max_size=5).launch(share=False, show_error=True)
