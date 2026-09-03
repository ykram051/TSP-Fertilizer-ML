from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "industrial_integration_architecture.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#123044")
GREEN = colors.HexColor("#008B72")
PALE = colors.HexColor("#EAF5F2")
LIGHT = colors.HexColor("#F3F6F8")
AMBER = colors.HexColor("#D58B00")
TEXT = colors.HexColor("#1E2B32")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=24, leading=29, textColor=NAVY, alignment=TA_CENTER,
                          spaceAfter=12))
styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=11, leading=15,
                          textColor=colors.HexColor("#45616E"), alignment=TA_CENTER))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
                          fontSize=17, leading=21, textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11, leading=14, textColor=GREEN, spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.6, leading=12,
                          textColor=TEXT, spaceAfter=4))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.4, leading=9.4,
                          textColor=TEXT))
styles.add(ParagraphStyle(name="Box", parent=styles["BodyText"], fontSize=8, leading=10,
                          textColor=colors.white, alignment=TA_CENTER))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5E1E5"))
    canvas.line(16 * mm, 11 * mm, 281 * mm, 11 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#607983"))
    canvas.drawString(16 * mm, 7 * mm, "Ikram Benfellah | SLURRY_FREE_ACID integration architecture")
    canvas.drawRightString(281 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def box(text, width=42 * mm, color=NAVY):
    table = Table([[Paragraph(text, styles["Box"])]], colWidths=[width], rowHeights=[18 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.7, color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def arrow():
    return Paragraph("&gt;", ParagraphStyle(name="Arrow", parent=styles["Bodyx"],
                                            fontName="Helvetica-Bold", fontSize=15,
                                            alignment=TA_CENTER, textColor=GREEN))


def bullet(text):
    return Paragraph(f"&#8226; {text}", styles["Bodyx"])


def data_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("LEADING", (0, 0), (-1, -1), 9.2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C9CF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


pagesize = landscape(A4)
doc = BaseDocTemplate(str(OUTPUT), pagesize=pagesize,
                      leftMargin=16 * mm, rightMargin=16 * mm,
                      topMargin=15 * mm, bottomMargin=16 * mm)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="standard", frames=[frame], onPage=footer)])
story = []

# Cover
story += [Spacer(1, 27 * mm), Paragraph("SLURRY_FREE_ACID", styles["Title2"]),
          Paragraph("Macro and Micro Integration Architecture", styles["Title2"]),
          Spacer(1, 5 * mm),
          Paragraph("Advisory industrial ML deployment - CSV replay, OPC UA, Docker and SQLite",
                    styles["Sub"]), Spacer(1, 19 * mm)]
cover = Table([
    [Paragraph("Prepared by", styles["Small"]), Paragraph("Ikram Benfellah", styles["Bodyx"])],
    [Paragraph("Current stage", styles["Small"]), Paragraph("Stage 1 laboratory simulation", styles["Bodyx"])],
    [Paragraph("Safety position", styles["Small"]), Paragraph("Read-only/advisory; no PLC or DCS control writes", styles["Bodyx"])],
    [Paragraph("Validated output", styles["Small"]), Paragraph("SQLite prediction audit plus JSON mirror", styles["Bodyx"])],
], colWidths=[35 * mm, 115 * mm], hAlign="CENTER")
cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE),
                           ("BOX", (0, 0), (-1, -1), 0.7, GREEN),
                           ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.white),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
story += [cover, PageBreak()]

# Macro
story += [Paragraph("1. Macro architecture: plant to operator", styles["H1x"]),
          Paragraph("Decision: the model is an isolated advisory layer. Plant control and safety remain in the PLC/DCS even when ML, Docker, OPC or storage fails.", styles["Bodyx"]),
          Spacer(1, 5 * mm)]
flow = [box("Sensors / manual laboratory", 38 * mm), arrow(), box("PLC / DCS<br/>control + safety", 38 * mm), arrow(),
        box("KEPServerEX<br/>protocol gateway", 38 * mm, GREEN), arrow(), box("OPC UA<br/>secure tags", 32 * mm, GREEN), arrow(),
        box("Docker ML<br/>inference", 36 * mm), arrow(), box("SQLite / HMI<br/>advisory", 37 * mm)]
story += [Table([flow], colWidths=[38*mm,7*mm,38*mm,7*mm,38*mm,7*mm,32*mm,7*mm,36*mm,7*mm,37*mm]), Spacer(1, 7 * mm)]
rows = [["Layer", "Component", "Responsibility", "If it fails"],
        ["Physical", "Sensors / laboratory", "Measure process and free-acid samples", "Controller applies existing safety behavior"],
        ["Control", "PLC / DCS", "Deterministic control, alarms, interlocks", "Independent of the ML service"],
        ["Translation", "Kepware", "Vendor drivers southbound; OPC UA northbound", "Inference disconnects and retries"],
        ["Compute", "Docker inference", "Validate, buffer, engineer features, predict", "No control function is lost"],
        ["Persistence", "SQLite + JSON", "Audit runs, results, quality and status", "No unaudited advisory result is released"],
        ["Presentation", "HMI / dashboard", "Show estimate, timestamp, quality and status", "Operator retains authority"]]
story += [data_table(rows, [25*mm,38*mm,100*mm,92*mm]), Spacer(1, 3*mm),
          Paragraph("Interfaces", styles["H2x"]),
          bullet("Field to controller: 4-20 mA or digital I/O; existing plant responsibility."),
          bullet("Controller to Kepware: vendor-specific industrial protocol, to be confirmed with IT/OT."),
          bullet("Kepware to Python: OPC UA subscription with timestamps and quality codes."),
          bullet("Python to user: persisted advisory result; optional dedicated HMI tags only after approval."),
          PageBreak()]

# Micro
story += [Paragraph("2. Micro architecture: one inference cycle", styles["H1x"])]
micro = [[box("1. Sequence<br/>notification", 32*mm, GREEN), arrow(), box("2. Coherent<br/>snapshot", 32*mm), arrow(), box("3. Quality<br/>shield", 32*mm), arrow(), box("4. Rolling<br/>buffer", 32*mm), arrow(), box("5. Saved feature<br/>contract", 34*mm), arrow(), box("6. Saved model<br/>pipeline", 34*mm), arrow(), box("7. SQLite<br/>transaction", 32*mm, GREEN)]]
story += [Table(micro, colWidths=[32*mm,6*mm,32*mm,6*mm,32*mm,6*mm,32*mm,6*mm,34*mm,6*mm,34*mm,6*mm,32*mm]), Spacer(1,5*mm)]
steps = [["Step", "Implementation", "Reason / output"],
         ["1", "Subscribe to the OPC sequence tag", "Avoid repeated polling; process each committed source row once"],
         ["2", "Read sequence before and after all mapped tags", "Reject a mixed snapshot if a new row was committed during reading"],
         ["3", "Check OPC quality, timestamp, cadence, ranges and missing share", "Isolated gaps use trained imputation; >25% missing required inputs is rejected"],
         ["4", "Append accepted rows to a bounded chronological buffer", "Preserve the exact history needed by temporal features"],
         ["5", "Wait for 31 rows on a cold start, then build saved columns", "30-minute lookback plus current row; no invented startup history"],
         ["6", "Apply persisted preprocessing and selected model without refitting", "Training/production consistency and leakage prevention"],
         ["7", "Commit SQLite row, mirror JSON, optionally publish advisory tags", "Auditable result before presentation; no control tags"]]
story += [data_table(steps, [14*mm,114*mm,127*mm]), Spacer(1, 5*mm),
          Paragraph("Operational modes", styles["H2x"])]
modes = [["Mode", "Target required at inference", "Meaning"],
         ["target_free", "No", "5-cluster Ridge virtual sensor estimates current free acid from process inputs"],
         ["target_anchored", "Fresh manual laboratory value + timestamp", "Predicts 1/5/10/15-minute change and adds it to the known current level"]]
story += [data_table(modes, [38*mm,78*mm,139*mm]), Spacer(1,4*mm),
          Paragraph("Important: the laboratory value is manual. It cannot be silently treated as a continuously available online analyzer value.",
                    ParagraphStyle(name="Warn", parent=styles["Bodyx"], textColor=AMBER, fontName="Helvetica-Bold")),
          PageBreak()]

# Database
story += [Paragraph("3. SQLite output and traceability", styles["H1x"]),
          Paragraph("Decision: SQLite is the authoritative Stage-1 audit store. It is embedded in the inference process, not a separate network server. JSONL/latest JSON remain readable mirrors.", styles["Bodyx"])]
dbflow = Table([[box("service_runs", 55*mm), arrow(), box("predictions", 70*mm, GREEN), Spacer(1,10), box("service_events", 60*mm)]],
               colWidths=[55*mm,10*mm,70*mm,15*mm,60*mm], hAlign="CENTER")
story += [Spacer(1,4*mm), dbflow, Spacer(1,6*mm)]
dbrows = [["Table", "One row represents", "Important fields and controls"],
          ["service_runs", "One service startup", "run UUID, service/model version, mode, horizon, start time, configuration snapshot"],
          ["predictions", "One successful prediction", "source and forecast timestamps, result/change, quality, imputed inputs, model; UNIQUE(run_id, sequence)"],
          ["service_events", "One status transition", "STARTING, CONNECTED, WARMING_UP, RUNNING, DATA_ERROR or DISCONNECTED with context"]]
story += [data_table(dbrows, [38*mm,65*mm,152*mm]), Spacer(1,5*mm),
          Paragraph("Reliability choices", styles["H2x"]),
          bullet("Each prediction is inserted transactionally; incomplete database rows are not possible."),
          bullet("WAL mode supports an inspection reader while the inference process writes."),
          bullet("A run UUID separates restarts and repeated CSV replays; sequence uniqueness is enforced within a run."),
          bullet("Timestamp and quality indexes support event review and performance validation."),
          bullet("The Docker bind mount keeps the database outside the container lifecycle."),
          Paragraph("Validated Stage-1 result", styles["H2x"]),
          Paragraph("The rebuilt container passed storage tests and produced live SQLite records from the OPC replay. The inspection command returned model name clustered_ridge_k5, process timestamps, predictions and GOOD input-quality status; CSV export also completed.", styles["Bodyx"]),
          PageBreak()]

# Roadmap/questions
story += [Paragraph("4. Integration sequence and decisions still required", styles["H1x"])]
road = [["Stage", "What is connected", "Acceptance evidence"],
        ["1 - completed prototype", "CSV replay OPC server -> Docker inference -> SQLite/JSON", "Build, subscription, warm-up, prediction, status, persistence and export"],
        ["2 - Kepware laboratory", "Approved Kepware instance -> Docker client", "Real Node IDs, certificates, SignAndEncrypt, reconnect and quality-code tests"],
        ["3 - production shadow", "Read-only plant tags -> silent prediction storage", "Timestamp alignment, laboratory matching, drift and prospective error review"],
        ["4 - advisory HMI", "Dedicated model output/status tags -> operator display", "Human factors, alarm policy, ownership and failure-state approval"]]
story += [data_table(road, [42*mm,96*mm,117*mm]), Spacer(1,5*mm),
          Paragraph("Information to obtain from the supervisor / IT-OT team", styles["H2x"]),
          bullet("DCS and PLC vendors/versions; exact meaning and scope of the plant 'automate'."),
          bullet("OPC UA versus OPC DA, Kepware endpoint, real Node IDs for all process variables, security policy and certificates."),
          bullet("Network route and firewall rule between the Windows Docker host and Kepware host."),
          bullet("Authoritative source timestamps, OPC quality semantics and clock synchronization."),
          bullet("Manual laboratory sampling/entry workflow and how a sample time differs from a database entry time."),
          bullet("Database retention, backup location, responsible owner and whether the plant historian becomes the long-term record."),
          bullet("Approved advisory tags and HMI behavior for warming-up, imputed data, data error and disconnection."),
          Spacer(1,5*mm),
          Paragraph("Safety conclusion", styles["H2x"]),
          Paragraph("The current architecture is suitable for laboratory and read-only shadow validation. It is not approved for automatic intervention. The next technical milestone is a secured Kepware laboratory connection followed by prospective comparison with timestamped laboratory results.", styles["Bodyx"])]

doc.build(story)
print(OUTPUT)
