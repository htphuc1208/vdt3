// ShardRCA VDT2026 defense deck — bilingual (VI body + EN technical terms), 13 slides.
const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.join(__dirname, "..", "..", "report", "figures", "shard_rca.png");

// ---- palette (telecom / scientific-rigor) ----
const NAVY   = "0E1B2E";   // dark bg
const NAVY2  = "12345B";   // primary
const TEAL   = "0FB8A6";   // accent (positive / constructive)
const CORAL  = "E8604C";   // accent (refutation / negative)
const AMBER  = "E7A33E";
const INK    = "1E293B";   // body text on light
const MUTE   = "64748B";
const ICE    = "EAF0F7";   // light card tint
const WHITE  = "FFFFFF";

const HEAD = "Cambria";    // safe serif header
const BODY = "Calibri";    // safe sans body

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";       // 13.3 x 7.5
pres.author = "Hoang Tan Phuc";
pres.title = "ShardRCA — VDT2026 Defense";
const W = 13.3, Hh = 7.5;

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.14 });

// small helpers
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || WHITE }, line: { color: "D7E0EA", width: 1 }, shadow: shadow() });
}
function chip(slide, x, y, w, txt, col) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, rectRadius: 0.17,
    fill: { color: col }, line: { type: "none" } });
  slide.addText(txt, { x, y, w, h: 0.34, align: "center", valign: "middle",
    fontFace: BODY, fontSize: 11.5, bold: true, color: WHITE, margin: 0 });
}
function title(slide, txt, sub) {
  slide.addText(txt, { x: 0.6, y: 0.32, w: 12.1, h: 0.66, fontFace: HEAD, fontSize: 27, bold: true, color: NAVY2, margin: 0 });
  if (sub) slide.addText(sub, { x: 0.6, y: 0.98, w: 12.1, h: 0.34, fontFace: BODY, fontSize: 13, italic: true, color: MUTE, margin: 0 });
}
function pageno(slide, n) {
  slide.addText(String(n) + " / 13", { x: 12.3, y: 7.05, w: 0.8, h: 0.3, align: "right",
    fontFace: BODY, fontSize: 9, color: "AAB4C0", margin: 0 });
}

// =============================================================== SLIDE 1 — TITLE
let s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(pres.shapes.OVAL, { x: 9.6, y: -2.2, w: 6, h: 6, fill: { color: NAVY2, transparency: 55 }, line: { type: "none" } });
s.addShape(pres.shapes.OVAL, { x: 11.0, y: 4.2, w: 4.5, h: 4.5, fill: { color: TEAL, transparency: 80 }, line: { type: "none" } });
chip(s, 0.9, 1.15, 3.6, "VDT2026 · CHUYÊN NGÀNH DSAI", TEAL);
s.addText("ShardRCA", { x: 0.85, y: 1.7, w: 11, h: 1.0, fontFace: HEAD, fontSize: 54, bold: true, color: WHITE, margin: 0 });
s.addText("Khi nào cô lập bằng chứng đa tác tử LLM giúp ích cho phân tích nguyên nhân gốc rễ?",
  { x: 0.9, y: 2.85, w: 11.3, h: 0.6, fontFace: BODY, fontSize: 20, color: "CDE3F2", margin: 0 });
s.addText("Một nghiên cứu phản chứng có kỷ luật  ·  A disciplined negative study on evidence-isolated multi-agent RCA",
  { x: 0.9, y: 3.45, w: 11.3, h: 0.5, fontFace: BODY, fontSize: 13.5, italic: true, color: "8FB2CC", margin: 0 });
s.addShape(pres.shapes.LINE, { x: 0.95, y: 4.5, w: 4.2, h: 0, line: { color: TEAL, width: 2 } });
s.addText([
  { text: "Hoàng Tấn Phúc", options: { bold: true, color: WHITE } },
  { text: "  (sinh viên)", options: { color: "9FB8CC" } },
], { x: 0.9, y: 4.7, w: 11, h: 0.35, fontFace: BODY, fontSize: 15, margin: 0 });
s.addText("Mentor: Nguyễn Duy Anh, Đặng Anh Quân  ·  Viettel Networks  ·  hoangtanphuc05@gmail.com",
  { x: 0.9, y: 5.1, w: 11, h: 0.35, fontFace: BODY, fontSize: 12.5, color: "8FB2CC", margin: 0 });
s.addNotes("Chào hội đồng, giới thiệu tên + tiêu đề. Nhấn: đây là một nghiên cứu phản chứng CÓ KỶ LUẬT, không phải khoe con số.");

// =============================================================== SLIDE 2 — PROBLEM
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Bài toán & động lực", "Root-Cause Analysis (RCA) trong mạng 5G / hệ vi dịch vụ");
s.addText([
  { text: "Khi sự cố xảy ra", options: { bold: true, color: NAVY2 } },
  { text: ", kỹ sư phải tìm ", options: {} },
  { text: "thành phần lỗi (component)", options: { bold: true } },
  { text: " + ", options: {} },
  { text: "nguyên nhân (reason)", options: { bold: true } },
  { text: " trong hàng chục GB telemetry đa phương thức: KPI time-series (CPU/RAM/latency), logs, traces (RPC), alarms — hàng trăm thành phần.", options: {} },
], { x: 0.6, y: 1.5, w: 12.1, h: 0.9, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.05, margin: 0 });

card(s, 0.6, 2.65, 5.85, 1.75, "FBECEA");
s.addText("Giới hạn 1 — Cửa sổ ngữ cảnh", { x: 0.85, y: 2.85, w: 5.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: CORAL, margin: 0 });
s.addText("Telemetry vượt xa context window → phải cắt / tóm lược → dễ đánh rơi bằng chứng quyết định.",
  { x: 0.85, y: 3.3, w: 5.4, h: 0.95, fontFace: BODY, fontSize: 13.5, color: INK, margin: 0 });

card(s, 6.85, 2.65, 5.85, 1.75, "FBECEA");
s.addText("Giới hạn 2 — Chẩn đoán vô căn cứ", { x: 7.1, y: 2.85, w: 5.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: CORAL, margin: 0 });
s.addText("Bị nhồi quá nhiều dữ liệu → LLM đưa chẩn đoán 'nghe hợp lý nhưng ungrounded' (không có căn cứ).",
  { x: 7.1, y: 3.3, w: 5.4, h: 0.95, fontFace: BODY, fontSize: 13.5, color: INK, margin: 0 });

card(s, 0.6, 4.7, 12.1, 1.9, NAVY);
s.addText("Xu hướng 2024–2026:  ném multi-agent LLM vào để giải.", { x: 0.9, y: 4.95, w: 11.5, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, margin: 0 });
s.addText([
  { text: "Câu hỏi bị bỏ quên → ", options: { color: "9FB8CC" } },
  { text: "nó có THỰC SỰ giúp không, hay chỉ là ảo giác do tốn nhiều token hơn?", options: { bold: true, color: TEAL } },
], { x: 0.9, y: 5.5, w: 11.5, h: 0.9, fontFace: BODY, fontSize: 17, margin: 0 });
pageno(s, 2);
s.addNotes("Kể vấn đề. Nhấn 2 giới hạn của single agent. Chuyển: xu hướng là multi-agent, nhưng câu hỏi 'có thật sự giúp' bị bỏ qua.");

// =============================================================== SLIDE 3 — RQ + contributions
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Câu hỏi nghiên cứu & đóng góp");
card(s, 0.6, 1.45, 12.1, 1.15, NAVY2);
s.addText([
  { text: "Câu hỏi trung tâm:  ", options: { color: "Bcd6ea", bold: true } },
  { text: "Một hệ đa tác tử cô lập bằng chứng có vượt tác tử đơn cho RCA không — và NHỜ CƠ CHẾ NÀO?", options: { bold: true, color: WHITE } },
], { x: 0.95, y: 1.55, w: 11.4, h: 0.95, fontFace: BODY, fontSize: 17, valign: "middle", margin: 0 });

const contribs = [
  ["1", "Quy trình đánh giá trung thực", "Tiền đăng ký (preregistered), ghép cặp (paired McNemar), cân bằng ngân sách token — đầu ra audit được.", TEAL],
  ["2", "Kết luận phản chứng có kỷ luật", "Dưới ngân sách công bằng: cô lập & phản biện chéo KHÔNG giúp; phân mảnh còn gây hại.", CORAL],
  ["3", "Khung chẩn đoán γ = N(x)/B_eff", "Dự báo TRƯỚC khi nào phân rã đa tác tử có lợi — kiểm chứng bằng same-budget test.", NAVY2],
  ["4", "Hai cải tiến có căn cứ", "Tầng quyết định LLM toàn cục (→0.72); de-collapse log-opinion pool trên telecom thật (×3).", AMBER],
];
let cy = 2.95;
contribs.forEach((c, i) => {
  const x = i % 2 === 0 ? 0.6 : 6.85;
  if (i % 2 === 0 && i > 0) cy += 1.9;
  const yy = i < 2 ? 2.95 : 4.85;
  card(s, x, yy, 5.85, 1.7, WHITE);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: yy + 0.28, w: 0.62, h: 0.62, fill: { color: c[3] }, line: { type: "none" } });
  s.addText(c[0], { x: x + 0.25, y: yy + 0.28, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: HEAD, fontSize: 24, bold: true, color: WHITE, margin: 0 });
  s.addText(c[1], { x: x + 1.05, y: yy + 0.22, w: 4.65, h: 0.5, fontFace: HEAD, fontSize: 15.5, bold: true, color: c[3], margin: 0 });
  s.addText(c[2], { x: x + 1.05, y: yy + 0.72, w: 4.65, h: 0.85, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
});
pageno(s, 3);
s.addNotes("Đọc câu hỏi trung tâm. 4 đóng góp: quy trình / phản chứng / khung γ / 2 cải tiến. Nhấn: đóng góp KHÔNG phải 'hệ của em thắng'.");

// =============================================================== SLIDE 4 — ARCHITECTURE
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Kiến trúc ShardRCA", "Chạy live gpt-4o-mini qua function-calling · mọi bước audit được · an toàn nhãn");
// figure left
s.addImage({ path: FIG, x: 0.5, y: 1.45, w: 4.05, h: 5.5, sizing: { type: "contain", w: 4.05, h: 5.5 } });
// pipeline steps right
const steps = [
  ["Planner / Shard builder", "Chia telemetry thành shard rời nhau: modality × nhóm thành phần × cửa sổ thời gian.", NAVY2],
  ["Tác tử điều tra cô lập ×5", "Mỗi tác tử CHỈ thấy shard của mình → đẩy findings có cấu trúc lên blackboard.", TEAL],
  ["Peer interaction", "Phản biện chéo support/challenge + hiệu chỉnh hậu nghiệm (tầng phân biệt với map-reduce).", AMBER],
  ["Log-opinion pool fusion", "Product-of-experts hợp nhất hậu nghiệm → ứng viên xếp hạng.", NAVY2],
  ["Verifier + đáp án", "Xác minh top-vs-runner-up → component + reason + thời điểm + audit trail.", CORAL],
];
let yy = 1.6;
steps.forEach((st) => {
  card(s, 4.95, yy, 7.75, 1.0, WHITE);
  s.addShape(pres.shapes.OVAL, { x: 5.15, y: yy + 0.28, w: 0.44, h: 0.44, fill: { color: st[2] }, line: { type: "none" } });
  s.addText(st[0], { x: 5.75, y: yy + 0.13, w: 6.8, h: 0.38, fontFace: HEAD, fontSize: 14.5, bold: true, color: st[2], margin: 0 });
  s.addText(st[1], { x: 5.75, y: yy + 0.5, w: 6.85, h: 0.45, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
  yy += 1.09;
});
pageno(s, 4);
s.addNotes("Đi từ trên xuống theo hình. Nhấn: mỗi tác tử chỉ thấy shard của mình (evidence isolation); peer interaction là tầng phân biệt. Falsifier gọi đúng tên: evidence-based reranker, KHÔNG phải Popperian.");

// =============================================================== SLIDE 5 — GAMMA
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Khung lý thuyết γ — điểm sáng tạo cốt lõi", "Khi nào phân rã đa tác tử THỰC SỰ có cơ sở?");
card(s, 0.6, 1.55, 12.1, 1.35, NAVY);
s.addText([
  { text: "γ(x) = N(x) / B", options: { fontFace: "Cambria", italic: true } },
  { text: "eff", options: { fontFace: "Cambria", italic: true, subscript: true } },
], { x: 0.6, y: 1.7, w: 4.6, h: 1.05, align: "center", valign: "middle", fontSize: 34, bold: true, color: TEAL, margin: 0 });
s.addText([
  { text: "N(x)", options: { bold: true, color: WHITE } },
  { text: " = kích thước ca sự cố   ·   ", options: { color: "9FB8CC" } },
  { text: "B_eff", options: { bold: true, color: WHITE } },
  { text: " = ngân sách ngữ cảnh thực cho telemetry (sau khi trừ prompt/schema).", options: { color: "9FB8CC" } },
], { x: 5.2, y: 1.7, w: 7.3, h: 1.05, valign: "middle", fontFace: BODY, fontSize: 14, margin: 0 });

card(s, 0.6, 3.15, 5.85, 3.3, "E9F7F4");
s.addText("Phân rã CÓ cơ sở khi CẢ HAI:", { x: 0.85, y: 3.35, w: 5.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: TEAL, margin: 0 });
s.addText([
  { text: "γ > 1 ", options: { bold: true, breakLine: true } },
  { text: "dữ liệu thật sự vượt cửa sổ ngữ cảnh; VÀ", options: { color: MUTE, italic: true, breakLine: true } },
  { text: "Tín hiệu phân tán (distributed)", options: { bold: true, breakLine: true } },
  { text: "thông điệp shard giữ nhiều thông tin về Y hơn một bản nén toàn cục cùng ngân sách.", options: { color: MUTE, italic: true } },
], { x: 0.85, y: 3.85, w: 5.4, h: 2.4, fontFace: BODY, fontSize: 14, color: INK, paraSpaceAfter: 8, margin: 0 });

card(s, 6.85, 3.15, 5.85, 3.3, "FBECEA");
s.addText("Tác tử đơn ĐÃ ĐỦ khi:", { x: 7.1, y: 3.35, w: 5.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: CORAL, margin: 0 });
s.addText([
  { text: "γ ≤ 1", options: { bold: true, breakLine: true } },
  { text: "hoặc có bản nén an toàn nhãn z với I(Y;X|Z) ≈ 0;", options: { color: MUTE, italic: true, breakLine: true } },
  { text: "Tín hiệu tập trung (concentrated)", options: { bold: true, breakLine: true } },
  { text: "→ cô lập chỉ pha loãng ngân sách vào các shard nhiễu.", options: { color: MUTE, italic: true } },
], { x: 7.1, y: 3.85, w: 5.4, h: 2.4, fontFace: BODY, fontSize: 14, color: INK, paraSpaceAfter: 8, margin: 0 });
s.addText("γ là LA BÀN dự báo trước khi nào nên bỏ compute cho MAS — không phải thước micromet.",
  { x: 0.6, y: 6.65, w: 12.1, h: 0.4, align: "center", fontFace: BODY, fontSize: 13, italic: true, bold: true, color: NAVY2, margin: 0 });
pageno(s, 5);
s.addNotes("Giải thích γ đơn giản: tỉ lệ dữ liệu / ngân sách. Điều kiện KÉP để MAS có lợi. Nhấn: γ dự báo TRƯỚC — đây là đóng góp sáng tạo, la bàn không phải thước.");

// =============================================================== SLIDE 6 — EVAL DESIGN
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Thiết kế đánh giá — nền của tính đúng đắn", "Thứ 95% công trình multi-agent bỏ qua");
const pillars = [
  ["Ghép cặp (paired)", "McNemar trên CÙNG một ca → loại nhiễu do độ khó dữ liệu, đo đúng hiệu ứng cơ chế.", TEAL],
  ["Cân bằng ngân sách", "Baseline single_equal_tokens: tác tử đơn với ngân sách mở rộng → tách 'MAS giúp' khỏi 'nhiều compute giúp'.", NAVY2],
  ["Tiền đăng ký", "Đóng băng holdout + kế hoạch phân tích TRƯỚC khi thấy nhãn → chống overfit-guard / p-hacking.", AMBER],
];
pillars.forEach((p, i) => {
  const x = 0.6 + i * 4.08;
  card(s, x, 1.7, 3.85, 3.0, WHITE);
  s.addShape(pres.shapes.OVAL, { x: x + 1.5, y: 2.0, w: 0.85, h: 0.85, fill: { color: p[2] }, line: { type: "none" } });
  s.addText(String(i + 1), { x: x + 1.5, y: 2.0, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  s.addText(p[0], { x: x + 0.2, y: 3.0, w: 3.45, h: 0.5, align: "center", fontFace: HEAD, fontSize: 16.5, bold: true, color: p[2], margin: 0 });
  s.addText(p[1], { x: x + 0.25, y: 3.55, w: 3.35, h: 1.1, align: "center", fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
});
card(s, 0.6, 5.05, 12.1, 1.5, NAVY);
s.addText([
  { text: "Vì sao quan trọng:  ", options: { bold: true, color: TEAL } },
  { text: "claim CŨ '0.60 vs 0.20, p=0.008' (từ validation) ", options: { color: WHITE } },
  { text: "KHÔNG tái lập", options: { bold: true, color: CORAL } },
  { text: " trên holdout tiền đăng ký n=50. Không có quy trình này → đã báo cáo một con số ảo.", options: { color: WHITE } },
], { x: 0.95, y: 5.05, w: 11.4, h: 1.5, valign: "middle", fontFace: BODY, fontSize: 15, margin: 0 });
pageno(s, 6);
s.addNotes("3 trụ cột. Nhấn ví dụ cụ thể: claim cũ đẹp không tái lập → chính quy trình này cứu em khỏi báo cáo sai. Đây là bản lề chuyển sang phần kết quả.");

// =============================================================== SLIDE 7 — MAIN REFUTATION
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Phản chứng chính — RCAEval-Hard, n=50, cân bằng ngân sách");
s.addChart(pres.charts.BAR, [{
  name: "Hit@1",
  labels: ["shardrca_full", "single_react_sc", "single_equal_tok", "no_shard (đọc gộp)"],
  values: [0.54, 0.42, 0.44, 0.74],
}], {
  x: 0.55, y: 1.55, w: 6.5, h: 4.9, barDir: "col",
  chartColors: [NAVY2, MUTE, MUTE, TEAL],
  valAxisMinVal: 0, valAxisMaxVal: 0.8,
  showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 13, dataLabelFontBold: true,
  catAxisLabelColor: INK, catAxisLabelFontSize: 10.5, valAxisLabelColor: MUTE,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: false, showTitle: false, chartArea: { fill: { color: WHITE } },
});
const finds = [
  ["Cân bằng token → không thắng", "full 0.54 vs đơn 0.42–0.44, p = 0.24 (ns), dù tốn 3.6–7.7× token.", CORAL],
  ["Phản biện chéo TRƠ", "no_interaction ≡ full, p = 1.0 — không đổi MỘT quyết định nào trên 50 ca.", CORAL],
  ["Phân mảnh GÂY HẠI", "no_shard (đọc gộp) 0.74 THẮNG full, p = 0.013, ở 1/3 chi phí.", TEAL],
];
let fy = 1.6;
finds.forEach((f) => {
  card(s, 7.3, fy, 5.45, 1.5, f[2] === TEAL ? "E9F7F4" : "FBECEA");
  s.addText(f[0], { x: 7.55, y: fy + 0.16, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: f[2], margin: 0 });
  s.addText(f[1], { x: 7.55, y: fy + 0.58, w: 5.0, h: 0.85, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
  fy += 1.63;
});
pageno(s, 7);
s.addNotes("Slide quan trọng nhất. 3 phát hiện: không thắng khi cân token / phản biện trơ p=1.0 / phân mảnh hại p=0.013. Chỉ vào cột no_shard xanh — đọc gộp thắng ở 1/3 chi phí.");

// =============================================================== SLIDE 8 — BARO + gamma test
s = pres.addSlide(); s.background = { color: WHITE };
title(s, '"Vượt BARO" = metric-prior, không phải MAS — và γ-test bác premise');
// left: BARO table
s.addText("Tách metric-prior (RE2-TT, n=20, Hit@1)", { x: 0.6, y: 1.5, w: 6.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY2, margin: 0 });
s.addTable([
  [ {text:"Hệ", options:{bold:true,color:WHITE,fill:{color:NAVY2}}},
    {text:"prior TẮT", options:{bold:true,color:WHITE,fill:{color:NAVY2},align:"center"}},
    {text:"prior BẬT", options:{bold:true,color:WHITE,fill:{color:NAVY2},align:"center"}} ],
  [ "shardrca_full", {text:"0.60",options:{align:"center"}}, {text:"0.80",options:{align:"center"}} ],
  [ "tác tử đơn", {text:"0.00",options:{align:"center"}}, {text:"0.75",options:{align:"center"}} ],
  [ "BARO (official AC@1)", {text:"0.55",options:{align:"center"}}, {text:"—",options:{align:"center"}} ],
], { x: 0.6, y: 1.95, w: 6.0, colW: [3.0, 1.5, 1.5], rowH: 0.44, fontFace: BODY, fontSize: 12.5, color: INK,
     border: { pt: 0.5, color: "D7E0EA" }, valign: "middle" });
card(s, 0.6, 4.25, 6.0, 2.25, "FBECEA");
s.addText([
  { text: "Prior nâng tác tử đơn 0.00 → 0.75 (p=6e-5). ", options: { breakLine: true, bold: true } },
  { text: "Bật prior: full 0.80 ≈ đơn 0.75, p = 1.0.", options: { breakLine: true, bold: true, color: CORAL } },
  { text: "→ 'Vượt BARO' là do PRIOR an toàn nhãn; MAS không thêm gì có ý nghĩa. (Đơn = 0.00 khi tắt prior chỉ vì bị bóp ngân sách — no_shard đạt 0.88.)", options: { color: INK } },
], { x: 0.85, y: 4.45, w: 5.5, h: 1.9, fontFace: BODY, fontSize: 12.5, paraSpaceAfter: 5, margin: 0 });

// right: gamma same-budget table
s.addText("Same-budget test (24 ca rộng nhất) — Hit@1 single / shard", { x: 6.9, y: 1.5, w: 6.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY2, margin: 0 });
s.addTable([
  [ {text:"B",options:{bold:true,color:WHITE,fill:{color:NAVY2},align:"center"}},
    {text:"γ",options:{bold:true,color:WHITE,fill:{color:NAVY2},align:"center"}},
    {text:"single / shard",options:{bold:true,color:WHITE,fill:{color:NAVY2},align:"center"}},
    {text:"p",options:{bold:true,color:WHITE,fill:{color:NAVY2},align:"center"}} ],
  [ {text:"4",options:{align:"center"}},{text:"271",options:{align:"center"}},{text:"0.83 / 0.21",options:{align:"center",bold:true}},{text:"6e-5",options:{align:"center"}} ],
  [ {text:"8",options:{align:"center"}},{text:"135",options:{align:"center"}},{text:"0.83 / 0.21",options:{align:"center",bold:true}},{text:"6e-5",options:{align:"center"}} ],
  [ {text:"16",options:{align:"center"}},{text:"68",options:{align:"center"}},{text:"0.83 / 0.42",options:{align:"center",bold:true}},{text:"0.013",options:{align:"center"}} ],
  [ {text:"32",options:{align:"center"}},{text:"34",options:{align:"center"}},{text:"0.96 / 1.0",options:{align:"center"}},{text:"1.0",options:{align:"center"}} ],
], { x: 6.9, y: 1.95, w: 6.0, colW: [1.0, 1.2, 2.6, 1.2], rowH: 0.4, fontFace: BODY, fontSize: 12.5, color: INK,
     border: { pt: 0.5, color: "D7E0EA" }, valign: "middle" });
card(s, 6.9, 4.25, 6.0, 2.25, "E9F7F4");
s.addText([
  { text: "Ngay ở γ = 271, ", options: {} },
  { text: "tóm lược toàn cục giữ nguyên nhân gốc TỐT HƠN", options: { bold: true, color: TEAL } },
  { text: " phân bổ shard (0.83 vs 0.21, p=6e-5). Trên RCAEval tín hiệu TẬP TRUNG → tiền đề 'bằng chứng phân tán' sai. γ được xác nhận là chẩn đoán đúng.", options: {} },
], { x: 7.15, y: 4.5, w: 5.5, h: 1.85, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
pageno(s, 8);
s.addNotes("Hai cột: trái = BARO thực chất là prior (full≈đơn khi bật prior, p=1.0). Phải = γ-test: ngay ở γ=271 single vẫn giữ root tốt hơn → premise phân tán sai trên RCAEval.");

// =============================================================== SLIDE 9 — CONSTRUCTIVE
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Mặt xây dựng — hai cải tiến có căn cứ", "Thứ tạo giá trị là lập luận LLM toàn cục, không phải sharding");
// improvement 1
card(s, 0.6, 1.6, 5.95, 4.9, WHITE);
chip(s, 0.85, 1.85, 3.4, "CẢI TIẾN 1 · RCAEval", NAVY2);
s.addText("Tầng quyết định LLM toàn cục", { x: 0.85, y: 2.35, w: 5.45, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY2, margin: 0 });
s.addChart(pres.charts.BAR, [{
  name: "Hit@1", labels: ["fusion cơ học", "shardrca_llmboard", "no_shard"], values: [0.54, 0.72, 0.74],
}], { x: 0.7, y: 2.95, w: 5.7, h: 2.6, barDir: "col", chartColors: [MUTE, TEAL, NAVY2],
  valAxisMinVal: 0, valAxisMaxVal: 0.8, showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontBold: true, dataLabelFontSize: 12,
  catAxisLabelColor: INK, catAxisLabelFontSize: 10, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, chartArea: { fill: { color: WHITE } } });
s.addText("Thay fusion cơ học bằng tổng hợp LLM đọc board gộp: 0.54 → 0.72 (p=0.022), ≈ no_shard, ở 22k thay vì 64k token.",
  { x: 0.85, y: 5.6, w: 5.5, h: 0.8, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0 });
// improvement 2
card(s, 6.75, 1.6, 5.95, 4.9, WHITE);
chip(s, 7.0, 1.85, 4.6, "CẢI TIẾN 2 · OpenRCA Telecom THẬT", CORAL);
s.addText("De-collapse log-opinion pool", { x: 7.0, y: 2.35, w: 5.45, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: CORAL, margin: 0 });
s.addText([
  { text: "Telecom = root là TÍN HIỆU YẾU bị át: tín hiệu mạnh nhất KHÔNG phải root ở 20/23 ca.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Log-opinion pool quá tự tin → sụp về 1 'nạn nhân' (hậu nghiệm ≈0.9), đẩy 22/23 root ra khỏi đáp án.", options: {} },
], { x: 7.0, y: 2.85, w: 5.5, h: 1.6, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
// big stat
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 4.55, w: 5.45, h: 1.05, rectRadius: 0.08, fill: { color: "E9F7F4" }, line: { type: "none" } });
s.addText([
  { text: "0.043 → 0.130", options: { bold: true, color: TEAL } },
  { text: "  ×3", options: { bold: true, color: CORAL } },
], { x: 7.0, y: 4.6, w: 5.45, h: 0.55, align: "center", fontFace: HEAD, fontSize: 26, margin: 0 });
s.addText("Component Hit@1 (de-collapse)", { x: 7.0, y: 5.15, w: 5.45, h: 0.35, align: "center", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
s.addText("Trung thực: chỉ 2/23 ca (p=0.5, thiếu lực); nhiễu LLM ±4% → cần re-score tất định / chạy lặp.",
  { x: 7.0, y: 5.65, w: 5.5, h: 0.75, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
pageno(s, 9);
s.addNotes("Cải tiến 1: LLM toàn cục kéo 0.54→0.72. Bài học: LLM reasoning là lever, không phải sharding. Cải tiến 2: telecom thật, de-collapse ×3. Chủ động nói giới hạn 2/23 ca + nhiễu ±4%.");

// =============================================================== SLIDE 10 — REGIME MAP
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Bản đồ chế độ telecom — hướng đi tiếp", "Simulator 3GPP-ish · chế độ alarm-flood (fan-out + nạn nhân khuếch đại)");
s.addChart(pres.charts.BAR, [{
  name: "Hit@1", labels: ["de_collapse\n(chọn tín hiệu mạnh nhất)", "topology_causal", "alarm_correlation"], values: [0.0, 0.797, 0.817],
}], { x: 0.55, y: 1.7, w: 6.7, h: 4.6, barDir: "col", chartColors: [CORAL, NAVY2, TEAL],
  valAxisMinVal: 0, valAxisMaxVal: 1.0, showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontBold: true, dataLabelFontSize: 13,
  catAxisLabelColor: INK, catAxisLabelFontSize: 10.5, valAxisLabelColor: MUTE, valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, chartArea: { fill: { color: WHITE } } });
card(s, 7.5, 1.9, 5.25, 2.15, "FBECEA");
s.addText("de_collapse HỎNG trên telecom", { x: 7.75, y: 2.1, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: CORAL, margin: 0 });
s.addText("Về 0.0 khi victim_amp ≥ 1.8 — chọn nhầm 'nạn nhân' khi triệu chứng bị khuếch đại (ngược với OpenRCA).",
  { x: 7.75, y: 2.55, w: 4.8, h: 1.4, fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
card(s, 7.5, 4.25, 5.25, 2.25, "E9F7F4");
s.addText("Topology / alarm-correlation THẮNG", { x: 7.75, y: 4.45, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: TEAL, margin: 0 });
s.addText("0.80+ và robust. Bản đồ γ giải thích OpenRCA (tín hiệu tập trung) và ĐẢO NGƯỢC kết luận cho telecom thật → đây là hướng phát triển.",
  { x: 7.75, y: 4.9, w: 4.8, h: 1.5, fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
pageno(s, 10);
s.addNotes("Synthetic — nói rõ. Payoff: giải thích OpenRCA + dự báo telecom. Trên alarm-flood: de_collapse hỏng, topology/alarm thắng 0.80+. INVERTS kết luận → hướng đi tiếp.");

// =============================================================== SLIDE 11 — DEMO (dark divider)
s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(pres.shapes.OVAL, { x: -1.5, y: 3.8, w: 6, h: 6, fill: { color: TEAL, transparency: 82 }, line: { type: "none" } });
chip(s, 0.9, 1.5, 2.2, "DEMO TRỰC TIẾP", TEAL);
s.addText("Live Telecom RCA Demo", { x: 0.85, y: 2.1, w: 11.5, h: 0.9, fontFace: HEAD, fontSize: 42, bold: true, color: WHITE, margin: 0 });
s.addText("OpenRCA Telecom case chạy live gpt-4o-mini · streaming 8 stage · label-safe",
  { x: 0.9, y: 3.05, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 15, italic: true, color: "9FB8CC", margin: 0 });
const watch = [
  ["Runtime task ẩn nhãn", "scoring_points + component/reason thật bị ẩn tới sau prediction."],
  ["5 tác tử cô lập → blackboard", "findings có cấu trúc theo modality, xem tab Evidence."],
  ["no_interaction ≡ full", "đổi System → kết quả GIỐNG HỆT: demo tự minh hoạ p=1.0."],
  ["Evaluator lộ sau cùng", "chấm protocol OpenRCA + Usage token = chi phí đã cân bằng."],
];
watch.forEach((wch, i) => {
  const x = 0.9 + (i % 2) * 6.0;
  const y = 3.85 + Math.floor(i / 2) * 1.55;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.7, h: 1.35, rectRadius: 0.08, fill: { color: NAVY2, transparency: 25 }, line: { color: "2E4A6B", width: 1 } });
  s.addText(wch[0], { x: x + 0.25, y: y + 0.18, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: TEAL, margin: 0 });
  s.addText(wch[1], { x: x + 0.25, y: y + 0.6, w: 5.25, h: 0.65, fontFace: BODY, fontSize: 12, color: "CDE0EE", margin: 0 });
});
s.addNotes("Chuyển sang trình duyệt. Chạy row 0 (strict pass 1.00). Theo runbook 02. Nếu lỗi mạng → video phao hoặc Deterministic fallback. Nhấn no_interaction≡full ngay trên demo.");

// =============================================================== SLIDE 12 — CONCLUSION + limits
s = pres.addSlide(); s.background = { color: WHITE };
title(s, "Kết luận & giới hạn");
card(s, 0.6, 1.5, 12.1, 1.7, NAVY);
s.addText([
  { text: "Đóng góp không phải một hệ đa tác tử vượt trội, mà là một ", options: { color: "CDE0EE" } },
  { text: "kết luận phản chứng có kỷ luật", options: { bold: true, color: TEAL } },
  { text: " + công cụ để đạt tới nó. ", options: { color: "CDE0EE" } },
  { text: "Biết khi nào KHÔNG dùng một kỹ thuật đắt (MAS tốn 3–7× compute) có giá trị vận hành trực tiếp.", options: { color: WHITE, bold: true } },
], { x: 0.95, y: 1.5, w: 11.4, h: 1.7, valign: "middle", fontFace: BODY, fontSize: 16, margin: 0 });

s.addText("Giới hạn (nói thẳng)", { x: 0.6, y: 3.45, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: CORAL, margin: 0 });
s.addText([
  { text: "Cỡ mẫu nhỏ: n=50 và telecom 23 ca có nhãn → hiệu ứng nhỏ thiếu lực.", options: { bullet: true, breakLine: true } },
  { text: "De-collapse robust nhưng chỉ 2/23 ca (p=0.5) — cần thêm ca, không ép bằng chạy lại (pseudo-replication).", options: { bullet: true, breakLine: true } },
  { text: "Nhiễu LLM ±4% ngay ở temp=0 → cần cache-only replay / chạy lặp để claim hiệu ứng nhỏ.", options: { bullet: true } },
], { x: 0.6, y: 3.85, w: 6.05, h: 2.6, fontFace: BODY, fontSize: 12.5, color: INK, paraSpaceAfter: 7, margin: 0 });

s.addText("Hướng phát triển", { x: 6.95, y: 3.45, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: TEAL, margin: 0 });
s.addText([
  { text: "Đưa topology-aware causal reasoning làm LÕI (thay reranker hậu kỳ tắt mặc định).", options: { bullet: true, breakLine: true } },
  { text: "Thay product-of-experts naive bằng hợp nhất nhận biết tương quan thực sự.", options: { bullet: true, breakLine: true } },
  { text: "Mở rộng ca telecom có nhãn (OpenRCA 2.0, TN-RCA530, Viettel OSS) để đủ lực thống kê.", options: { bullet: true } },
], { x: 6.95, y: 3.85, w: 5.75, h: 2.6, fontFace: BODY, fontSize: 12.5, color: INK, paraSpaceAfter: 7, margin: 0 });
pageno(s, 12);
s.addNotes("Tóm đóng góp. Chủ động liệt kê 3 giới hạn TRƯỚC khi bị hỏi. 3 hướng phát triển. Thái độ: trung thực = điểm mạnh.");

// =============================================================== SLIDE 13 — CLOSING
s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(pres.shapes.OVAL, { x: 8.8, y: -2.5, w: 7, h: 7, fill: { color: NAVY2, transparency: 50 }, line: { type: "none" } });
s.addShape(pres.shapes.OVAL, { x: 9.8, y: 4.0, w: 5, h: 5, fill: { color: TEAL, transparency: 82 }, line: { type: "none" } });
s.addText("“Khoa học không phải chứng minh mình đúng,", { x: 0.9, y: 2.0, w: 11.3, h: 0.7, fontFace: HEAD, fontSize: 28, bold: true, color: WHITE, margin: 0 });
s.addText("mà kiểm tra xem mình có đúng không.”", { x: 0.9, y: 2.7, w: 11.3, h: 0.7, fontFace: HEAD, fontSize: 28, bold: true, color: TEAL, margin: 0 });
s.addText("Em đã tự bác bỏ claim đẹp ban đầu của chính mình để đưa ra một bản đồ trung thực về khi nào phân rã đa tác tử có — và không có — giá trị.",
  { x: 0.9, y: 3.75, w: 10.8, h: 0.9, fontFace: BODY, fontSize: 15, color: "CDE0EE", margin: 0 });
s.addShape(pres.shapes.LINE, { x: 0.95, y: 4.8, w: 4.2, h: 0, line: { color: TEAL, width: 2 } });
s.addText("Xin cảm ơn hội đồng — em sẵn sàng nhận câu hỏi phản biện.",
  { x: 0.9, y: 5.0, w: 11, h: 0.5, fontFace: BODY, fontSize: 17, bold: true, color: WHITE, margin: 0 });
s.addText("Hoàng Tấn Phúc  ·  ShardRCA  ·  VDT2026 DSAI", { x: 0.9, y: 5.6, w: 11, h: 0.4, fontFace: BODY, fontSize: 12, color: "8FB2CC", margin: 0 });
s.addNotes("Câu chốt = câu mở. Nhấn: em tự bác bỏ claim của chính mình. Mời câu hỏi tự tin. Xem 01_QA_DEFENSE.md.");

pres.writeFile({ fileName: path.join(__dirname, "..", "ShardRCA_VDT2026_Defense.pptx") })
  .then((f) => console.log("wrote", f));
