// Generate the VDT2026 ShardRCA report (.docx), Vietnamese, academic style, <= 6 A4 pages.
// Follows the VDT2026 template: 1 Giới thiệu, 2 Nội dung & phương pháp, 3 Kết quả thực hiện,
// 4 Đánh giá hiệu quả, 5 Kết luận, 6 Tài liệu tham khảo.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ExternalHyperlink, ImageRun,
  AlignmentType, Footer, PageNumber, convertInchesToTwip,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const FONT = "Times New Roman";
// Figures: reuse the original figure-script outputs already used by the current report.
const FIG = fs.readFileSync(path.join(__dirname, "..", "figures", "shard_rca.png"));       // Hình 1
const FIG2 = fs.readFileSync(path.join(__dirname, "..", "figures", "peer_interaction.png")); // Hình 2

// ---- helpers ---------------------------------------------------------------
const R = (text, opts = {}) => new TextRun({ text, font: FONT, size: opts.size || 22, ...opts });
const I = (text, opts = {}) => new TextRun({ text, font: FONT, italics: true, size: opts.size || 22, ...opts });

const P = (runs, opts = {}) => new Paragraph({
  children: Array.isArray(runs) ? runs : [runs],
  alignment: opts.alignment || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after == null ? 84 : Math.round(opts.after * 0.8), line: 240, ...(opts.spacing || {}) },
  ...(opts.bullet ? { bullet: { level: 0 } } : {}),
});

const H = (text) => new Paragraph({
  children: [R(text, { bold: true, size: 24 })],
  spacing: { before: 150, after: 70 }, keepNext: true,
});
const SH = (text) => new Paragraph({
  children: [R(text, { bold: true, italics: true, size: 22 })],
  spacing: { before: 90, after: 40 }, keepNext: true,
});

const cite = (n, url) => new ExternalHyperlink({
  link: url,
  children: [new TextRun({ text: `[${n}]`, font: FONT, size: 22, style: "Hyperlink" })],
});
const refEntry = (n, label, url) => new Paragraph({
  spacing: { after: 8, line: 224 },
  children: [
    R(`[${n}] `, { size: 18 }),
    new ExternalHyperlink({ link: url, children: [new TextRun({ text: label, font: FONT, size: 18, style: "Hyperlink" })] }),
  ],
});

const U = {
  rcaeval: "https://github.com/phamquiluan/RCAEval",
  openrca: "https://github.com/microsoft/OpenRCA",
  openrca2: "https://arxiv.org/abs/2606.27154",
  tnrca: "https://arxiv.org/abs/2507.18190",
  react: "https://arxiv.org/abs/2210.03629",
  sc: "https://arxiv.org/abs/2203.11171",
  rcacopilot: "https://arxiv.org/abs/2305.15778",
  dbot: "https://www.vldb.org/pvldb/vol17/p2514-li.pdf",
  mabc: "https://aclanthology.org/2024.findings-emnlp.232/",
  debate_du: "https://arxiv.org/abs/2305.14325",
  debate_liang: "https://arxiv.org/abs/2305.19118",
  survey_rca: "https://arxiv.org/abs/2105.12378",
  survey_mas: "https://arxiv.org/abs/2402.01680",
  mast: "https://arxiv.org/abs/2503.13657",
  sasmas: "https://arxiv.org/abs/2604.02460",
  dnc: "https://arxiv.org/abs/2506.16411",
  topoevo: "https://arxiv.org/abs/2605.15611",
};

// table helpers
const BORDER = { style: BorderStyle.SINGLE, size: 2, color: "AAAAAA" };
const cell = (w, runs, opts = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: opts.fill ? { type: ShadingType.CLEAR, color: "auto", fill: opts.fill } : undefined,
  margins: { top: 34, bottom: 34, left: 88, right: 88 },
  children: [new Paragraph({ spacing: { after: 0, line: 236 }, children: Array.isArray(runs) ? runs : [runs] })],
});
const trow = (cells) => new TableRow({ children: cells });
const mkTable = (widths, rows) => new Table({
  columnWidths: widths,
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER, insideHorizontal: BORDER, insideVertical: BORDER },
  rows,
});
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 36, after: 120 },
  children: [R(t, { italics: true, size: 18 })] });

// ---- title block -----------------------------------------------------------
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [R("ShardRCA: Khi nào cô lập bằng chứng đa tác tử LLM giúp ích", { bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
    children: [R("cho phân tích nguyên nhân gốc rễ? Một nghiên cứu phản chứng có kỷ luật", { bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [ R("Hoàng Tấn Phúc", {}), R(" (sinh viên), ", {}), R("Nguyễn Duy Anh, Đặng Anh Quân", {}), R(" (mentor - người hướng dẫn)", {}), R("¹", { size: 16, superScript: true }) ] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
    children: [R("¹ Viettel Networks – hoangtanphuc05@gmail.com", { italics: true, size: 18 })] }),
];

// ---- body ------------------------------------------------------------------
const body = [];

// TÓM TẮT
body.push(new Paragraph({ spacing: { after: 50 }, children: [R("TÓM TẮT", { bold: true, size: 22 })] }));
body.push(P([
  I("Phân tích nguyên nhân gốc rễ – RCA trong mạng viễn thông 5G phải xử lý dữ liệu quan trắc đa phương thức khổng lồ. Xu hướng gần đây dùng hệ đa tác tử (multi-agent) LLM; báo cáo này đặt câu hỏi phản biện: "),
  I("một hệ đa tác tử cô lập bằng chứng có thực sự tốt hơn một tác tử đơn cho RCA không, và nhờ cơ chế nào?", { bold: true }),
  I(" Chúng tôi hiện thực "),
  I("ShardRCA", { bold: true }),
  I(" (phân mảnh telemetry → tác tử điều tra cô lập → phản biện chéo → hợp nhất log-opinion pool → xác minh bằng chứng) và đánh giá nó dưới một giao thức "),
  I("tiền đăng ký, ghép cặp, cân bằng ngân sách tính toán", { bold: true }),
  I(". Kết quả chính là "),
  I("phản chứng", { bold: true }),
  I(": trên holdout RCAEval-Hard mới (n=50), khi cân bằng ngân sách, ShardRCA "),
  I("không", { bold: true }),
  I(" vượt tác tử đơn có ý nghĩa thống kê (0.54 so với 0.42–0.44; p ≈ 0.24); vòng phản biện chéo "),
  I("không đổi một quyết định nào", { bold: true }),
  I(" (giống hệt bản bỏ phản biện); và phân mảnh bằng chứng "),
  I("gây hại", { bold: true }),
  I(": một tác tử đọc gộp không phân mảnh đạt 0.74, cao hơn có ý nghĩa (p = 0.013) ở 1/3 chi phí. Ưu thế trên BARO chỉ đến từ một tiên nghiệm định lượng heuristic, không phải từ hệ đa tác tử. Phép thử cùng-ngân-sách còn cho thấy: ngay cả khi ép "),
  I("γ = N(x)/B"), I("eff", { subScript: true }),
  I(" > 1, tóm lược toàn cục vẫn giữ nguyên nhân gốc tốt hơn phân bổ theo shard — vì trên RCAEval bằng chứng tập trung, không phân tán. Về mặt xây dựng, thay tầng hợp nhất cơ học bằng một "),
  I("tầng quyết định LLM toàn cục", { bold: true }),
  I(" kéo hệ phân mảnh lên 0.72 ≈ tác tử đọc gộp; và trên telemetry viễn thông thật (OpenRCA), nơi nguyên nhân gốc là "),
  I("tín hiệu yếu bị triệu chứng lan truyền át", { bold: true }),
  I(", việc gỡ hiện tượng quá tự tin của log-opinion pool nâng định vị thành phần từ 0.043 lên 0.130 (Hit@1) — dù cỡ mẫu còn nhỏ. Khung γ vì thế được xác nhận như một công cụ "),
  I("chẩn đoán", { bold: true }),
  I(" đúng đắn, còn premise 'đa tác tử luôn tốt hơn' thì không được dữ liệu ủng hộ."),
], { after: 150 }));

// 1. GIỚI THIỆU CHUNG
body.push(H("1. GIỚI THIỆU CHUNG"));
body.push(P([
  R("Mạng viễn thông 5G và các hệ thống vận hành hiện đại sinh ra dữ liệu quan trắc (telemetry) khổng lồ và đa dạng: chuỗi thời gian KPI (chỉ số hiệu năng thiết yếu) như CPU, RAM, băng thông, độ trễ, log (nhật ký) hệ thống, và trace (dấu vết) mô tả quan hệ phụ thuộc RPC (lệnh gọi hàm từ xa) giữa hàng trăm thành phần. Khi sự cố xảy ra, kỹ sư vận hành phải xác định nhanh "),
  R("nguyên nhân gốc rễ (root cause)", { bold: true }),
  R(" trong biển dữ liệu này; chẩn đoán sai hoặc chậm đều trực tiếp gây gián đoạn dịch vụ và thiệt hại lớn."),
]));
body.push(P([
  R("Một agent (tác tử) LLM đơn "),
  cite(5, U.react),
  R(" gặp hai giới hạn khi tự động hoá RCA. Thứ nhất, telemetry vượt xa cửa sổ ngữ cảnh nên phải cắt bỏ hoặc tóm lược, dễ đánh rơi bằng chứng quyết định. Thứ hai, khi bị nhồi quá nhiều dữ liệu, mô hình dễ đưa ra chẩn đoán nghe hợp lý nhưng ungrounded (không có căn cứ). Từ đó, câu hỏi nghiên cứu trung tâm của đề tài là: "),
  R("một hệ đa tác tử có thực sự tốt hơn một tác tử đơn cho RCA hay không, và nếu có thì nhờ cơ chế nào?", { bold: true }),
  R(" Đề tài không mặc định rằng nhiều tác tử thì tốt hơn, mà đi tìm điều kiện và cơ chế khiến việc phân rã đa tác tử đem lại giá trị."),
]));
body.push(P([
  R("Để trả lời câu hỏi đó, tôi xây dựng "),
  R("ShardRCA", { bold: true }),
  R(" — một hệ đa tác tử LLM tự trị, "),
  R("evidence-isolated (cô lập bằng chứng)", { bold: true }),
  R(". Mỗi tác tử chỉ quan sát một phân mảnh (shard) telemetry, tự hình thành giả thuyết, phản biện chéo với các tác tử khác, rồi được hợp nhất có kiểm soát để đưa ra nguyên nhân gốc kèm cơ sở. Hệ thống được chạy trực tiếp bằng LLM (gpt-4o-mini) và được thử nghiệm trên các benchmark RCA công khai."),
]));
// Related work
body.push(P([
  R("Công trình liên quan. ", { bold: true, italics: true }),
  R("RCA cho hệ vi dịch vụ và AIOps đã được khảo sát rộng rãi "),
  cite(12, U.survey_rca),
  R(", từ phát hiện bất thường tới suy diễn nhân quả trên đồ thị phụ thuộc. Gần đây, LLM được đưa vào RCA: RCACopilot "),
  cite(7, U.rcacopilot),
  R(" thu thập ngữ cảnh chẩn đoán cho sự cố cloud, D-Bot "),
  cite(8, U.dbot),
  R(" chẩn đoán cơ sở dữ liệu bằng LLM kèm tri thức vận hành, còn mABC "),
  cite(9, U.mabc),
  R(" đề xuất cộng tác/bỏ phiếu đa tác tử cho RCA vi dịch vụ. Song song, hướng multi-agent debate "),
  cite(10, U.debate_du),
  R(" "),
  cite(11, U.debate_liang),
  R(" và khảo sát về hệ đa tác tử dựa trên LLM "),
  cite(13, U.survey_mas),
  R(" cho thấy phản biện chéo cải thiện tính đúng đắn. ShardRCA thừa hưởng tinh thần này nhưng đặt thêm ràng buộc vận hành RCA: cô lập bằng chứng theo phân mảnh, con trỏ bằng chứng bắt buộc, và giao thức đánh giá an toàn nhãn."),
]));
body.push(P(R("Đóng góp chính của đề tài gồm bốn điểm:", { bold: true }), { after: 30 }));
body.push(P(R("một giao thức đánh giá RCA đa tác tử tiền đăng ký, ghép cặp và cân bằng ngân sách tính toán, cho đầu ra kiểm toán được;"), { bullet: true, after: 30 }));
body.push(P(R("một kết luận phản chứng có kỷ luật: dưới cân bằng ngân sách, cô lập bằng chứng và phản biện chéo không giúp RCA vượt tác tử đơn trên RCAEval, và phân mảnh còn gây hại;"), { bullet: true, after: 30 }));
body.push(P(R("một khung chẩn đoán γ = N(x)/Beff dự báo đúng khi nào phân rã có lợi, được kiểm chứng bằng phép thử cùng-ngân-sách;"), { bullet: true, after: 30 }));
body.push(P(R("và hai cải tiến có căn cứ: tầng quyết định LLM toàn cục (đưa hệ phân mảnh về ngang tác tử đọc gộp) và gỡ hiện tượng quá tự tin của log-opinion pool (nâng định vị thành phần trên telemetry viễn thông thật)."), { bullet: true, after: 90 }));
body.push(P([
  R("Vai trò của sinh viên: ", { bold: true }),
  R("trực tiếp thiết kế kiến trúc và cài đặt toàn bộ pipeline (planner, tác tử điều tra cô lập, tầng tương tác peer, hợp nhất, tái xếp hạng nhân quả, xác minh bằng chứng), xây dựng bộ công cụ đánh giá cùng các adapter dữ liệu và công cụ trực quan hoá."),
]));

// 2. NỘI DUNG VÀ PHƯƠNG PHÁP
body.push(H("2. NỘI DUNG VÀ PHƯƠNG PHÁP"));
body.push(SH("2.1. Tổng quan kiến trúc"));
body.push(P([
  R("Hình 1 mô tả đường đi suy luận của ShardRCA. Từ một ca sự cố và danh mục ứng viên (component (thành phần) × reason (nguyên nhân)) an toàn nhãn, "),
  R("Planner", { bold: true }),
  R(" chia khả quan sát thành các shard theo modality (phương thức dữ liệu) × nhóm thành phần × cửa sổ thời gian. Các "),
  R("tác tử điều tra cô lập", { bold: true }),
  R(" chạy song song, mỗi tác tử chỉ thấy shard của mình và ghi findings (phát hiện) có cấu trúc lên một blackboard (bảng tin chung). Tiếp theo là vòng "),
  R("tương tác peer", { bold: true }),
  R(", rồi "),
  R("hợp nhất bằng log-opinion pool", { bold: true }),
  R(", tái xếp hạng nhân quả và tinh chỉnh có mục tiêu, và cuối cùng là tầng "),
  R("xác minh bằng chứng top-vs-runner-up", { bold: true }),
  R(" trước khi chốt nguyên nhân gốc."),
]));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 50, after: 34 },
  children: [new ImageRun({ data: FIG, type: "png", transformation: { width: 352, height: 478 } })] }));
body.push(caption("Hình 1. Kiến trúc ShardRCA — RCA đa tác tử cô lập bằng chứng."));
body.push(P([
  R("Candidate catalog (danh mục ứng viên) là tập các cặp (component, reason) khả dĩ — được suy ra từ cấu trúc telemetry quan sát được: các thành phần xuất hiện trong metrics (các số liệu đo lường), logs, traces cùng một tập họ nguyên nhân chuẩn hoá (ví dụ quá tải CPU, nghẽn mạng, giới hạn kết nối). Planner dùng danh mục này để lập kế hoạch shard: mỗi shard gắn với một modality, một nhóm thành phần, và một cửa sổ thời gian quanh thời điểm sự cố, sao cho các shard rời nhau và không tác tử nào phải đọc dữ liệu ngoài phạm vi của mình."),
]));

body.push(SH("2.2. Phát biểu bài toán"));
body.push(P([
  R("Một ca RCA được mô hình hoá thành đầu vào "),
  I("x = (G, t"), I("0", { subScript: true }), I(", M, L, T, A, C) ∈ 𝒳"),
  R(". Trong đó "),
  I("G"), R(" là đồ thị thành phần/phụ thuộc, "),
  I("t"), I("0", { subScript: true }), R(" là thời điểm phát hiện sự cố, "),
  I("M"), R(" là chuỗi thời gian KPI đa biến, "),
  I("L"), R(" là tập log đã gắn thời gian, "),
  I("T"), R(" là trace/RPC hoặc dependency events (các sự kiện phụ thuộc), "),
  I("A"), R(" là cảnh báo, và "),
  I("C"), R(" là danh mục ứng viên sinh từ telemetry quan sát được. Không gian đầu ra là "),
  I("𝒴 = 𝒞 × ℛ"),
  R(", với mỗi nhãn "),
  I("y = (c, r)"),
  R(" gồm thành phần lỗi "),
  I("c ∈ 𝒞"),
  R(" và họ nguyên nhân "),
  I("r ∈ ℛ"),
  R(" (ví dụ CPU, memory (bộ nhớ), disk (ổ đĩa), delay (độ trễ), loss (mất gói), socket (cổng kết nối)). Hệ chạy một chính sách xếp hạng "),
  I("π: 𝒳 → Δ(𝒴)"),
  R(", trả về hậu nghiệm trên các cặp component × reason và chọn "),
  I("ŷ = argmax"), I("y", { subScript: true }), I(" π(y|x)"),
  R("."),
]));
body.push(P([
  R("Tôi định nghĩa quy trình chẩn đoán sự cố của hệ thống là bài toán xếp hạng dưới các ràng buộc nghiêm ngặt về tài nguyên: hệ cần đưa ra danh sách xếp hạng các nguyên nhân lỗi tiềm năng sao cho giảm thiểu sai số chẩn đoán "),
  I("ℓ"), I("rank", { subScript: true }),
  R(" (đo bằng Hit@1, Hit@3, MRR), đồng thời khống chế lượng token, số lần gọi công cụ và độ trễ dưới các ngân sách "),
  I("B"), I("tok", { subScript: true }), I(", B"), I("call", { subScript: true }), I(", L"), I("max", { subScript: true }),
  R(". Một ràng buộc cốt lõi là "),
  R("an toàn nhãn (label-safe)", { bold: true }),
  R(": planner, tác tử và fusion chỉ dùng telemetry, ontology chung và candidate catalog sinh từ quan sát, tuyệt đối không đọc nhãn vàng hay metadata đánh giá; và mọi finding/critique phải trỏ về bằng chứng nằm trong shard hợp lệ. Các cấu hình được so sánh ghép cặp trên cùng một ca để phép đo chi phí không bị nhiễu bởi độ khó dữ liệu."),
]));

body.push(SH("2.3. Ngưỡng thông tin cần đến đa tác tử"));
body.push(P([
  R("Gọi "),
  I("B"), I("eff", { subScript: true }),
  R(" là ngân sách ngữ cảnh thực dùng cho telemetry sau khi trừ prompt hệ thống, mô tả công cụ và schema; "),
  I("N(x)"),
  R(" là kích thước chuẩn hoá của ca sự cố. Tỉ lệ vượt ngữ cảnh là "),
  I("γ(x) = N(x)/B"), I("eff", { subScript: true }),
  R(". Nếu "),
  I("γ ≤ 1"),
  R(", hoặc tồn tại bộ nén an toàn nhãn "),
  I("z"),
  R(" với "),
  I("|z| ≤ B"), I("eff", { subScript: true }),
  R(" và "),
  I("I(Y; X | Z) ≈ 0"),
  R(", thì một tác tử đơn kèm tóm lược đã gần đủ thông tin. Phân rã "),
  R("có cơ sở", { bold: true }),
  R(" khi "),
  I("γ > 1"),
  R(" và các thông điệp shard giữ nhiều thông tin về Y hơn một bản nén toàn cục cùng ngân sách — điều kiện đòi hỏi tín hiệu quyết định phải "),
  R("phân tán", { bold: true }),
  R(". Mục 4.4 kiểm định trực tiếp bất đẳng thức này; và quan trọng, khi tín hiệu "),
  R("tập trung", { bold: true }),
  R(" hoặc các shard trùng thông tin, đa tác tử không có lợi thế — nên fusion phải báo cáo redundancy và mọi ablation phải so với tác tử đơn."),
]));

body.push(SH("2.4. Cô lập bằng chứng và bài học thiết kế"));
body.push(P([
  R("Phiên bản đầu là hội đồng nhiều persona miền (RAN, Core…) cùng nhìn một khối telemetry toàn cục; thiết kế này cho "),
  R("giá trị gia tăng gần như bằng không", { bold: true }),
  R(" so với một tác tử đơn cùng tầm nhìn (ở ~1/3 chi phí). Vì thế thiết kế chuyển sang "),
  R("phân mảnh thông tin", { bold: true }),
  R(", với giả thuyết làm việc: "),
  R("đa tác tử chỉ nên vượt tác tử đơn khi (a) dữ liệu vượt cửa sổ ngữ cảnh, hoặc (b) thông tin quyết định bị phân tán trên các phân mảnh độc lập.", { bold: true }),
  R(" Toàn bộ Mục 4 chính là phép kiểm định — và bác bỏ — giả thuyết này trên RCAEval."),
]));

body.push(SH("2.5. Tác tử điều tra cô lập"));
body.push(P([
  R("Quy trình gồm hai tầng. Tầng "),
  R("khai thác (mining)", { bold: true }),
  R(" dùng công cụ chuyên biệt theo phương thức (metric bằng pandas, log theo template/bất thường, trace dựng đồ thị phụ thuộc và độ trễ RPC) để đẩy findings có cấu trúc lên Blackboard. Sau đó các tác tử LLM cô lập lập luận trên findings để đưa ra "),
  R("hậu nghiệm cục bộ", { bold: true }),
  R(" trên tập ứng viên lỗi, kèm con trỏ bằng chứng dẫn chiếu telemetry gốc để truy vết."),
]));

body.push(SH("2.6. Tương tác ngang hàng tự trị và hiệu chỉnh hậu nghiệm"));
body.push(P([
  R("Tầng này phân biệt ShardRCA với một pipeline chia nhỏ thông thường. Sau vòng cục bộ, mỗi tác tử công bố giả thuyết của mình, đọc giả thuyết của các peer, rồi phát một thông điệp phản biện gồm phán quyết "),
  R("support/challenge (ủng hộ/phản bác)", { bold: true }),
  R(", con trỏ bằng chứng, và một lý do ngắn. Dựa trên các phản biện nhận được, tác tử "),
  R("hiệu chỉnh chính phân phối hậu nghiệm", { bold: true }),
  R(" của mình. Mỗi thông điệp được kiểm tra tính hợp lệ, chỉ được dùng con trỏ bằng chứng có thật trong shard cục bộ; khi LLM trả về dữ liệu sai định dạng, hệ dùng một đường lui tất định để vẫn tạo ra phản biện hợp lệ."),
]));
body.push(P([
  R("Mỗi thông điệp gồm phán quyết (support/challenge/abstain), cặp (component, reason), điểm ủng hộ/phản bác và con trỏ bằng chứng — cô đọng và kiểm toán được theo từng bước. Hình 2 minh hoạ về mặt cơ chế: hậu nghiệm ban đầu gần hoà giữa hai ứng viên, sau phản biện có bằng chứng thì nguyên nhân gốc vượt lên."),
]));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 50, after: 34 },
  children: [new ImageRun({ data: FIG2, type: "png", transformation: { width: 486, height: 211 } })] }));
body.push(caption("Hình 2. Cơ chế hiệu chỉnh hậu nghiệm sau một vòng tương tác peer (minh hoạ khái niệm)."));
body.push(P([
  I("Lưu ý trung thực: dù về mặt cơ chế vòng phản biện có thể lật một quyết định như minh hoạ, đánh giá thực nghiệm ghép cặp (Mục 4.2) cho thấy trên các benchmark khảo sát, vòng này "),
  I("không đổi một quyết định nào", { bold: true }),
  I(" so với bản bỏ phản biện — một phát hiện chính của báo cáo."),
], { after: 90 }));

body.push(SH("2.7. Hợp nhất bằng log-opinion pool"));
body.push(P([
  R("ShardRCA hợp nhất các hậu nghiệm đã hiệu chỉnh bằng một "),
  R("log-opinion pool", { bold: true }),
  R(" (product-of-experts (tích các ý kiến chuyên gia)). Với mỗi ứng viên (component c, reason r), log-xác suất hợp nhất là tổng có trọng số của log-xác suất từ các tác tử, "),
  I("log P(c,r) ∝ Σ"), I("w", { subScript: true }), I(" w"), I("pool", { subScript: true }), I(" · log P"), I("w", { subScript: true }), I("(c,r)"),
  R(", sau đó chuẩn hoá bằng softmax theo nhiệt độ. Product-of-experts chỉ đúng khi các nguồn độc lập có điều kiện, nhưng các tác tử cô lập không độc lập hoàn toàn: chúng cùng quan sát triệu chứng lan truyền của một nguyên nhân, nên sai số tương quan. Cơ chế tuỳ chọn đo tương quan Pearson dương trung bình "),
  I("r̄"),
  R(", ước lượng số chuyên gia độc lập hiệu dụng "),
  I("N"), I("eff", { subScript: true }), I(" = 1 + (N − 1)(1 − ρ·r̄)"),
  R(", và áp trọng số "),
  I("N"), I("eff", { subScript: true }), I("/N"),
  R(" để tránh đếm trùng cùng một tín hiệu gốc."),
]));
body.push(P([
  I("Minh bạch: cấu hình xác nhận không fit các trọng số này (ρ = 0, temperature = 1), nên pool rút về product-of-experts trọng số bằng nhau — chính hiện tượng quá tự tin với nguồn tương quan mà Mục 4.6 chỉ ra là lỗi chính trên telemetry viễn thông."),
]));

body.push(SH("2.8. Tái xếp hạng nhân quả, tinh chỉnh và xác minh bằng chứng"));
body.push(P([
  R("Ba tầng muộn tinh lọc câu trả lời: "),
  R("(i)", { bold: true }),
  R(" tái xếp hạng nhân quả theo độ phủ topology và thứ tự thời gian; "),
  R("(ii)", { bold: true }),
  R(" tinh chỉnh có mục tiêu (một vòng hợp nhất thứ hai khi hai ứng viên quá sát); "),
  R("(iii)", { bold: true }),
  R(" xác minh top-vs-runner-up (một evidence-based reranker, không phải falsification Popperian). Trong cấu hình xác nhận, topology_gamma = beta = 0 nên hai tầng nhân quả là no-op — một lựa chọn mà Mục 4.5–4.6 cho thấy chính là điểm cần bật lại và tăng cường cho telemetry viễn thông."),
]));

// 3. KẾT QUẢ THỰC HIỆN
body.push(H("3. KẾT QUẢ THỰC HIỆN"));
body.push(P(I("Phần này trình bày sản phẩm và bộ công cụ; đánh giá định lượng ở Mục 4."), { after: 80 }));
body.push(SH("3.1. Hệ thống end-to-end và benchmark thực"));
body.push(P([
  R("ShardRCA là prototype chạy trực tiếp bằng LLM (gpt-4o-mini): nhận một ca sự cố, chạy toàn bộ pipeline ở Hình 1, và trả về "),
  R("component, reason, thời điểm", { bold: true }),
  R(" kèm transcript kiểm toán được; mọi tầng bật–tắt qua tham số (Bảng 1). Đề tài xây dựng adapter cho "),
  R("RCAEval", { bold: true }), R(" "), cite(1, U.rcaeval),
  R(" (735 ca trên ba hệ vi dịch vụ, nhiều họ lỗi) và "),
  R("OpenRCA Telecom", { bold: true }), R(" "), cite(2, U.openrca),
  R(" (telemetry vận hành đa nguồn hàng chục GB), cùng bản tái lập BARO để đối chiếu."),
]));
body.push(SH("3.2. Runner ghép cặp và đầu ra kiểm toán được"));
body.push(P([
  R("Một "),
  R("runner ghép cặp", { bold: true }),
  R(" chạy mọi hệ trên cùng tập ca với cùng bộ công cụ, ghi token/gọi-công-cụ/độ-trễ, và sinh artifact đầy đủ (phân phối từng tác tử, transcript phản biện, ứng viên sau hợp nhất, topology, kết quả xác minh) để mọi kết luận truy vết được về bằng chứng. Vì đánh giá theo cặp trên từng ca, bộ công cụ hỗ trợ "),
  R("paired McNemar", { bold: true }),
  R(", bootstrap CI, và tính công suất/cỡ mẫu — điều kiện để báo cáo trung thực cả tín hiệu lẫn giới hạn thống kê."),
]));

// 4. ĐÁNH GIÁ HIỆU QUẢ VÀ THIẾT KẾ THỰC NGHIỆM
body.push(H("4. ĐÁNH GIÁ HIỆU QUẢ VÀ THIẾT KẾ THỰC NGHIỆM"));

body.push(SH("4.1. Công cụ, baseline và thiết kế ablation"));
body.push(P([
  R("Hệ dùng LLM (gpt-4o-mini) qua function-calling; baseline là tác tử đơn ReAct "),
  cite(5, U.react),
  R(" và ReAct + self-consistency "),
  cite(6, U.sc),
  R(", dùng chung bộ công cụ. Mỗi tầng có một cấu hình ablation để cô lập đóng góp (Bảng 1). Đặc biệt, chúng tôi bổ sung "),
  R("single_equal_tokens", { bold: true }),
  R(" (tác tử đơn với ngân sách mở rộng) làm đối chứng parity, để phép so độ chính xác không bị lệch bởi chênh lệch chi phí."),
]));
const C1 = [2100, 2500, 4900];
body.push(mkTable(C1, [
  trow([
    cell(C1[0], R("Cấu hình", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C1[1], R("Vô hiệu hoá", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C1[2], R("Đo / cô lập điều gì", { bold: true, size: 20 }), { fill: "E7EEF7" }),
  ]),
  trow([ cell(C1[0], R("shardrca_full", { size: 20 })), cell(C1[1], R("—", { size: 20 })), cell(C1[2], R("Hệ đầy đủ (mốc tham chiếu)", { size: 20 })) ]),
  trow([ cell(C1[0], R("no_interaction", { size: 20 })), cell(C1[1], R("Vòng tương tác peer", { size: 20 })), cell(C1[2], R("Giá trị của phản biện chéo và hiệu chỉnh hậu nghiệm", { size: 20 })) ]),
  trow([ cell(C1[0], R("no_falsifier", { size: 20 })), cell(C1[1], R("Bộ xác minh bằng chứng", { size: 20 })), cell(C1[2], R("Giá trị của kiểm tra top-vs-runner-up", { size: 20 })) ]),
  trow([ cell(C1[0], R("no_refinement", { size: 20 })), cell(C1[1], R("Vòng tinh chỉnh thứ hai", { size: 20 })), cell(C1[2], R("Giá trị của tinh chỉnh có mục tiêu", { size: 20 })) ]),
  trow([ cell(C1[0], R("no_shard", { size: 20 })), cell(C1[1], R("Phân mảnh bằng chứng", { size: 20 })), cell(C1[2], R("Giá trị của bản thân việc cô lập, so với một tác tử đơn", { size: 20 })) ]),
  trow([ cell(C1[0], R("no_topology", { size: 20 })), cell(C1[1], R("Tái xếp hạng nhân quả", { size: 20 })), cell(C1[2], R("Sanity check; mặc định gamma=0, beta=0 nên chỉ ý nghĩa khi bật causal weights", { size: 20 })) ]),
]));
body.push(caption("Bảng 1. Các cấu hình ablation và mục tiêu cô lập tương ứng."));

body.push(SH("4.2. Phản chứng dưới cân bằng ngân sách và cô lập cơ chế (RCAEval-Hard, n=50)"));
body.push(P([
  R("Chúng tôi đóng băng một holdout RCAEval-Hard tiền đăng ký (n=50, an toàn nhãn) và chạy ghép cặp năm cấu hình, tắt mọi heuristic augmentation. Ba đối chứng quyết định: "),
  R("(a)", { bold: true }),
  R(" single_equal_tokens (một tác tử ReAct với ngân sách công cụ mở rộng) để loại nhiễu do chênh lệch chi phí; "),
  R("(b)", { bold: true }),
  R(" no_shard (một tác tử đọc gộp toàn bộ, kèm self-consistency) để tách giá trị của bản thân việc phân mảnh; "),
  R("(c)", { bold: true }),
  R(" no_interaction để tách giá trị của phản biện chéo. Kết quả (Bảng 2) là "),
  R("phản chứng và nhất quán với văn bản khoa học gần đây", { bold: true }),
  R(" "),
  cite(14, U.mast), R(" "), cite(15, U.sasmas),
  R(": khi cân bằng ngân sách, ShardRCA không vượt tác tử đơn có ý nghĩa (Δ ≈ +0.10–0.12, p ≈ 0.24–0.33) dù tốn 3.6–7.7× token; vòng phản biện chéo cho kết quả "),
  R("giống hệt", { bold: true }),
  R(" bản bỏ nó (p = 1.0) — tức không đổi một quyết định nào; và một tác tử đọc gộp không phân mảnh (no_shard) "),
  R("thắng có ý nghĩa", { bold: true }),
  R(" (0.74 so với 0.54; p = 0.013) ở 1/3 chi phí. Phân mảnh một bài toán có phụ thuộc chéo giữa các shard đúng là chế độ mà chia-để-trị gây hại "),
  cite(16, U.dnc),
  R("."),
]));
const C2 = [2650, 1300, 1450, 3050];
body.push(mkTable(C2, [
  trow([
    cell(C2[0], R("Hệ (RCAEval-Hard, n=50)", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C2[1], R("Hit@1", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C2[2], R("token/ca", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C2[3], R("So với shardrca_full (McNemar ghép cặp)", { bold: true, size: 20 }), { fill: "E7EEF7" }),
  ]),
  trow([ cell(C2[0], R("shardrca_full (mốc)", { bold: true, size: 20 })), cell(C2[1], R("0.540", { size: 20 })), cell(C2[2], R("63.6k", { size: 20 })), cell(C2[3], R("—", { size: 20 })) ]),
  trow([ cell(C2[0], R("single_react_sc (tác tử đơn)", { size: 20 })), cell(C2[1], R("0.420", { size: 20 })), cell(C2[2], R("17.9k", { size: 20 })), cell(C2[3], R("Δ +0.12; p = 0.24 (không ý nghĩa)", { size: 20 })) ]),
  trow([ cell(C2[0], R("single_equal_tokens (parity ngân sách)", { size: 20 })), cell(C2[1], R("0.440", { size: 20 })), cell(C2[2], R("8.3k", { size: 20 })), cell(C2[3], R("Δ +0.10; p = 0.33 (không ý nghĩa)", { size: 20 })) ]),
  trow([ cell(C2[0], R("no_shard (đọc gộp, không phân mảnh)", { bold: true, size: 20 })), cell(C2[1], R("0.740", { bold: true, size: 20 })), cell(C2[2], R("20.4k", { size: 20 })), cell(C2[3], R("full THUA: Δ −0.20; p = 0.013", { bold: true, size: 20 })) ]),
  trow([ cell(C2[0], R("no_interaction (bỏ phản biện chéo)", { size: 20 })), cell(C2[1], R("0.540", { size: 20 })), cell(C2[2], R("≈0 (tất định)", { size: 20 })), cell(C2[3], R("giống hệt full: p = 1.0", { size: 20 })) ]),
]));
body.push(caption("Bảng 2. Đối chứng cân bằng ngân sách + cô lập cơ chế trên RCAEval-Hard. Đọc gộp thắng phân mảnh; phản biện chéo trơ."));

body.push(SH("4.3. “Vượt BARO” đến từ tiên nghiệm định lượng, không phải hệ đa tác tử (RE2-TT)"));
body.push(P([
  R("Trên đúng 20 ca RE2-TT của phép so BARO, chúng tôi tách riêng đóng góp của metric-prior bằng cách bật/tắt nó (Bảng 3). Khi "),
  R("tắt prior", { bold: true }),
  R(", tác tử đơn đạt 0.00 (do bị bóp ngân sách, không quét hết bảng metric rộng) còn ShardRCA đạt 0.60; nhưng khi "),
  R("bật prior", { bold: true }),
  R(", một tác tử đơn cũng đạt 0.75 ≈ ShardRCA 0.80 (Δ +0.05; p = 1.0). Bản thân prior nâng tác tử đơn từ 0.00 lên 0.75 (p = 6·10⁻⁵). Nói trung thực: câu 'ShardRCA vượt BARO' nên đọc là "),
  R("'một tiên nghiệm định lượng an toàn nhãn vượt BARO; hệ đa tác tử không thêm gì có ý nghĩa lên trên'", { bold: true }),
  R("."),
]));
const C3 = [4200, 2050, 2050];
body.push(mkTable(C3, [
  trow([
    cell(C3[0], R("Hệ (RE2-TT, n=20) — Hit@1", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C3[1], R("prior TẮT", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C3[2], R("prior BẬT", { bold: true, size: 20 }), { fill: "E7EEF7" }),
  ]),
  trow([ cell(C3[0], R("shardrca_full", { size: 20 })), cell(C3[1], R("0.60", { size: 20 })), cell(C3[2], R("0.80", { size: 20 })) ]),
  trow([ cell(C3[0], R("tác tử đơn (single_react_sc / equal_tokens)", { size: 20 })), cell(C3[1], R("0.00", { size: 20 })), cell(C3[2], R("0.75", { size: 20 })) ]),
  trow([ cell(C3[0], R("BARO (official, AC@1)", { size: 20 })), cell(C3[1], R("0.55", { size: 20 })), cell(C3[2], R("—", { size: 20 })) ]),
]));
body.push(caption("Bảng 3. Tách metric-prior trên RE2-TT: prior gánh gần hết; full so tác tử đơn khi prior bật là p = 1.0."));

body.push(SH("4.4. Ngay ở chế độ vượt ngữ cảnh (γ > 1), phân mảnh vẫn không giữ nguyên nhân gốc tốt hơn"));
body.push(P([
  R("Để kiểm định trực tiếp premise của khung γ, chúng tôi so cùng-ngân-sách trên 24 ca rộng nhất: cùng một tập findings và cùng bộ giải mã LLM, chỉ khác cách phân bổ ngân sách B findings — "),
  I("single"), R(" giữ top-B toàn cục, "),
  I("shard"), R(" luân phiên qua các shard. Bảng 4 cho thấy ngay cả khi γ tới 271, "),
  R("tóm lược toàn cục giữ nguyên nhân gốc TỐT HƠN", { bold: true }),
  R(" (retention 0.83 so với 0.21) và thắng Hit@1 có ý nghĩa (p = 6·10⁻⁵). Lý do: trên RCAEval, nguyên nhân gốc có tín hiệu mạnh "),
  R("tập trung", { bold: true }),
  R(", nên top-B toàn cục luôn bắt được; phân bổ đều theo shard lại pha loãng ngân sách vào các shard nhiễu. Vậy điều kiện 'bằng chứng quyết định phân tán qua các shard' — tiền đề để phân rã có lợi — "),
  R("không đúng trên RCAEval", { bold: true }),
  R("."),
]));
const C4 = [1000, 1250, 2700, 2400, 950];
body.push(mkTable(C4, [
  trow([
    cell(C4[0], R("B", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C4[1], R("γ", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C4[2], R("Retention single / shard", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C4[3], R("Hit@1 single / shard", { bold: true, size: 20 }), { fill: "E7EEF7" }),
    cell(C4[4], R("p", { bold: true, size: 20 }), { fill: "E7EEF7" }),
  ]),
  trow([ cell(C4[0], R("4", { size: 20 })), cell(C4[1], R("271", { size: 20 })), cell(C4[2], R("0.83 / 0.21", { size: 20 })), cell(C4[3], R("0.83 / 0.21", { size: 20 })), cell(C4[4], R("6e-5", { size: 20 })) ]),
  trow([ cell(C4[0], R("8", { size: 20 })), cell(C4[1], R("135", { size: 20 })), cell(C4[2], R("0.92 / 0.21", { size: 20 })), cell(C4[3], R("0.83 / 0.21", { size: 20 })), cell(C4[4], R("6e-5", { size: 20 })) ]),
  trow([ cell(C4[0], R("16", { size: 20 })), cell(C4[1], R("68", { size: 20 })), cell(C4[2], R("0.96 / 0.42", { size: 20 })), cell(C4[3], R("0.83 / 0.42", { size: 20 })), cell(C4[4], R("0.013", { size: 20 })) ]),
  trow([ cell(C4[0], R("32", { size: 20 })), cell(C4[1], R("34", { size: 20 })), cell(C4[2], R("1.0 / 1.0", { size: 20 })), cell(C4[3], R("0.96 / 1.0", { size: 20 })), cell(C4[4], R("1.0", { size: 20 })) ]),
]));
body.push(caption("Bảng 4. Cùng-ngân-sách single vs shard (24 ca rộng nhất). Tóm lược toàn cục giữ root tốt hơn ở mọi γ chặt."));

body.push(SH("4.5. Xây dựng: tầng quyết định LLM toàn cục làm hệ phân mảnh cạnh tranh trở lại"));
body.push(P([
  R("Chẩn đoán lỗi cho thấy hệ chọn dịch vụ kề topology mang triệu chứng lan truyền thay vì nguyên nhân gốc — đúng như "),
  cite(17, U.topoevo),
  R(". Thay tầng hợp nhất cơ học bằng một "),
  R("tầng tổng hợp LLM toàn cục", { bold: true }),
  R(" (shardrca_llmboard) đọc board đã gộp — đúng “bộ não” của tác tử đọc gộp — nâng Hit@1 từ 0.54 lên "),
  R("0.72", { bold: true }),
  R(" (vượt bản cơ học có ý nghĩa, p = 0.022) và ngang tác tử đọc gộp 0.74 (p = 1.0) ở 22k token/ca. Bài học trung thực: thứ tạo ra kết quả là "),
  R("lập luận LLM toàn cục", { bold: true }),
  R(", không phải bản thân việc phân mảnh hay phản biện chéo; sharding chỉ có lý do tồn tại khi bằng chứng thực sự vượt ngữ cảnh và phân tán."),
]));

body.push(SH("4.6. Trên telemetry viễn thông thật (OpenRCA Telecom, n=51)"));
body.push(P([
  R("Khác RCAEval, telemetry viễn thông thật đúng là chế độ mà phương pháp nhắm tới: nguyên nhân gốc là "),
  R("tín hiệu yếu bị át", { bold: true }),
  R(" — tín hiệu mạnh nhất KHÔNG phải nguyên nhân gốc ở 20/23 ca (rank trung vị ≈ 8 trong ~32 thành phần). Nhưng chẩn đoán cho thấy lỗi lớn nhất, dễ sửa nhất là log-opinion pool "),
  R("quá tự tin", { bold: true }),
  R(" (đúng như lý thuyết product-of-experts với nguồn tương quan): nó sụp về một “nạn nhân” duy nhất (hậu nghiệm ≈ 0.9), đẩy 22/23 nguyên nhân gốc thật ra khỏi đáp án. Gỡ hiện tượng sụp này — dùng thành phần có bằng chứng board mạnh nhất — nâng định vị thành phần từ "),
  R("0.043 lên 0.130 (Hit@1)", { bold: true }),
  R(", và strict Hit@1 từ 0.196 lên 0.235 (re-score tất định). Cần thẳng thắn: hiệu ứng này chỉ trên 2/23 ca (p = 0.5, thiếu lực), và một lần chạy live đơn lẻ bị "),
  R("nhiễu tái lập ±4%", { bold: true }),
  R(" (LLM không tất định ngay ở temperature = 0) che mất — muốn claim phải dùng re-score tất định hoặc chạy lặp nhiều lần. Đồ thị phụ thuộc xuyên tầng (host→container→service) dựng được từ trace nhưng việc chốt nguyên nhân gốc yếu vẫn là bài toán mở, cần suy luận topology sâu kiểu "),
  cite(17, U.topoevo),
  R("."),
]));

// 5. KẾT LUẬN
body.push(H("5. KẾT LUẬN"));
body.push(P([
  R("Đóng góp chính của đề tài không phải là một hệ đa tác tử vượt trội, mà là một "),
  R("kết luận phản chứng có kỷ luật", { bold: true }),
  R(" cùng công cụ để đạt tới nó. Dưới giao thức tiền đăng ký, ghép cặp và cân bằng ngân sách, cô lập bằng chứng đa tác tử "),
  R("không", { bold: true }),
  R(" vượt một tác tử đơn cho RCA trên RCAEval: phản biện chéo trơ, phân mảnh gây hại, và ưu thế trên BARO là do một tiên nghiệm heuristic. Phép thử cùng-ngân-sách bác bỏ trực tiếp tiền đề 'bằng chứng phân tán' ngay cả khi γ ≫ 1. Những kết quả này nhất quán với làn sóng nghiên cứu 2025–2026 cho thấy ưu thế đa tác tử phần lớn là ảo khi kiểm soát chi phí "),
  cite(14, U.mast), R(" "), cite(15, U.sasmas),
  R(". Mặt xây dựng: thứ thực sự tạo giá trị là "),
  R("lập luận LLM toàn cục", { bold: true }),
  R(" (đưa hệ phân mảnh lên 0.72 ≈ tác tử đọc gộp), còn trên telemetry viễn thông thật, gỡ hiện tượng quá tự tin của log-opinion pool nâng định vị thành phần gấp ba lần. Khung "),
  I("γ = N(x)/B"), I("eff", { subScript: true }),
  R(" vì thế nên được dùng như một "),
  R("công cụ chẩn đoán", { bold: true }),
  R(": nó dự báo đúng rằng RCAEval (bằng chứng tập trung, phụ thuộc chéo) là chế độ bất lợi cho phân mảnh."),
]));
body.push(P([
  R("Hạn chế: ", { bold: true }),
  R("(i) holdout RCAEval-Hard (n=50) và đặc biệt OpenRCA Telecom (23 ca có nhãn thành phần) đều nhỏ, nên các hiệu ứng nhỏ thiếu lực thống kê; (ii) fix de-collapse tuy robust qua nhiều lần lấy mẫu board nhưng chỉ tác động 2/23 ca (p = 0.5) — cần nhiều ca telecom hơn để đạt significance, không thể ép bằng cách chạy lại (pseudo-replication); (iii) tính tái lập phụ thuộc LLM API — đo được nhiễu run-to-run ±4% ngay ở temperature = 0, nên cache-only replay hoặc chạy lặp là bắt buộc để claim hiệu ứng nhỏ; (iv) một số kết quả (γ-regime, de-collapse strict) đo bằng re-score tất định trên board đã cache, cần được xác nhận thêm bằng nhiều lần chạy live."),
]));
body.push(P([
  R("Hướng phát triển: ", { bold: true }),
  R("(a) đưa suy luận topology xuyên tầng làm lõi (thay vì tái xếp hạng hậu kỳ) theo tinh thần "),
  cite(17, U.topoevo),
  R(" để chốt nguyên nhân gốc yếu trên telemetry viễn thông; (b) thay product-of-experts naive bằng hợp nhất nhận biết tương quan thực sự; (c) mở rộng bộ ca telecom có nhãn thành phần để đạt lực thống kê, và bổ sung OpenRCA 2.0 "),
  cite(3, U.openrca2),
  R(" cùng TN-RCA530 "),
  cite(4, U.tnrca),
  R(". Nói ngắn gọn: đóng góp lâu bền của đề tài là một quy trình đánh giá trung thực cho RCA đa tác tử, và một bản đồ rõ ràng về khi nào phân rã có — và không có — giá trị."),
]));

// 6. TÀI LIỆU THAM KHẢO
body.push(H("6. TÀI LIỆU THAM KHẢO"));
body.push(refEntry(1, "L. Pham et al. — RCAEval: Benchmark và thư viện RCA đa nguồn", U.rcaeval));
body.push(refEntry(2, "Microsoft — OpenRCA: Benchmark RCA vận hành đa phương thức", U.openrca));
body.push(refEntry(3, "OpenRCA 2.0 / PAVE — RCA liên hệ thống với chú giải đường nhân quả", U.openrca2));
body.push(refEntry(4, "TN-AutoRCA / TN-RCA530 — RCA cảnh báo mạng viễn thông theo đồ thị tri thức", U.tnrca));
body.push(refEntry(5, "S. Yao et al. — ReAct: Synergizing Reasoning and Acting in Language Models", U.react));
body.push(refEntry(6, "X. Wang et al. — Self-Consistency Improves Chain-of-Thought Reasoning", U.sc));
body.push(refEntry(7, "Y. Chen et al. — RCACopilot: Automatic Root Cause Analysis via LLMs for Cloud Incidents", U.rcacopilot));
body.push(refEntry(8, "X. Zhou et al. — D-Bot: Database Diagnosis System using LLMs (PVLDB 2024)", U.dbot));
body.push(refEntry(9, "mABC: Multi-Agent Blockchain-inspired Collaboration for RCA in Micro-Services", U.mabc));
body.push(refEntry(10, "Y. Du et al. — Improving Factuality and Reasoning through Multiagent Debate", U.debate_du));
body.push(refEntry(11, "T. Liang et al. — Encouraging Divergent Thinking in LLMs through Multi-Agent Debate", U.debate_liang));
body.push(refEntry(12, "J. Soldani, A. Brogi — Anomaly Detection and Failure RCA in (Micro)Service Cloud Applications: A Survey (ACM CSUR 2022)", U.survey_rca));
body.push(refEntry(13, "T. Guo et al. — Large Language Model based Multi-Agents: A Survey of Progress and Challenges (IJCAI 2024)", U.survey_mas));
body.push(refEntry(14, "M. Cemri et al. — Why Do Multi-Agent LLM Systems Fail? (MAST, NeurIPS 2025)", U.mast));
body.push(refEntry(15, "Single-Agent LLMs Outperform Multi-Agent Systems under Equal Thinking-Token Budgets (2026)", U.sasmas));
body.push(refEntry(16, "When Does Divide and Conquer Work for Long-Context LLM? A Noise Decomposition Framework", U.dnc));
body.push(refEntry(17, "J. Wang et al. — TopoEvo: Topology-Aware Self-Evolving Multi-Agent RCA in Microservices (2026)", U.topoevo));

// ---- document --------------------------------------------------------------
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: { page: { margin: {
      top: convertInchesToTwip(0.75), bottom: convertInchesToTwip(0.75),
      left: convertInchesToTwip(0.82), right: convertInchesToTwip(0.82),
    } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Trang ", PageNumber.CURRENT, " / ", PageNumber.TOTAL_PAGES], font: FONT, size: 18, color: "666666" })] })] }) },
    children: [...titleBlock, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "..", "BaoCao_ShardRCA_VDT2026.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
