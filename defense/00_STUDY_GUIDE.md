# ShardRCA — Sổ tay ôn tập bảo vệ (VDT2026 DSAI)

> Mục tiêu: nắm chắc **mọi số liệu, mọi quyết định thiết kế, và logic khoa học** để trả lời phản biện không vấp.
> Đọc file này + `01_QA_DEFENSE.md` là đủ để defend. Thời lượng nói: 8–10 phút + Q&A.

---

## 0. Câu chốt (một câu để mở đầu và để kết thúc)

**"Tôi đặt câu hỏi phản biện — *hệ đa tác tử LLM cô lập bằng chứng có thực sự tốt hơn tác tử đơn cho RCA không, và nhờ cơ chế nào?* — rồi xây ShardRCA và trả lời nó dưới một giao thức tiền đăng ký, cân bằng ngân sách. Đóng góp của tôi không phải 'hệ của tôi thắng', mà là một **quy trình đánh giá trung thực + khung chẩn đoán γ dự báo đúng khi nào phân rã có lợi**, cộng hai cải tiến có căn cứ."**

Đây là điểm khác biệt với 95% mini-project khác: **không khoe con số, mà kiểm chứng con số**. Hội đồng DSAI (nhiều PhD) sẽ đánh giá cao thái độ khoa học này ở tiêu chí *sáng tạo* và *phần bảo vệ*.

---

## 1. Bài toán & động lực (Slide 1–2)

- **RCA (Root-Cause Analysis)** trong mạng 5G / hệ vi dịch vụ: khi sự cố xảy ra, phải tìm **thành phần lỗi (component)** + **nguyên nhân (reason)** trong biển telemetry đa phương thức: KPI time-series (CPU/RAM/latency), logs, traces (RPC dependency), alarms. Hàng trăm thành phần, hàng chục GB.
- Một **agent LLM đơn** có 2 giới hạn: (1) telemetry vượt cửa sổ ngữ cảnh → phải cắt/tóm lược → rơi bằng chứng; (2) nhồi quá nhiều dữ liệu → chẩn đoán "nghe hợp lý nhưng vô căn cứ" (ungrounded).
- **Xu hướng 2024–2026:** ném multi-agent LLM vào để giải. Câu hỏi bị bỏ qua: *nó có thực sự giúp không, hay chỉ là ảo giác do tốn nhiều token hơn?*

**Câu hỏi nghiên cứu trung tâm:** một MAS cô lập bằng chứng có vượt tác tử đơn cho RCA không, và **nhờ cơ chế nào**?

---

## 2. Kiến trúc ShardRCA (Slide 3–5) — Hình 1

Đường đi suy luận (pipeline), tất cả chạy **live gpt-4o-mini** qua function-calling:

1. **Planner / shard builder** — từ ca sự cố + candidate catalog (component × reason, sinh từ telemetry, **an toàn nhãn**), chia khả quan sát thành **shard rời nhau** theo `modality × nhóm-thành-phần × cửa-sổ-thời-gian`.
2. **Tác tử điều tra cô lập (isolated investigators)** — chạy song song, **mỗi tác tử chỉ thấy shard của mình**. Hai tầng: (a) *mining* bằng công cụ chuyên biệt (pandas cho metric, template/anomaly cho log, đồ thị phụ thuộc cho trace) → đẩy *findings* có cấu trúc lên **Blackboard**; (b) LLM lập luận trên findings → **hậu nghiệm cục bộ** trên tập ứng viên + con trỏ bằng chứng.
3. **Vòng tương tác peer (peer critique + posterior revision)** — mỗi tác tử công bố giả thuyết, đọc của peer, phát thông điệp `support/challenge` + con trỏ bằng chứng, rồi **tự hiệu chỉnh hậu nghiệm**. (Hình 2 minh hoạ cơ chế.)
4. **Hợp nhất log-opinion pool (product-of-experts)** — `log P(c,r) ∝ Σ w · log P_worker(c,r)`, softmax theo nhiệt độ. Có tuỳ chọn discount tương quan (N_eff) nhưng **mặc định tắt** (ρ=0, temp=1) để tái lập.
5. **Tái xếp hạng nhân quả** (topology coverage + temporal) — **mặc định no-op** (γ=β=0) trong cấu hình xác nhận.
6. **Tinh chỉnh + xác minh bằng chứng top-vs-runner-up** (một evidence-based reranker, **không phải** Popperian falsification — nói đúng tên để không bị bắt lỗi).
7. **Audit artifacts** — ghi lại phân phối từng tác tử, transcript, ứng viên hợp nhất, usage, provenance an toàn nhãn.

**Cấu hình ablation (Bảng 1):** `shardrca_full` (đầy đủ) · `no_interaction` (bỏ peer) · `no_falsifier` · `no_refinement` · `no_shard` (đọc gộp, không phân mảnh) · `no_topology`.

---

## 3. Khung lý thuyết γ (Slide 6) — điểm sáng tạo cốt lõi

- `B_eff` = ngân sách ngữ cảnh thực cho telemetry (sau khi trừ prompt/schema).
- `N(x)` = kích thước chuẩn hoá của ca sự cố.
- **Tỉ lệ vượt ngữ cảnh** `γ(x) = N(x) / B_eff`.

**Định lý làm việc:** Phân rã đa tác tử **chỉ có cơ sở** khi **cả hai**:
1. `γ > 1` (dữ liệu thật sự vượt cửa sổ), **VÀ**
2. thông điệp shard giữ **nhiều thông tin về Y hơn** một bản nén toàn cục cùng ngân sách → đòi hỏi **tín hiệu quyết định phân tán (distributed)**.

Nếu `γ ≤ 1`, hoặc tồn tại bản nén an toàn nhãn `z` với `I(Y; X | Z) ≈ 0`, hoặc **tín hiệu tập trung (concentrated)** → tác tử đơn đã gần đủ. γ là **công cụ chẩn đoán**: nó *dự báo trước* khi nào nên/không nên phân rã.

---

## 4. Kết quả — thuộc lòng bảng số (Slide 7–10)

### 4.1 Phản chứng chính — RCAEval-Hard, n=50, cân bằng ngân sách (Bảng 2)

| Hệ | Hit@1 | token/ca | So với `shardrca_full` (McNemar ghép cặp) |
|---|---:|---:|---|
| **shardrca_full** (mốc) | **0.540** | 63.6k | — |
| single_react_sc (tác tử đơn) | 0.420 | 17.9k | Δ+0.12, **p=0.24 ns** |
| single_equal_tokens (parity) | 0.440 | 8.3k | Δ+0.10, **p=0.33 ns** |
| **no_shard** (đọc gộp) | **0.740** | 20.4k | **full THUA: Δ−0.20, p=0.013** |
| no_interaction (bỏ peer) | 0.540 | ≈0 | **giống hệt full, p=1.0** |

**Ba phát hiện:** (1) cân bằng token → MAS **không** thắng có ý nghĩa; (2) phản biện chéo **trơ** (đổi 0 quyết định); (3) phân mảnh **có hại** (đọc gộp thắng ở 1/3 chi phí).

### 4.2 "Vượt BARO" = metric-prior, không phải MAS — RE2-TT, n=20 (Bảng 3)

| Hệ (Hit@1) | prior TẮT | prior BẬT |
|---|---:|---:|
| shardrca_full | 0.60 | 0.80 |
| tác tử đơn | 0.00 | 0.75 |
| BARO (official AC@1) | 0.55 | — |

Prior nâng tác tử đơn 0.00→0.75 (p=6e-5). Bật prior: full 0.80 vs đơn 0.75, **p=1.0**. → "vượt BARO" là do **prior an toàn nhãn**, MAS không thêm gì có ý nghĩa. (Tác tử đơn = 0.00 khi tắt prior chỉ vì **bị bóp ngân sách**, không quét hết bảng metric — không phải vì kém bản chất.)

### 4.3 γ-regime same-budget — 24 ca rộng nhất (Bảng 4)

| B | γ | Retention single/shard | Hit@1 single/shard | p |
|---:|---:|---|---|---:|
| 4 | 271 | 0.83 / 0.21 | 0.83 / 0.21 | 6e-5 |
| 8 | 135 | 0.92 / 0.21 | 0.83 / 0.21 | 6e-5 |
| 16 | 68 | 0.96 / 0.42 | 0.83 / 0.42 | 0.013 |
| 32 | 34 | 1.0 / 1.0 | 0.96 / 1.0 | 1.0 |

Ngay ở γ=271, **top-B toàn cục giữ root TỐT HƠN** phân bổ shard → tiền đề "bằng chứng phân tán" **sai trên RCAEval** (tín hiệu tập trung). γ được xác nhận là chẩn đoán đúng.

### 4.4 Xây dựng #1 — tầng LLM toàn cục (Slide 9)

`shardrca_llmboard` (thay fusion cơ học bằng tổng hợp LLM đọc board đã gộp): Hit@1 **0.54 → 0.72** (p=0.022 vs cơ học), **≈ no_shard 0.74** (p=1.0), ở 22k token/ca (vs 64k). → thứ tạo giá trị là **lập luận LLM toàn cục**, không phải sharding.

### 4.5 Xây dựng #2 — OpenRCA Telecom thật, n=51 (Slide 9)

- Telecom = chế độ **tín hiệu yếu bị át**: tín hiệu mạnh nhất KHÔNG phải root ở **20/23 ca** (rank trung vị ≈ 8 trong ~32 thành phần).
- Lỗi lớn nhất: log-opinion pool **quá tự tin** (product-of-experts với nguồn tương quan) → sụp về 1 "nạn nhân" (hậu nghiệm ≈0.9), đẩy 22/23 root ra khỏi đáp án.
- **Gỡ sụp (de-collapse)** → dùng thành phần có board-evidence mạnh nhất: component Hit@1 **0.043 → 0.130** (×3); strict **0.196 → 0.235** (re-score tất định).
- **Trung thực:** chỉ 2/23 ca (p=0.5, thiếu lực); live đơn lẻ bị **nhiễu ±4%** (LLM không tất định ngay ở temp=0). Muốn claim → cache-only replay hoặc chạy lặp.

---

## 5. Bản đồ chế độ (regime map) — telecom tổng hợp (bổ trợ, nếu bị hỏi sâu)

Simulator 3GPP-ish (core→transport→gNB→cell) + fault injection + fan-out propagation. Trên **propagating faults (alarm-flood — chế độ telecom thật):**
- `de_collapse` (chọn tín hiệu mạnh nhất): **0.0 ở amp≥1.8** — chọn nhầm "nạn nhân" khi victim bị khuếch đại.
- `topology_causal` (explanatory coverage): **0.797**, robust.
- `alarm_correlation` (coverage + earliest onset): **0.817**, tốt nhất.

→ **INVERTS** kết luận OpenRCA: ở telecom thật, topology/alarm-correlation là cơ chế đúng (0.80+), còn de_collapse có hại. Bản đồ γ giải thích OpenRCA (tín hiệu tập trung) và dự báo telecom (phân tán).

---

## 6. Những quyết định thiết kế "sao lại thế?" (hay bị hỏi)

| Quyết định | Vì sao |
|---|---|
| Dùng gpt-4o-mini, không phải model xịn | Cân bằng ngân sách + chi phí; **kết luận là về *cơ chế*, không phải về sức mạnh model** — model mạnh hơn nâng cả single lẫn MAS. |
| Mặc định tắt topology/temporal/ρ | Để cấu hình **xác nhận** tái lập được (no-fit, no-op); chúng chỉ bật khi có artifact validation tiền đăng ký riêng. Đây chính là honesty. |
| Đánh giá ghép cặp (paired) + McNemar | Loại nhiễu do độ khó dữ liệu; đo đúng hiệu ứng cơ chế trên **cùng một ca**. |
| Tiền đăng ký (preregister) holdout | Chống overfit-guard / p-hacking — quyết định trước khi thấy nhãn test. |
| Chọn kết quả phản chứng thay vì giấu | Group-A confirmatory (n=50) đã **bác** claim cũ "0.60 vs 0.20 p=0.008" (không tái lập). Báo cáo trung thực > báo cáo đẹp. |

---

## 7. Điểm mạnh để nhấn (map vào rubric)

| Tiêu chí (trọng số) | Bằng chứng để nói |
|---|---|
| Độ khó/phức tạp (1) | RCA multi-modal telemetry hàng chục GB; MAS 7 tầng; thống kê ghép cặp; 3 benchmark (RCAEval 735 ca, OpenRCA thật 51 ca, synth). |
| Quy mô/khối lượng (2) | Toàn bộ pipeline + 6 ablation + 2 baseline + runner ghép cặp + adapter 3 dataset + demo UI streaming + báo cáo học thuật + ~115 test pass. |
| Sáng tạo (2) | Khung chẩn đoán **γ = N(x)/B_eff**; same-budget A/B test; **evidence-isolated sharding**; de-collapse fix; regime map INVERTS OpenRCA→telecom. |
| Hoàn thiện sản phẩm (2) | Chạy live end-to-end, demo UI real-time SSE, audit artifacts, test suite, Makefile/Docker, báo cáo 6 trang. |
| Phần bảo vệ (3) | File `01_QA_DEFENSE.md` — trả lời được mọi câu khó, kể cả câu "hệ của bạn không thắng thì có gì hay?". |

---

## 8. Ba con số phải thuộc lòng (nếu chỉ nhớ được 3)

1. **0.54 vs 0.74** — full THUA no_shard, p=0.013 (phân mảnh có hại).
2. **p=1.0** — no_interaction ≡ full (phản biện chéo trơ).
3. **0.043 → 0.130** — de-collapse ×3 trên telecom thật (đóng góp xây dựng).
