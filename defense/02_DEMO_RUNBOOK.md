# ShardRCA — Kịch bản Demo (Live LLM + phao cached)

> Demo chạy **live gpt-4o-mini**. Rủi ro lớn nhất khi bảo vệ: mạng/API lỗi hoặc chậm. **Bắt buộc chuẩn bị phao.**

---

## 0. Chuẩn bị TRƯỚC ngày bảo vệ (làm 1 lần)

```bash
cd /home/phucht/project/vdt3

# (1) Kiểm tra key + model
grep -E "OPENAI_API_KEY|OPENAI_MODEL|OPENAI_BASE_URL" .env   # phải có key + gpt-4o-mini

# (2) Chạy thử demo end-to-end 1 lần để "làm nóng" cache và xác nhận không lỗi
make demo-ui        # mở http://127.0.0.1:8765
#   -> chọn "row 0 · protocol strict pass · expected 1.00", System=shardrca_full, Reasoning=Reported protocol
#   -> bấm Run Live, xem chạy hết 8 stage tới Score 1.00 ✓

# (3) QUAY MÀN HÌNH bản chạy thành công này (30–60s) làm PHAO.
#     Nếu lúc bảo vệ mạng lỗi -> phát video này, không đứng hình.

# (4) Phao thứ 2 (không cần mạng): Deterministic fallback mode chạy offline
#     -> trên UI chọn Reasoning = "Deterministic fallback" -> Run -> vẫn ra kết quả, KHÔNG gọi API.
```

**Case nên demo (đã whitelist trong `scripts/demo_shardrca_live.py`):**
- **row 0** — protocol strict pass, expected **1.00** ⭐ (case mở màn an toàn nhất).
- **row 1, 3, 17** — cũng strict pass 1.00 (dự phòng nếu row 0 nhiễu).
- **row 15** — partial 0.33: dùng để giải thích **field-level scoring** (component/reason/time).
- **row 12** — protocol fail 0.00: chỉ dùng nếu muốn show *failure analysis*, KHÔNG mở màn bằng case này.

---

## 1. Lệnh chạy khi bảo vệ

```bash
cd /home/phucht/project/vdt3
make demo-ui                      # = python scripts/demo_shardrca_live.py
# Mở trình duyệt: http://127.0.0.1:8765
```

Nếu cổng 8765 bận: `python scripts/demo_shardrca_live.py --port 8080`.

---

## 2. Kịch bản nói khi demo (~2–3 phút)

**Bước 1 — Nạp ca sự cố (label-safe).**
> "Đây là một ca sự cố thật từ OpenRCA Telecom. Lưu ý panel Incident: hệ **chỉ** thấy mô tả runtime — `scoring_points`, thành phần thật, nguyên nhân thật đều **bị ẩn**. Đây là ràng buộc an toàn nhãn."

**Bước 2 — Candidate catalog.**
> "Catalog các cặp (component × reason) được **suy từ cấu trúc telemetry**, không phải từ nhãn — đây là không gian tìm kiếm an toàn nhãn."

**Bước 3 — Tác tử cô lập chạy (tab Agents/Evidence).**
> "5 tác tử cô lập chạy song song trên các shard: node metrics, container metrics, service/middleware, application symptoms, trace dependencies. Mỗi tác tử chỉ thấy shard của mình và đẩy *findings* có cấu trúc lên blackboard — xem tab Evidence, findings kèm modality + hướng + score."

**Bước 4 — Peer review (tab Peer Review).**
> "Đây là tầng phân biệt ShardRCA với map-reduce thường: các tác tử phản biện chéo `support/challenge` và hiệu chỉnh hậu nghiệm. *Trung thực* — trong đánh giá ghép cặp tầng này hoá ra trơ, nhưng cơ chế thì có thật và audit được, xem transcript."

**Bước 5 — Fusion (tab Fusion).**
> "Log-opinion pool hợp nhất hậu nghiệm các tác tử thành ứng viên xếp hạng, rồi qua verifier top-vs-runner-up."

**Bước 6 — Đáp án + Evaluator (tab Evaluator).**
> "Hệ chốt component + reason + thời điểm. **Bây giờ** evaluator panel mới lộ nhãn thật và chấm theo protocol OpenRCA — Score **1.00 ✓**. Chú ý Usage: số LLM call và token — đây chính là chi phí tôi cân bằng trong đánh giá."

**Điểm nhấn khi demo (kéo về luận điểm):**
> "Nút System cho phép đổi sang `no_interaction` — nếu chạy, kết quả **giống hệt** `shardrca_full`, đúng như phát hiện phản biện chéo trơ (p=1.0). Demo này *tự minh hoạ* kết luận của báo cáo."

---

## 3. Nếu có sự cố (xử lý bình tĩnh)

| Tình huống | Xử lý |
|---|---|
| API chậm > 30s | Nói: "Đang gọi LLM thật nên có độ trễ — trong lúc chờ tôi giải thích tầng tiếp theo." Nếu quá lâu → chuyển **video phao**. |
| API lỗi / hết quota | Đổi Reasoning → **Deterministic fallback** (không gọi API, vẫn chạy pipeline) HOẶC phát **video phao**. |
| Mạng phòng bảo vệ chết | Deterministic fallback chạy hoàn toàn offline → vẫn demo được luồng. |
| Cổng bận | `--port 8080`. |
| Case ra 0.00 bất ngờ (nhiễu ±4%) | Bình tĩnh: "Đây đúng là nhiễu run-to-run ±4% mà báo cáo chỉ ra — để tôi chạy lại / đổi sang row 1." Chọn case strict-pass khác. **Biến sự cố thành minh chứng cho luận điểm về nhiễu LLM.** |

---

## 4. Câu chốt sau demo
> "Điểm mấu chốt: đây là hệ chạy **live thật**, mọi bước **audit được**, và nó **an toàn nhãn** — chính sự nghiêm ngặt này cho phép tôi đưa ra kết luận phản chứng đáng tin ở phần tiếp theo."
