# ShardRCA — Talk-track theo từng slide (8–10 phút)

> Lời nói kèm mỗi slide. Tập nói to 2–3 lần, canh giờ. Mỗi slide ghi thời lượng mục tiêu.
> File PPTX: `defense/ShardRCA_VDT2026_Defense.pptx` (13 slide).

---

**Slide 1 — Tiêu đề (0:20)**
> "Kính chào hội đồng. Em là Hoàng Tấn Phúc. Đề tài của em: *ShardRCA — Khi nào cô lập bằng chứng đa tác tử LLM giúp ích cho phân tích nguyên nhân gốc rễ? Một nghiên cứu phản chứng có kỷ luật.*"

**Slide 2 — Bài toán & động lực (1:00)**
> "RCA trong mạng 5G: khi sự cố, phải tìm thành phần lỗi + nguyên nhân trong hàng chục GB telemetry đa phương thức. Một agent LLM đơn gặp 2 giới hạn: dữ liệu vượt cửa sổ ngữ cảnh, và dễ chẩn đoán vô căn cứ khi bị nhồi quá tải. Xu hướng 2024–2026 là ném multi-agent vào. Nhưng em đặt câu hỏi bị bỏ quên: **nó có thực sự giúp không, hay chỉ là ảo giác do tốn nhiều token hơn?**"

**Slide 3 — Câu hỏi nghiên cứu & đóng góp (0:50)**
> "Câu hỏi trung tâm: MAS cô lập bằng chứng có vượt tác tử đơn cho RCA không, và nhờ cơ chế nào? Em không mặc định nhiều tác tử thì tốt. Bốn đóng góp: (1) quy trình đánh giá tiền đăng ký, cân bằng ngân sách; (2) kết luận phản chứng có kỷ luật; (3) khung chẩn đoán γ; (4) hai cải tiến có căn cứ."

**Slide 4 — Kiến trúc ShardRCA (1:10)** *(Hình 1)*
> "Đây là pipeline, chạy live gpt-4o-mini. Planner chia telemetry thành shard rời nhau theo modality × thành phần × thời gian. Các tác tử điều tra cô lập chạy song song — mỗi tác tử chỉ thấy shard của mình — đẩy findings lên blackboard. Rồi vòng phản biện chéo, hợp nhất log-opinion pool, và xác minh bằng chứng. Mọi bước audit được và an toàn nhãn."

**Slide 5 — Khung γ (1:00)**
> "Đây là công cụ lý thuyết trung tâm. γ = N(x) / B_eff, tỉ lệ giữa kích thước ca sự cố và ngân sách ngữ cảnh thực. Phân rã đa tác tử **chỉ** có cơ sở khi γ>1 VÀ tín hiệu quyết định **phân tán** qua các shard. Nếu tín hiệu **tập trung**, tác tử đơn đã đủ. γ là la bàn dự báo *trước* khi nào nên bỏ compute cho MAS."

**Slide 6 — Thiết kế đánh giá (0:50)**
> "Điểm mấu chốt về tính đúng đắn: em đánh giá **ghép cặp** trên cùng một ca (McNemar), **cân bằng ngân sách token**, và **tiền đăng ký** holdout trước khi thấy nhãn. Có thêm baseline `single_equal_tokens` để phép so độ chính xác không bị lệch bởi chênh chi phí. Đây là thứ 95% công trình MAS bỏ qua."

**Slide 7 — Phản chứng chính (1:20)** *(Hình 3 / Bảng 2)*
> "Kết quả trên RCAEval-Hard, n=50, cân bằng ngân sách. Ba phát hiện: Một — khi cân bằng token, ShardRCA 0.54 KHÔNG vượt tác tử đơn 0.42–0.44 có ý nghĩa, p=0.24, dù tốn 3.6–7.7× token. Hai — bỏ phản biện chéo cho kết quả **giống hệt** full, p=1.0: tầng đó trơ. Ba — một tác tử **đọc gộp** không phân mảnh đạt **0.74**, thắng full có ý nghĩa p=0.013, ở 1/3 chi phí. Phân mảnh **gây hại**."

**Slide 8 — Vượt BARO là do prior + γ-test (1:00)** *(Bảng 3–4)*
> "Còn 'vượt baseline BARO'? Em tách metric-prior ra: prior nâng tác tử đơn 0.00→0.75; bật prior thì full 0.80 ≈ đơn 0.75, p=1.0. Vậy 'vượt BARO' là do **prior**, không phải MAS. Và γ-test cùng-ngân-sách: ngay ở γ=271, tóm lược toàn cục giữ root **tốt hơn** phân bổ shard, p=6e-5 — tiền đề 'bằng chứng phân tán' sai trên RCAEval."

**Slide 9 — Hai cải tiến có căn cứ (1:10)**
> "Mặt xây dựng. Một — thay fusion cơ học bằng **tầng quyết định LLM toàn cục**: kéo hệ phân mảnh từ 0.54 lên **0.72**, ngang tác tử đọc gộp, ở 22k thay vì 64k token. Thứ tạo giá trị là *lập luận LLM toàn cục*, không phải sharding. Hai — trên **telemetry viễn thông thật** OpenRCA, root là tín hiệu yếu bị át; log-opinion pool quá tự tin sụp về nạn nhân ồn. **Gỡ sụp** nâng định vị thành phần **0.043 → 0.130**, gấp ba lần."

**Slide 10 — Bản đồ chế độ telecom (0:50)**
> "Để dự báo *khi nào* MAS/topology thắng, em dựng simulator telecom. Trên chế độ alarm-flood — fan-out + nạn nhân khuếch đại — chọn-tín-hiệu-mạnh-nhất **hỏng** (về 0 khi amp≥1.8), còn **topology-causal và alarm-correlation thắng 0.80+**. Bản đồ này giải thích OpenRCA và **đảo ngược** kết luận cho telecom thật: đây là hướng đi tiếp."

**Slide 11 — Demo (chuyển sang trình duyệt, ~2:30)**
> "Em xin demo live." → chạy theo `02_DEMO_RUNBOOK.md`.

**Slide 12 — Kết luận & giới hạn (0:50)**
> "Đóng góp không phải một hệ vượt trội, mà là **kết luận phản chứng có kỷ luật** + công cụ đạt tới nó. Giới hạn em nói thẳng: cỡ mẫu nhỏ, đóng góp de-collapse thiếu lực (2/23 ca), nhiễu LLM ±4%. Hướng phát triển: topology reasoning làm lõi, hợp nhất nhận biết tương quan, và dữ liệu telecom thật có nhãn."

**Slide 13 — Câu chốt & cảm ơn (0:30)**
> "Khoa học không phải chứng minh mình đúng, mà kiểm tra xem mình có đúng không. Em đã tự bác bỏ claim đẹp ban đầu của chính mình để đưa ra một bản đồ trung thực về *khi nào phân rã đa tác tử có — và không có — giá trị*. Em xin cảm ơn và sẵn sàng nhận câu hỏi."

---

## Canh giờ tổng
| Phần | Phút |
|---|---|
| Slide 1–6 (bối cảnh + phương pháp) | ~4:10 |
| Slide 7–10 (kết quả) | ~4:20 |
| Slide 11 demo | ~2:30 (đệm) |
| Slide 12–13 kết | ~1:20 |
| **Tổng nói** | **~8–10 phút** (chưa tính demo đệm) |

> Nếu bị nhắc rút gọn: bỏ Slide 10 (regime map) và rút demo còn 1 case. Giữ bằng mọi giá: Slide 7 (phản chứng) + Slide 9 (cải tiến).
