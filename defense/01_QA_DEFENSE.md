# ShardRCA — Ngân hàng câu hỏi phản biện & đáp án

> Tiêu chí "Đánh giá phần bảo vệ" có **trọng số 3 (cao nhất)**: chấm *tính đúng đắn & đầy đủ khi trả lời câu hỏi* + *kỹ năng trình bày*.
> Nguyên tắc vàng: **thừa nhận giới hạn TRƯỚC khi bị chỉ ra, rồi biến nó thành đóng góp.** Không phòng thủ, không giấu.

---

## NHÓM A — Câu "hiểm" nhất (phải trả lời hoàn hảo)

### A1. "Hệ đa tác tử của bạn KHÔNG thắng tác tử đơn. Vậy đề tài này có giá trị gì?"
**Đây là câu quyết định điểm. Trả lời:**

> "Đúng, và đó chính là **đóng góp** chứ không phải thất bại. Đa số công trình 2024–2026 khoe multi-agent RCA thắng, nhưng **không ai kiểm soát ngân sách tính toán**. Tôi là người đặt câu hỏi đó và trả lời nó một cách có kỷ luật: dưới cân bằng token, ưu thế MAS biến mất. Kết quả này **nhất quán với MAST (NeurIPS 2025)** và các công trình 2026 cho thấy MAS thắng chủ yếu là *compute artifact*. Giá trị của tôi là **ba thứ**: (1) một quy trình đánh giá trung thực (tiền đăng ký, ghép cặp, cân bằng ngân sách); (2) khung chẩn đoán γ dự báo *đúng* khi nào phân rã có lợi; (3) hai cải tiến có căn cứ — tầng LLM toàn cục kéo hệ lên 0.72, và de-collapse nâng định vị telecom ×3. Một kết quả phản chứng *có kiểm chứng* mạnh hơn một con số đẹp *không kiểm chứng*."

Chốt: **"Khoa học không phải là chứng minh mình đúng, mà là kiểm tra xem mình có đúng không."**

### A2. "Nếu MAS không giúp, tại sao lại xây cả một hệ MAS phức tạp?"
> "Vì câu hỏi 'MAS có giúp không' **chỉ trả lời được nếu tôi xây MAS đàng hoàng và so công bằng**. Tôi không xây để nó thắng bằng mọi giá; tôi xây để *kiểm định giả thuyết*. Và hệ vẫn còn giá trị: cùng bộ artifact đó cho phép tôi cô lập *từng cơ chế* (Bảng 2) và chỉ ra chính xác cái nào trơ (phản biện chéo), cái nào hại (phân mảnh), cái nào cứu được (LLM toàn cục). Không có hệ đầy đủ + ablation thì không có phát hiện này."

### A3. "Bạn dùng gpt-4o-mini — model yếu. Model mạnh hơn (GPT-4o, Claude) có đổi kết luận không?"
> "Câu hỏi rất hay. Kết luận của tôi là về **cơ chế phân rã**, không phải về sức mạnh model. Model mạnh hơn nâng *cả* tác tử đơn *lẫn* MAS — nó không tạo ra lợi thế *tương đối* cho MAS, vì rào cản không phải năng lực suy luận mà là **cấu trúc bài toán**: trên RCAEval tín hiệu *tập trung* (γ-test cho thấy top-B toàn cục giữ root tốt hơn ở mọi γ). Thực ra model mạnh hơn còn *bất lợi* cho luận điểm MAS, vì nó làm tác tử đơn càng gần trần. Đây cũng là lý do cân bằng ngân sách quan trọng: nó tách 'MAS giúp' khỏi 'nhiều compute hơn giúp'."

### A4. "Cỡ mẫu quá nhỏ (n=50, telecom 23 ca). Kết luận có đáng tin không?"
> "Tôi chủ động báo cáo giới hạn này — đó là lý do tôi dùng **McNemar ghép cặp** (mạnh hơn so sánh độc lập vì kiểm soát độ khó từng ca) và tính công suất. Với phản chứng chính, số liệu *đủ mạnh* ở đúng chỗ cần: no_shard thắng full **p=0.013** (có ý nghĩa), no_interaction ≡ full **p=1.0** (bằng chứng mạnh cho 'trơ' vì giống *hệt* từng ca, không phải chỉ trung bình gần nhau). Ngược lại, với đóng góp *tích cực* nhỏ (de-collapse 2/23 ca, p=0.5) tôi nói thẳng là **thiếu lực thống kê** và cần thêm dữ liệu — tôi không thổi phồng. Chính sự phân biệt 'chỗ nào đủ lực, chỗ nào chưa' là dấu hiệu đánh giá trung thực."

### A5. "LLM không tất định — làm sao tin kết quả tái lập được?"
> "Tôi đo trực tiếp nhiễu này: **±4% run-to-run ngay ở temperature=0**. Vì biết vậy, các kết luận nhỏ tôi dùng **re-score tất định** (cố định board đã cache, chỉ đổi bước quyết định) hoặc chạy lặp — chứ không dựa một lần chạy live. Còn phản chứng chính (no_shard, no_interaction) thì robust vượt nhiễu: no_interaction giống *hệt* full trên cả 50 ca. Việc tôi *chỉ ra* nhiễu ±4% và thiết kế quanh nó chính là phần methodology đóng góp — nhiều paper RCA claim hiệu ứng nhỏ trên 1 lần chạy live mà không kiểm tra điều này."

---

## NHÓM B — Kỹ thuật chi tiết

### B1. "Log-opinion pool / product-of-experts là gì, tại sao nó 'quá tự tin'?"
> "Log-opinion pool hợp nhất bằng cách cộng log-xác suất của các tác tử: `log P(c,r) ∝ Σ w·log P_worker`. PoE chỉ đúng khi các nguồn **độc lập có điều kiện**. Nhưng các tác tử cô lập của tôi **không độc lập** — chúng cùng quan sát *triệu chứng lan truyền* của một nguyên nhân. Khi nhiều tác tử cùng 'thấy' một nạn nhân ồn ào, PoE nhân các xác suất lại → **sụp về hậu nghiệm ≈0.9 cho nạn nhân đó**, đẩy root thật (tín hiệu yếu) ra ngoài. Đó là lý do de-collapse (bỏ qua winner sụp, lấy board-strongest) cứu được ×3 trên telecom."

### B2. "Sharding của bạn có phải map-reduce/chia-để-trị thông thường không? Khác gì?"
> "Khác ở tầng **peer interaction** — tác tử không chỉ chia việc rồi gộp, mà *đọc và phản biện* giả thuyết của nhau rồi hiệu chỉnh hậu nghiệm. Nhưng — và đây là phát hiện trung thực — thực nghiệm cho thấy tầng đó **trơ** trên các benchmark này (p=1.0). Về lý thuyết chia-để-trị: công trình 2506.16411 chứng minh divide-and-conquer *gây hại* khi có **phụ thuộc chéo giữa các chunk**, mà RCA đúng là bài toán phụ thuộc chéo (root ở shard này, triệu chứng ở shard khác) — khớp với việc no_shard thắng."

### B3. "'Evidence-isolated' — cô lập bằng chứng — lợi ích lý thuyết là gì?"
> "Giả thuyết: cô lập buộc mỗi tác tử grounding trên bằng chứng cục bộ, giảm ảo giác, và cho phép song song vượt cửa sổ ngữ cảnh. Điều kiện để nó *thật sự* lợi là γ>1 VÀ tín hiệu phân tán. γ-test (Bảng 4) cho thấy trên RCAEval điều kiện thứ hai **sai** — tín hiệu tập trung nên cô lập chỉ pha loãng ngân sách. Đó là ranh giới tôi vẽ ra: cô lập lợi ở *distributed-evidence regime*, mà RCAEval không phải."

### B4. "Candidate catalog 'an toàn nhãn' — chứng minh không leak nhãn?"
> "Catalog là tập (component × reason) khả dĩ **suy từ cấu trúc telemetry quan sát được** — các thành phần xuất hiện trong metrics/logs/traces + họ nguyên nhân chuẩn hoá (CPU, memory, delay...). Planner/agent/fusion **chỉ** đọc telemetry + ontology chung + catalog này, **tuyệt đối không** đọc scoring_points hay nhãn vàng. Evaluator panel chỉ lộ *sau* khi prediction chốt. Mọi finding phải trỏ về bằng chứng trong shard hợp lệ — có kiểm tra validity. Trong demo, bạn thấy runtime task **ẩn** `scoring_points` và nhãn thật."

### B5. "Metric-prior là gì? Có phải bạn 'gian lận' để vượt BARO?"
> "Ngược lại — tôi **tự bóc trần** nó. Metric-prior là một heuristic an toàn nhãn ưu tiên thành phần có bất thường metric mạnh. Khi tôi *tách* nó ra (Bảng 3): prior nâng tác tử đơn 0.00→0.75; bật prior thì full 0.80 ≈ đơn 0.75 (p=1.0). Nên tôi *sửa lại câu claim*: 'một prior an toàn nhãn vượt BARO; MAS không thêm gì có ý nghĩa'. Việc tách và báo cáo minh bạch này là điều một đánh giá gian lận sẽ không bao giờ làm."

### B6. "Tác tử đơn = 0.00 khi tắt prior (Bảng 3) — nghe vô lý, giải thích?"
> "0.00 đó **không** phải vì tác tử đơn kém bản chất, mà vì nó bị **bóp ngân sách công cụ** trên RE2-TT: bảng metric rất rộng, với ngân sách hẹp nó không quét hết để tìm tín hiệu. Bằng chứng: `no_shard` (đọc gộp, ngân sách rộng) đạt **0.88** trên chính RE2-TT prior-off. Nên câu chuyện nhất quán: cho tác tử đơn đủ ngân sách → nó mạnh; chênh lệch là *ngân sách*, không phải *số lượng tác tử*."

### B7. "Falsifier của bạn có phải falsification theo Popper không?"
> "Không, và tôi gọi đúng tên trong báo cáo: nó là **evidence-based reranker top-vs-runner-up**, không phải falsification Popperian. Nó so ứng viên #1 và #2 bằng bằng chứng, không phải chứng minh sai một cách logic. Tôi cố ý không thổi phồng thuật ngữ."

---

## NHÓM C — Câu về phương pháp & khoa học

### C1. "Preregistration (tiền đăng ký) là gì và tại sao quan trọng?"
> "Là đóng băng holdout test + quyết định phân tích **trước khi thấy nhãn**, lưu thành artifact (`prereg_*.json`). Nó chống **overfit-guard** và p-hacking: bạn không thể thử nhiều cấu hình rồi chọn cái đẹp trên test. Tôi có bằng chứng cụ thể vì sao cần: claim *cũ* '0.60 vs 0.20 p=0.008' (từ validation) **không tái lập** trên holdout tiền đăng ký n=50 → nếu không prereg tôi đã báo cáo một con số ảo."

### C2. "γ = N(x)/B_eff — đo N(x) và B_eff cụ thể thế nào?"
> "`B_eff` = context window trừ đi tokens của system prompt + tool schema + reasoning scaffold (đo được). `N(x)` = kích thước chuẩn hoá telemetry của ca (số điểm dữ liệu × chi phí token/điểm). γ là **tỉ số chẩn đoán** — tôi không cần nó chính xác tuyệt đối, chỉ cần nó *dự báo đúng dấu*: γ≤1 → single đủ; γ≫1 nhưng tín hiệu tập trung → single vẫn thắng (đã kiểm chứng ở γ=271). Nó là la bàn, không phải thước micromet."

### C3. "Same-budget test (Bảng 4) thiết kế thế nào để công bằng?"
> "Cố định **mọi thứ** trừ *cách phân bổ ngân sách*: cùng tập findings, cùng bộ giải mã LLM. `single` giữ top-B findings toàn cục; `shard` luân phiên B findings đều qua các shard. Nếu premise 'bằng chứng phân tán' đúng, shard phải giữ root tốt hơn. Kết quả ngược lại ở mọi γ chặt (p=6e-5). Đây là **A/B test có đối chứng sạch** cho đúng câu hỏi lý thuyết."

### C4. "Kết quả synthetic telecom (regime map) — có phải bạn tự bịa data để ra kết quả đẹp?"
> "Tôi minh bạch đây là **synthetic** và dùng nó *không phải để claim thắng*, mà để **giải thích + dự báo**. Simulator dựng theo topology 3GPP với fan-out propagation + victim amplification. Payoff khoa học không vòng vo: bản đồ *giải thích* OpenRCA (tín hiệu tập trung → de_collapse ổn) và *dự báo* telecom thật (fan-out + victim khuếch đại → topology/alarm-correlation thắng 0.80+, de_collapse hại). Nó cần dữ liệu thật (TN-RCA530/Viettel OSS) xác nhận — tôi ghi rõ caveat này."

### C5. "Vì sao không dùng dữ liệu Viettel thật? Đề tài cho Viettel mà?"
> "TN-RCA530 (RCA cảnh báo viễn thông thật) là **gated** — không tải công khai được, tôi đã email tác giả chưa có hồi đáp. OpenRCA Telecom là telemetry vận hành thật gần nhất tôi tiếp cận được (51 ca, hàng chục GB), và tôi đã chạy trên đó. Hướng phát triển rõ ràng: cắm dữ liệu Viettel OSS vào pipeline đã sẵn sàng (4 cơ chế đã cài) — đây là bước tiếp theo tự nhiên, không phải lỗ hổng."

---

## NHÓM D — Câu về sản phẩm / demo / kỹ thuật hệ thống

### D1. "Demo chạy live thật hay quay sẵn?"
> "Live thật — server Python stdlib stream Server-Sent Events, gọi gpt-4o-mini thật qua OpenAI-compatible endpoint. Runtime task **ẩn nhãn**; evaluator panel chỉ lộ sau prediction. Bạn chọn case + system + reasoning mode ngay tại chỗ." *(Có bản cached phòng lỗi mạng — xem `02_DEMO_RUNBOOK.md`.)*

### D2. "Chi phí/độ trễ vận hành thực tế?"
> "shardrca_full ≈ 63k token/ca, độ trễ ~vài chục giây (nhiều LLM call). no_shard chỉ 20k token và *chính xác hơn* — đó là lý do tôi kết luận full **không đáng chi phí** trên các benchmark này. Đây là con số vận hành thật, không phải ước lượng."

### D3. "Kiến trúc scale được không cho mạng thật hàng nghìn node?"
> "Pipeline dựng để scale theo shard (song song, chunked pandas cho RAM hạn chế), và γ chính là công cụ quyết định *khi nào* cần thêm shard. Nhưng tôi trung thực: bằng chứng hiện tại nói trên benchmark hiện có, thêm shard *chưa* giúp — nên trước khi scale phải xác nhận regime distributed-evidence (telecom thật). Đó là điều γ giúp quyết định *trước*, tiết kiệm compute."

### D4. "Có bao nhiêu test? Chất lượng code?"
> "~115 test pass (`make test`), gồm test cho de-collapse, fusion, evaluator an toàn nhãn. Runner ghép cặp + audit artifacts + Makefile/Docker + demo UI. Toàn bộ chạy live LLM, không có fallback suy luận tất định giả (chỉ simulator data là tất định)."

---

## NHÓM E — Câu "bẫy" / thái độ

### E1. "Đề tài của bạn nghe như một bài phê bình hơn là một sản phẩm."
> "Nó là **cả hai**: một sản phẩm hoàn chỉnh (hệ MAS chạy live + demo + benchmark suite) VÀ một kết luận khoa học từ sản phẩm đó. Trong kỹ thuật, biết *khi nào KHÔNG dùng* một kỹ thuật đắt đỏ (MAS tốn 3–7× compute) có giá trị vận hành trực tiếp — nó cứu tiền và cứu độ trễ. γ cho ta biết *trước* khi nào nên bỏ tiền cho MAS."

### E2. "Nếu làm lại, bạn sẽ làm khác gì?"
> "Ba điều: (1) đặt **topology-aware causal reasoning làm lõi** ngay từ đầu (thay vì reranker hậu kỳ tắt mặc định) — dữ liệu telecom cho thấy đó mới là đòn bẩy; (2) thay PoE naive bằng hợp nhất nhận biết tương quan *ngay*, vì overconfidence là lỗi lớn nhất; (3) đầu tư sớm vào dữ liệu telecom thật có nhãn thành phần để đủ lực thống kê. Nhưng quy trình đánh giá thì tôi giữ nguyên — nó chính là thứ cứu tôi khỏi báo cáo kết quả ảo."

### E3. "Điểm bạn tự hào nhất?"
> "Việc tôi *tự bác bỏ* claim ban đầu của chính mình. Tôi từng có con số '0.60 vs 0.20, p=0.008' trông rất đẹp. Group-A confirmatory cho thấy nó không tái lập, và tôi đã **viết lại toàn bộ báo cáo thành nghiên cứu phản chứng** thay vì giữ con số đẹp. Đó là kỷ luật khoa học mà tôi tự hào nhất."

### E4. "Đóng góp mới (novelty) so với TopoEvo, MAST là gì?"
> "MAST nói 'MAS hay fail' ở mức tổng quát; tôi *định lượng* nó cho RCA bằng khung γ và same-budget test — chỉ ra **cơ chế nào** fail (phản biện trơ, phân mảnh hại) và **điều kiện** (tín hiệu tập trung vs phân tán). TopoEvo giải bằng topology learning; tôi *độc lập* tái phát hiện cùng chẩn đoán 'chọn nạn nhân ồn thay vì root yếu' và chỉ ra de-collapse là đòn bẩy rẻ đầu tiên. Novelty của tôi: **γ như công cụ chẩn đoán vận hành** + **quy trình đánh giá cân bằng ngân sách cho RCA** + **regime map INVERTS OpenRCA→telecom**."

---

## Checklist thái độ khi trả lời (đọc trước khi vào phòng)
- [ ] Nghe hết câu hỏi, **nhắc lại** ý nếu cần ("Ý thầy là...?") để không lạc.
- [ ] Thừa nhận giới hạn **chủ động**, đừng đợi bị dồn.
- [ ] Mỗi câu → dẫn về **1 số liệu cụ thể** (p-value, Hit@1) → thể hiện nắm chắc.
- [ ] Không nói "em nghĩ chắc là..."; nói "số liệu cho thấy..." hoặc "em chưa kiểm chứng phần đó, nhưng...".
- [ ] Nếu không biết: "Câu hỏi hay, em chưa test điều đó; giả thuyết của em là X, và để kiểm chứng thì cần Y." — **không bịa**.
- [ ] Luôn kéo về câu chốt: **đánh giá trung thực + khung γ + 2 cải tiến có căn cứ**.
