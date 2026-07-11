# Bộ tài liệu bảo vệ ShardRCA — VDT2026 DSAI

Chuẩn bị cho buổi bảo vệ mini-project (hội đồng phản biện). Đọc theo thứ tự:

| File | Dùng để | Khi nào |
|---|---|---|
| **[ShardRCA_VDT2026_Defense.pptx](ShardRCA_VDT2026_Defense.pptx)** | Slide trình bày (13 slide, song ngữ) — có speaker notes trong Presenter View | Trình chiếu |
| [00_STUDY_GUIDE.md](00_STUDY_GUIDE.md) | Ôn tập toàn bộ dự án: kiến trúc, số liệu, quyết định thiết kế | Học thuộc trước |
| [01_QA_DEFENSE.md](01_QA_DEFENSE.md) | Ngân hàng câu hỏi phản biện + đáp án (tiêu chí trọng số 3) | Luyện Q&A |
| [02_DEMO_RUNBOOK.md](02_DEMO_RUNBOOK.md) | Kịch bản demo live + phao dự phòng | Chuẩn bị + lúc demo |
| [03_SLIDE_TALKTRACK.md](03_SLIDE_TALKTRACK.md) | Lời nói kèm từng slide + canh giờ | Tập nói |

## Lịch quan trọng (theo biểu mẫu VDT2026)
- **10/07/2026:** đơn vị gửi 20% top + 20% thấp điểm.
- **11–14/07/2026:** hội đồng phản biện, chuẩn bị slide + trình bày (**buổi bảo vệ chính**).
- **15/07/2026:** Ban CNTT tổng hợp.

## Tiêu chí chấm (thang 0–10 × trọng số)
| # | Tiêu chí | Trọng số | Vũ khí trong bộ này |
|---|---|:--:|---|
| 1 | Độ khó/phức tạp | 1 | Study guide §7 |
| 2 | Quy mô/khối lượng | 2 | Study guide §7 |
| 3 | Sáng tạo/độc đáo | 2 | Khung γ + regime map (slide 5, 8, 10) |
| 4 | Hoàn thiện sản phẩm | 2 | Demo live (slide 11) + test suite |
| 5 | **Phần bảo vệ** (trả lời + trình bày) | **3** | **01_QA_DEFENSE.md** |

## 3 việc phải làm trước ngày bảo vệ
1. **Chạy thử demo 1 lần + QUAY MÀN HÌNH** bản thành công (phao chống lỗi mạng) — xem `02_DEMO_RUNBOOK.md §0`.
2. **Học thuộc 3 con số** (Study guide §8): `0.54 vs 0.74 p=0.013` · `p=1.0` · `0.043→0.130`.
3. **Luyện to 3 câu hiểm nhất** (A1–A3 trong `01_QA_DEFENSE.md`): "hệ không thắng thì có gì hay", "sao xây MAS phức tạp", "model yếu có đổi kết luận".

## Dựng lại slide (nếu cần sửa)
```bash
cd defense/pptx_build && node build.js      # sinh lại ShardRCA_VDT2026_Defense.pptx
```
Nội dung/số liệu slide khớp `report/BaoCao_ShardRCA_VDT2026.docx` và `report/abstract.md` (nghiên cứu phản chứng).
