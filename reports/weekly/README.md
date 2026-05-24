# Báo cáo tuần — Đồ án tốt nghiệp

Slide báo cáo hàng tuần cho đồ án **Vietnamese GraphRAG over Wikipedia + Neo4j**.

## Cách dùng

```bash
make weekly-slides           # tạo reports/weekly/<YYYY>-W<WW>.md từ git log 7 ngày
make weekly-slides-pdf       # thêm bước render PDF qua npx @marp-team/marp-cli
```

Tuỳ biến khoảng thời gian:

```bash
uv run python scripts/gen_weekly_slides.py --days 14
uv run python scripts/gen_weekly_slides.py --since 2026-05-15 --until 2026-05-21
uv run python scripts/gen_weekly_slides.py --force          # ghi đè file đã có
```

## Quy trình hàng tuần

1. Chạy `make weekly-slides` cuối tuần.
2. Mở `reports/weekly/<YYYY>-W<WW>.md`, điền các block `<!-- TODO: ... -->`:
   - Tóm tắt thành quả lớn nhất.
   - Demo / screenshot (lưu vào `reports/weekly/assets/`).
   - Số liệu (coverage, EM/F1, kích thước KG, latency...).
   - Khó khăn cần GVHD góp ý.
   - Kế hoạch tuần tới.
3. `make weekly-slides-pdf` để xuất PDF nộp.
4. Commit cả `.md` lẫn `.pdf` — hoặc chỉ commit `.md` cho gọn.

## Vì sao dùng Marp

- Markdown thuần → diff git rõ ràng, dễ review.
- Một file template duy nhất, không phụ thuộc Google Slides / PowerPoint.
- `npx @marp-team/marp-cli` không cần cài global, chỉ cần Node.
- Xuất được PDF, HTML, PPTX khi cần.
