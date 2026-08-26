# Các model đang dùng — LegalIR + LegalQA

> Cập nhật 2026-08-25. Nguồn kết quả retrieval: `docs/submission_log_task1.md`.
> Khi nộp chính thức, nhóm cần khóa revision/commit SHA và đối chiếu lại giấy phép/model card.

## Pipeline chính thức hiện tại

| Vai trò | Model/công cụ | Cấu hình | Tham số xấp xỉ |
|---|---|---|---:|
| Dense anchor | `AITeamVN/Vietnamese_Embedding` | pretrained, không prefix, max sequence length 2048, RRF weight 1.0 | 568M |
| Dense bổ trợ | `mainguyen9/vietlegal-harrier-0.6b` | pretrained, query instruction pháp luật, max sequence length 1024, RRF weight 0.2 | 596M |
| Sparse | BM25 (`rank_bm25`) | top 100, RRF weight 0.1 | 0 |
| Sinh câu trả lời | `AITeamVN/Vi-Qwen2-1.5B-RAG` | local greedy generation | 1.5B |

RRF dùng `rrf_k=60`, top 100 từ mỗi nguồn, sau đó gộp chunk thành document bằng điểm tối đa.
Không dùng reranker và không dùng dedup/backfill.

- Task 1 public score của ensemble: **0.9208**.
- Tổng model retrieval: khoảng **1.16B**.
- Tổng retrieval + LLM Task 2: khoảng **2.66B**, dưới ngưỡng 4B nếu hai task được tính chung.

## Model đã thử nhưng không dùng trong pipeline chính thức

| Model/hướng | Kết luận |
|---|---|
| `models/vietnamese-legal-embedding-ft-full` từ `bqbbao6/VN_legal_embedding_512` | Thấp hơn AITeamVN pretrained |
| Fine-tune lại `AITeamVN/Vietnamese_Embedding` | Giảm public score; nghi catastrophic forgetting |
| `AITeamVN/Vietnamese_Embedding_v2` | Thấp hơn v1 |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | Không bổ trợ được ensemble chính |
| `BAAI/bge-reranker-v2-m3` và reranker fine-tune | Làm giảm kết quả qua nhiều phép thử |
| Dedup/backfill, entity/citation injection | Không cải thiện cấu hình chính thức |

Chi tiết toàn bộ thí nghiệm và submission ID nằm trong `docs/submission_log_task1.md`.

## Candidate LLM dự phòng, chưa dùng

- `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`
- `ntphuc149/ViLegalQwen2.5-1.5B-Base`
- `ntphuc149/ViLegalQwen3-1.7B-Base`

Không tự ý thay model trong submission chính thức nếu chưa kiểm tra quy định cho phép của BTC,
giấy phép, tổng tham số và chất lượng trên validation.
