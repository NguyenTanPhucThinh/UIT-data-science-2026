"""
Kiem tra xem prefix instruction bat buoc cua mainguyen9/vietlegal-harrier-0.6b co THUC SU tao khac
biet so voi khong dung hay khong - vi buoc sanity_check truoc chay VOI query_prefix='' (rong), CHUA
ap dung dung dinh dang "Instruct: {task}\nQuery: {query}" da xac nhan qua model card. Script nay
encode CUNG 1 cap cau hoi/passage 2 LAN (co va khong co instruction), so sanh ma tran similarity.

Cach dung (tu thu muc goc UIT-data-science-2026, SAU KHI da chay sanity_check_embedding_model.py
thanh cong voi model nay):
    python scripts/test_instruct_prefix.py
"""
from __future__ import annotations

MODEL_NAME = "mainguyen9/vietlegal-harrier-0.6b"
MAX_SEQ_LENGTH = 1024
INSTRUCT_TASK = "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question"

QUESTIONS = [
    "Người lao động được nghỉ phép năm bao nhiêu ngày?",
    "Doanh nghiệp cần đáp ứng điều kiện gì để được cấp giấy phép kinh doanh?",
]
PASSAGES = [
    "Điều 113. Nghỉ hằng năm: Người lao động làm việc đủ 12 tháng cho một người sử dụng lao động thì được nghỉ hằng năm, hưởng nguyên lương theo hợp đồng lao động 12 ngày làm việc.",
    "Điều 27. Điều kiện cấp Giấy chứng nhận đăng ký doanh nghiệp: Ngành, nghề đăng ký kinh doanh không bị cấm đầu tư kinh doanh; tên doanh nghiệp được đặt theo đúng quy định.",
]


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"Load {MODEL_NAME} (trust_remote_code=True, max_seq_length={MAX_SEQ_LENGTH})...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH

    passage_emb = model.encode(PASSAGES, convert_to_numpy=True, normalize_embeddings=True)

    # --- Truong hop 1: KHONG co instruction (giong dung sanity_check truoc, query_prefix='') ---
    q_emb_no_instruct = model.encode(QUESTIONS, convert_to_numpy=True, normalize_embeddings=True)
    sim_no_instruct = q_emb_no_instruct @ passage_emb.T

    # --- Truong hop 2: CO instruction dung dinh dang model card yeu cau ---
    questions_with_instruct = [f"Instruct: {INSTRUCT_TASK}\nQuery: {q}" for q in QUESTIONS]
    q_emb_instruct = model.encode(questions_with_instruct, convert_to_numpy=True, normalize_embeddings=True)
    sim_instruct = q_emb_instruct @ passage_emb.T

    np.set_printoptions(precision=4, suppress=True)
    print("\n" + "=" * 70)
    print("Ma tran similarity KHONG co instruction (hang=cau hoi, cot=passage):")
    print(sim_no_instruct)
    print(f"  Do chenh dung/sai (duong cheo - ngoai duong cheo): "
          f"{(sim_no_instruct[0,0]-sim_no_instruct[0,1]) + (sim_no_instruct[1,1]-sim_no_instruct[1,0]):.4f}")

    print("\nMa tran similarity CO instruction dung dinh dang model card:")
    print(sim_instruct)
    print(f"  Do chenh dung/sai (duong cheo - ngoai duong cheo): "
          f"{(sim_instruct[0,0]-sim_instruct[0,1]) + (sim_instruct[1,1]-sim_instruct[1,0]):.4f}")
    print("=" * 70)
    print(
        "\nDoc ket qua: 'Do chenh dung/sai' CANG CAO nghia la model phan biet CANG TOT giua passage\n"
        "dung/sai chu de cho tung cau hoi. So sanh 2 con so nay: neu ban CO instruction cho ra do chenh\n"
        "RO RET CAO HON, xac nhan prefix la BAT BUOC/QUAN TRONG (giong dung model card) - PHAI dung dung\n"
        "dinh dang nay khi encode full corpus, KHONG dung query_prefix='' nhu sanity_check truoc do.\n"
        "Neu 2 con so gan bang nhau, co the model khong nhay cam lam voi prefix - nhung van nen dung\n"
        "dung theo model card de an toan (khong co ly do gi de KHONG dung, chi ton 1 chuoi text co dinh)."
    )


if __name__ == "__main__":
    main()
