# Script Thuyet Trinh — Capstone Review 3
**A Study and Evaluation of Graph RAG on Wikipedia Data**
FPT University SE+AI — Bao cao tong hop tu Review 1, Review 2, va ket qua cuoi ky

> Huong dan su dung: moi slide ~35-50 giay, tong thoi gian 10-12 phut.

---

## Slide 1 — Title

Kinh chao thay co va hoi dong.

Nhom chung em gom 4 thanh vien: Huynh Quoc Trung, Nhu Quang Anh, Pham Ngo Dinh Khoi, va Truong Dinh Thien.

Hom nay, chung em bao cao de tai ve GraphRAG cho bai toan hoi dap da-hop tren Wikipedia tieng Viet, tap trung vao bo du lieu, kien truc he thong, va ket qua danh gia cuoi cung.

---

## Slide 2 — Research Problem and Objectives

Van de cua chung em la: voi cau hoi da-hop tieng Viet, bang chung thuong nam o nhieu bai Wikipedia khac nhau, nen text-only retrieval de bo sot moi lien he quan trong.

Vi vay, muc tieu cua nhom co 3 diem:
- xay dung corpus Wikipedia tieng Viet co provenance ro rang,
- ket hop lexical, vector, graph, va community retrieval trong Neo4j,
- va danh gia day du chat luong cau tra loi, bang chung ho tro, abstention, do tre, va ablation.

---

## Slide 3 — Users, Applicability, and Innovation

Slide nay bo sung cac y quan trong tu Review 1 va Review 2.

Ve practical applicability, doi tuong huong den la sinh vien, nha nghien cuu, nha bao, va cac nhom AI can mot he thong truy vet bang chung bang tieng Viet.

Ve innovation, nhom nhan manh 3 diem:
- local-first GraphRAG voi evidence ID co the kiem toan,
- WRRF dung dong thoi 4 tin hieu retrieval,
- va ViWikiHop gom ca cau hoi answerable va unanswerable co quality control.

---

## Slide 4 — System Architecture

Kien truc he thong gom 3 khoi chinh.

ViWikiHop quan ly phan corpus va bang chung. Wikipedia-Neo4j dam nhan xay dung do thi, retrieval, va suy luan. Aster la lop giao dien trinh bay cau tra loi va trich dan.

Muc tieu cua kien truc nay la giu duoc tinh traceable tu du lieu goc den cau tra loi cuoi cung.

---

## Slide 5 — End-to-End Data Flow

Dong du lieu bat dau tu dump Wikipedia tieng Viet, sau do duoc loc, tach doan, sinh stable IDs, dua vao Neo4j voi cac nut Page, Chunk, Entity, va Community.

Khi co cau hoi, he thong thuc hien WRRF, rerank, va tra ve answer kem citations tren giao dien.

Diem can nhan manh la toan bo pipeline van giu duoc kha nang audit evidence.

---

## Slide 6 — Corpus Construction

Tu 1.29 trieu trang dau vao, sau nhieu buoc loc cau truc, chat luong, orphan, va deduplication, nhom giu lai 589,742 trang.

Noi dung duoc giu lai khong chi la text, ma con co wikilinks, provenance, va segment IDs on dinh de phuc vu truy vet va danh gia.

---

## Slide 7 — ViWikiHop Dataset and Quality Controls

Bo du lieu ViWikiHop cuoi cung co 1,600 mau, trong do 1,118 answerable va 482 unanswerable.

Nhom tach split train, development, va test theo cach group-aware de giam ro ri thong tin va giu kha nang danh gia cong bang hon.

Day la mot phan noi dung duoc ke thua va mo rong tu phan dataset planning o Review 1.

---

## Slide 8 — Knowledge Graph and Retrieval

Do thi tri thuc cua chung em gom Page, Chunk, cac Entity co type, va Community.

He thong retrieval khong chi tim theo text va vector, ma con su dung entity graph va community summaries de bo sung bang chung cho cau hoi da-hop.

Day la diem noi bat ve mat architecture va detail design da duoc hoi dong quan tam tu Review 2.

---

## Slide 9 — Technology Choices and Engineering Quality

Slide nay tong hop phan con thieu tu Review 2.

Ve cong nghe, nhom dung FastAPI va Python 3.12 cho API, Neo4j 5.x cho graph plus vector plus full-text, va cac model tieng Viet cho embedding, NER, reranking, va local answer generation.

Ve engineering quality, he thong dung config theo moi truong, pipeline co script tai lap, seed co dinh, kien truc retrieval theo module, va demo khong hard-code ma chay tren input that.

---

## Slide 10 — Weighted Reciprocal Rank Fusion

WRRF la co che hop nhat 4 bang xep hang retrieval thanh mot diem tong hop.

Thay vi tron raw score khac thang do, chung em hop nhat theo thu hang, sau do deduplicate theo `chunk_id`, rerank bang cross-encoder, va bo sung link expansion cho chuoi bang chung da-hop.

Day la thanh phan cot loi giup he thong vuot tung kenh retrieval don le.

---

## Slide 11 — Evaluation Protocol

Chung em danh gia theo 3 setting: gold-context QA, open-corpus retrieval, va end-to-end QA.

Cac metric gom Hit@5, MRR, support recall, chain recall, EM, F1, joint F1, abstention accuracy, va latency.

Ngoai danh gia tu dong, nhom con co human evaluation tren 120 mau voi 3 annotator co kha nang tieng Viet.

---

## Slide 12 — Final Retrieval Results

Ket qua retrieval cuoi cung cho thay WRRF + reranking dat Hit@5 = 0.711 va MRR = 0.501, cao nhat trong cac cau hinh.

So voi BM25, vector, va graph rieng le, hybrid fusion giup tang do phu bang chung ro ret. Doi lai, latency tang len 118.6 ms, nhung van o muc tot cho he thong demo va nghien cuu.

---

## Slide 13 — Ablation Study

Ablation cho thay hai thanh phan dong gop lon nhat la link expansion va reranking.

Neu bo multi-hop expansion thi Hit@5 giam 0.063. Neu bo reranking thi MRR giam 0.029 nhung latency giam dang ke.

Nhu vay, ket qua cuoi cung khong den tu mot thanh phan don le, ma tu su ket hop hop ly giua retrieval, graph structure, va ranking.

---

## Slide 14 — End-to-End QA and Answerability

O bai toan end-to-end, he thong dat Answer F1 = 0.683, Support F1 = 0.667, Joint F1 = 0.602, va abstention accuracy = 0.792.

Dieu nay cho thay retrieval da tien bo ro, nhung van con khoang cach giua tim dung bang chung va tong hop cau tra loi cuoi cung.

Day cung la ly do nhom xac dinh evidence-aware generation va abstention la huong cai thien tiep theo.

---

## Slide 15 — Human Evaluation

Danh gia thu cong cho thay muc diem tot o naturalness, answer correctness, va evidence correctness; do dong thuan giua annotator cung o muc chap nhan duoc.

Tieu chi kho nhat la hop necessity, vi day la noi rat de phat sinh tranh cai khi xac dinh mot cau hoi co that su can da-hop hay khong.

---

## Slide 16 — Error Analysis and Limitations

Bon nhom loi chinh gom: thieu mat xich bang chung, noise trong entity resolution, sinh cau tra loi chua duoc ho tro day du boi evidence, va abstention sai.

Ve gioi han he thong, viec alignment giua segment va chunk chua hoan hao, NER van co noise, va mot so phan giao dien demo van uu tien tinh minh hoa hon la production polish.

---

## Slide 17 — Contributions

Dong gop cua de tai co 4 diem:
- corpus Wikipedia tieng Viet co the tai lap va kiem toan provenance,
- ViWikiHop la bo du lieu MHQA co answerable va unanswerable,
- GraphRAG tren Neo4j voi 4 kenh retrieval thong qua WRRF,
- va bo danh gia support-aware cho phep nhin thay ro trade-off giua chat luong va latency.

---

## Slide 18 — Team Contribution

Phan cong cuoi cung cua nhom nhu sau:
- Huynh Quoc Trung: xay dung knowledge graph, NER, entity resolution, dataset generation.
- Nhu Quang Anh: WRRF retrieval, reranking, evaluation pipeline, va tong hop bao cao.
- Pham Ngo Dinh Khoi: agent reasoning, local LLM, curating multi-hop QA, va Text2Cypher experiments.
- Truong Dinh Thien: FastAPI services, job management, demo interface, va dashboard integration.

Slide nay da bo sung lai thanh vien va phan cong con thieu trong ban review 3 truoc do.

---

## Slide 19 — Conclusion

Tong ket lai, cau hinh GraphRAG day du dat Hit@5 = 0.711, MRR = 0.501, va complete-chain recall = 0.417 tren test set ViWikiHop.

Hybrid fusion vuot tung retrieval channel rieng le. Link expansion va reranking mang lai loi ich ro rang, du co trade-off ve latency. Quan trong hon, he thong giu duoc citations va provenance de cau tra loi co the kiem chung.

---

## Slide 20 — Q&A

Cam on thay co va hoi dong da lang nghe.

Nhom chung em san sang tra loi cau hoi ve dataset, retrieval, ablation, va cac huong cai thien tiep theo.
