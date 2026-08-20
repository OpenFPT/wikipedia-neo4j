# Script thuyết trình - Huỳnh Quốc Trung

Tài liệu này bám theo deck `reports/review3/final-slides.pdf` sau khi rút gọn các slide trùng ý và các slide phụ.

Mục tiêu:
- Nói mạch lạc, không đọc nguyên văn slide.
- Chia phần đúng cho 3 người: Quang Anh phụ trách dataset, Khôi phụ trách ingestion + Neo4j, Trung phụ trách phần còn lại.
- Bảo đảm cover đủ mọi slide còn lại trong deck.

## Mở đầu

### Slide 1 - Title Page

### Lời thoại gợi ý
"Em xin chào thầy cô và các bạn. Em là Huỳnh Quốc Trung. Hôm nay nhóm em xin trình bày đề tài `A Study and Evaluation of Graph RAG on Wikipedia Data`, tập trung vào bài toán hỏi đáp đa bước trên Wikipedia tiếng Việt.

Phần trình bày sẽ chia làm 3 phần. Em trình bày bài toán, kiến trúc tổng quan, thiết kế retrieval và các kết quả cuối. Bạn Nhữ Quang Anh sẽ phụ trách phần dataset. Bạn Phạm Ngô Đình Khôi sẽ phụ trách phần data ingestion và Neo4j."

### Slide 2 - Table of Contents

### Lời thoại gợi ý
"Đây là bố cục chính của bài trình bày. Nhóm em sẽ đi từ bài toán, dữ liệu, quá trình ingest vào graph, sau đó đến thiết kế retrieval, evaluation và kết quả cuối."

## Phần 1 - Trung mở đầu

### Slide 3 - Research Problem and Objectives

### Lời thoại gợi ý
"Bài toán nhóm em hướng tới là multi-hop question answering cho tiếng Việt. Điểm khó không chỉ là tìm ra đáp án, mà còn là tìm đủ chuỗi bằng chứng nằm trên nhiều trang Wikipedia khác nhau.

Nếu chỉ dùng text retrieval thông thường, hệ thống có thể lấy được một mảnh thông tin nhưng bỏ lỡ mối liên hệ nối các mảnh bằng chứng lại với nhau.

Vì vậy, nhóm em đặt ra 3 mục tiêu chính:
- xây dựng một corpus Wikipedia tiếng Việt có thể tái lập và có evidence ID ổn định;
- kết hợp lexical, dense, graph và community retrieval trong Neo4j;
- đánh giá cả câu trả lời, bằng chứng, khả năng abstain và độ trễ hệ thống."

### Slide 4 - Users, Applicability, and Innovation

### Lời thoại gợi ý
"Hệ thống này hướng tới 3 nhóm giá trị chính.

Thứ nhất là sinh viên và nhà nghiên cứu, vì họ cần tổng hợp thông tin từ nhiều bài viết nhanh hơn.

Thứ hai là nhà báo hoặc analyst, vì họ cần citation rõ ràng thay vì chỉ một câu trả lời tóm tắt.

Thứ ba là AI developer, vì họ có thể tái sử dụng corpus, graph schema và evaluation pipeline của nhóm em.

Điểm mới nằm ở cách tiếp cận local-first cho GraphRAG tiếng Việt, có evidence ID rõ ràng, có WRRF để hợp nhất nhiều kênh retrieval, và có bộ dữ liệu ViWikiHop gồm cả câu hỏi answerable lẫn unanswerable."

### Slide 5 - System Architecture

### Lời thoại gợi ý
"Đây là kiến trúc tổng quan của toàn bộ hệ thống.

ViWikiHop là bộ dữ liệu dùng để kiểm thử và đánh giá.
Wikipedia-Neo4j là phần lõi thực hiện ingestion, retrieval, graph traversal và reasoning trên bằng chứng.
Aster là giao diện chat để trình diễn kết quả cho người dùng.

Điểm quan trọng là toàn bộ pipeline được nối thành một hệ thống thống nhất: từ dữ liệu nguồn, graph, retrieval đến câu trả lời cuối cùng đều có liên hệ rõ ràng."

## Chuyển giao 1 - Sang Quang Anh

### Ghi chú slide
- Slide 6 - Dataset Motivation
- Slide 7 - Data Pipeline --- Corpus Filtering
- Slide 8 - Corpus Construction
- Slide 9 - Data Pipeline --- ViWikiHop Schema v0.4
- Slide 10 - Data Pipeline --- Reasoning Types & Difficulty
- Slide 11 - Data Pipeline --- QA Generation & QC
- Slide 12 - ViWikiHop Dataset and Quality Controls

### Lời thoại gợi ý
"Sau phần bài toán và kiến trúc tổng quan, em xin chuyển sang bạn Nhữ Quang Anh để trình bày về dataset và quá trình xây dựng dữ liệu đánh giá của đề tài."

## Chuyển giao 2 - Nhận lại từ Khôi

### Ghi chú slide
- Slide 13 - AI Pipeline --- Data Ingestion
- Slide 14 - Knowledge Graph and Retrieval
- Slide 15 - Neo4j Ingestion Status
- Slide 16 - Neo4j Backbone --- Page to Chunk
- Slide 17 - Neo4j Backbone --- Chunk to Entity
- Slide 18 - Neo4j Backbone --- Page Links and Typed Entities
- Slide 19 - Neo4j Semantic Relations --- Entity Layer
- Slide 20 - Neo4j Semantic Relations --- Community Layer

### Lời thoại gợi ý
"Cảm ơn Khôi. Như vậy, sau khi đã có pipeline ingest và cấu trúc Neo4j phục vụ retrieval, em xin quay lại phần thiết kế retrieval, evaluation và các kết quả cuối của hệ thống."

## Phần 2 - Trung quay lại với method và results

### Slide 21 - Algorithm Complexity and Design Rationale

### Lời thoại gợi ý
"Slide này trả lời câu hỏi: tại sao các thành phần này được giữ lại trong hệ thống cuối.

WRRF được giữ lại vì nó hợp nhất tốt các kênh retrieval khác loại.
Graph traversal được giữ lại vì nó bổ sung khả năng tìm chuỗi bằng chứng đa bước.
Cross-encoder reranking được giữ lại vì nó cải thiện thứ tự ứng viên sau khi fusion.
Agent routing được giữ lại vì nó giúp tách câu hỏi đơn giản và câu hỏi phức tạp.

Nói ngắn gọn, mỗi thành phần đều tồn tại vì có đóng góp rõ ràng vào retrieval quality hoặc chất lượng đáp án cuối."

### Slide 22 - Weighted Reciprocal Rank Fusion

### Lời thoại gợi ý
"Đây là công thức fusion trung tâm của hệ thống.

Thay vì trộn raw score giữa các kênh retrieval vốn có thang đo khác nhau, nhóm em fusion theo thứ hạng.

Mỗi tài liệu được cộng điểm dựa trên vị trí xếp hạng của nó trong từng kênh, với trọng số riêng cho BM25, vector, graph và community.

Sau khi fusion, hệ thống deduplicate theo `chunk_id`, rồi mới đưa qua cross-encoder reranking.

Ý nghĩa thực tế là hệ thống tận dụng được điểm mạnh của từng kênh retrieval mà vẫn giữ được quá trình hợp nhất đơn giản và ổn định."

### Slide 23 - Evaluation Protocol

### Lời thoại gợi ý
"Để đánh giá hệ thống, nhóm em chia evaluation thành 3 mức.

Mức thứ nhất là gold-context QA. Ở mức này, hệ thống đã được cấp đúng bằng chứng, nên mục tiêu là kiểm tra phần answer generation khi retrieval không còn là nút thắt.

Mức thứ hai là open-corpus retrieval. Đây là mức quan trọng để đo riêng khả năng tìm đúng bằng chứng trong kho dữ liệu thật.

Mức thứ ba là end-to-end QA. Đây là setting gần với thực tế nhất, vì hệ thống phải tự tìm bằng chứng, sắp xếp bằng chứng, rồi mới sinh câu trả lời và quyết định có nên abstain hay không.

Về metric, nhóm em dùng theo từng vai trò:
- Hit@5 để đo bằng chứng đúng có xuất hiện trong top 5 hay không;
- MRR để đo chất lượng thứ hạng;
- support recall để đo mức độ thu hồi các mảnh bằng chứng cần thiết;
- complete-chain recall để đo khả năng lấy đủ chuỗi bằng chứng đa bước;
- EM, F1 và joint F1 để đo chất lượng câu trả lời và bằng chứng;
- abstention accuracy để đo khả năng từ chối đúng;
- latency để đo chi phí vận hành thực tế.

Toàn bộ thực nghiệm được cố định cùng test set 240 mẫu, cùng graph snapshot, cùng model setup và cùng các seeds 42, 43, 44 để bảo đảm tính so sánh công bằng giữa các cấu hình."

### Slide 24 - Final Retrieval Results

### Lời thoại gợi ý
"Đây là bảng kết quả retrieval chính, và nên đọc theo hai nhóm chỉ số.

Nhóm thứ nhất là khả năng tìm đúng tài liệu liên quan, gồm Hit@5 và MRR.
Nhóm thứ hai là khả năng gom đủ bằng chứng để trả lời câu hỏi đa bước, gồm support recall và chain recall.

Nếu nhìn từng kênh đơn lẻ:
- BM25 nhanh nhưng coverage còn thấp;
- vector retrieval cải thiện Hit@5 và MRR so với BM25;
- entity graph chưa mạnh nhất về rank, nhưng kéo support recall và chain recall tốt hơn vì bám vào quan hệ giữa các thực thể.

Điểm quan trọng là không có kênh riêng lẻ nào vừa mạnh về xếp hạng vừa mạnh về chuỗi bằng chứng. Vì vậy hybrid mới tạo ra khác biệt.

Khi dùng WRRF, nhóm em đẩy được Hit@5 lên 0.684 và chain recall lên 0.391. Sau khi thêm reranking, cấu hình tốt nhất đạt:
- Hit@5 = 0.711
- MRR = 0.501
- support recall = 0.628
- chain recall = 0.417

Thông điệp chính ở slide này là fusion cải thiện cả khả năng tìm đúng lẫn độ đầy đủ của bằng chứng, còn reranking giúp đưa bằng chứng tốt lên sớm hơn, đổi lại latency tăng từ 63.8 ms lên 118.6 ms."

### Slide 25 - Ablation Study

### Lời thoại gợi ý
"Slide này dùng để kiểm chứng nguyên nhân tạo ra kết quả tốt.

Full hybrid là mốc chuẩn, với Hit@5 = 0.711 và MRR = 0.501.

Khi chỉ dùng text-only, hệ thống giảm mạnh cả Hit@5 lẫn MRR. Điều này cho thấy nếu không có graph signal thì retrieval trên câu hỏi đa bước sẽ bỏ sót khá nhiều bằng chứng liên trang.

Khi chỉ dùng graph-only, kết quả vẫn có ích cho suy luận nhưng không đủ để thay thế hoàn toàn tín hiệu lexical và dense.

Khi bỏ reranking, MRR giảm 0.029. Điều này cho thấy fusion mới chỉ tạo candidate set tốt, còn reranker là bước giúp đưa bằng chứng đúng lên đầu.

Khi bỏ multi-hop expansion, Hit@5 giảm 0.063. Đây là bằng chứng trực tiếp cho thấy link expansion thực sự hữu ích với bài toán multi-hop.

Vì vậy, hai thành phần có tác động rõ nhất tới chất lượng cuối là multi-hop expansion và reranking."

### Slide 26 - End-to-End QA and Answerability

### Lời thoại gợi ý
"Khi ghép toàn bộ pipeline lại thành bài toán end-to-end, chúng ta sẽ thấy rõ khoảng cách giữa retrieval quality và final answer quality.

Trên toàn bộ tập test, hệ thống đạt answer F1 = 0.683, support F1 = 0.667 và joint F1 = 0.602.

Nếu nhìn riêng nhóm answerable, answer F1 là 0.653 nhưng joint F1 chỉ còn 0.536. Điều này cho thấy có những trường hợp hệ thống trả lời gần đúng nhưng bằng chứng chưa đủ sạch hoặc chưa đủ trọn chuỗi.

Nếu nhìn nhóm unanswerable, abstention đạt 0.753. Nghĩa là hệ thống đã có khả năng từ chối trong phần lớn trường hợp không có đáp án, nhưng vẫn còn khoảng trống để giảm false answer và false abstain.

Điểm cần nhấn mạnh là retrieval tốt là điều kiện cần, nhưng chưa phải điều kiện đủ. Khi đi vào end-to-end, hệ thống phải đồng thời tìm đúng, hiểu đúng và quyết định trả lời hay từ chối đúng."

### Slide 27 - Error Analysis and Limitations

### Lời thoại gợi ý
"Những hạn chế chính của hệ thống hiện tại gồm 4 nhóm.

Thứ nhất là incomplete evidence chains, tức là lấy được một phần bằng chứng nhưng chưa đủ chuỗi.
Thứ hai là entity-resolution noise.
Thứ ba là unsupported answer synthesis.
Thứ tư là incorrect abstention.

Ở mức hệ thống, nhóm em cũng ghi nhận rằng segment-to-chunk alignment chưa hoàn toàn đầy đủ, NER noise vẫn ảnh hưởng neighborhood của graph, và phần giao diện Aster hiện là live demo nên một số visual fixture còn mang tính trình diễn."

### Slide 28 - Contributions

### Lời thoại gợi ý
"Tóm lại, nhóm em có 4 đóng góp chính.

Một là pipeline corpus Wikipedia tiếng Việt có tính tái lập và có evidence identity rõ ràng.
Hai là bộ dữ liệu ViWikiHop gồm 1,600 mẫu answerable và unanswerable có group-aware split.
Ba là hệ thống Neo4j GraphRAG kết hợp lexical, vector, graph và community retrieval qua WRRF.
Bốn là quy trình đánh giá support-aware để chỉ ra lợi ích cũng như trade-off về latency của reranking và multi-hop expansion."

### Slide 29 - Conclusion

### Lời thoại gợi ý
"Kết luận lại, cấu hình GraphRAG đầy đủ của nhóm em đạt Hit@5 = 0.711, MRR = 0.501 và complete-chain recall = 0.417 trên tập test ViWikiHop.

Kết quả này cho thấy hybrid fusion mạnh hơn từng kênh retrieval đơn lẻ, và graph expansion cùng với reranking tạo ra lợi ích rõ ràng, dù có đánh đổi bằng latency.

Thông điệp sau cùng mà nhóm em muốn nhấn mạnh là: với bài toán multi-hop QA tiếng Việt, giá trị của hệ thống không chỉ nằm ở câu trả lời, mà nằm ở khả năng truy vết và kiểm chứng bằng chứng."

## Kết thúc

### Slide 30 - Thank You / Q&A

### Lời thoại gợi ý
"Phần trình bày của nhóm em xin kết thúc tại đây. Nhóm em rất mong nhận được các câu hỏi và góp ý từ thầy cô."

## Checklist cover slide

- Trung mở đầu: slide 1, 2, 3, 4, 5
- Quang Anh: slide 6, 7, 8, 9, 10, 11, 12
- Đình Khôi: slide 13, 14, 15, 16, 17, 18, 19, 20
- Trung kết thúc: slide 21, 22, 23, 24, 25, 26, 27, 28, 29, 30

Ghi chú:
- Script này đã bỏ các phần `Related Works and Early Validation`, `End-to-End Data Flow`, `Model Training and Implementation`, `Human Evaluation`, và `Team Contribution`.
- Nếu deck tiếp tục chỉnh thêm, chỉ cần cập nhật lại số slide trong phần checklist và các tiêu đề tương ứng.
