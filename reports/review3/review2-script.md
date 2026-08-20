# Kịch bản thuyết trình — Capstone Review 2
## Vietnamese GraphRAG: Multi-hop Question Answering over Wikipedia
### FPT University — Tháng 7/2026

---

## Phân công trình bày

| Thành viên | Slides |
|---|---|
| Truong Dinh Thien (QE180018) | 1, 2, 9, 10, 19, 20 |
| Huynh Quoc Trung (QE180038) | 3, 5, 6, 7, 13, 14 |
| Nhu Quang Anh (QE180005) | 4, 11, 12, 15, 16, 18 |
| Pham Ngo Dinh Khoi (QE180110) | 8, 17, 21 |

---

## Kịch bản chi tiết

### [SLIDE 1 — Title] Giới thiệu nhóm

**[THIEN]**

Kính chào thầy cô và các bạn, em tên là Truong Dinh Thien, đại diện nhóm chúng em xin phép bắt đầu buổi báo cáo Capstone Review 2 hôm nay. Dự án của nhóm em mang tên "Vietnamese GraphRAG — Multi-hop Question Answering over Wikipedia", tập trung vào việc xây dựng hệ thống hỏi đáp tiếng Việt có khả năng suy luận đa bước dựa trên đồ thị tri thức từ Wikipedia. Nhóm em gồm bốn thành viên: Huynh Quoc Trung phụ trách xây dựng đồ thị và trích xuất thực thể, Nhu Quang Anh phụ trách module truy xuất và đánh giá, Pham Ngo Dinh Khoi phụ trách agent suy luận và huấn luyện mô hình, còn em phụ trách phần hạ tầng API, giao diện người dùng và dashboard. Chúng em rất mong nhận được phản hồi quý báu từ thầy cô trong buổi hôm nay ạ.

---

### [SLIDE 2 — TOC] Nội dung trình bày

**[THIEN]**

Thưa thầy cô, trên slide này các bạn có thể thấy toàn bộ nội dung chúng em sẽ trình bày hôm nay. Buổi báo cáo được chia làm bốn phần lớn: phần đầu là kiến trúc hệ thống và thiết kế kỹ thuật, phần hai là chất lượng kỹ thuật phần mềm, phần ba là kết quả đánh giá và tập dữ liệu, và phần cuối là chất lượng kỹ thuật AI cùng tiến độ dự án. Chúng em sẽ trình bày theo đúng thứ tự như vậy, mỗi phần sẽ có một thành viên phụ trách để đảm bảo đi sâu vào từng khía cạnh. Nếu thầy cô có câu hỏi trong quá trình trình bày, nhóm em rất hoan nghênh được giải đáp ngay ạ. Bây giờ em xin nhường lời cho bạn Trung để bắt đầu phần kiến trúc hệ thống.

---

### [SLIDE 3 — Architecture] Kiến trúc tổng thể

**[TRUNG]**

Cảm ơn Thien, em xin tiếp tục ạ. Thưa thầy cô, nhìn vào slide này, các bạn thấy kiến trúc tổng thể của hệ thống GraphRAG mà nhóm em đã xây dựng. Ý tưởng cốt lõi là thay vì chỉ dùng vector search đơn giản như RAG truyền thống, chúng em kết hợp đồ thị tri thức để có thể suy luận qua nhiều bước, ví dụ như câu hỏi "Người sáng lập tổ chức mà nhân vật X là thành viên sinh năm bao nhiêu?" — loại câu hỏi này cần đi qua ít nhất hai nút trong đồ thị. Luồng dữ liệu đi từ trái sang phải: bên trái là pipeline nạp dữ liệu từ Wikipedia tiếng Việt vào Neo4j, bên phải là pipeline truy vấn khi người dùng đặt câu hỏi. Hai pipeline này hoàn toàn tách biệt, nghĩa là sau khi nạp dữ liệu xong, hệ thống có thể phục vụ hàng nghìn câu hỏi mà không cần đọc lại nguồn gốc. Điều này giúp độ trễ truy vấn của chúng em chỉ ở mức 51 mili giây, so với khoảng 2,5 giây nếu dùng RAG thuần túy gọi API ngoài.

---

### [SLIDE 4 — Deployment] Triển khai hệ thống

**[ANH]**

Cảm ơn Trung ạ. Thưa thầy cô, em là Nhu Quang Anh, em xin trình bày về phần triển khai. Như các bạn thấy trên slide, hệ thống của chúng em chạy hoàn toàn on-premise, không phụ thuộc vào dịch vụ đám mây bên ngoài cho phần lõi — đây là lựa chọn có chủ đích để đảm bảo tính tái lập và kiểm soát chi phí trong môi trường nghiên cứu. Thành phần trung tâm là Neo4j chạy như một systemd service trên máy chủ Linux, phía trước là FastAPI làm lớp API với xác thực API key và giới hạn tốc độ 120 request mỗi phút mỗi client. Mô hình ngôn ngữ cục bộ Vi-Qwen2-7B-RAG được tải lazy khi có request đầu tiên, nên không chiếm RAM khi hệ thống chưa cần đến. Môi trường Python được quản lý hoàn toàn bằng `uv`, giúp đảm bảo reproducibility và build nhanh hơn pip thông thường rất nhiều ạ.

---

### [SLIDE 5 — Ingestion Dataflow] Luồng nạp dữ liệu

**[TRUNG]**

Cảm ơn Anh ạ. Bây giờ em sẽ đi sâu hơn vào phần mà em trực tiếp xây dựng, đó là pipeline nạp dữ liệu. Thưa thầy cô, việc xử lý 1,6 triệu bài Wikipedia tiếng Việt không thể làm theo cách thông thường được, vì nếu gọi API từng bài một thì mất hàng tuần. Vì vậy nhóm em thiết kế một pipeline ba bước rõ ràng: bước một là export từ HuggingFace dataset ra file JSONL, bước hai là tạo embedding theo batch, bước ba là nạp hàng loạt vào Neo4j bằng câu lệnh UNWIND. Mỗi bước đều có checkpoint, nghĩa là nếu máy tắt giữa chừng thì lần sau chạy lại sẽ tiếp tục từ điểm đã dừng chứ không phải làm lại từ đầu. Đặc biệt ở bước trích xuất thực thể, chúng em dùng backend wikilink — khai thác chính các siêu liên kết có sẵn trong Wikipedia thay vì chạy mô hình NER nặng — điều này giúp tốc độ tăng lên rất nhiều trong khi Typed F1 vẫn đạt 46,9% ạ.

---

### [SLIDE 6 — Graph Schema] Lược đồ đồ thị

**[TRUNG]**

Tiếp theo, em xin trình bày về lược đồ đồ thị — đây là thiết kế trung tâm của toàn bộ hệ thống. Như các bạn thấy trên slide, lược đồ xoay quanh ba loại nút chính: Page đại diện cho bài viết Wikipedia, Chunk là đoạn văn bản được cắt ra từ bài viết, và Entity là các thực thể được nhận diện như Person, Organization, Location, Work. Quan hệ cơ bản là Page có HAS_CHUNK, Chunk có MENTIONS đến Entity, và Page có LINKS_TO đến Page khác — chính cái LINKS_TO này cho phép chúng em traversal đồ thị khi suy luận đa bước. Điều quan trọng là chúng em dùng typed mention edges, ví dụ MENTIONS_PERSON, MENTIONS_ORG, MENTIONS_LOCATION — điều này giúp câu truy vấn Cypher có thể filter theo loại thực thể một cách hiệu quả hơn là đọc toàn bộ MENTIONS rồi mới lọc. Toàn bộ lược đồ được thiết lập tự động khi khởi động hệ thống lần đầu, bao gồm cả full-text index và vector index ạ.

---

### [SLIDE 7 — Physical Design] Thiết kế vật lý

**[TRUNG]**

Về thiết kế vật lý, đây là phần mà nhiều bạn thường bỏ qua nhưng thực ra ảnh hưởng trực tiếp đến hiệu năng. Chúng em tạo ba loại index trên Neo4j: full-text index cho BM25 search trên thuộc tính text của Chunk, vector index 1024 chiều cho similarity search dùng thuật toán HNSW, và composite index trên các thuộc tính truy vấn thường xuyên như chunk_id và page_title. Lý do chọn 1024 chiều là vì mô hình embedding cục bộ GreenNode-Embedding-Large-VN-Mixed-V1 có output size là 1024, phù hợp với dữ liệu tiếng Việt hơn các mô hình đa ngữ thông thường. Sau khi nạp đầy đủ corpus, tổng dung lượng Neo4j vào khoảng vài chục GB — con số này hợp lý cho một hệ thống nghiên cứu chạy on-premise. Bây giờ em xin nhường lời cho bạn Khoi để trình bày về Job State Machine ạ.

---

### [SLIDE 8 — Job State Machine] Quản lý trạng thái công việc

**[KHOI]**

Cảm ơn Trung ạ, em là Pham Ngo Dinh Khoi. Thưa thầy cô, phần em trình bày tiếp theo là cơ chế quản lý các background job trong hệ thống — đây là thứ chạy ngầm phía sau mỗi khi người dùng kích hoạt nạp dữ liệu qua API. Như trên slide, mỗi job đi qua một state machine rõ ràng với các trạng thái: pending khi mới tạo, running khi đang xử lý, completed khi hoàn thành, failed khi gặp lỗi, cancelling và cancelled khi người dùng yêu cầu hủy. Điểm thiết kế quan trọng là job state được lưu xuống file JSON bằng cơ chế atomic write — tức là ghi ra file tạm rồi mới rename — để tránh mất dữ liệu nếu server crash giữa chừng. Khi server khởi động lại, các job đang ở trạng thái running hoặc cancelling sẽ tự động bị đánh dấu là interrupted, tránh trường hợp hệ thống nghĩ job vẫn đang chạy trong khi thực tế đã chết từ lâu. Cơ chế cancellation dùng threading.Event, cho phép người dùng dừng một job đang nạp hàng nghìn bài mà không cần restart server ạ.

---

### [SLIDE 9 — Screen Design] Thiết kế giao diện

**[THIEN]**

Cảm ơn Khoi ạ. Em xin tiếp tục với phần giao diện người dùng. Thưa thầy cô, hệ thống của chúng em có hai giao diện: một là Gradio demo cho phép thử nghiệm hỏi đáp trực tiếp, hai là dashboard theo dõi hệ thống. Gradio được chọn vì nó phù hợp với prototype nghiên cứu — triển khai nhanh, có sẵn text box và streaming output, không cần viết frontend riêng. Dashboard thì được xây bằng HTML thuần với một chút JavaScript, hiển thị các metrics quan trọng như số lượng node trong đồ thị, số job đang chạy, lịch sử query gần đây, và biểu đồ phân phối loại thực thể. Điểm quan trọng là dashboard đọc dữ liệu từ Neo4j và log file theo thời gian thực, không phải hardcode — vì vậy mỗi khi nạp thêm dữ liệu mới, các con số tự động cập nhật mà không cần refresh thủ công ạ.

---

### [SLIDE 10 — Technology Choices] Lựa chọn công nghệ

**[THIEN]**

Đây là slide tóm tắt các lựa chọn công nghệ chính của nhóm em, và em muốn giải thích ngắn gọn lý do đằng sau mỗi lựa chọn thay vì chỉ liệt kê tên. Neo4j được chọn vì nó là graph database trưởng thành nhất, hỗ trợ native full-text search và vector search trong cùng một câu truy vấn Cypher — điều mà các graph database khác chưa làm được mượt mà. FastAPI được chọn vì async-native, type-safe với Pydantic, và có OpenAPI docs tự động. Về mô hình ngôn ngữ, Vi-Qwen2-7B-RAG được chọn vì đây là mô hình tiếng Việt mạnh nhất trong tầm kích thước có thể chạy được trên phần cứng nghiên cứu với quantization 4-bit NF4. Embedding dùng GreenNode-Embedding-Large-VN-Mixed-V1 vì đây là embedding model tiếng Việt tốt nhất hiện có trên HuggingFace với 1024 chiều. Toàn bộ dependency được pin version và quản lý bằng `uv` để đảm bảo build reproducible trên mọi máy ạ. Tiếp theo, em xin nhường lời cho bạn Anh để trình bày về chất lượng code.

---

### [SLIDE 11 — Coding Quality] Chất lượng code & Design Patterns

**[ANH]**

Cảm ơn Thien ạ. Thưa thầy cô, em xin trình bày về chất lượng code của hệ thống. Thay vì chỉ nói rằng code đẹp, nhóm em enforce chất lượng bằng tooling tự động: ruff và ruff-format chạy trong pre-commit hook, mypy kiểm tra type ở chế độ strict, và toàn bộ CI pipeline phải pass trước khi merge. Điều này có   nghĩa là không ai có thể push code thiếu type hint hay vi phạm naming convention mà không bị chặn lại ngay. Về design pattern, nhóm em áp dụng bốn pattern chính một cách có chủ đích: Strategy cho NER backends — chỉ đổi biến môi trường `NER_BACKEND` là toàn bộ hệ thống chuyển sang dùng backend khác mà không cần sửa một dòng code nào; Singleton cho Neo4j driver để tránh tạo nhiều connection pool; Factory qua hàm `get_ner_backend()` để tách logic khởi tạo; và Facade cho WRRF pipeline để agent layer không cần biết bên trong có bốn tín hiệu đang được kết hợp. Cấu hình và secret hoàn toàn đi qua `pydantic_settings.BaseSettings` từ file `.env`, không có bất kỳ credential nào được hardcode trong source code ạ.

---

### [SLIDE 12 — Algorithm Complexity] Độ phức tạp thuật toán

**[ANH]**

Tiếp theo, em sẽ nói về phần thuật toán. Nhìn vào slide, các bạn thấy công thức WRRF — đây là trái tim của module truy xuất mà em xây dựng. Ý tưởng cơ bản là thay vì chọn một phương pháp tìm kiếm duy nhất, chúng em kết hợp bốn tín hiệu: BM25 fulltext với trọng số 0.4, vector similarity với trọng số 0.4, graph traversal 2-hop với trọng số 0.2, và community-based retrieval với trọng số 0.15. Hằng số k=60 trong công thức là smoothing factor giúp tránh tình trạng một tín hiệu có rank rất thấp bị kéo score về 0. Kết quả ablation cho thấy mỗi tín hiệu thêm vào đều cải thiện MRR: BM25 đơn lẻ đạt 0.35, thêm vector lên 0.40, thêm graph lên 0.43, và khi thêm tín hiệu community thứ tư thì MRR đạt 0.4497 — tức là tổng cộng tăng 28.5% so với baseline. Ngoài WRRF, chúng em còn dùng Louvain community detection của Leiden algorithm để phân cụm các trang Wikipedia có liên quan, ReAct loop tối đa 6 iteration để tránh vòng lặp vô hạn, và multi-trajectory voting để tăng độ ổn định cho câu hỏi phức tạp ạ.

---

### [SLIDE 13 — Corpus Filter] Data Pipeline — Lọc corpus

**[TRUNG]**

Cảm ơn Anh ạ. Bây giờ chúng ta chuyển sang phần data pipeline mà em xây dựng. Thưa thầy cô, điểm xuất phát là dump Wikipedia tiếng Việt snapshot `viwiki-20260501` với 1.614.015 bài viết. Con số này nghe lớn nhưng thực ra rất nhiều bài không dùng được cho multi-hop QA — có bài chỉ là redirect, có bài là trang định hướng, có bài chỉ vài chục chữ nội dung thực sự. Vì vậy nhóm em thiết kế một pipeline 11 bước lọc từng loại nội dung rác ra. Bước lớn nhất là word count pre-filter — loại bỏ 861.000 bài quá ngắn — vì một bài viết chỉ 50 từ thì không thể đặt câu hỏi đa bước được. Tiếp theo là orphan filter loại 17.000 bài không có bài nào liên kết đến — bài cô lập như vậy không tham gia được vào đồ thị tri thức. Cuối cùng là MinHash near-deduplication với ngưỡng Jaccard 0.85 để loại các bài sao chép gần giống nhau. Kết quả cuối cùng là 353.498 bài — chỉ 21.9% so với dump gốc nhưng đây là phần chất lượng nhất, đủ để xây dựng một đồ thị tri thức có chiều sâu ạ.

---

### [SLIDE 14 — ViWikiHop Schema] Data Pipeline — ViWikiHop Schema v0.4

**[TRUNG]**

Sau khi có corpus sạch, nhóm em xây dựng bộ dataset ViWikiHop riêng cho tiếng Việt. Schema v0.4 được thiết kế với 13 trường chính, trong đó có một số điểm đáng chú ý. Trường `_id` có tiền tố để phân biệt nguồn gốc: `hw-` cho câu hỏi do người viết tay, `syn-mh-` cho câu hỏi được sinh tự động từ KG walks, và `uvq-` cho các câu hỏi được import từ UIT-ViQuAD 2.0. Trường `decomposition` lưu chuỗi sub-question và sub-answer theo từng hop — đây là điểm khác biệt so với HotpotQA hay MultiHop-RAG, giúp model có thể học từng bước suy luận riêng lẻ. Schema còn có invariants rõ ràng: câu hỏi 1-hop thì `decomposition` phải rỗng, câu không trả lời được thì `answer` phải rỗng nhưng `plausible_answer` phải có nội dung. Những ràng buộc này được kiểm tra tự động trong QC pipeline trước khi một entry được chấp nhận vào dataset ạ. Bây giờ em xin nhường lời cho bạn Anh để trình bày về reasoning types và QC pipeline.

---

### [SLIDE 15 — Reasoning Types] Data Pipeline — Reasoning Types & Difficulty

**[ANH]**

Cảm ơn Trung ạ. Thưa thầy cô, một trong những đóng góp quan trọng của nhóm em là hệ thống phân loại 8 kiểu suy luận cho câu hỏi multi-hop tiếng Việt. Các kiểu này không phải tùy tiện — chúng được thiết kế để đánh giá từng khả năng suy luận riêng biệt của hệ thống. `bridge` là kiểu phổ biến nhất: đi từ thực thể A qua một thực thể trung gian để đến đáp án. `comparison` yêu cầu lấy thông tin từ ít nhất hai bài rồi so sánh. `temporal` cần hiểu thứ tự thời gian. `unanswerable` kiểm tra xem model có biết từ chối trả lời khi context không đủ hay không — đây là điểm yếu của nhiều hệ thống RAG hiện tại. Về độ khó, chúng em dùng rubric 3 chiều: số hop, lexical overlap giữa câu hỏi và đáp án, và loại thao tác suy luận. Mỗi chiều cho điểm từ 0 đến 2, tổng điểm 0-1 là easy, 2-3 là medium, 4-6 là hard. Cách tính này minh bạch và reproducible hơn là để con người tự đánh giá độ khó một cách cảm tính ạ.

---

### [SLIDE 16 — QA Gen & QC] Data Pipeline — QA Generation & QC

**[ANH]**

Về pipeline sinh câu hỏi và kiểm soát chất lượng, quy trình gồm hai phần song song. Phía sinh dữ liệu: chúng em extract các đường đi 2-hop, 3-hop và broken-link từ đồ thị Neo4j, sau đó dùng template tiếng Việt để sinh câu hỏi từ các triple trích xuất được. Bước LLM rewrite tùy chọn giúp câu hỏi tự nhiên hơn và tránh pattern nhàm chán như "Theo bài viết, X là gì?" — một dấu hiệu rõ ràng câu hỏi được sinh tự động. Phía QC, chúng em có 5 bước kiểm tra tuần tự: well-formedness để loại câu hỏi rỗng hoặc sai định dạng, grounding để đảm bảo thực thể thực sự xuất hiện trong đoạn văn bản, dedup để loại câu trùng lặp, answerability để kiểm tra đáp án có nằm trong supporting facts không, và cuối cùng là natural phrasing để block các câu có template language lộ liễu. Kết quả là dataset ViWikiHop với tỷ lệ câu hỏi chất lượng cao, sẵn sàng dùng cho evaluation ạ. Bây giờ em xin nhường lời cho bạn Khoi để trình bày về phần huấn luyện mô hình.

---

### [SLIDE 17 — Model Training] Huấn luyện mô hình QLoRA

**[KHOI]**
Cảm ơn Anh ạ. Thưa thầy cô, phần em sẽ trình bày là fine-tuning mô hình ngôn ngữ cho task Text-to-Cypher — tức là chuyển câu hỏi tiếng Việt thành câu truy vấn Cypher để chạy trên Neo4j. Chúng em chọn QLoRA vì đây là phương pháp hiệu quả nhất để fine-tune mô hình 7 tỷ tham số trên phần cứng nghiên cứu có hạn. Cụ thể, base model là Vi-Qwen2-7B-RAG được quantize xuống 4-bit NF4, và chúng em chỉ train thêm các LoRA adapter với rank r=32, lora_alpha=64, và dropout 0.05 — tổng số tham số train thực tế chỉ bằng một phần nhỏ so với toàn bộ mô hình. Các target modules là q_proj và v_proj trong attention layers, vì đây là nơi model học cách align input tiếng Việt với cú pháp Cypher. Adapter được lưu theo `LORA_ADAPTER_PATH` và lazy-load khi có request đầu tiên, không chiếm GPU khi hệ thống chưa cần. Ngoài fine-tuning, hệ thống còn dùng ba mô hình pre-trained: GreenNode embedding cho retrieval, ViDeBERTa cho NER, và BAAI/bge-reranker cho reranking — tất cả đều không cần fine-tune thêm ạ.


---

### [SLIDE 18 — Evaluation Results] Kết quả đánh giá

**[ANH]**

Thưa thầy cô, đây là slide kết quả mà em muốn trình bày kỹ nhất. Nhìn vào bảng so sánh, hệ thống WRRF kết hợp reranker của chúng em đạt MRR=0.4497 và Hit@5=64.67% — quan trọng hơn, độ trễ chỉ 51 mili giây. Con số 51ms này đặc biệt ấn tượng vì naive RAG gọi API ngoài mất khoảng 2,5 giây cho cùng một câu hỏi. Sở dĩ hệ thống của chúng em nhanh hơn 50 lần là vì toàn bộ inference chạy local — Neo4j vector search, reranker, và LLM đều trên máy chủ cùng mạng nội bộ, không có round-trip qua internet. Về đánh giá NER, backend wikilink đạt Typed F1=46.9% — con số này khiêm tốn ở nhìn tuyệt đối nhưng là cao nhất trong 6 backend và đặc biệt phù hợp với bulk ingestion vì tốc độ rất nhanh. Chúng em cũng thành thật với thầy cô về các điểm yếu: abstain accuracy bằng 0 nghĩa là model không bao giờ từ chối trả lời câu không có đáp án, hit rate trên câu hỏi 3-hop fan-out còn thấp, và khoảng 5% entity bị mismatch do dấu thanh tiếng Việt. Đây là ba vấn đề ưu tiên cho giai đoạn tiếp theo ạ. Bây giờ em xin nhường lời cho bạn Thien.

---

### [SLIDE 19 — AI Engineering Quality] Chất lượng kỹ thuật AI

**[THIEN]**

Cảm ơn Anh ạ. Thưa thầy cô, một tiêu chí quan trọng trong đánh giá AI engineering là liệu hệ thống có chạy pipeline thực sự hay chỉ là demo giả lập. Nhóm em tự hào là tất cả các interface — Gradio demo, REST API `/query`, dashboard GraphPulse, và MCP server — đều gọi đúng pipeline thực tế từ đầu đến cuối: NER trích xuất thực thể, embedding tạo vector, WRRF kết hợp bốn tín hiệu, reranker cross-encoder, rồi mới đến LLM sinh câu trả lời. Không có path nào là hardcoded hay mock response. Về engineering nâng cao, nhóm em đã tự xây dựng WRRF fusion có configurable weights thay vì dùng thư viện có sẵn, Leiden community detection làm tín hiệu retrieval thứ tư, cơ chế complexity routing để phân loại câu hỏi đơn giản và phức tạp, multi-trajectory majority voting cho câu hỏi multi-hop, entity resolution với diacritic normalization cho tiếng Việt, và Cypher safety validation để chặn các lệnh ghi lén vào database. Điểm còn tồn tại mà nhóm em muốn minh bạch: chúng em chưa có experiment tracking với MLflow hay W&B — kết quả eval hiện lưu dưới dạng JSON timestamped. Đây là việc cần làm trong tuần tới ạ.

---

### [SLIDE 20 — Team & Timeline] Đóng góp nhóm & Tiến độ

**[THIEN]**

Về phần tiến độ dự án, nhóm em đang bám sát kế hoạch. Các milestone chính đã hoàn thành: core pipeline ổn định từ Review 1, full ingestion pipeline và ablation study hoàn thành trong Review 2. Milestone đang trong tiến trình: QLoRA Text-to-Cypher fine-tuning và WRRF weight tuning với final evaluation. Điểm đáng chú ý là UCP — Use Case Points — của dự án đạt 96.5, tính từ UUCP=134 với TCF=0.9 và ECF=0.8. Theo thang đo này, nhóm em đang on-schedule so với kế hoạch capstone. Mỗi thành viên đều có module rõ ràng và không có sự chồng chéo không cần thiết: Trung lo phần đồ thị và dữ liệu, Anh lo phần truy xuất và đánh giá, Khoi lo phần agent và mô hình, còn em lo phần hạ tầng và giao diện. Sự phân công này giúp chúng em có thể làm việc song song và review chéo nhau hiệu quả trong suốt quá trình phát triển ạ.

---

### [SLIDE 21 — Thank You / Q&A] Kết thúc & Câu hỏi

**[KHOI]**

Thưa thầy cô và các bạn, đó là toàn bộ nội dung nhóm em chuẩn bị cho Capstone Review 2 hôm nay. Tóm lại, chúng em đã xây dựng một hệ thống GraphRAG tiếng Việt hoàn chỉnh, từ pipeline lọc corpus 1.6 triệu bài xuống còn 353 nghìn bài chất lượng, xây dựng đồ thị tri thức với NER và entity resolution, thiết kế WRRF hybrid retrieval kết hợp bốn tín hiệu, cho đến fine-tuning mô hình với QLoRA và xây dựng bộ dataset ViWikiHop riêng cho tiếng Việt. Các kết quả đo được cho thấy hệ thống hoạt động ổn định với MRR=0.4497 và latency 51ms — đủ để chứng minh tính khả thi của hướng tiếp cận GraphRAG cho tiếng Việt. Chúng em ý thức được các điểm còn hạn chế và có kế hoạch rõ ràng để cải thiện trong giai đoạn tiếp theo. Nhóm em rất mong nhận được phản hồi và câu hỏi từ thầy cô — tất cả bốn thành viên đều có mặt và sẵn sàng trả lời ạ. Xin chân thành cảm ơn thầy cô!

---

## Phụ lục — Ghi chú cho Q&A

### Appendix A: So sánh NER backends

Nếu thầy cô hỏi về lý do chọn backend NER:
- Simple backend (regex + keyword) đạt Typed F1 ~20%, nhanh nhưng không dùng ML
- underthesea ~35%, phonlp ~38%, phobert ~42% — tăng dần nhưng chậm hơn
- videberta (ViDeBERTa/NlpHUST electra-base) đạt 44%
- wikilink đạt 46.9% và đặc biệt nhanh vì tái dụng siêu liên kết có sẵn trong Wikipedia — phù hợp nhất cho bulk ingestion với dataset tiếng Việt

### Appendix B: WRRF Ablation chi tiết

Nếu thầy cô hỏi về ablation WRRF:
- BM25 only: MRR=0.35, Hit@5=55%
- + Vector: MRR=0.40 (+14.3%), Hit@5=60%
- + Graph: MRR=0.43 (+22.9%), Hit@5=63%
- + Community (4 tín hiệu): MRR=0.4497 (+28.5%), Hit@5=64.67%
- Mỗi tín hiệu đều đóng góp có ý nghĩa, không có tín hiệu nào thừa

### Appendix C: MHQA Generation Pipeline

Nếu thầy cô hỏi về pipeline sinh dataset:
- `scripts/mhqa_extract_walks.py`: extract 2-hop, 3-hop, broken-link walks từ Neo4j
- `scripts/mhqa_generate.py`: sinh câu hỏi từ template tiếng Việt, có option --rewrite dùng Gemini
- `scripts/mhqa_qc.py`: 5-stage QC pipeline
- `scripts/mhqa_retrievability.py`: kiểm tra mỗi cặp QA có thể được retrieve bởi WRRF không trước khi đưa vào dataset

---

*Kịch bản kết thúc — Tổng số slide chính: 21 | Phụ lục: 3 slides*
