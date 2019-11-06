
# TD5 - Chia nhóm các bài báo thời sự - xã hội trên VNExpress

## Mô tả

Ở TD3, chúng ta đã học kĩ năng thu thập dữ liệu từ Internet và xử lí dữ liệu để thu được các đoạn văn bản tương đối "sạch". Trong TD này, ta sẽ tiến hành chia nhóm các văn bản này.

### Tải dữ liệu

Khi chạy chương trình ở TD3 cho năm 2017 với thể loại **Thời sự**, ta thu được dữ liệu dưới dạng thư mục `RawData_News` <a href="https://drive.google.com/drive/folders/1u2IQzbDPvZ7Q_bE_uHMGkt5n9YF5QnmG?usp=sharing">ở đây</a>. Thư mục này gồm 12 file ứng với các tháng. Bạn cần tải thư mục này và lưu tại vị trí `Lesson5/TD/RawData`. Ta cũng cần làm việc với một danh sách các từ trong tiếng Việt, được trích xuất từ từ điển tiếng Việt của Hoàng Phê ([1]), nằm ở thư mục `VietnameseDictionary`. Bạn cần tải thư mục này và lưu tại `Lesson5/TD/VietnameseDictionary`. Cuối cùng, bạn có thể tải thư mục `FullData`, trong đó chứa kết quả bước 1 của TD trong trường hợp bạn không thực hiện thành công bước này.

### Quan sát sơ lược dữ liệu

Ta nhắc lại rằng ở TD3, mỗi bài báo của VNExpress được trích lọc tiêu đề, đoạn giới thiệu và nội dung. Ba thành phần này của mỗi bài báo đã được chúng ta lưu bằng một hàng của file dữ liệu, chúng cách nhau bởi hai khoảng trắng tab (`"\t\t"`). Riêng với thành phần thứ ba (nội dung của bài báo), các đoạn văn cách nhau bởi một khoảng trắng tab (`"\t"`).

Bạn có thể xem ví dụ từ <a href="https://drive.google.com/drive/folders/1nmsx4QAUwy2gQKMxqoTqshmTESoHUmYD">RawData_News/News_012017.txt</a>.

### Mục tiêu

Từ 12 file dữ liệu "thô" này, ta sẽ thực hiện việc tìm hiểu các bài báo thể thao năm 2017 của vnexpress có thể được chia thành những nhóm nào dựa trên sự tương đồng về nội dung. Ta sẽ thực hiện theo quy trình sau:

- Bước 1: Preprocessing 1 - Biến bài báo thành bag-of-words
 - Ghép 12 file dữ liệu thành file duy nhất và xoá các hàng trắng. File này tương tự như `FullData/News2017_Solution.txt`.
 - Tách nội dung mỗi bài báo thành một túi từ (bag of words), gồm mỗi từ và các tần số của nó trong mỗi bài báo, và lưu nó vào một file. File này tương tự như `FullData/FrequencyByNewsArticle_Solution.txt`
 - Tính tần số tổng cộng của các từ trên tất cả các bài báo và lưu nó vào một file. File này tương tự như `GlobalNewsFrequency_Solution.txt`. File này sẽ giúp ta xác định những từ nào là thông dụng, ít thông dụng trong tiếng Việt để xử lí ở các phần sau.
 
- Bước 2: Preprocessing 2 - Chọn biến quan trọng và biến bag-of-words thành vectors
 - Từ kết quả bước 1, ta sẽ chọn ra những từ được xem là quan trọng để phân loại các bài báo thể thao, sau đó trên cở sở này biến mỗi bài báo thành một vector trong $[0, 1]^d$ ($d \in \mathbf N$ là số từ quan trọng được lựa chọn, cũng chính là số chiều của không gian). 
 - Do mỗi vector có số lượng 0 rất lớn, ta có thể lưu tất cả các vector dưới dạng một ma trận sparse thông qua scipy.sparse.

- Bước 3: Chia nhóm với K-Means. 

 Đến bây giờ khi đã có các vector, ta sẽ dùng K-Means để chia nhóm, rồi thử thay đổi các tham số, rồi phân tích kết quả.

- Bước 4: Thử thực hiện một số phương pháp khác để so sánh kết quả
 - Cho $k$ thay đổi (3, 4, 6, 8, 10, 20, ...)
 - Thử với khoảng cách Euclide và cosine
 - Chia nhóm với Hierarchical Clustering.
 - Bạn cũng có thể tự do thử với các mô hình khác có sẵn trong Python và so sánh hiệu quả của các phương pháp.
 
Bạn cần hoàn thành các hàm trong file <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson5/TD/SportArticlesClustering.py">SportArticlesClustering.py</a>  ứng với yêu cầu của các bài tập, và test theo chỉ dẫn ở từng bài tập.

Bạn có thể tham khảo lời giải ở <a  href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson5/TD/SportArticlesClustering_Solution.py">SportArticlesClustering_Solution.py</a>  sau khi hoàn thành.


```python
from SportArticlesClustering_Solution import *
import pandas as pd
```

Dưới đây là các hằng số cơ bản chỉ đường dẫn đến các tập tin sẽ làm việc. Khi test, nhớ copy và chạy các dòng này.


```python
RAW_DATA_FOLDER = "RawData/"
ALL_DATA_FILE = "FullData/News2017.txt"
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyByNewsArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalNewsFrequency.txt"
```

## Phần 1. Preprocessing 1 - Chuyển đổi bài báo thành túi từ

### Bài 1. Tổng hợp các bài báo theo tháng thành file chung

Giả sử trong thư mục **`raw_data_folder`** chứa các file dữ liệu con (trong trường hợp của chúng ta: 12 file, mỗi file ứng với một tháng), và các file này có dạng như output ở TD3. Một số hàng trong file có thể trắng do bài báo không có nội dung. 

*Viết hàm **`concatenateDataFiles(raw_data_folder, full_data_folder, all_data_file)`** nhận đối số là tên thư mục **`raw_data_folder`**, nối tất cả dữ liệu trong các file này thành một file duy nhất tại đường dẫn **`all_data_file`**, đồng thời xoá tất cả các hàng trắng trong file. *

Bạn nên nối các file theo thứ tự tên của chúng, để các bài báo xuất hiện theo thứ tự từ tháng 1 đến tháng 12.

Đoạn code dưới đây giúp test hàm của bạn.


```python
concatenateDataFiles(RAW_DATA_FOLDER, ALL_DATA_FILE)
data = pd.read_csv(ALL_DATA_FILE, sep="\t\t", header = None)
len(data)
```




    6748




```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cơ quan khí tượng triều Nguyễn</td>
      <td>Ngoài việc xác định khí tượng thời tiết có gì ...</td>
      <td>Sau khi lên ngôi năm 1802, vua Gia Long cho th...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
      <td>Người chở hàng hóa không che chắn bị phạt đến ...</td>
      <td>Chế độ bồi dưỡng cho cán bộ tiếp công dân\tThe...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Đường vành đai Hà Nội ùn tắc ngày mùng 4 Tết</td>
      <td>Chưa hết kỳ nghỉ Tết, cửa ngõ phía Nam Hà Nội ...</td>
      <td>Chiều mùng 4 tết Đinh Dậu (31/1), đường dẫn và...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
      <td>Theo Bí thư Thành ủy TP HCM, một số cán bộ chi...</td>
      <td>Thăm và làm việc với Phòng Cảnh sát giao thông...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Người dân dựng biển cảnh báo 'Công an bắn tốc độ'</td>
      <td>Tấm bảng được cột vào thùng rác, dựng thêm một...</td>
      <td>Mạng xã hội hôm 30/1 đăng tải tấm biển bằng ca...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6743</th>
      <td>Ông Nguyễn Thiện Nhân: 'Thành ủy TP HCM sẽ xem...</td>
      <td>Theo Bí thư Thành ủy TP HCM, năm nay số cán bộ...</td>
      <td>Ngày 30/11, phát biểu khai mạc Hội nghị lần th...</td>
    </tr>
    <tr>
      <th>6744</th>
      <td>Không chỉ định thầu xây sân bay Long Thành</td>
      <td>"Chúng ta cố gắng đấu thầu dự án để đảm bảo tố...</td>
      <td>Tại cuộc họp bàn công tác triển khai dự án sân...</td>
    </tr>
    <tr>
      <th>6745</th>
      <td>Người lái xe container bằng chân tại Sài Gòn b...</td>
      <td>Tài xế phân bua "chân bị tê nên gác lên vô lăn...</td>
      <td>Ngày 1/12, Phòng CSGT TP HCM cho biết đã làm v...</td>
    </tr>
    <tr>
      <th>6746</th>
      <td>Hai người tử vong, xe máy nát vụn sau cú đâm đ...</td>
      <td>Sau cú đâm đối đầu, cả hai xe máy bị nát vụn, ...</td>
      <td>Hơn 19h tối 30/11 tại quốc lộ 6, (đoạn qua địa...</td>
    </tr>
    <tr>
      <th>6747</th>
      <td>Bốn thứ trưởng nghỉ hưu từ ngày 1/12</td>
      <td>Thủ tướng Nguyễn Xuân Phúc ký quyết định nghỉ ...</td>
      <td>Ông Thạch Dư được bổ nhiệm làm Thứ trưởng Bộ N...</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 2. Hàm phụ [1]: Tách đoạn văn thành câu

Ta bắt đầu thực hiện việc chính của preprocessing 1: biến nội dung mỗi bài báo thành túi từ.

Trong TD3, ta đã chuyển đổi nội dung bài báo thành các "từ" với giả sử rằng mỗi từ luôn là một từ đơn. Ví dụ "long lanh" được xem là hai từ, "long" và "lanh". Điều này không thực tế đối với tiếng Việt. Do vậy ta cần thực hiện một quy trình "gần đúng" như sau:

- Biến mỗi đoạn văn thành câu
- Biến mỗi câu thành từng "thành phần"
- Biến mỗi thành phần thành từng tiếng (tương đương từ đơn)
- Ghép các tiếng thành từ (từ đơn, từ ghép hoặc từ láy)

Lưu ý rằng quy trình này chỉ là gần đúng, không nên hy vọng nó thực hiện chính xác 100% mong muốn của chúng ta.

Bài tập 2, 3, 4, 6 thực hiện 4 thao tác trên.

*Hãy viết hàm **`paragraphToSentences(paragraph)`** nhận đối số **`paragraph`** là một đoạn văn dưới dạng `str`, và biến nó thành một list các câu theo thuật toán sau: *

*- Xoá tất cả các kí tự '\xc2\xa0' có trong đoạn văn (đây là một kí tự vô dụng thường gặp ở vnexpress)*

*- Những nơi có dấu chấm theo sau bởi dấu cách (. ), ba chấm hoặc nhiều chấm theo sau bởi dấu cách (... ), chấm hỏi theo sau bởi dấu cách (? ), chấm than theo sau bởi dấu cách (! ) được xem là các vị trí kết thúc câu. Split đoạn văn thành các câu tại các vị trí đó.*

Đoạn code dưới đây giúp test hàm của bạn.

Đoạn văn gốc:
*Ngày 8/3, Văn phòng Trung ương Đảng có công văn về việc xử lý vụ Tổng công ty viễn thông Mobifone mua 95% cổ phần của Công ty cổ phần nghe nhìn Toàn Cầu (AVG). Theo đó, vừa qua, Ban Bí thư đã họp dưới sự chủ trì của Tổng Bí thư Nguyễn Phú Trọng để nghe Ban cán sự đảng Thanh tra Chính phủ báo cáo kết quả việc thanh tra dự án nêu trên! Ban Bí thư cho rằng, đây là một vụ việc rất nghiêm trọng, phức tạp, nhạy cảm, dư luận xã hội đặc biệt quan tâm. Thanh tra Chính phủ đã có nhiều cố gắng tiến hành thanh tra toàn diện, kết luận và báo cáo với Ban Bí thư... Ban Bí thư đề nghị Thường trực Chính phủ, Thanh tra Chính phủ chỉ đạo và chịu trách nhiệm về Kết luận thanh tra, sớm công bố Kết luận thanh tra theo quy định của pháp luật. Các cơ quan có trách nhiệm khẩn trương xem xét, xử lý vụ việc bảo đảm khách quan, chính xác theo quy định của Đảng và pháp luật Nhà nước với tinh thần kiên quyết, chặt chẽ, làm rõ đến đâu xử lý đến đó, đúng người, đúng vi phạm, đúng pháp luật và thu hồi tài sản Nhà nước bị thất thoát.*


```python
p = "Ngày 8/3, Văn phòng Trung ương Đảng có công văn về việc xử lý vụ Tổng công ty viễn thông Mobifone mua " +\
    "95% cổ phần của Công ty cổ phần nghe nhìn Toàn Cầu (AVG). " +\
    "Theo đó, vừa qua, Ban Bí thư đã họp dưới sự chủ trì của Tổng Bí thư Nguyễn Phú Trọng để nghe Ban cán sự đảng Thanh tra " +\
    "Chính phủ báo cáo kết quả việc thanh tra dự án nêu trên!" +\
    "Ban Bí thư cho rằng, đây là một vụ việc rất nghiêm trọng, phức tạp, " +\
    "nhạy cảm, dư luận xã hội đặc biệt quan tâm. Thanh tra "+\
    "Chính phủ đã có nhiều cố gắng tiến hành thanh tra toàn diện, kết luận và báo cáo với Ban Bí thư... " +\
    "Ban Bí thư đề nghị Thường trực Chính phủ, Thanh tra Chính phủ chỉ đạo và chịu trách nhiệm về Kết luận thanh tra, " +\
    "sớm công bố Kết luận thanh tra theo quy định của pháp luật. Các cơ quan có trách nhiệm khẩn trương xem xét, " +\
    "xử lý vụ việc bảo đảm khách quan, chính xác theo quy định của Đảng và pháp luật Nhà nước với tinh thần kiên quyết, " +\
    "chặt chẽ, làm rõ đến đâu xử lý đến đó, đúng người, đúng vi phạm, đúng pháp luật và thu hồi tài sản Nhà nước bị thất thoát."
s = paragraphToSentences(p) # Kết quả là 1 list
pd.DataFrame(s) # Chuyển list thành DataFrame để dễ quan sát
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ngày 8/3, Văn phòng Trung ương Đảng có công vă...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Theo đó, vừa qua, Ban Bí thư đã họp dưới sự ch...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thanh tra Chính phủ đã có nhiều cố gắng tiến h...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ban Bí thư đề nghị Thường trực Chính phủ, Than...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Các cơ quan có trách nhiệm khẩn trương xem xét...</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 3. Hàm phụ [2]: Tách câu thành thành phần

Trong mỗi câu, ta có thể gặp các dấu câu khác như dấu phẩy, chấm phẩy, hai chấm, gạch ngang... Chúng chia câu thành các thành phần nhỏ hơn. Ví dụ, câu *"Theo đó, vừa qua, Ban Bí thư đã họp dưới sự chủ trì của Tổng Bí thư Nguyễn Phú Trọng để nghe Ban cán sự đảng Thanh tra Chính phủ báo cáo kết quả việc thanh tra dự án nêu trên!"* có thể được xem là tổng hợp của 3 thành phần *"Theo đó,", "vừa qua,", "Ban Bí thư đã họp dưới sự chủ trì của Tổng Bí thư Nguyễn Phú Trọng để nghe Ban cán sự đảng Thanh tra Chính phủ báo cáo kết quả việc thanh tra dự án nêu trên!"*

*Hãy viết hàm **`sentenceToSegments(sentence)`** nhận đối số **`sentence`** là một đoạn câu dưới dạng `str`, tức một phần tử của list trong output của bài 2, và biến nó thành một list các thành phần theo thuật toán tương tự như ở bài 2, nhưng đối với các dấu phẩy theo sau bởi dấu cách (, ), hai chấm theo sau bởi dấu cách (: ), chấm phẩy theo sau bởi dấu cách (; ), gạch ngang theo sau bởi dấu cách (- ), đóng ngoặc đơn theo sau bởi dấu cách () ), dấu cách theo sau bởi mở ngoặc đơn (( ).*

Đoạn code dưới đây giúp test hàm của bạn.


```python
s = "Các cơ quan có trách nhiệm khẩn trương xem xét, xử lý vụ việc bảo đảm khách quan, chính xác; " +\
    "theo quy định của Đảng và pháp luật Nhà nước: với tinh thần kiên quyết, chặt chẽ, làm rõ đến đâu xử lý đến đó, " +\
    "đúng người, đúng vi phạm, đúng pháp luật và thu hồi tài sản Nhà nước bị thất thoát."
sg = sentenceToSegments(s) #Kết quả là  1 list
pd.DataFrame(sg)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Các cơ quan có trách nhiệm khẩn trương xem xét,</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xử lý vụ việc bảo đảm khách quan,</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chính xác;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>theo quy định của Đảng và pháp luật Nhà nước:</td>
    </tr>
    <tr>
      <th>4</th>
      <td>với tinh thần kiên quyết,</td>
    </tr>
    <tr>
      <th>5</th>
      <td>chặt chẽ,</td>
    </tr>
    <tr>
      <th>6</th>
      <td>làm rõ đến đâu xử lý đến đó,</td>
    </tr>
    <tr>
      <th>7</th>
      <td>đúng người,</td>
    </tr>
    <tr>
      <th>8</th>
      <td>đúng vi phạm,</td>
    </tr>
    <tr>
      <th>9</th>
      <td>đúng pháp luật và thu hồi tài sản Nhà nước bị ...</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 4. Hàm phụ [3]: Tách thành phần thành tiếng
*Hãy viết hàm **`segmentToUnits(segment)`** nhận đối số **`segment`** là một thành phần dưới dạng `str`, tức một phần tử của list trong output của bài 3, và trả lại kết quả là một list tất cả các tiếng (từ đơn) sau khi bỏ hết các dấu câu, số và kí tự lạ.*

Regular expression (biểu thức chính quy) sau có thể giúp bạn: `r'[,;$\:\^\=\+\-\"\'\(\)\/\@\*\&\%\“\.{1,}\?\!\d]'` (bạn có thể thêm những dấu khác có thể gặp)

Đoạn code dưới đây giúp test hàm của bạn.


```python
e = 'Cho phương trình "x^3 + 3mx^2 + 2x - 1 = 0" tìm m để phương trình có 3 nghiệm phân biệt.'
u = segmentToUnits(e) # Kết quả là 1 list
pd.DataFrame(u)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cho</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phương</td>
    </tr>
    <tr>
      <th>2</th>
      <td>trình</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mx</td>
    </tr>
    <tr>
      <th>5</th>
      <td>x</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tìm</td>
    </tr>
    <tr>
      <th>7</th>
      <td>m</td>
    </tr>
    <tr>
      <th>8</th>
      <td>để</td>
    </tr>
    <tr>
      <th>9</th>
      <td>phương</td>
    </tr>
    <tr>
      <th>10</th>
      <td>trình</td>
    </tr>
    <tr>
      <th>11</th>
      <td>có</td>
    </tr>
    <tr>
      <th>12</th>
      <td>nghiệm</td>
    </tr>
    <tr>
      <th>13</th>
      <td>phân</td>
    </tr>
    <tr>
      <th>14</th>
      <td>biệt</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 5. Tải (load) danh sách từ tiếng Việt

Việc tách câu thành đoạn, đoạn thành thành phần, thành phần thành tiếng có thể xem là các việc thuần tuý kĩ thuật. Thao tác còn lại: ghép tiếng thành từ để thực hiện được cần có sự trợ giúp của một từ điển được xem là chứa tất cả các từ (đơn, ghép, láy) trong tiếng Việt. Ta load từ điển trong bài này.


*Hãy viết hàm **`loadWordSet(wordlist_filename)`** nhận đối số **`wordlist_filename`** là đường dẫn của file chứa danh sách các từ (file này có dạng như `VietnameseDictionary/WordList.txt`), sau đó trả lại một **set** là tập hợp tất cả các từ trong file đó.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
s = loadWordSet(WORDLIST_FILE)
"nhân dân" in s, "người" in s, "công nghệ" in s
```




    (True, True, True)




```python
len(s)
```




    29568



### Bài 6. Hàm phụ [4]: Ghép tiếng thành từ

Thông thường, không có từ điển nào liệt kê đủ mọi từ, bởi các từ mới luôn xuất hiện theo thời gian, và hàng loạt thuật ngữ, từ mượn, thành ngữ, danh từ riêng ... không phải lúc nào cũng xuất hiện trong từ điển.

Kể cả khi tồn tại một từ điển liệt kê đầy đủ các từ, thì việc ghép các từ đơn thành từ phức (ghép, láy) cũng không thể giải quyết tốt 100%, do hiện tượng đồng âm, kiêm nhiệm chức năng... của từ. Ví dụ, xét 2 câu:

- *"Mây được hình thành từ hơi nước."*

- *"Ta chia hình thành 3 phần bằng nhau."*

Trong câu 1, tổ hợp "hình thành" là một từ ghép, còn trong câu 2, tổ hợp "hình thành" là 2 từ đơn. Để biết câu đang xét thuộc trường hợp nào, đó là một bài toán data science khác vượt ngoài phạm vi TD này.

Vì vậy, ta chỉ có thể giải quyết gần đúng bài toán ghép tiếng thành từ. Chẳng hạn, với list các tiếng `['Ta', 'chia', 'hình', 'thành', 'phần', 'bằng', 'nhau']`, ta có thể chấp nhận để chương trình trả lại list các từ sau khi ghép `['ta', 'chia', 'hình thành', 'phần', 'bằng', 'nhau']`.

Một trong những thuật toán có thể ghép gần đúng tiếng thành từ là thuật toán **tham lam** hay **so khớp từ dài nhất** (longest matching) sau: 

- Đọc list các tiếng từ trái sang phải.

- Nếu một tiếng được viết hoa, **có mặt trong từ điển** nhưng không nằm kề một tiếng viết hoa nào, thì chuyển nó thành dạng viết thường. Các trường hợp khác giữ nguyên trạng thái viết hoa hoặc thường của nó: `['Trung', 'tâm', 'Hữu', 'nghị', 'Việt', 'Đức'] -> ['trung', 'tâm', 'hữu', 'nghị', 'Việt', 'Đức']`; `['Anh', 'và', 'Tây', 'Ban', 'Nha', 'và', 'Ireland', 'đều', 'thuộc', 'châu', 'Âu'] -> ['anh', 'và', 'Tây', 'Ban', 'Nha', 'và', 'Ireland', 'đều', 'thuộc', 'châu', 'âu']`.

- Ứng với mỗi tiếng, ghép nó với số lượng lớn nhất các tiếng ở sát bên phải nó sao cho tổ hợp tạo thành hoặc là một từ có mặt trong từ điển, hoặc là một dãy viết hoa liên tiếp cực đại. `['nhân', 'dân', 'ta', 'có', 'một', 'lòng', 'nồng', 'nàn', 'yêu', 'nước'] -> ['nhân dân', 'ta', 'có', 'một lòng', 'nồng nàn', 'yêu', 'nước']`; `['anh', 'và', 'Tây', 'Ban', 'Nha', 'và', 'Ireland', 'đều', 'thuộc', 'châu', 'âu'] -> ['anh', 'và', 'Tây Ban Nha', 'và', 'Ireland', 'đều', 'thuộc', 'châu', 'âu']`

 (tại các vị trí **nhân**, **một**, **nồng**; tổ hợp **nhân dân**, **một lòng**, **nồng nàn** có mặt trong từ điển còn **nhân dân ta**, **một lòng nồng**, **nồng nàn yêu** không có mặt trong từ điển, cũng không được viết hoa; tổ hợp "Tây Ban Nha" được viết hoa còn **Tây Ban Nha và** không được viết hoa, v.v.).

- Cuối cùng, nếu bản thân tiếng đó không có mặt trong từ điển và cũng không ghép được với tổ hợp nào ở sát bên phải để tạo từ, ta cũng xem nó là một từ: `['tích', 'xy', 'của', 'x', 'và', 'y'] -> ['tích', 'xy', 'của', 'x', 'và', 'y']` (**xy** trở thành từ dù nó không có trong từ điển)


*Hãy viết hàm **`unitsToWords(units, word_set)`** nhận đối số **`units`** là list các tiếng như output của bài 4 và **`word_set`** là danh sách các từ dưới dạng set như output của bài 5, sau đó thực hiện thuật toán tham lam nêu trên để trả lại list các từ được tạo thành.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
e = 'Cho phương trình "x^3 + 3mx^2 + 2x - 1 = 0" tìm m để phương trình có 3 nghiệm phân biệt.'
u = segmentToUnits(e)
s = loadWordSet(WORDLIST_FILE) # Kết quả là 1 set
w = unitsToWords(u, s) # Kết quả là 1 list
pd.DataFrame(w)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cho</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phương trình</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mx</td>
    </tr>
    <tr>
      <th>4</th>
      <td>x</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tìm</td>
    </tr>
    <tr>
      <th>6</th>
      <td>m</td>
    </tr>
    <tr>
      <th>7</th>
      <td>để</td>
    </tr>
    <tr>
      <th>8</th>
      <td>phương trình</td>
    </tr>
    <tr>
      <th>9</th>
      <td>có</td>
    </tr>
    <tr>
      <th>10</th>
      <td>nghiệm</td>
    </tr>
    <tr>
      <th>11</th>
      <td>phân biệt</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 7. Tổng hợp các bước: biến văn bản thành túi từ

Bây giờ ta có thể tổng hợp tất cả các bước từ các bài trên để từ file dữ liệu tổng hợp (`FullData/Sport2017_Solution.txt`) tạo ra các túi từ.

*Viết hàm **`getWordFrequencyFromArticles(data_file, word_list_file)`** nhận đối số **`data_file`** là đường dẫn của file dữ liệu tổng hợp và **`word_list_file`** là đường dẫn của file danh sách từ (`VietnameseDictionary/WordList.txt`), và trả lại một **list** có số phần tử bằng số bài báo trong **`data_file`**, mỗi phần tử là một **dict** của Python có dạng **từ : tần số xuất hiện của từ trong nội dung bài báo**).*

Đoạn code dưới đây giúp test hàm của bạn. (Thời gian chạy khoảng 1-2 phút)


```python
freq_dict = getWordFrequencyFromArticles(ALL_DATA_FILE, WORDLIST_FILE) #Kết quả freq_dict là một list các dict
len(freq_dict)
```




    6748




```python
c1 = pd.Series(freq_dict[0].keys()) #freq_dict[0] là một dict, lấy tất cả các key (từ) của nó
c2 = pd.Series(freq_dict[0].values()) # Lấy tần số tương ứng
pd.DataFrame([c1, c2]) # Quan sát bằng DataFrame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>276</th>
      <th>277</th>
      <th>278</th>
      <th>279</th>
      <th>280</th>
      <th>281</th>
      <th>282</th>
      <th>283</th>
      <th>284</th>
      <th>285</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bảo</td>
      <td>và</td>
      <td>lập</td>
      <td>thời</td>
      <td>đổ nát</td>
      <td>tham mưu</td>
      <td>xem</td>
      <td>bóng</td>
      <td>đầu</td>
      <td>còn</td>
      <td>...</td>
      <td>quan</td>
      <td>hàng</td>
      <td>là</td>
      <td>hoàn thành</td>
      <td>ngày</td>
      <td>kiêm</td>
      <td>cấp</td>
      <td>ghi</td>
      <td>đuôi</td>
      <td>Bắc Kỳ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>14</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 286 columns</p>
</div>



Ví dụ trên cho thấy nội dung bài báo thứ nhất đã được chuyển đổi thành các bộ từ : tần số. Từ *"Bảo"* xuất hiện 1 lần, từ *"và"* xuất hiện 5 lần, từ *"tham mưu"* xuất hiện 1 lần v.v...

### Bài 8. Lưu túi từ vào file

Để các bước tiếp theo không phải thực hiện lại quá trình preprocessing1 này, ta lưu kết quả đã thực hiện vào file (Trong thực tế, người ta lưu vào cơ sở dữ liệu thay vì file). Ta sẽ cần 2 file:

- Một file như `FullData/FrequencyByNewsArticle_Solution.txt` chứa tiêu đề các bài báo và túi từ tương ứng của nó.
- Một file như `FullData/GlobalNewsFrequency_Solution.txt` chứa tổng tần số của tất cả các từ xuất hiện ít nhất trong 1 bài báo, lấy trên tất cả các bài báo.

*Viết hàm **`saveWordFrequencyToFile(data_file, word_list_file, frequency_file, global_frequency_file)`** nhận đối số **`data_file`**, **`word_list_file`** là file các bài báo và file danh sách từ, rồi thực hiện việc biến các bài báo thành list các túi từ rồi ghi vào **`frequency_file`** theo quy tắc: Mỗi hàng tương ứng là một bài báo gồm: tiêu đề bài báo, theo sau bởi hai dấu tab (`\t\t`), theo sau bởi các cặp từ:tần số (ở giữa là dấu hai chấm), các cặp này cách nhau bởi 1 dấu tab (`\t`). Ví dụ hàng đầu tiên có dạng: **tieude1**\t\t**tu1**:**tanso1**\t**tu2**:**tanso2**...\t**tuN**:**tansoN**\n. *

*Tiếp theo, hàm tính tổng tần số của các từ trên tất cả các bài báo và lưu vào **`global_frequency_file`** sao cho mỗi từ ứng với một hàng, mỗi hàng gồm từ, theo sau bởi dấu hai chấm, theo sau bởi tổng tần số tương ứng. Để thuận tiện cho tính toán sau này, bạn nên lưu theo thứ tự tần số giảm dần, hàng tương ứng với từ có tần số cao hơn xuất hiện trước.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
saveWordFrequencyToFile(ALL_DATA_FILE, WORDLIST_FILE, FREQUENCY_FILE, GLOBAL_FREQUENCY_FILE)
```


```python
pd.read_csv(FREQUENCY_FILE, sep="\t\t", names=["Tiêu đề", "Túi từ"]).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tiêu đề</th>
      <th>Túi từ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cơ quan khí tượng triều Nguyễn</td>
      <td>Bảo:1\tvà:5\tlập:1\tthời:5\tđổ nát:1\ttham mưu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
      <td>sai sót:1\ttham dự:2\tđang:1\tphụ cấp:2\tnhận:...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Đường vành đai Hà Nội ùn tắc ngày mùng 4 Tết</td>
      <td>đón:1\tchôn chân:1\tChiều:1\tHưng:1\tKhuất Duy...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
      <td>sức lực:1\tđông:3\tTP HCM:9\ttrở lại:1\tlần:1\...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Người dân dựng biển cảnh báo 'Công an bắn tốc độ'</td>
      <td>đông:1\tđang:1\tCông:1\ttrăm:2\tở:1\tchốt:2\td...</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv(GLOBAL_FREQUENCY_FILE, sep=":", names=["Từ", "Tần số"]).head(20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Từ</th>
      <th>Tần số</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>và</td>
      <td>5973</td>
    </tr>
    <tr>
      <th>1</th>
      <td>được</td>
      <td>5572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cho</td>
      <td>5353</td>
    </tr>
    <tr>
      <th>3</th>
      <td>có</td>
      <td>5264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>trong</td>
      <td>5128</td>
    </tr>
    <tr>
      <th>5</th>
      <td>đã</td>
      <td>5107</td>
    </tr>
    <tr>
      <th>6</th>
      <td>của</td>
      <td>5088</td>
    </tr>
    <tr>
      <th>7</th>
      <td>các</td>
      <td>4840</td>
    </tr>
    <tr>
      <th>8</th>
      <td>không</td>
      <td>4766</td>
    </tr>
    <tr>
      <th>9</th>
      <td>người</td>
      <td>4706</td>
    </tr>
    <tr>
      <th>10</th>
      <td>trên</td>
      <td>4685</td>
    </tr>
    <tr>
      <th>11</th>
      <td>một</td>
      <td>4679</td>
    </tr>
    <tr>
      <th>12</th>
      <td>đến</td>
      <td>4670</td>
    </tr>
    <tr>
      <th>13</th>
      <td>với</td>
      <td>4583</td>
    </tr>
    <tr>
      <th>14</th>
      <td>là</td>
      <td>4568</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ở</td>
      <td>4543</td>
    </tr>
    <tr>
      <th>16</th>
      <td>để</td>
      <td>4464</td>
    </tr>
    <tr>
      <th>17</th>
      <td>đó</td>
      <td>4229</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ra</td>
      <td>4189</td>
    </tr>
    <tr>
      <th>19</th>
      <td>vào</td>
      <td>4133</td>
    </tr>
  </tbody>
</table>
</div>



## Phần 2: Preprocessing 2 - Số hoá túi từ

Từ thời điểm này, ta chỉ cần làm việc với 3 file, trong đó 2 file sau chính là kết quả của bước 1.
```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyByNewsArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalNewsFrequency.txt"
```

Để thuận tiện cho việc thống nhất số liệu để so sánh kết quả với hướng dẫn của TD, bạn có thể dùng 2 file `..._Solution.txt` và đổi tên thành các file trên.


### Bài 9. Giảm số chiều bằng cách chọn các từ quan trọng

Để các thuật toán clustering có thể chạy được, ta phải biến các túi từ thành các vector, trong đó mỗi toạ độ thể hiện một từ. Vấn đề đầu tiên đặt ra: ta sử dụng hết tất cả các từ hiện có hay có thể chọn lọc trước một số từ quan trọng. Đây là bài toán giảm số chiều (reduction of dimension) trong machine learning. Ta chưa đi sâu vào các kĩ thuật cho bài toán này, mà sử dụng phương pháp đơn giản dựa vào quan sát file `FullData/GlobalNewsFrequency.txt`.

Ta thấy:

- Những từ có tần số quá lớn thường không có tính phân loại, tức không quan trọng. Ví dụ các từ "và", "của", "hoặc"... có thể xuất hiện trong rất nhiều bài báo, không thể căn cứ vào đó để chia nhóm nội dung.

- Những từ có tần số quá nhỏ cũng không quan trọng. Đó là những từ chỉ xuất hiện đơn lẻ trong một vài bài báo, không đặc trưng cho một nhóm.

- Những từ xuất hiện đều đặn trong các bài báo (có tần số gần như nhau trong hầu hết tất cả bài báo) thường cũng không thể được dùng để phân loại các bài báo. 

Do đó, ta sẽ chọn ra những từ thoả mãn điều kiện sau:

- Có tần số nhỏ hơn hoặc bằng một giá trị chặn trên **upperbound**

- Có tần số lớn hơn hoặc bằng một giá trị chặn dưới **lowerbound**

- Có phương sai của list tần số lấy trên tất cả các bài báo lớn hơn hoặc bằng một giá trị **var_lowerbound**

Giả sử sau khi chọn xong, ta giữ lại được 5 từ (`"kinh tế", "xã hội", "Việt Nam", "Hà Nội", "TPHCM")`, ta có thể gán cho chúng 5 toạ độ, ví dụ `("kinh tế" -> 0, "Hà Nội" -> 1, "TPHCM" -> 2, "Việt Nam" -> 3, "xã hội" -> 4)`. Khi đó túi từ `("Việt Nam":2, "kinh tế":1, "tăng trưởng": 2, "là": 1)` sẽ được biến thành vector `[1, 0, 0, 2, 0]`.

*Hãy viết hàm **`getExplicativeFeatures(global_frequency_file, frequency_file, lowerbound, upperbound, var_lower_bound)`** nhận 5 đối số theo thứ tự là đường dẫn file túi từ theo bài báo, đường dẫn file tần số tổng quát, chặn dưới của tần số cần chọn, chặn trên của tần số cần chọn, chặn dưới của phương sai cần chọn, và trả lại một **dict** của Python gồm các key là các từ "có tính giải thích" được chọn và value tương ứng là số thứ tự của toạ độ.*

Trong ví dụ trên, kết quả trả lại cần là: `{"kinh tế": 0, "Hà Nội": 1, "TPHCM": 2, "Việt Nam": 3, "xã hội": 4}`

Đoạn code dưới đây giúp kiểm tra hàm của bạn.


```python
MIN_AVG = 300
MAX_AVG = 1500
MIN_DEV = 0.8
```


```python
features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
features
```




    {'Ban': 39,
     'B\xe1\xbb\x99': 6,
     'C': 76,
     'Ch\xc3\xadnh': 108,
     'Qu\xe1\xbb\x91c': 90,
     'TP HCM': 65,
     'Th\xc3\xa0nh': 46,
     'Th\xe1\xbb\xa7': 105,
     'Trung': 75,
     'T\xe1\xbb\x95ng': 22,
     'UBND': 63,
     'Vi\xe1\xbb\x87t Nam': 24,
     'anh': 86,
     'ban': 70,
     'bi\xe1\xbb\x83n': 16,
     'b\xc3\xa0': 110,
     'b\xc3\xa3o': 117,
     'b\xe1\xbb\x99': 122,
     'ch\xc3\xa1y': 114,
     'ch\xc3\xadnh': 26,
     'ch\xe1\xbb\x8b': 101,
     'con': 79,
     'c\xc3\xa1': 126,
     'c\xc3\xa1n b\xe1\xbb\x99': 74,
     'c\xc3\xa2y': 115,
     'c\xc3\xb4ng tr\xc3\xacnh': 9,
     'c\xe1\xba\xa7n': 27,
     'c\xe1\xba\xa7u': 29,
     'c\xe1\xbb\xa9u': 21,
     'doanh': 54,
     'du l\xe1\xbb\x8bch': 58,
     'd\xe1\xbb\xb1 \xc3\xa1n': 77,
     'em': 41,
     'gia \xc4\x91\xc3\xacnh': 125,
     'giao th\xc3\xb4ng': 8,
     'gi\xe1\xba\xa3i': 14,
     'gi\xe1\xbb\x9d': 82,
     'ho\xe1\xba\xa1ch': 17,
     'ho\xe1\xba\xa1t \xc4\x91\xe1\xbb\x99ng': 36,
     'h\xc3\xb3a': 20,
     'h\xe1\xbb\x8dc': 33,
     'h\xe1\xbb\x8dc sinh': 59,
     'h\xe1\xbb\x93': 47,
     'h\xe1\xbb\x99': 48,
     'h\xe1\xbb\x99i': 78,
     'khoa': 61,
     'kh\xc3\xa1ch': 95,
     'kh\xc3\xad': 50,
     'k\xe1\xbb\xb3': 67,
     'lao \xc4\x91\xe1\xbb\x99ng': 0,
     'l\xc3\xa3nh \xc4\x91\xe1\xba\xa1o': 55,
     'l\xc5\xa9': 28,
     'l\xe1\xbb\x99': 119,
     'l\xe1\xbb\x9bp': 83,
     'm\xc3\xa1y': 40,
     'm\xc6\xb0a': 7,
     'm\xe1\xba\xa1nh': 15,
     'm\xe1\xbb\x97i': 25,
     'm\xe1\xbb\xa9c': 60,
     'nghi\xc3\xaan': 72,
     'ngh\xe1\xbb\x81': 49,
     'ngh\xe1\xbb\x89': 69,
     'nguy\xc3\xaan': 93,
     'ng\xc3\xa0nh': 10,
     'ng\xe1\xba\xadp': 13,
     'ph\xc3\xa1t tri\xe1\xbb\x83n': 124,
     'ph\xc3\xad': 32,
     'ph\xc3\xb2ng': 42,
     'ph\xc6\xb0\xc6\xa1ng \xc3\xa1n': 18,
     'ph\xc6\xb0\xe1\xbb\x9dng': 56,
     'ph\xe1\xba\xa1m': 106,
     'ph\xe1\xbb\xa7': 84,
     'quy\xe1\xbb\x81n': 116,
     'qu\xc3\xa2n': 94,
     'qu\xe1\xba\xadn': 85,
     'r\xe1\xba\xb1ng': 81,
     's\xc3\xa2n bay': 12,
     's\xc3\xb4ng': 123,
     's\xe1\xbb\xad d\xe1\xbb\xa5ng': 30,
     's\xe1\xbb\xb1': 87,
     'thanh tra': 118,
     'thi': 99,
     'thu': 97,
     'th\xc3\xaam': 68,
     'th\xc3\xb4ng': 51,
     'th\xc3\xb4ng tin': 80,
     'th\xc6\xb0': 38,
     'ti\xe1\xbb\x81n': 120,
     'tra': 11,
     'tri\xe1\xbb\x87u': 121,
     'tr\xc6\xb0\xe1\xbb\x9dng': 100,
     'tr\xc6\xb0\xe1\xbb\x9fng': 112,
     'tr\xe1\xba\xa1m': 2,
     'tr\xe1\xba\xbb': 111,
     'tr\xe1\xbb\x93ng': 98,
     'tuy\xe1\xba\xbfn': 31,
     't\xc3\xa0i x\xe1\xba\xbf': 89,
     't\xc3\xa0u': 109,
     't\xc3\xacm': 104,
     't\xc3\xb4i': 35,
     't\xc4\x83ng': 102,
     't\xc6\xb0\xe1\xbb\x9bng': 4,
     't\xe1\xba\xbf': 3,
     't\xe1\xbb\x8bch': 66,
     't\xe1\xbb\xb7': 113,
     'vi': 91,
     'vi\xc3\xaan': 1,
     'v\xe1\xbb\x89a h\xc3\xa8': 62,
     'v\xe1\xbb\x91n': 23,
     'x\xc3\xa3 h\xe1\xbb\x99i': 107,
     'x\xe1\xbb\xad l\xc3\xbd': 92,
     '\xc3\xb4t\xc3\xb4': 45,
     '\xc4\x90\xc3\xa0 N\xe1\xba\xb5ng': 52,
     '\xc4\x90\xe1\xba\xa1i': 19,
     '\xc4\x90\xe1\xba\xa3ng': 53,
     '\xc4\x91i\xe1\xbb\x83m': 73,
     '\xc4\x91i\xe1\xbb\x87n': 57,
     '\xc4\x91o\xc3\xa0n': 34,
     '\xc4\x91\xc3\xb3ng': 44,
     '\xc4\x91\xc3\xb4 th\xe1\xbb\x8b': 88,
     '\xc4\x91\xc6\xa1n v\xe1\xbb\x8b': 103,
     '\xc4\x91\xe1\xba\xa5t': 64,
     '\xc4\x91\xe1\xba\xa7u t\xc6\xb0': 71,
     '\xc4\x91\xe1\xbb\x8ba ph\xc6\xb0\xc6\xa1ng': 96,
     '\xc4\x91\xe1\xbb\x99': 43,
     '\xc6\xb0\xc6\xa1ng': 37,
     '\xe1\xbb\xa7y': 5}




```python
# Xem dưới dạng bảng để rõ hơn: hàng 1 là các từ, hàng 2 là chỉ số của chúng trong từ điển
c1 = pd.Series(features.keys()) 
c2 = pd.Series(features.values()) 
pd.DataFrame([c1, c2]) #Brazil ứng với toạ độ 12, Neymar ứng với toạ độ 9, v.v... Của bạn có thể khác
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>trường</td>
      <td>lao động</td>
      <td>TP HCM</td>
      <td>công trình</td>
      <td>viên</td>
      <td>trạm</td>
      <td>tế</td>
      <td>tịch</td>
      <td>ương</td>
      <td>tướng</td>
      <td>...</td>
      <td>tôi</td>
      <td>khoa</td>
      <td>vỉa hè</td>
      <td>hoạt động</td>
      <td>địa phương</td>
      <td>thu</td>
      <td>UBND</td>
      <td>đất</td>
      <td>thi</td>
      <td>phát triển</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>0</td>
      <td>65</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>37</td>
      <td>4</td>
      <td>...</td>
      <td>35</td>
      <td>61</td>
      <td>62</td>
      <td>36</td>
      <td>96</td>
      <td>97</td>
      <td>63</td>
      <td>64</td>
      <td>99</td>
      <td>124</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 127 columns</p>
</div>



### Bài 10. Biến túi từ thành các vector có toạ độ 0/1

Ta cần phân nhóm các bài báo theo độ tương đồng về nội dung. Theo đó, các vector ứng với các bài báo tương đồng (sử dụng chung một lượng từ "quan trọng" giống nhau) cần có khoảng cách nhỏ, còn các vector ứng với các bài báo không tương đồng có khoảng cách lớn. 

Vì điều kiện này, việc số hoá các túi từ thành các vector có toạ độ là tần số các từ không phải là lựa chọn tốt. 

Ví dụ xét 3 túi từ: 

- Túi từ `A {"Việt Nam": 1, "xã hội": 2}`
- Túi từ `B {"Việt Nam": 2, "xã hội": 4}`
- Túi từ `C {"Hà Nội": 1, "TPHCM": 1}`

Nếu biến mỗi túi từ thành vector tần số các từ thì `A, B, C` trở thành `[0, 0, 0, 1, 1]`, `[0, 0, 0, 2, 4]`, `[0, 1, 1, 0, 0]`. Khi đó `AB`$ = \sqrt{10} > $ `AC`$= 2$, trong khi `A` tương đồng với `B` hơn `C`.

Để khắc phục bất tiện này, ta nghĩ đến phương pháp sử dụng phép số hoá 0-1. Nếu một từ "quan trọng" có mặt trong bài báo (túi từ), toạ độ tương ứng của nó bằng 1. Nếu không, toạ độ tương ứng bằng 0.

Với ví dụ trên, `A, B, C` trở thành `[0, 0, 0, 1, 1]`, `[0, 0, 0, 1, 1]`, `[0, 1, 1, 0, 0]`. Khi đó `AB = 0, AC = 2`, một kết quả hợp lí hơn.

*Hãy viết hàm **`articlesToSparseVector(frequency_file, features_dict, coordinates_coding_mode = "0-1")`** nhận đối số **`frequency_file`** là file túi từ, **`features_dict`** là từ điển các từ quan trọng như output của bài 9, **coordinates_coding_mode** là một str cho biết cách số hoá, có giá trị "0-1" trong bài này và sẽ thay đổi ở một số bài sau; và trả lại một numpy array hoặc một sparse matrix hoặc một DataFrame (tuỳ bạn lựa chọn) tương ứng với một ma trận có số hàng bằng số bài báo, mỗi hàng là vector 0-1 đã được chuyển đổi từ túi từ của bài báo đó.*

Đoạn code dưới đây giúp test hàm của bạn (trong đó `X` được chọn là một sparse matrix)


```python
COORDINATES_CODING_MODE = "0-1"
features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
pd.DataFrame(X.toarray()).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 127 columns</p>
</div>



### Bài 11. Hàm phụ - Lấy tiêu đề của các bài báo

*Để tiện cho việc phân tích kết quả sau này, hãy viết hàm **`getTitles(frequency_file)`** nhận đối số **`frequency_file`** là file túi từ, và trả lại list tiêu đề theo đúng thứ tự.*


```python
tt = getTitles(FREQUENCY_FILE)
pd.DataFrame(tt).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cơ quan khí tượng triều Nguyễn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Đường vành đai Hà Nội ùn tắc ngày mùng 4 Tết</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Người dân dựng biển cảnh báo 'Công an bắn tốc độ'</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nữ sinh bị bạn dùng gạch hành hung đến ngất xỉu</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lễ hội trâu rơm, bò rạ ở đồng bằng sông Hồng</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9 câu hỏi về ông vua dời đô từ Hoa Lư về Thăng...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Đường hoa Nguyễn Huệ kéo dài thêm một ngày</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5 ngày nghỉ Tết, gần 120 người chết vì tai nạn...</td>
    </tr>
  </tbody>
</table>
</div>



## Phần 3 - KMeans

### Bài 12. Chia nhóm với KMeans và khoảng cách Euclide

*Hãy viết hàm **`train(vectors, nb_clusters, model = "KMeans")`** nhận đối số **`vectors`** là output của bài 10, **`nb_clusters`** là số nhóm cần chia, **`model`** là tên mô hình (trong bài này là **KMeans** by default), và trả lại mô hình dự đoán (predictive model) ứng với thuật toán KMeans với khoảng cách Euclide.*

*Sau đó, hãy viết hàm **`predict(predictive_model, vectors)`** để từ **`predictive_model`** là kết quả của hàm **`train`** trên, trả lại dự đoán của mô hình dành cho **`vectors`** dưới dạng một list gồm các số thứ tự của nhóm tương ứng với từng vector (bài báo).*

Đoạn code dưới đây giúp test hàm của bạn.


```python
COORDINATES_CODING_MODE = "0-1"
NB_CLUSTER = 4
MODEL = "KMeans"
MIN_AVG = 300
MAX_AVG = 1500
MIN_DEV = 0.8

features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
predictive_model = train(X, NB_CLUSTER, model=MODEL)
prediction = predict(predictive_model, X)
prediction
```




    array([3, 1, 3, ..., 2, 3, 3])




```python
titles = getTitles(FREQUENCY_FILE)
c1 = pd.Series(titles)
c2 = pd.Series(prediction)
pd.concat([c1, c2], axis=1).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cơ quan khí tượng triều Nguyễn</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Đường vành đai Hà Nội ùn tắc ngày mùng 4 Tết</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Người dân dựng biển cảnh báo 'Công an bắn tốc độ'</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nữ sinh bị bạn dùng gạch hành hung đến ngất xỉu</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lễ hội trâu rơm, bò rạ ở đồng bằng sông Hồng</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9 câu hỏi về ông vua dời đô từ Hoa Lư về Thăng...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Đường hoa Nguyễn Huệ kéo dài thêm một ngày</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5 ngày nghỉ Tết, gần 120 người chết vì tai nạn...</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Theo kết quả trên, bài báo 0 và 2 thuộc cùng một nhóm (ta chưa biết vì sao chúng thuộc cùng một nhóm).

### Bài 13. Quan sát kết quả [1] - Nhìn vào từng nhóm

Bảng trên cho phép ta biết bài báo nào được phân vào nhóm nào, nhưng khá "khó nhìn" để phân tích kế quả. Ta tìm cách biểu diễn ở một dạng dễ phân tích hơn. 

*Hãy viết hàm **`getClusters(titles, prediction)`** nhận đối số **`titles`** là list tiêu đề các bài báo theo thứ tự, **`prediction`** là kết quả chia nhóm ở bài 12 dưới dạng một array (như array([0, 2, 0, ..., 4, 2, 3]), sau đó trả lại kết quả là một list gồm $k$ phần tử, trong đó phần tử thứ $i$ là một list tiêu đề các bài báo thuộc nhóm $i$.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
clusters = getClusters(titles, prediction)
pd.DataFrame(clusters[0]).head(10) #Các bài báo thuộc nhóm 0...
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nữ sinh bị bạn dùng gạch hành hung đến ngất xỉu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Đường hoa Nguyễn Huệ kéo dài thêm một ngày</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bia cổ ghi công trạng vị vua anh minh bậc nhất...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vị thần 'trấn Bắc' của kinh thành Thăng Long xưa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cựu chiến binh hiến 2.000 m2 đất mở đường</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vụ án 'hóa hổ' giết vua và vị Trạng nguyên đầu...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Tên đường ở Sài Gòn xưa được đặt như thế nào</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nam sinh người Mông ăn mì tôm dành tiền nghiên...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Người Cơ Tu đốt trứng gà chọn đất tốt</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Công trường khai thác đá cổ xây thành nhà Hồ</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(clusters[1]).head(10) #Các bài báo thuộc nhóm 1...
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bộ trưởng Khoa học: Cần thay đổi tư duy chiến ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thứ trưởng Hồ Thị Kim Thoa bị khiển trách</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Người tuyển dụng 'vụ phó 26 tuổi' bị kiểm tra ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Phần mềm giám định chống trục lợi thẻ bảo hiểm...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thứ trưởng Hồ Thị Kim Thoa bị xem xét kỷ luật ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thủ tướng: 'Hạn chế xây cao ốc ở trung tâm TP ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ông Vũ Huy Hoàng bị xóa tư cách nguyên Bộ trưở...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Trụ sở nhiều bộ ngành sẽ được xây dựng trên kh...</td>
    </tr>
  </tbody>
</table>
</div>



Như vậy, ta đã gộp tiêu đề các bài báo thuộc cùng một nhóm với nhau để thử xem chúng có liên quan gì. Nhưng cách nhìn này vẫn chưa cho phép phân tích kết quả.

### Bài 14. Quan sát kết quả [2] - Nhìn vào tâm của từng nhóm

Như ta biết, thành phần toạ độ của mỗi vector chỉ có thể là 0 hoặc 1. Do đó tâm của mỗi nhóm là một vector có các thành phần nằm giữa 0 và 1. Thành phần nào càng lớn (càng gần 1) thì các bài báo ứng với nhóm đó càng hay chứa từ quan trọng tương ứng. Nghĩa là các bài báo tương đồng với nhau dựa trên việc dùng chung từ quan trọng đó.

Vậy, tâm của mỗi nhóm cho ta thông tin về nguyên nhân của sự tương đồng trong nhóm.

*Hãy viết hàm **`getClusterCenters(predictive_model, vectors, prediction)`** nhận đối số **`predictive_model`** là mô hình dự đoán đã học (train) từ bài 12, **`vectors`** là output từ bài 10 (ma trận 0-1), **`prediction`** là kết quả của việc chia nhóm (output của bài 12) và trả lại một array hoặc DataFrame gồm $k$ hàng, mỗi hàng là vector tâm của mỗi nhóm.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
centers = getClusterCenters(predictive_model, X, prediction) #Một array hoặc DataFrame
pd.DataFrame(centers)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.110735</td>
      <td>0.226543</td>
      <td>0.027895</td>
      <td>0.295013</td>
      <td>0.039730</td>
      <td>0.078614</td>
      <td>0.110735</td>
      <td>0.102282</td>
      <td>0.052409</td>
      <td>0.103973</td>
      <td>...</td>
      <td>0.048183</td>
      <td>0.036348</td>
      <td>0.045647</td>
      <td>0.246830</td>
      <td>0.293322</td>
      <td>0.172443</td>
      <td>0.110735</td>
      <td>0.224852</td>
      <td>0.397295</td>
      <td>0.078614</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.128008</td>
      <td>0.341675</td>
      <td>0.042348</td>
      <td>0.528393</td>
      <td>0.431184</td>
      <td>0.312801</td>
      <td>0.450433</td>
      <td>0.045236</td>
      <td>0.148219</td>
      <td>0.154957</td>
      <td>...</td>
      <td>0.052936</td>
      <td>0.174206</td>
      <td>0.035611</td>
      <td>0.178056</td>
      <td>0.215592</td>
      <td>0.365736</td>
      <td>0.078922</td>
      <td>0.398460</td>
      <td>0.127045</td>
      <td>0.053898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.031050</td>
      <td>0.102283</td>
      <td>0.211872</td>
      <td>0.233790</td>
      <td>0.156164</td>
      <td>0.096804</td>
      <td>0.114155</td>
      <td>0.090411</td>
      <td>0.519635</td>
      <td>0.357991</td>
      <td>...</td>
      <td>0.028311</td>
      <td>0.050228</td>
      <td>0.223744</td>
      <td>0.234703</td>
      <td>0.268493</td>
      <td>0.265753</td>
      <td>0.166210</td>
      <td>0.196347</td>
      <td>0.058447</td>
      <td>0.028311</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.027106</td>
      <td>0.068785</td>
      <td>0.034392</td>
      <td>0.078694</td>
      <td>0.032935</td>
      <td>0.034975</td>
      <td>0.039930</td>
      <td>0.147187</td>
      <td>0.156806</td>
      <td>0.042553</td>
      <td>...</td>
      <td>0.069368</td>
      <td>0.015447</td>
      <td>0.123579</td>
      <td>0.056252</td>
      <td>0.107549</td>
      <td>0.088895</td>
      <td>0.107549</td>
      <td>0.038181</td>
      <td>0.102303</td>
      <td>0.075488</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 127 columns</p>
</div>



*Hãy viết tiếp hàm **`getExplicatveFeaturesForEachCluster(predictive_model, vectors, prediction, explicative_features)`** nhận các đối số **`predictive_model`** (mô hình học từ hàm **`train`** ở bài 12), **`vectors`** (ma trận  0-1 như output bài 10), **`prediction`** (kết quả chia nhóm như output hàm **`predict`** ở bài 12), **`explicative_features`** (từ điển các từ quan trọng như output bài 9), và trả lại một numpy array hoặc DataFrame gồm $k$ hàng ứng với $k$ cluster, mỗi phần tử là một list các tuple (từ quan trọng, toạ độ tương ứng) xếp theo thứ tự giảm dần của toạ độ.*

Đoạn code sau giúp test hàm của bạn.


```python
features_by_cluster = getExplicatveFeaturesForEachCluster(predictive_model, X, prediction, features)
pd.DataFrame(features_by_cluster)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(học, 0.618765849535)</td>
      <td>(trường, 0.485207100592)</td>
      <td>(mỗi, 0.42180896027)</td>
      <td>(con, 0.409129332206)</td>
      <td>(gia đình, 0.39729501268)</td>
      <td>(sự, 0.395604395604)</td>
      <td>(tôi, 0.392223161454)</td>
      <td>(em, 0.377852916314)</td>
      <td>(thêm, 0.372781065089)</td>
      <td>(rằng, 0.345731191885)</td>
      <td>...</td>
      <td>(tướng, 0.039729501268)</td>
      <td>(Đảng, 0.039729501268)</td>
      <td>(ương, 0.0380388841927)</td>
      <td>(thanh tra, 0.0363482671175)</td>
      <td>(đô thị, 0.028740490279)</td>
      <td>(trạm, 0.0278951817413)</td>
      <td>(cháy, 0.0278951817413)</td>
      <td>(vỉa hè, 0.0211327134404)</td>
      <td>(sân bay, 0.0177514792899)</td>
      <td>(tài xế, 0.00676246830093)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Chính, 0.611164581328)</td>
      <td>(phủ, 0.55822906641)</td>
      <td>(hội, 0.557266602502)</td>
      <td>(tế, 0.528392685274)</td>
      <td>(rằng, 0.528392685274)</td>
      <td>(trưởng, 0.526467757459)</td>
      <td>(quyền, 0.512030798845)</td>
      <td>(cần, 0.48604427334)</td>
      <td>(lãnh đạo, 0.484119345525)</td>
      <td>(sự, 0.479307025987)</td>
      <td>...</td>
      <td>(lũ, 0.0394610202117)</td>
      <td>(lộ, 0.0356111645813)</td>
      <td>(hồ, 0.0298363811357)</td>
      <td>(lớp, 0.0279114533205)</td>
      <td>(ngập, 0.0259865255053)</td>
      <td>(học sinh, 0.0259865255053)</td>
      <td>(cháy, 0.0230991337825)</td>
      <td>(vỉa hè, 0.0230991337825)</td>
      <td>(chị, 0.0163618864293)</td>
      <td>(tài xế, 0.0125120307988)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(đầu tư, 0.649315068493)</td>
      <td>(tuyến, 0.568949771689)</td>
      <td>(thông, 0.536073059361)</td>
      <td>(dự án, 0.531506849315)</td>
      <td>(giao thông, 0.519634703196)</td>
      <td>(tỷ, 0.493150684932)</td>
      <td>(quận, 0.437442922374)</td>
      <td>(đơn vị, 0.3899543379)</td>
      <td>(TP HCM, 0.38904109589)</td>
      <td>(phí, 0.382648401826)</td>
      <td>...</td>
      <td>(bão, 0.0283105022831)</td>
      <td>(cá, 0.0283105022831)</td>
      <td>(em, 0.0237442922374)</td>
      <td>(nghề, 0.0228310502283)</td>
      <td>(lớp, 0.02100456621)</td>
      <td>(chị, 0.0200913242009)</td>
      <td>(lũ, 0.0200913242009)</td>
      <td>(trẻ, 0.0191780821918)</td>
      <td>(Đảng, 0.013698630137)</td>
      <td>(học sinh, 0.013698630137)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(biển, 0.23754007578)</td>
      <td>(khách, 0.192946662781)</td>
      <td>(ôtô, 0.176916350918)</td>
      <td>(Trung, 0.174293208977)</td>
      <td>(máy, 0.170212765957)</td>
      <td>(giao thông, 0.156805596036)</td>
      <td>(giờ, 0.155639755173)</td>
      <td>(mạnh, 0.155348294958)</td>
      <td>(anh, 0.152142232585)</td>
      <td>(quận, 0.15185077237)</td>
      <td>...</td>
      <td>(lao động, 0.0271058000583)</td>
      <td>(học sinh, 0.0268143398426)</td>
      <td>(thư, 0.0259399591956)</td>
      <td>(dự án, 0.0259399591956)</td>
      <td>(phí, 0.0238997376858)</td>
      <td>(vốn, 0.0204022150976)</td>
      <td>(nghiên, 0.0174876129408)</td>
      <td>(Đảng, 0.0166132322938)</td>
      <td>(thanh tra, 0.0154473914311)</td>
      <td>(thi, 0.0154473914311)</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 127 columns</p>
</div>



Đến đây ta cảm thấy rõ hơn về cách chia nhóm do KMeans thực hiện:

- Nhóm 0 có vẻ liên quan đến giáo dục và trẻ em (nhìn những từ quan trọng nhất với toạ độ của nó)

- Nhóm 1 có vẻ liên quan đến chính phủ

- Nhóm 2 có vẻ liên quan đến đầu tư dự án

- Nhóm 3 có vẻ liên quan đến giao thông, nhưng để ý rằng toạ độ các từ quan trọng nhỏ hơn hẳn so với 3 nhóm trên. Nhóm này có thể là tổ hợp của một số nhóm khác nhau hoặc bao gồm những từ không phân loại được nhóm.

Thứ tự các nhóm có thể thay đổi trong chương trình của bạn.

## Phần 4 - Thử nghiệm các thao tác khác

### Bài 15 - Thay đổi k

Việc chọn $k=4$ có thể dẫn đến một số nhóm bị gộp vào một nhóm chung (thường là nhóm ở gốc toạ độ 0). Ta thử thay đổi $k$ xem kết quả có tiến triển không.

*Bạn hãy thử thay đổi $k$, tính thế năng tương ứng, vẽ đồ thị thế năng theo $k$ và nhận xét. Không có hàm nào cần viết thêm trong bài này.*

Kết quả có dạng như hình sau:

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson5/TD/figure_3.png" width=600></img>

Thực tế, đồ thị trên không cho ta rõ ràng thông tin nên chọn $k$ bằng bao nhiêu. $k=5, 10, 20$ đều có thể là những lựa chọn hợp lí. Ví dụ, với $k = 10.$


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyByNewsArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalNewsFrequency.txt"
COORDINATES_CODING_MODE = "0-1"
NB_CLUSTER = 10
MODEL = "KMeans"
MIN_AVG = 300
MAX_AVG = 1500
MIN_DEV = 0.8

features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
titles = getTitles(FREQUENCY_FILE)

predictive_model = train(X, NB_CLUSTER, model=MODEL)
prediction = predict(predictive_model, X)
clusters = getClusters(titles, prediction)

centers = getClusterCenters(predictive_model, X, prediction)
explicativeFeatures = getExplicatveFeaturesForEachCluster(predictive_model, X, prediction, features)
```


```python
pd.DataFrame(explicativeFeatures)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(khí, 0.867749419954)</td>
      <td>(mưa, 0.849187935035)</td>
      <td>(Trung, 0.712296983759)</td>
      <td>(mạnh, 0.69837587007)</td>
      <td>(biển, 0.587006960557)</td>
      <td>(bão, 0.517401392111)</td>
      <td>(độ, 0.452436194896)</td>
      <td>(ương, 0.436194895592)</td>
      <td>(lũ, 0.419953596288)</td>
      <td>(tăng, 0.408352668213)</td>
      <td>...</td>
      <td>(thi, 0.0116009280742)</td>
      <td>(tài xế, 0.0092807424594)</td>
      <td>(vi, 0.0092807424594)</td>
      <td>(xã hội, 0.0046403712297)</td>
      <td>(tiền, 0.0046403712297)</td>
      <td>(phí, 0.0046403712297)</td>
      <td>(phạm, 0.00232018561485)</td>
      <td>(thanh tra, 0.00232018561485)</td>
      <td>(Đảng, 0.00232018561485)</td>
      <td>(vỉa hè, 0.00232018561485)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(quận, 0.914798206278)</td>
      <td>(xử lý, 0.807174887892)</td>
      <td>(UBND, 0.7533632287)</td>
      <td>(vỉa hè, 0.726457399103)</td>
      <td>(vi, 0.717488789238)</td>
      <td>(phạm, 0.708520179372)</td>
      <td>(đô thị, 0.654708520179)</td>
      <td>(phường, 0.645739910314)</td>
      <td>(TP HCM, 0.556053811659)</td>
      <td>(lãnh đạo, 0.551569506726)</td>
      <td>...</td>
      <td>(trạm, 0.0134529147982)</td>
      <td>(C, 0.0134529147982)</td>
      <td>(học sinh, 0.0134529147982)</td>
      <td>(khí, 0.00896860986547)</td>
      <td>(sông, 0.00896860986547)</td>
      <td>(ngập, 0.00448430493274)</td>
      <td>(bão, 0.0)</td>
      <td>(lũ, 0.0)</td>
      <td>(khoa, 0.0)</td>
      <td>(thi, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(học, 0.784114052953)</td>
      <td>(trường, 0.77800407332)</td>
      <td>(em, 0.682281059063)</td>
      <td>(lớp, 0.582484725051)</td>
      <td>(học sinh, 0.580448065173)</td>
      <td>(con, 0.501018329939)</td>
      <td>(gia đình, 0.480651731161)</td>
      <td>(tôi, 0.454175152749)</td>
      <td>(mỗi, 0.401221995927)</td>
      <td>(giờ, 0.362525458248)</td>
      <td>...</td>
      <td>(phủ, 0.0244399185336)</td>
      <td>(trạm, 0.0224032586558)</td>
      <td>(Đảng, 0.0162932790224)</td>
      <td>(sân bay, 0.0142566191446)</td>
      <td>(đô thị, 0.0142566191446)</td>
      <td>(Thủ, 0.0122199592668)</td>
      <td>(tướng, 0.010183299389)</td>
      <td>(du lịch, 0.010183299389)</td>
      <td>(vỉa hè, 0.010183299389)</td>
      <td>(tài xế, 0.0020366598778)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(UBND, 0.5)</td>
      <td>(quyền, 0.447791164659)</td>
      <td>(tra, 0.410642570281)</td>
      <td>(chính, 0.403614457831)</td>
      <td>(tịch, 0.386546184739)</td>
      <td>(địa phương, 0.36546184739)</td>
      <td>(xử lý, 0.347389558233)</td>
      <td>(lãnh đạo, 0.34437751004)</td>
      <td>(đơn vị, 0.330321285141)</td>
      <td>(phòng, 0.295180722892)</td>
      <td>...</td>
      <td>(nghiên, 0.035140562249)</td>
      <td>(trẻ, 0.0331325301205)</td>
      <td>(bão, 0.0321285140562)</td>
      <td>(học sinh, 0.031124497992)</td>
      <td>(cháy, 0.0281124497992)</td>
      <td>(lũ, 0.0240963855422)</td>
      <td>(ngập, 0.0220883534137)</td>
      <td>(lớp, 0.0190763052209)</td>
      <td>(tài xế, 0.0120481927711)</td>
      <td>(vỉa hè, 0.0110441767068)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(ôtô, 0.691158156912)</td>
      <td>(giao thông, 0.53798256538)</td>
      <td>(tài xế, 0.474470734745)</td>
      <td>(máy, 0.415940224159)</td>
      <td>(lộ, 0.39103362391)</td>
      <td>(khách, 0.328767123288)</td>
      <td>(biển, 0.318804483188)</td>
      <td>(TP HCM, 0.217932752179)</td>
      <td>(mạnh, 0.21295143213)</td>
      <td>(quận, 0.211706102117)</td>
      <td>...</td>
      <td>(cá, 0.00622665006227)</td>
      <td>(lớp, 0.00622665006227)</td>
      <td>(phát triển, 0.00622665006227)</td>
      <td>(bão, 0.00498132004981)</td>
      <td>(lao động, 0.00373599003736)</td>
      <td>(ủy, 0.00373599003736)</td>
      <td>(thư, 0.00249066002491)</td>
      <td>(ương, 0.00124533001245)</td>
      <td>(trồng, 0.00124533001245)</td>
      <td>(Đảng, 0.00124533001245)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(học, 0.769857433809)</td>
      <td>(cứu, 0.716904276986)</td>
      <td>(nghiên, 0.684317718941)</td>
      <td>(Việt Nam, 0.668024439919)</td>
      <td>(sự, 0.627291242363)</td>
      <td>(khoa, 0.584521384929)</td>
      <td>(tế, 0.566191446029)</td>
      <td>(rằng, 0.533604887984)</td>
      <td>(phát triển, 0.521384928717)</td>
      <td>(cần, 0.49083503055)</td>
      <td>...</td>
      <td>(hồ, 0.0386965376782)</td>
      <td>(lộ, 0.030549898167)</td>
      <td>(trạm, 0.0285132382892)</td>
      <td>(lũ, 0.0264765784114)</td>
      <td>(ngập, 0.0244399185336)</td>
      <td>(bão, 0.020366598778)</td>
      <td>(vỉa hè, 0.0142566191446)</td>
      <td>(cháy, 0.0122199592668)</td>
      <td>(thanh tra, 0.0081466395112)</td>
      <td>(tài xế, 0.0061099796334)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(con, 0.197175732218)</td>
      <td>(tìm, 0.193514644351)</td>
      <td>(anh, 0.192991631799)</td>
      <td>(Việt Nam, 0.173640167364)</td>
      <td>(cứu, 0.167887029289)</td>
      <td>(khách, 0.167364016736)</td>
      <td>(biển, 0.1589958159)</td>
      <td>(quận, 0.155334728033)</td>
      <td>(giờ, 0.148535564854)</td>
      <td>(gia đình, 0.144874476987)</td>
      <td>...</td>
      <td>(thư, 0.0214435146444)</td>
      <td>(vỉa hè, 0.0193514644351)</td>
      <td>(trạm, 0.0188284518828)</td>
      <td>(ủy, 0.0188284518828)</td>
      <td>(ôtô, 0.0167364016736)</td>
      <td>(tài xế, 0.015690376569)</td>
      <td>(thi, 0.015690376569)</td>
      <td>(ương, 0.0130753138075)</td>
      <td>(Đảng, 0.010460251046)</td>
      <td>(thanh tra, 0.00575313807531)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Chính, 0.869198312236)</td>
      <td>(phủ, 0.852320675105)</td>
      <td>(hội, 0.729957805907)</td>
      <td>(rằng, 0.677215189873)</td>
      <td>(tế, 0.675105485232)</td>
      <td>(đầu tư, 0.654008438819)</td>
      <td>(cần, 0.651898734177)</td>
      <td>(Quốc, 0.594936708861)</td>
      <td>(phát triển, 0.592827004219)</td>
      <td>(trưởng, 0.569620253165)</td>
      <td>...</td>
      <td>(C, 0.0400843881857)</td>
      <td>(bão, 0.0379746835443)</td>
      <td>(mưa, 0.0295358649789)</td>
      <td>(học sinh, 0.0295358649789)</td>
      <td>(vỉa hè, 0.0274261603376)</td>
      <td>(hồ, 0.0253164556962)</td>
      <td>(lũ, 0.0210970464135)</td>
      <td>(cháy, 0.0210970464135)</td>
      <td>(lớp, 0.0168776371308)</td>
      <td>(chị, 0.0105485232068)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(đầu tư, 0.833850931677)</td>
      <td>(dự án, 0.669254658385)</td>
      <td>(tuyến, 0.652173913043)</td>
      <td>(thông, 0.649068322981)</td>
      <td>(tỷ, 0.614906832298)</td>
      <td>(giao thông, 0.560559006211)</td>
      <td>(phí, 0.475155279503)</td>
      <td>(vốn, 0.42701863354)</td>
      <td>(mức, 0.416149068323)</td>
      <td>(sử dụng, 0.402173913043)</td>
      <td>...</td>
      <td>(cán bộ, 0.026397515528)</td>
      <td>(bão, 0.0248447204969)</td>
      <td>(em, 0.0248447204969)</td>
      <td>(nghề, 0.0201863354037)</td>
      <td>(cá, 0.0201863354037)</td>
      <td>(học sinh, 0.0201863354037)</td>
      <td>(lũ, 0.0170807453416)</td>
      <td>(trẻ, 0.0170807453416)</td>
      <td>(chị, 0.0139751552795)</td>
      <td>(Đảng, 0.00465838509317)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Trung, 0.826855123675)</td>
      <td>(ương, 0.777385159011)</td>
      <td>(ban, 0.759717314488)</td>
      <td>(Đảng, 0.731448763251)</td>
      <td>(cán bộ, 0.727915194346)</td>
      <td>(ủy, 0.717314487633)</td>
      <td>(thư, 0.713780918728)</td>
      <td>(Ban, 0.706713780919)</td>
      <td>(lãnh đạo, 0.628975265018)</td>
      <td>(tra, 0.618374558304)</td>
      <td>...</td>
      <td>(mưa, 0.0106007067138)</td>
      <td>(bão, 0.0106007067138)</td>
      <td>(học sinh, 0.0106007067138)</td>
      <td>(trồng, 0.00706713780919)</td>
      <td>(lũ, 0.00706713780919)</td>
      <td>(ngập, 0.00706713780919)</td>
      <td>(vỉa hè, 0.00706713780919)</td>
      <td>(tàu, 0.00353356890459)</td>
      <td>(hồ, 0.00353356890459)</td>
      <td>(tài xế, 0.00353356890459)</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 127 columns</p>
</div>



Các bài báo tương ứng của các nhóm (lấy khoảng 10 bài đầu tiên)


```python
clusters_as_array = np.array([cluster[:10] for cluster in clusters])
pd.DataFrame(clusters_as_array)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bắc Bộ giảm nhiệt nhẹ, miền Trung khả năng có mưa</td>
      <td>Ngày cuối năm miền Bắc trời rét về đêm, Nam Bộ...</td>
      <td>Chọn nước mắm thơm ngon cho bữa cơm ngày Tết</td>
      <td>Không khí Hà Nội ô nhiễm nhất thời điểm nào?</td>
      <td>Trung Bộ có mưa trên diện rộng</td>
      <td>Thời tiết cả nước thuận lợi du xuân dịp Tết</td>
      <td>Chủ tịch Hà Nội: ‘Chăm sóc cây xanh như đối vớ...</td>
      <td>Miền Bắc rét nhất 4 độ C</td>
      <td>Hà Nội sẽ rét 13 độ C</td>
      <td>Đêm nay, miền Bắc đón không khí lạnh mạnh</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39 điểm trông giữ xe khu vực Hồ Gươm đêm giao ...</td>
      <td>Tân Sơn Nhất 'trong tầm kiểm soát' dù có 120.0...</td>
      <td>Tổ công tác đặc biệt túc trực 'giải cứu' kẹt x...</td>
      <td>Lãnh đạo cấp vụ cầm đường dây nóng giao thông ...</td>
      <td>Ông Đinh La Thăng: 'Quy hoạch huyện lên quận đ...</td>
      <td>TP HCM muốn quy hoạch 3 huyện thành quận</td>
      <td>Ôtô đậu trái phép ở Sài Gòn, chuyên gia nói 'c...</td>
      <td>Vỉa hè ở Hải Phòng bị tái chiếm khi nhà chức t...</td>
      <td>Bộ trưởng Công an: 'Quyết không để tái diễn tì...</td>
      <td>Bí thư Hà Nội: 'Lập lại trật tự vỉa hè phải gắ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nữ sinh bị bạn dùng gạch hành hung đến ngất xỉu</td>
      <td>Cựu chiến binh hiến 2.000 m2 đất mở đường</td>
      <td>Người Cơ Tu đốt trứng gà chọn đất tốt</td>
      <td>Người phụ nữ mở xưởng đóng tàu vươn Hoàng Sa</td>
      <td>Trò chơi đánh quay miền sơn cước</td>
      <td>Người đầu tiên mang rau, hoa Đà Lạt về Sài Gòn</td>
      <td>Kỷ lục gia sang Nhật biểu diễn cắt tóc bằng kiếm</td>
      <td>Thầy giáo 9x đào tạo nhiều học sinh giỏi quốc gia</td>
      <td>'Cột mốc' Trường Sa, đảo Gạc Ma trên đỉnh Trườ...</td>
      <td>Chào xuân Đinh Dậu</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Đường hoa Nguyễn Huệ kéo dài thêm một ngày</td>
      <td>Xác pháo vương vãi đường làng ở Hải Phòng</td>
      <td>Nhà thờ đá hơn 120 tuổi độc nhất Việt Nam</td>
      <td>Làng chài Khe Gà với hải đăng hơn trăm tuổi ở ...</td>
      <td>Xác pháo đầy quốc lộ ở Hà Tĩnh</td>
      <td>Hà Nội lập đoàn kiểm tra việc chấp hành kỷ cươ...</td>
      <td>Cựu binh 20 năm giữ rừng dưới chân núi Hoành Sơn</td>
      <td>Chủ tịch Quảng Ngãi viết thư kêu gọi mua hoa g...</td>
      <td>Nhiều nhà xe tăng cước 60% dịp Tết</td>
      <td>Tiếp viên trả lại gần 100 triệu đồng cho khách...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Người dân dựng biển cảnh báo 'Công an bắn tốc độ'</td>
      <td>5 ngày nghỉ Tết, gần 120 người chết vì tai nạn...</td>
      <td>Xe chở 29 người đi lễ chùa đâm vách núi, 2 ngư...</td>
      <td>25 người chết vì tai nạn giao thông trong mùng...</td>
      <td>Xe nhích từng mét qua cầu Rạch Miễu ngày mùng ...</td>
      <td>3 người tháo chạy khi căn nhà bị ôtô khách tôn...</td>
      <td>15 người chết vì tai nạn giao thông trong ngày...</td>
      <td>Sài Gòn mưa lớn bất thường ngày Tết</td>
      <td>Rửa xe những ngày 'bão giá'</td>
      <td>Bắt khách trên cao tốc, 3 tài xế bị phạ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
      <td>Bia cổ ghi công trạng vị vua anh minh bậc nhất...</td>
      <td>Vụ án 'hóa hổ' giết vua và vị Trạng nguyên đầu...</td>
      <td>Tên đường ở Sài Gòn xưa được đặt như thế nào</td>
      <td>Bộ trưởng Khoa học: Cần thay đổi tư duy chiến ...</td>
      <td>Nam sinh người Mông ăn mì tôm dành tiền nghiên...</td>
      <td>Công trường khai thác đá cổ xây thành nhà Hồ</td>
      <td>Từ cậu bé làm ruộng đến giáo sư nổi tiếng ở Mỹ</td>
      <td>Chiếc ấn vàng truyền quốc của triều Nguyễn</td>
      <td>Chữ 'Thần' bí ẩn trên vách núi ở cửa biển Thần...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cơ quan khí tượng triều Nguyễn</td>
      <td>Đường vành đai Hà Nội ùn tắc ngày mùng 4 Tết</td>
      <td>Lễ hội trâu rơm, bò rạ ở đồng bằng sông Hồng</td>
      <td>9 câu hỏi về ông vua dời đô từ Hoa Lư về Thăng...</td>
      <td>Vị thần 'trấn Bắc' của kinh thành Thăng Long xưa</td>
      <td>Đường hoa lớn nhất miền Bắc khoe sắc</td>
      <td>Cháy rừng dữ dội ở Hải Phòng</td>
      <td>Hà Nội khai trương phố sách Xuân 2017</td>
      <td>Ngư dân Quảng Trị thu tiền triệu từ 'lộc biển'...</td>
      <td>Học sinh Đồng Nai tri ân 'người đưa đò' ngày T...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bộ Giao thông xây dựng cơ chế làm 1.376 km cao...</td>
      <td>Phần mềm giám định chống trục lợi thẻ bảo hiểm...</td>
      <td>Thủ tướng: 'Hạn chế xây cao ốc ở trung tâm TP ...</td>
      <td>Trụ sở nhiều bộ ngành sẽ được xây dựng trên kh...</td>
      <td>Thủ tướng: Tổng cục Tình báo có vị trí đặc biệ...</td>
      <td>Những chính sách, chỉ đạo nổi bật của Chính ph...</td>
      <td>Phó thủ tướng chỉ đạo đẩy nhanh dịch vụ công t...</td>
      <td>3 huyện ngoại thành TP HCM gần đủ tiêu chí lên...</td>
      <td>80 triệu USD đầu tư cho dự án đổi mới giáo dục...</td>
      <td>Hiệu trưởng cao đẳng: 'Trường nghề tranh nhau ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5 công trình giao thông kỳ vọng năm 2017 ở TP HCM</td>
      <td>3 công trình giao thông hàng chục nghìn tỷ hoà...</td>
      <td>Xây dựng trung tâm cơ sở dữ liệu về thu phí kh...</td>
      <td>Thủ tướng yêu cầu xử lý nghiêm việc lấn làn xe...</td>
      <td>Khởi động dự án cao tốc Tuyên Quang - Phú Thọ</td>
      <td>Hà Nội khẳng định 'không có rối loạn' sau điều...</td>
      <td>Thông xe tuyến Thái Nguyên - Bắc Kạn ngày giáp...</td>
      <td>Phó thủ tướng chốt phương án đầu tư gần 20.000...</td>
      <td>Đà Nẵng chi hơn 70 triệu USD phát triển xe buý...</td>
      <td>Xe máy được qua hầm Thủ Thiêm thêm 2 giờ</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
      <td>Một phó tổng cục trưởng bị cách chức vì liên q...</td>
      <td>Thứ trưởng Hồ Thị Kim Thoa bị khiển trách</td>
      <td>Người tuyển dụng 'vụ phó 26 tuổi' bị kiểm tra ...</td>
      <td>Thứ trưởng Hồ Thị Kim Thoa bị xem xét kỷ luật ...</td>
      <td>Ông Vũ Huy Hoàng bị xóa tư cách nguyên Bộ trưở...</td>
      <td>Trưng bày hơn 200 kỷ vật về Tổng bí thư Trường...</td>
      <td>Ông Võ Văn Thưởng: Báo chí sẽ tụt hậu nếu khôn...</td>
      <td>Hai Thứ trưởng Nội vụ bị kỷ luật</td>
      <td>Nữ Bí thư Thành đoàn bị kỷ luật vì nhờ người đ...</td>
    </tr>
  </tbody>
</table>
</div>



Căn cứ vào kết quả, ta cảm thấy, nếu chia thành 10 nhóm: các bài báo liên quan đến chủ đề sau.

(0) Thời tiết, thiên tai

(1) Vi phạm, xử lí ở thành phố, quận (cần xem xét các bài báo để hiểu thêm đề tài)

(2) Giáo dục, trẻ em

(3) Chính quyền địa phương (?)

(4) Giao thông

(5) Khoa học, nghiên cứu

(6) Nhóm hỗn tạp (toạ độ các cột rất nhỏ)

(7) Chính phủ, Quốc hội (chính quyền Trung ương)

(8) Đầu tư, dự án

(9) Trung ương, Đảng

Cuối cùng, ta tìm size của các nhóm: ta thấy nhóm 6 (hỗn tạp) có size lớn nhất vì nó có thể bao gồm cả các bài báo không phân nhóm được.


```python
[len(cluster) for cluster in clusters]
```




    [431, 223, 491, 996, 803, 491, 1912, 474, 644, 283]



Với k = 20. Ta có kết quả sau:


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyByNewsArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalNewsFrequency.txt"
COORDINATES_CODING_MODE = "0-1"
NB_CLUSTER = 20
MODEL = "KMeans"
MIN_AVG = 300
MAX_AVG = 1500
MIN_DEV = 0.8

features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
titles = getTitles(FREQUENCY_FILE)

predictive_model = train(X, NB_CLUSTER, model=MODEL)
prediction = predict(predictive_model, X)
clusters = getClusters(titles, prediction)

centers = getClusterCenters(predictive_model, X, prediction)
explicativeFeatures = getExplicatveFeaturesForEachCluster(predictive_model, X, prediction, features)

pd.DataFrame(explicativeFeatures)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(mưa, 0.859459459459)</td>
      <td>(bão, 0.789189189189)</td>
      <td>(lũ, 0.735135135135)</td>
      <td>(mạnh, 0.713513513514)</td>
      <td>(biển, 0.686486486486)</td>
      <td>(khí, 0.681081081081)</td>
      <td>(sông, 0.589189189189)</td>
      <td>(địa phương, 0.589189189189)</td>
      <td>(Trung, 0.551351351351)</td>
      <td>(Ban, 0.491891891892)</td>
      <td>...</td>
      <td>(nghiên, 0.0162162162162)</td>
      <td>(vốn, 0.0162162162162)</td>
      <td>(tiền, 0.0162162162162)</td>
      <td>(vi, 0.0162162162162)</td>
      <td>(thi, 0.0162162162162)</td>
      <td>(xã hội, 0.00540540540541)</td>
      <td>(thanh tra, 0.00540540540541)</td>
      <td>(phạm, 0.0)</td>
      <td>(Đảng, 0.0)</td>
      <td>(vỉa hè, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(tìm, 0.179324894515)</td>
      <td>(anh, 0.17194092827)</td>
      <td>(con, 0.154008438819)</td>
      <td>(cháy, 0.152953586498)</td>
      <td>(cứu, 0.137130801688)</td>
      <td>(giờ, 0.132911392405)</td>
      <td>(đất, 0.130801687764)</td>
      <td>(máy, 0.119198312236)</td>
      <td>(sông, 0.113924050633)</td>
      <td>(phường, 0.10970464135)</td>
      <td>...</td>
      <td>(hoạch, 0.0116033755274)</td>
      <td>(Thành, 0.0116033755274)</td>
      <td>(vỉa hè, 0.0116033755274)</td>
      <td>(tướng, 0.0105485232068)</td>
      <td>(ương, 0.0084388185654)</td>
      <td>(vi, 0.0084388185654)</td>
      <td>(Đảng, 0.00632911392405)</td>
      <td>(quận, 0.00632911392405)</td>
      <td>(thanh tra, 0.00527426160338)</td>
      <td>(phạm, 0.0042194092827)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Việt Nam, 0.502772643253)</td>
      <td>(khách, 0.439926062847)</td>
      <td>(sự, 0.343807763401)</td>
      <td>(hoạt động, 0.343807763401)</td>
      <td>(tế, 0.314232902033)</td>
      <td>(Tổng, 0.271719038817)</td>
      <td>(thông, 0.266173752311)</td>
      <td>(hóa, 0.264325323475)</td>
      <td>(đoàn, 0.26247689464)</td>
      <td>(hội, 0.260628465804)</td>
      <td>...</td>
      <td>(trạm, 0.0184842883549)</td>
      <td>(chị, 0.0184842883549)</td>
      <td>(trồng, 0.0184842883549)</td>
      <td>(tài xế, 0.0184842883549)</td>
      <td>(học sinh, 0.0166358595194)</td>
      <td>(lũ, 0.0147874306839)</td>
      <td>(vi, 0.0129390018484)</td>
      <td>(vỉa hè, 0.0129390018484)</td>
      <td>(phạm, 0.00739371534196)</td>
      <td>(ngập, 0.00739371534196)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(UBND, 0.837209302326)</td>
      <td>(tịch, 0.581395348837)</td>
      <td>(quyền, 0.505285412262)</td>
      <td>(chính, 0.435517970402)</td>
      <td>(tra, 0.380549682875)</td>
      <td>(lãnh đạo, 0.359408033827)</td>
      <td>(địa phương, 0.357293868922)</td>
      <td>(cán bộ, 0.317124735729)</td>
      <td>(xử lý, 0.274841437632)</td>
      <td>(thông tin, 0.272727272727)</td>
      <td>...</td>
      <td>(Việt Nam, 0.0253699788584)</td>
      <td>(sân bay, 0.0211416490486)</td>
      <td>(tàu, 0.0211416490486)</td>
      <td>(C, 0.0211416490486)</td>
      <td>(khí, 0.0211416490486)</td>
      <td>(lớp, 0.0211416490486)</td>
      <td>(bão, 0.0190274841438)</td>
      <td>(vỉa hè, 0.0190274841438)</td>
      <td>(nghề, 0.0169133192389)</td>
      <td>(tài xế, 0.00634249471459)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(đầu tư, 0.816326530612)</td>
      <td>(dự án, 0.707482993197)</td>
      <td>(tỷ, 0.641723356009)</td>
      <td>(quận, 0.575963718821)</td>
      <td>(tuyến, 0.541950113379)</td>
      <td>(thông, 0.526077097506)</td>
      <td>(giao thông, 0.498866213152)</td>
      <td>(hoạch, 0.46485260771)</td>
      <td>(công trình, 0.442176870748)</td>
      <td>(đô thị, 0.439909297052)</td>
      <td>...</td>
      <td>(lũ, 0.0226757369615)</td>
      <td>(cá, 0.0226757369615)</td>
      <td>(trẻ, 0.0226757369615)</td>
      <td>(anh, 0.0226757369615)</td>
      <td>(học sinh, 0.0204081632653)</td>
      <td>(bão, 0.0181405895692)</td>
      <td>(nghề, 0.015873015873)</td>
      <td>(Đảng, 0.0113378684807)</td>
      <td>(chị, 0.00907029478458)</td>
      <td>(tài xế, 0.00453514739229)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(khí, 0.977777777778)</td>
      <td>(mưa, 0.844444444444)</td>
      <td>(Trung, 0.844444444444)</td>
      <td>(độ, 0.742222222222)</td>
      <td>(mạnh, 0.702222222222)</td>
      <td>(C, 0.626666666667)</td>
      <td>(ương, 0.586666666667)</td>
      <td>(biển, 0.528888888889)</td>
      <td>(tăng, 0.408888888889)</td>
      <td>(mức, 0.382222222222)</td>
      <td>...</td>
      <td>(vốn, 0.0)</td>
      <td>(thanh tra, 0.0)</td>
      <td>(tiền, 0.0)</td>
      <td>(Đảng, 0.0)</td>
      <td>(tài xế, 0.0)</td>
      <td>(sử dụng, 0.0)</td>
      <td>(xử lý, 0.0)</td>
      <td>(phí, 0.0)</td>
      <td>(học sinh, 0.0)</td>
      <td>(vỉa hè, 0.0)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(phí, 0.988165680473)</td>
      <td>(thu, 0.976331360947)</td>
      <td>(trạm, 0.940828402367)</td>
      <td>(đầu tư, 0.881656804734)</td>
      <td>(lộ, 0.834319526627)</td>
      <td>(thông, 0.816568047337)</td>
      <td>(tuyến, 0.786982248521)</td>
      <td>(dự án, 0.721893491124)</td>
      <td>(giao thông, 0.692307692308)</td>
      <td>(mức, 0.591715976331)</td>
      <td>...</td>
      <td>(lớp, 0.00591715976331)</td>
      <td>(điện, 0.00591715976331)</td>
      <td>(lao động, 0.0)</td>
      <td>(trồng, 0.0)</td>
      <td>(lũ, 0.0)</td>
      <td>(nghề, 0.0)</td>
      <td>(tàu, 0.0)</td>
      <td>(hồ, 0.0)</td>
      <td>(Đảng, 0.0)</td>
      <td>(ngập, 0.0)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(phạm, 0.973597359736)</td>
      <td>(vi, 0.973597359736)</td>
      <td>(xử lý, 0.650165016502)</td>
      <td>(tra, 0.561056105611)</td>
      <td>(đơn vị, 0.366336633663)</td>
      <td>(giao thông, 0.359735973597)</td>
      <td>(triệu, 0.316831683168)</td>
      <td>(hoạt động, 0.313531353135)</td>
      <td>(doanh, 0.293729372937)</td>
      <td>(UBND, 0.270627062706)</td>
      <td>...</td>
      <td>(C, 0.023102310231)</td>
      <td>(học sinh, 0.023102310231)</td>
      <td>(thi, 0.023102310231)</td>
      <td>(mạnh, 0.019801980198)</td>
      <td>(trẻ, 0.019801980198)</td>
      <td>(em, 0.013201320132)</td>
      <td>(mưa, 0.00990099009901)</td>
      <td>(lũ, 0.00660066006601)</td>
      <td>(ngập, 0.00660066006601)</td>
      <td>(bão, 0.0)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(cây, 0.753768844221)</td>
      <td>(mỗi, 0.738693467337)</td>
      <td>(trồng, 0.713567839196)</td>
      <td>(phát triển, 0.582914572864)</td>
      <td>(đất, 0.56783919598)</td>
      <td>(địa phương, 0.537688442211)</td>
      <td>(triệu, 0.517587939698)</td>
      <td>(hộ, 0.507537688442)</td>
      <td>(thêm, 0.43216080402)</td>
      <td>(gia đình, 0.422110552764)</td>
      <td>...</td>
      <td>(cháy, 0.0251256281407)</td>
      <td>(Bộ, 0.0201005025126)</td>
      <td>(Thành, 0.0201005025126)</td>
      <td>(thi, 0.0201005025126)</td>
      <td>(thanh tra, 0.0150753768844)</td>
      <td>(Đảng, 0.0150753768844)</td>
      <td>(phạm, 0.0100502512563)</td>
      <td>(sân bay, 0.00502512562814)</td>
      <td>(tài xế, 0.00502512562814)</td>
      <td>(vỉa hè, 0.00502512562814)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(trường, 0.945454545455)</td>
      <td>(học, 0.927272727273)</td>
      <td>(học sinh, 0.772727272727)</td>
      <td>(em, 0.745454545455)</td>
      <td>(lớp, 0.724242424242)</td>
      <td>(con, 0.466666666667)</td>
      <td>(gia đình, 0.415151515152)</td>
      <td>(phòng, 0.39696969697)</td>
      <td>(điểm, 0.387878787879)</td>
      <td>(mỗi, 0.369696969697)</td>
      <td>...</td>
      <td>(tàu, 0.0181818181818)</td>
      <td>(du lịch, 0.0181818181818)</td>
      <td>(phủ, 0.0151515151515)</td>
      <td>(trạm, 0.0121212121212)</td>
      <td>(Thủ, 0.0121212121212)</td>
      <td>(sân bay, 0.0121212121212)</td>
      <td>(đô thị, 0.0121212121212)</td>
      <td>(tướng, 0.00606060606061)</td>
      <td>(tài xế, 0.00606060606061)</td>
      <td>(vỉa hè, 0.0030303030303)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(quận, 0.973170731707)</td>
      <td>(TP HCM, 0.565853658537)</td>
      <td>(phường, 0.331707317073)</td>
      <td>(máy, 0.273170731707)</td>
      <td>(tuyến, 0.258536585366)</td>
      <td>(giao thông, 0.253658536585)</td>
      <td>(giờ, 0.253658536585)</td>
      <td>(ôtô, 0.243902439024)</td>
      <td>(cháy, 0.19512195122)</td>
      <td>(tôi, 0.168292682927)</td>
      <td>...</td>
      <td>(nghề, 0.00731707317073)</td>
      <td>(hội, 0.00731707317073)</td>
      <td>(thanh tra, 0.00731707317073)</td>
      <td>(ương, 0.00487804878049)</td>
      <td>(tướng, 0.00487804878049)</td>
      <td>(ủy, 0.00243902439024)</td>
      <td>(Bộ, 0.00243902439024)</td>
      <td>(thư, 0.00243902439024)</td>
      <td>(lũ, 0.00243902439024)</td>
      <td>(Đảng, 0.0)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(quận, 0.963768115942)</td>
      <td>(vỉa hè, 0.942028985507)</td>
      <td>(xử lý, 0.876811594203)</td>
      <td>(đô thị, 0.855072463768)</td>
      <td>(phạm, 0.789855072464)</td>
      <td>(vi, 0.789855072464)</td>
      <td>(UBND, 0.775362318841)</td>
      <td>(phường, 0.753623188406)</td>
      <td>(lãnh đạo, 0.659420289855)</td>
      <td>(TP HCM, 0.652173913043)</td>
      <td>...</td>
      <td>(lớp, 0.00724637681159)</td>
      <td>(ngập, 0.00724637681159)</td>
      <td>(sông, 0.00724637681159)</td>
      <td>(bão, 0.0)</td>
      <td>(lũ, 0.0)</td>
      <td>(tàu, 0.0)</td>
      <td>(hồ, 0.0)</td>
      <td>(học sinh, 0.0)</td>
      <td>(khoa, 0.0)</td>
      <td>(thi, 0.0)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(Trung, 0.905660377358)</td>
      <td>(ương, 0.853773584906)</td>
      <td>(Ban, 0.820754716981)</td>
      <td>(thư, 0.806603773585)</td>
      <td>(ban, 0.806603773585)</td>
      <td>(ủy, 0.792452830189)</td>
      <td>(cán bộ, 0.77358490566)</td>
      <td>(Đảng, 0.77358490566)</td>
      <td>(viên, 0.688679245283)</td>
      <td>(tra, 0.683962264151)</td>
      <td>...</td>
      <td>(trạm, 0.00943396226415)</td>
      <td>(mưa, 0.00943396226415)</td>
      <td>(bão, 0.00943396226415)</td>
      <td>(vỉa hè, 0.00943396226415)</td>
      <td>(trồng, 0.00471698113208)</td>
      <td>(lũ, 0.00471698113208)</td>
      <td>(tài xế, 0.00471698113208)</td>
      <td>(ngập, 0.00471698113208)</td>
      <td>(tàu, 0.0)</td>
      <td>(hồ, 0.0)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(đầu tư, 0.889344262295)</td>
      <td>(Chính, 0.852459016393)</td>
      <td>(phủ, 0.840163934426)</td>
      <td>(dự án, 0.823770491803)</td>
      <td>(tế, 0.774590163934)</td>
      <td>(hoạch, 0.75)</td>
      <td>(phát triển, 0.717213114754)</td>
      <td>(cần, 0.709016393443)</td>
      <td>(tỷ, 0.672131147541)</td>
      <td>(hội, 0.672131147541)</td>
      <td>...</td>
      <td>(mưa, 0.0450819672131)</td>
      <td>(C, 0.0450819672131)</td>
      <td>(tài xế, 0.0368852459016)</td>
      <td>(vỉa hè, 0.0327868852459)</td>
      <td>(học sinh, 0.0286885245902)</td>
      <td>(lũ, 0.0204918032787)</td>
      <td>(cá, 0.0204918032787)</td>
      <td>(cháy, 0.0204918032787)</td>
      <td>(lớp, 0.0122950819672)</td>
      <td>(chị, 0.0)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(hội, 0.817164179104)</td>
      <td>(rằng, 0.764925373134)</td>
      <td>(cần, 0.679104477612)</td>
      <td>(Quốc, 0.638059701493)</td>
      <td>(xã hội, 0.585820895522)</td>
      <td>(trưởng, 0.582089552239)</td>
      <td>(quyền, 0.563432835821)</td>
      <td>(ngành, 0.555970149254)</td>
      <td>(Chính, 0.55223880597)</td>
      <td>(Việt Nam, 0.548507462687)</td>
      <td>...</td>
      <td>(lộ, 0.0223880597015)</td>
      <td>(sông, 0.0223880597015)</td>
      <td>(bão, 0.0186567164179)</td>
      <td>(lũ, 0.0186567164179)</td>
      <td>(tài xế, 0.0149253731343)</td>
      <td>(mưa, 0.00746268656716)</td>
      <td>(cháy, 0.00746268656716)</td>
      <td>(ngập, 0.00746268656716)</td>
      <td>(vỉa hè, 0.00746268656716)</td>
      <td>(hồ, 0.00373134328358)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(ôtô, 0.735779816514)</td>
      <td>(tài xế, 0.565137614679)</td>
      <td>(lộ, 0.499082568807)</td>
      <td>(giao thông, 0.477064220183)</td>
      <td>(biển, 0.411009174312)</td>
      <td>(máy, 0.379816513761)</td>
      <td>(khách, 0.37247706422)</td>
      <td>(mạnh, 0.25504587156)</td>
      <td>(anh, 0.183486238532)</td>
      <td>(Quốc, 0.159633027523)</td>
      <td>...</td>
      <td>(ương, 0.00366972477064)</td>
      <td>(thanh tra, 0.00366972477064)</td>
      <td>(nguyên, 0.00366972477064)</td>
      <td>(thi, 0.00366972477064)</td>
      <td>(lao động, 0.00183486238532)</td>
      <td>(thư, 0.00183486238532)</td>
      <td>(trồng, 0.00183486238532)</td>
      <td>(Đảng, 0.00183486238532)</td>
      <td>(phương án, 0.0)</td>
      <td>(phát triển, 0.0)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(Chính, 0.928853754941)</td>
      <td>(phủ, 0.920948616601)</td>
      <td>(tướng, 0.865612648221)</td>
      <td>(Thủ, 0.849802371542)</td>
      <td>(phòng, 0.529644268775)</td>
      <td>(Bộ, 0.494071146245)</td>
      <td>(trưởng, 0.474308300395)</td>
      <td>(địa phương, 0.466403162055)</td>
      <td>(ngành, 0.458498023715)</td>
      <td>(xử lý, 0.422924901186)</td>
      <td>...</td>
      <td>(mưa, 0.0197628458498)</td>
      <td>(vỉa hè, 0.0197628458498)</td>
      <td>(chị, 0.0118577075099)</td>
      <td>(hồ, 0.0118577075099)</td>
      <td>(lớp, 0.0118577075099)</td>
      <td>(trồng, 0.00790513833992)</td>
      <td>(tài xế, 0.00790513833992)</td>
      <td>(lộ, 0.00395256916996)</td>
      <td>(ngập, 0.00395256916996)</td>
      <td>(học sinh, 0.00395256916996)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(con, 0.698717948718)</td>
      <td>(tôi, 0.698717948718)</td>
      <td>(gia đình, 0.653846153846)</td>
      <td>(anh, 0.608974358974)</td>
      <td>(bà, 0.400641025641)</td>
      <td>(tìm, 0.371794871795)</td>
      <td>(em, 0.365384615385)</td>
      <td>(giờ, 0.352564102564)</td>
      <td>(mỗi, 0.352564102564)</td>
      <td>(chị, 0.346153846154)</td>
      <td>...</td>
      <td>(học sinh, 0.025641025641)</td>
      <td>(đô thị, 0.0224358974359)</td>
      <td>(tài xế, 0.0224358974359)</td>
      <td>(sân bay, 0.0192307692308)</td>
      <td>(phạm, 0.0192307692308)</td>
      <td>(Thủ, 0.0160256410256)</td>
      <td>(Đảng, 0.0160256410256)</td>
      <td>(vi, 0.00961538461538)</td>
      <td>(ương, 0.00641025641026)</td>
      <td>(thanh tra, 0.0)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(biển, 0.904564315353)</td>
      <td>(tàu, 0.780082987552)</td>
      <td>(cá, 0.564315352697)</td>
      <td>(cứu, 0.502074688797)</td>
      <td>(tìm, 0.44398340249)</td>
      <td>(Việt Nam, 0.311203319502)</td>
      <td>(Trung, 0.290456431535)</td>
      <td>(anh, 0.236514522822)</td>
      <td>(con, 0.232365145228)</td>
      <td>(tịch, 0.207468879668)</td>
      <td>...</td>
      <td>(thi, 0.00829875518672)</td>
      <td>(trồng, 0.00414937759336)</td>
      <td>(ôtô, 0.00414937759336)</td>
      <td>(thanh tra, 0.00414937759336)</td>
      <td>(đô thị, 0.00414937759336)</td>
      <td>(tài xế, 0.00414937759336)</td>
      <td>(thư, 0.0)</td>
      <td>(lộ, 0.0)</td>
      <td>(Đảng, 0.0)</td>
      <td>(vỉa hè, 0.0)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(cứu, 0.91961414791)</td>
      <td>(nghiên, 0.906752411576)</td>
      <td>(học, 0.855305466238)</td>
      <td>(Việt Nam, 0.733118971061)</td>
      <td>(khoa, 0.691318327974)</td>
      <td>(sự, 0.627009646302)</td>
      <td>(tế, 0.614147909968)</td>
      <td>(phát triển, 0.569131832797)</td>
      <td>(rằng, 0.536977491961)</td>
      <td>(hóa, 0.514469453376)</td>
      <td>...</td>
      <td>(chị, 0.032154340836)</td>
      <td>(trạm, 0.0289389067524)</td>
      <td>(lộ, 0.0289389067524)</td>
      <td>(ngập, 0.0257234726688)</td>
      <td>(lũ, 0.0128617363344)</td>
      <td>(cháy, 0.0128617363344)</td>
      <td>(vỉa hè, 0.0128617363344)</td>
      <td>(bão, 0.0064308681672)</td>
      <td>(thanh tra, 0.0064308681672)</td>
      <td>(tài xế, 0.0)</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 127 columns</p>
</div>



Ta thấy so với trường hợp $k=10$, có các hiện tượng: 
- 1 cluster bị tách thành nhiều cluster: Ví dụ, cluster thời tiết ở trường hợp $k=10$ bị tách thành các cluster 0, 5.
- Xuất hiện các cluster mới bị tách ra từ cluster hỗn tạp hoặc 1 cluster lớn cũ: ví dụ cluster 18 liên quan đến biển và tàu; cluster 8 liên quan đến nông nghiệp nhưng không xuất hiện trong trường hợp $k=10$.

Bạn có thể chọn $k$ giữa 10 và 20 và có những nhận xét tương tự.

### Bài 16. Thay đổi cách tính khoảng cách

Một hiện tượng có thể phát sinh khi dùng khoảng cách 0-1 là việc không để ý đến tần số của các từ trong bài báo có thể khiến một số từ kém quan trọng trở nên quan trọng. Hiện tượng này chưa được tìm hiểu có xuất hiện trong dữ liệu của ta hay không, nhưng xuất hiện trong dữ liệu về thể thao, bạn có thể tìm hiểu ở TD5 dành cho thể thao nếu quan tâm.

Như vậy, biến mỗi bài báo thành một vector 0-1 có thể là một cách quá đơn giản và không hiệu quả trong một số trường hợp.

Trước đó ta cũng thấy việc số hoá mỗi bài báo thành một vector chứa tần số từ là không hợp lí.

Ta tìm cách dung hoà hai phương pháp này bằng cách xây dựng một hàm `f` biến tần số $a$ mỗi từ thành thành phần toạ độ tương ứng với từ đó, sao cho:

Ứng với 3 bài báo `A`, `B`, `C` có tần số `a`, `b`, `c`.

- Nếu `a = 0 thì f(a) = 0`

- Nếu `a = 0`, `b, c > 0` thì `|f(b) - f(c)| < |f(b) - f(a)|` (Hai bài báo cùng chứa 1 từ có thành phần khoảng cách tương ứng nhỏ hơn 1 bài báo chứa và 1 báo không chứa)

- Nếu `a > b` thì  `f(a) > f(b)` (Tần số càng lớn, toạ độ càng lớn)

- Nếu `a = 1, b` rất lớn, `c = 0` thì `|f(b) - f(a)| \approx |f(c) - f(a)|`. Nghĩa là ta xem như "có 1" là trung bình cộng giữa "không có" và "có rất nhiều".

Theo cách này, ta có thể cho chẳng hạn $f(1) = 2$, đặt một chặn trên tại 4 và xây dựng 1 hàm tăng nhận giá trị trong [2, 4]. Điều kiện thứ 3 cho thấy tần số của 1 từ càng lớn thì độ liên quan đến từ đó càng lớn

Có thể giả sử rằng 1 từ không xuất hiện quá 8 lần trong bài báo, ta có thể chọn hàm $f(a) = 1+ \sqrt{a}$. (Có nhiều cách lựa chọn khác, ví dụ $f(a) = \log_2{(3 + a)}$)

*Hãy sửa chữa hàm **`freqToCoordinates`** đã viết ở bài 10, để khi đối số **`coordinates_coding_mode`** nhận giá trị **"sqrt"** thì output của hàm này sẽ là một ma trận mà mỗi hàng là các vector có thành phần toạ độ được tính theo hàm $f(a) = 1 + \sqrt{a}$.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyByNewsArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalNewsFrequency.txt"
COORDINATES_CODING_MODE = "sqrt"
NB_CLUSTER = 10
MODEL = "KMeans"
MIN_AVG = 300
MAX_AVG = 1500
MIN_DEV = 0.8

features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
titles = getTitles(FREQUENCY_FILE)

predictive_model = train(X, NB_CLUSTER, model=MODEL)
prediction = predict(predictive_model, X)
clusters = getClusters(titles, prediction)

centers = getClusterCenters(predictive_model, X, prediction)
explicativeFeatures = getExplicatveFeaturesForEachCluster(predictive_model, X, prediction, features)
```


```python
pd.DataFrame(explicativeFeatures)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(học, 3.34943952534)</td>
      <td>(trường, 2.81038093591)</td>
      <td>(học sinh, 1.91108589443)</td>
      <td>(em, 1.68792132206)</td>
      <td>(lớp, 1.46743381884)</td>
      <td>(khoa, 1.23090397151)</td>
      <td>(thi, 1.18338806119)</td>
      <td>(con, 1.07152205969)</td>
      <td>(sự, 1.05937612139)</td>
      <td>(Đại, 1.03761930152)</td>
      <td>...</td>
      <td>(ngập, 0.0658479144876)</td>
      <td>(đô thị, 0.0650426446199)</td>
      <td>(hồ, 0.0530674655426)</td>
      <td>(cháy, 0.0458704216268)</td>
      <td>(bão, 0.0401261721057)</td>
      <td>(sân bay, 0.0335655275365)</td>
      <td>(trạm, 0.0312672745388)</td>
      <td>(tàu, 0.0234889959322)</td>
      <td>(tài xế, 0.0102647522941)</td>
      <td>(vỉa hè, 0.00957530056914)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(mưa, 2.40173551975)</td>
      <td>(khí, 2.31948422887)</td>
      <td>(mạnh, 1.9870310817)</td>
      <td>(bão, 1.96260943316)</td>
      <td>(Trung, 1.7176347139)</td>
      <td>(biển, 1.66764457551)</td>
      <td>(độ, 1.57998495154)</td>
      <td>(C, 1.31913497952)</td>
      <td>(lũ, 1.13259255436)</td>
      <td>(tăng, 0.946670841233)</td>
      <td>...</td>
      <td>(vi, 0.0195599022005)</td>
      <td>(phí, 0.0164597819256)</td>
      <td>(tiền, 0.0146699266504)</td>
      <td>(thi, 0.0138037929366)</td>
      <td>(tài xế, 0.00977995110024)</td>
      <td>(phạm, 0.00488997555012)</td>
      <td>(xã hội, 0.00488997555012)</td>
      <td>(thanh tra, 0.00488997555012)</td>
      <td>(Đảng, 0.0)</td>
      <td>(vỉa hè, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(mỗi, 1.46874745469)</td>
      <td>(con, 1.32213002108)</td>
      <td>(gia đình, 1.19823549228)</td>
      <td>(anh, 1.19323615662)</td>
      <td>(cây, 1.18473175888)</td>
      <td>(triệu, 1.18397613645)</td>
      <td>(tôi, 1.17543883742)</td>
      <td>(đất, 1.02212403219)</td>
      <td>(trồng, 0.957992165648)</td>
      <td>(thêm, 0.932250186237)</td>
      <td>...</td>
      <td>(vỉa hè, 0.0792040583156)</td>
      <td>(ủy, 0.0791526230436)</td>
      <td>(Thủ, 0.0749988699439)</td>
      <td>(vi, 0.0706235859566)</td>
      <td>(phạm, 0.0686155707237)</td>
      <td>(ương, 0.0683280875263)</td>
      <td>(Đảng, 0.0574036937594)</td>
      <td>(thanh tra, 0.038729421923)</td>
      <td>(sân bay, 0.03777725142)</td>
      <td>(tài xế, 0.0169158754167)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Việt Nam, 0.450902925358)</td>
      <td>(tàu, 0.40811640257)</td>
      <td>(tìm, 0.402411439634)</td>
      <td>(khách, 0.397719915089)</td>
      <td>(cứu, 0.386974515994)</td>
      <td>(UBND, 0.385224280475)</td>
      <td>(con, 0.38075057389)</td>
      <td>(biển, 0.374583841918)</td>
      <td>(tịch, 0.363343651003)</td>
      <td>(Trung, 0.335356217706)</td>
      <td>...</td>
      <td>(ương, 0.057247472568)</td>
      <td>(ôtô, 0.0559234694521)</td>
      <td>(Đảng, 0.0556039731472)</td>
      <td>(trồng, 0.0535632590619)</td>
      <td>(đô thị, 0.0489124215059)</td>
      <td>(vốn, 0.0484934399369)</td>
      <td>(trạm, 0.0481817077687)</td>
      <td>(thanh tra, 0.04053845239)</td>
      <td>(tài xế, 0.0345792062881)</td>
      <td>(vỉa hè, 0.0311320528592)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Trung, 2.71026641688)</td>
      <td>(ương, 2.54339736462)</td>
      <td>(ủy, 2.53530657902)</td>
      <td>(Ban, 2.38311777777)</td>
      <td>(Đảng, 2.28141054526)</td>
      <td>(ban, 2.25377040727)</td>
      <td>(thư, 2.20120268276)</td>
      <td>(cán bộ, 2.19851410123)</td>
      <td>(tra, 1.93107125552)</td>
      <td>(vi, 1.81361453917)</td>
      <td>...</td>
      <td>(cây, 0.0202486860659)</td>
      <td>(tuyến, 0.0202486860659)</td>
      <td>(bão, 0.0183486238532)</td>
      <td>(học sinh, 0.0183486238532)</td>
      <td>(tài xế, 0.012532343154)</td>
      <td>(trồng, 0.0110743741393)</td>
      <td>(tàu, 0.00917431192661)</td>
      <td>(ngập, 0.00917431192661)</td>
      <td>(vỉa hè, 0.00917431192661)</td>
      <td>(hồ, 0.0)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(phí, 4.06186659841)</td>
      <td>(thu, 3.7804769533)</td>
      <td>(trạm, 3.57647806596)</td>
      <td>(đầu tư, 2.86399687759)</td>
      <td>(lộ, 2.40729405998)</td>
      <td>(dự án, 2.4007600122)</td>
      <td>(thông, 2.25317325816)</td>
      <td>(tuyến, 2.1245325728)</td>
      <td>(giao thông, 1.75835980561)</td>
      <td>(mức, 1.66373029308)</td>
      <td>...</td>
      <td>(trẻ, 0.0113636363636)</td>
      <td>(cây, 0.0113636363636)</td>
      <td>(lớp, 0.0113636363636)</td>
      <td>(ngập, 0.0113636363636)</td>
      <td>(điện, 0.0113636363636)</td>
      <td>(trồng, 0.0)</td>
      <td>(lũ, 0.0)</td>
      <td>(nghề, 0.0)</td>
      <td>(tàu, 0.0)</td>
      <td>(hồ, 0.0)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(xử lý, 1.75078360744)</td>
      <td>(UBND, 1.73044495552)</td>
      <td>(quận, 1.69609495604)</td>
      <td>(tra, 1.42669859212)</td>
      <td>(vi, 1.35674227499)</td>
      <td>(phạm, 1.31151165748)</td>
      <td>(phường, 1.16298996254)</td>
      <td>(lãnh đạo, 1.14134548383)</td>
      <td>(quyền, 1.1347224667)</td>
      <td>(tịch, 1.07211365964)</td>
      <td>...</td>
      <td>(học sinh, 0.0669484458631)</td>
      <td>(mạnh, 0.0587270610461)</td>
      <td>(thi, 0.0555893788503)</td>
      <td>(khoa, 0.0527085896514)</td>
      <td>(trẻ, 0.0497460267822)</td>
      <td>(ngập, 0.0448504962882)</td>
      <td>(C, 0.0414065552235)</td>
      <td>(lớp, 0.0387222742595)</td>
      <td>(bão, 0.0382166030424)</td>
      <td>(lũ, 0.0107334525939)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Chính, 1.93320639435)</td>
      <td>(hội, 1.88902475171)</td>
      <td>(phủ, 1.86663725385)</td>
      <td>(tế, 1.75801682738)</td>
      <td>(phát triển, 1.60211252264)</td>
      <td>(trưởng, 1.58717972755)</td>
      <td>(rằng, 1.57655284365)</td>
      <td>(Việt Nam, 1.54018943193)</td>
      <td>(cần, 1.52769281638)</td>
      <td>(Bộ, 1.50804575768)</td>
      <td>...</td>
      <td>(lũ, 0.0671258062378)</td>
      <td>(ngập, 0.0659930403416)</td>
      <td>(lộ, 0.0646453422298)</td>
      <td>(lớp, 0.0607770066205)</td>
      <td>(học sinh, 0.0485156321847)</td>
      <td>(vỉa hè, 0.0463456273488)</td>
      <td>(hồ, 0.0351807157056)</td>
      <td>(cháy, 0.0338640775625)</td>
      <td>(tài xế, 0.0236443603776)</td>
      <td>(chị, 0.0216987321983)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(ôtô, 1.68183806644)</td>
      <td>(giao thông, 1.31328143899)</td>
      <td>(máy, 1.07366335735)</td>
      <td>(tài xế, 1.03356795677)</td>
      <td>(khách, 0.940639377756)</td>
      <td>(lộ, 0.829937118745)</td>
      <td>(biển, 0.68069612335)</td>
      <td>(quận, 0.589830552912)</td>
      <td>(tuyến, 0.562496294771)</td>
      <td>(TP HCM, 0.533815286411)</td>
      <td>...</td>
      <td>(cá, 0.0161855374103)</td>
      <td>(Bộ, 0.012915717694)</td>
      <td>(bão, 0.0115874855156)</td>
      <td>(ủy, 0.00974995777795)</td>
      <td>(lao động, 0.00926998841251)</td>
      <td>(thư, 0.00926998841251)</td>
      <td>(nghề, 0.00926998841251)</td>
      <td>(ương, 0.00743246067482)</td>
      <td>(Đảng, 0.00743246067482)</td>
      <td>(trồng, 0.00695249130939)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(đầu tư, 2.22498410958)</td>
      <td>(dự án, 2.07565684395)</td>
      <td>(tỷ, 1.6291885425)</td>
      <td>(tuyến, 1.54244045939)</td>
      <td>(giao thông, 1.39839572042)</td>
      <td>(thông, 1.39343790556)</td>
      <td>(hoạch, 1.31449875279)</td>
      <td>(quận, 1.13815540405)</td>
      <td>(TP HCM, 1.09106218875)</td>
      <td>(công trình, 1.05905868784)</td>
      <td>...</td>
      <td>(anh, 0.0629752338647)</td>
      <td>(lớp, 0.0592540309772)</td>
      <td>(trẻ, 0.0578132881515)</td>
      <td>(em, 0.0554405481348)</td>
      <td>(nghề, 0.0518755254343)</td>
      <td>(bão, 0.0431798732604)</td>
      <td>(học sinh, 0.0292668297822)</td>
      <td>(chị, 0.0198508061954)</td>
      <td>(Đảng, 0.0146334148911)</td>
      <td>(tài xế, 0.0128942844563)</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 127 columns</p>
</div>




```python
clusters_as_array = np.array([cluster[:10] for cluster in clusters])
pd.DataFrame(clusters_as_array)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nữ sinh bị bạn dùng gạch hành hung đến ngất xỉu</td>
      <td>Vụ án 'hóa hổ' giết vua và vị Trạng nguyên đầu...</td>
      <td>Tên đường ở Sài Gòn xưa được đặt như thế nào</td>
      <td>Nam sinh người Mông ăn mì tôm dành tiền nghiên...</td>
      <td>Từ cậu bé làm ruộng đến giáo sư nổi tiếng ở Mỹ</td>
      <td>Thầy giáo 9x đào tạo nhiều học sinh giỏi quốc gia</td>
      <td>'Cột mốc' Trường Sa, đảo Gạc Ma trên đỉnh Trườ...</td>
      <td>Nam sinh 12 tuổi ẵm nhiều giải thưởng</td>
      <td>'Cây đại thụ' của ngành sử Việt Nam Đinh Xuân ...</td>
      <td>Giáo viên Thanh Hóa đã nhận lương thưởng Tết</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bắc Bộ giảm nhiệt nhẹ, miền Trung khả năng có mưa</td>
      <td>Ngày cuối năm miền Bắc trời rét về đêm, Nam Bộ...</td>
      <td>Không khí Hà Nội ô nhiễm nhất thời điểm nào?</td>
      <td>Trung Bộ có mưa trên diện rộng</td>
      <td>Thời tiết cả nước thuận lợi du xuân dịp Tết</td>
      <td>Miền Bắc rét nhất 4 độ C</td>
      <td>Hà Nội sẽ rét 13 độ C</td>
      <td>Đêm nay, miền Bắc đón không khí lạnh mạnh</td>
      <td>Miền Bắc sắp đón đợt rét mạnh</td>
      <td>Miền Bắc tiếp tục rét, Nam Bộ có mưa to</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cựu chiến binh hiến 2.000 m2 đất mở đường</td>
      <td>Ngư dân Quảng Trị thu tiền triệu từ 'lộc biển'...</td>
      <td>Người Cơ Tu đốt trứng gà chọn đất tốt</td>
      <td>Công trường khai thác đá cổ xây thành nhà Hồ</td>
      <td>Người phụ nữ mở xưởng đóng tàu vươn Hoàng Sa</td>
      <td>Nhà thờ đá hơn 120 tuổi độc nhất Việt Nam</td>
      <td>Trò chơi đánh quay miền sơn cước</td>
      <td>Người đầu tiên mang rau, hoa Đà Lạt về Sài Gòn</td>
      <td>Làng chài Khe Gà với hải đăng hơn trăm tuổi ở ...</td>
      <td>Thú chơi giống gà nhỏ nhất thế giới ở Sài Gòn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cơ quan khí tượng triều Nguyễn</td>
      <td>Đường vành đai Hà Nội ùn tắc ngày mùng 4 Tết</td>
      <td>Lễ hội trâu rơm, bò rạ ở đồng bằng sông Hồng</td>
      <td>9 câu hỏi về ông vua dời đô từ Hoa Lư về Thăng...</td>
      <td>Vị thần 'trấn Bắc' của kinh thành Thăng Long xưa</td>
      <td>Đường hoa lớn nhất miền Bắc khoe sắc</td>
      <td>Cháy rừng dữ dội ở Hải Phòng</td>
      <td>Hà Nội khai trương phố sách Xuân 2017</td>
      <td>Học sinh Đồng Nai tri ân 'người đưa đò' ngày T...</td>
      <td>Sờ đầu rùa, rải tiền lẻ cầu may ở Văn Miếu - Q...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thứ trưởng Hồ Thị Kim Thoa bị khiển trách</td>
      <td>Người tuyển dụng 'vụ phó 26 tuổi' bị kiểm tra ...</td>
      <td>Thứ trưởng Hồ Thị Kim Thoa bị xem xét kỷ luật ...</td>
      <td>Ông Vũ Huy Hoàng bị xóa tư cách nguyên Bộ trưở...</td>
      <td>Trưng bày hơn 200 kỷ vật về Tổng bí thư Trường...</td>
      <td>Hai Thứ trưởng Nội vụ bị kỷ luật</td>
      <td>Nữ Bí thư Thành đoàn bị kỷ luật vì nhờ người đ...</td>
      <td>Ban thường vụ Đảng ủy ngoài nước bị khiển trách</td>
      <td>Vụ trưởng Bộ Tài chính làm Phó ban kinh tế Tru...</td>
      <td>Bài toán nhân sự ngày Đà Nẵng chia tay Quảng Nam</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Xây dựng trung tâm cơ sở dữ liệu về thu phí kh...</td>
      <td>Bộ Giao thông xây dựng cơ chế làm 1.376 km cao...</td>
      <td>Quảng Trị đề nghị giảm phí, dời trạm BOT</td>
      <td>Bộ Giao thông phản hồi việc giảm thời gian thu...</td>
      <td>Giám đốc Sở Giao thông Hà Nội thuyết phục các ...</td>
      <td>Khởi công mở rộng hầm Hải Vân 2</td>
      <td>Bộ trưởng Giao thông đốc thúc thu phí không dừ...</td>
      <td>Đề xuất tăng phí cao tốc TP HCM - Long Thành v...</td>
      <td>Doanh nghiệp Nhật 'choáng' vì phí cảng biển ở ...</td>
      <td>Phát hiện chênh lệch thu phí ở dự án BOT Hà Nộ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Những chính sách nổi bật có hiệu lực từ tháng 2</td>
      <td>Đường hoa Nguyễn Huệ kéo dài thêm một ngày</td>
      <td>Xác pháo đầy quốc lộ ở Hà Tĩnh</td>
      <td>39 điểm trông giữ xe khu vực Hồ Gươm đêm giao ...</td>
      <td>Tân Sơn Nhất 'trong tầm kiểm soát' dù có 120.0...</td>
      <td>Nhiều nhà xe tăng cước 60% dịp Tết</td>
      <td>Phó chủ tịch Hà Nội: Không chủ trương rung chu...</td>
      <td>Sở có biên chế 44 cán bộ bổ nhiệm 'phù hợp quy...</td>
      <td>Lãnh đạo huyện Bình Chánh: 'Quy hoạch thành qu...</td>
      <td>Lãnh đạo cấp vụ cầm đường dây nóng giao thông ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ông Đinh La Thăng: 'CSGT TP HCM phải lấy lại h...</td>
      <td>Bia cổ ghi công trạng vị vua anh minh bậc nhất...</td>
      <td>Bộ trưởng Khoa học: Cần thay đổi tư duy chiến ...</td>
      <td>Phần mềm giám định chống trục lợi thẻ bảo hiểm...</td>
      <td>Thủ tướng: 'Hạn chế xây cao ốc ở trung tâm TP ...</td>
      <td>Trụ sở nhiều bộ ngành sẽ được xây dựng trên kh...</td>
      <td>Thủ tướng: Tổng cục Tình báo có vị trí đặc biệ...</td>
      <td>Những chính sách, chỉ đạo nổi bật của Chính ph...</td>
      <td>Phó thủ tướng chỉ đạo đẩy nhanh dịch vụ công t...</td>
      <td>Ông Võ Văn Thưởng: Báo chí sẽ tụt hậu nếu khôn...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Người dân dựng biển cảnh báo 'Công an bắn tốc độ'</td>
      <td>5 ngày nghỉ Tết, gần 120 người chết vì tai nạn...</td>
      <td>Xe chở 29 người đi lễ chùa đâm vách núi, 2 ngư...</td>
      <td>25 người chết vì tai nạn giao thông trong mùng...</td>
      <td>Xe nhích từng mét qua cầu Rạch Miễu ngày mùng ...</td>
      <td>Sài Gòn khác lạ ngày đầu năm</td>
      <td>3 người tháo chạy khi căn nhà bị ôtô khách tôn...</td>
      <td>15 người chết vì tai nạn giao thông trong ngày...</td>
      <td>Sài Gòn mưa lớn bất thường ngày Tết</td>
      <td>Rửa xe những ngày 'bão giá'</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5 công trình giao thông kỳ vọng năm 2017 ở TP HCM</td>
      <td>3 công trình giao thông hàng chục nghìn tỷ hoà...</td>
      <td>Thủ tướng yêu cầu xử lý nghiêm việc lấn làn xe...</td>
      <td>Khởi động dự án cao tốc Tuyên Quang - Phú Thọ</td>
      <td>Thông xe tuyến Thái Nguyên - Bắc Kạn ngày giáp...</td>
      <td>Tổ công tác đặc biệt túc trực 'giải cứu' kẹt x...</td>
      <td>Phó thủ tướng chốt phương án đầu tư gần 20.000...</td>
      <td>Đà Nẵng chi hơn 70 triệu USD phát triển xe buý...</td>
      <td>Thứ trưởng Giao thông yêu cầu thêm xe buýt chạ...</td>
      <td>Đề xuất hơn 2.700 tỷ đồng đầu tư chống ùn tắc ...</td>
    </tr>
  </tbody>
</table>
</div>



Việc sử dụng khoảng cách khác 0-1 có lợi ích: ví dụ, thông qua clustering với 10 nhóm, nhóm nào có toạ độ thành phần quan trọng lớn (cỡ >2.5, như nhóm 0, 1, 4, 9) ta xem chúng như 1 cluster có độ thuần nhất cao. Những nhóm có toạ độ cluster trung bình (như 2, 6, 8), chúng có thể là hợp của 1 số cluster con khác. Ví dụ: cluster 2 có khả năng là hợp của 1 cluster "gia đình" và 1 cluster "nông nghiệp". Nếu lọc riêng cluster này và thực hiện lại KMeans, rất có thể ta sẽ tìm ra các cluster con.

Ta kiểm tra lại size của các cluster. Cluster hỗn tạp vẫn có size lớn.


```python
[len(cluster) for cluster in clusters]
```




    [461, 409, 546, 2385, 218, 176, 559, 556, 863, 575]



### Bài 17. KMeans với cosine metric

Trong PC, ta biết rằng thực hiện k-means với cosine metric tương đương với thực hiện k-means với khoảng cách Euclide, nhưng cần preprocess để co dãn mỗi vector thành vector đơn vị.

*Hãy sửa chữa hàm **`train, predict, getClusterCenters, getExplicatveFeaturesForEachCluster`** để khi đối số **`model`** trong hàm **`train`** nhận giá trị **"KMeans_Cosine"** thì thuật toán KMeans với metric cosine được thực hiện. So sánh kết quả với KMeans với khoảng cách Euclide*

Đoạn code dưới đây giúp test hàm của bạn.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyByNewsArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalNewsFrequency.txt"
COORDINATES_CODING_MODE = "sqrt"
NB_CLUSTER = 10
MODEL = "KMeans_Cosine"
MIN_AVG = 300
MAX_AVG = 1500
MIN_DEV = 0.8


features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
titles = getTitles(FREQUENCY_FILE)

predictive_model = train(X, NB_CLUSTER, model=MODEL)
prediction = predict(predictive_model, X)
clusters = getClusters(titles, prediction)

centers = getClusterCenters(predictive_model, X, prediction)
explicativeFeatures = getExplicatveFeaturesForEachCluster(predictive_model, X, prediction, features)
pd.DataFrame(explicativeFeatures)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(tàu, 0.425027879247)</td>
      <td>(biển, 0.207557195816)</td>
      <td>(cá, 0.131726886203)</td>
      <td>(cứu, 0.111612556534)</td>
      <td>(tìm, 0.0932773887503)</td>
      <td>(Việt Nam, 0.084130748844)</td>
      <td>(Trung, 0.0636058310306)</td>
      <td>(máy, 0.0594226030898)</td>
      <td>(khách, 0.0588677472427)</td>
      <td>(đoàn, 0.0574035673474)</td>
      <td>...</td>
      <td>(hồ, 0.00254248472715)</td>
      <td>(nghiên, 0.00237139804783)</td>
      <td>(thư, 0.00209776920078)</td>
      <td>(C, 0.00198892070077)</td>
      <td>(trạm, 0.00157155416507)</td>
      <td>(thanh tra, 0.000610433961411)</td>
      <td>(thi, 0.000586303922058)</td>
      <td>(trồng, 0.000434316533957)</td>
      <td>(Đảng, 0.0)</td>
      <td>(vỉa hè, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(UBND, 0.1496609819)</td>
      <td>(tra, 0.137928379524)</td>
      <td>(xử lý, 0.111550332397)</td>
      <td>(vi, 0.102092631474)</td>
      <td>(tịch, 0.100830870096)</td>
      <td>(phạm, 0.100106747034)</td>
      <td>(cán bộ, 0.0975802270117)</td>
      <td>(lãnh đạo, 0.0960859969327)</td>
      <td>(quận, 0.0875295211256)</td>
      <td>(quyền, 0.0856548281804)</td>
      <td>...</td>
      <td>(nghề, 0.0056917029072)</td>
      <td>(mạnh, 0.0055641196748)</td>
      <td>(cháy, 0.00535283638084)</td>
      <td>(học sinh, 0.00506767085374)</td>
      <td>(C, 0.00486487957528)</td>
      <td>(lớp, 0.00475887601816)</td>
      <td>(bão, 0.00428788028582)</td>
      <td>(ngập, 0.00286080046518)</td>
      <td>(lũ, 0.00278189517132)</td>
      <td>(tàu, 0.00229084788845)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(trường, 0.258653913991)</td>
      <td>(học, 0.242216062514)</td>
      <td>(học sinh, 0.176613015551)</td>
      <td>(em, 0.168364999073)</td>
      <td>(lớp, 0.138601425604)</td>
      <td>(thi, 0.0828093821326)</td>
      <td>(con, 0.0750761195066)</td>
      <td>(điểm, 0.0748217640712)</td>
      <td>(phòng, 0.0720620993659)</td>
      <td>(khoa, 0.0706131828204)</td>
      <td>...</td>
      <td>(Thủ, 0.00501724809317)</td>
      <td>(cá, 0.00408921111131)</td>
      <td>(cháy, 0.00337944654084)</td>
      <td>(đô thị, 0.00318159843544)</td>
      <td>(tài xế, 0.0030580175836)</td>
      <td>(tướng, 0.00301814060678)</td>
      <td>(trạm, 0.00287521849577)</td>
      <td>(tàu, 0.00198572416668)</td>
      <td>(sân bay, 0.00118371197835)</td>
      <td>(vỉa hè, 0.000368699182003)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(khí, 0.268511706128)</td>
      <td>(mưa, 0.256565487676)</td>
      <td>(độ, 0.218542178784)</td>
      <td>(mạnh, 0.197605922719)</td>
      <td>(Trung, 0.192120649208)</td>
      <td>(C, 0.172726003877)</td>
      <td>(bão, 0.158610287587)</td>
      <td>(biển, 0.158155993286)</td>
      <td>(ương, 0.109532465604)</td>
      <td>(tăng, 0.0940925664809)</td>
      <td>...</td>
      <td>(hội, 0.000848768731633)</td>
      <td>(cháy, 0.000796183487618)</td>
      <td>(xã hội, 0.000582906652692)</td>
      <td>(phạm, 0.000576266449173)</td>
      <td>(tiền, 0.000448685643883)</td>
      <td>(thanh tra, 0.000377100536253)</td>
      <td>(nghề, 0.0)</td>
      <td>(Đảng, 0.0)</td>
      <td>(phí, 0.0)</td>
      <td>(vỉa hè, 0.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(khách, 0.357013549187)</td>
      <td>(tuyến, 0.10371589336)</td>
      <td>(giao thông, 0.0916776788073)</td>
      <td>(hoạt động, 0.072613373482)</td>
      <td>(thông, 0.0706060528375)</td>
      <td>(TP HCM, 0.0666732073288)</td>
      <td>(sân bay, 0.0606328908184)</td>
      <td>(điểm, 0.0581519429325)</td>
      <td>(ôtô, 0.0579326538642)</td>
      <td>(biển, 0.056025531486)</td>
      <td>...</td>
      <td>(phủ, 0.00441674751716)</td>
      <td>(học sinh, 0.00440019234933)</td>
      <td>(lao động, 0.00415783429459)</td>
      <td>(cá, 0.00400994045621)</td>
      <td>(nguyên, 0.003553838099)</td>
      <td>(lũ, 0.0035309599934)</td>
      <td>(lớp, 0.002850488355)</td>
      <td>(trồng, 0.00240637492655)</td>
      <td>(ương, 0.00135022932472)</td>
      <td>(Đảng, 0.000860231254374)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(cháy, 0.477750923266)</td>
      <td>(cứu, 0.184855004457)</td>
      <td>(quận, 0.125228782474)</td>
      <td>(phường, 0.0983469383427)</td>
      <td>(giờ, 0.0929170099141)</td>
      <td>(phòng, 0.0765657682313)</td>
      <td>(ôtô, 0.0665709111664)</td>
      <td>(điện, 0.0660884508149)</td>
      <td>(máy, 0.0636483125902)</td>
      <td>(TP HCM, 0.0591469299284)</td>
      <td>...</td>
      <td>(phát triển, 0.00103609189795)</td>
      <td>(du lịch, 0.000992907716881)</td>
      <td>(tàu, 0.00084739660508)</td>
      <td>(phạm, 0.0)</td>
      <td>(xã hội, 0.0)</td>
      <td>(nghiên, 0.0)</td>
      <td>(hội, 0.0)</td>
      <td>(thanh tra, 0.0)</td>
      <td>(vi, 0.0)</td>
      <td>(mức, 0.0)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(con, 0.131278239915)</td>
      <td>(anh, 0.109593552663)</td>
      <td>(gia đình, 0.0980813981641)</td>
      <td>(tìm, 0.0862758177878)</td>
      <td>(đất, 0.0841687265553)</td>
      <td>(cây, 0.0780173583826)</td>
      <td>(tôi, 0.0733207308039)</td>
      <td>(mỗi, 0.0657747574104)</td>
      <td>(địa phương, 0.0646097522698)</td>
      <td>(sông, 0.0634975091531)</td>
      <td>...</td>
      <td>(sân bay, 0.00436049476767)</td>
      <td>(tướng, 0.00389921522001)</td>
      <td>(cháy, 0.00378857116692)</td>
      <td>(phạm, 0.00374739692524)</td>
      <td>(Thủ, 0.00365546010664)</td>
      <td>(vi, 0.003595245166)</td>
      <td>(Đảng, 0.00286183455503)</td>
      <td>(ương, 0.00243605200584)</td>
      <td>(tài xế, 0.00225353471201)</td>
      <td>(thanh tra, 0.00138239176123)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(đầu tư, 0.19560318513)</td>
      <td>(dự án, 0.169985797563)</td>
      <td>(tỷ, 0.136952517415)</td>
      <td>(tuyến, 0.129064603946)</td>
      <td>(thông, 0.123450483234)</td>
      <td>(phí, 0.116180356167)</td>
      <td>(giao thông, 0.111383629331)</td>
      <td>(thu, 0.0924882236736)</td>
      <td>(vốn, 0.0876700559417)</td>
      <td>(mức, 0.0832514200544)</td>
      <td>...</td>
      <td>(cháy, 0.0049408024796)</td>
      <td>(lũ, 0.00453640458887)</td>
      <td>(lớp, 0.00440513109881)</td>
      <td>(em, 0.00406643214559)</td>
      <td>(cá, 0.0037073830474)</td>
      <td>(trẻ, 0.00333761638998)</td>
      <td>(học sinh, 0.00273714650781)</td>
      <td>(chị, 0.00201791050005)</td>
      <td>(nghề, 0.00198292234875)</td>
      <td>(Đảng, 0.00105852201129)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(ôtô, 0.295098843386)</td>
      <td>(máy, 0.214222904537)</td>
      <td>(tài xế, 0.184626300465)</td>
      <td>(giao thông, 0.172624420118)</td>
      <td>(lộ, 0.145948029001)</td>
      <td>(biển, 0.112773607608)</td>
      <td>(quận, 0.0883645929755)</td>
      <td>(mạnh, 0.087734570529)</td>
      <td>(TP HCM, 0.0790193058316)</td>
      <td>(khách, 0.0752148452997)</td>
      <td>...</td>
      <td>(ủy, 0.00147193323881)</td>
      <td>(thư, 0.00138881492186)</td>
      <td>(phương án, 0.00116539083558)</td>
      <td>(bão, 0.00088845056627)</td>
      <td>(thanh tra, 0.000824647488927)</td>
      <td>(Đảng, 0.000701953876085)</td>
      <td>(trồng, 0.000565639268792)</td>
      <td>(lao động, 0.000535739604877)</td>
      <td>(ương, 0.000422646340588)</td>
      <td>(Bộ, 0.000409181213012)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Việt Nam, 0.130057422298)</td>
      <td>(hội, 0.111090349666)</td>
      <td>(tế, 0.107090174045)</td>
      <td>(sự, 0.105528489683)</td>
      <td>(Chính, 0.100107803037)</td>
      <td>(trưởng, 0.0984365506085)</td>
      <td>(phủ, 0.0948783355742)</td>
      <td>(Bộ, 0.0861916772158)</td>
      <td>(tướng, 0.0823381146365)</td>
      <td>(rằng, 0.0810921836496)</td>
      <td>...</td>
      <td>(lũ, 0.00501928422535)</td>
      <td>(học sinh, 0.00442790727743)</td>
      <td>(lộ, 0.00404650718081)</td>
      <td>(hồ, 0.00393760151782)</td>
      <td>(lớp, 0.00365836087905)</td>
      <td>(chị, 0.00357281131546)</td>
      <td>(cháy, 0.00253508085076)</td>
      <td>(ngập, 0.00236097574902)</td>
      <td>(vỉa hè, 0.00185207874131)</td>
      <td>(tài xế, 0.00159962680789)</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 127 columns</p>
</div>



So với KMeans với khoảng cách Euclide, khoảng cách Cosine không cho ta một nhóm hỗn tạp ở toạ độ 0 quá lớn. 


```python
[len(cluster) for cluster in clusters]
```




    [343, 1009, 540, 357, 545, 260, 1208, 800, 584, 1102]



Điều này do khoảng cách cosine là khoảng cách giữa các phương. Nhóm hỗn tạp đã bị chia và tách vào 10 nhóm thành phần. Kết quả về mặt nội dung tuy phần lớn tương tự, nhưng có sự xuất hiện của một số nội dung không tìm thấy khi dùng khoảng cách Euclide. 10 nhóm đó là:

```
0. Tàu, biển
1. UBND, xử lí vi phạm (toạ độ nhỏ, có thể được chia nhóm tiếp)
2. Giáo dục, trẻ em
3. Thời tiết, thiên tai
4. Giao thông
5. Cháy, cứu hộ (nhóm này có size nhỏ nhất)
6. Gia đình + nông nghiệp (toạ độ nhỏ, có thể được chia nhóm tiếp)
7. Đầu tư, dự án
8. Giao thông (có thể được hợp với nhóm 4)
9. Quốc hội, Chính phủ
```

Đây chỉ là kết quả về mặt tính toán. Tương tự như K-Means, người dùng có thể từ kết quả này, quyết định tăng $k$, phân nhóm các nhóm có size lớn thành nhỏ hơn (ví dụ chia 6 thành 2 nhóm mới, chia 9 thành 1 số nhóm mới), hợp nhóm các nhóm có size nhỏ và có liên quan về nội dung thành lớn hơn (ví dụ gộp 4, 8 thành 1 nhóm).

### Bài 18. Hierarchical Clustering

*Hãy sửa chữa hàm **`train, predict, getClusterCenters, getExplicatveFeaturesForEachCluster`** để khi đối số model trong hàm train nhận giá trị **"Hierarchical_Euclidean"** thì thuật toán Hierachical với khoảng cách Euclide được thực hiện. So sánh kết quả và thời gian chạy với **KMeans.** *

(Kết quả nhìn chung tương tự với KMeans, do cả 2 phương pháp đều dựa trên giả thiết khoảng cách nhỏ khi tương đồng)

## Bình luận thêm

TD3 và 5 minh hoạ một project thực tế nhỏ về xử lí văn bản (text-mining). Bước preprocessing thường mất rất nhiều thời gian. Việc chọn model hợp lí cũng không hiển nhiên, thường qua nhiều bước thử chọn. Do khuôn khổ TD, ta không thực hiện bước dự đoán với các văn bản mới (có thể không lấy từ vnexpress). Bạn có thể tự thực hiện phần này.

Bài toán clustering nhìn chung "ill-defined" vì không có tiêu chuẩn rõ ràng về mặt toán học để đánh giá phương pháp nào là tốt hay xấu. Dựa trên hiểu biết và kinh nghiệm về lĩnh vực, ta có thể chọn ra một mô hình hợp lí.

Mô hình có thể được cải tiến theo nhiều hướng sau:

- Thay đổi cách giảm số chiều của vector bằng cách chọn từ quan trọng (ví dụ, thay đổi chặn trên, dưới của tần số và phương sai. Trong TD, ta dùng các tham số 300, 1500, 0.8. Chúng có thể được thay đổi)

- Thay đổi metric

- Thay đổi cách số hoá vector

- Dùng mô hình khác

- Tái phân nhóm cluster hỗn tạp 1 hoặc nhiều lần...

- Dùng $k$ lớn, rồi hợp các nhóm nhỏ có nội dung liên quan

- Dùng $k$ nhỏ, rồi với từng nhóm lại chia nhóm lần nữa, nhiều lần...

Bạn có thể tự thực hiện một hướng nếu tò mò.

## Tham khảo

[1] Viện Ngôn ngữ học, GS. Hoàng Phê chủ biên, *Từ điển tiếng Việt*, NXB Hồng Đức (2003)

[2] http://vnexpress.net

[3] http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

[4] http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
