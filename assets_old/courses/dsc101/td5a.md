
# TD5 - Chia nhóm các bài báo thể thao trên VNExpress

## Mô tả

Ở TD3, chúng ta đã học kĩ năng thu thập dữ liệu từ Internet và xử lí dữ liệu để thu được các đoạn văn bản tương đối "sạch". Trong TD này, ta sẽ tiến hành chia nhóm các văn bản này.

### Tải dữ liệu

Khi chạy chương trình ở TD3 cho năm 2017 với thể loại **Thể thao**, ta thu được dữ liệu dưới dạng thư mục `RawData` <a href="https://drive.google.com/drive/folders/1u2IQzbDPvZ7Q_bE_uHMGkt5n9YF5QnmG?usp=sharing">ở đây</a>. Thư mục này gồm 12 file ứng với các tháng. Bạn cần tải thư mục này và lưu tại vị trí `Lesson5/TD/RawData`. Ta cũng cần làm việc với một danh sách các từ trong tiếng Việt, được trích xuất từ từ điển tiếng Việt của Hoàng Phê ([1]), nằm ở thư mục `VietnameseDictionary`. Bạn cần tải thư mục này và lưu tại `Lesson5/TD/VietnameseDictionary`. Cuối cùng, bạn có thể tải thư mục `FullData`, trong đó chứa kết quả bước 1 của TD trong trường hợp bạn không thực hiện thành công bước này.

### Quan sát sơ lược dữ liệu

Ta nhắc lại rằng ở TD3, mỗi bài báo của VNExpress được trích lọc tiêu đề, đoạn giới thiệu và nội dung. Ba thành phần này của mỗi bài báo đã được chúng ta lưu bằng một hàng của file dữ liệu, chúng cách nhau bởi hai khoảng trắng tab (`"\t\t"`). Riêng với thành phần thứ ba (nội dung của bài báo), các đoạn văn cách nhau bởi một khoảng trắng tab (`"\t"`).

Một số bài báo không có nội dung được lưu bằng một hàng trắng, không có kí tự nào khác ngoài kí tự xuống dòng (`"\n"`). Bạn có thể xem ví dụ từ <a href="https://drive.google.com/file/d/1aTV-WckP8idERHe33Kd7Fby3pwX1nZQD/view">RawData/Sport_012017</a>, hàng 31 là một hàng trắng.

### Mục tiêu

Từ 12 file dữ liệu "thô" này, ta sẽ thực hiện việc tìm hiểu các bài báo thể thao năm 2017 của vnexpress có thể được chia thành những nhóm nào dựa trên sự tương đồng về nội dung. Ta sẽ thực hiện theo quy trình sau:

- Bước 1: Preprocessing 1 - Biến bài báo thành bag-of-words
 - Ghép 12 file dữ liệu thành file duy nhất và xoá các hàng trắng. File này tương tự như `FullData/Sport2017_Solution.txt`.
 - Tách nội dung mỗi bài báo thành một túi từ (bag of words), gồm mỗi từ và các tần số của nó trong mỗi bài báo, và lưu nó vào một file. File này tương tự như `FullData/FrequencyBySportArticle_Solution.txt`
 - Tính tần số tổng cộng của các từ trên tất cả các bài báo và lưu nó vào một file. File này tương tự như `GlobalSportFrequency_Solution.txt`. File này sẽ giúp ta xác định những từ nào là thông dụng, ít thông dụng trong tiếng Việt để xử lí ở các phần sau.
 
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
ALL_DATA_FILE = "FullData/Sport2017.txt"
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
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




    5949




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
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
      <td>Thất bại 0-1 trên sân Burnley hôm 31/1 nối dài...</td>
      <td>Kết quả trận đấu tại Turf Moor được định đoạt ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
      <td>Thầy trò HLV Mauricio Pochettino hoà 0-0 trên ...</td>
      <td>Tottenham hành quân đến sân Ánh Sáng với tham ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
      <td>Cựu tiền đạo của AC Milan là tên tuổi mới nhất...</td>
      <td>Alexandre Pato đã hoàn tất vụ chuyển nhượng đế...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
      <td>Cựu thủ quân đội tuyển Anh tiết lộ sự thật đằn...</td>
      <td>"Tôi không hận thù, nhưng thực sự lúc đó cảm t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
      <td>Mauricio Pochettino tỏ ra rất thực tế về cơ hộ...</td>
      <td>Chelsea hiện dẫn đầu qua 22 vòng đấu với 55 đi...</td>
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
      <th>5944</th>
      <td>Real Madrid áp đảo các đề cử giải thưởng Bóng ...</td>
      <td>Đội chủ sân Bernabeu góp ứng cử viên ở một loạ...</td>
      <td>Cristiano Ronaldo và Sergio Ramos là hai ứng c...</td>
    </tr>
    <tr>
      <th>5945</th>
      <td>Văn Quyết bị gạch tên khỏi đề cử Quả Bóng Vàng...</td>
      <td>Tiền đạo vừa nhận án kỷ luật do lỗi đánh cầu t...</td>
      <td>"Ban tổ chức, sau khi tiến hành họp chiều 30/1...</td>
    </tr>
    <tr>
      <th>5946</th>
      <td>Real 'bị loại' khỏi Cup Nhà Vua vì lỗi đánh máy</td>
      <td>Trang web của LĐBĐ Tây Ban Nha hiển thị thông ...</td>
      <td>Lỗi văn bản xuất hiện khi trang web Liên đoàn ...</td>
    </tr>
    <tr>
      <th>5947</th>
      <td>Conte bị phạt hơn 10.000 đôla vì lỗi phản ứng ...</td>
      <td>HLV của Chelsea chỉ nộp phạt tiền sau khi bị t...</td>
      <td>BBC và Sky Sports cho biết, Antonio Conte sẽ n...</td>
    </tr>
    <tr>
      <th>5948</th>
      <td>Allardyce ký hợp đồng 18 tháng với Everton</td>
      <td>Cựu HLV tuyển Anh nhận lời giải cứu đội bóng t...</td>
      <td>Sam Allardyce trở lại Anh hôm 29/11, sau kỳ ng...</td>
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




    29567



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




    5949




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
      <th>216</th>
      <th>217</th>
      <th>218</th>
      <th>219</th>
      <th>220</th>
      <th>221</th>
      <th>222</th>
      <th>223</th>
      <th>224</th>
      <th>225</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lập</td>
      <td>thời</td>
      <td>Turf Moor</td>
      <td>đang</td>
      <td>tinh thần</td>
      <td>hạng</td>
      <td>nhất</td>
      <td>bóng</td>
      <td>còn</td>
      <td>Mike Dean</td>
      <td>...</td>
      <td>lục</td>
      <td>sân</td>
      <td>ghi</td>
      <td>đấu</td>
      <td>trước</td>
      <td>một mình</td>
      <td>ngày</td>
      <td>trận</td>
      <td>vẫn</td>
      <td>tại</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 226 columns</p>
</div>



Ví dụ trên cho thấy nội dung bài báo thứ nhất đã được chuyển đổi thành các bộ từ : tần số. Từ *"lập"* xuất hiện 1 lần, từ *"tinh thần"* xuất hiện 1 lần, từ *"Turf Moor"* xuất hiện 2 lần v.v...

### Bài 8. Lưu túi từ vào file

Để các bước tiếp theo không phải thực hiện lại quá trình preprocessing1 này, ta lưu kết quả đã thực hiện vào file (Trong thực tế, người ta lưu vào cơ sở dữ liệu thay vì file). Ta sẽ cần 2 file:

- Một file như `FullData/FrequencyBySportArticle_Solution.txt` chứa tiêu đề các bài báo và túi từ tương ứng của nó.
- Một file như `FullData/GlobalSportFrequency_Solution.txt` chứa tổng tần số của tất cả các từ xuất hiện ít nhất trong 1 bài báo, lấy trên tất cả các bài báo.

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
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
      <td>lập:1\tthời:1\tTurf Moor:2\tđang:3\ttinh thần:...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
      <td>đánh rơi:1\tphần nhiều:1\tđang:1\thạng:1\tnhất...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
      <td>Brazil:1\thạng:1\tnhất:2\tỞ:1\tCựu:1\tgiá:1\th...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
      <td>đang:1\tReal Madrid:2\txem:2\tgiành:1\tbóng:1\...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
      <td>đua:2\txem:1\tđang:2\tgiúp:1\thạng:4\tnhất:1\t...</td>
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
      <td>của</td>
      <td>5568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>và</td>
      <td>5552</td>
    </tr>
    <tr>
      <th>2</th>
      <td>là</td>
      <td>5321</td>
    </tr>
    <tr>
      <th>3</th>
      <td>với</td>
      <td>5311</td>
    </tr>
    <tr>
      <th>4</th>
      <td>trong</td>
      <td>5301</td>
    </tr>
    <tr>
      <th>5</th>
      <td>có</td>
      <td>5114</td>
    </tr>
    <tr>
      <th>6</th>
      <td>khi</td>
      <td>5078</td>
    </tr>
    <tr>
      <th>7</th>
      <td>một</td>
      <td>4984</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ở</td>
      <td>4805</td>
    </tr>
    <tr>
      <th>9</th>
      <td>không</td>
      <td>4784</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cho</td>
      <td>4776</td>
    </tr>
    <tr>
      <th>11</th>
      <td>được</td>
      <td>4686</td>
    </tr>
    <tr>
      <th>12</th>
      <td>người</td>
      <td>4448</td>
    </tr>
    <tr>
      <th>13</th>
      <td>trận</td>
      <td>4228</td>
    </tr>
    <tr>
      <th>14</th>
      <td>đã</td>
      <td>4184</td>
    </tr>
    <tr>
      <th>15</th>
      <td>vào</td>
      <td>4160</td>
    </tr>
    <tr>
      <th>16</th>
      <td>này</td>
      <td>4140</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sau</td>
      <td>4137</td>
    </tr>
    <tr>
      <th>18</th>
      <td>để</td>
      <td>3964</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ra</td>
      <td>3957</td>
    </tr>
  </tbody>
</table>
</div>



## Phần 2: Preprocessing 2 - Số hoá túi từ

Từ thời điểm này, ta chỉ cần làm việc với 3 file, trong đó 2 file sau chính là kết quả của bước 1.
```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
```

Để thuận tiện cho việc thống nhất số liệu để so sánh kết quả với hướng dẫn của TD, bạn có thể dùng 2 file `..._Solution.txt` và đổi tên thành các file trên.


### Bài 9. Giảm số chiều bằng cách chọn các từ quan trọng

Để các thuật toán clustering có thể chạy được, ta phải biến các túi từ thành các vector, trong đó mỗi toạ độ thể hiện một từ. Vấn đề đầu tiên đặt ra: ta sử dụng hết tất cả các từ hiện có hay có thể chọn lọc trước một số từ quan trọng. Đây là bài toán giảm số chiều (reduction of dimension) trong machine learning. Ta chưa đi sâu vào các kĩ thuật cho bài toán này, mà sử dụng phương pháp đơn giản dựa vào quan sát file `FullData/GlobalSportFrequency.txt`.

Ta thấy:

- Những từ có tần số quá lớn thường không có tính phân loại, tức không quan trọng. Ví dụ các từ "và", "của", "hoặc"... có thể xuất hiện trong rất nhiều bài báo, không thể căn cứ vào đó để chia nhóm nội dung.

- Những từ có tần số quá nhỏ cũng không quan trọng. Đó là những từ chỉ xuất hiện đơn lẻ trong một vài bài báo, không đặc trưng cho một nhóm.

- Những từ xuất hiện đều đặn trong các bài báo (có tần số gần như nhau trong hầu hết tất cả bài báo) thường cũng không thể được dùng để phân loại các bài báo. 

Do đó, ta sẽ chọn ra những từ thoả mãn điều kiện sau:

- Có tần số nhỏ hơn hoặc bằng một giá trị chặn trên **upperbound**

- Có tần số lớn hơn hoặc bằng một giá trị chặn dưới **lowerbound**

- Có phương sai của list tần số lấy trên tất cả các bài báo lớn hơn hoặc bằng một giá trị **var_lowerbound**

Giả sử sau khi chọn xong, ta giữ lại được 5 từ (`"thể thao", "bóng đá", "Việt Nam", "Messi", "Ronaldo")`, ta có thể gán cho chúng 5 toạ độ, ví dụ `("thể thao" -> 0, "Messi" -> 1, "Ronaldo" -> 2, "Việt Nam" -> 3, "bóng đá" -> 4)`. Khi đó túi từ `("Việt Nam":2, "thể thao":1, "huy chương": 2, "là": 1)` sẽ được biến thành vector `[1, 0, 0, 2, 0]`.

*Hãy viết hàm **`getExplicativeFeatures(global_frequency_file, frequency_file, lowerbound, upperbound, var_lower_bound)`** nhận 5 đối số theo thứ tự là đường dẫn file túi từ theo bài báo, đường dẫn file tần số tổng quát, chặn dưới của tần số cần chọn, chặn trên của tần số cần chọn, chặn dưới của phương sai cần chọn, và trả lại một **dict** của Python gồm các key là các từ "có tính giải thích" được chọn và value tương ứng là số thứ tự của toạ độ.*

Trong ví dụ trên, kết quả trả lại cần là: `{"thể thao": 0, "Messi": 1, "Ronaldo": 2, "Việt Nam": 3, "bóng đá": 4}`

Đoạn code dưới đây giúp kiểm tra hàm của bạn.


```python
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5
```


```python
features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
features
```




    {'Argentina': 28,
     'Arsenal': 11,
     'Atletico': 10,
     'Bayern': 7,
     'Brazil': 12,
     'Chelsea': 18,
     'HC': 32,
     'Juventus': 30,
     'La Liga': 13,
     'Liverpool': 0,
     'Man City': 35,
     'Messi': 21,
     'Monaco': 19,
     'Mourinho': 16,
     'M\xe1\xbb\xb9': 1,
     'Neymar': 9,
     'PSG': 23,
     'Ronaldo': 8,
     'Tottenham': 40,
     'T\xc3\xa2y Ban Nha': 15,
     'Vi\xe1\xbb\x87t Nam': 3,
     'World Cup': 4,
     'Zidane': 22,
     'b\xc3\xa1n': 14,
     'b\xe1\xba\xa1n': 26,
     'c\xe1\xba\xadu': 42,
     'danh hi\xe1\xbb\x87u': 6,
     'gi\xc3\xa2y': 29,
     'hi\xe1\xbb\x87p': 5,
     'h\xe1\xbb\xa3p \xc4\x91\xe1\xbb\x93ng': 33,
     'm\xc3\xacnh': 25,
     'n\xe1\xbb\xaf': 41,
     'ph\xe1\xba\xa1t': 31,
     'quy\xe1\xbb\x81n': 36,
     'th\xc3\xac': 24,
     'ti\xe1\xbb\x81n': 38,
     'tr\xe1\xba\xbb': 34,
     'tr\xe1\xbb\x8dng t\xc3\xa0i': 37,
     'tuy\xe1\xbb\x83n': 39,
     'v\xc3\xa0ng': 20,
     'v\xe1\xbb\xa3t': 2,
     '\xc4\x91ua': 17,
     '\xc4\x91\xc3\xa1nh': 27}




```python
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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brazil</td>
      <td>bán</td>
      <td>Neymar</td>
      <td>đua</td>
      <td>mình</td>
      <td>Monaco</td>
      <td>Tây Ban Nha</td>
      <td>PSG</td>
      <td>hiệp</td>
      <td>HC</td>
      <td>...</td>
      <td>Chelsea</td>
      <td>Messi</td>
      <td>Man City</td>
      <td>Zidane</td>
      <td>Arsenal</td>
      <td>vợt</td>
      <td>danh hiệu</td>
      <td>phạt</td>
      <td>thì</td>
      <td>quyền</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>14</td>
      <td>9</td>
      <td>17</td>
      <td>25</td>
      <td>19</td>
      <td>15</td>
      <td>23</td>
      <td>5</td>
      <td>32</td>
      <td>...</td>
      <td>18</td>
      <td>21</td>
      <td>35</td>
      <td>22</td>
      <td>11</td>
      <td>2</td>
      <td>6</td>
      <td>31</td>
      <td>24</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 43 columns</p>
</div>



### Bài 10. Biến túi từ thành các vector có toạ độ 0/1

Ta cần phân nhóm các bài báo theo độ tương đồng về nội dung. Theo đó, các vector ứng với các bài báo tương đồng (sử dụng chung một lượng từ "quan trọng" giống nhau) cần có khoảng cách nhỏ, còn các vector ứng với các bài báo không tương đồng có khoảng cách lớn. 

Vì điều kiện này, việc số hoá các túi từ thành các vector có toạ độ là tần số các từ không phải là lựa chọn tốt. 

Ví dụ xét 3 túi từ: 

- Túi từ `A {"Việt Nam": 1, "bóng đá": 2}`
- Túi từ `B {"Việt Nam": 2, "bóng đá": 4}`
- Túi từ `C {"Messi": 1, "Ronaldo": 1}`

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
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
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
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
      <th>3</th>
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
    <tr>
      <th>4</th>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
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
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tết buồn của cựu võ sĩ Trần Kim Tuyến</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Arsenal, Man Utd và Chelsea tránh nhau ở FA Cup</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cầu thủ Leicester City: 'Ranieri đã phản bội tôi'</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tiến Minh và Vũ Thị Trang lần đầu đánh giải sa...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Suarez: 'Bóng đã qua vạch vôi cả mét'</td>
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
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5

features = getExplicativeFeatures(GLOBAL_FREQUENCY_FILE, FREQUENCY_FILE, MIN_AVG, MAX_AVG, MIN_DEV)
X = articlesToSparseVector(FREQUENCY_FILE, features, COORDINATES_CODING_MODE)
predictive_model = train(X, NB_CLUSTER, model=MODEL)
prediction = predict(predictive_model, X)
prediction
```




    array([2, 3, 3, ..., 1, 3, 2])




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
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tết buồn của cựu võ sĩ Trần Kim Tuyến</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Arsenal, Man Utd và Chelsea tránh nhau ở FA Cup</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cầu thủ Leicester City: 'Ranieri đã phản bội tôi'</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tiến Minh và Vũ Thị Trang lần đầu đánh giải sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Suarez: 'Bóng đã qua vạch vôi cả mét'</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Theo kết quả trên, bài báo 1 và 4 thuộc cùng 1 nhóm (chúng cùng liên quan đến Tottenham).

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
      <td>Tiến Minh và Vũ Thị Trang lần đầu đánh giải sa...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Việt Nam cùng bảng với Campuchia ở vòng loại c...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ban trọng tài VFF: ‘Samson chỉ trượt lên đầu g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HAGL bất bình với pha ra đòn của cầu thủ Hà Nộ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>'Nữ hoàng đấu kiếm' Lệ Dung liên tục bị thất h...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Man City đón tân binh trị giá 33,3 triệu đôla</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Tài năng bóng đá Việt Nam có cơ hội tập luyện ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bác sĩ tuyển Việt Nam: 'Không điều trị sai cho...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hoàng Xuân Vinh, Ánh Viên được vinh danh tại C...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HLV Hữu Thắng: 'Bác sĩ làm lỡ hết kế hoạch của...</td>
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
      <td>Suarez: 'Bóng đã qua vạch vôi cả mét'</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Real nới rộng cách biệt tại Liga bằng chiến th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Suarez giải cứu Barca bằng bàn thắng ở phút 90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tháng 2 đầy chông gai đang chờ đón Barca</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alba: 'Real bị loại và lập tức hạ thấp Cup Nhà...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Neymar thất bại khi lôi kéo Coutinho về Barca</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Real thua trong vụ kiện kênh truyền hình bôi x...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Scolari: 'Ronaldo giành Quả Bóng Vàng nhờ ý ch...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mất Modric và Marcelo, Real lâm vào khủng hoản...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Zidane họp khẩn với cả đội Real sau trận thắng...</td>
    </tr>
  </tbody>
</table>
</div>



Như ví dụ trên, nhóm 0 gồm các bài báo có vẻ tương tự nhau, liên quan đến thể thao Việt Nam (không chính xác 100%). Nhóm 1, tương tự, nếu bạn có hiểu biết về bóng đá, sẽ cảm thấy nó liên quan đến bóng đá Tây Ban Nha. 

Trong bài này ta dùng $k = 4$. Bạn có thể kiểm tra kết quả 4 nhóm xem chúng có đặc trưng gì.

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002759</td>
      <td>0.082759</td>
      <td>0.048276</td>
      <td>0.986207</td>
      <td>0.099310</td>
      <td>0.160000</td>
      <td>0.056552</td>
      <td>0.000000</td>
      <td>0.002759</td>
      <td>0.001379</td>
      <td>...</td>
      <td>0.095172</td>
      <td>0.280000</td>
      <td>0.005517</td>
      <td>0.204138</td>
      <td>0.100690</td>
      <td>0.086897</td>
      <td>0.288276</td>
      <td>0.002759</td>
      <td>0.220690</td>
      <td>0.073103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.053763</td>
      <td>0.030722</td>
      <td>0.005376</td>
      <td>0.000768</td>
      <td>0.141321</td>
      <td>0.099846</td>
      <td>0.211982</td>
      <td>0.053763</td>
      <td>0.294163</td>
      <td>0.324885</td>
      <td>...</td>
      <td>0.196621</td>
      <td>0.109063</td>
      <td>0.055300</td>
      <td>0.145929</td>
      <td>0.078341</td>
      <td>0.138249</td>
      <td>0.135177</td>
      <td>0.046851</td>
      <td>0.013057</td>
      <td>0.160522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.042113</td>
      <td>0.122541</td>
      <td>0.101829</td>
      <td>0.000000</td>
      <td>0.089058</td>
      <td>0.166724</td>
      <td>0.150155</td>
      <td>0.059027</td>
      <td>0.050742</td>
      <td>0.013462</td>
      <td>...</td>
      <td>0.127028</td>
      <td>0.161201</td>
      <td>0.053849</td>
      <td>0.158440</td>
      <td>0.087332</td>
      <td>0.108733</td>
      <td>0.086296</td>
      <td>0.027270</td>
      <td>0.037625</td>
      <td>0.124957</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.317073</td>
      <td>0.035122</td>
      <td>0.004878</td>
      <td>0.003902</td>
      <td>0.069268</td>
      <td>0.200976</td>
      <td>0.174634</td>
      <td>0.074146</td>
      <td>0.014634</td>
      <td>0.020488</td>
      <td>...</td>
      <td>0.234146</td>
      <td>0.130732</td>
      <td>0.448780</td>
      <td>0.128780</td>
      <td>0.074146</td>
      <td>0.131707</td>
      <td>0.111220</td>
      <td>0.292683</td>
      <td>0.005854</td>
      <td>0.164878</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 43 columns</p>
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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Việt Nam, 0.986206896552)</td>
      <td>(tuyển, 0.288275862069)</td>
      <td>(trẻ, 0.28)</td>
      <td>(nữ, 0.220689655172)</td>
      <td>(thì, 0.216551724138)</td>
      <td>(vàng, 0.213793103448)</td>
      <td>(HC, 0.212413793103)</td>
      <td>(quyền, 0.204137931034)</td>
      <td>(mình, 0.198620689655)</td>
      <td>(đánh, 0.198620689655)</td>
      <td>...</td>
      <td>(Ronaldo, 0.00275862068966)</td>
      <td>(Neymar, 0.00137931034483)</td>
      <td>(La Liga, 0.00137931034483)</td>
      <td>(Juventus, 0.00137931034483)</td>
      <td>(Chelsea, 0.00137931034483)</td>
      <td>(Arsenal, 0.00137931034483)</td>
      <td>(Monaco, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Zidane, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(La Liga, 0.625960061444)</td>
      <td>(Messi, 0.367895545315)</td>
      <td>(Neymar, 0.324884792627)</td>
      <td>(Tây Ban Nha, 0.323348694316)</td>
      <td>(Ronaldo, 0.294162826421)</td>
      <td>(PSG, 0.275729646697)</td>
      <td>(Brazil, 0.251152073733)</td>
      <td>(danh hiệu, 0.21198156682)</td>
      <td>(Argentina, 0.198156682028)</td>
      <td>(hợp đồng, 0.196620583717)</td>
      <td>...</td>
      <td>(Monaco, 0.042242703533)</td>
      <td>(Chelsea, 0.0353302611367)</td>
      <td>(Mỹ, 0.0307219662058)</td>
      <td>(Arsenal, 0.0245775729647)</td>
      <td>(Mourinho, 0.0238095238095)</td>
      <td>(nữ, 0.0130568356375)</td>
      <td>(giây, 0.00844854070661)</td>
      <td>(HC, 0.00537634408602)</td>
      <td>(vợt, 0.00537634408602)</td>
      <td>(Việt Nam, 0.000768049155146)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(đánh, 0.198481187435)</td>
      <td>(hiệp, 0.166724197446)</td>
      <td>(trẻ, 0.161201242665)</td>
      <td>(quyền, 0.158439765274)</td>
      <td>(danh hiệu, 0.150155333103)</td>
      <td>(bán, 0.148429409734)</td>
      <td>(thì, 0.140490162237)</td>
      <td>(mình, 0.128408698654)</td>
      <td>(hợp đồng, 0.127027959959)</td>
      <td>(cậu, 0.124956851916)</td>
      <td>...</td>
      <td>(Brazil, 0.0355540214014)</td>
      <td>(Atletico, 0.0279599585778)</td>
      <td>(Tottenham, 0.0272695892302)</td>
      <td>(HC, 0.0245081118398)</td>
      <td>(Zidane, 0.0165688643424)</td>
      <td>(Messi, 0.0162236796686)</td>
      <td>(Neymar, 0.0134622022782)</td>
      <td>(La Liga, 0.0031066620642)</td>
      <td>(Chelsea, 0.000690369347601)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Chelsea, 0.767804878049)</td>
      <td>(Arsenal, 0.453658536585)</td>
      <td>(Man City, 0.448780487805)</td>
      <td>(Liverpool, 0.317073170732)</td>
      <td>(Tottenham, 0.292682926829)</td>
      <td>(hợp đồng, 0.234146341463)</td>
      <td>(hiệp, 0.200975609756)</td>
      <td>(Tây Ban Nha, 0.185365853659)</td>
      <td>(thì, 0.185365853659)</td>
      <td>(Mourinho, 0.184390243902)</td>
      <td>...</td>
      <td>(giây, 0.0214634146341)</td>
      <td>(Neymar, 0.020487804878)</td>
      <td>(vàng, 0.020487804878)</td>
      <td>(Ronaldo, 0.0146341463415)</td>
      <td>(Messi, 0.0117073170732)</td>
      <td>(Zidane, 0.00878048780488)</td>
      <td>(nữ, 0.00585365853659)</td>
      <td>(vợt, 0.00487804878049)</td>
      <td>(Việt Nam, 0.00390243902439)</td>
      <td>(HC, 0.00292682926829)</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 43 columns</p>
</div>



Đến đây ta cảm thấy rõ hơn về cách chia nhóm do KMeans thực hiện:

- Nhóm 0 liên quan đến thể thao Việt Nam (toạ độ của "Việt Nam" chiếm áp đảo với 0.98)

- Nhóm 1 liên quan đến bóng đá Tây Ban Nha (bạn có thể tìm hiểu "La Liga", "Messi", "Neymar", ... là ai/gì nếu chưa biết)

- Nhóm 2 là một nhóm hỗn tạp (như quan sát, các toạ độ thành phần của tâm đều nhỏ, gần 0 hơn 1, tức không có từ quan trọng nổi trội quyết định nội dung chung của nhóm)

- Nhóm 3 liên quan đến bóng đá Anh (bạn có thể tìm hiểu "Chelse"a, "Arsenal", "Man City", ... là gì nếu chưa biết)

Thứ tự các nhóm có thể thay đổi trong chương trình của bạn.

## Phần 4 - Thử nghiệm các thao tác khác

### Bài 15 - Thay đổi k

Việc chọn $k=4$ có thể dẫn đến một số nhóm bị gộp vào một nhóm chung (thường là nhóm ở gốc toạ độ 0). Ta thử thay đổi $k$ xem kết quả có tiến triển không.

*Bạn hãy thử thay đổi $k$, tính thế năng tương ứng, vẽ đồ thị thế năng theo $k$ và nhận xét. Không có hàm nào cần viết thêm trong bài này.*

Kết quả có dạng như hình sau:

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson5/TD/figure_2.png" width=600></img>

Thực tế, đồ thị trên đề xuất ta chọn $k$ rất lớn, nhưng theo quan sát kết quả, ta có thể hài lòng với $k$ nhỏ. Chẳng hạn $k = 8$ dưới đây.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
COORDINATES_CODING_MODE = "0-1"
NB_CLUSTER = 8
MODEL = "KMeans"
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(đánh, 0.875179340029)</td>
      <td>(hiệp, 0.583931133429)</td>
      <td>(phạt, 0.272596843615)</td>
      <td>(quyền, 0.253945480631)</td>
      <td>(trọng tài, 0.233859397418)</td>
      <td>(thì, 0.170731707317)</td>
      <td>(bán, 0.163558106169)</td>
      <td>(trẻ, 0.153515064562)</td>
      <td>(mình, 0.124820659971)</td>
      <td>(Mỹ, 0.123385939742)</td>
      <td>...</td>
      <td>(Messi, 0.0444763271162)</td>
      <td>(Brazil, 0.0401721664275)</td>
      <td>(Atletico, 0.0401721664275)</td>
      <td>(nữ, 0.0315638450502)</td>
      <td>(PSG, 0.0286944045911)</td>
      <td>(Neymar, 0.0272596843615)</td>
      <td>(Monaco, 0.0243902439024)</td>
      <td>(HC, 0.0143472022956)</td>
      <td>(danh hiệu, 0.0143472022956)</td>
      <td>(Zidane, 0.0100430416069)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Việt Nam, 0.980625931446)</td>
      <td>(trẻ, 0.286140089419)</td>
      <td>(tuyển, 0.283159463487)</td>
      <td>(nữ, 0.239940387481)</td>
      <td>(HC, 0.235469448584)</td>
      <td>(vàng, 0.233979135618)</td>
      <td>(thì, 0.207153502235)</td>
      <td>(quyền, 0.202682563338)</td>
      <td>(mình, 0.18479880775)</td>
      <td>(đánh, 0.160953800298)</td>
      <td>...</td>
      <td>(Mourinho, 0.00149031296572)</td>
      <td>(Juventus, 0.00149031296572)</td>
      <td>(Chelsea, 0.00149031296572)</td>
      <td>(Arsenal, 0.00149031296572)</td>
      <td>(Monaco, 0.0)</td>
      <td>(La Liga, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Tottenham, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Zidane, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Chelsea, 0.725341426404)</td>
      <td>(Arsenal, 0.61153262519)</td>
      <td>(Man City, 0.608497723824)</td>
      <td>(Liverpool, 0.433990895296)</td>
      <td>(Tottenham, 0.402124430956)</td>
      <td>(hợp đồng, 0.215477996965)</td>
      <td>(hiệp, 0.212443095599)</td>
      <td>(thì, 0.197268588771)</td>
      <td>(Mourinho, 0.177541729894)</td>
      <td>(Tây Ban Nha, 0.166919575114)</td>
      <td>...</td>
      <td>(Mỹ, 0.0257966616085)</td>
      <td>(vàng, 0.0242792109256)</td>
      <td>(Ronaldo, 0.0121396054628)</td>
      <td>(Messi, 0.01062215478)</td>
      <td>(Zidane, 0.00758725341426)</td>
      <td>(Neymar, 0.00606980273141)</td>
      <td>(Việt Nam, 0.00455235204856)</td>
      <td>(vợt, 0.00303490136571)</td>
      <td>(HC, 0.00151745068285)</td>
      <td>(nữ, 0.00151745068285)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(hợp đồng, 0.152542372881)</td>
      <td>(trẻ, 0.138241525424)</td>
      <td>(bán, 0.127118644068)</td>
      <td>(quyền, 0.126588983051)</td>
      <td>(cậu, 0.125)</td>
      <td>(thì, 0.116525423729)</td>
      <td>(Tây Ban Nha, 0.112288135593)</td>
      <td>(mình, 0.105402542373)</td>
      <td>(tiền, 0.105402542373)</td>
      <td>(bạn, 0.104343220339)</td>
      <td>...</td>
      <td>(nữ, 0.0259533898305)</td>
      <td>(Tottenham, 0.0233050847458)</td>
      <td>(Zidane, 0.0195974576271)</td>
      <td>(HC, 0.0169491525424)</td>
      <td>(Neymar, 0.00900423728814)</td>
      <td>(World Cup, 0.0)</td>
      <td>(La Liga, 0.0)</td>
      <td>(đánh, 0.0)</td>
      <td>(Việt Nam, 0.0)</td>
      <td>(danh hiệu, 0.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(La Liga, 0.991084695394)</td>
      <td>(Tây Ban Nha, 0.337295690936)</td>
      <td>(Ronaldo, 0.323922734027)</td>
      <td>(Messi, 0.27191679049)</td>
      <td>(danh hiệu, 0.245170876672)</td>
      <td>(Atletico, 0.197622585438)</td>
      <td>(Zidane, 0.194650817236)</td>
      <td>(đua, 0.145616641902)</td>
      <td>(Argentina, 0.139673105498)</td>
      <td>(bán, 0.132243684993)</td>
      <td>...</td>
      <td>(World Cup, 0.0326894502229)</td>
      <td>(Mỹ, 0.0282317979198)</td>
      <td>(Monaco, 0.0267459138187)</td>
      <td>(Mourinho, 0.0267459138187)</td>
      <td>(Arsenal, 0.0252600297177)</td>
      <td>(nữ, 0.00594353640416)</td>
      <td>(giây, 0.00297176820208)</td>
      <td>(vợt, 0.00297176820208)</td>
      <td>(Việt Nam, 0.00148588410104)</td>
      <td>(HC, 0.0)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(World Cup, 0.964028776978)</td>
      <td>(tuyển, 0.553956834532)</td>
      <td>(Argentina, 0.306954436451)</td>
      <td>(Tây Ban Nha, 0.275779376499)</td>
      <td>(Brazil, 0.258992805755)</td>
      <td>(Messi, 0.189448441247)</td>
      <td>(trẻ, 0.179856115108)</td>
      <td>(cậu, 0.172661870504)</td>
      <td>(quyền, 0.170263788969)</td>
      <td>(hợp đồng, 0.16067146283)</td>
      <td>...</td>
      <td>(nữ, 0.0383693045564)</td>
      <td>(Bayern, 0.0359712230216)</td>
      <td>(Việt Nam, 0.0335731414868)</td>
      <td>(Mourinho, 0.0287769784173)</td>
      <td>(Zidane, 0.0287769784173)</td>
      <td>(HC, 0.0239808153477)</td>
      <td>(đua, 0.0215827338129)</td>
      <td>(Atletico, 0.0143884892086)</td>
      <td>(giây, 0.00719424460432)</td>
      <td>(vợt, 0.00719424460432)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Neymar, 0.872817955112)</td>
      <td>(PSG, 0.817955112219)</td>
      <td>(Brazil, 0.551122194514)</td>
      <td>(hợp đồng, 0.418952618454)</td>
      <td>(tiền, 0.326683291771)</td>
      <td>(Messi, 0.316708229426)</td>
      <td>(Tây Ban Nha, 0.254364089776)</td>
      <td>(quyền, 0.229426433915)</td>
      <td>(cậu, 0.224438902743)</td>
      <td>(La Liga, 0.197007481297)</td>
      <td>...</td>
      <td>(Arsenal, 0.0374064837905)</td>
      <td>(vàng, 0.0349127182045)</td>
      <td>(Atletico, 0.0299251870324)</td>
      <td>(giây, 0.0199501246883)</td>
      <td>(Mourinho, 0.0199501246883)</td>
      <td>(Zidane, 0.0199501246883)</td>
      <td>(nữ, 0.0174563591022)</td>
      <td>(vợt, 0.0074812967581)</td>
      <td>(HC, 0.00498753117207)</td>
      <td>(Việt Nam, 0.00249376558603)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(danh hiệu, 1.0)</td>
      <td>(mình, 0.217311233886)</td>
      <td>(trẻ, 0.21546961326)</td>
      <td>(vợt, 0.213627992634)</td>
      <td>(bán, 0.206261510129)</td>
      <td>(thì, 0.198895027624)</td>
      <td>(bạn, 0.195211786372)</td>
      <td>(cậu, 0.19152854512)</td>
      <td>(Mỹ, 0.16758747698)</td>
      <td>(hợp đồng, 0.16758747698)</td>
      <td>...</td>
      <td>(World Cup, 0.036832412523)</td>
      <td>(giây, 0.036832412523)</td>
      <td>(Argentina, 0.0349907918969)</td>
      <td>(Brazil, 0.0331491712707)</td>
      <td>(La Liga, 0.0276243093923)</td>
      <td>(Việt Nam, 0.0184162062615)</td>
      <td>(Atletico, 0.0184162062615)</td>
      <td>(Zidane, 0.0184162062615)</td>
      <td>(trọng tài, 0.0147329650092)</td>
      <td>(Neymar, 0.0110497237569)</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 43 columns</p>
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
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
      <td>Luis van Gaal và con đường dang dở</td>
      <td>Man Utd đứt mạch bất bại khi vào chung kết Cup...</td>
      <td>Trút mưa bàn thắng, Barca vào bán kết Cup Nhà vua</td>
      <td>Liverpool thất thủ, bị loại ở bán kết Cúp Liên...</td>
      <td>Ronaldo sút phạt ghi bàn, nhưng Real bị loại k...</td>
      <td>ĐKVĐ Bờ Biển Ngà bị loại ngay từ vòng bảng CAN...</td>
      <td>Sao tiền vệ PSG nhận thẻ vàng kỳ lạ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tiến Minh và Vũ Thị Trang lần đầu đánh giải sa...</td>
      <td>Việt Nam cùng bảng với Campuchia ở vòng loại c...</td>
      <td>Ban trọng tài VFF: ‘Samson chỉ trượt lên đầu g...</td>
      <td>HAGL bất bình với pha ra đòn của cầu thủ Hà Nộ...</td>
      <td>'Nữ hoàng đấu kiếm' Lệ Dung liên tục bị thất h...</td>
      <td>Tài năng bóng đá Việt Nam có cơ hội tập luyện ...</td>
      <td>Bác sĩ tuyển Việt Nam: 'Không điều trị sai cho...</td>
      <td>Hoàng Xuân Vinh, Ánh Viên được vinh danh tại C...</td>
      <td>HLV Hữu Thắng: 'Bác sĩ làm lỡ hết kế hoạch của...</td>
      <td>Em trai Lê Văn Duẩn giành cú đúp ở giải đua xe...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
      <td>Arsenal, Man Utd và Chelsea tránh nhau ở FA Cup</td>
      <td>Man City nhấn chìm đội bóng của Sam Allardyce ...</td>
      <td>Cựu danh thủ Arsenal khoe vạch ra được điểm yế...</td>
      <td>Wenger bị cấm chỉ đạo bốn trận vì vụ xô trọng tài</td>
      <td>Graham Poll: 'Wenger đáng bị phạt sáu trận vì ...</td>
      <td>Wenger bị khép tội hành xử thiếu chuẩn mực</td>
      <td>Wenger đối mặt án cấm chỉ đạo dài hạn vì xô xá...</td>
      <td>Cầu thủ Hull nứt sọ sau cú va chạm với Gary Ca...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tết buồn của cựu võ sĩ Trần Kim Tuyến</td>
      <td>Cầu thủ Leicester City: 'Ranieri đã phản bội tôi'</td>
      <td>Payet hoàn tất vụ chuyển nhượng về Marseille</td>
      <td>Mourinho công khai tên cầu thủ duy nhất có thể...</td>
      <td>Juventus thắng nhẹ, Roma và Milan thua ngược t...</td>
      <td>Bolt: 'Sự vĩ đại của tôi không thể bị hoen ố k...</td>
      <td>Man City sắp nhận án phạt do vi phạm luật chốn...</td>
      <td>Barca chạm trán Atletico ở bán kết Cup Nhà vua</td>
      <td>Mourinho: 'Man Utd không thua, trận đấu có tỷ ...</td>
      <td>Usain Bolt mất kỷ lục siêu hattrick vì đồng độ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Suarez: 'Bóng đã qua vạch vôi cả mét'</td>
      <td>Real nới rộng cách biệt tại Liga bằng chiến th...</td>
      <td>Suarez giải cứu Barca bằng bàn thắng ở phút 90</td>
      <td>Tháng 2 đầy chông gai đang chờ đón Barca</td>
      <td>Alba: 'Real bị loại và lập tức hạ thấp Cup Nhà...</td>
      <td>Mất Modric và Marcelo, Real lâm vào khủng hoản...</td>
      <td>Zidane họp khẩn với cả đội Real sau trận thắng...</td>
      <td>MSN cùng lập công, Barca đại thắng</td>
      <td>Real đã vô địch La Liga nếu tính từ thời điểm ...</td>
      <td>Cựu danh thủ Barca và Real thay thế Riedl, dẫn...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ferguson: ‘Kỷ lục ghi bàn của Rooney có thể kh...</td>
      <td>Rooney: 'Tôi chưa từng nghĩ đến kỷ lục ghi bàn...</td>
      <td>Rivaldo tin Gabriel Jesus sẽ đi vào lịch sử Ma...</td>
      <td>Man City đón tân binh trị giá 33,3 triệu đôla</td>
      <td>Louis van Gaal nghỉ hưu</td>
      <td>Jorge Sampaoli: Kẻ cắt mạch bất bại của Real v...</td>
      <td>Juventus giữ chặt sao tiền đạo Dybala bằng mác...</td>
      <td>Lá thư cho bản thân của Ronaldinho: 'Đừng đá b...</td>
      <td>Suarez không dự lễ trao giải FIFA vì còn bực v...</td>
      <td>Cầu thủ già nhất Nhật Bản tiếp tục thi đấu ở t...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Neymar thất bại khi lôi kéo Coutinho về Barca</td>
      <td>Scolari: 'Ronaldo giành Quả Bóng Vàng nhờ ý ch...</td>
      <td>Cầu thủ Sociedad tố trọng tài bỏ qua thẻ đỏ ch...</td>
      <td>Thắng đối thủ kỵ giơ, Barca đặt một chân vào b...</td>
      <td>Neymar qua mặt Messi, dẫn đầu thế giới về giá ...</td>
      <td>Oscar để ngỏ khả năng quay lại Chelsea dù đã t...</td>
      <td>Barca sa thải vị giám đốc phát biểu về Messi</td>
      <td>HLV Enrique: 'Cầu thủ Barca tự quyết định việc...</td>
      <td>Real áp đảo trong đội hình tiêu biểu FIFA FIFP...</td>
      <td>Barca viết lại lịch sử khi dùng đội hình chỉ c...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mesut Ozil phát cuồng vì Federer vô địch Austr...</td>
      <td>Liverpool bị đội hạng dưới đá bay khỏi Cup FA</td>
      <td>Valencia từng nghĩ sự quan tâm của Man Utd là ...</td>
      <td>Federer từng không tin còn khả năng vào chung ...</td>
      <td>Harry Kane: 'Chỉ có điên mới rời Tottenham lúc...</td>
      <td>Federer bất ngờ với chính mình khi vào bán kết...</td>
      <td>Serena vào tứ kết Australia Mở rộng</td>
      <td>Chapecoense chơi trận đầu tiên sau tai nạn máy...</td>
      <td>Serena thẳng tiến vào vòng ba Australia Mở rộng</td>
      <td>Djokovic thua tay vợt thứ 117 thế giới, dừng b...</td>
    </tr>
  </tbody>
</table>
</div>



Căn cứ vào kết quả, ta thấy nhóm 3 vẫn là nhóm hỗn tạp (nhưng với size nhỏ hơn, có thể kiểm tra). Nhóm 1, 2, 4 nói về thể thao Việt Nam, bóng đá Anh và Tây Ban Nha. Một số nhóm mới xuất hiện, căn cứ vào toạ độ tâm ta có thể gọi tên *"các bài báo mô tả một trận đấu"* (nhóm 0), *"World Cup và các đội tuyển quốc gia"* (nhóm 5), *"Neymar và các chuyện liên quan"* (nhóm 6, nhớ rằng thương vụ chuyển nhượng Neymar từ Barcelona sang PSG tốn nhiều giấy mực của báo chí năm 2017), *"danh hiệu"* (nhóm 7). Tuy nhiên, ta có thể không thực sự hài lòng với sự liên quan của nội dung các bài báo, ví dụ ở nhóm 5, nhiều bài không liên quan đến World Cup. Có thể World Cup chỉ xuất hiện 1 lần trong bài báo trong văn cảnh ngẫu nhiên.

Tuỳ thuộc mục đích sử dụng kết quả, ta có thể hài lòng với $k=4$ (3 chủ đề chính), $k=8$ (cộng thêm một số chủ đề mới).

Cuối cùng về size của các cluster, ta thấy cluster hỗn tạp có size lớn nhất. Ta có thể tách riêng nó ra và thực hiện một lần clustering mới để hoặc thêm vào các cluster cũ, hoặc phát hiện ra các cluster có chủ đề mới với size nhỏ hơn.


```python
[len(cluster) for cluster in clusters]
```




    [697, 671, 659, 1888, 673, 417, 401, 543]



### Bài 16. Thay đổi cách tính khoảng cách

Cuối bài 15, ta đã thấy một hiện tượng phát sinh là việc không để ý đến tần số của các từ trong bài báo có thể khiến một số từ kém quan trọng (như "World Cup") trở nên quan trọng. Đây là vấn đề phát sinh từ việc số hoá 0-1.

Trước đó ta cũng thấy việc số hoá mỗi bài báo thành một vector chứa tần số từ là không hợp lí.

Ta tìm cách dung hoà hai phương pháp này bằng cách xây dựng một hàm `f` biến tần số $a$ mỗi từ thành thành phần toạ độ tương ứng với từ đó, sao cho:

Ứng với 3 bài báo `A`, `B`, `C` có tần số `a`, `b`, `c`.

- Nếu `a = 0 thì f(a) = 0`

- Nếu `a = 0`, `b, c > 0` thì `|f(b) - f(c)| < |f(b) - f(a)|` (Hai bài báo cùng chứa 1 từ có thành phần khoảng cách tương ứng nhỏ hơn 1 bài báo chứa và 1 báo không chứa)

- Nếu `a > b` thì  `f(a) > f(b)` (Tần số càng lớn, toạ độ càng lớn)

Theo cách này, ta có thể cho chẳng hạn $f(1) = 2$, đặt một chặn trên tại 4 và xây dựng 1 hàm tăng nhận giá trị trong [2, 4]. Điều kiện thứ 3 cho thấy tần số của 1 từ càng lớn thì độ liên quan đến từ đó càng lớn

Có thể giả sử rằng 1 từ không xuất hiện quá 8 lần trong bài báo, ta có thể chọn hàm $f(a) = 1+ \sqrt{a}$. (Có nhiều cách lựa chọn khác, ví dụ $f(a) = \log_2{(3 + a)}$)

*Hãy sửa chữa hàm **`freqToCoordinates`** đã viết ở bài 10, để khi đối số **`coordinates_coding_mode`** nhận giá trị **"sqrt"** thì output của hàm này sẽ là một ma trận mà mỗi hàng là các vector có thành phần toạ độ được tính theo hàm $f(a) = 1 + \sqrt{a}$.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
COORDINATES_CODING_MODE = "sqrt"
NB_CLUSTER = 8
MODEL = "KMeans"
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(hiệp, 2.3699809451)</td>
      <td>(đánh, 0.965998961995)</td>
      <td>(phạt, 0.594122550533)</td>
      <td>(trọng tài, 0.572241879888)</td>
      <td>(quyền, 0.469221284562)</td>
      <td>(thì, 0.355790974737)</td>
      <td>(Man City, 0.353154465299)</td>
      <td>(bán, 0.300041101448)</td>
      <td>(trẻ, 0.281914660519)</td>
      <td>(Arsenal, 0.279069161658)</td>
      <td>...</td>
      <td>(Brazil, 0.105085875555)</td>
      <td>(Monaco, 0.0827137797314)</td>
      <td>(tuyển, 0.0813766216166)</td>
      <td>(hợp đồng, 0.0764822190314)</td>
      <td>(Neymar, 0.0706447686946)</td>
      <td>(PSG, 0.057830353843)</td>
      <td>(Zidane, 0.0507760721064)</td>
      <td>(nữ, 0.0244265850691)</td>
      <td>(HC, 0.0185185185185)</td>
      <td>(vợt, 0.00617283950617)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Chelsea, 2.3254744897)</td>
      <td>(Arsenal, 1.33963831412)</td>
      <td>(Man City, 1.10799496006)</td>
      <td>(Liverpool, 0.826120098671)</td>
      <td>(Tottenham, 0.782647220009)</td>
      <td>(hợp đồng, 0.566800947501)</td>
      <td>(Mourinho, 0.523850083375)</td>
      <td>(danh hiệu, 0.461419676369)</td>
      <td>(Tây Ban Nha, 0.431511083967)</td>
      <td>(cậu, 0.428805646929)</td>
      <td>...</td>
      <td>(giây, 0.0450561799803)</td>
      <td>(vàng, 0.0435405617717)</td>
      <td>(Messi, 0.0315424739294)</td>
      <td>(Neymar, 0.0263678296862)</td>
      <td>(Zidane, 0.0201000248228)</td>
      <td>(Ronaldo, 0.0164709583539)</td>
      <td>(nữ, 0.0108851404429)</td>
      <td>(Việt Nam, 0.00905562742561)</td>
      <td>(HC, 0.00776196636481)</td>
      <td>(vợt, 0.00517464424321)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Neymar, 2.7959805111)</td>
      <td>(PSG, 2.26531783988)</td>
      <td>(Brazil, 1.36841644585)</td>
      <td>(hợp đồng, 0.94259267976)</td>
      <td>(tiền, 0.70047877398)</td>
      <td>(Messi, 0.696399332293)</td>
      <td>(La Liga, 0.566831809717)</td>
      <td>(cậu, 0.526841004919)</td>
      <td>(Tây Ban Nha, 0.494922332565)</td>
      <td>(quyền, 0.469114428461)</td>
      <td>...</td>
      <td>(Mỹ, 0.0862648344813)</td>
      <td>(Mourinho, 0.0656213569577)</td>
      <td>(Arsenal, 0.0640849164636)</td>
      <td>(Atletico, 0.0592332335292)</td>
      <td>(giây, 0.0426780615616)</td>
      <td>(Zidane, 0.0340231494956)</td>
      <td>(nữ, 0.0235692256649)</td>
      <td>(HC, 0.0135250059517)</td>
      <td>(vợt, 0.0112044817927)</td>
      <td>(Việt Nam, 0.00560224089636)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Việt Nam, 2.60567170468)</td>
      <td>(tuyển, 0.72572090513)</td>
      <td>(HC, 0.72394296146)</td>
      <td>(trẻ, 0.707170880204)</td>
      <td>(vàng, 0.669229568357)</td>
      <td>(nữ, 0.612548406283)</td>
      <td>(thì, 0.535601432896)</td>
      <td>(mình, 0.49457397255)</td>
      <td>(quyền, 0.471726823682)</td>
      <td>(đánh, 0.401087097383)</td>
      <td>...</td>
      <td>(Tottenham, 0.00415205289904)</td>
      <td>(Neymar, 0.003669017572)</td>
      <td>(Arsenal, 0.003669017572)</td>
      <td>(PSG, 0.00303951367781)</td>
      <td>(Juventus, 0.00303951367781)</td>
      <td>(Chelsea, 0.00303951367781)</td>
      <td>(Monaco, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Zidane, 0.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Ronaldo, 2.7546898077)</td>
      <td>(La Liga, 1.20032932048)</td>
      <td>(Messi, 0.60278211361)</td>
      <td>(Tây Ban Nha, 0.599879871247)</td>
      <td>(danh hiệu, 0.542351509961)</td>
      <td>(Zidane, 0.47254364085)</td>
      <td>(cậu, 0.422621268577)</td>
      <td>(bán, 0.373362553631)</td>
      <td>(quyền, 0.28812472287)</td>
      <td>(hiệp, 0.287360556852)</td>
      <td>...</td>
      <td>(Liverpool, 0.0727378906835)</td>
      <td>(Chelsea, 0.0717287041501)</td>
      <td>(Monaco, 0.0523566176274)</td>
      <td>(Mourinho, 0.0485076158328)</td>
      <td>(Arsenal, 0.0409204745831)</td>
      <td>(nữ, 0.0364760301386)</td>
      <td>(vợt, 0.0231426968053)</td>
      <td>(HC, 0.0)</td>
      <td>(giây, 0.0)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(hợp đồng, 0.384774356762)</td>
      <td>(danh hiệu, 0.323396418294)</td>
      <td>(quyền, 0.32289515296)</td>
      <td>(trẻ, 0.311180411699)</td>
      <td>(bán, 0.302445990177)</td>
      <td>(tiền, 0.295843613289)</td>
      <td>(La Liga, 0.293283426008)</td>
      <td>(đua, 0.273566142067)</td>
      <td>(Tây Ban Nha, 0.261776937219)</td>
      <td>(Mỹ, 0.255595264007)</td>
      <td>...</td>
      <td>(Argentina, 0.0630199637413)</td>
      <td>(Messi, 0.0586264937297)</td>
      <td>(Chelsea, 0.057005850599)</td>
      <td>(nữ, 0.0510986478456)</td>
      <td>(HC, 0.0405613706718)</td>
      <td>(Neymar, 0.0385774158723)</td>
      <td>(Việt Nam, 0.00174291938998)</td>
      <td>(Ronaldo, 0.000871459694989)</td>
      <td>(hiệp, 0.0)</td>
      <td>(vợt, 0.0)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(World Cup, 1.61036471516)</td>
      <td>(Argentina, 1.37537471749)</td>
      <td>(Messi, 1.36625766004)</td>
      <td>(tuyển, 1.11061237527)</td>
      <td>(Tây Ban Nha, 0.83605949656)</td>
      <td>(La Liga, 0.686189015107)</td>
      <td>(Brazil, 0.595607724501)</td>
      <td>(cậu, 0.592783489406)</td>
      <td>(hợp đồng, 0.419230604188)</td>
      <td>(danh hiệu, 0.407010559534)</td>
      <td>...</td>
      <td>(Liverpool, 0.0796168264202)</td>
      <td>(Bayern, 0.0771894807401)</td>
      <td>(Zidane, 0.0614958441828)</td>
      <td>(Tottenham, 0.0606733976919)</td>
      <td>(HC, 0.0425020928324)</td>
      <td>(nữ, 0.0425020928324)</td>
      <td>(Mourinho, 0.0388174998255)</td>
      <td>(giây, 0.0280230554455)</td>
      <td>(Việt Nam, 0.0189937100731)</td>
      <td>(vợt, 0.00451467268623)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(vợt, 3.14842856041)</td>
      <td>(đánh, 0.840580284289)</td>
      <td>(danh hiệu, 0.826462301495)</td>
      <td>(bán, 0.660969985692)</td>
      <td>(Tây Ban Nha, 0.518831124094)</td>
      <td>(trẻ, 0.461116172362)</td>
      <td>(Mỹ, 0.362508244793)</td>
      <td>(nữ, 0.359942347988)</td>
      <td>(thì, 0.35614131969)</td>
      <td>(cậu, 0.299236552013)</td>
      <td>...</td>
      <td>(Liverpool, 0.00615384615385)</td>
      <td>(Atletico, 0.00615384615385)</td>
      <td>(Zidane, 0.00615384615385)</td>
      <td>(Neymar, 0.0)</td>
      <td>(PSG, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Tottenham, 0.0)</td>
      <td>(Mourinho, 0.0)</td>
      <td>(Juventus, 0.0)</td>
      <td>(Man City, 0.0)</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 43 columns</p>
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
      <td>Man City nhấn chìm đội bóng của Sam Allardyce ...</td>
      <td>Trút mưa bàn thắng, Barca vào bán kết Cup Nhà vua</td>
      <td>Liverpool thất thủ, bị loại ở bán kết Cúp Liên...</td>
      <td>Ronaldo sút phạt ghi bàn, nhưng Real bị loại k...</td>
      <td>Sao tiền vệ PSG nhận thẻ vàng kỳ lạ</td>
      <td>Chapecoense chơi trận đầu tiên sau tai nạn máy...</td>
      <td>Klopp thừa nhận Liverpool đáng thua tại Anfield</td>
      <td>HAGL thua trận thứ ba liên tiếp tại V-League 2017</td>
      <td>HLV từng vô địch C1 châu Âu thắng trận thứ ba ...</td>
      <td>Sao Man City mất 50.000 đôla vì một dòng viết ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
      <td>Arsenal, Man Utd và Chelsea tránh nhau ở FA Cup</td>
      <td>Liverpool bị đội hạng dưới đá bay khỏi Cup FA</td>
      <td>Cựu danh thủ Arsenal khoe vạch ra được điểm yế...</td>
      <td>Wenger bị cấm chỉ đạo bốn trận vì vụ xô trọng tài</td>
      <td>Harry Kane: 'Chỉ có điên mới rời Tottenham lúc...</td>
      <td>Sir Alex: 'Van Gaal làm tốt, còn Mourinho thật...</td>
      <td>Graham Poll: 'Wenger đáng bị phạt sáu trận vì ...</td>
      <td>Wenger bị khép tội hành xử thiếu chuẩn mực</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Neymar thất bại khi lôi kéo Coutinho về Barca</td>
      <td>MSN cùng lập công, Barca đại thắng</td>
      <td>Messi nhường đồng đội đá 15 quả phạt đền của B...</td>
      <td>Thắng đối thủ kỵ giơ, Barca đặt một chân vào b...</td>
      <td>Neymar, Suarez đua nhau chọc tức Pique</td>
      <td>Bộ ba MSN đạt mốc 300 bàn chỉ trong 26 tháng</td>
      <td>Cựu thần đồng Real sẵn sàng giảm lương, để thá...</td>
      <td>Qua mặt Neymar, Coutinho giành giải Cầu thủ Br...</td>
      <td>Barca viết lại lịch sử khi dùng đội hình chỉ c...</td>
      <td>Đá phạt đền ở phút 90, Messi cứu Barca khỏi ng...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tiến Minh và Vũ Thị Trang lần đầu đánh giải sa...</td>
      <td>Bolt: 'Sự vĩ đại của tôi không thể bị hoen ố k...</td>
      <td>Usain Bolt mất kỷ lục siêu hattrick vì đồng độ...</td>
      <td>Việt Nam cùng bảng với Campuchia ở vòng loại c...</td>
      <td>Ban trọng tài VFF: ‘Samson chỉ trượt lên đầu g...</td>
      <td>HAGL bất bình với pha ra đòn của cầu thủ Hà Nộ...</td>
      <td>'Nữ hoàng đấu kiếm' Lệ Dung liên tục bị thất h...</td>
      <td>Tài năng bóng đá Việt Nam có cơ hội tập luyện ...</td>
      <td>Bác sĩ tuyển Việt Nam: 'Không điều trị sai cho...</td>
      <td>Hoàng Xuân Vinh, Ánh Viên được vinh danh tại C...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Real nới rộng cách biệt tại Liga bằng chiến th...</td>
      <td>Scolari: 'Ronaldo giành Quả Bóng Vàng nhờ ý ch...</td>
      <td>Real thua trận thứ hai liên tiếp</td>
      <td>Ronaldo tái xuất tại Cup Nhà vua sau hai năm v...</td>
      <td>Neymar qua mặt Messi, dẫn đầu thế giới về giá ...</td>
      <td>Ronaldo gây phẫn nộ khi ném bóng vào cầu thủ S...</td>
      <td>Ronaldo san bằng kỷ lục đá phạt đền tại La Liga</td>
      <td>Ramos phản lưới, Real nhận thất bại đầu tiên s...</td>
      <td>HLV Simeone đá xoáy Ronaldo về Quả Bóng Vàng</td>
      <td>Ramos sút phạt đền kiểu Panenka, Real phá kỷ l...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
      <td>Tết buồn của cựu võ sĩ Trần Kim Tuyến</td>
      <td>Cầu thủ Leicester City: 'Ranieri đã phản bội tôi'</td>
      <td>Suarez: 'Bóng đã qua vạch vôi cả mét'</td>
      <td>Payet hoàn tất vụ chuyển nhượng về Marseille</td>
      <td>Mourinho công khai tên cầu thủ duy nhất có thể...</td>
      <td>Juventus thắng nhẹ, Roma và Milan thua ngược t...</td>
      <td>Suarez giải cứu Barca bằng bàn thắng ở phút 90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ferguson: ‘Kỷ lục ghi bàn của Rooney có thể kh...</td>
      <td>Tevez: 'Tôi không nhận mức lương cao nhất thế ...</td>
      <td>Rivaldo tin Gabriel Jesus sẽ đi vào lịch sử Ma...</td>
      <td>Man City đón tân binh trị giá 33,3 triệu đôla</td>
      <td>Louis van Gaal nghỉ hưu</td>
      <td>Jorge Sampaoli: Kẻ cắt mạch bất bại của Real v...</td>
      <td>Juventus giữ chặt sao tiền đạo Dybala bằng mác...</td>
      <td>Messi quân bình kỷ lục của Raul Gonzalez</td>
      <td>Barca sa thải vị giám đốc phát biểu về Messi</td>
      <td>Lá thư cho bản thân của Ronaldinho: 'Đừng đá b...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mesut Ozil phát cuồng vì Federer vô địch Austr...</td>
      <td>Federer từng không tin còn khả năng vào chung ...</td>
      <td>Federer bất ngờ với chính mình khi vào bán kết...</td>
      <td>Serena vào tứ kết Australia Mở rộng</td>
      <td>ĐKVĐ Kerber bị loại, Venus vào tứ kết Australi...</td>
      <td>Murray bị loại ở vòng bốn Australia Mở rộng</td>
      <td>Bouchard dừng bước tại vòng ba Australia Mở rộng</td>
      <td>Djokovic tâm phục, khẩu phục khi thua đối thủ ...</td>
      <td>Nadal vào vòng ba Australia Mở rộng</td>
      <td>Serena thẳng tiến vào vòng ba Australia Mở rộng</td>
    </tr>
  </tbody>
</table>
</div>



Kiểm tra lại rằng có điểm khác biệt: các cluster là: *mô tả trận đấu, bóng đá Anh, Neymar và chuyện liên quan, thể thao Việt Nam, bóng đá Tay Ban Nha, cluster hỗn tạp, World Cup và các đội tuyển quốc gia* (nhưng toạ độ của "World Cup" bằng 1.61 nhỏ hơn hẳn so với các từ quan trọng ở các cluster khác), *quần vợt* (toạ độ của "vợt" bằng 3.14 lớn hơn hẳn so với các từ quan trọng ở các cluster khác, lí do nó xuất hiện trong thuật toán này).

Vì vậy, nếu giảm còn 7 cluster, cluster về World Cup sẽ biến mất (tách và tan trong các cluster khác), như quan sát dưới đây.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
COORDINATES_CODING_MODE = "sqrt"
NB_CLUSTER = 7
MODEL = "KMeans"
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(vợt, 3.14842856041)</td>
      <td>(đánh, 0.840580284289)</td>
      <td>(danh hiệu, 0.826462301495)</td>
      <td>(bán, 0.660969985692)</td>
      <td>(Tây Ban Nha, 0.518831124094)</td>
      <td>(trẻ, 0.461116172362)</td>
      <td>(Mỹ, 0.362508244793)</td>
      <td>(nữ, 0.359942347988)</td>
      <td>(thì, 0.35614131969)</td>
      <td>(cậu, 0.299236552013)</td>
      <td>...</td>
      <td>(Liverpool, 0.00615384615385)</td>
      <td>(Atletico, 0.00615384615385)</td>
      <td>(Zidane, 0.00615384615385)</td>
      <td>(Neymar, 0.0)</td>
      <td>(PSG, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Tottenham, 0.0)</td>
      <td>(Mourinho, 0.0)</td>
      <td>(Juventus, 0.0)</td>
      <td>(Man City, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(quyền, 0.374476700099)</td>
      <td>(hiệp, 0.364047290606)</td>
      <td>(đánh, 0.349193648869)</td>
      <td>(hợp đồng, 0.32936044584)</td>
      <td>(trẻ, 0.329137597244)</td>
      <td>(danh hiệu, 0.300820357162)</td>
      <td>(bán, 0.293070749954)</td>
      <td>(Tây Ban Nha, 0.276467636627)</td>
      <td>(tiền, 0.265427335951)</td>
      <td>(mình, 0.262890608196)</td>
      <td>...</td>
      <td>(vàng, 0.0935656113865)</td>
      <td>(Tottenham, 0.0779046671629)</td>
      <td>(Chelsea, 0.0679332869413)</td>
      <td>(nữ, 0.0538972796894)</td>
      <td>(HC, 0.0422266464506)</td>
      <td>(Neymar, 0.0279347614935)</td>
      <td>(Messi, 0.0128571428571)</td>
      <td>(vợt, 0.00142857142857)</td>
      <td>(Việt Nam, 0.000714285714286)</td>
      <td>(Ronaldo, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Việt Nam, 2.58231875235)</td>
      <td>(tuyển, 0.678563152679)</td>
      <td>(HC, 0.66802042186)</td>
      <td>(trẻ, 0.663927270165)</td>
      <td>(vàng, 0.619859780696)</td>
      <td>(nữ, 0.563842378554)</td>
      <td>(thì, 0.513350910546)</td>
      <td>(mình, 0.471850493718)</td>
      <td>(quyền, 0.463445886533)</td>
      <td>(đánh, 0.408813105354)</td>
      <td>...</td>
      <td>(Tottenham, 0.00386976035066)</td>
      <td>(Neymar, 0.00341956595237)</td>
      <td>(Arsenal, 0.00341956595237)</td>
      <td>(PSG, 0.0028328611898)</td>
      <td>(Juventus, 0.0028328611898)</td>
      <td>(Chelsea, 0.0028328611898)</td>
      <td>(Monaco, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Zidane, 0.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Neymar, 2.7745002913)</td>
      <td>(PSG, 2.30343842231)</td>
      <td>(Brazil, 1.44707363673)</td>
      <td>(hợp đồng, 0.981576165549)</td>
      <td>(tiền, 0.716535594014)</td>
      <td>(Messi, 0.558476280565)</td>
      <td>(cậu, 0.529902505404)</td>
      <td>(Tây Ban Nha, 0.500536598068)</td>
      <td>(La Liga, 0.490395978787)</td>
      <td>(quyền, 0.479867767795)</td>
      <td>...</td>
      <td>(Tottenham, 0.0916180272054)</td>
      <td>(Mourinho, 0.0740430888145)</td>
      <td>(Arsenal, 0.0724714290541)</td>
      <td>(Atletico, 0.0605910153867)</td>
      <td>(giây, 0.0436563552364)</td>
      <td>(Zidane, 0.0348030497706)</td>
      <td>(nữ, 0.0241094944481)</td>
      <td>(HC, 0.0207525521121)</td>
      <td>(vợt, 0.0114613180516)</td>
      <td>(Việt Nam, 0.00573065902579)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Ronaldo, 2.73420193876)</td>
      <td>(La Liga, 1.11054441477)</td>
      <td>(Tây Ban Nha, 0.59203820224)</td>
      <td>(danh hiệu, 0.495138656012)</td>
      <td>(Zidane, 0.46119205394)</td>
      <td>(Messi, 0.456080894386)</td>
      <td>(hiệp, 0.435019610346)</td>
      <td>(cậu, 0.415802078928)</td>
      <td>(bán, 0.410902493172)</td>
      <td>(phạt, 0.311840626076)</td>
      <td>...</td>
      <td>(Liverpool, 0.0658073893096)</td>
      <td>(Mourinho, 0.0632986679493)</td>
      <td>(Argentina, 0.0574484520873)</td>
      <td>(Monaco, 0.0504507022105)</td>
      <td>(Arsenal, 0.0479961746518)</td>
      <td>(nữ, 0.035148208913)</td>
      <td>(vợt, 0.018017587928)</td>
      <td>(HC, 0.00428265524625)</td>
      <td>(giây, 0.0)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Chelsea, 1.97774540346)</td>
      <td>(Arsenal, 1.35832962035)</td>
      <td>(Man City, 1.19722629105)</td>
      <td>(Liverpool, 0.839233468166)</td>
      <td>(Tottenham, 0.736060358164)</td>
      <td>(hợp đồng, 0.544180444428)</td>
      <td>(Mourinho, 0.5127261347)</td>
      <td>(hiệp, 0.508883805648)</td>
      <td>(cậu, 0.465808356971)</td>
      <td>(danh hiệu, 0.425366936119)</td>
      <td>...</td>
      <td>(vàng, 0.0511448115762)</td>
      <td>(giây, 0.0430679611021)</td>
      <td>(Messi, 0.0265255953269)</td>
      <td>(Neymar, 0.024382260712)</td>
      <td>(Zidane, 0.0225033235351)</td>
      <td>(Ronaldo, 0.0134304333413)</td>
      <td>(nữ, 0.00887575270293)</td>
      <td>(HC, 0.0084388185654)</td>
      <td>(Việt Nam, 0.00738396624473)</td>
      <td>(vợt, 0.00632911392405)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Messi, 2.64054121332)</td>
      <td>(Argentina, 1.5445698982)</td>
      <td>(La Liga, 1.16840249352)</td>
      <td>(World Cup, 0.663943871653)</td>
      <td>(Tây Ban Nha, 0.661420385935)</td>
      <td>(cậu, 0.492668801105)</td>
      <td>(Neymar, 0.475834377215)</td>
      <td>(hiệp, 0.473934893814)</td>
      <td>(tuyển, 0.456991316273)</td>
      <td>(phạt, 0.428904781601)</td>
      <td>...</td>
      <td>(Arsenal, 0.0447130709739)</td>
      <td>(Tottenham, 0.0444408214903)</td>
      <td>(Chelsea, 0.0345651637782)</td>
      <td>(Mỹ, 0.0317588719975)</td>
      <td>(giây, 0.0294186823796)</td>
      <td>(nữ, 0.0169491525424)</td>
      <td>(HC, 0.0112994350282)</td>
      <td>(Mourinho, 0.0077176576485)</td>
      <td>(vợt, 0.00564971751412)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 43 columns</p>
</div>



Nếu tăng thành 9 hay 10 cluster, một số cluster mới xuất hiện làm cấu trúc clustering, thay đổi, như ở dưới đây, cluster *"đua xe"* xuất hiện (cluster 8), cluster "Bóng đá Tây Ban Nha" bị tách đôi (cluster 2 liên quan đến Messi, cluster 7 liên quan đến Ronaldo). 

Có thể giải thích: khi $k$ nhỏ, cluster "đua xe" bị hợp vào các cluster quan trọng khác có size lớn hơn. Nó chỉ xuất hiện khi $k$ đủ lớn. Một hay một số lớn cũng được phân thành nhiều cluster nhỏ hơn (trường hợp của *"bóng đá Tây Ban Nha"*) nếu trong cluster này lại tồn tại một cấu trúc chia được rõ ràng hơn các cluster khác.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
COORDINATES_CODING_MODE = "sqrt"
NB_CLUSTER = 10
MODEL = "KMeans"
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(hiệp, 2.409420513)</td>
      <td>(đánh, 0.99521684314)</td>
      <td>(quyền, 0.437982479221)</td>
      <td>(phạt, 0.411793009915)</td>
      <td>(Man City, 0.391267396054)</td>
      <td>(thì, 0.330219225896)</td>
      <td>(trẻ, 0.298503566344)</td>
      <td>(bán, 0.288237555764)</td>
      <td>(Arsenal, 0.277610942978)</td>
      <td>(Liverpool, 0.27393814132)</td>
      <td>...</td>
      <td>(Monaco, 0.0890458335115)</td>
      <td>(hợp đồng, 0.0782055259108)</td>
      <td>(Atletico, 0.0720428275425)</td>
      <td>(PSG, 0.0600970723344)</td>
      <td>(Messi, 0.057256154072)</td>
      <td>(Zidane, 0.0421699841339)</td>
      <td>(Neymar, 0.0315268903632)</td>
      <td>(HC, 0.0215439856373)</td>
      <td>(nữ, 0.0151063080114)</td>
      <td>(vợt, 0.00718132854578)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(vợt, 3.15992120948)</td>
      <td>(đánh, 0.844339351231)</td>
      <td>(danh hiệu, 0.826875774956)</td>
      <td>(bán, 0.665047641719)</td>
      <td>(Tây Ban Nha, 0.526937860408)</td>
      <td>(trẻ, 0.454526695173)</td>
      <td>(Mỹ, 0.368172436118)</td>
      <td>(nữ, 0.365566447175)</td>
      <td>(thì, 0.35545602781)</td>
      <td>(cậu, 0.291412123139)</td>
      <td>...</td>
      <td>(Liverpool, 0.00625)</td>
      <td>(Atletico, 0.00625)</td>
      <td>(Neymar, 0.0)</td>
      <td>(PSG, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Tottenham, 0.0)</td>
      <td>(Mourinho, 0.0)</td>
      <td>(Juventus, 0.0)</td>
      <td>(Man City, 0.0)</td>
      <td>(Zidane, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(La Liga, 1.81813244438)</td>
      <td>(Messi, 1.21082394442)</td>
      <td>(Tây Ban Nha, 0.848009826605)</td>
      <td>(Argentina, 0.812534419082)</td>
      <td>(danh hiệu, 0.519669520709)</td>
      <td>(World Cup, 0.445407362728)</td>
      <td>(Atletico, 0.441550906916)</td>
      <td>(hợp đồng, 0.362662828268)</td>
      <td>(Juventus, 0.313522750241)</td>
      <td>(tuyển, 0.311258434864)</td>
      <td>...</td>
      <td>(Arsenal, 0.0578820524305)</td>
      <td>(Monaco, 0.0560711003317)</td>
      <td>(Mourinho, 0.0506172567334)</td>
      <td>(Ronaldo, 0.0439329696053)</td>
      <td>(Mỹ, 0.040508312071)</td>
      <td>(nữ, 0.0178325574698)</td>
      <td>(giây, 0.0178325574698)</td>
      <td>(HC, 0.00684931506849)</td>
      <td>(Việt Nam, 0.00467816919104)</td>
      <td>(vợt, 0.00342465753425)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(hợp đồng, 0.430648915155)</td>
      <td>(trẻ, 0.365737095804)</td>
      <td>(quyền, 0.329915621427)</td>
      <td>(tiền, 0.32274069765)</td>
      <td>(danh hiệu, 0.317410153795)</td>
      <td>(bán, 0.314813127746)</td>
      <td>(cậu, 0.305749922905)</td>
      <td>(Mỹ, 0.305528886182)</td>
      <td>(Juventus, 0.272452144308)</td>
      <td>(tuyển, 0.270991496424)</td>
      <td>...</td>
      <td>(giây, 0.0466752852228)</td>
      <td>(HC, 0.0452376961532)</td>
      <td>(Messi, 0.0328957731427)</td>
      <td>(Neymar, 0.0253305764851)</td>
      <td>(La Liga, 0.00617283950617)</td>
      <td>(trọng tài, 0.00514403292181)</td>
      <td>(hiệp, 0.00205761316872)</td>
      <td>(Việt Nam, 0.00102880658436)</td>
      <td>(Ronaldo, 0.00102880658436)</td>
      <td>(vợt, 0.00102880658436)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Neymar, 2.8066071635)</td>
      <td>(PSG, 2.34770865928)</td>
      <td>(Brazil, 1.50495627378)</td>
      <td>(hợp đồng, 0.971658467685)</td>
      <td>(tiền, 0.75245051436)</td>
      <td>(Messi, 0.630527736821)</td>
      <td>(cậu, 0.558454444048)</td>
      <td>(Tây Ban Nha, 0.475005357593)</td>
      <td>(quyền, 0.470583354734)</td>
      <td>(La Liga, 0.423013144737)</td>
      <td>...</td>
      <td>(Mourinho, 0.0699308192056)</td>
      <td>(Atletico, 0.0690933264774)</td>
      <td>(Arsenal, 0.0623233288881)</td>
      <td>(trọng tài, 0.0621744093276)</td>
      <td>(giây, 0.0454807999328)</td>
      <td>(Zidane, 0.0362575055819)</td>
      <td>(HC, 0.0275899722004)</td>
      <td>(nữ, 0.0251170554101)</td>
      <td>(vợt, 0.0119402985075)</td>
      <td>(Việt Nam, 0.00597014925373)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(trọng tài, 2.74226876271)</td>
      <td>(phạt, 1.55934679105)</td>
      <td>(quyền, 0.641471671452)</td>
      <td>(hiệp, 0.569293979069)</td>
      <td>(thì, 0.525214363719)</td>
      <td>(đánh, 0.484500376891)</td>
      <td>(mình, 0.339283640285)</td>
      <td>(La Liga, 0.318033452042)</td>
      <td>(Messi, 0.302479777122)</td>
      <td>(Việt Nam, 0.290587608424)</td>
      <td>...</td>
      <td>(Liverpool, 0.0854675599875)</td>
      <td>(nữ, 0.0703454373909)</td>
      <td>(Bayern, 0.0685840601176)</td>
      <td>(danh hiệu, 0.063994399882)</td>
      <td>(Tottenham, 0.0532915360502)</td>
      <td>(Mỹ, 0.0339449126168)</td>
      <td>(Zidane, 0.0211036075472)</td>
      <td>(HC, 0.0188087774295)</td>
      <td>(vợt, 0.0101444137226)</td>
      <td>(Monaco, 0.00856442259426)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Việt Nam, 2.68013550671)</td>
      <td>(tuyển, 0.767752287049)</td>
      <td>(trẻ, 0.733561497707)</td>
      <td>(HC, 0.733482349645)</td>
      <td>(vàng, 0.654045866441)</td>
      <td>(nữ, 0.624700184828)</td>
      <td>(thì, 0.519616706432)</td>
      <td>(mình, 0.489982465683)</td>
      <td>(quyền, 0.472976473329)</td>
      <td>(đánh, 0.38534761296)</td>
      <td>...</td>
      <td>(Liverpool, 0.00333333333333)</td>
      <td>(Juventus, 0.00333333333333)</td>
      <td>(Chelsea, 0.00333333333333)</td>
      <td>(Monaco, 0.0)</td>
      <td>(PSG, 0.0)</td>
      <td>(La Liga, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Zidane, 0.0)</td>
      <td>(Arsenal, 0.0)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Ronaldo, 2.76593069354)</td>
      <td>(La Liga, 1.13474356454)</td>
      <td>(Messi, 0.622955991034)</td>
      <td>(Tây Ban Nha, 0.579957928925)</td>
      <td>(danh hiệu, 0.561748815784)</td>
      <td>(Zidane, 0.484742696839)</td>
      <td>(cậu, 0.481155590131)</td>
      <td>(bán, 0.372490178632)</td>
      <td>(tuyển, 0.31590350845)</td>
      <td>(bạn, 0.285678600967)</td>
      <td>...</td>
      <td>(trọng tài, 0.0760874544069)</td>
      <td>(Liverpool, 0.0693725751864)</td>
      <td>(Monaco, 0.0631482877984)</td>
      <td>(Mourinho, 0.0547237938761)</td>
      <td>(Arsenal, 0.0515314382048)</td>
      <td>(nữ, 0.037052400818)</td>
      <td>(vợt, 0.0235083827593)</td>
      <td>(HC, 0.00451467268623)</td>
      <td>(giây, 0.0)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(đua, 4.03455800249)</td>
      <td>(giây, 1.8962701653)</td>
      <td>(vàng, 0.6791946304)</td>
      <td>(mình, 0.586819550102)</td>
      <td>(trẻ, 0.462284250298)</td>
      <td>(thì, 0.430111398509)</td>
      <td>(HC, 0.405703489339)</td>
      <td>(danh hiệu, 0.375521361485)</td>
      <td>(Mỹ, 0.323797223554)</td>
      <td>(Việt Nam, 0.318099094281)</td>
      <td>...</td>
      <td>(La Liga, 0.0)</td>
      <td>(Tottenham, 0.0)</td>
      <td>(Mourinho, 0.0)</td>
      <td>(Ronaldo, 0.0)</td>
      <td>(Juventus, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Chelsea, 0.0)</td>
      <td>(Man City, 0.0)</td>
      <td>(Zidane, 0.0)</td>
      <td>(Arsenal, 0.0)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Chelsea, 2.35924930956)</td>
      <td>(Arsenal, 1.36305064953)</td>
      <td>(Man City, 1.11404695307)</td>
      <td>(Liverpool, 0.862322574781)</td>
      <td>(Tottenham, 0.799004925532)</td>
      <td>(hợp đồng, 0.577709042857)</td>
      <td>(Mourinho, 0.528034615826)</td>
      <td>(danh hiệu, 0.463741788069)</td>
      <td>(cậu, 0.44304749753)</td>
      <td>(Tây Ban Nha, 0.430268381247)</td>
      <td>...</td>
      <td>(vàng, 0.0460422082756)</td>
      <td>(giây, 0.0449089290352)</td>
      <td>(Neymar, 0.0251468294767)</td>
      <td>(Messi, 0.0234559020109)</td>
      <td>(Zidane, 0.0152162867656)</td>
      <td>(Ronaldo, 0.0109439124487)</td>
      <td>(Việt Nam, 0.00957592339261)</td>
      <td>(nữ, 0.00877457395673)</td>
      <td>(HC, 0.00820793433653)</td>
      <td>(vợt, 0.00547195622435)</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 43 columns</p>
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
      <td>Man City nhấn chìm đội bóng của Sam Allardyce ...</td>
      <td>Liverpool thất thủ, bị loại ở bán kết Cúp Liên...</td>
      <td>Ronaldo sút phạt ghi bàn, nhưng Real bị loại k...</td>
      <td>Chapecoense chơi trận đầu tiên sau tai nạn máy...</td>
      <td>Klopp thừa nhận Liverpool đáng thua tại Anfield</td>
      <td>HAGL thua trận thứ ba liên tiếp tại V-League 2017</td>
      <td>HLV từng vô địch C1 châu Âu thắng trận thứ ba ...</td>
      <td>Mourinho: 'Man Utd tấn công, còn Liverpool phò...</td>
      <td>Ibrahimovic tiếc khi Man Utd phải chia điểm vì...</td>
      <td>Guardiola lập kỷ lục buồn khi Man City thua 0-4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mesut Ozil phát cuồng vì Federer vô địch Austr...</td>
      <td>Federer từng không tin còn khả năng vào chung ...</td>
      <td>Federer bất ngờ với chính mình khi vào bán kết...</td>
      <td>Serena vào tứ kết Australia Mở rộng</td>
      <td>ĐKVĐ Kerber bị loại, Venus vào tứ kết Australi...</td>
      <td>Murray bị loại ở vòng bốn Australia Mở rộng</td>
      <td>Bouchard dừng bước tại vòng ba Australia Mở rộng</td>
      <td>Djokovic tâm phục, khẩu phục khi thua đối thủ ...</td>
      <td>Nadal vào vòng ba Australia Mở rộng</td>
      <td>Serena thẳng tiến vào vòng ba Australia Mở rộng</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Suarez giải cứu Barca bằng bàn thắng ở phút 90</td>
      <td>Tháng 2 đầy chông gai đang chờ đón Barca</td>
      <td>Alba: 'Real bị loại và lập tức hạ thấp Cup Nhà...</td>
      <td>Real thua trong vụ kiện kênh truyền hình bôi x...</td>
      <td>Mất Modric và Marcelo, Real lâm vào khủng hoản...</td>
      <td>Zidane họp khẩn với cả đội Real sau trận thắng...</td>
      <td>MSN cùng lập công, Barca đại thắng</td>
      <td>Real đã vô địch La Liga nếu tính từ thời điểm ...</td>
      <td>Cựu danh thủ Barca và Real thay thế Riedl, dẫn...</td>
      <td>Messi nhường đồng đội đá 15 quả phạt đền của B...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pato đầu quân cho đội bóng của Trung Quốc</td>
      <td>Beckham: 'Tôi không biết mình suýt bị Man Utd ...</td>
      <td>Tết buồn của cựu võ sĩ Trần Kim Tuyến</td>
      <td>Cầu thủ Leicester City: 'Ranieri đã phản bội tôi'</td>
      <td>Payet hoàn tất vụ chuyển nhượng về Marseille</td>
      <td>Mourinho công khai tên cầu thủ duy nhất có thể...</td>
      <td>Juventus thắng nhẹ, Roma và Milan thua ngược t...</td>
      <td>Luis van Gaal và con đường dang dở</td>
      <td>Man City sắp nhận án phạt do vi phạm luật chốn...</td>
      <td>Valencia từng nghĩ sự quan tâm của Man Utd là ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar thất bại khi lôi kéo Coutinho về Barca</td>
      <td>Man City đón tân binh trị giá 33,3 triệu đôla</td>
      <td>Thắng đối thủ kỵ giơ, Barca đặt một chân vào b...</td>
      <td>Qua mặt Neymar, Coutinho giành giải Cầu thủ Br...</td>
      <td>Barca viết lại lịch sử khi dùng đội hình chỉ c...</td>
      <td>Cầu thủ Barca bàn về sai lầm chiến thuật ngay ...</td>
      <td>CĐV Tây Ban Nha chỉ mặt những 'con sâu làm rầu...</td>
      <td>Lavezzi ăn lương khủng cả năm nhưng vẫn tịt ng...</td>
      <td>PSG - Barca: Trận cầu của dàn sao trị giá 1,3 ...</td>
      <td>Ronaldinho: 'Chẳng có gì là bất khả thi trong ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Leicester thua trận, trở thành nhà ĐKVĐ tệ nhấ...</td>
      <td>Suarez: 'Bóng đã qua vạch vôi cả mét'</td>
      <td>Bolt: 'Sự vĩ đại của tôi không thể bị hoen ố k...</td>
      <td>Wenger bị cấm chỉ đạo bốn trận vì vụ xô trọng tài</td>
      <td>Mourinho: 'Man Utd không thua, trận đấu có tỷ ...</td>
      <td>Trút mưa bàn thắng, Barca vào bán kết Cup Nhà vua</td>
      <td>Graham Poll: 'Wenger đáng bị phạt sáu trận vì ...</td>
      <td>Wenger bị khép tội hành xử thiếu chuẩn mực</td>
      <td>Ban trọng tài VFF: ‘Samson chỉ trượt lên đầu g...</td>
      <td>Sao tiền vệ PSG nhận thẻ vàng kỳ lạ</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Tiến Minh và Vũ Thị Trang lần đầu đánh giải sa...</td>
      <td>Usain Bolt mất kỷ lục siêu hattrick vì đồng độ...</td>
      <td>Việt Nam cùng bảng với Campuchia ở vòng loại c...</td>
      <td>'Nữ hoàng đấu kiếm' Lệ Dung liên tục bị thất h...</td>
      <td>Tài năng bóng đá Việt Nam có cơ hội tập luyện ...</td>
      <td>Bác sĩ tuyển Việt Nam: 'Không điều trị sai cho...</td>
      <td>Hoàng Xuân Vinh, Ánh Viên được vinh danh tại C...</td>
      <td>HLV Hữu Thắng: 'Bác sĩ làm lỡ hết kế hoạch của...</td>
      <td>Taekwondo Việt Nam giành bốn HC vàng trên đất ...</td>
      <td>Giải golf lớn nhất Việt Nam có tổng thưởng 60 ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Real nới rộng cách biệt tại Liga bằng chiến th...</td>
      <td>Scolari: 'Ronaldo giành Quả Bóng Vàng nhờ ý ch...</td>
      <td>Real thua trận thứ hai liên tiếp</td>
      <td>Ronaldo tái xuất tại Cup Nhà vua sau hai năm v...</td>
      <td>Neymar qua mặt Messi, dẫn đầu thế giới về giá ...</td>
      <td>Ronaldo san bằng kỷ lục đá phạt đền tại La Liga</td>
      <td>Ramos phản lưới, Real nhận thất bại đầu tiên s...</td>
      <td>HLV Simeone đá xoáy Ronaldo về Quả Bóng Vàng</td>
      <td>Lá thư cho bản thân của Ronaldinho: 'Đừng đá b...</td>
      <td>Ramos sút phạt đền kiểu Panenka, Real phá kỷ l...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tay lái mới của Mercedes mong đua tranh công b...</td>
      <td>Em trai Lê Văn Duẩn giành cú đúp ở giải đua xe...</td>
      <td>Maverick Vinales ra mắt Yamaha bằng chiến thắn...</td>
      <td>Lật ngược thế cờ, Vettel về nhất chặng ...</td>
      <td>Hamilton giành pole tại chặng mở màn F1 2017</td>
      <td>VĐV đua thuyền tỉnh Hải Dương chết đuối khi tập</td>
      <td>Cua-rơ Việt mất chiến thắng vì mừng quá sớm ở ...</td>
      <td>ĐKVĐ Cleverland thua sốc trong cuộc đại chi...</td>
      <td>Tám đội quốc tế dự giải xe đạp mừng ngày 8/3</td>
      <td>Cua-rơ Lào đoạt Áo Vàng chung cuộc Cup Truyền ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tottenham vượt Arsenal nhưng chưa thể thu hẹp ...</td>
      <td>HLV Tottenham: 'Bắt kịp Chelsea là điều bất kh...</td>
      <td>Arsenal, Man Utd và Chelsea tránh nhau ở FA Cup</td>
      <td>Liverpool bị đội hạng dưới đá bay khỏi Cup FA</td>
      <td>Cựu danh thủ Arsenal khoe vạch ra được điểm yế...</td>
      <td>Harry Kane: 'Chỉ có điên mới rời Tottenham lúc...</td>
      <td>Sir Alex: 'Van Gaal làm tốt, còn Mourinho thật...</td>
      <td>Cầu thủ bị nứt sọ trong trận gặp Chelsea đã tỉ...</td>
      <td>Cầu thủ Hull nứt sọ sau cú va chạm với Gary Ca...</td>
      <td>Conte: 'Đưa Costa trở lại là quyết định tốt nh...</td>
    </tr>
  </tbody>
</table>
</div>



Ta kiểm tra lại size của các cluster. Cluster hỗn tạp vẫn có size lớn. Các cluster mới xuất hiện như "quần vợt", "đua xe" có size thuộc nhóm nhỏ nhất.


```python
[len(cluster) for cluster in clusters]
```




    [557, 320, 584, 1944, 335, 319, 600, 443, 116, 731]



### Bài 17. KMeans với cosine metric

Trong PC, ta biết rằng thực hiện k-means với cosine metric tương đương với thực hiện k-means với khoảng cách Euclide, nhưng cần preprocess để co dãn mỗi vector thành vector đơn vị.

*Hãy sửa chữa hàm **`train, predict, getClusterCenters, getExplicatveFeaturesForEachCluster`** để khi đối số **`model`** trong hàm **`train`** nhận giá trị **"KMeans_Cosine"** thì thuật toán KMeans với metric cosine được thực hiện. So sánh kết quả với KMeans với khoảng cách Euclide*

Đoạn code dưới đây giúp test hàm của bạn.


```python
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "FullData/FrequencyBySportArticle.txt"
GLOBAL_FREQUENCY_FILE = "FullData/GlobalSportFrequency.txt"
COORDINATES_CODING_MODE = "sqrt"
NB_CLUSTER = 10
MODEL = "KMeans_Cosine"
MIN_AVG = 200
MAX_AVG = 1000
MIN_DEV = 0.5

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
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(La Liga, 0.479387503192)</td>
      <td>(Messi, 0.18459970148)</td>
      <td>(Tây Ban Nha, 0.158884548502)</td>
      <td>(Atletico, 0.121204892175)</td>
      <td>(Argentina, 0.0914813835899)</td>
      <td>(danh hiệu, 0.0912992233222)</td>
      <td>(Zidane, 0.0784765365384)</td>
      <td>(hợp đồng, 0.0594358426404)</td>
      <td>(đua, 0.0540370238948)</td>
      <td>(Neymar, 0.050732304198)</td>
      <td>...</td>
      <td>(Ronaldo, 0.0122896898179)</td>
      <td>(Arsenal, 0.00906195259973)</td>
      <td>(Mourinho, 0.0066862893088)</td>
      <td>(Mỹ, 0.00644721198195)</td>
      <td>(Monaco, 0.00623628022197)</td>
      <td>(giây, 0.00208296274138)</td>
      <td>(nữ, 0.0018463218014)</td>
      <td>(Việt Nam, 0.000949027629226)</td>
      <td>(vợt, 0.000766477879141)</td>
      <td>(HC, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(hiệp, 0.390231347615)</td>
      <td>(đánh, 0.222241775408)</td>
      <td>(trọng tài, 0.193534282257)</td>
      <td>(phạt, 0.193376885171)</td>
      <td>(quyền, 0.104338808544)</td>
      <td>(thì, 0.07942197592)</td>
      <td>(mình, 0.0463954377068)</td>
      <td>(trẻ, 0.038870370192)</td>
      <td>(Juventus, 0.0380101056318)</td>
      <td>(Arsenal, 0.037478866348)</td>
      <td>...</td>
      <td>(Man City, 0.0145356154564)</td>
      <td>(tuyển, 0.0140494825131)</td>
      <td>(danh hiệu, 0.0120574635079)</td>
      <td>(PSG, 0.0101464651811)</td>
      <td>(Neymar, 0.00750516117932)</td>
      <td>(Monaco, 0.00736847061758)</td>
      <td>(nữ, 0.00537027433431)</td>
      <td>(Zidane, 0.00512041224721)</td>
      <td>(HC, 0.00403821692674)</td>
      <td>(vợt, 0.000481282507128)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Ronaldo, 0.551769722873)</td>
      <td>(La Liga, 0.22537339193)</td>
      <td>(Tây Ban Nha, 0.104375399782)</td>
      <td>(Messi, 0.101638201258)</td>
      <td>(danh hiệu, 0.0953522186881)</td>
      <td>(Zidane, 0.0818693874948)</td>
      <td>(cậu, 0.0753773624292)</td>
      <td>(bán, 0.0573555030241)</td>
      <td>(hiệp, 0.0535033135737)</td>
      <td>(bạn, 0.0497090511429)</td>
      <td>...</td>
      <td>(Man City, 0.00810314995735)</td>
      <td>(vàng, 0.00763618992154)</td>
      <td>(Monaco, 0.0068328445197)</td>
      <td>(nữ, 0.00591927289783)</td>
      <td>(Mourinho, 0.00552580988348)</td>
      <td>(Arsenal, 0.00529373978016)</td>
      <td>(vợt, 0.00450130417079)</td>
      <td>(HC, 0.0)</td>
      <td>(giây, 0.0)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Chelsea, 0.551870692421)</td>
      <td>(Arsenal, 0.150796875717)</td>
      <td>(Tottenham, 0.106275428516)</td>
      <td>(Mourinho, 0.101929772114)</td>
      <td>(hợp đồng, 0.0988422432161)</td>
      <td>(Liverpool, 0.0831885734543)</td>
      <td>(danh hiệu, 0.0769631543727)</td>
      <td>(Tây Ban Nha, 0.0755676838808)</td>
      <td>(cậu, 0.0643421498918)</td>
      <td>(bạn, 0.0609720908351)</td>
      <td>...</td>
      <td>(Neymar, 0.00494637797074)</td>
      <td>(Messi, 0.00404563930265)</td>
      <td>(giây, 0.00395274883008)</td>
      <td>(Ronaldo, 0.00374004452802)</td>
      <td>(vàng, 0.00351639769241)</td>
      <td>(nữ, 0.00304400539681)</td>
      <td>(vợt, 0.00269834334465)</td>
      <td>(Zidane, 0.00215055142682)</td>
      <td>(HC, 0.00140324969223)</td>
      <td>(Việt Nam, 0.000477328374345)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(World Cup, 0.38754826347)</td>
      <td>(tuyển, 0.294787885065)</td>
      <td>(Argentina, 0.159586532135)</td>
      <td>(Brazil, 0.101232734057)</td>
      <td>(Tây Ban Nha, 0.0950425703704)</td>
      <td>(Messi, 0.0782092562606)</td>
      <td>(trẻ, 0.0669982936821)</td>
      <td>(bạn, 0.063457712468)</td>
      <td>(hợp đồng, 0.0530387942105)</td>
      <td>(quyền, 0.0505576386252)</td>
      <td>...</td>
      <td>(PSG, 0.0148086800364)</td>
      <td>(Man City, 0.0118381444824)</td>
      <td>(Zidane, 0.00938167965744)</td>
      <td>(đua, 0.00839571904069)</td>
      <td>(HC, 0.0077288787919)</td>
      <td>(Mourinho, 0.00771526159536)</td>
      <td>(Việt Nam, 0.00760407446352)</td>
      <td>(Atletico, 0.00373013390599)</td>
      <td>(giây, 0.00248407873272)</td>
      <td>(vợt, 0.00106485383797)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(vợt, 0.67112531805)</td>
      <td>(đánh, 0.168707461122)</td>
      <td>(danh hiệu, 0.158870191794)</td>
      <td>(bán, 0.124744353087)</td>
      <td>(Tây Ban Nha, 0.104347434487)</td>
      <td>(trẻ, 0.0823038594984)</td>
      <td>(thì, 0.0621639327821)</td>
      <td>(nữ, 0.0620731403033)</td>
      <td>(Mỹ, 0.0604305656362)</td>
      <td>(cậu, 0.0536769249507)</td>
      <td>...</td>
      <td>(Messi, 0.00135145866041)</td>
      <td>(Ronaldo, 0.00111958501226)</td>
      <td>(Neymar, 0.0)</td>
      <td>(PSG, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Tottenham, 0.0)</td>
      <td>(Mourinho, 0.0)</td>
      <td>(Juventus, 0.0)</td>
      <td>(Man City, 0.0)</td>
      <td>(Zidane, 0.0)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(hợp đồng, 0.104003065351)</td>
      <td>(trẻ, 0.0929393490632)</td>
      <td>(danh hiệu, 0.090434524437)</td>
      <td>(Mỹ, 0.0855671623033)</td>
      <td>(quyền, 0.0834150793595)</td>
      <td>(tiền, 0.0814523493279)</td>
      <td>(bán, 0.0805193468589)</td>
      <td>(mình, 0.0756735706854)</td>
      <td>(cậu, 0.0743042970123)</td>
      <td>(Juventus, 0.0702955413653)</td>
      <td>...</td>
      <td>(trọng tài, 0.00708330327537)</td>
      <td>(hiệp, 0.00589627645837)</td>
      <td>(World Cup, 0.00583538807184)</td>
      <td>(Man City, 0.00538832251217)</td>
      <td>(La Liga, 0.00415215954811)</td>
      <td>(Chelsea, 0.0027684097658)</td>
      <td>(Neymar, 0.002141686953)</td>
      <td>(Ronaldo, 0.00165739267442)</td>
      <td>(Việt Nam, 0.00165504712307)</td>
      <td>(vợt, 0.000735860221911)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Việt Nam, 0.582045931535)</td>
      <td>(trẻ, 0.133043036524)</td>
      <td>(tuyển, 0.131163985364)</td>
      <td>(HC, 0.101953940855)</td>
      <td>(nữ, 0.0946068289748)</td>
      <td>(vàng, 0.0941947540446)</td>
      <td>(thì, 0.0857880372495)</td>
      <td>(quyền, 0.0823007608606)</td>
      <td>(mình, 0.0790459516183)</td>
      <td>(đánh, 0.0686188493742)</td>
      <td>...</td>
      <td>(Chelsea, 0.000517682408059)</td>
      <td>(Ronaldo, 0.000469391267238)</td>
      <td>(Monaco, 0.0)</td>
      <td>(PSG, 0.0)</td>
      <td>(Liverpool, 0.0)</td>
      <td>(La Liga, 0.0)</td>
      <td>(Bayern, 0.0)</td>
      <td>(Atletico, 0.0)</td>
      <td>(Zidane, 0.0)</td>
      <td>(Arsenal, 0.0)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(PSG, 0.391296835718)</td>
      <td>(Neymar, 0.366181753513)</td>
      <td>(Brazil, 0.188748068939)</td>
      <td>(hợp đồng, 0.136117704781)</td>
      <td>(Messi, 0.106055047753)</td>
      <td>(tiền, 0.0857090991776)</td>
      <td>(cậu, 0.0736351312139)</td>
      <td>(Tây Ban Nha, 0.0689326082933)</td>
      <td>(quyền, 0.0673967005495)</td>
      <td>(La Liga, 0.0632921501341)</td>
      <td>...</td>
      <td>(Mỹ, 0.0106929369919)</td>
      <td>(vàng, 0.0106710079383)</td>
      <td>(Arsenal, 0.00985217293274)</td>
      <td>(Mourinho, 0.00979113877431)</td>
      <td>(giây, 0.00637382475342)</td>
      <td>(Zidane, 0.00502667812309)</td>
      <td>(nữ, 0.00433164641829)</td>
      <td>(vợt, 0.00133964366668)</td>
      <td>(HC, 0.00133697522488)</td>
      <td>(Việt Nam, 0.0)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Man City, 0.456230305095)</td>
      <td>(Arsenal, 0.220553931166)</td>
      <td>(Liverpool, 0.157796209929)</td>
      <td>(Chelsea, 0.127474758696)</td>
      <td>(Tottenham, 0.121913138287)</td>
      <td>(hiệp, 0.0796328017343)</td>
      <td>(Mourinho, 0.0782184984598)</td>
      <td>(hợp đồng, 0.0692996929933)</td>
      <td>(cậu, 0.0684718601372)</td>
      <td>(Tây Ban Nha, 0.0587065940033)</td>
      <td>...</td>
      <td>(Ronaldo, 0.00717509146703)</td>
      <td>(Mỹ, 0.00698389203357)</td>
      <td>(vàng, 0.00687312326249)</td>
      <td>(giây, 0.00660382144629)</td>
      <td>(Zidane, 0.00230406611571)</td>
      <td>(Neymar, 0.00172859965884)</td>
      <td>(Việt Nam, 0.00123968718923)</td>
      <td>(HC, 0.00103038160134)</td>
      <td>(nữ, 0.000895401481367)</td>
      <td>(vợt, 0.0)</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 43 columns</p>
</div>



Kết quả tương đối tương tự KMeans do trong bài toán này ta chú trọng về sự khác biệt theo phương hơn là về độ lớn. (Nhắc lại điều kiện xây dựng hàm $f$, hai bài báo cùng chứa 1 từ có thành phần khoảng cách tương ứng với từ đó nhỏ hơn 1 bài báo chứa và 1 báo không chứa.)

### Bài 18. Hierarchical Clustering

*Hãy sửa chữa hàm **`train, predict, getClusterCenters, getExplicatveFeaturesForEachCluster`** để khi đối số model trong hàm train nhận giá trị **"Hierarchical_Euclidean"** thì thuật toán Hierachical với khoảng cách Euclide được thực hiện. So sánh kết quả và thời gian chạy với **KMeans.** *

(Kết quả nhìn chung tương tự với KMeans, do cả 2 phương pháp đều dựa trên giả thiết khoảng cách nhỏ khi tương đồng)

## Bình luận thêm

TD3 và 5 minh hoạ một project thực tế nhỏ về xử lí văn bản (text-mining). Bước preprocessing thường mất rất nhiều thời gian. Việc chọn model hợp lí cũng không hiển nhiên, thường qua nhiều bước thử chọn. Do khuôn khổ TD, ta không thực hiện bước dự đoán với các văn bản mới (có thể không lấy từ vnexpress). Bạn có thể tự thực hiện phần này.

Bài toán clustering nhìn chung "ill-defined" vì không có tiêu chuẩn rõ ràng về mặt toán học để đánh giá phương pháp nào là tốt hay xấu. Dựa trên hiểu biết và kinh nghiệm về lĩnh vực, ta có thể chọn ra một mô hình hợp lí.

Mô hình có thể được cải tiến theo nhiều hướng sau:

- Thay đổi cách giảm số chiều của vector bằng cách chọn từ quan trọng (ví dụ, thay đổi chặn trên, dưới của tần số và phương sai. Trong TD, ta dùng các tham số 200, 1000, 0.5. Chúng có thể được thay đổi)

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
