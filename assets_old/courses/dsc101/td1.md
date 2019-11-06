
# TD1 - Multiple Choice Questions on History

Trong TD đầu tiên này, ta sẽ thực hiện các thao tác lập trình cơ bản với Python thông qua việc thiết kế hai "ứng dụng" cơ bản làm việc với dữ liệu là bộ câu hỏi trắc nghiệm về chủ đề lịch sử.

## Mô tả

Tập tin <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/TD/QCM.csv">QCM.csv</a> chứa dữ liệu gồm 77 câu hỏi về lịch sử. Bạn có thể mở file với spyder (đừng thử với Excel, sẽ gặp vấn đề về font chữ). Dữ liệu trong tập tin có cấu trúc như sau:
- Mỗi câu hỏi tương ứng với một hàng (kết thúc bởi "\n")
- Mỗi hàng gồm 12 thông tin cách nhau bởi "\t" (do vậy luôn có 11 "\t"). 12 thông tin đó là: 
    - (0) Phạm vi: một str 2 kí tự ("TG" = Lịch sử thế giới, "VN" = Lịch sử Việt Nam)
    - (1) Thời đại: một số nguyên (1: Nguyên thuỷ và cổ đại, 2: Trung đại, 3: Cận đại, 4: Hiện đại)
    - (2) Số thứ tự câu hỏi trong file: một số nguyên
    - (3) Nội dung câu hỏi: một str
    - (4), (5), (6), (7) Phương án A, B, C, D: các str
    - (8) Đáp án: một str 1 kí tự ("A", "B", "C" hoặc "D")
    - (9) Độ khó: một str 1 kí tự ("E": dễ, "M": trung bình, "H": khó)
    - (10) Các hashtags: Hiện tại chưa có thông tin nên được kí hiệu bằng một str "-", nhưng có thể được cập nhật trong tương lai dưới dạng một str gồm các hashtag ngăn cách nhau bằng dấu "|"
    - (11) Giải thích: Hiện tại chưa có thông tin nên được kí hiệu bằng một str "-", có thể được cập nhật trong tương lai. 
    

Ta sẽ tìm cách lưu dữ liệu bằng một type thích hợp của Python trong bộ nhớ máy tính, sau đó tạo hai ứng dụng sau:
- Ứng dụng 1: Tạo ra một bộ *N* câu hỏi ngẫu nhiên từ dữ liệu, cho người chơi nhập câu trả lời mỗi câu hỏi và đánh giá độ chính xác câu trả lời của người chơi.
- Ứng dụng 2: Lọc ra tất cả các câu hỏi mà trong nội dung có từ khoá *S* (ví dụ "Lý"), sau đó thêm hashtag *R* (ví dụ "nhà Lý") vào cột (10) của dữ liệu, rồi lưu lại bản cập nhật này của dữ liệu.

File <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/TD/HisMCQ.py">HisMCQ.py</a> là nơi bạn cần viết code. Tập tin <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/TD/TestHisMCQ.py">TestHisMCQ.py</a> chứa các test cần chạy để kiểm tra code bạn viết trong HisMCQ.py có chạy tốt không. Hãy download các file vừa đề cập về cùng một thư mục. Bạn không cần sửa chữa gì trong TestHisMCQ.py trừ điều chỉnh các syntax trong Python 2 thành Python 3.

## Yêu cầu

Trong <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/TD/HisMCQ.py">HisMCQ.py</a>, bạn cần hoàn thành tất cả các hàm có chú thích #TODO ứng với mỗi bài tập. Một số đoạn code đã được viết sẵn, ví dụ các hằng số nêu rõ ý nghĩa các cột.


```python
ZONE = 0
PERIOD = 1
INDEX = 2
CONTENT = 3
OPTION_A = 4
OPTION_B = 5
OPTION_C = 6
OPTION_D = 7
CORRECTION = 8
LEVEL = 9
TAGS = 10
EXPLANATION = 11

DATAFILE = "QCM.csv"
NEWDATAFILE = "NewQCM.csv"
TAG_DELIMITOR = "|"
```

Sau khi hoàn thành xong mỗi bài tập, bạn có thể chạy test trong TestHisMCQ.py để kiểm tra.

Để chạy từng test, trong Spyder, bạn chọn Run -> Configure và gõ tên test tương ứng với bài tập (test_1, test_2, test_3, …)

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/TD/TestConfig.png"/>

Nếu code viết đúng, khi chạy code bạn sẽ nhìn thấy dòng “Test … OK”. Nếu sai, lỗi của Python hoặc kết quả test sẽ giúp bạn xác định code đang gặp vấn đề gì.

Bạn được tuỳ ý sử dụng các thư viện của Python, viết thêm các hàm phụ trong HisMCQ.py nếu thấy cần thiết. Bạn cũng có thể tự test các hàm đang viết bằng cách tạo một file mới, import HisMCQ và thử các phép test của chính bạn. (Ví dụ, dưới đây sẽ import History_Solution là lời giải hoàn chỉnh của TD. Bạn chưa nên đọc lời giải lúc này).


```python
from HisMCQ_Solution import *
```

## Phần 1 - Đọc và biểu diễn dữ liệu

### Bài 1 - Đọc dữ liệu
*(1) Hãy viết hàm **readQuestionFileAsString(filename)** nhận đối số **filename** là tên (đường dẫn) của file dữ liệu, và trả lại toàn bộ nội dung của file dưới dạng một **str**.*

Gợi ý: Sau khi viết, bạn có thể thử bằng test sau. Độ dài của **str** cần là 16812 nếu dùng Python 2 hoặc 13258 nếu dùng Python 3.


```python
text = readQuestionFileAsString("QCM.csv")
print(len(text))
```

    16812
    

*(2) Hãy viết tiếp hàm **readQuestionFileAsLines(filename)** nhận đối số **filename** là tên (đường dẫn) của file dữ liệu, và trả lại một **list** các **str**, mỗi **str** là một hàng trong file dữ liệu, chứa cả kí tự chuyển hàng "\n" nếu có.*

Gợi ý: Sau khi viết, tự test bằng đoạn code dưới đây.


```python
lines = readQuestionFileAsLines("QCM.csv")
print(lines[0])
print(lines[76])
print("\n" in lines[0]) #Should be True
```

    TG	1	1	Người tối cổ xuất hiện trên Trái Đất vào niên đại nào?	2.000.000.000 năm trước	80.000.000 năm trước	6.000.000 năm trước	400.000 năm trước	C	M	-	-
    
    VN	3	77	Năm 1866, "không tốn một viên đạn", thực dân Pháp đã lấy được 3 tỉnh nào của nước ta?	Gia Định, Định Tường, Biên Hoà	Nam Định, Hà Nam, Kinh Bắc	Thanh Hoá, Nghệ An, Hà Tĩnh	Vĩnh Long, An Giang, Hà Tiên	D	H	-	-
    True
    

*(3) Hãy viết tiếp hàm **readQuestionFileAsCleanLines(filename)** nhận đối số **filename** là tên (đường dẫn) của file dữ liệu, và trả lại một **list** các **str**, mỗi **str** là một hàng trong file dữ liệu, nhưng lần này xoá bỏ tất cả các kí tự chuyển hàng "\n" nếu có.*

Gợi ý: Sau khi viết, tự test bằng đoạn code dưới đây.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
print(lines[0])
print(lines[76])
print("\n" in lines[0]) #Should be False
```

    TG	1	1	Người tối cổ xuất hiện trên Trái Đất vào niên đại nào?	2.000.000.000 năm trước	80.000.000 năm trước	6.000.000 năm trước	400.000 năm trước	C	M	-	-
    VN	3	77	Năm 1866, "không tốn một viên đạn", thực dân Pháp đã lấy được 3 tỉnh nào của nước ta?	Gia Định, Định Tường, Biên Hoà	Nam Định, Hà Nam, Kinh Bắc	Thanh Hoá, Nghệ An, Hà Tĩnh	Vĩnh Long, An Giang, Hà Tiên	D	H	-	-
    False
    

### Bài 2 - Biểu diễn dữ liệu bằng list của list
Sau khi hoàn thành bài 1.(3), ta đã có một **list** các **str**, mỗi **str** tương ứng với một câu hỏi. Biểu diễn này chưa tốt vì chưa cho phép ta làm việc với từng thành phần của mỗi câu hỏi. Để tiếp cận mỗi thành phần, ta sẽ biểu diễn các **str** này thành các list có độ dài 12, tương ứng với 12 thành phần của câu hỏi.

*Hãy viết hàm **parseQuestionsAsListOfList(lines)** nhận đối số **lines** là một **list** các **str** có dạng như output của bài 1.(3), và trả lại kết quả là một **list** các **list** các **str**, mỗi **str** tương ứng với một thành phần (như phạm vi, nội dung, phương án A, hashtags...) của một câu hỏi, xuất hiện theo đúng thứ tự như trong file.*

Bạn có thể dùng đoạn code sau để test.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
print(questions[48][3])
print(questions[48][5])
```

    Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?
    Vạn Xuân
    

### Bài 3 - Tìm đáp án của câu hỏi

Giả sử ta có tình huống như sau: biết nội dung của câu hỏi và cần tìm đáp án của nó dưới dạng tự luận (tức là đáp án hiển thị một cách rõ ràng chứ không chỉ là một kí tự A, B, C, D).

*Hãy viết hàm **answer(questions_list, question_content)** nhận đối số **questions_list** là một **list** các **list** như output của bài 2, và một đối số **question_content** là một **str** nội dung của câu hỏi, rồi trả lại đáp án đúng dưới dạng một **str**.*

Ví dụ, đoạn code để test dưới đây cần ra đúng kết quả "Vạn Xuân"


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
print(answer(questions, "Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?"))
```

    Vạn Xuân
    

**Hiểu thêm: ** Ta có thể tìm tốc độ tìm kiếm bằng cách cho hàm chạy 10000 lần và xem thời gian cần thiết để đưa ra kết quả trung bình là bao nhiêu.


```python
import time
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
begin = time.time() #Get current time
for i in range(10000):
    a = answer(questions, "Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?")
end = time.time() #Get current time
print("Find the answer using list of list took " + str(end - begin) + " seconds")
```

    Find the answer using list of list took 0.108000040054 seconds
    

### Bài 4 - Thử một cách biểu diễn dữ liệu khác, dùng từ điển

Giả sử tình huống của bài 3 có xuất hiện trong thực tế và ta cần biểu diễn dữ liệu để thời gian tìm kiếm đáp án theo nội dung của câu hỏi được tối ưu. Ta sẽ thử biểu diễn dữ liệu theo một **dict** các **list**, trong đó các key trong từ điển do ta tuỳ chọn. Ta thử dùng **hash(nội dung câu hỏi)** làm key và list 12 thành phần của câu hỏi làm value.

*Hãy viết hàm **parseQuestionsAsDictionary(lines)**  nhận đối số **lines** là một **list** các **str** có dạng như output của bài 1.(3), và trả lại kết quả là một **dict** với các **key** là **hash** của nội dung câu hỏi, và **value** là list 12 thành phần của câu hỏi.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsDictionary(lines)
A = hash("Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?")
print(A) #May have different result in Python 3
print(questions[hash(A)])
```

    -520583810
    ['VN', '1', '49', 'Qu\xe1\xbb\x91c hi\xe1\xbb\x87u n\xc6\xb0\xe1\xbb\x9bc ta d\xc6\xb0\xe1\xbb\x9bi th\xe1\xbb\x9di L\xc3\xbd Nam \xc4\x90\xe1\xba\xbf l\xc3\xa0 g\xc3\xac?', '\xc3\x82u L\xe1\xba\xa1c', 'V\xe1\xba\xa1n Xu\xc3\xa2n', '\xc4\x90\xe1\xba\xa1i C\xe1\xbb\x93 Vi\xe1\xbb\x87t', '\xc4\x90\xe1\xba\xa1i Vi\xe1\xbb\x87t', 'B', 'E', '-', '-']
    

### Bài 5 - Tìm đáp án với biểu diễn dữ liệu bằng từ điển

Cùng một tình huống với bài 3, ta muốn tìm câu trả lời cho một câu hỏi bằng cách dùng từ điển đã viết ở bài 4.

*Hãy viết hàm **answer_2(questions_dictionary, question_content)** nhận đối số **questions_dictionary** là một **dict** các **list** như output của bài 4, và một đối số **question_content** là một **str** nội dung của câu hỏi, rồi trả lại đáp án đúng dưới dạng một **str**.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsDictionary(lines)
print(answer_2(questions, "Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?"))
```

    Vạn Xuân
    

**Hiểu thêm**: Ta thử tìm kiếm 10000 lần bằng phương pháp dùng từ điển.


```python
import time
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsDictionary(lines)
begin = time.time() #Get current time
for i in range(10000):
    a = answer_2(questions, "Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?")
end = time.time() #Get current time
print("Find the answer using dict of list took " + str(end - begin) + " seconds")
```

    Find the answer using dict of list took 0.0090000629425 seconds
    

So sánh với phương pháp dùng **list**, kết quả như thế nào? Sự bất cẩn của phương pháp thứ hai này là gì?

## Phần 2 - Một số hàm phụ chuẩn bị cho ứng dụng

### Bài 6 - Tìm câu hỏi chứa một từ

Đọc lại về hai ứng dụng ta muốn tạo ra ban đầu, ta thấy tình huống "tìm kết quả của câu hỏi" không thực sự liên quan đến 2 trường hợp cần giải quyết, do đó cách biểu diễn ở bài 4, 5 không phát huy được . Vì vậy, ta sẽ chọn phương pháp biểu diễn bằng **list** các **list** như bài 2, 3.

Bây giờ, quay lại với hai ứng dụng, ta thấy có một số hàm cần viết. Ở ứng dụng 2, đó là việc tìm ra tất cả các câu hỏi chứa 1 từ khoá tuỳ ý.

*Hãy viết hàm **searchQuestionsContainingWord(questions_list, keyword)** nhận đối số **questions_list** là **list** các **list** như output của bài 2, và đối số **keyword** là từ khoá có thể xuất hiện trong nội dung của câu hỏi, và trả lại một list con của **questions_list** chứa tất cả các câu hỏi mà nội dung chứa từ khoá đó.*

Bạn có thể dùng **filter** để làm gọn code của mình. Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
S = searchQuestionsContainingWord(questions, "Trần")
print(len(S))
print(S[0])
print(S[0][CONTENT]) #CONTENT = 3
```

    5
    ['VN', '1', '26', 'Nh\xc3\xa2n v\xe1\xba\xadt l\xe1\xbb\x8bch s\xe1\xbb\xad n\xc3\xa0o ch\xe1\xbb\x91ng l\xe1\xba\xa1i qu\xc3\xa2n x\xc3\xa2m l\xc6\xb0\xe1\xbb\xa3c Tr\xe1\xba\xa7n (Trung Qu\xe1\xbb\x91c), gi\xe1\xbb\xaf v\xe1\xbb\xafng \xc4\x91\xe1\xbb\x99c l\xe1\xba\xadp t\xe1\xba\xa1m th\xe1\xbb\x9di c\xe1\xbb\xa7a d\xc3\xa2n t\xe1\xbb\x99c?', '\xc4\x90inh Ki\xe1\xba\xbfn, L\xc3\xbd T\xe1\xbb\xb1 Ti\xc3\xaan (th\xe1\xba\xbf k\xe1\xbb\x89 VII)', 'D\xc6\xb0\xc6\xa1ng Thanh (th\xe1\xba\xbf k\xe1\xbb\x89 IX)', 'L\xc3\xbd Ph\xe1\xba\xadt T\xe1\xbb\xad (th\xe1\xba\xbf k\xe1\xbb\x89 VI, VII)', 'Tri\xe1\xbb\x87u Quang Ph\xe1\xbb\xa5c (th\xe1\xba\xbf k\xe1\xbb\x89 VI)', 'D', 'H', '-', '-']
    Nhân vật lịch sử nào chống lại quân xâm lược Trần (Trung Quốc), giữ vững độc lập tạm thời của dân tộc?
    

### Bài 7 - Cập nhật một thành phần của câu hỏi

Cũng trong ứng dụng 2, ta cần sửa chữa một câu hỏi (ví dụ: thêm hashtag ở cột 10).

*Hãy viết hàm **modify(question, column_index, text)** nhận đối số **question** là một câu hỏi (tức một **list** 12 thành phần),  **column_index** là chỉ số của thành phần cần thay đổi (ví dụ, thành phần "Thời đại" có chỉ số cột là PERIOD = 1), **text** là nội dung cần ghi đè vào cột đó; và thực hiện sự ghi đè. Hàm này không cần return gì cả.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
print(questions[12][CONTENT]) #CONTENT = 3
modify(questions[12], CONTENT, "Thời nào của Trung Quốc cổ đại được chia thành Xuân Thu và Chiến Quốc?")
print("Updated!")
print(questions[12][CONTENT])
```

    Xuân Thu và Chiến Quốc là hai giai đoạn của thời kì nào sau đây của Trung Quốc cổ đại?
    Updated!
    Thời nào của Trung Quốc cổ đại được chia thành Xuân Thu và Chiến Quốc?
    

*Hãy viết hàm **addTag(question, newtag)** nhận đối số **question** là một câu hỏi (tức một **list** 12 thành phần) và **newtag** là một hashtag mới cần thêm vào ở cột có chỉ số 10, và thực hiện việc thêm hashtag này. Yêu cầu: nếu chưa có hashtag nào, tức nội dung của thành phần HASHTAG đó đang có giá trị "-", ta xoá "-" và ghi đè **newtag** vào đó. Nếu đã có các hashtag và **newtag** chưa xuất hiện trong các hashtag cũ, ta thêm "|" và **newtag** vào bên phải.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
print(questions[48][TAGS])
addTag(questions[48], "Vua")
print(questions[48][TAGS])
addTag(questions[48], "Bắc thuộc")
print(questions[48][TAGS])
addTag(questions[48], "Bắc thuộc")
print(questions[48][TAGS])
addTag(questions[48], "Khởi nghĩa")
print(questions[48][TAGS])
```

    -
    Vua
    Vua|Bắc thuộc
    Vua|Bắc thuộc
    Vua|Bắc thuộc|Khởi nghĩa
    

### Bài 8 - Xuất dữ liệu ra file

Giả sử dữ liệu đã được thay đổi và đang được lưu trong một **list** các **list**. Ta muốn lưu nó vào một file mới.

*Hãy viết hàm **saveDatabaseToFile(questions_list, newfilename)** nhận đối số **questions_list** là **list** các **list** (có dạng như output của bài 2) và lưu nó vào **newfilename** với định dạng như file hệt như file gốc (mỗi hàng là một câu hỏi, các câu hỏi cách nhau bởi tab)*

Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
addTag(questions[48], "Vua")
addTag(questions[48], "Bắc thuộc")
addTag(questions[48], "Khởi nghĩa")
saveDatabaseToFile(questions, "QCM2.csv")
newquestions = parseQuestionsAsListOfList(readQuestionFileAsCleanLines("QCM2.csv"))
print(newquestions[48][TAGS])
```

    Vua|Bắc thuộc|Khởi nghĩa
    

### Bài 9 - Phát sinh một câu hỏi ngẫu nhiên

Ta xem như đã viết xong các hàm con cho ứng dụng 2. Quay lại với ứng dụng 1, đầu tiên cần tạo ra các câu hỏi ngẫu nhiên.

*Hãy viết hàm **generateRandomQuestion(questions_list)** nhận đối số là **questions_list** (có dạng như output của bài 2) trả lại một câu hỏi ngẫu nhiên trong đó (tức một **list** 12 thành phần).*

Đề bài không yêu cầu, nhưng bạn có thể để các câu hỏi được chọn ngẫu nhiên với xác suất như nhau.

Thư viện **random** với hàm **randint** có thể giúp bạn. Đoạn code dưới đây giúp test hàm của bạn. (Khi chạy nó nhiều lần sẽ ra các kết quả khác nhau).


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
print(generateRandomQuestion(questions)[3])
```

    Thái thú nhà Hán phải cắt tóc, cạo râu, lẩn trong đám loạn quân chạy về nước trong cuộc khởi nghĩa Hai Bà Trưng là ai?
    

### Bài10 - Phát sinh *N* câu hỏi ngẫu nhiên

*Hãy viết hàm **generateRandomQuestionList(questions_list, nb_questions)** nhận đối số là **questions_list** (có dạng như output của bài 2) và trả lại một **tuple** gồm hai phần tử, *
- *Phần tử thứ nhất là list chỉ số của các câu hỏi được chọn.*
- *Phần tử thứ hai là list con của **questiosn_list** gồm N câu hỏi khác nhau một cách ngẫu nhiên. Nếu N lớn hơn số lượng câu hỏi N' có trong **question_list**, trả lại N' câu hỏi.*

Thư viện **random** với hàm **shuffle** có thể giúp bạn. Đoạn code dưới đây giúp test hàm của bạn. (Khi chạy nhiều lần sẽ ra kết quả khác nhau)


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
L = generateRandomQuestionList(questions, 5)
print(type(L))
print(L[0]) #Should return a random list of 5 elements, each of which is a random number between 0 and 76
print(L[1][0][CONTENT])
print("")
print(L[1][4][CONTENT])
```

    <type 'tuple'>
    [1, 47, 19, 30, 9]
    Đâu trong các sự kiện dưới đây đánh dấu bước ngoặt cho thấy loài vượn đã tiến hoá thành loài người?
    
    Kim tự tháp Giza được xây dựng ở thời kì nào của Ai Cập cổ đại?
    

### Bài 11 - Kiểm tra kết quả

*Hãy viết hàm **isCorrectAnswer(question, answer)** nhận đối số **question** là một câu hỏi (tức một **list** 12 thành phần) và đối số **answer** là một trong 4 **str** "A", "B", "C" hoặc "D", và trả lại **True** hoặc **False** tuỳ theo đó có phải là câu trả lời đúng cho câu hỏi không.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
lines = readQuestionFileAsCleanLines("QCM.csv")
questions = parseQuestionsAsListOfList(lines)
print(isCorrectAnswer(questions[12], "B"))
```

    True
    

## Phần 3 - Ứng dụng thứ nhất - Trắc nghiệm

Ta xây dựng một hàm có tương tác với người sử dụng thực hiện các thao tác sau:
- Hỏi người chơi muốn trả lời một bộ gồm bao nhiêu câu hỏi.
- Đợi người dùng nhập một số nguyên dương *N*
- Tạo một bộ câu hỏi ngẫu nhiên với *N* câu hoặc với số câu trong dữ liệu (nếu *N* quá lớn)
- In ra câu hỏi 1, đợi đáp án của người chơi, người chơi nhập vào một trong các kí tự 'A', 'B', 'C', 'D'
- In ra câu hỏi 2, đợi đáp án của người chơi, người chơi nhập vào một trong các kí tự 'A', 'B', 'C', 'D'
- ...
- In ra câu hỏi *N* (hoặc cuối cùng), đợi đáp án của người chơi, người chơi nhập vào một trong các kí tự 'A', 'B', 'C', 'D'
- Tính số câu trả lời đúng người chơi đã đạt được, in ra số câu trả lời đúng và kết thúc chương trình.

*Hãy chạy hàm **generateHistoryTest()** trong iPython console, nếu các hàm trước đó đã hoàn thiện, chương trình sẽ chạy kiểu như sau*


```python
generateHistoryTest()
```

    Please choose the number of question: 5
    Bạo loạn do ai gây ra thời Lý Cao Tông khiến Lý Cao Tông, hoàng hậu và thái tử Lý Hạo Sảm phải bỏ kinh thành ẩn nấp tại Quy Hoá và Hải Ấp?
    A. Phạm Du
    B. Phạm Bỉnh Di
    C. Quách Bốc
    D. Đoàn Thượng
    Please answer by typing A, B, C or D: A
    Vua nào ở thời Lý khi lên ngôi có sự tranh chấp với em, phải dẹp bằng binh đao? (Sử cũ gọi là Loạn Tam vương)
    A. Lý Thái Tông
    B. Lý Thánh Tông
    C. Lý Nhân Tông
    D. Lý Anh Tông
    Please answer by typing A, B, C or D: A
    Thời Lý, Đại Việt không có chiến tranh với quốc gia nào sau đây?
    A. Trảo Oa
    B. Chân Lạp
    C. Đại Lý
    D. Chiêm Thành
    Please answer by typing A, B, C or D: A
    Kinh đô nước ta trong thời kì Hùng Vương thuộc địa phương nào hiện nay?
    A. Phong Châu, Phú Thọ
    B. Đông Anh, Hà Nội
    C. Hoa Lư, Ninh Bình
    D. Thanh Hoá
    Please answer by typing A, B, C or D: A
    Trận chiến ở phòng tuyến sông nào đánh dấu kết thúc thắng lợi của chiến tranh chống quân Tống năm 1077?
    A. Bạch Đằng
    B. Đản Nãi
    C. Như Nguyệt
    D. Lãnh Kinh
    Please answer by typing A, B, C or D: A
    Your score is 3 / 5
    Game finished.
    

### Cải tiến trò chơi

Bây giờ, ta muốn cải tiến hàm này. Sau khi người chơi trả lời xong *N* câu (*N* $\leq$ 50), chương trình sẽ hỏi người chơi có muốn tiếp tục hay không. Nếu người chơi nhấn "Y" (in hoa), chương trình sẽ tạo lại ngẫu nhiên *N* câu hỏi sao cho" không quá *N/3* đã được sử dụng trước đó" (ta gọi đây là **điều kiện 1**). Nếu điều kiện 1 không thoả mãn, chương trình sẽ lặp lại việc tạo ngẫu nhiên cho đến khi hoặc điều kiện 1 được thoả mãn, hoặc đã thực hiện đến bước lặp thứ 50. 

- Nếu đã thực hiện bước lặp 50 mà điều kiện 1 vẫn không được thoả mãn. Chương trình thông báo không còn câu hỏi, in số câu trả lời đúng của người chơi và kết thúc.
- Nếu chọn được bộ câu hỏi thoả mãn điều kiện 1, trò chơi tiếp tục với việc in từng câu hỏi trong số *N* câu hỏi mới cho người chơi và nhận đáp án. Cuối cùng, cũng hỏi người chơi có muốn tiếp tục một bộ câu hỏi nữa hay không.

Chương trình tiếp tục như vậy cho đến khi người chơi gõ một kí tự khác 'Y' cho câu hỏi có muốn tiếp tục không, hoặc khi đã lặp 50 lần nhưng không lần nào thoả mãn điều kiện 1 (tìm được bộ câu hỏi có số câu trùng nhỏ hơn *N*/3).

Lấy ý tưởng từ hàm **generateHistoryTest()**, bạn có thể viết hàm **generateRepeatedHistoryTest()** cho kịch bản trên. Ta viết một hàm phụ trước.

### Bài 12. Làm việc với set

*Hãy viết hàm **isGoodChoice(indices_1, indices_2, ceil)** nhận các đối số **indices_1, indices_2** là 2 **list** các phần tử phân biệt cùng với một số nguyên khồng âm **ceil**, và trả lại **True** nếu số phần tử chung của **indices_1, indices_2** không lớn hơn **ceil**, và **False** nếu ngược lại.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
A = [1, 2, 3, 4, 5, 6]
B = [10, 8, 6, 4, 2, 0]
print(isGoodChoice(A, B, 3))
print(isGoodChoice(B, A, 2))
```

    True
    False
    

### Bài 13. Cải tiến trò chơi

*Hãy viết hàm **generateRepeatedHistoryTest()** cho kịch bản cải tiến trên*.

Lưu ý rằng trong file TestHisMCQ.py không có test13. Bạn sẽ tự chạy hàm đã viết trong iPython console để kiểm tra hàm. Dưới đây là một ví dụ.


```python
generateRepeatedHistoryTest()
```

    Please choose the number of question: 2
    Xuân Thu và Chiến Quốc là hai giai đoạn của thời kì nào sau đây của Trung Quốc cổ đại?
    A. Tây Chu
    B. Đông Chu
    C. Tây Hán
    D. Đông Hán
    Please answer by typing A, B, C or D: A
    Địa danh nào gắn với cuộc kháng chiến chống quân Trần Bá Tiên của Triệu Quang Phục?
    A. Đầm Dơi
    B. Sông Bạch Đằng
    C. Đầm Dạ Trạch
    D. Núi Tản Viên
    Please answer by typing A, B, C or D: A
    Your score this time is 0 / 2
    Number of questions has been used in the database: 2.
    Do you want to replay? Press 'Y' if yes, any other key if no: Y
    Bạo loạn do ai gây ra thời Lý Cao Tông khiến Lý Cao Tông, hoàng hậu và thái tử Lý Hạo Sảm phải bỏ kinh thành ẩn nấp tại Quy Hoá và Hải Ấp?
    A. Phạm Du
    B. Phạm Bỉnh Di
    C. Quách Bốc
    D. Đoàn Thượng
    Please answer by typing A, B, C or D: B
    Một trong những sự kiện đánh dấu sự chuyển thời đại từ Xuân Thu sang Chiến Quốc ở Trung Quốc cổ đại là việc nước Tấn bị chia thành 3 nước nào sau đây?
    A. Triệu, Nguỵ, Yên
    B. Yên, Nguỵ, Hàn
    C. Triệu, Nguỵ, Hàn
    D. Yên, Hàn, Triệu
    Please answer by typing A, B, C or D: D
    Your score this time is 0 / 2
    Number of questions has been used in the database: 4.
    Do you want to replay? Press 'Y' if yes, any other key if no: Y
    Ba người con của vua Lê Thái Tông từng ở ngôi vua là những ai?
    A. Lê Nhân Tông, Lê Thánh Tông, Lê Hiến Tông
    B. Lê Nhân Tông, Lê Thánh Tông, Lê Nghi Dân (Lạng Sơn Vương)
    C. Lê Thánh Tông, Lê Hiến Tông, Lê Nghi Dân (Lạng Sơn Vương)
    D. Lê Nhân Tông, Lê Hiến Tông, Lê Nghi Dân (Lạng Sơn Vương)
    Please answer by typing A, B, C or D: B
    Thủ khoa thi đình đầu tiên trong lịch sử Việt Nam (thời Lý Nhân Tông) là ai?
    A. Mạc Đĩnh Chi
    B. Nguyễn Hiền
    C. Lê Văn Thịnh
    D. Lương Thế Vinh
    Please answer by typing A, B, C or D: C
    Your score this time is 2 / 2
    Number of questions has been used in the database: 6.
    Do you want to replay? Press 'Y' if yes, any other key if no: N
    Your highest score in an attempt is 2 / 2
    Game finished.
    

## Phần 4 - Ứng dụng thứ hai - Cập nhật tag bằng tìm kiếm từ khoá

Kịch bản cho ứng dụng thứ hai như sau.

- Chương trình hỏi người dùng cần tìm từ khoá nào, người dùng nhập vào một từ
- Chương trình hỏi người dùng cần nhập hashtag nào, người dùng nhập hashtag
- Sau đó chương trình in ra từng câu hỏi chứa từ khoá, người dùng ấn 'Y' để quyết định thêm hashtag vào câu hỏi đó, và phím khác nếu không.
- Chương trình cập nhật dữ liệu vào một file mới có tên "NewQCM.csv",

*Hãy chạy hàm **generateTagUpdateApplication()** trong iPython console, lưu ý chỉ dùng các hashtag không dấu như "vua", "Nam"( do iPython console không xử lí tốt tiếng Việt). Nếu các hàm trước viết đúng, chương trình sẽ chạy như mẫu sau*


```python
generateTagUpdateApplication()
```

    Which keyword do you want to search? Please type in Vietnamese without accent: Nam
    16 questions found.
    What is the hashtag you would like to use?: Vietnam
    Thời kì Nam Bắc triều ở Trung Quốc diễn ra trong những thế kì nào?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: N
    Next question: 
    Tại di chỉ nào dưới đây các nhà khảo cổ đã tìm thấy hoá thạch của răng người tối cổ ở Việt Nam?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Tướng giặc nào của Nam Hán tử trận trên sông Bạch Đằng trong trận chiến với Ngô Quyền năm 938?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Thời Triệu (Nam Việt) đô hộ (thế kỉ II TCN), nước ta được chia thành các đơn vị hành chính nào?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Cuối thế kỉ I, đầu thế kỉ II, thái thú nào của nhà Hán đã thực hiện nhiều chính sách tiến bộ trong việc quản lí vùng đất nước ta hiện nay, được người đời sau gọi là "Nam Giao học tổ"?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Năm 866, nhà Đường đổi vùng lãnh thổ Bắc Việt Nam thành đơn vị hành chính nào?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Người Việt Nam nào đã phát minh ra súng thần công?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Những vua nào thời Tiền Lê được truy tôn miếu hiệu? (Miếu hiệu của một vua Việt Nam bắt đầu bằng họ của vua, tiếp đến là một chữ và cuối cùng là chữ "Tổ" hoặc "Tông", ví dụ Lí Thái Tổ, Trần Thái Tông)
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Vân Đồn, thương cảng Việt Nam đầu tiên thời phong kiến, được mở dưới thời vua nào?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Quốc hiệu Đại Việt được sử dụng từ thời vua nào của Việt Nam?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Vương triều nào dưới đây có thời gian trị vì lâu nhất tại Việt Nam?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Thủ khoa thi đình đầu tiên trong lịch sử Việt Nam (thời Lý Nhân Tông) là ai?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Với 56 năm cai trị, vua nào ở ngôi lâu nhất trong lịch sử Việt Nam thời phong kiến?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Quốc giáo của Việt Nam thời Lý là tôn giáo nào?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    Trong 4 công trình dưới đây ("An Nam tứ đại khí" của Đại Việt), công trình nào không được chế tác, xây dựng thời Lý?
    Do you want to tag this question with the hashtag Vietnam? Press 'Y' if yes, any other key if no: Y
    No more question!
    Process finished!
    

*Kiểm tra lại file <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/TD/NewQCM.csv">NewQCM.csv</a>. Hashtag "king" sẽ xuất hiện trong các câu hỏi được chọn.*

<center>Hết</center>
