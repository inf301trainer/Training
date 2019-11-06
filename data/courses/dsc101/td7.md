
# TD7 - Đọc phiếu chấm học bổng Đồng Hành

## Mô tả

<a href="donghanh.net">Quỹ học bổng Đồng Hành</a> được thành lập và quản lí bởi một nhóm du học sinh Việt Nam tại Pháp, thực hiện ý tưởng gây quỹ và trao học bổng cho những sinh viên nghèo hiếu học tại hơn 15 trường đại học ở Việt Nam. Ở mỗi học kì, sinh viên có nguyện vọng sẽ làm hồ sơ xin học bổng Đồng Hành. Một hội đồng giám khảo ở Pháp (thường là các du học sinh) sẽ nghiên cứu các hồ sơ này cùng kết quả một vòng phỏng vấn để quyết định chọn những ứng viên xứng đáng nhất để trao học bổng.

Để đánh giá và so sánh các ứng viên với nhau, mỗi giám khảo sẽ đọc các hồ sơ và cho 4 điểm số ứng với các tiêu chí: hoàn cảnh, học tập, ước mơ và điểm cộng dưới dạng một điểm số từ 0 đến 10 và ghi vào một phiếu chấm. Trong TD này ta giả định điểm luôn là một số **nguyên**.

Hình dưới đây minh hoạ một phiếu chấm như vậy. (Tên các ứng viên có thể đã được thay đổi)

<img src="https://raw.githubusercontent.com/riduan91/DSC101/ba2b4d2aa94cc56bfa6757666b13731d335fe547/Lesson7/TD/RawForm/BKDN_1.jpg"/ width=800; >

Sau đó, ban tổ chức sẽ thu lại phiếu chấm của tất cả các giám khảo và nhập điểm vào máy tính (cơ sở dữ liệu), thực hiện việc tổng hợp điểm để tính toán kết quả cuối cùng. Việc nhập điểm hiện nay được thực hiện bằng tay.


## Yêu cầu


Trong TD này và TD sau, ta xây dựng một chương trình để từ bản scan phiếu chấm, tự động đọc ra kết quả dưới dạng các điểm số ứng với các ứng viên để thay cho việc con người tự đọc và nhập điểm.

- Ở phần 1, ta sẽ sử dụng thư viện **`opencv`** (hay **cv2**), một thư viện hỗ trợ xử lí hình ảnh để giúp biến mỗi hình ảnh thành một numpy array, và xử lí những thao tác cơ bản như biến ảnh màu thành trắng đen, tìm các đường ngang dọc trong hình. Sau đó, lọc ra các ô chứa điểm (4 ô ứng với cột "hoàn cảnh", "học tập", "ước mơ", "điểm cộng" giao với các hàng). Dữ liệu là hình ảnh các phiếu chấm được scan với chất lượng khá tốt, định dạng jpg, dung lượng không quá lớn (khoảng 500KB)

- Ở phần 2, ta sẽ thực hiện preprocessing để biến các hình ảnh thành các vector.

- Ở phần 3, ta sẽ sử dụng các mô hình đã học để phân loại chữ số dựa trên dữ liệu là những ô điểm mà ta đã lọc ra. Trong giới hạn kiến thức ở amphi, ta sẽ phân loại các cặp chữ số (binary classification, chẳng hạn 0 với 1, 4 với 5).

- Ở phần 4, ta sẽ lặp lại phần 2, nhưng thay vì train với dữ liệu từ các phiếu chấm, ta sẽ train với dữ liệu từ dataset MNIST (với 70000 hình ảnh).

Ở TD sau, ta sẽ chuyển binary classification thành multiclass classification và hoàn thiện chương trình tự động đọc phiếu chấm.

TD này cũng có mục đích giới thiệu bài toán xử lí ảnh, nhưng không đi sâu vào chuyên đề này.

Bạn cần viết các hàm và hoàn thành các class trong file `DHEvaluation.py` và chạy các test theo hướng dẫn ở các bài tập để kiểm tra hàm đã viết.


```python
from DHEvaluation_Solution import *
```

## Phần 1 - Lọc ra các ô điểm. Class `EvaluationForm`

Ta sẽ xây dựng class `EvaluationForm`, mỗi instance của class này mô tả một phiếu chấm. Mỗi instance của class có các attribute cơ bản sau:

- `img`: là một numpy array 2 chiều mô tả hình ảnh dưới dạng các điểm trắng đen

và được xây dựng các hàm sau (sẽ được yêu cầu viết trong các bài tập)

- `__init__(source_img)`: khởi tạo instance từ một hình ảnh có định dạng .jpg 
- `getImage()`, `getHeight()`, `getWidth()`: các getter cơ bản trả lại hình ảnh dưới dạng numpy array, kích thước ảnh
- `getEdge()`: lấy các đường viền của ảnh
- `getHorizontalLines()`, `getHorizontalLineGroups()`, `getCleanHorizontalLineGroups()`, `getVerticalLines()`, `getVerticalLineGroups()`, `getCleanVerticalLineGroups()`: lấy và xử lí các đường kẻ ngang và dọc của bảng điểm
- `getCellLeftEdges()`, `getCellRightEdges()`, `getCellTopEdges()`, `getCellBottomEdges()`: lấy toạ độ của các ô điểm
- `saveCells()`: lưu tất cả các ô điểm dưới dạng các tập tin hình ảnh hình vuông kích thước 28 pixel x 28 pixel

### Bài 1 - Cài đặt thư viện opencv

*Hãy tìm cách cài đặt thư viện opencv. (Bạn có thể google, "Install opencv with conda"). *

Đoạn code test sau cần chạy không có lỗi.


```python
import cv2
```

### Bài 2 - Đọc hình ảnh

Chức năng chính của **`cv2`** là biến mỗi hình ảnh thành một numpy array. Ví dụ trong đoạn code dưới đây, ta có một hình ảnh tại đường dẫn `RawForm/BKDN_1.jpg`. Hàm **`imread`** của **`cv2`** sẽ đọc hình ảnh và biểu diễn nó dưới dạng một numpy array như sau:


```python
img = cv2.imread("RawForm/BKDN_1.jpg", cv2.IMREAD_GRAYSCALE)
print(img)
print(img.shape)
```

    [[ 47  49  66 ... 246  55  20]
     [204 203 210 ... 245  54  21]
     [255 254 252 ... 241  51  23]
     ...
     [255 255 255 ...  52  32  17]
     [255 255 255 ...  29  24  13]
     [255 255 255 ...  12  17   9]]
    (2480L, 3504L)
    

Do `RawForm/BKDN_1.jpg` là hình ảnh có kích thước 2480 pixel x 3504 pixel, numpy array được trả lại cũng có kích thước tương ứng. Tham số `cv2.IMREAD_GRAYSCALE` nói rằng ta muốn đọc ảnh dưới dạng đen trắng. Mỗi phần tử trong numpy array là một số nguyên trong đoạn \[0, 255\]: 0 là màu đen, 255 là màu trắng, giữa hai giá trị là các màu xám đậm nhạt khác nhau. Thư viện **`matplotlib`** sẽ giúp ta "vẽ" lại các hình ảnh từ numpy array trên. 


```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(20,12))
plt.imshow(img, cmap='gray')
```




    <matplotlib.image.AxesImage at 0xc83db00>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_11_1.png)


Trong nhiều bài toán, ta không cần quan tâm đến các màu xám, mà sẽ biến hình ảnh thành các ảnh thuần tuý trắng đen. Ví dụ: ta sẽ cho các điểm có giá trị $<$168 thành màu đen và $\geq$168 thành màu trắng. Khi đó, hàm **`threshold`** sẽ giúp ta. (Tham số `cv2.THRESH_BINARY` nói rằng ta muốn đưa ảnh về trắng đen)


```python
thres, black_white_img = cv2.threshold(img, 168, 255, cv2.THRESH_BINARY)
print thres
print black_white_img
```

    168.0
    [[  0   0   0 ... 255   0   0]
     [255 255 255 ... 255   0   0]
     [255 255 255 ... 255   0   0]
     ...
     [255 255 255 ...   0   0   0]
     [255 255 255 ...   0   0   0]
     [255 255 255 ...   0   0   0]]
    


```python
plt.figure(figsize=(20,12))
plt.imshow(black_white_img, cmap='gray')
```




    <matplotlib.image.AxesImage at 0xf343a58>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_14_1.png)


Ta sẽ thiết lập một class **`EvaluationForm`** mô tả một phiếu chấm, trong đó attribute **`self.img`** là biểu diễn dạng trắng đen của hình ảnh ứng với phiếu chấm đó. **`self.img`** sẽ luôn là một array có kích thước bằng kích thước hình ảnh, và có mỗi phần tử nhận giá trị 0 hoặc 255.

*Hãy viết instant method **`__init__(self, source_image)`** nhận đối số **`source_image`** là đường dẫn của hình ảnh và khởi tạo instance **`self`** của class **`EvaluationForm`** bằng cách gán cho attribute **`self.img`** numpy array biểu diễn dạng đen trắng của hình ảnh đó. Ta chọn giá trị `self.BLACK_WHITE_THRESHOLD = 168` để chia đen trắng.*

Đoạn code dưới đây test hàm của bạn.


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
print myform.img
print myform.img.shape
```

    [[  0   0   0 ... 255   0   0]
     [255 255 255 ... 255   0   0]
     [255 255 255 ... 255   0   0]
     ...
     [255 255 255 ...   0   0   0]
     [255 255 255 ...   0   0   0]
     [255 255 255 ...   0   0   0]]
    (2480L, 3504L)
    

### Bài 3 - Xác định các đường viền (contour)

*Đọc tài liệu sau đây: https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html để hiểu cách cv2 xác định các đường viền.*

*Viết instant method **`getEdges()`** trong class **`EvaluationForm`** trả lại một numpy array biểu diễn tất cả các đường viền của hình ảnh tương ứng (các điểm thuộc một đường viền sẽ có màu trắng, những điểm không thuộc đường viền nào sẽ có màu đen). Bạn có thể sử dụng các giá trị sau làm $minVal$ và $maxVal$ trong tài liệu.*

``` python
EDGE_LOW_VALUE = 100
EDGE_HIGH_VALUE = 800
```

Đoạn code dưới đây giúp test hàm của bạn.


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
print myform.getEdges()
```

    [[255 255 255 ... 255   0   0]
     [  0   0   0 ... 255   0   0]
     [  0   0   0 ... 255   0   0]
     ...
     [  0   0   0 ...   0   0   0]
     [  0   0   0 ...   0   0   0]
     [  0   0   0 ...   0   0   0]]
    


```python
plt.figure(figsize=(20,12))
plt.imshow(myform.getEdges(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0xf7ca7b8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_19_1.png)


Bạn có thể zoom một khu vực để nhìn rõ hơn, ví dụ: 400 pixel trung tâm theo mỗi chiều của hình.


```python
edges = myform.getEdges()
height, width = edges.shape
plt.imshow(edges[height//2 - 200: height//2 + 200, width//2 - 200: width//2 + 200], cmap='gray')
```




    <matplotlib.image.AxesImage at 0xf9c3940>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_21_1.png)


### Bài 4 - Xác định các đường ngang dọc

Sau khi xác định các đường viền, điều ta quan tâm là các đường kẻ ngang và dọc của bảng. Xác định được các đường ngang dọc này, ta sẽ xác định được các ô của bảng điểm.

Giả sử `edges` là numpy array biểu diễn các đường viền (output của bài 3), phép toán

``` python
lines = cv2.HoughLinesP(edges, 2, np.pi/2, VOTE)
```

sẽ giúp ta xác định các đường ngang hoặc dọc có đạt đến một ngưỡng `VOTE`. (Hãy tìm hiểu thuật toán HoughLine trong tìm kiếm hình ảnh để hiểu `VOTE` là gì và vì sao như vậy).

Kết quả sẽ được biểu diễn dưới dạng tung và hoành độ của điểm đầu, rồi tung và hoành độ của điểm cuối (điểm góc trên, bên trái có toạ độ 0, 0).

Ví dụ:


```python
lines = cv2.HoughLinesP(edges, 2, np.pi/2, 500)
lines
```




    array([[[2238, 1100, 2918, 1100],
            [ 200,  537,  776,  537],
            [3500, 2337, 3500,  339],
            ...,
            [1800,  305, 1802,  305],
            [2250,  247, 2259,  247],
            [1531,  362, 1531,  362]]])



Kết quả cho thấy thuật toán tìm ra một đoạn thẳng nằm ngang từ toạ độ (2238, 1100) đến (2918, 1100); một đoạn thẳng nằm ngang nằm từ toạ độ (200, 537) đến (776, 537); một đoạn thẳng nằm dọc từ (3500, 2337) đến (3500, 339).

Dùng `matplotlib`, ta có thể vẽ lại chúng như sau:


```python
plt.figure(figsize=(20,12))
black_img = np.zeros(edges.shape) # Tạo 1 ảnh nền đen
for line in lines[0]:
    cv2.line(black_img, (line[0], line[1]),(line[2], line[3]),(255,255,255), 1) 
plt.imshow(black_img, 'gray')
cv2.imwrite('black_img.jpg', black_img)
```




    True




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_25_1.png)


Từ biểu diễn trên, ta nhận thấy các cạnh được trả lại không toàn vẹn. Hơn nữa, ta còn quan sát thấy một số đường thẳng "lạ" nằm ở vị trí tiêu đề của bảng điểm. Chúng không phải là các đường thẳng mà chỉ là những đường tạo ra bởi các vệt thẳng hàng từ chân các chữ cái trong tiêu đề.

Vì vậy ta sẽ loại những đường nằm ở bên ngoài bằng cách giới hạn các lề trái, phải, trên, dưới và chỉ nhận những đường thẳng nằm trong vùng này.

Trong class đã có các attribute sau:
``` python
    MARGIN_UP = 600
    MARGIN_DOWN = 100
    MARGIN_LEFT = 1140
    MARGIN_RIGHT = 300
```

*Hãy viết instance method **`getHorizontalLines()`** trả lại list tung độ của tất cả các đường nằm ngang được tìm ra từ thuật toán HoughLineP áp dụng lên ảnh các cạnh (output bài 3), sao cho các đường đó cách biên trên tối thiểu `MARGIN_UP` pixels và cách biên dưới tối thiểu `MARGIN_DOWN` pixels. Trong thuật toán HoughLineP, bạn nên sử dụng `MIN_VOTE_HORIZONTAL` làm số vote tối thiểu.*

*Tương tự, viết instance method **`getVerticalLines()`** trả lại list hoành độ của tất cả các đường nằm dọc được tìm ra từ thuật toán HoughLineP áp dụng lên ảnh các cạnh (output bài 3), sao cho các đường đó cách biên trái tối thiểu `MARGIN_LEFT` pixels và cách biên phải tối thiểu `MARGIN_RIGHT` pixels. Trong thuật toán HoughLineP, bạn nên sử dụng `MIN_VOTE_VERTICAL` làm số vote tối thiểu.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
print myform.getHorizontalLines() 
print myform.getVerticalLines() 
```

    [627, 628, 628, 628, 628, 628, 628, 628, 628, 631, 631, 631, 631, 631, 632, 705, 706, 706, 706, 709, 709, 709, 710, 710, 710, 783, 784, 784, 784, 784, 784, 785, 785, 787, 787, 787, 787, 788, 788, 789, 861, 861, 862, 863, 863, 863, 864, 865, 865, 866, 867, 867, 868, 940, 940, 941, 942, 942, 943, 943, 943, 943, 944, 944, 945, 946, 946, 947, 1018, 1019, 1021, 1022, 1023, 1025, 1025, 1096, 1096, 1096, 1096, 1097, 1099, 1100, 1101, 1103, 1103, 1175, 1177, 1177, 1178, 1178, 1178, 1179, 1180, 1181, 1253, 1253, 1254, 1254, 1254, 1255, 1257, 1258, 1258, 1258, 1258, 1331, 1332, 1332, 1332, 1332, 1335, 1335, 1335, 1336, 1336, 1336, 1336, 1409, 1410, 1410, 1410, 1410, 1412, 1412, 1412, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1414, 1414, 1414, 1414, 1487, 1487, 1488, 1488, 1491, 1491, 1492, 1492, 1492, 1565, 1565, 1566, 1566, 1566, 1566, 1567, 1567, 1571, 1572]
    [1143, 1143, 1144, 1144, 1144, 1147, 1147, 1147, 1147, 1148, 1287, 1290, 1458, 1458, 1458, 1458, 1458, 1461, 1461, 1462, 1462, 1462, 1462, 1496, 1676, 1689, 1689, 1689, 1689, 1690, 1690, 1693, 1693, 1693, 1861, 1924, 1924, 1924, 1925, 1925, 1928, 1928, 1928, 1928, 2143, 2232, 2232, 2232, 2233, 2233, 2236, 2236, 2236, 2236, 2236, 2236]
    

### Bài 5 - Nhóm các đường ngang dọc

Đến đây ta đã có được hoành độ và tung độ các đường ngang dọc. Tuy nhiên, ta nhìn thấy chúng lặp lại và lệch sai số, ví dụ các tung độ 627, 628, 631, 632 có thể được xem là của cùng một đường ngang, nhưng bị lặp lại do nét vẽ đường ngang lớn hơn 1 pixel.

Vì vậy ta muốn nhóm chúng thành từng nhóm.

*Hãy viết instance method **`getHorizontalLineGroups()`** để từ output bài 4 (một list các tung độ tăng dần), trả lại một list các list con, sao cho 2 tung độ liền nhau nếu cách nhau không quá `ROW_GROUP_DISTANCE` (10 pixels) thì rơi vào cùng một list con.*

*Tương tự, hãy viết instance method **`getVerticalLineGroups()`** để từ output bài 4 (một list các hoành độ tăng dần), trả lại một list các list con, sao cho 2 hoành độ liền nhau nếu cách nhau không quá `COLUMN_GROUP_DISTANCE` (10 pixels) thì rơi vào cùng một list con.*

Đoạn code dưới đây giúp test hàm của bạn


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
print myform.getHorizontalLineGroups() 
print ""
print myform.getVerticalLineGroups() 
```

    [[627, 628, 628, 628, 628, 628, 628, 628, 628, 631, 631, 631, 631, 631, 632], [705, 706, 706, 706, 709, 709, 709, 710, 710, 710], [783, 784, 784, 784, 784, 784, 785, 785, 787, 787, 787, 787, 788, 788, 789], [861, 861, 862, 863, 863, 863, 864, 865, 865, 866, 867, 867, 868], [940, 940, 941, 942, 942, 943, 943, 943, 943, 944, 944, 945, 946, 946, 947], [1018, 1019, 1021, 1022, 1023, 1025, 1025], [1096, 1096, 1096, 1096, 1097, 1099, 1100, 1101, 1103, 1103], [1175, 1177, 1177, 1178, 1178, 1178, 1179, 1180, 1181], [1253, 1253, 1254, 1254, 1254, 1255, 1257, 1258, 1258, 1258, 1258], [1331, 1332, 1332, 1332, 1332, 1335, 1335, 1335, 1336, 1336, 1336, 1336], [1409, 1410, 1410, 1410, 1410, 1412, 1412, 1412, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1414, 1414, 1414, 1414], [1487, 1487, 1488, 1488, 1491, 1491, 1492, 1492, 1492], [1565, 1565, 1566, 1566, 1566, 1566, 1567, 1567, 1571, 1572]]
    
    [[1143, 1143, 1144, 1144, 1144, 1147, 1147, 1147, 1147, 1148], [1287, 1290], [1458, 1458, 1458, 1458, 1458, 1461, 1461, 1462, 1462, 1462, 1462], [1496], [1676], [1689, 1689, 1689, 1689, 1690, 1690, 1693, 1693, 1693], [1861], [1924, 1924, 1924, 1925, 1925, 1928, 1928, 1928, 1928], [2143], [2232, 2232, 2232, 2233, 2233, 2236, 2236, 2236, 2236, 2236, 2236]]
    

Biểu diễn trên hình ảnh (bạn cần tự viết được đoạn code này)


```python
plt.figure(figsize=(40,20))
black_img = np.zeros(edges.shape) # Tạo 1 ảnh nền đen
for group in myform.getHorizontalLineGroups():
    for line in group:
        cv2.line(black_img, (0, line), (myform.img.shape[1], line),(255,255,255), 1) 
for group in myform.getVerticalLineGroups():
    for line in group:
        cv2.line(black_img, (line, 0), (line, myform.img.shape[0]),(255,255,255), 1) 
plt.imshow(black_img, 'gray')
```




    <matplotlib.image.AxesImage at 0xf8bff60>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_31_1.png)


### Bài 6 - Vấn đề của hình ảnh

Sau khi khảo sát các hình ảnh, ta phát hiện ra một vấn đề như sau: một số vị trí vốn không có đường ngang dọc nào, nhưng do lỗi của máy in hoặc máy scan, hoặc do ảnh được scan từ giấy A4 in 2 mặt có độ dài mỏng, nên 1 số đường ngang dọc "lỗi" xuất hiện trong output.

Ví dụ với output của bài 5, các đường nằm ngang không bị lỗi, nhưng các đường nằm dọc tại vị trí \[1287, 1290\], \[1496\], \[1676\], \[2143\] là các đường lỗi, không ứng với các đường kẻ của bảng điểm, mà ứng với 1 vệt bất thường do lỗi từ giấy, máy in hay máy scan.

Khảo sát cho thấy lỗi thuộc về máy in do việc in được thực hiện bằng cách quét theo các đường dọc, và một số đường không được quét.

Vì vậy, ta cần xử lí thêm hàm ở bài 5 để thu được các đường kẻ thực sự của bảng điểm. Khảo sát (từ một dự án khác) cho thấy các đường kẻ thực sự sẽ ứng với 1 list con gồm từ 3 phần tử trở lên, các đường giả sẽ ứng với 1 list con gồm 1 hoặc 2 phần tử.

*Hãy viết instance method **`getCleanHorizontalLineGroups()`** để trả lại 1 list các list con như output bài 5, nhưng chỉ giữ lại những list con gồm từ 3 phần tử trở lên. Tương tự với instance method **`getCleanVerticalLineGroups()`***

Chú ý rằng cách giải quyết của ta chỉ mang tính cá biệt (ta biết tất cả các ảnh được in và scan từ cùng máy). Việc xử lí chính xác hơn nằm ngoài mục đích của TD.

Đoạn code dưới đây giúp test hàm của bạn.


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
print myform.getCleanHorizontalLineGroups() 
print ""
print myform.getCleanVerticalLineGroups() 
```

    [[627, 628, 628, 628, 628, 628, 628, 628, 628, 631, 631, 631, 631, 631, 632], [705, 706, 706, 706, 709, 709, 709, 710, 710, 710], [783, 784, 784, 784, 784, 784, 785, 785, 787, 787, 787, 787, 788, 788, 789], [861, 861, 862, 863, 863, 863, 864, 865, 865, 866, 867, 867, 868], [940, 940, 941, 942, 942, 943, 943, 943, 943, 944, 944, 945, 946, 946, 947], [1018, 1019, 1021, 1022, 1023, 1025, 1025], [1096, 1096, 1096, 1096, 1097, 1099, 1100, 1101, 1103, 1103], [1175, 1177, 1177, 1178, 1178, 1178, 1179, 1180, 1181], [1253, 1253, 1254, 1254, 1254, 1255, 1257, 1258, 1258, 1258, 1258], [1331, 1332, 1332, 1332, 1332, 1335, 1335, 1335, 1336, 1336, 1336, 1336], [1409, 1410, 1410, 1410, 1410, 1412, 1412, 1412, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1413, 1414, 1414, 1414, 1414], [1487, 1487, 1488, 1488, 1491, 1491, 1492, 1492, 1492], [1565, 1565, 1566, 1566, 1566, 1566, 1567, 1567, 1571, 1572]]
    
    [[1143, 1143, 1144, 1144, 1144, 1147, 1147, 1147, 1147, 1148], [1458, 1458, 1458, 1458, 1458, 1461, 1461, 1462, 1462, 1462, 1462], [1689, 1689, 1689, 1689, 1690, 1690, 1693, 1693, 1693], [1924, 1924, 1924, 1925, 1925, 1928, 1928, 1928, 1928], [2232, 2232, 2232, 2233, 2233, 2236, 2236, 2236, 2236, 2236, 2236]]
    

Biểu diễn trên hình ảnh:


```python
plt.figure(figsize=(40,20))
black_img = np.zeros(edges.shape) # Tạo 1 ảnh nền đen
for group in myform.getCleanHorizontalLineGroups():
    for line in group:
        cv2.line(black_img, (0, line), (myform.img.shape[1], line),(255,255,255), 1) 
for group in myform.getCleanVerticalLineGroups():
    for line in group:
        cv2.line(black_img, (line, 0), (line, myform.img.shape[0]),(255,255,255), 1) 
plt.imshow(black_img, 'gray')
cv2.imwrite('black_img.jpg', black_img)
```




    True




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_35_1.png)


### Bài 7 - Xác định cạnh của các ô điểm

Đến đây ta đã có các nhóm đường ngang dọc, chúng cho phép xác định toạ độ các ô điểm. Trong mỗi nhóm dọc, ta sẽ chọn đường ngoài cùng bên phải làm biên trái của ô điểm và đường ngoài cùng bên trái của nhóm dọc tiếp theo làm biên phải của ô điểm.

Ví dụ với `[[1143, 1143, 1144, 1144, 1144, 1147, 1147, 1147, 1147, 1148], [1458, 1458, 1458, 1458, 1458, 1461, 1461, 1462, 1462, 1462, 1462], [1689, 1689, 1689, 1689, 1690, 1690, 1693, 1693, 1693], [1924, 1924, 1924, 1925, 1925, 1928, 1928, 1928, 1928], [2232, 2232, 2232, 2233, 2233, 2236, 2236, 2236, 2236, 2236, 2236]]`, ta xác định biên trái và phải của 4 ô: ô thứ nhất có biên trái 1148 và biên phải 1458, ô thứ hai có biên trái 1462 và biên phải 1689, ô thứ ba 1693 -> 1924, ô thứ tư 1928 -> 2232.

Tương tự ở mỗi nhóm ngang, ta sẽ chọn đường dưới cùng làm biên trên của ô điểm và đường trên cùng ở nhóm tiếp theo làm biên dưới của ô điểm.

*Hãy viết 4 instance method: **`getCellLeftEdges()`**, **`getCellRightEdges()`**, **`getCellTopEdges()`**, **`getCellBottomEdges()`**: method thứ nhất trả lại list hoành độ các biên trái của các ô, method thứ hai trả lại list hoành độ biên phải của các ô, method thứ ba trả lại list tung độ biên trên của các ô, method thứ tư trả lại list tung độ biên dưới của các ô.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
print myform.getCellLeftEdges()
```

    [1148, 1462, 1693, 1928]
    


```python
print myform.getCellRightEdges()
```

    [1458, 1689, 1924, 2232]
    


```python
print myform.getCellTopEdges()
```

    [632, 710, 789, 868, 947, 1025, 1103, 1181, 1258, 1336, 1414, 1492]
    


```python
print myform.getCellBottomEdges()
```

    [705, 783, 861, 940, 1018, 1096, 1175, 1253, 1331, 1409, 1487, 1565]
    

Để cắt ra một ô điểm: ví dụ ô ứng với cột đầu tiên ("Hoàn cảnh") của ứng viên đầu tiên, ta có thể dùng


```python
left, right, top, bottom = myform.getCellLeftEdges(), myform.getCellRightEdges(), myform.getCellTopEdges(), myform.getCellBottomEdges()
my_cell = myform.img[top[0]: bottom[0], left[0]: right[0]]
plt.imshow(my_cell, 'gray')
```




    <matplotlib.image.AxesImage at 0xfaacac8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/output_42_1.png)


### Bài 8 - Cắt các ô điểm

Đến hết bài 7, ta đã biết cách xác định đường biên của các ô điểm cũng như cắt ra ô điểm. Bây giờ, ta sẽ cắt ra tất cả các ô điểm đó và lưu chúng vào các hình ảnh, theo phương pháp sau:

- Đầu tiên, loại bỏ 5 pixels ở 4 mép hình ảnh. Điều này giúp loại bỏ các đường kẻ màu đen của các ô điểm nếu chúng vô tình lọt vào hình ảnh. Mép 5 pixels này được biểu diễn bằng attribute `CELL_MARGIN = 5`.

- Tiếp theo, xét phần hình ảnh còn lại, cắt ra hình chữ nhật con (hình chữ nhật này có các cạnh song song với các đường biên) nhỏ nhất sao cho toàn bộ các pixel đen của hình nằm trong hình chữ nhật con đó. Nói cách khác, cắt ra hình chữ nhật con nhỏ nhất mà các pixel nằm ngoài hình chữ nhật này đều trắng. Thao tác này gọi là "đóng khung" điểm số trong ô.

- Bây giờ, ta đã có một hình chữ nhật con chứa gọn số điểm. Ta resize hình này thành hình có kích thước 28x28 pixels bằng
``` python
cell = cv2.resize(cell, (self.SMALL_SIZE, self.SMALL_SIZE), interpolation = cv2.INTER_LINEAR)
```

- Cuối cùng, ta lưu mỗi điểm số như vậy vào một file trong thư mục `SCORE_DATA = "ScoreData/"`. Tên file được đặt như sau: `[Tên hình ảnh gôc]`\_`[Số thứ tự ứng viên, bắt đầu từ 0]`-`[Mã cột].jpg`; trong đó mã cột như sau: Hoàn cảnh: A; Học tập: B; Ước mơ: C; Điểm cộng: D (`CRITERIA = ["A", "B", "C", "D"]`).

*Hãy viết instance method **`saveCells()`** thực hiện quy trình trên.*

Gợi ý: hàm ngược của **`imread`** là **`imwrite`**.

Sau khi thực hiện đoạn code sau:


```python
myform = EvaluationForm("RawForm/BKDN_1.jpg")
myform.saveCells()
```

thư mục `ScoreData` của bạn cần giống như sau:

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/Fig1.png" width=800></img>

Cuối cùng, chạy đoạn code dưới đây để thực hiện cho tất cả các hình ảnh còn lại.


```python
for imgfile in os.listdir(RAW_DATA):
    form = EvaluationForm(RAW_DATA + imgfile)
    form.saveCells()
```

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson7/TD/Fig2.png" ></img>

# Phần 2 - Preprocessing - Class `Mark`

Ta sẽ xây dựng class **`Mark`**, mỗi instance của class này mô tả cho một tập dữ liệu với một số biến và hàm được định nghĩa sẵn. Chẳng hạn ở phần 2 và 3, ta sẽ tạo một instance mô tả tập dữ liệu lấy từ các phiếu chấm trong thư mục `RawForm`; ở phần 4 ta sẽ tạo một instance khác mô tả tập dữ liệu MNIST.

Cấu trúc class `Mark` như sau:

Các attributes:

- `X`: Một 2D numpy array: Một ma trận mà mỗi hàng là một vector tạo từ hình ảnh, có số chiều 28 x 28 = 784, mỗi thành phần của hàng là 0 hoặc 1.
- `y`: Một 1D numpy array: Nhãn của vector: các số nguyên từ 0 đến 10
- `names`: Một 1D numpy array: Tên tập tin ứng với các vector đó

Các method:

- `__init__(X_folder, y_file)`: Khởi tạo một instance từ thư mục chứa các hình ảnh (như `ScoreData`) và một file chứa nhãn ứng với các vector (như `LabelData/Labels.csv`)
- `getX()`, `gety()`, `getNames()`: các getter cơ bản lấy ra `X`, `y` và `names`
- `draw()`: vẽ minh hoạ một vector
- `filterData()`, `getFilteredX()`, `getFilteredy()`, `getFilteredNames()`: lọc ra các dữ liệu ứng với tiêu chuẩn nào đó để train (ta sẽ không train hết toàn bộ dữ liệu, mà chỉ train chẳng hạn trên các dữ liệu ứng với 2 lớp nào đó như 4 và 5)

Ở phần này ta có thêm dữ liệu từ file csv `LabelData/Labels.csv` đã được nhập tay như sau:


```python
import pandas as pd
y_data = pd.read_csv('LabelData/Labels.csv', header=None)
y_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>BKDN_1_0-A.jpg</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BKDN_1_0-B.jpg</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BKDN_1_0-C.jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BKDN_1_0-D.jpg</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BKDN_1_1-A.jpg</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BKDN_1_1-B.jpg</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BKDN_1_1-C.jpg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BKDN_1_1-D.jpg</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BKDN_1_10-A.jpg</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BKDN_1_10-B.jpg</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



Nó cho biết nhãn của dữ liệu: chẳng hạn ở file "BKHN_1_0-A.jpg" điểm được giám khảo cho là 10. Tên các file đã được xếp theo thứ tự ABC. Cùng với các file output hình ảnh của phần 1, đây sẽ là nguồn dữ liệu cho phần này.

Bạn cũng có thể sử dụng folder `ScoreData_Solution` nếu chưa thành công ở phần 1 trong việc tạo ra folder chứa các hình ảnh chữ số.

### Bài 9 - Khởi tạo instance từ các hình ảnh

Trong class `Mark`, hãy:

*Viết instance method **`__init__(self, X_folder=None, y_file=None)`** nhận đối số **`X_folder`** là thư mục chứa các hình ảnh output của phần 1 (tức `ScoreData`), **`y_file`** là một file csv chứa nhãn của các hình ảnh (tức `LabelData/Labels.csv`) và trả lại một instance chứa các attributes:*

- *`X`: một numpy array có kích thước ($N$, 784) trong đó $N$ là số hình ảnh trong thư mục **`X_folder`**. Mỗi hàng của `X` là một vector được xác định như sau: Đọc hình ảnh tương ứng dưới dạng một numpy array 28x28; Biến các pixel lớn hơn `EvaluationForm.BLACK_WHITE_THRESHOLD = 168` thành trắng và nhỏ hơn giá trị này thành đen; Biến các điểm đen thành 1 và trắng thành 0; Sau đó ghép 28 hàng (mỗi hàng 28 phần tử) thành 1 hàng (784 phần tử) bằng cách lần lượt nối hàng sau vào hàng trước.*
 
- *`y`: một numpy array có kích thước ($N$), theo lí thuyết sẽ bằng số hàng trong file csv **`y_file`**, mỗi phần tử nhận giá trị là số điểm của giám khảo ứng với hình ảnh, tức một số nguyên từ 0 đến 10.*

- *`names`: một numpy array có kích thước ($N$), một phần tử nhận giá trị là tên các file tương ứng với dữ liệu.*

*Nếu **`X_folder`** hoặc **`y_file`** không được khai báo hoặc bằng `None`, thì `self.X`, `self.y` nhận giá trị bằng 'None' tương ứng.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
pd.DataFrame(mark.X).head(5) # X là numpy array, biểu diễn ở DataFrame để dễ nhìn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>774</th>
      <th>775</th>
      <th>776</th>
      <th>777</th>
      <th>778</th>
      <th>779</th>
      <th>780</th>
      <th>781</th>
      <th>782</th>
      <th>783</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>0</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <th>3</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>




```python
pd.DataFrame(mark.y).T # Dạng DataFrame của y, viết theo hàng ngang. Kết quả có thể là số thực hoặc số nguyên, không quan trọng
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>758</th>
      <th>759</th>
      <th>760</th>
      <th>761</th>
      <th>762</th>
      <th>763</th>
      <th>764</th>
      <th>765</th>
      <th>766</th>
      <th>767</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 768 columns</p>
</div>




```python
pd.DataFrame(mark.names).T # Dạng DataFrame của names, viết theo hàng ngang
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>758</th>
      <th>759</th>
      <th>760</th>
      <th>761</th>
      <th>762</th>
      <th>763</th>
      <th>764</th>
      <th>765</th>
      <th>766</th>
      <th>767</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BKDN_1_0-A.jpg</td>
      <td>BKDN_1_0-B.jpg</td>
      <td>BKDN_1_0-C.jpg</td>
      <td>BKDN_1_0-D.jpg</td>
      <td>BKDN_1_1-A.jpg</td>
      <td>BKDN_1_1-B.jpg</td>
      <td>BKDN_1_1-C.jpg</td>
      <td>BKDN_1_1-D.jpg</td>
      <td>BKDN_1_10-A.jpg</td>
      <td>BKDN_1_10-B.jpg</td>
      <td>...</td>
      <td>BKHN_6_7-C.jpg</td>
      <td>BKHN_6_7-D.jpg</td>
      <td>BKHN_6_8-A.jpg</td>
      <td>BKHN_6_8-B.jpg</td>
      <td>BKHN_6_8-C.jpg</td>
      <td>BKHN_6_8-D.jpg</td>
      <td>BKHN_6_9-A.jpg</td>
      <td>BKHN_6_9-B.jpg</td>
      <td>BKHN_6_9-C.jpg</td>
      <td>BKHN_6_9-D.jpg</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 768 columns</p>
</div>



Nếu không nêu rõ thư mục và tập tin chứa điểm, code trả lại các giá trị `None`:


```python
mark = Mark()
mark.X, mark.y, mark.names
```




    (None, None, None)



### Bài 10 - Các getter cơ bản

*Trong class `Mark`, viết các instance method **`getX(self)`, `gety(self)`, `getNames(self)`** trả lại các attributes tương ứng `X`, `y`, `names` của instance `self`.*

Kiểm tra bằng đoạn code dưới đây.


```python
mark = Mark(SCORE_DATA)
print mark.getX()
print mark.gety()
```

    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    None
    

### Bài 11 - Phác hoạ chữ số

*Trong class `Mark`, viết instance method **`draw(self, index)`** phác hoạ dữ liệu thứ `index` (tức hàng thứ `index` của `self.X`) sao cho ta hình dung ra chữ số mà nó thể hiện là gì.* 

*Bạn có thể dùng `matplotlib` hoặc phác hoạ một cách đơn giản bằng một string, ví dụ như minh hoạ dưới đây (các điểm bằng 1 được thể hiện bằng X và 0 bằng khoảng trắng).`*


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
mark.draw(10) # Phác hoạ hình ảnh thứ 10, khi nhìn thấy hình vẽ cần hiểu ngay nó là số 9
```

```
    ------------------------------
    |         XXXXXX             |
    |      XXX   XXX   XXX       |
    |    XXXX          XXXX      |
    |  XXXXXX        XXXXXX      |
    |  XXXXX           XXXX      |
    |  XXX             XXXXX     |
    | XXXX             XXXXX     |
    |XXXX               XXXX     |
    |XXX                XXXX     |
    |XXX                XXXXX    |
    |XXX                 XXXXX   |
    |XXXX                 XXXXX  |
    |XXXXX                XXXXX  |
    | XXXXXXX             XXXXXXX|
    |   XXXXXXXXXX         XXXXXX|
    |      XXXXXXX          XXXXX|
    |                       XXXXX|
    |                       XXXXX|
    |                       XXXXX|
    |                        XXXX|
    |                       XXXXX|
    |                       XXXXX|
    |                      XXXXXX|
    |                    XXXXXXX |
    |                   XXXXXXX  |
    |          XXXXXXXXXXXXXX    |
    |      XX XXXXXXXXXXXX       |
    |      XXXXXXX               |
    ------------------------------
``` 


```python
mark.gety()[10] # Kiểm tra lại bằng nhãn y, kết quả cần bằng 9
```




    9.0



### Bài 12 - Loại bỏ những hình ảnh "chất lượng kém"



Nếu nhìn vào hình ảnh '`BKHN_6_11-D.jpg`' ta sẽ thấy số 9 không được trải rộng ở toàn bộ khung hình vuông, mà do một chấm đen ở viền trái, nên số 9 bị đẩy sang biên phải. Nễu dùng hình ảnh này để train, rất có khả năng ta sẽ làm "nhiễu" mô hình. Đây là ví dụ cho thấy trong tập dự liệu của chúng ta vẫn có những hình ảnh không chất lượng.


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
index = [i for i in range(len(mark.getNames())) if mark.getNames()[i] == "BKHN_6_11-D.jpg"][0] #Tìm chỉ số của hình ảnh
print(mark.getNames()[index])
mark.draw(index)
```

    BKHN_6_11-D.jpg
```
    ------------------------------
    |                         XXX|
    |                        XXX |
    |                       XX   |
    |                       X    |
    |                      XX    |
    |                      X     |
    |                      X   X |
    |                     X    X |
    |                     X    X |
    |                     X   XX |
    |                     X   XX |
    |                     X    X |
    |                     X  X X |
    |                     X XX X |
    |                     XXX  XX|
    |                      X   XX|
    |                          X |
    |                          X |
    |                          X |
    |                          X |
    |                         XX |
    |                         X  |
    |                        XX  |
    |                X       X   |
    |                       XX   |
    |                      XX    |
    |                  X   X     |
    |                    XX      |
    ------------------------------
```
    

Quan sát cho ta thấy những hình ảnh chất lượng phải có số pixel đen đủ lớn. Trường hợp của '`BKHN_6_11-D.jpg`' khiến phần "chữ số" bị thu hẹp nên số pixel đen không nhiều, đây không phải là hình ảnh chất lượng. Chọn `MIN_BLACK = 64`, ta quy ước chỉ dùng những hình ảnh có số pixel đen lớn hơn hoặc bằng `MIN_BLACK` để train.

*Trong class `Mark`, hãy viết instance method **`filterData(self, criterion = "FilterByQuality", args = None)`** nhận đối số **`criterion`** là một str sao cho nếu giá trị của nó bằng string "FilterByQuality" thì instance sẽ khởi tạo các attribute **`X_filtered`**, **`y_filtered`** và **`names_filtered`** tương ứng với những hình ảnh có số pixel đen (bằng 1) lớn hơn hoặc bằng `MIN_BLACK`*.

Trong bài này ta chưa quan tâm đến vai trò của đối số **`args`**.

*Sau đó, viết các getter **`getFilteredX(self)`, `getFilteredy(self)`, `getFilteredNames(self)`** trả lại các attributes **`X_filtered`**, **`y_filtered`** và **`names_filtered`** nói trên.*

Ví dụ, đoạn code sau cần chạy thành công


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
mark.filterData(criterion = "FilterByQuality")
mark.getFilteredX()
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])




```python
print ('BKHN_6_11-D.jpg' in mark.getNames()) #Cần bằng True
print ('BKHN_6_11-D.jpg' in mark.getFilteredNames()) #Sau khi lọc, cần bằng False
```

    True
    False
    

### Bài 13 - Lọc dữ liệu thuộc 2 class
Dữ liệu của chúng ta đến giờ gồm 11 class, từ 0 đến 10. Ta muốn lọc ra các ô điểm thuộc 2 class, chẳng hạn 4 và 5.

*Trong class `Mark`, hãy bổ sung vào instance method **`filterData(self, criterion = "FilterByQuality", args = None)`** sao cho khi đối số **`criterion`** nhận giá trị "FilterByQualityAndScore" và khi đối số **`args`** nhận giá trị là list \[$i$, $j$\] thì instance sẽ khởi tạo các attribute **`X_filtered`**, **`y_filtered`** và **`names_filtered`** tương ứng với những hình ảnh có số pixel đen (bằng 1) lớn hơn hoặc bằng `MIN_BLACK` và có nhãn (`y`) bằng $i$ hoặc $j$*.

Đoạn code dưới đây giúp test hàm của bạn.


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
mark.filterData(criterion = "FilterByQualityAndScore", args = [0, 9])
print(mark.getFilteredX())
mark.getFilteredy() # Kết quả cần chỉ chứa 0 và 9.
```

    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 1 1 0]
     [0 0 0 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    




    array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9., 9., 9.,
           9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 0., 0., 0., 0., 0., 0.,
           9., 0., 9., 9., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9., 9.,
           9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           9., 9., 9., 9., 9., 9., 0., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           9., 9., 0., 9., 9., 0., 9., 9., 0., 9., 0., 9., 9., 0., 0., 0., 9.,
           9., 9., 0., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])




```python
print ('BKHN_6_11-D.jpg' in mark.getFilteredNames()) # Tương tự bài trên, kết quả không chứa hình ảnh chất lượng kém
print ('BKHN_3_1-D.jpg' in mark.getFilteredNames()) # Nhưng chứa hình ảnh chất lượng tốt
```

    False
    True
    

Kết quả cũng cần tốt với việc lọc 3 lớp, 5 lớp, ...


```python
mark.filterData(criterion = "FilterByQualityAndScore", args = [1, 3, 7])
mark.getFilteredy() # Kết quả cần chỉ chứa 1, 3 và 7.
```




    array([7., 7., 7., 7., 7., 7., 7., 7., 1., 7., 1., 7., 1., 1., 7., 3., 1.,
           3., 1., 7., 3., 1., 1., 7., 1., 7., 3., 3., 1., 3., 1., 1., 3., 1.,
           1., 3., 1., 1., 3., 1., 1., 1., 1., 3., 1., 1., 3., 3., 1., 1., 1.,
           3., 1., 1., 3., 1., 1., 7., 7., 7., 7., 7., 7., 7., 3., 3., 7., 7.,
           7., 7., 7., 7., 3., 7., 7., 1., 7., 1., 3., 7., 1., 7., 3., 7., 3.,
           3., 1., 3., 3., 1., 7., 7., 3., 1., 7., 7., 1., 3., 3., 3., 1., 1.,
           1., 3., 3., 3., 3., 1., 1., 3., 1., 1., 1., 1., 3., 1., 3., 3., 3.,
           3., 1., 1., 3., 1., 3., 1., 3., 1., 1., 3., 1., 3., 3., 7., 7., 7.,
           7., 7., 7., 7., 7., 3., 7., 3., 7., 7., 7., 3., 7., 7., 7., 3., 7.,
           7., 7., 7., 7., 7., 7., 3., 7., 3., 3., 7., 1., 7., 1., 7., 1., 7.,
           3., 3., 1., 1., 7., 1., 1., 3., 1., 1., 7., 7., 1., 1., 3., 1., 7.,
           1., 7., 1., 7., 7., 7., 7., 3., 7., 3., 7., 7., 1., 3., 3., 3., 3.,
           7., 7., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 3.,
           7., 7., 3., 7., 3., 7., 7., 7., 1., 3., 7., 7., 3., 7., 1.])



### Bài 14 - Phác hoạ chữ số (tiếp theo)

*Hãy viết hàm **`drawAfterFiltering(self, index)`** phác hoạ hình ảnh thứ **`index`** sau khi lọc (tức **`self.X_filtered[index]`** *

Đoạn code dưới đây giúp test hàm của bạn.


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
mark.filterData(criterion = "FilterByQualityAndScore", args = [0, 9])
mark.drawAfterFiltering(5) # Kết quả cần là một hình vẽ số 0 hoặc 9 dù thay index trong ngoặc đơn bởi bất kì chỉ số nào
```

```
    ------------------------------
    |                       XX   |
    |                XXXXXXXXX   |
    |             XXXXXXXXXXXXX  |
    |            XXXXXXXXXXXXXX  |
    |           XXXXXXXXXXXXXX   |
    |         XXXXXXX     XXXX   |
    |        XXXXX       XXXXX   |
    |        XXXXX       XXXXX   |
    |        XXX           XXX   |
    |       XXXX            XXX  |
    |       XXX             XXX  |
    |       XXXX            XXX  |
    |       XXXXXXX  XX     XXX  |
    |        XXXXXXXXXX     XXXX |
    |         XXXXXXXX      XXXX |
    |                       XXXX |
    |                       XXXX |
    |                       XXXX |
    |                       XXXXX|
    |                       XXXX |
    |                       XXXX |
    |                       XXXX |
    |                      XXXXX |
    |                     XXXXXX |
    |         XX   XXXXXXXXXXXX  |
    |XXX      XXXXXXXXXXXXXXXX   |
    | XX      XXX   XXXXXXXX     |
    |     XXXXXXXXX  XXXX        |
    ------------------------------
```
    

## Phần 3 - Tiến hành phân loại

Đến phần này ta sẽ dùng các thuật toán đã học để phân loại các cặp chữ số. Ta sẽ bổ sung vào class `Mark` các hàm sau.

- `getTrainTestScore(model, i, j)`: train và test với một mô hình nào đó và với dữ liệu ứng với 2 chữ số `i`, `j`, đánh giá kết quả dựa trên tỉ lệ dự đoán chính sách
- `massiveTrainTest(models)`: train và test trên dữ liệu ứng tất cả các cặp chữ số và với một tập các mô hình để so sánh kết quả.
- `getWrongCase(model, i, j)`: trả lại những hình ảnh dễ bị phân loại nhầm.

Ta sẽ sử dụng các mô hình đã học trong bài (riêng "Support Vector Machine" hay "SVM" sẽ được giới thiệu sau). 

```python
MODELS = [MyPerceptron, MyLDA, MyQDA, MyLogisticRegression, MyGaussianNaiveBayes, MyKNN5, MyKNN1, MySVM, MyLinearRegression]
MODEL_NAMES = ["Perceptron", "LDA", "QDA", "LogisticRegression", "GaussianNaiveBayes", "KNN5", "KNN1", "SVM", "LinearRegression"]
```

Ngoài ra ta cũng sẽ thử dùng `LinearRegression` cho bài toán classification để xem hệ quả của việc chọn mô hình không thích hợp như thế nào.

Các hàm trong `MODELS` hiện đang trả lại giá trị `None` và cần được sửa lại.

### Bài 15 - Implement các mô hình

*Hãy sửa chữa các hàm **`MyPerceptron, MyLDA, MyQDA, MyLogisticRegression, MyGaussianNaiveBayes, MyKNN5, MyKNN1, MySVM, MyLinearRegression`** để khởi tạo chính xác các mô hình. Nếu các mô hình có tham số, bạn được chọn tham số phù hợp. Riêng **`MyKNN3`** và **`MyKNN1`** lấy $K=5$, $K=1$*.

Ví dụ, mô hình Perceptron được viết như sau
```python
from sklearn.linear_model import Perceptron

def MyPerceptron():
    return Perceptron()
```
Bạn khởi tạo tương tự cho các mô hình còn lại. Ví dụ, đoạn code dưới đây cần chạy đúng.


```python
model = MyLogisticRegression()
mark = Mark(SCORE_DATA, LABEL_DATA)
mark.filterData(criterion = "FilterByQualityAndScore", args = [0, 9]) # Lọc các dữ liệu thuộc lớp 0, 9
model.fit(mark.getFilteredX(), mark.getFilteredy()) # Mô hình cần phải train được dữ liệu
model.predict(mark.getFilteredX()) # Rồi dự đoán được dữ liệu
```




    array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9., 9., 9.,
           9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 0., 0., 0., 0., 0., 0.,
           9., 0., 9., 9., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9., 9.,
           9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           9., 9., 9., 9., 9., 9., 0., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           9., 9., 0., 9., 9., 0., 9., 9., 0., 9., 0., 9., 9., 0., 0., 0., 9.,
           9., 9., 0., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
           9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])



### Bài 16 - Độ chính xác của từng mô hình

Ta sẽ đo độ chính xác của từng mô hình bằng cách dùng Cross Validation, chia dữ liệu thành 3 phần, lần lượt train trên 2 tập và test trên tập còn lại, tính tỉ số dữ liệu được dự đoán đúng trên tập test (một số thực nằm giữa 0, 1), rồi lấy trung bình cộng của 3 tỉ số đó và xem nó là **độ chính xác của mô hình**. Do yếu tố ngẫu nhiên trong cách chia của Cross Validation, độ chính xác này có thể dao động nhẹ.

*Trong class `Mark`, hãy viết instance method **`getTrainTestScore(model, i, j)`** nhận đối số **`model`** là một trong các hàm trong `MODELS`, **`i`, `j`** là 2 số nguyên chỉ 2 class đang phân loại, và trả lại **độ chính xác của mô hình** trên tập dữ liệu đang dùng.*

Đoạn code dưới đây giúp test hàm của bạn (kết quả của bạn có thể khác nhưng không sai lệch quá nhiều, và phải có LogisticRegression cao hơn QDA và LinearRegression)


```python
mark = Mark(SCORE_DATA, LABEL_DATA)
mark.getTrainTestScore(MyLogisticRegression, 3, 5) # Độ chính xác khi dùng LogisticRegression
```




    0.9855072463768115




```python
mark.getTrainTestScore(MyQDA, 3, 5)
```

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\lib\site-packages\sklearn\discriminant_analysis.py:682: UserWarning: Variables are collinear
      warnings.warn("Variables are collinear")
    




    0.6096837944664032




```python
mark.getTrainTestScore(MyLinearRegression, 3, 5)
```




    0.9135283243323652



### Bài 17 - Lập bảng độ chính xác của tất cả các mô hình trên mọi lớp.

*Trong class `Mark`, hãy viết instance method **`massiveTrainTest(self, models)`** nhận đối số **`models`** là list các mô hình (như `MODELS`), rồi tính tất cả các độ chính xác của các mô hình trên cặp 2 lớp, và trả lại kết quả dưới dạng một numpy array 3 chiều `R` trong đó `R[i, j, m]` là độ chính xác khi dùng mô hình thứ `m` trên các lớp `i`, `j`.*

Đoạn code dưới đây giúp test hàm của bạn.



```python
mark = Mark(SCORE_DATA, LABEL_DATA)
MODELS = [MyPerceptron, MyLDA, MyQDA, MyLogisticRegression, MyGaussianNaiveBayes, MyKNN5, MyKNN1, MySVM, MyLinearRegression]
result = mark.massiveTrainTest(MODELS)
result.shape
```

    Classifying 0 and 1
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.perceptron.Perceptron'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Classifying 0 and 2
    Classifying 0 and 3
    Classifying 0 and 4
    Classifying 0 and 5
    Classifying 0 and 6
    Classifying 0 and 7
    Classifying 0 and 8
    Classifying 0 and 9
    Classifying 0 and 10
    Classifying 1 and 2
    Classifying 1 and 3
    Classifying 1 and 4
    Classifying 1 and 5
    Classifying 1 and 6
    Classifying 1 and 7
    Classifying 1 and 8
    Classifying 1 and 9
    Classifying 1 and 10
    Classifying 2 and 3
    Classifying 2 and 4
    Classifying 2 and 5
    Classifying 2 and 6
    Classifying 2 and 7
    Classifying 2 and 8
    Classifying 2 and 9
    Classifying 2 and 10
    Classifying 3 and 4
    Classifying 3 and 5
    Classifying 3 and 6
    Classifying 3 and 7
    Classifying 3 and 8
    Classifying 3 and 9
    Classifying 3 and 10
    Classifying 4 and 5
    Classifying 4 and 6
    Classifying 4 and 7
    Classifying 4 and 8
    Classifying 4 and 9
    Classifying 4 and 10
    Classifying 5 and 6
    Classifying 5 and 7
    Classifying 5 and 8
    Classifying 5 and 9
    Classifying 5 and 10
    Classifying 6 and 7
    Classifying 6 and 8
    Classifying 6 and 9
    Classifying 6 and 10
    Classifying 7 and 8
    Classifying 7 and 9
    Classifying 7 and 10
    Classifying 8 and 9
    Classifying 8 and 10
    Classifying 9 and 10
    




    (11L, 11L, 9L)



Ta cũng có thể dùng DataFrame để xem độ chính xác của 9 mô hình. Ví dụ, dùng đoạn code sau:


```python
import pandas as pd
MODEL_NAMES = ["Perceptron", "LDA", "QDA", "LogisticRegression", "GaussianNaiveBayes", "KNN5", "KNN1", "SVM", "LinearRegression"]
frame = []
group_identifier = []
for i in range(11):
    for j in range(i+1, 11):
        group_identifier.append("%d vs %d" % (i, j))
        frame.append(result[i, j])
frame = np.array(frame)
data = pd.DataFrame(frame, index = group_identifier, columns = MODEL_NAMES)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Perceptron</th>
      <th>LDA</th>
      <th>QDA</th>
      <th>LogisticRegression</th>
      <th>GaussianNaiveBayes</th>
      <th>KNN5</th>
      <th>KNN1</th>
      <th>SVM</th>
      <th>LinearRegression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 vs 1</th>
      <td>0.972222</td>
      <td>0.981481</td>
      <td>0.653021</td>
      <td>0.972222</td>
      <td>0.870858</td>
      <td>0.990741</td>
      <td>0.990741</td>
      <td>0.972222</td>
      <td>0.859739</td>
    </tr>
    <tr>
      <th>0 vs 2</th>
      <td>0.978831</td>
      <td>1.000000</td>
      <td>0.711873</td>
      <td>0.989247</td>
      <td>0.978136</td>
      <td>1.000000</td>
      <td>0.978136</td>
      <td>0.989247</td>
      <td>0.911993</td>
    </tr>
    <tr>
      <th>0 vs 3</th>
      <td>0.990991</td>
      <td>1.000000</td>
      <td>0.685278</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.923986</td>
    </tr>
    <tr>
      <th>0 vs 4</th>
      <td>0.988095</td>
      <td>1.000000</td>
      <td>0.654458</td>
      <td>1.000000</td>
      <td>0.987654</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.943274</td>
    </tr>
    <tr>
      <th>0 vs 5</th>
      <td>0.971693</td>
      <td>0.981217</td>
      <td>0.796697</td>
      <td>0.981217</td>
      <td>0.971958</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.981217</td>
      <td>0.886036</td>
    </tr>
    <tr>
      <th>0 vs 6</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.782609</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.908311</td>
    </tr>
    <tr>
      <th>0 vs 7</th>
      <td>1.000000</td>
      <td>0.985500</td>
      <td>0.762041</td>
      <td>0.992908</td>
      <td>0.985185</td>
      <td>0.992908</td>
      <td>0.992908</td>
      <td>0.992908</td>
      <td>0.913655</td>
    </tr>
    <tr>
      <th>0 vs 8</th>
      <td>0.976190</td>
      <td>1.000000</td>
      <td>0.569048</td>
      <td>1.000000</td>
      <td>0.952381</td>
      <td>0.988095</td>
      <td>0.988095</td>
      <td>1.000000</td>
      <td>0.916393</td>
    </tr>
    <tr>
      <th>0 vs 9</th>
      <td>0.986928</td>
      <td>0.993333</td>
      <td>0.676078</td>
      <td>0.993333</td>
      <td>0.973464</td>
      <td>0.973595</td>
      <td>1.000000</td>
      <td>0.993333</td>
      <td>0.865277</td>
    </tr>
    <tr>
      <th>0 vs 10</th>
      <td>0.973909</td>
      <td>0.973909</td>
      <td>0.757310</td>
      <td>1.000000</td>
      <td>0.903509</td>
      <td>0.991453</td>
      <td>0.991453</td>
      <td>0.982681</td>
      <td>0.893883</td>
    </tr>
    <tr>
      <th>1 vs 2</th>
      <td>0.934127</td>
      <td>0.983333</td>
      <td>0.594715</td>
      <td>0.966667</td>
      <td>0.803407</td>
      <td>0.991667</td>
      <td>0.991667</td>
      <td>0.983333</td>
      <td>0.839083</td>
    </tr>
    <tr>
      <th>1 vs 3</th>
      <td>0.985185</td>
      <td>0.992593</td>
      <td>0.630486</td>
      <td>0.985185</td>
      <td>0.839894</td>
      <td>0.992593</td>
      <td>0.992593</td>
      <td>0.992593</td>
      <td>0.867257</td>
    </tr>
    <tr>
      <th>1 vs 4</th>
      <td>0.938347</td>
      <td>0.973447</td>
      <td>0.632120</td>
      <td>0.973684</td>
      <td>0.832099</td>
      <td>0.990991</td>
      <td>0.982219</td>
      <td>0.990991</td>
      <td>0.857936</td>
    </tr>
    <tr>
      <th>1 vs 5</th>
      <td>0.956515</td>
      <td>0.978408</td>
      <td>0.593919</td>
      <td>0.941861</td>
      <td>0.898537</td>
      <td>0.985500</td>
      <td>0.992593</td>
      <td>0.977939</td>
      <td>0.852368</td>
    </tr>
    <tr>
      <th>1 vs 6</th>
      <td>0.975750</td>
      <td>0.975750</td>
      <td>0.586861</td>
      <td>0.987875</td>
      <td>0.884480</td>
      <td>0.993827</td>
      <td>0.993827</td>
      <td>0.987654</td>
      <td>0.854806</td>
    </tr>
    <tr>
      <th>1 vs 7</th>
      <td>0.958535</td>
      <td>0.982243</td>
      <td>0.577708</td>
      <td>0.958535</td>
      <td>0.852749</td>
      <td>0.988091</td>
      <td>0.988091</td>
      <td>0.982243</td>
      <td>0.825034</td>
    </tr>
    <tr>
      <th>1 vs 8</th>
      <td>0.956140</td>
      <td>0.973684</td>
      <td>0.542982</td>
      <td>0.938596</td>
      <td>0.860965</td>
      <td>0.973684</td>
      <td>0.964912</td>
      <td>0.965351</td>
      <td>0.758436</td>
    </tr>
    <tr>
      <th>1 vs 9</th>
      <td>0.972495</td>
      <td>0.978051</td>
      <td>0.480965</td>
      <td>0.983515</td>
      <td>0.884426</td>
      <td>0.983515</td>
      <td>0.983515</td>
      <td>0.983515</td>
      <td>0.820871</td>
    </tr>
    <tr>
      <th>1 vs 10</th>
      <td>0.958617</td>
      <td>0.993056</td>
      <td>0.648243</td>
      <td>0.986253</td>
      <td>0.875709</td>
      <td>0.993056</td>
      <td>0.993056</td>
      <td>0.993056</td>
      <td>0.864646</td>
    </tr>
    <tr>
      <th>2 vs 3</th>
      <td>0.991870</td>
      <td>0.991870</td>
      <td>0.653117</td>
      <td>1.000000</td>
      <td>0.967063</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.882114</td>
    </tr>
    <tr>
      <th>2 vs 4</th>
      <td>0.979146</td>
      <td>0.989247</td>
      <td>0.577387</td>
      <td>0.989247</td>
      <td>0.949495</td>
      <td>0.979146</td>
      <td>0.989899</td>
      <td>0.979146</td>
      <td>0.871864</td>
    </tr>
    <tr>
      <th>2 vs 5</th>
      <td>0.932458</td>
      <td>0.983323</td>
      <td>0.660830</td>
      <td>0.958516</td>
      <td>0.908901</td>
      <td>0.983323</td>
      <td>1.000000</td>
      <td>0.958516</td>
      <td>0.834505</td>
    </tr>
    <tr>
      <th>2 vs 6</th>
      <td>0.972500</td>
      <td>0.986111</td>
      <td>0.558668</td>
      <td>0.993056</td>
      <td>0.965975</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.841349</td>
    </tr>
    <tr>
      <th>2 vs 7</th>
      <td>0.973056</td>
      <td>0.979859</td>
      <td>0.628651</td>
      <td>0.986395</td>
      <td>0.946112</td>
      <td>0.973323</td>
      <td>0.986661</td>
      <td>0.986395</td>
      <td>0.856495</td>
    </tr>
    <tr>
      <th>2 vs 8</th>
      <td>0.949142</td>
      <td>0.989583</td>
      <td>0.643791</td>
      <td>0.969363</td>
      <td>0.857880</td>
      <td>0.938131</td>
      <td>0.958965</td>
      <td>1.000000</td>
      <td>0.887411</td>
    </tr>
    <tr>
      <th>2 vs 9</th>
      <td>0.981818</td>
      <td>0.981706</td>
      <td>0.500112</td>
      <td>0.987879</td>
      <td>0.981706</td>
      <td>0.981706</td>
      <td>0.981706</td>
      <td>0.987879</td>
      <td>0.876720</td>
    </tr>
    <tr>
      <th>2 vs 10</th>
      <td>0.944629</td>
      <td>0.952750</td>
      <td>0.664083</td>
      <td>0.976375</td>
      <td>0.921189</td>
      <td>0.984312</td>
      <td>0.984496</td>
      <td>0.976375</td>
      <td>0.881499</td>
    </tr>
    <tr>
      <th>3 vs 4</th>
      <td>0.973684</td>
      <td>0.955166</td>
      <td>0.571150</td>
      <td>0.973197</td>
      <td>0.973684</td>
      <td>0.981969</td>
      <td>0.991228</td>
      <td>0.973197</td>
      <td>0.905475</td>
    </tr>
    <tr>
      <th>3 vs 5</th>
      <td>0.956522</td>
      <td>0.978261</td>
      <td>0.609684</td>
      <td>0.985507</td>
      <td>0.971014</td>
      <td>0.985507</td>
      <td>0.985507</td>
      <td>0.985507</td>
      <td>0.913528</td>
    </tr>
    <tr>
      <th>3 vs 6</th>
      <td>0.993711</td>
      <td>1.000000</td>
      <td>0.636296</td>
      <td>1.000000</td>
      <td>0.987879</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.919030</td>
    </tr>
    <tr>
      <th>3 vs 7</th>
      <td>0.951279</td>
      <td>0.981922</td>
      <td>0.552910</td>
      <td>0.987875</td>
      <td>0.964286</td>
      <td>0.981922</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.925319</td>
    </tr>
    <tr>
      <th>3 vs 8</th>
      <td>0.991228</td>
      <td>0.965125</td>
      <td>0.533720</td>
      <td>0.965362</td>
      <td>0.893989</td>
      <td>0.947806</td>
      <td>0.947806</td>
      <td>0.973909</td>
      <td>0.857153</td>
    </tr>
    <tr>
      <th>3 vs 9</th>
      <td>0.955556</td>
      <td>0.983333</td>
      <td>0.548023</td>
      <td>0.983333</td>
      <td>0.955461</td>
      <td>0.988889</td>
      <td>0.994444</td>
      <td>0.988889</td>
      <td>0.815590</td>
    </tr>
    <tr>
      <th>3 vs 10</th>
      <td>1.000000</td>
      <td>0.993056</td>
      <td>0.615396</td>
      <td>1.000000</td>
      <td>0.936613</td>
      <td>0.993056</td>
      <td>0.993056</td>
      <td>0.993056</td>
      <td>0.922689</td>
    </tr>
    <tr>
      <th>4 vs 5</th>
      <td>0.973197</td>
      <td>1.000000</td>
      <td>0.572612</td>
      <td>1.000000</td>
      <td>0.946394</td>
      <td>0.991228</td>
      <td>0.991228</td>
      <td>1.000000</td>
      <td>0.891142</td>
    </tr>
    <tr>
      <th>4 vs 6</th>
      <td>0.985346</td>
      <td>1.000000</td>
      <td>0.484229</td>
      <td>1.000000</td>
      <td>0.978100</td>
      <td>0.985507</td>
      <td>0.992754</td>
      <td>1.000000</td>
      <td>0.847233</td>
    </tr>
    <tr>
      <th>4 vs 7</th>
      <td>0.951087</td>
      <td>0.971920</td>
      <td>0.492150</td>
      <td>0.964976</td>
      <td>0.873188</td>
      <td>0.979167</td>
      <td>0.972222</td>
      <td>0.964976</td>
      <td>0.789088</td>
    </tr>
    <tr>
      <th>4 vs 8</th>
      <td>0.967384</td>
      <td>0.989247</td>
      <td>0.588592</td>
      <td>1.000000</td>
      <td>0.911445</td>
      <td>0.977395</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.816745</td>
    </tr>
    <tr>
      <th>4 vs 9</th>
      <td>0.948592</td>
      <td>0.974359</td>
      <td>0.497360</td>
      <td>0.974359</td>
      <td>0.967949</td>
      <td>0.974359</td>
      <td>0.980769</td>
      <td>0.974359</td>
      <td>0.863937</td>
    </tr>
    <tr>
      <th>4 vs 10</th>
      <td>0.916453</td>
      <td>0.966667</td>
      <td>0.564957</td>
      <td>0.991453</td>
      <td>0.882265</td>
      <td>0.991667</td>
      <td>0.983333</td>
      <td>0.983120</td>
      <td>0.790976</td>
    </tr>
    <tr>
      <th>5 vs 6</th>
      <td>0.938034</td>
      <td>0.975309</td>
      <td>0.691053</td>
      <td>0.987654</td>
      <td>0.963412</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.993827</td>
      <td>0.853772</td>
    </tr>
    <tr>
      <th>5 vs 7</th>
      <td>0.957892</td>
      <td>0.975970</td>
      <td>0.584656</td>
      <td>0.970018</td>
      <td>0.940035</td>
      <td>0.982143</td>
      <td>0.988095</td>
      <td>0.969797</td>
      <td>0.846725</td>
    </tr>
    <tr>
      <th>5 vs 8</th>
      <td>0.947119</td>
      <td>0.964675</td>
      <td>0.612254</td>
      <td>0.955903</td>
      <td>0.903272</td>
      <td>0.947119</td>
      <td>0.973672</td>
      <td>0.964675</td>
      <td>0.831383</td>
    </tr>
    <tr>
      <th>5 vs 9</th>
      <td>0.972222</td>
      <td>0.972222</td>
      <td>0.480414</td>
      <td>0.972222</td>
      <td>0.944444</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.972222</td>
      <td>0.873473</td>
    </tr>
    <tr>
      <th>5 vs 10</th>
      <td>0.972074</td>
      <td>0.986111</td>
      <td>0.587914</td>
      <td>0.986111</td>
      <td>0.930408</td>
      <td>0.986111</td>
      <td>0.986111</td>
      <td>0.986111</td>
      <td>0.901412</td>
    </tr>
    <tr>
      <th>6 vs 7</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.646100</td>
      <td>1.000000</td>
      <td>0.994792</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.898540</td>
    </tr>
    <tr>
      <th>6 vs 8</th>
      <td>0.985809</td>
      <td>0.985507</td>
      <td>0.484903</td>
      <td>0.992754</td>
      <td>0.957428</td>
      <td>0.978261</td>
      <td>0.971014</td>
      <td>1.000000</td>
      <td>0.871670</td>
    </tr>
    <tr>
      <th>6 vs 9</th>
      <td>0.990267</td>
      <td>0.990267</td>
      <td>0.634839</td>
      <td>0.990338</td>
      <td>0.985436</td>
      <td>0.995098</td>
      <td>0.995098</td>
      <td>0.985436</td>
      <td>0.919154</td>
    </tr>
    <tr>
      <th>6 vs 10</th>
      <td>0.982352</td>
      <td>0.988200</td>
      <td>0.704574</td>
      <td>0.988200</td>
      <td>0.952799</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.988200</td>
      <td>0.839437</td>
    </tr>
    <tr>
      <th>7 vs 8</th>
      <td>0.971921</td>
      <td>0.979013</td>
      <td>0.623167</td>
      <td>0.965697</td>
      <td>0.910561</td>
      <td>0.923008</td>
      <td>0.965272</td>
      <td>0.986105</td>
      <td>0.839478</td>
    </tr>
    <tr>
      <th>7 vs 9</th>
      <td>0.971360</td>
      <td>0.980952</td>
      <td>0.612422</td>
      <td>0.980952</td>
      <td>0.980952</td>
      <td>0.976190</td>
      <td>0.980952</td>
      <td>0.976190</td>
      <td>0.851125</td>
    </tr>
    <tr>
      <th>7 vs 10</th>
      <td>0.971264</td>
      <td>0.988506</td>
      <td>0.699234</td>
      <td>0.988506</td>
      <td>0.959669</td>
      <td>0.988506</td>
      <td>0.982759</td>
      <td>0.988506</td>
      <td>0.839851</td>
    </tr>
    <tr>
      <th>8 vs 9</th>
      <td>0.949081</td>
      <td>0.955612</td>
      <td>0.496734</td>
      <td>0.974722</td>
      <td>0.930213</td>
      <td>0.967949</td>
      <td>0.974480</td>
      <td>0.968312</td>
      <td>0.815583</td>
    </tr>
    <tr>
      <th>8 vs 10</th>
      <td>0.966870</td>
      <td>0.983537</td>
      <td>0.676016</td>
      <td>1.000000</td>
      <td>0.958740</td>
      <td>0.983333</td>
      <td>0.991667</td>
      <td>0.991870</td>
      <td>0.854297</td>
    </tr>
    <tr>
      <th>9 vs 10</th>
      <td>0.946237</td>
      <td>0.978495</td>
      <td>0.586022</td>
      <td>0.978495</td>
      <td>0.935484</td>
      <td>0.983871</td>
      <td>0.967742</td>
      <td>0.978495</td>
      <td>0.804352</td>
    </tr>
  </tbody>
</table>
</div>



### Nhận xét kết quả:

*Hãy thảo luận các câu hỏi sau:*

*1. Những mô hình nào chạy không tốt? Vì sao?*

*2. Những mô hình nào chạy tốt? Trong số chúng, những mô hình nào có độ phức tạp về thời gian lớn nhất?*

*3. Những cặp chữ số nào nhìn chung có kết quả kém hơn các cặp khác?*

### Bài 18 - Quan sát những chữ số dễ nhầm lẫn

Ta muốn tìm lại những hình ảnh bị đoán nhầm để phân tích. Việc train test tự động bằng cross validation của scikit learn không giúp tìm được những dữ liệu vị test nhầm vì chỉ cho ta thông tin về score của mô hình. Do đó, ta có thể thực hiện bằng cách dùng nửa đầu số dữ liệu để train và nửa sau để test, sau đó lấy ra những dữ liệu bị dự đoán sai ở nửa sau. Sau đó, ta dùng nửa sau để train và lấy ra những dữ liệu bị đoán sai ở nửa đầu.

*Trong class `Mark`, viết instance method **`getWrongCase(self, model, i, j)`** nhận đối số **`model`** là mô hình, **`i`, `j`** là 2 số nguyên chỉ các class để phân loại, và trả lại một `set` gồm tên các tập tin bị phân loại nhầm khi train với các hình ảnh thuộc nửa kia của tập dữ liệu.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
mark = Mark()
mark = Mark(SCORE_DATA, LABEL_DATA)
WrongCases = mark.getWrongCase(MyLogisticRegression, 4, 9)
WrongCases
```




    {'BKDN_4_11-A.jpg', 'BKDN_4_2-B.jpg', 'BKHCM_3_8-A.jpg'}




```python
indices = [i for i in range(len(mark.getNames())) if mark.getNames()[i] in WrongCases] #Tìm chỉ số của hình ảnh
for i in indices:
    mark.draw(i)
```

```
    ------------------------------
    |                           X|
    |                            |
    |                            |
    |                            |
    |                            |
    |                            |
    |                            |
    |                            |
    |                            |
    |        XXXX                |
    |        XXXX                |
    |       XX  XX               |
    |      XX    X X             |
    |     XXX    XXXX            |
    |     XX      XX             |
    |     X       XX             |
    |     XX    XXXX             |
    |      XXXXXX  XX            |
    |        XX    XX            |
    |               X            |
    |               XX           |
    |               XX           |
    |               XX           |
    |               XX           |
    |               XX           |
    |X              XX           |
    | XXX     X XX XX            |
    |   XXXXXXXXXXXX             |
    ------------------------------
```
```
    ------------------------------
    |          XXXX              |
    |       XXXXXXXX             |
    |       X     XX             |
    |    X XX     XX             |
    |   XX        XX             |
    |   XX        XX             |
    |  XX        XXX          XXX|
    | XXX        XXX           X |
    | XX        XXXX             |
    |XXX        XXXX             |
    |XXX      XXX XX             |
    |XXX     XX   XX             |
    | XXXXXXXXX   XXX            |
    |  XXXX       XXX            |
    |             XXX            |
    |             XXX            |
    |             XXX            |
    |             XXX            |
    |             XXX            |
    |             XXX            |
    |             XX             |
    |             XX             |
    |            XXX             |
    |            XXX             |
    |           XXX              |
    |          XXX               |
    |         XXX                |
    |        XXX                 |
    ------------------------------
```
```
    ------------------------------
    |              XX            |
    |              XXX           |
    |              XXX           |
    |              XX            |
    |              X             |
    |               X            |
    |                            |
    |              X             |
    |             XX       X     |
    |             XX      XX     |
    |             XX      XXX    |
    |X           XXX      XXX    |
    |            XXX      XX X   |
    |           XXX      XXX X   |
    |           XXX      XX  X   |
    |           XX       XX      |
    |           XX        X      |
    |          XX         X  X XX|
    |          XX        XXXXXXXX|
    |          XXXXXXXXXXXXXXXX  |
    |          XXXXXXXXXXXXXXX   |
    |          XXXXXXXXXXXXXXX   |
    |           XXXXX     XX     |
    |                     X      |
    |                     XX     |
    |                     XX     |
    |                      X     |
    |                      X     |
    ------------------------------
```
    

Đây vẫn là những hình ảnh chất lượng kém dù có số pixel đen đủ lớn, vì nó bị dính các chấm đen không mong muốn. Việc tẩy các chấm đen này có thể được xử lí bởi các thuật toán xử lí ảnh.

## Phần 4 - Train với MNIST

MNIST là một cơ sở dữ liệu gồm 70000 hình ảnh 28x28 đã được vector hoá, biểu diễn các chữ số từ 0 đến 9.

Để tải dữ liệu MNIST, bạn có thể dùng: 

```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='MNIST')
```

Tại đây, dữ liệu sẽ có dạng một dữ liệu `.mat` dùng trong Matlab. Để đọc dữ liệu `.mat` từ file, bạn có thể dùng `scipy.io.loadmat(filename)`. Với MNIST, sau khi đọc, output sẽ là một từ điển có key `data` biểu diễn $X$ (dưới dạng numpy array 784 x 70000) và key `label` biểu diễn $y$ (dưới dạng numpy array 1 x 70000). Vì dữ liệu là hình ảnh nên các phần tử của $X$ là một số từ 0 đến 255. 

### Bài 19 - Đọc MNIST

*Hãy viết instance method **`loadMatData(self, filename)`** đọc dữ liệu trong `filename` có dạng `.mat` và gán $X$, $y$ tương ứng cho `self.X`, `self.y`. $X$ cần được chuyển thành ma trận có các phần tử là 0, 1 theo nguyên tắc nếu giá trị của pixel tương ứng nhỏ hơn hay không nhỏ hơn `Evaluation.BLACK_WHITE_THRESHOLD`. Hãy thay đổi `self.names` để có độ dài bằng độ dài `self.X, self.y`.*

Đoạn code dưới đây test hàm của bạn.


```python
mnist_mark = Mark()
mnist_mark.loadMatData(MNIST_DATA)
mnist_mark.getX()
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)




```python
mnist_mark.gety()
```




    array([0., 0., 0., ..., 9., 9., 9.])




```python
mnist_mark.getX()[0] # Cần là 1 vector có các thành phần trong [0, 1] thay vì [0, 255]
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)




```python
mnist_mark.draw(0) # Method draw cũng cần hoạt động với mnist_mark mà không cẩn sửa chữa gì trong hàm.
```

```
    ------------------------------
    |                            |
    |                            |
    |                            |
    |                            |
    |                 X          |
    |               XXXXX        |
    |              XXXXXX        |
    |             XXXXX XX       |
    |            XXXXXX XX       |
    |           XXXX XX  X       |
    |          XXXX      XX      |
    |          XXX       XX      |
    |        XXX         XXX     |
    |        XX          XXX     |
    |       XXX          XXX     |
    |       XX           XX      |
    |       XX           XX      |
    |       XX          XX       |
    |       X          XX        |
    |       XX       XX          |
    |       XX    XXXX           |
    |       XXXXXXXXX            |
    |       XXXXXXX              |
    |         XXX                |
    |                            |
    |                            |
    |                            |
    |                            |
    ------------------------------
```
    

### Bài 20 - Train và test trong MNIST

*Dùng **`getTrainTestScore`** đã viết ở phần trước để tính độ chính xác khi train dữ liệu bằng các mô hình khác nhau với MNIST với 2 class nào đó, chẳng hạn 4 và 9. Chú ý rằng bây giờ MNIST chỉ có 10 lớp, nên bạn cần sửa chữa hàm `getTrainTestScore` để không bị báo lỗi "không có class 10". Nhận xét về độ chính xác và thời gian chạy của các mô hình. Nếu mô hình có thời gian chạy quá lớn (VD >1 phút) hoặc độ chính xác quá nhỏ, ta sẽ rút nó ra khỏi danh sách mô hình.*


```python
mnist_mark = Mark()
mnist_mark.loadMatData(MNIST_DATA)
import time
for name, model in zip(MODEL_NAMES, MODELS):
    print name
    t = time.time()
    print mnist_mark.getTrainTestScore(model, 4, 9)
    print "Running time: ", time.time() - t, "seconds"
```

    Perceptron
    0.9490559585783673
    Running time:  4.92000007629 seconds
    LDA
    0.9583516522689735
    Running time:  7.55599999428 seconds
    QDA
    0.7376277776121195
    Running time:  8.43000006676 seconds
    LogisticRegression
    0.96117633686006
    Running time:  6.4889998436 seconds
    GaussianNaiveBayes
    0.7286960338106209
    Running time:  5.57899999619 seconds
    KNN5
    0.9806798160441318
    Running time:  97.7709999084 seconds
    KNN1
    0.9730249110869923
    Running time:  98.1720001698 seconds
    SVM
    0.9576223574740722
    Running time:  32.4670000076 seconds
    LinearRegression
    -5.891759727231892e+16
    Running time:  6.32499980927 seconds
    

**Câu hỏi**: *Trên lí thuyết, độ phức tạp của KNN là bao nhiêu?*

### Bài 21 - Train với MNIST và test với DONGHANH

Trong các mô hình được giữ lại theo các tiêu chí trên (Perceptron, LDA, SVM, LogisticRegression) thì Perceptron không mở rộng được cho bài toán multiclass, SVM-LogisticRegression-LDA có độ chính xác thường tương đương nhau. Ta giữ lại LogisticRegression là mô hình được lựa chọn.

*Hãy train với LogisticRegression trên MNIST và test với dữ liệu ta đã tạo từ các phiếu chấm (từ đây gọi là DONGHANH).*

Bạn cần tự viết được đoạn code dưới đây.


```python
mnist_mark = Mark()
mnist_mark.loadMatData(MNIST_DATA)

donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)

group_identifier = []
scores = []

for i in range(10):
    for j in range(i+1, 10):
        print(i, " and ", j)
        group_identifier.append("%d vs %d" % (i, j))
        mnist_mark.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        donghanh_mark.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        model = MyLogisticRegression().fit(mnist_mark.getFilteredX(), mnist_mark.getFilteredy())
        score =  model.score(donghanh_mark.getFilteredX(), donghanh_mark.getFilteredy())
        scores.append(score)   
```

    (0, ' and ', 1)
    (0, ' and ', 2)
    (0, ' and ', 3)
    (0, ' and ', 4)
    (0, ' and ', 5)
    (0, ' and ', 6)
    (0, ' and ', 7)
    (0, ' and ', 8)
    (0, ' and ', 9)
    (1, ' and ', 2)
    (1, ' and ', 3)
    (1, ' and ', 4)
    (1, ' and ', 5)
    (1, ' and ', 6)
    (1, ' and ', 7)
    (1, ' and ', 8)
    (1, ' and ', 9)
    (2, ' and ', 3)
    (2, ' and ', 4)
    (2, ' and ', 5)
    (2, ' and ', 6)
    (2, ' and ', 7)
    (2, ' and ', 8)
    (2, ' and ', 9)
    (3, ' and ', 4)
    (3, ' and ', 5)
    (3, ' and ', 6)
    (3, ' and ', 7)
    (3, ' and ', 8)
    (3, ' and ', 9)
    (4, ' and ', 5)
    (4, ' and ', 6)
    (4, ' and ', 7)
    (4, ' and ', 8)
    (4, ' and ', 9)
    (5, ' and ', 6)
    (5, ' and ', 7)
    (5, ' and ', 8)
    (5, ' and ', 9)
    (6, ' and ', 7)
    (6, ' and ', 8)
    (6, ' and ', 9)
    (7, ' and ', 8)
    (7, ' and ', 9)
    (8, ' and ', 9)
    


```python
data = pd.DataFrame(scores, index = group_identifier, columns = ["LogisticRegression"])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LogisticRegression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 vs 1</th>
      <td>0.245455</td>
    </tr>
    <tr>
      <th>0 vs 2</th>
      <td>0.548387</td>
    </tr>
    <tr>
      <th>0 vs 3</th>
      <td>0.620370</td>
    </tr>
    <tr>
      <th>0 vs 4</th>
      <td>0.547619</td>
    </tr>
    <tr>
      <th>0 vs 5</th>
      <td>0.453704</td>
    </tr>
    <tr>
      <th>0 vs 6</th>
      <td>0.641791</td>
    </tr>
    <tr>
      <th>0 vs 7</th>
      <td>0.485507</td>
    </tr>
    <tr>
      <th>0 vs 8</th>
      <td>0.534884</td>
    </tr>
    <tr>
      <th>0 vs 9</th>
      <td>0.695364</td>
    </tr>
    <tr>
      <th>1 vs 2</th>
      <td>0.439024</td>
    </tr>
    <tr>
      <th>1 vs 3</th>
      <td>0.463768</td>
    </tr>
    <tr>
      <th>1 vs 4</th>
      <td>0.508772</td>
    </tr>
    <tr>
      <th>1 vs 5</th>
      <td>0.550725</td>
    </tr>
    <tr>
      <th>1 vs 6</th>
      <td>0.457317</td>
    </tr>
    <tr>
      <th>1 vs 7</th>
      <td>0.345238</td>
    </tr>
    <tr>
      <th>1 vs 8</th>
      <td>0.413793</td>
    </tr>
    <tr>
      <th>1 vs 9</th>
      <td>0.602210</td>
    </tr>
    <tr>
      <th>2 vs 3</th>
      <td>0.487603</td>
    </tr>
    <tr>
      <th>2 vs 4</th>
      <td>0.577320</td>
    </tr>
    <tr>
      <th>2 vs 5</th>
      <td>0.380165</td>
    </tr>
    <tr>
      <th>2 vs 6</th>
      <td>0.360544</td>
    </tr>
    <tr>
      <th>2 vs 7</th>
      <td>0.536424</td>
    </tr>
    <tr>
      <th>2 vs 8</th>
      <td>0.767677</td>
    </tr>
    <tr>
      <th>2 vs 9</th>
      <td>0.506098</td>
    </tr>
    <tr>
      <th>3 vs 4</th>
      <td>0.544643</td>
    </tr>
    <tr>
      <th>3 vs 5</th>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>3 vs 6</th>
      <td>0.376543</td>
    </tr>
    <tr>
      <th>3 vs 7</th>
      <td>0.379518</td>
    </tr>
    <tr>
      <th>3 vs 8</th>
      <td>0.614035</td>
    </tr>
    <tr>
      <th>3 vs 9</th>
      <td>0.530726</td>
    </tr>
    <tr>
      <th>4 vs 5</th>
      <td>0.794643</td>
    </tr>
    <tr>
      <th>4 vs 6</th>
      <td>0.731884</td>
    </tr>
    <tr>
      <th>4 vs 7</th>
      <td>0.598592</td>
    </tr>
    <tr>
      <th>4 vs 8</th>
      <td>0.722222</td>
    </tr>
    <tr>
      <th>4 vs 9</th>
      <td>0.838710</td>
    </tr>
    <tr>
      <th>5 vs 6</th>
      <td>0.197531</td>
    </tr>
    <tr>
      <th>5 vs 7</th>
      <td>0.530120</td>
    </tr>
    <tr>
      <th>5 vs 8</th>
      <td>0.710526</td>
    </tr>
    <tr>
      <th>5 vs 9</th>
      <td>0.385475</td>
    </tr>
    <tr>
      <th>6 vs 7</th>
      <td>0.489583</td>
    </tr>
    <tr>
      <th>6 vs 8</th>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>6 vs 9</th>
      <td>0.526829</td>
    </tr>
    <tr>
      <th>7 vs 8</th>
      <td>0.381944</td>
    </tr>
    <tr>
      <th>7 vs 9</th>
      <td>0.358852</td>
    </tr>
    <tr>
      <th>8 vs 9</th>
      <td>0.535032</td>
    </tr>
  </tbody>
</table>
</div>



Việc train trên MNIST và test trên DONGHANH dẫn đến độ chính xác giảm.

Nguyên nhân là do 2 tập dữ liệu, dù cùng gồm những hình ảnh có kích thước 28x28, nhưng có cách preprocess khác nhau. Cụ thể, ở DONGHANH, các chữ số được lấy sát biên, ở MNIST thì không. Ta cần đưa 2 tập dữ liệu về cùng một cách preprocessing.

### Bài 22 - Preprocessing với MNIST

Ta sẽ sử dụng cùng một cách preprocessing với DONGHANH trên MNIST. Hàm **`transformMatData(inputfile, outputfile)`** đã được viết sẵn để chuyển file `.mat` trong `inputfile` thành một file mới trong `outputfile`. File mới này có các chữ số được dãn ra sát biên, và mỗi pixel được biến thành 0, 1.

Khi chạy đoạn code
```python
transformMatData(MNIST_DATA, MNIST_DATA_TRANFORMED)
```
file `.mat` mới sẽ được tạo ở vị trí `MNIST_DATA_TRANFORMED`.

*Hãy viết instance method **`loadTransformedMatData(self, filename)`** tương tự như **`loadMatData`** đọc file `.mat` mới từ `filename` và gán các giá trị của $X$, $y$ cho `self.X`, `self.y`.*

Đoạn code dưới đây cần chạy và cho hình ảnh sát biên.


```python
mnist_mark = Mark()
mnist_mark.loadTransformedMatData(MNIST_DATA_TRANSFORMED)
mnist_mark.draw(0)
```

```
    ------------------------------
    |                  X         |
    |               XXXXXXX      |
    |              XXXXXXXX      |
    |            XXXXXXXXXXX     |
    |           XXXXXXXXX XXX    |
    |          XXXXXXXXX  XXX    |
    |         XXXXXXXXXX  XXX    |
    |       XXXXXXX  XXX    X    |
    |      XXXXXX           XXX  |
    |      XXXXXX           XXX  |
    |     XXXXXX            XXX  |
    |  XXXXX                XXXXX|
    |  XXXX                 XXXXX|
    |  XXX                  XXXXX|
    |XXXXX                  XXXXX|
    |XXXX                   XXX  |
    |XXX                    XXX  |
    |XXXX                   XXX  |
    |XXX                  XXXX   |
    |XXX                  XXX    |
    |XX                 XXXX     |
    |XXX             XXX         |
    |XXX        XXXXXXX          |
    |XXXX      XXXXXXX           |
    |XXXXXXXXXXXXXXXX            |
    |XXXXXXXXXXXXX               |
    |XXXXXXXXXXXX                |
    |    XXXXX                   |
    ------------------------------
```
    

### Bài 23 - Train với MNIST đã preprocess và test với DONGHANH

*Thực hiện lại việc train với MNIST và test với DONGHANH, nhưng bây giờ dùng dữ liệu MNIST đã preprocess sát biên.* 


```python
mnist_mark = Mark()
mnist_mark.loadTransformedMatData(MNIST_DATA_TRANSFORMED)

donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)

group_identifier = []
scores = []

for i in range(10):
    for j in range(i+1, 10):
        print(i, " and ", j)
        group_identifier.append("%d vs %d" % (i, j))
        mnist_mark.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        donghanh_mark.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        model = MyLogisticRegression().fit(mnist_mark.getFilteredX(), mnist_mark.getFilteredy())
        score =  model.score(donghanh_mark.getFilteredX(), donghanh_mark.getFilteredy())
        scores.append(score)   
```

    (0, ' and ', 1)
    (0, ' and ', 2)
    (0, ' and ', 3)
    (0, ' and ', 4)
    (0, ' and ', 5)
    (0, ' and ', 6)
    (0, ' and ', 7)
    (0, ' and ', 8)
    (0, ' and ', 9)
    (1, ' and ', 2)
    (1, ' and ', 3)
    (1, ' and ', 4)
    (1, ' and ', 5)
    (1, ' and ', 6)
    (1, ' and ', 7)
    (1, ' and ', 8)
    (1, ' and ', 9)
    (2, ' and ', 3)
    (2, ' and ', 4)
    (2, ' and ', 5)
    (2, ' and ', 6)
    (2, ' and ', 7)
    (2, ' and ', 8)
    (2, ' and ', 9)
    (3, ' and ', 4)
    (3, ' and ', 5)
    (3, ' and ', 6)
    (3, ' and ', 7)
    (3, ' and ', 8)
    (3, ' and ', 9)
    (4, ' and ', 5)
    (4, ' and ', 6)
    (4, ' and ', 7)
    (4, ' and ', 8)
    (4, ' and ', 9)
    (5, ' and ', 6)
    (5, ' and ', 7)
    (5, ' and ', 8)
    (5, ' and ', 9)
    (6, ' and ', 7)
    (6, ' and ', 8)
    (6, ' and ', 9)
    (7, ' and ', 8)
    (7, ' and ', 9)
    (8, ' and ', 9)
    


```python
data = pd.DataFrame(scores, index = group_identifier, columns = ["LogisticRegression"])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LogisticRegression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 vs 1</th>
      <td>0.963636</td>
    </tr>
    <tr>
      <th>0 vs 2</th>
      <td>0.989247</td>
    </tr>
    <tr>
      <th>0 vs 3</th>
      <td>0.990741</td>
    </tr>
    <tr>
      <th>0 vs 4</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>0 vs 5</th>
      <td>0.972222</td>
    </tr>
    <tr>
      <th>0 vs 6</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>0 vs 7</th>
      <td>0.978261</td>
    </tr>
    <tr>
      <th>0 vs 8</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>0 vs 9</th>
      <td>0.920530</td>
    </tr>
    <tr>
      <th>1 vs 2</th>
      <td>0.829268</td>
    </tr>
    <tr>
      <th>1 vs 3</th>
      <td>0.891304</td>
    </tr>
    <tr>
      <th>1 vs 4</th>
      <td>0.912281</td>
    </tr>
    <tr>
      <th>1 vs 5</th>
      <td>0.804348</td>
    </tr>
    <tr>
      <th>1 vs 6</th>
      <td>0.932927</td>
    </tr>
    <tr>
      <th>1 vs 7</th>
      <td>0.898810</td>
    </tr>
    <tr>
      <th>1 vs 8</th>
      <td>0.663793</td>
    </tr>
    <tr>
      <th>1 vs 9</th>
      <td>0.839779</td>
    </tr>
    <tr>
      <th>2 vs 3</th>
      <td>0.958678</td>
    </tr>
    <tr>
      <th>2 vs 4</th>
      <td>0.979381</td>
    </tr>
    <tr>
      <th>2 vs 5</th>
      <td>0.966942</td>
    </tr>
    <tr>
      <th>2 vs 6</th>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>2 vs 7</th>
      <td>0.867550</td>
    </tr>
    <tr>
      <th>2 vs 8</th>
      <td>0.969697</td>
    </tr>
    <tr>
      <th>2 vs 9</th>
      <td>0.890244</td>
    </tr>
    <tr>
      <th>3 vs 4</th>
      <td>0.973214</td>
    </tr>
    <tr>
      <th>3 vs 5</th>
      <td>0.985294</td>
    </tr>
    <tr>
      <th>3 vs 6</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3 vs 7</th>
      <td>0.879518</td>
    </tr>
    <tr>
      <th>3 vs 8</th>
      <td>0.938596</td>
    </tr>
    <tr>
      <th>3 vs 9</th>
      <td>0.625698</td>
    </tr>
    <tr>
      <th>4 vs 5</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4 vs 6</th>
      <td>0.963768</td>
    </tr>
    <tr>
      <th>4 vs 7</th>
      <td>0.964789</td>
    </tr>
    <tr>
      <th>4 vs 8</th>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>4 vs 9</th>
      <td>0.896774</td>
    </tr>
    <tr>
      <th>5 vs 6</th>
      <td>0.981481</td>
    </tr>
    <tr>
      <th>5 vs 7</th>
      <td>0.951807</td>
    </tr>
    <tr>
      <th>5 vs 8</th>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>5 vs 9</th>
      <td>0.765363</td>
    </tr>
    <tr>
      <th>6 vs 7</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6 vs 8</th>
      <td>0.985714</td>
    </tr>
    <tr>
      <th>6 vs 9</th>
      <td>0.951220</td>
    </tr>
    <tr>
      <th>7 vs 8</th>
      <td>0.868056</td>
    </tr>
    <tr>
      <th>7 vs 9</th>
      <td>0.775120</td>
    </tr>
    <tr>
      <th>8 vs 9</th>
      <td>0.675159</td>
    </tr>
  </tbody>
</table>
</div>



Kết quả khả quan hơn so với trước preprocessing.

Ta sẽ tiếp tục khai thác bài toán ở khía cạnh multiclass ở TD sau.

## References

[1] http://scikit-learn.org/stable/datasets/mldata.html
