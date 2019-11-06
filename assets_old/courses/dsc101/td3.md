
# TD3 - Làm việc với các thư viện Python

Khác với các TD trước, TD3 gồm 2 phần độc lập.

- Phần thứ nhất: Sử dụng các thư viện của Python để thực hiện một số tính toán về đại số tuyến tính, giải tích và lí thuyết xác suất. Ở phần này chỉ có các câu hỏi, không có yêu cầu viết code với dạng input và output quy định sẵn và không có file test. Bạn viết code tuỳ ý cho phép tính ra kết quả bài toán.

- Phần thứ hai: Mục đích của phần này là minh hoạ việc lấy dữ liệu từ Internet và xử lí bước đầu (preprocess) nó để có dữ liệu "sạch" để sử dụng cho các bài toán data science về sau. Phần này yêu cầu viết hàm với kiểu input, output quy định trước như ở các TD trước.

# Phần 1 - Dùng Python để tính toán

## A. Đại số tuyến tính

Cho các ma trận
$$
A = 
\begin{pmatrix}
0 & 1 & 0 & 0 \\\\
1 & 2 & -2 & 0 \\\\
1 & 0 & 1 & -1 \\\\
-1 & 2 & -1 & 1
\end{pmatrix},
B =
\begin{pmatrix}
-1 & 2 & 2 & -1 \\\\
0 & 1 & -1 & 0 \\\\
2 & -2 & -1 & 1 \\\\
0 & 0 & 1 & 1
\end{pmatrix}, 
\mathbf v = (2 \qquad 3 \qquad 1 \qquad 3)^t
$$
### Câu hỏi 1: (Nhân và luỹ thừa ma trận)
Tính $(AB)^5$

**Kết quả:**
$$
(AB)^5 = 
\begin{pmatrix}
-1284 & 2072 & 1328 & -1436 \\\\
-7337 & 12017 & 8073 & -8735 \\\\
259 & -403 & -227 &  245 \\\\
-2827 & 4547 & 2883 & -3117\\\\
\end{pmatrix}
$$


### Câu hỏi 2: (Tích giữa ma trận và vector)
Tính $A\mathbf v, B^t \mathbf v, \mathbf v^t A \mathbf v $.

**Kết quả:**
$$
A\mathbf v =
\begin{pmatrix}
3\\\\
6\\\\
0\\\\
6
\end{pmatrix},
B^t\mathbf v =
\begin{pmatrix}
0\\\\
5\\\\
3\\\\
2\\\\
\end{pmatrix},
\mathbf v^t A \mathbf v = 42
$$

### Câu hỏi 3: (Nghịch đảo ma trận)
Tính $(A+I_4)^{-1}$, trong đó $I_4$ là ma trận đơn vị cấp 4

**Kết quả:**
$$
\begin{pmatrix}
1.625 & -0.375 & -0.5 & -0.25 \\\\
-0.625 & 0.375 & 0.5 & 0.25 \\\\
-0.125 & -0.125 & 0.5 & 0.25 \\\\
1.375 & -0.625 & -0.5 & 0.25
\end{pmatrix}
$$

### Câu hỏi 4: (SVD)
Gọi $C$ là ma trận 3x4 được tạo bởi 3 hàng đầu của ma trận $A$. Tìm một ma trận trực giao $U$ cấp 3x3, một ma trận $\Sigma$ cấp 3x4 có dạng "đường chéo" (tức có phần tử $a_{ij}=0$ nếu $i \neq j$) và một ma trận trực giao $V$ cấp 4x4 sao cho $C = U\Sigma V^t$. (Singular value decomposition)

**Một kết quả:**
$$
\begin{pmatrix}
 0 & 1 & 0 & 0 \\\\
 1 & 2 & -2 & 0 \\\\
 1 & 0 & 1 & -1 \\\\
\end{pmatrix} 
= 
\begin{pmatrix}
-0.223716    & 0.12705277 & -0.9663378  \\\\
-0.96370632  & 0.11935282 &  0.23879913 \\\\
 0.14567523  & 0.98468904 &  0.09574042
\end{pmatrix}
\begin{pmatrix}
3.10087793 & 0.         & 0.        &  0.     \\\\
0.         & 1.69670014 & 0.        &  0.     \\\\
0.         & 0.         & 0.71117135&  0.     
\end{pmatrix}
\begin{pmatrix}
-0.26380629  & 0.65069946 &  0.47040639 & 0.53452248 \\\\
-0.693716    & 0.21557044 & -0.68723176 & 0.         \\\\
 0.66854869  & 0.4396672  & -0.53694211 & 0.26726124 \\\\
-0.04697871  &-0.58035537 & -0.13462356 & 0.80178373 
\end{pmatrix}^t
$$

### Câu hỏi 5: (Chéo hoá ma trận)
Tìm một ma trận $Q$ cấp $4 \times 4$ và một ma trận đường chéo $D$ cấp 4 sao cho $Q^{-1}AQ = D$. Từ đó suy ra các giá trị riêng và một cơ sở gồm các vector riêng của $A$. Suy ra hạng của $A$.

**Một kết quả:**
$$
Q = \begin{pmatrix}
-0.5 & -0.5 & 0.5 & 0.2236 \\\\
0 & 0 & 0.5 & 0.6708 \\\\
-0.2673 & -0.2673 & 0.5 & -0.2236 \\\\
-0.8018 & -0.8018 & 0.5 & 0.6708
\end{pmatrix},
D = \begin{pmatrix}
0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 \\\\
0 & 0 & 1 & 0 \\\\
0 & 0 & 0 & 3 \\\\
\end{pmatrix}
$$

### Câu hỏi 6: (Tìm đa thức đặc trưng)
Tìm đa thức đặc trưng của $A$. Suy ra định thức và trace của $A$. Thử tìm định thức và trace trực tiếp bằng hàm trong numpy.

**Kết quả**
$$P_A(X) = X^4 - 4X^3 + 3X^2$$

### Câu hỏi 7: (Giải hệ phương trình).
Tìm $\mathbf x \in \mathbf R^4$ sao cho $(A + I_4)\mathbf x = \mathbf v$.

**Kết quả**
$$\mathbf x =  (0.875 \qquad 1.125 \qquad 0.625 \qquad 1.125)^t. $$

## B. Giải tích thực

Cho
$$
f(x) = 4\sin x - \cos x + 2e^{-3x^2} + \frac1 x 
$$
$$
F(x, y) = \frac{(x-y)^2}{x^2 + y^2}
$$

### Câu hỏi 8: (Đạo hàm)
Tính đạo hàm cấp 1, 2, 3, 4 của $f(x)$ tại $x = 3$.

**Kết quả**
$$
-3.92996109022, -1.47992729183, 3413.93580072, -3774758283.73
$$

### Câu hỏi 9: (Đạo hàm hàm nhiều biến)
Tính gradient của $F$ tại $(x, y) = (1, -2)$.

**Kết quả:**
$$
(0.48, 0.24)^t
$$

### Câu hỏi 10: (Tích phân)
Tính tích phân 
$$
\int_1^2 f(x) dx
$$
$$
\int_0^2 \int_{\max(0, \sqrt{1-x^2})}^{\sqrt{4-x^2}} F(x, y) dy dx
$$

**Kết quả:**
$$
4.4658, 0.8562
$$

### Câu hỏi 11: (Tìm nghiệm)
Vẽ đồ thị $y = f(x)$ bằng **pyplot** trên đoạn [-10, 10]. Từ đồ thị ước lượng 6 nghiệm của $f$ trên khoảng này. Tìm gần đúng các nghiệm này (chính xác đến chẳng hạn 5 chữ số thập phân) bằng một hàm của **scipy** với thuật toán tuỳ chọn.


```python
%matplotlib inline
from Maths_Solution import ex11
ex11() #This is the function name in the solution
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/output_17_0.png)


    -9.20614709658
    -5.9977578786
    -2.97814285691
    3.45679107929
    6.4907892106
    9.69477637904
    

### Câu hỏi 12: (Tìm cực trị)
Từ đồ thị ở câu hỏi 11, tìm tất cả các cực tiểu và cực đại địa phương của $f$ trên [-10, 10].


```python
from Maths_Solution import ex12
ex12() #This is the function name in the solution
```

    Local minima: 
    [-7.60478516]
    [-1.20351563]
    [ 0.72998047]
    [ 4.96722412]
    Local maxima: 
    [-4.47949219]
    [-0.39951172]
    [ 1.73447266]
    [ 8.09521484]
    

### Câu hỏi 13: (Đồ thị hàm nhiều biến)

Vẽ hai hình:
- Hình thứ nhất là đồ thị biểu diễn các đường đồng mức của $F$ với giá trị 0.5, 1, 1.5, 2 khi $|x|, |y| \leq 5$. Thông qua đồ thị, giải phương trình $F(x, y) = 1$, $F(x, y) = 0.5$
- Hình thứ hai là đồ thị 3D của $F(x, y)$ theo $x, y$ khi $|x|, |y| \leq 5$


```python
%matplotlib inline
from Maths_Solution import ex13
ex13() #This is the function name in the solution
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/output_22_0.png)



![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/output_22_1.png)


### Câu hỏi 14: (Cực trị hàm nhiều biến)
Tìm $(x_0, y_0)$ thoả mãn $x_0 + y_0 = 5$ và maximize $F(x, y)$.

**Kết quả**
$$
(2.5, 2.5)
$$

## C. Lí thuyết xác suất

### Câu hỏi 15: (Biến liên tục)
Giả sử $X$ tuân theo phân phối chuẩn tắc với kì vọng $\mu = 2$ và phương sai 6. Tính $\mathbf P(-1 \leq X \leq 5)$. Tìm $\alpha$ để $\mathbf P(\mu - \alpha \leq X \leq \mu + \alpha) \geq 0.9$.

**Kết quả**
$$
0.77932863808, 4.0290520876
$$

### Câu hỏi 16: (Biến rời rạc)
Giả sử $X$ tuân theo phân phối nhị thức với tham số $N = 10, \mu = 0.7$. Tính $\mathbf P(X = 0.8), \mathbf P(X \geq 0.8)$. Phát sinh ngẫu nhiên 1000 quan sát của $X$ và biểu diễn chúng bằng một histogram.


```python
%matplotlib inline
from Maths_Solution import ex16
ex16() #This is the function name in the solution
```

    0.2334744405
    0.3827827864
    7.0
    


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/output_27_1.png)


### Câu hỏi 17: (Minh hoạ định lí giá trị trung tâm)

Giả sử $\mathbf X = (X_1, \ldots, X_n)$, $n = 100$ là một vector gồm 100 biến ngẫu nhiên iid theo luật Poisson(1.5). Gọi $\mu$ và $\sigma$ là kì vọng và độ lệch chuẩn của luật Poisson(1.5). Đặt:
$$
S = \frac{X_1 + \ldots + X_n - n\mu}{\sigma \sqrt{n}}
$$
Phát sinh ngẫu nhiên 1000 quan sát của $\mathbf X$ (tức phát sinh $ (X_1^{(i)}, \ldots, X_n^{(i)})$, $(i =1, \ldots, 1000)$) và tính các giá trị tương ứng $S^{(i)}$, $i = 1, \ldots, 1000$. Hãy biểu diên histogram của các $S^{(i)}$. So sánh với pdf của $\mathcal N(0, 1)$.


```python
%matplotlib inline
from Maths_Solution import ex17
ex17() #This is the function name in the solution
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/output_30_0.png)


# Phần 2 - Vnexpress

# Mô tả

Trong phần này ta sẽ luyện tập thực hiện các kĩ năng:
- Cài đặt một thư viện chưa có trong Python
- Download một trang web dưới dạng string
- Tìm hiểu cấu trúc một trang web (dưới dạng html)
- Trích xuất một chi tiết trong file html
- Xử lí dữ liệu ở dạng string và dạng bảng.

Mục đích của phần này là download tất cả các bài báo thuộc mảng khoa học ở vnexpress.net được xuất bản vào tháng 1 năm 2018, sau đó trích ra những thông tin ta cần trong một bài báo. Ở TD5, ta sẽ tìm cách biến mỗi bài báo này thành một vector từ. 

Bạn sẽ cần hoàn thành các hàm trong <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/VNExpressPreProcessing.py"> VNExpressPreProcessing.py</a> và chạy các test trong <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/TestVNExpressPreProcessing.py"> TestVNExpressPreProcessing.py</a> như các bài trước để kiểm tra (các test từ 9 trở đi sẽ tương đối mất thời gian, khoảng 1-2 phút). Trong **TestVNExpressPreProcessing.py**, bạn import tập tin source code bạn đã viết, ví dụ như dưới đây.


```python
from VNExpressPreProcessing_Solution import *
```

## Bước 1. Chuẩn bị

### Bài 1. Download thư viện BeautifulSoup
Thư viện beautifulsoup4 cung cấp nhiều công cụ xử lí các trang web được download ở dạng html.

*Hãy download thư viện **beautifulsoup4.** *

Gợi ý: Bạn có thể download từ terminal của Linux/Mac hoặc từ anaconda terminal với câu lệnh **conda install -c anaconda beautifulsoup4**

Sau khi download, bạn có thể test với đoạn code sau. Đoạn code cần chạy tốt, không có lỗi **ImportError** nào được báo.


```python
from bs4 import BeautifulSoup
```

### Bài 2. Tạo thư mục Data
*Hãy viết hàm **prepareDataFolder(data_folder)** nhận đối số là một str **data_folder** và tạo một thư mục mang tên **data_folder** tại nơi mà bạn làm việc nếu thư mục này chưa có.*

*Nếu thư mục này đã tồn tại hoặc tại nơi bạn làm việc đã có một tập tin mang tên giống với **data_folder**, hàm này in một dòng thông báo và không làm gì khác.* 

Bạn có thể test bằng đoạn code sau.


```python
prepareDataFolder("Data")
```

Sau khi test, thư mục **Data** cần xuất hiện trong thư mục bạn đang làm việc. Nếu chạy dòng code trên một lần nữa, cần nhận được một thông báo kiểu như sau:


```python
prepareDataFolder("Data")
```

    Folder 'Data' already existed.
    

## Bước 2. Tìm tất cả các đường dẫn bài báo

Việc đầu tiên là đi tìm tất cả các bài báo về chủ đề khoa học trên VNExpress được xuất bản vào tháng 1 năm 2018. Muốn vậy, bạn truy cập vào <a href="vnexpress.net">vnexpress.net</a>, chọn mục "Khoa học" trên thanh menu chính. Kéo xuống dưới cùng, chọn "Xem theo ngày". Nhấn vào 01/01/2018, rồi giữ shift và nhấn vào 31/01/2018. 

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/Fig1.png" width=600></img>

Click "Hoàn tất", lúc này bạn sẽ được dẫn đến một trang liệt kê 30 bài báo đầu tiên (hoặc tất cả nếu có ít hơn 30 bài báo) về chủ đề khoa học trong khoảng thời gian tương ứng. Ta gọi trang này là **trang liệt k 1ê**.

**Đường dẫn của trang liệt kê 1 như sau**:

https://vnexpress.net/category/day/?cateid=1001009&fromdate=1514761200&todate=1517438160&allcate=1001009|1001009|

### Câu hỏi:
*Bạn hiểu thế nào về các thành phần của các đường dẫn trên?*

### Gợi ý trả lời câu hỏi:

- Phần https://vnexpress.net/category/day/ là phần đầu của đường dẫn.
- Sau đó là một dấu chấm hỏi **?** bắt đầu phần lọc theo 4 tiêu chí cách nhau bởi dấu **&**. Mỗi tiêu chí lọc được viết theo quy tắc **thamsố=giátrị**. 
- Tiêu chí thứ nhất có nghĩa là lọc các bài báo có thể loại 1001009 (đây là mã số của thể loại khoa học)
- Tiêu chí thứ hai nghĩa là lọc các bài báo từ thời điểm 1514761200 (theo <a href="https://www.unixtimeconverter.io/1514761200">https://www.unixtimeconverter.io/1514761200</a>, đây là 23h ngày 31/12/2017 theo giờ GMT.)
- Tiêu chí thứ ba là thời điểm kết thúc.
- Tiêu chí thứ tư là một sự lặp lại của tiêu chí đầu.

Dựa vào phân tích trên, ta có thể tìm cách download tự động một trang liệt kê.

### Bài 3. Chuyển đổi timestamp

Số 1514761200 trong đường dẫn trên là timestamp hay Unix Time của thời điểm 23h ngày 31/12/2017 (GMT). Nó được định nghĩa là số giây đã trôi qua kể từ 0h00 ngày 1/1/1970 (GMT) cho đến thời điểm đang tính.

*Hãy viết hàm **dayToTimestamp(day)** nhận đối số **day** dưới dạng "DD/MM/YYYY" (tức 1 str có 10 kí tự như "21/01/2018") và chuyển nó thành timestamp của thời điểm 0h ngày tương ứng dưới dạng **int**.*

Gợi ý: Tìm trên google với từ khoá **convert date to timestamp in python**

Chạy đoạn code dưới đây để test hàm của bạn.


```python
dayToTimestamp("21/01/2018")
```




    1516489200



### Bài 4. Download trang liệt kê 1
Trong file **VNExpressPreProcessing.py**, một đoạn code đã được viết sẵn tương ứng mỗi thể loại với mã số của nó.


```python
THOI_SU = 1001005 
GIA_DINH = 1002966 
SUC_KHOE = 1003750 
THE_GIOI = 1001002 
KINH_DOANH = 1003159 
GIAI_TRI = 1002691 
THE_THAO = 1002565 
PHAP_LUAT = 1001007 
GIAO_DUC = 1003497 
DU_LICH = 1003231 
KHOA_HOC = 1001009 
SO_HOA = 1002592 
XE = 1001006 
CONG_DONG = 1001012 
TAM_SU = 1001014
```

Bây giờ ta có thể viết hàm download trang liệt kê các tin khoa học trên VNExpress trong tháng 1.

*Hãy viết hàm **downloadFirstTitlePage(category, fromdate, todate)** nhận 3 đối số:*

*- **category** là một số nguyên (int) (ví dụ: 1001005) là mã số của một trong 15 thể loại trên*

*- **fromdate**, **todate** là hai string có dạng "DD/MM/YYYY" chỉ thời điểm bắt đầu và kết thúc của các bài báo muốn nhận. Ví dụ nếu fromdate = "01/01/2018" và todate="01/02/2018", ta tính các bài báo từ 0h ngày 01/01/2018 đến 0h ngày 01/02/2018.*

*và download trang liệt kê 1 tương ứng, trả lại nội dung của trang liệt kê 1 dưới dạng một str.*

Chạy đoạn code dưới đây để test hàm của bạn (mất <20s)


```python
s = downloadFirstTitlePage(THOI_SU, "01/01/2018", "02/01/2018")
s[100:200] #Print part of the file
```




    'tible" content="IE=100" />\n        <meta property="fb:app_id" content="1547540628876392" />\n        '



### Bài 5. Tìm tất cả các đường dẫn bài báo có trong trang liệt kê 1

Đến bước này ta đã có nội dung của trang liệt kê 1 được download dưới dạng một str. str này có dạng html tương đối phức tạp về cấu trúc. Cụ thể, một trang liệt kê 1 ở dạng html thường có cấu trúc như ví dụ sau:


```python
<html>
    <head>
        ...
    </head>
    <body>
        ...
        <article class="list_news">
               ...
               <h3 class="title_news">
                      <a href="https://vnexpress.net/tin-tuc/khoa-hoc/tran-ngoam-dau-loi-thu-co-tui-len-ngon-cay-de-nuot-chung-3706463.html"> Trăn ngoạm đầu lôi thú có túi lên ngọn cây để nuốt chửng </a>
               </h3>
               <h3 class="title_news">
                      <a href="https://vnexpress.net/tin-tuc/khoa-hoc/thuong-thuc/hien-tuong-nam-cham-bi-day-khi-roi-tren-mat-tam-dong-3706503.html"> Hiện tượng nam châm bị đẩy khi rơi trên mặt tấm đồng </a>
               </h3>
               <h3>
               ...
               </h3>
               ...
        </article>
    </body>
</html>
```

Trong đó, nội dung của cả file sẽ được ghi giữa thẻ $<html>$ và thẻ $</html>$. Nội dung này lại được chia thành hai phần: một phần "head" được ghi giữa $<head>$ và $</head>$, một phần "body" được ghi giữa $<body>$ và $</body>$. Ta nói $<head>$ và $<body>$ là các thẻ con của $<html>$. Cứ như vậy, mỗi thẻ lại chia thành các thẻ con nhỏ hơn, tạo thành thứ bậc như hình dưới đây.

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/Fig2.png" width=600></img>

Điều ta cần quan tâm là đường dẫn của tất cả các bài báo có trong trang. Như biểu diễn, đường dẫn này là giá trị của tham số **href** ở trong các thẻ **a**, mà mỗi thẻ **a** là thẻ con của một thẻ **h3** thuộc class **title_news**.

Thư viện BeautifulSoup giúp ta lấy ra được các đường dẫn này bằng câu lệnh đơn giản như đã được viết trong hàm **getLinksFromTitlePage(page_content)**. Hàm này nhận đối số **page_content** là nội dung trang liệt kê 1 đã được download và in ra tất cả các đường dẫn đến các bài báo có trong trang này bằng hàm **print**.

*Hãy sửa chữa hàm **getLinksFromTitlePage(page_content)** để thay vì dùng hàm **print** in tất cả các đường link ra màn hình, thì trả lại một list chứa tất cả các đường link đó.*

Chạy đoạn code dưới đây để test hàm của bạn.


```python
s = downloadFirstTitlePage(THOI_SU, "12/02/2018", "14/03/2018")
getLinksFromTitlePage(s)
```




    ['https://video.vnexpress.net/tin-tuc/xa-hoi/nha-xe-ha-noi-than-e-khach-ve-que-an-tet-3712083.html',
     'https://vnexpress.net/tin-tuc/thoi-su/oto-camry-tong-lien-hoan-nam-xe-tren-pho-ha-noi-3712086.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/csgt-mien-tay-phat-nuoc-suoi-khan-lanh-cho-dan-ve-que-an-tet-3712084.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/ga-sai-gon-dong-nghit-nguoi-ve-que-don-tet-3711933.html',
     'https://vnexpress.net/tin-tuc/thoi-su/giao-thong/20-duong-day-nong-phan-anh-giao-thong-dip-tet-3712061.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/lang-bien-khoac-ao-bich-hoa-don-tet-3712013.html',
     'https://vnexpress.net/tin-tuc/thoi-su/ca-nuoc-nang-dep-dip-tet-nguyen-dan-3712027.html',
     'https://vnexpress.net/tin-tuc/thoi-su/gan-800-000-nguoi-ngheo-duoc-nhan-gao-ho-tro-dip-tet-3711904.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/cho-neu-lang-bien-o-quang-binh-nhon-nhip-ngay-tet-3711910.html',
     'https://vnexpress.net/tin-tuc/thoi-su/ha-noi-khong-to-chuc-pho-di-bo-dip-tet-nguyen-dan-3711889.html',
     'https://vnexpress.net/tin-tuc/thoi-su/bo-van-hoa-yeu-cau-khong-to-chuc-hau-dong-nhu-ca-nhac-duong-pho-3711885.html',
     'https://vnexpress.net/tin-tuc/thoi-su/nhieu-tai-xe-dinh-dinh-tren-quoc-lo-o-nghe-an-3711854.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/tran-ra-giua-duong-buon-ban-tren-pho-thoi-trang-sai-gon-3711882.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/cu-ba-92-tuoi-cung-con-chau-goi-banh-tet-don-tet-3711637.html',
     'https://vnexpress.net/tin-tuc/thoi-su/ong-le-phuoc-hoai-bao-bi-xoa-ten-khoi-danh-sach-dang-vien-3711852.html',
     'https://vnexpress.net/tin-tuc/thoi-su/bao-sanba-do-bo-philippines-kha-nang-suy-yeu-tren-bien-dong-3711861.html',
     'https://vnexpress.net/tin-tuc/thoi-su/cuc-chong-tham-nhung-nhan-nhieu-cuoc-goi-ve-qua-tet-trai-quy-dinh-3711766.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/rung-hoa-anh-dao-trong-hoi-hoa-xuan-vung-tau-3711632.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/y-nghia-cua-ba-chu-de-tren-duong-hoa-nguyen-hue-3711623.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/hue-ban-sung-than-cong-tren-ky-dai-3711607.html',
     'https://vnexpress.net/tin-tuc/thoi-su/be-gai-bon-thang-tuoi-bi-bo-roi-truoc-cong-nha-3711643.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/nguoi-sai-gon-chi-tien-trieu-mua-la-dong-goi-banh-tet-3711570.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/loi-chuc-tet-cua-linh-dao-truong-sa-gui-dat-lien-3711542.html',
     'https://vnexpress.net/tin-tuc/thoi-su/ong-nguyen-thien-nhan-2018-la-thoi-co-phat-trien-tp-hcm-3710955.html',
     'https://vnexpress.net/photo/thoi-su/nguoi-ha-noi-chen-chan-mua-sam-tren-via-he-3711624.html',
     'https://vnexpress.net/tin-tuc/giao-duc/con-gai-nguoi-lao-cong-nhan-hoc-bong-6-ty-dong-cua-dai-hoc-my-3711163.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/cong-nhan-dong-nai-tranh-tai-mua-lan-mung-xuan-mau-tuat-3711463.html',
     'https://vnexpress.net/tin-tuc/giao-duc/giao-vien-tay-hat-ngay-tet-que-em-tang-hoc-tro-3711565.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/deo-noi-binh-thuan-lam-dong-sat-lo-4-thang-van-chua-sua-xong-3711551.html']



### Bài 6. Các trang liệt kê tiếp theo
Trang liệt kê 1 chỉ liệt kê 30 bài báo đầu tiên, nếu click vào trang thứ hai hoặc biểu tượng next (">"), ta sẽ thấy trang liệt kê 2 có đường dẫn như sau:

https://vnexpress.net/category/day/page/2.html?cateid=1001009&fromdate=1514761200&todate=1517438160&allcate=1001009||

Ta thấy điều khác biệt duy nhất so với trang liệt kê 1 là có thêm phần "2.html". Điều này cũng đúng với các trang liệt kê 3, 4, ..., 10, 11, ...

*Tương tự như hàm **downloadFirstTitlePage(category, fromdate, todate)**, hãy viết hàm **downloadTitlePage(index, category, fromdate, todate)**, lần này nhận 4 đối số:*

*- **index** là một số nguyên (int) (ví dụ: 2) là số thứ tự của trang liệt kê

*- **category** là một số nguyên (int) (ví dụ: 1001005) là mã số của một trong 15 thể loại trên*

*- **fromdate**, **todate** là hai string có dạng "DD/MM/YYYY" chỉ thời điểm bắt đầu và kết thúc của các bài báo muốn nhận.*

*và download trang liệt kê thứ **index** tương ứng, trả lại nội dung của trang này dưới dạng một str.*

Chạy đoạn code dưới đây để test hàm của bạn.


```python
s = downloadTitlePage(2, THOI_SU, "12/02/2018", "14/02/2018")
getLinksFromTitlePage(s)
```




    ['https://vnexpress.net/tin-tuc/thoi-su/le-hoi-den-hung-2018-co-5-tinh-cung-to-chuc-3711418.html',
     'https://vnexpress.net/tin-tuc/thoi-su/trung-ve-bui-tien-dung-ky-tang-anh-gay-quy-giup-nguoi-ngheo-don-tet-3711371.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/100-canh-sat-can-tho-gom-rac-tren-duong-pho-ngay-giap-tet-3711396.html',
     'https://vnexpress.net/tin-tuc/thoi-su/ao-va-bong-doi-tuyen-u23-tang-thu-tuong-duoc-dau-gia-20-ty-dong-3711377.html',
     'https://vnexpress.net/tin-tuc/thoi-su/nhieu-cong-nghe-lan-dau-xuat-hien-tren-duong-hoa-nguyen-hue-3711294.html',
     'https://vnexpress.net/tin-tuc/thoi-su/no-lo-luyen-phoi-thep-o-ha-noi-hai-nguoi-nguy-kich-3711323.html',
     'https://vnexpress.net/tin-tuc/thoi-su/giao-thong/lat-xe-khach-hai-nguoi-tu-vong-3711203.html',
     'https://vnexpress.net/tin-tuc/thoi-su/bao-sanba-nhieu-kha-nang-tan-tren-bien-dong-3711232.html',
     'https://vnexpress.net/photo/thoi-su/nguoi-ha-noi-chi-tien-trieu-lam-dep-thu-cung-don-tet-3711212.html',
     'https://video.vnexpress.net/tin-tuc/xa-hoi/spa-cho-thu-cung-dong-khach-tu-sang-den-dem-3711146.html']



### Bài 7. Lưu tất cả các liên kết bài báo vào một file

*Từ bài 5, 6, hãy viết hàm **saveTitlesFromTitlePages(category, fromdate, todate, title_file, data_folder)** nhận đối số: *

*- **category, fromdate, todate** như mô tả ở bài 6, *

*- **title_file** là một str, chỉ tên một file mà ta sẽ ghi dữ liệu vào đó*

*- **data_folder** là một thư mục chứa file sẽ ghi.*

*và thực hiện việc lưu tất cả các đường link đến các bài báo hiển thị trên tất cả các trang liệt kê 1, 2, 3, ... vào trong file **title_file** nằm trong thư mục **data_folder**, mỗi đường link là một dòng. (Các dòng cách nhau bởi dấu xuống hàng "\n")*.

Đoạn code sau giúp test hàm của bạn. Sau khi chạy, khi mở file **Science_Jan2018_Titles.txt** trong thư mục **Data**, file cần có nội dung như kết quả dưới. (Chú ý các dòng đầu chỉ là các dòng print để dễ theo dõi, đoạn từ "Output: " mới là output)


```python
prepareDataFolder("Data")
saveLinksFromTitlePages(THOI_SU, "12/02/2018", "14/02/2018", "Science_Jan2018_Titles.txt", "Data")
f = open("Data/Science_Jan2018_Titles.txt").read()
print("Output: ")
print(f)
```

    Folder 'Data' already existed.
    Downloading page 1
    Downloading page 2
    Downloading page 3
    Output: 
    https://video.vnexpress.net/tin-tuc/xa-hoi/nha-xe-ha-noi-than-e-khach-ve-que-an-tet-3712083.html
    https://vnexpress.net/tin-tuc/thoi-su/oto-camry-tong-lien-hoan-nam-xe-tren-pho-ha-noi-3712086.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/csgt-mien-tay-phat-nuoc-suoi-khan-lanh-cho-dan-ve-que-an-tet-3712084.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/ga-sai-gon-dong-nghit-nguoi-ve-que-don-tet-3711933.html
    https://vnexpress.net/tin-tuc/thoi-su/giao-thong/20-duong-day-nong-phan-anh-giao-thong-dip-tet-3712061.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/lang-bien-khoac-ao-bich-hoa-don-tet-3712013.html
    https://vnexpress.net/tin-tuc/thoi-su/ca-nuoc-nang-dep-dip-tet-nguyen-dan-3712027.html
    https://vnexpress.net/tin-tuc/thoi-su/gan-800-000-nguoi-ngheo-duoc-nhan-gao-ho-tro-dip-tet-3711904.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/cho-neu-lang-bien-o-quang-binh-nhon-nhip-ngay-tet-3711910.html
    https://vnexpress.net/tin-tuc/thoi-su/ha-noi-khong-to-chuc-pho-di-bo-dip-tet-nguyen-dan-3711889.html
    https://vnexpress.net/tin-tuc/thoi-su/bo-van-hoa-yeu-cau-khong-to-chuc-hau-dong-nhu-ca-nhac-duong-pho-3711885.html
    https://vnexpress.net/tin-tuc/thoi-su/nhieu-tai-xe-dinh-dinh-tren-quoc-lo-o-nghe-an-3711854.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/tran-ra-giua-duong-buon-ban-tren-pho-thoi-trang-sai-gon-3711882.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/cu-ba-92-tuoi-cung-con-chau-goi-banh-tet-don-tet-3711637.html
    https://vnexpress.net/tin-tuc/thoi-su/ong-le-phuoc-hoai-bao-bi-xoa-ten-khoi-danh-sach-dang-vien-3711852.html
    https://vnexpress.net/tin-tuc/thoi-su/bao-sanba-do-bo-philippines-kha-nang-suy-yeu-tren-bien-dong-3711861.html
    https://vnexpress.net/tin-tuc/thoi-su/cuc-chong-tham-nhung-nhan-nhieu-cuoc-goi-ve-qua-tet-trai-quy-dinh-3711766.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/rung-hoa-anh-dao-trong-hoi-hoa-xuan-vung-tau-3711632.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/y-nghia-cua-ba-chu-de-tren-duong-hoa-nguyen-hue-3711623.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/hue-ban-sung-than-cong-tren-ky-dai-3711607.html
    https://vnexpress.net/tin-tuc/thoi-su/be-gai-bon-thang-tuoi-bi-bo-roi-truoc-cong-nha-3711643.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/nguoi-sai-gon-chi-tien-trieu-mua-la-dong-goi-banh-tet-3711570.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/loi-chuc-tet-cua-linh-dao-truong-sa-gui-dat-lien-3711542.html
    https://vnexpress.net/tin-tuc/thoi-su/ong-nguyen-thien-nhan-2018-la-thoi-co-phat-trien-tp-hcm-3710955.html
    https://vnexpress.net/photo/thoi-su/nguoi-ha-noi-chen-chan-mua-sam-tren-via-he-3711624.html
    https://vnexpress.net/tin-tuc/giao-duc/con-gai-nguoi-lao-cong-nhan-hoc-bong-6-ty-dong-cua-dai-hoc-my-3711163.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/cong-nhan-dong-nai-tranh-tai-mua-lan-mung-xuan-mau-tuat-3711463.html
    https://vnexpress.net/tin-tuc/giao-duc/giao-vien-tay-hat-ngay-tet-que-em-tang-hoc-tro-3711565.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/deo-noi-binh-thuan-lam-dong-sat-lo-4-thang-van-chua-sua-xong-3711551.html
    https://vnexpress.net/tin-tuc/thoi-su/le-hoi-den-hung-2018-co-5-tinh-cung-to-chuc-3711418.html
    https://vnexpress.net/tin-tuc/thoi-su/trung-ve-bui-tien-dung-ky-tang-anh-gay-quy-giup-nguoi-ngheo-don-tet-3711371.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/100-canh-sat-can-tho-gom-rac-tren-duong-pho-ngay-giap-tet-3711396.html
    https://vnexpress.net/tin-tuc/thoi-su/ao-va-bong-doi-tuyen-u23-tang-thu-tuong-duoc-dau-gia-20-ty-dong-3711377.html
    https://vnexpress.net/tin-tuc/thoi-su/nhieu-cong-nghe-lan-dau-xuat-hien-tren-duong-hoa-nguyen-hue-3711294.html
    https://vnexpress.net/tin-tuc/thoi-su/no-lo-luyen-phoi-thep-o-ha-noi-hai-nguoi-nguy-kich-3711323.html
    https://vnexpress.net/tin-tuc/thoi-su/giao-thong/lat-xe-khach-hai-nguoi-tu-vong-3711203.html
    https://vnexpress.net/tin-tuc/thoi-su/bao-sanba-nhieu-kha-nang-tan-tren-bien-dong-3711232.html
    https://vnexpress.net/photo/thoi-su/nguoi-ha-noi-chi-tien-trieu-lam-dep-thu-cung-don-tet-3711212.html
    https://video.vnexpress.net/tin-tuc/xa-hoi/spa-cho-thu-cung-dong-khach-tu-sang-den-dem-3711146.html
    

## Bước 3. Xử lí từng bài báo

### Bài 8. Download một bài báo

*Hãy viết hàm **downloadArticle(link)** nhận đối số **link** là đường dẫn của một bài báo (ví dụ, https://vnexpress.net/tin-tuc/khoa-hoc/hoi-dap/co-phai-dien-thoai-nhanh-sut-pin-hon-vao-mua-dong-3694402.html) và trả lại nội dung được download dưới dạng một str.*

Test hàm của bạn bằng đoạn code dưới đây.


```python
s = downloadArticle("https://vnexpress.net/tin-tuc/thoi-su/ca-nuoc-nang-dep-dip-tet-nguyen-dan-3712027.html")
s[100:200] #Print part of the article
```




    '100"/>\n<meta property="fb:app_id" content="1547540628876392"/>\n<link rel="amphtml" href="https://amp'



### Bài 9. Tìm tiêu đề, đoạn giới thiệu và nội dung
Sau khi thực hiện bài 8, ta sẽ có nội dung dạng html của bài báo cần download. Bây giờ, điều ta cần quan tâm là:

- Tiêu đề bài báo (ví dụ với bài báo https://vnexpress.net/tin-tuc/khoa-hoc/nhen-tho-san-gia-chet-van-bi-ong-bap-cay-truy-giet-3694170.html, đó là "Nhện thợ săn giả chết vẫn bị ong bắp cày truy giết")

- Đoạn giới thiệu (ví dụ "Chiêu giả chết của nhện thợ săn không thể qua mắt ong bắp cày, nó phải chịu cái chết chậm đau đớn bị ấu trùng ong ăn sống từ trong." trong bài báo trên)

- Nội dung (ví dụ "Jeremy Wittber, một cư dân ở Perth, ... từ bên trong để phát triển.". Nội dung này gồm 5 đoạn văn: đoạn 1 từ "Jeremy Wittber" đến "mặt đất", đoạn 2 từ "Nhện thợ săn" đến "bắp cày"..., đoạn 4 từ "Hành vi giao chiến" đến "phát triển.", đoạn 5 là tên tác giả bài báo "Phương Hoa".)

Hàm **getComponents(article)** trong file nhận đối số **article** là nội dung bài báo ở dạng str (như đã download ở bài 8), sau đó in tiêu đề, đoạn giới thiệu và nội dung ra màn hình.

*Hãy sửa lại các hàm này để thay vì in kết quả ra màn hình thì nó trả lại một **tuple** gồm 3 thành phần:*

*- Thành phần thứ nhất là tiêu đề ở dạng str*

*- Thành phần thứ hai là đoạn giới thiệu ở dạng str*

*- Thành phần thứ ba là một list, mỗi phần tử của list là một đoạn văn trong nội dung ở dạng str.*

*Hãy tìm cách "làm sạch kết quả", tức xoá những dấu cách, dấu "\r", "\t", "\n" không cần thiết.*

Ví dụ trong bài báo trên, kết quả cần có dạng
("Nhện thợ săn giả chết vẫn bị ong bắp cày truy giết", "Chiêu giả chết của nhện thợ săn không thể qua mắt ong bắp cày, nó phải chịu cái chết chậm đau đớn bị ấu trùng ong ăn sống từ trong.", ["Jeremy Wittber, một cư dân ở Perth, ... mặt đất.", "Nhện thợ săn ... bắp cày", ..., "Phương Hoa"])
Test hàm của bạn bằng đoạn code dưới đây.


```python
s = downloadArticle("https://vnexpress.net/tin-tuc/khoa-hoc/nhen-tho-san-gia-chet-van-bi-ong-bap-cay-truy-giet-3694170.html")
r = getComponents(s)
r
```




    (u'Nh\u1ec7n th\u1ee3 s\u0103n gi\u1ea3 ch\u1ebft v\u1eabn b\u1ecb ong b\u1eafp c\xe0y truy gi\u1ebft',
     u'Chi\xeau gi\u1ea3 ch\u1ebft c\u1ee7a nh\u1ec7n th\u1ee3 s\u0103n kh\xf4ng th\u1ec3 qua m\u1eaft ong b\u1eafp c\xe0y, n\xf3 ph\u1ea3i ch\u1ecbu c\xe1i ch\u1ebft ch\u1eadm \u0111au \u0111\u1edbn b\u1ecb \u1ea5u tr\xf9ng ong \u0103n s\u1ed1ng t\u1eeb trong.',
     [u'Jeremy Wittber, m\u1ed9t c\u01b0 d\xe2n \u1edf Perth, Australia, b\u1eaft g\u1eb7p ong b\u1eafp c\xe0y v\xe0 nh\u1ec7n th\u1ee3 s\u0103n k\u1ecbch chi\u1ebfn trong khi s\u1eeda nh\xe0 h\u1ed3i \u0111\u1ea7u n\u0103m nay, theo Storyful . D\xf9 c\u1ea3 hai con v\u1eadt \u0111\u1ec1u n\u1eb1m trong s\u1ed1 nh\u1eefng lo\xe0i c\xf4n tr\xf9ng \u0111\xe1ng s\u1ee3 nh\u1ea5t, tr\u1eadn chi\u1ebfn d\u01b0\u1eddng nh\u01b0 nghi\xeang v\u1ec1 m\u1ed9t ph\xeda. Ong b\u1eafp c\xe0y t\u1ecf ra hung d\u1eef h\u01a1n h\u1eb3n k\u1ebb th\xf9, kh\xf4ng ng\u1eebng \u0111\u1ea9y nh\u1ec7n th\u1ee3 s\u0103n v\u1ec1 ph\xeda sau v\xe0 ch\xedch n\u1ecdc khi con nh\u1ec7n n\u1eb1m tr\xean m\u1eb7t \u0111\u1ea5t.',
      u'Nh\u1ec7n th\u1ee3 s\u0103n nhi\u1ec1u l\u1ea7n vung ch\xe2n c\u1ed1 xua \u0111u\u1ed5i ong b\u1eafp c\xe0y, nh\u01b0ng \u0111\u1ed1i th\u1ee7 c\u1ee7a n\xf3 kh\xf4ng t\u1ecf ra y\u1ebfu th\u1ebf. D\xf9 n\u1ed7 l\u1ef1c ch\u1ea1y tr\u1ed1n, con nh\u1ec7n li\xean t\u1ee5c b\u1ecb \u0111\xe1nh cho ng\xe3 ng\u1eeda v\xe0 h\u1ee9ng th\xeam nh\u1eefng c\xfa ch\xedch \u0111au \u0111i\u1ebfng. Nh\u1ec7n th\u1ee3 s\u0103n ph\u1ea3i d\xf9ng t\u1edbi m\xe1nh kh\xf3e t\u1ef1 v\u1ec7 cu\u1ed1i c\xf9ng l\xe0 cu\u1ed9n tr\xf2n nh\u01b0 qu\u1ea3 b\xf3ng v\xe0 gi\u1ea3 ch\u1ebft nh\u01b0ng v\u1eabn th\u1ea5t b\u1ea1i tr\u01b0\u1edbc ong b\u1eafp c\xe0y.',
      u'K\u1ebft th\xfac tr\u1eadn chi\u1ebfn, nh\u1ec7n th\u1ee3 s\u0103n ng\u1eebng gi\xe3y gi\u1ee5a, ho\xe0n to\xe0n t\xea li\u1ec7t tr\u01b0\u1edbc n\u1ecdc \u0111\u1ed9c c\u1ee7a ong b\u1eafp c\xe0y. N\xf3 n\u1eb1m b\u1ea5t \u0111\u1ed9ng tr\xean m\u1eb7t \u0111\u1ea5t, \u0111\u1ec3 m\u1eb7c k\u1ebb th\u1eafng cu\u1ed9c k\xe9o l\xea \u0111i.',
      u'H\xe0nh vi giao chi\u1ebfn gi\u1eefa hai lo\xe0i kh\xf4ng hi\u1ebfm g\u1eb7p, v\xe0 th\u01b0\u1eddng d\u1eabn t\u1edbi k\u1ebft c\u1ee5c \u0111au \u0111\u1edbn cho nh\u1ec7n th\u1ee3 s\u0103n. Sau khi b\u1ecb ong b\u1eafp c\xe0y k\xe9o v\xe0o t\u1ed5, n\xf3 s\u1ebd tr\u1edf th\xe0nh l\u1ed3ng \u1ea5p s\u1ed1ng \u0111\u1ec3 k\u1ebb th\xf9 \u0111\u1ebb tr\u1ee9ng b\xean trong c\u01a1 th\u1ec3. \u1ea4u tr\xf9ng ong s\u1ebd \u0103n d\u1ea7n c\u01a1 th\u1ec3 nh\u1ec7n t\u1eeb b\xean trong \u0111\u1ec3 ph\xe1t tri\u1ec3n.',
      u'Ph\u01b0\u01a1ng Hoa'])



### Bài 10. Lưu dữ liệu
Bây giờ ta đã có các bài báo dưới dạng tiêu đề, đoạn giới thiệu và nội dung. Ta muốn lưu chúng vào một file.

*Hãy viết hàm **saveArticles(title_file, content_file, data_folder)** nhận 3 đối số:*

*- **title_file**: một str, tên tập tin chứa các đường dẫn đến các bài báo như đã thực hiện ở bài 7, bước 2. Tập tin này cần nằm trong thư mục **data_folder**.*

*- **content_file**, một str, tên tập tin mà ta sẽ lưu các bài báo vào. Tập tin này cũng cần nằm trong thư mục **data_folder**.*

*- **data_folder**, một str, tên thư mục chứa dữ liệu.*

và thực hiện việc:

*- Đọc **title_file***

*- Với mỗi đường dẫn trong **title_file**, download bài báo từ đường dẫn tương ứng và lấy ra tiêu đề, đoạn giới thiệu và nội dung bài báo (dùng bài 9).*

*- Cuối cùng, lưu mỗi bộ ba (tiêu đề, đoạn giới thiệu, nội dung) dưới dạng một hàng trong file mới **content_file**, theo cú pháp: tiêu đề cách đoạn giới thiệu bởi 2 dấu tab (\t), đoạn giới thiệu cách nội dung bởi 2 dấu tab, và hai đoạn văn liên tiếp trong nội dung cách nhau bởi 1 dấu tab. *

<center>[Tiêu đề]\t\t[Đoạn giới thiệu]\t\t[Đoạn 1 nội dung]\t[Đoạn 2 nội dung]\t...[Đoạn cuối nội dung]\n</center>

*- Nếu bài báo nào có lỗi (không có nội dung hoặc tiêu đề, đoạn giới thiệu), bỏ qua, không cần ghi vào file.*

Lưu ý: trong câu lệnh ghi vào file, bạn có thể dùng **some_file.write(some_string.encode('utf-8'))** để hiển thị đúng tiếng Việt.

Test hàm của bạn bằng đoạn code dưới đây (Mất khoảng 1 phút. Các dòng trên chỉ là các dòng print trong hàm để dễ theo dõi).


```python
prepareDataFolder("Data")
saveLinksFromTitlePages(THOI_SU, "12/02/2018", "14/02/2018", "Science_Jan2018_Titles.txt", "Data")
saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
f = open("Data/Science_Jan2018_Articles.txt").readlines()
f[0]
```

    Folder 'Data' already existed.
    Downloading page 1
    Downloading page 2
    Downloading page 3
    ('Downloading %s', 'https://vnexpress.net/photo/thoi-su/ben-xe-ha-noi-vang-tanh-ngay-giap-tet-3712088.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/nha-xe-ha-noi-than-e-khach-ve-que-an-tet-3712083.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/oto-camry-tong-lien-hoan-nam-xe-tren-pho-ha-noi-3712086.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/csgt-mien-tay-phat-nuoc-suoi-khan-lanh-cho-dan-ve-que-an-tet-3712084.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/ga-sai-gon-dong-nghit-nguoi-ve-que-don-tet-3711933.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/giao-thong/20-duong-day-nong-phan-anh-giao-thong-dip-tet-3712061.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/lang-bien-khoac-ao-bich-hoa-don-tet-3712013.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/ca-nuoc-nang-dep-dip-tet-nguyen-dan-3712027.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/gan-800-000-nguoi-ngheo-duoc-nhan-gao-ho-tro-dip-tet-3711904.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/cho-neu-lang-bien-o-quang-binh-nhon-nhip-ngay-tet-3711910.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/ha-noi-khong-to-chuc-pho-di-bo-dip-tet-nguyen-dan-3711889.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/bo-van-hoa-yeu-cau-khong-to-chuc-hau-dong-nhu-ca-nhac-duong-pho-3711885.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/nhieu-tai-xe-dinh-dinh-tren-quoc-lo-o-nghe-an-3711854.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/tran-ra-giua-duong-buon-ban-tren-pho-thoi-trang-sai-gon-3711882.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/cu-ba-92-tuoi-cung-con-chau-goi-banh-tet-don-tet-3711637.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/ong-le-phuoc-hoai-bao-bi-xoa-ten-khoi-danh-sach-dang-vien-3711852.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/bao-sanba-do-bo-philippines-kha-nang-suy-yeu-tren-bien-dong-3711861.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/cuc-chong-tham-nhung-nhan-nhieu-cuoc-goi-ve-qua-tet-trai-quy-dinh-3711766.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/rung-hoa-anh-dao-trong-hoi-hoa-xuan-vung-tau-3711632.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/y-nghia-cua-ba-chu-de-tren-duong-hoa-nguyen-hue-3711623.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/hue-ban-sung-than-cong-tren-ky-dai-3711607.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/be-gai-bon-thang-tuoi-bi-bo-roi-truoc-cong-nha-3711643.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/nguoi-sai-gon-chi-tien-trieu-mua-la-dong-goi-banh-tet-3711570.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/loi-chuc-tet-cua-linh-dao-truong-sa-gui-dat-lien-3711542.html')
    ('Downloading %s', 'https://vnexpress.net/photo/thoi-su/nguoi-ha-noi-chen-chan-mua-sam-tren-via-he-3711624.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/ong-nguyen-thien-nhan-2018-la-thoi-co-phat-trien-tp-hcm-3710955.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/giao-duc/con-gai-nguoi-lao-cong-nhan-hoc-bong-6-ty-dong-cua-dai-hoc-my-3711163.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/cong-nhan-dong-nai-tranh-tai-mua-lan-mung-xuan-mau-tuat-3711463.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/giao-duc/giao-vien-tay-hat-ngay-tet-que-em-tang-hoc-tro-3711565.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/deo-noi-binh-thuan-lam-dong-sat-lo-4-thang-van-chua-sua-xong-3711551.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/le-hoi-den-hung-2018-co-5-tinh-cung-to-chuc-3711418.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/trung-ve-bui-tien-dung-ky-tang-anh-gay-quy-giup-nguoi-ngheo-don-tet-3711371.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/100-canh-sat-can-tho-gom-rac-tren-duong-pho-ngay-giap-tet-3711396.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/ao-va-bong-doi-tuyen-u23-tang-thu-tuong-duoc-dau-gia-20-ty-dong-3711377.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/nhieu-cong-nghe-lan-dau-xuat-hien-tren-duong-hoa-nguyen-hue-3711294.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/no-lo-luyen-phoi-thep-o-ha-noi-hai-nguoi-nguy-kich-3711323.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/giao-thong/lat-xe-khach-hai-nguoi-tu-vong-3711203.html')
    ('Downloading %s', 'https://vnexpress.net/tin-tuc/thoi-su/bao-sanba-nhieu-kha-nang-tan-tren-bien-dong-3711232.html')
    ('Downloading %s', 'https://vnexpress.net/photo/thoi-su/nguoi-ha-noi-chi-tien-trieu-lam-dep-thu-cung-don-tet-3711212.html')
    ('Downloading %s', 'https://video.vnexpress.net/tin-tuc/xa-hoi/spa-cho-thu-cung-dong-khach-tu-sang-den-dem-3711146.html')
    




    'B\xe1\xba\xbfn xe H\xc3\xa0 N\xe1\xbb\x99i v\xe1\xba\xafng tanh ng\xc3\xa0y gi\xc3\xa1p T\xe1\xba\xbft\t\tB\xe1\xba\xbfn xe M\xe1\xbb\xb9 \xc4\x90\xc3\xacnh, Gi\xc3\xa1p B\xc3\xa1t... b\xe1\xba\xa5t ng\xe1\xbb\x9d v\xe1\xba\xafng kh\xc3\xa1ch do th\xc3\xb3i quen di chuy\xe1\xbb\x83n c\xe1\xbb\xa7a ng\xc6\xb0\xe1\xbb\x9di d\xc3\xa2n thay \xc4\x91\xe1\xbb\x95i.\t\tH\xc3\xb4m nay (28 T\xe1\xba\xbft) l\xc3\xa0 ng\xc3\xa0y l\xc3\xa0m vi\xe1\xbb\x87c cu\xe1\xbb\x91i c\xc3\xb9ng tr\xc6\xb0\xe1\xbb\x9bc khi b\xc6\xb0\xe1\xbb\x9bc v\xc3\xa0o k\xe1\xbb\xb3 ngh\xe1\xbb\x89 T\xe1\xba\xbft Nguy\xc3\xaan \xc4\x91\xc3\xa1n 2018. Tuy nhi\xc3\xaan, b\xe1\xba\xbfn xe M\xe1\xbb\xb9 \xc4\x90\xc3\xacnh, Gi\xc3\xa1p B\xc3\xa1t th\xc6\xb0a th\xe1\xbb\x9bt. C\xc3\xa1c tuy\xe1\xba\xbfn ph\xe1\xbb\x91 Ph\xe1\xba\xa1m H\xc3\xb9ng, Khu\xe1\xba\xa5t Duy Ti\xe1\xba\xbfn, Gi\xe1\xba\xa3i Ph\xc3\xb3ng th\xc6\xb0\xe1\xbb\x9dng \xc3\xb9n t\xe1\xba\xafc v\xc3\xa0o c\xc3\xa1c d\xe1\xbb\x8bp ngh\xe1\xbb\x89 l\xe1\xbb\x85 nay c\xc5\xa9ng th\xc3\xb4ng tho\xc3\xa1ng.\tAnh l\xc3\xadnh v\xe1\xbb\x81 Qu\xe1\xba\xa3ng Tr\xe1\xbb\x8b loay hoay t\xc3\xacm ng\xc6\xb0\xe1\xbb\x9di b\xc3\xa1n v\xc3\xa9 \xe1\xbb\x9f b\xe1\xba\xbfn xe M\xe1\xbb\xb9 \xc4\x90\xc3\xacnh. Tr\xc6\xb0\xe1\xbb\x9bc \xc4\x91\xc3\xb3 anh \xc4\x91\xc3\xa3 sang b\xe1\xba\xbfn N\xc6\xb0\xe1\xbb\x9bc Ng\xe1\xba\xa7m song kh\xc3\xb4ng c\xc3\xb2n xe. Anh ph\xe1\xba\xa3i v\xe1\xba\xadt v\xe1\xbb\x9d ch\xe1\xbb\x9d mua v\xc3\xa9 xe B\xe1\xba\xafc Nam \xc4\x91\xe1\xbb\x83 v\xe1\xbb\x81 qua Qu\xe1\xba\xa3ng Tr\xe1\xbb\x8b.\tG\xe1\xba\xa7n 18h, h\xc3\xa0nh lang gi\xe1\xbb\xafa b\xe1\xba\xbfn xe M\xe1\xbb\xb9 \xc4\x90\xc3\xacnh v\xe1\xba\xafng v\xe1\xba\xbb h\xc6\xa1n c\xe1\xba\xa3 ng\xc3\xa0y th\xc6\xb0\xe1\xbb\x9dng. L\xc6\xb0\xe1\xbb\xa3ng ng\xc6\xb0\xe1\xbb\x9di di chuy\xe1\xbb\x83n r\xe1\xba\xa5t \xc3\xadt. \xc3\x94ng Nguy\xe1\xbb\x85n Quang Nam (qu\xc3\xaa Tuy\xc3\xaan Quang) cho bi\xe1\xba\xbft d\xe1\xbb\x8bp ngh\xe1\xbb\x89 l\xe1\xbb\x85 n\xc3\xa0o c\xc5\xa9ng v\xe1\xbb\x81 qu\xc3\xaa song ch\xc6\xb0a l\xe1\xba\xa7n n\xc3\xa0o \xc3\xb4ng l\xe1\xba\xa1i \xc4\x91\xc6\xb0\xe1\xbb\xa3c th\xe1\xba\xa3nh th\xc6\xa1i nh\xc6\xb0 v\xe1\xba\xady. "H\xc3\xb4m nay xe chuy\xe1\xba\xbfn cu\xe1\xbb\x91i trong ng\xc3\xa0y v\xe1\xbb\x81 Tuy\xc3\xaan Quang nh\xc6\xb0ng kh\xc3\xa1ch ch\xc6\xb0a l\xe1\xba\xa5p \xc4\x91\xe1\xba\xa7y c\xc3\xa1c gh\xe1\xba\xbf", \xc3\xb4ng Nam n\xc3\xb3i.\tTr\xc3\xaan m\xe1\xbb\x99t chuy\xe1\xba\xbfn xe l\xc3\xbac 19h ng\xc3\xa0y 28 T\xe1\xba\xbft ch\xe1\xbb\x89 c\xc3\xb3 hai ng\xc6\xb0\xe1\xbb\x9di kh\xc3\xa1ch.\tKhu nh\xc3\xa0 ch\xe1\xbb\x9d b\xe1\xba\xbfn xe M\xe1\xbb\xb9 \xc4\x90\xc3\xacnh l\xc6\xb0a th\xc6\xb0a v\xc3\xa0i ng\xc6\xb0\xe1\xbb\x9di d\xc3\xa2n \xc4\x91ang \xc4\x91\xe1\xbb\xa3i xe.\tNhi\xe1\xbb\x81u nh\xc3\xa0 xe n\xe1\xba\xb1m im l\xc3\xacm, trong xe t\xe1\xbb\x91i om kh\xc3\xb4ng b\xe1\xba\xadt \xc4\x91i\xe1\xbb\x87n \xc4\x91\xc3\xb3n kh\xc3\xa1ch. Anh H\xe1\xbb\x93ng, ch\xe1\xbb\xa7 xe ch\xe1\xba\xa1y tuy\xe1\xba\xbfn Tam \xc4\x90i\xe1\xbb\x87p (Ninh B\xc3\xacnh) cho bi\xe1\xba\xbft, x\xc3\xa1c \xc4\x91\xe1\xbb\x8bnh ng\xe1\xbb\xa7 l\xe1\xba\xa1i xe t\xe1\xbb\x91i nay \xc4\x91\xe1\xbb\x83 ch\xe1\xbb\x9d kh\xc3\xa1ch ch\xe1\xbb\xa9 v\xc3\xa0i gi\xe1\xbb\x9d \xc4\x91\xe1\xbb\x93ng h\xe1\xbb\x93 m\xe1\xbb\x9bi c\xc3\xb3 hai ng\xc6\xb0\xe1\xbb\x9di kh\xc3\xa1ch. "Nh\xc6\xb0 v\xe1\xba\xady s\xe1\xba\xbd l\xe1\xbb\x97 ti\xe1\xbb\x81n x\xc4\x83ng d\xe1\xba\xa7u m\xe1\xba\xa5t", anh H\xe1\xbb\x93ng n\xc3\xb3i.\tNh\xc3\xa2n vi\xc3\xaan v\xe1\xba\xa1 v\xe1\xba\xadt \xe1\xbb\x9f ph\xc3\xb2ng b\xc3\xa1n v\xc3\xa9. L\xc3\xa3nh \xc4\x91\xe1\xba\xa1o b\xe1\xba\xbfn xe Gi\xc3\xa1p B\xc3\xa1t cho bi\xe1\xba\xbft, d\xe1\xbb\x8bp T\xe1\xba\xbft Nguy\xc3\xaan \xc4\x91\xc3\xa1n n\xc4\x83m nay l\xc6\xb0\xe1\xbb\xa3ng kh\xc3\xa1ch v\xe1\xba\xafng nhi\xe1\xbb\x81u so v\xe1\xbb\x9bi c\xc3\xa1c n\xc4\x83m tr\xc6\xb0\xe1\xbb\x9bc. Nguy\xc3\xaan nh\xc3\xa2n m\xe1\xbb\x99t ph\xe1\xba\xa7n l\xc3\xa0 do ng\xc6\xb0\xe1\xbb\x9di d\xc3\xa2n th\xc6\xb0\xe1\xbb\x9dng \xc4\x91i xe t\xe1\xbb\xb1 thu\xc3\xaa b\xc3\xaan ngo\xc3\xa0i, \xc3\xadt ng\xc6\xb0\xe1\xbb\x9di \xc4\x91i xe kh\xc3\xa1ch ch\xe1\xba\xa1y theo tuy\xe1\xba\xbfn trong b\xe1\xba\xbfn.\tTheo l\xe1\xbb\x8bch, ng\xc6\xb0\xe1\xbb\x9di lao \xc4\x91\xe1\xbb\x99ng s\xe1\xba\xbd c\xc3\xb3 hai ng\xc3\xa0y ngh\xe1\xbb\x89 tr\xc6\xb0\xe1\xbb\x9bc T\xe1\xba\xbft, g\xe1\xbb\x93m 29 v\xc3\xa0 30 th\xc3\xa1ng 12 \xc3\xa2m l\xe1\xbb\x8bch. Ti\xe1\xba\xbfp \xc4\x91\xc3\xb3, ngh\xe1\xbb\x89 m\xc3\xb9ng m\xe1\xbb\x99t, hai, ba c\xe1\xbb\x99ng v\xe1\xbb\x9bi hai ng\xc3\xa0y ngh\xe1\xbb\x89 b\xc3\xb9, t\xe1\xba\xa5t c\xe1\xba\xa3 l\xc3\xa0 b\xe1\xba\xa3y ng\xc3\xa0y.\tNhi\xe1\xbb\x81u ng\xc6\xb0\xe1\xbb\x9di ch\xe1\xbb\x8dn ph\xc6\xb0\xc6\xa1ng ti\xe1\xbb\x87n l\xc3\xa0 xe m\xc3\xa1y \xc4\x91\xe1\xbb\x83 v\xe1\xbb\x81 qu\xc3\xaa \xc4\x83n T\xe1\xba\xbft thay v\xc3\xac \xc4\x91i xe kh\xc3\xa1ch theo tuy\xe1\xba\xbfn c\xe1\xbb\x91 \xc4\x91\xe1\xbb\x8bnh.\n'



Để nhìn rõ hơn, bạn có thể mở file "Data/Science_Jan2018_Articles.txt", nội dung file cần giống như trong hình sau.

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson3/TD/Fig3.png" width=80%></img>

## Một số thống kê

### Bài 11. Pandas
*Hãy viết hàm **readContent(content_file, data_folder)** đọc dữ liệu từ file **content_file** trong thư mục **data_folder** và trả lại kết quả dưới dạng một DataFrame trong pandas.*

Test hàm của bạn bằng đoạn code dưới đây (nếu cần, chạy lại đoạn code test của bài 10 trước để tạo file dữ liệu.)


```python
r = readContent("Science_Jan2018_Articles.txt", "Data")
r.head()
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
      <td>Bến xe Hà Nội vắng tanh ngày giáp Tết</td>
      <td>Bến xe Mỹ Đình, Giáp Bát... bất ngờ vắng khách...</td>
      <td>Hôm nay (28 Tết) là ngày làm việc cuối cùng tr...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ôtô Camry tông liên hoàn năm xe trên phố Hà Nội</td>
      <td>Tài xế lái Camry tông liên tiếp hai ôtô đi cùn...</td>
      <td>Khoảng 16h ngày 13/2, ôtô Camry di chuyển trên...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20 đường dây nóng phản ánh giao thông dịp Tết</td>
      <td>Gặp bất kỳ rắc rối nào khi tham gia giao thông...</td>
      <td>Ủy ban An toàn giao thông Quốc gia vừa công bố...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cả nước nắng đẹp dịp Tết Nguyên đán</td>
      <td>Thời tiết cả ba miền Bắc-Trung-Nam đều thuận l...</td>
      <td>Do khối không khí lạnh suy yếu, thời tiết ở cá...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gần 800.000 người nghèo được nhận gạo hỗ trợ d...</td>
      <td>Theo quyết định của Thủ tướng Nguyễn Xuân Phúc...</td>
      <td>Ông Lê Văn Thời, Phó tổng cục trưởng Tổng cục ...</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 12. Tác giả các bài báo
*Hãy viết hàm **addAuthorColumn(articles_table)** nhận đối số **articles_table** là một DataFrame trong pandas và thêm một cột chứa trả lại tên tác giả bài báo vào DataFrame mang tên "author", và trả lại DataFrame vừa được thay đổi. (Ta định nghĩa tác giả là đoạn văn cuối cùng của bài báo nếu đoạn văn này dài không quá 20 kí tự, và bằng "" nếu đoạn văn cuối dài hơn 20 kí tự.)*

Test hàm của bạn bằng đoạn code dưới đây (nếu cần, chạy lại đoạn code test của bài 10 trước để tạo file dữ liệu.)


```python
r = readContent("Science_Jan2018_Articles.txt", "Data")
s = addAuthorColumn(r)
s.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bến xe Hà Nội vắng tanh ngày giáp Tết</td>
      <td>Bến xe Mỹ Đình, Giáp Bát... bất ngờ vắng khách...</td>
      <td>Hôm nay (28 Tết) là ngày làm việc cuối cùng tr...</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ôtô Camry tông liên hoàn năm xe trên phố Hà Nội</td>
      <td>Tài xế lái Camry tông liên tiếp hai ôtô đi cùn...</td>
      <td>Khoảng 16h ngày 13/2, ôtô Camry di chuyển trên...</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>20 đường dây nóng phản ánh giao thông dịp Tết</td>
      <td>Gặp bất kỳ rắc rối nào khi tham gia giao thông...</td>
      <td>Ủy ban An toàn giao thông Quốc gia vừa công bố...</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cả nước nắng đẹp dịp Tết Nguyên đán</td>
      <td>Thời tiết cả ba miền Bắc-Trung-Nam đều thuận l...</td>
      <td>Do khối không khí lạnh suy yếu, thời tiết ở cá...</td>
      <td>Xuân Hoa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gần 800.000 người nghèo được nhận gạo hỗ trợ d...</td>
      <td>Theo quyết định của Thủ tướng Nguyễn Xuân Phúc...</td>
      <td>Ông Lê Văn Thời, Phó tổng cục trưởng Tổng cục ...</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hà Nội không tổ chức phố đi bộ dịp Tết Nguyên đán</td>
      <td>Để tạo điều kiện cho nhân dân và du khách đi l...</td>
      <td>UBND quận Hoàn Kiếm (Hà Nội) vừa thông báo sẽ ...</td>
      <td>Võ Hải</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bộ Văn hóa yêu cầu không tổ chức hầu đồng như ...</td>
      <td>Các địa phương được yêu cầu chỉ cho tổ chức hầ...</td>
      <td>Bộ Văn hóa, Thể thao và Du lịch vừa chỉ đạo Sở...</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nhiều tài xế dính đinh trên quốc lộ ở Nghệ An</td>
      <td>Nhiều ôtô khi đi qua quốc lộ 16 ở Nghệ An đã d...</td>
      <td>"Xe đang leo dốc thì thấy lốp có biểu hiện bất...</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ông Lê Phước Hoài Bảo bị xóa tên khỏi danh sác...</td>
      <td>Dự kiến sau Tết Nguyên đán, tỉnh Quảng Nam sẽ ...</td>
      <td>Ngày 13/2, ông Đinh Nguyên Vũ, Bí thư Đảng ủy ...</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bão Sanba đổ bộ Philippines, khả năng suy yếu ...</td>
      <td>Đài quốc tế dự báo ngày 16/2 bão Sanba sẽ tan ...</td>
      <td>Trung tâm dự báo khí tượng thủy văn trung ương...</td>
      <td>Xuân Hoa</td>
    </tr>
  </tbody>
</table>
</div>



### Bài 13. Những từ đơn thông dụng nhất trong tiếng Việt

Giả sử mỗi tiếng/âm tiết trong tiếng Việt đều là một từ, như vậy ta có thể tìm tất cả các từ của một bài báo bằng phương pháp đơn giản sau:

- Xoá hết các kí tự không phải chữ cái như: dấu chấm, dấu phẩy, chấm than, số, ...

- Biến mỗi đoạn văn thành list các từ, sau đó tính tần số của từng từ trên tập hợp tất cả các đoạn văn

Để đơn giản, ta xem như các từ viết hoa khác các từ viết thường, ví dụ "Việt" khác "việt".

*Hãy viết hàm **getSimpleWordFrequency(articles_table)** nhận đối số **articles_table** là một DataFrame của pandas có cột nội dung bài báo, sau đó đọc tất cả các nội dung bài báo và trả lại một từ điển gồm các từ đơn xuất hiện trong ít nhất một bài báo, cùng với tần số của chúng trên tất cả các bài báo trong dữ liệu. Chú ý rằng nếu từ rỗng "" xuất hiện trong từ điển, bạn cần xoá nó khỏi từ điển.*

Test hàm của bạn bằng đoạn code dưới đây (nếu cần, chạy lại đoạn code test của bài 10 trước để tạo file dữ liệu.)


```python
r = readContent("Science_Jan2018_Articles.txt", "Data")
d = getSimpleWordFrequency(r)

# Print the 20 most frequent simple word
for key, value in sorted(d.items(), key=lambda x: -x[1])[:20] :
    print key, value
```

    và 101
    có 82
    người 75
    của 69
    các 68
    cho 68
    được 66
    ngày 65
    trong 60
    không 55
    để 51
    từ 46
    là 45
    đến 44
    thành 43
    về 43
    học 42
    thông 41
    Tết 41
    dân 40
    