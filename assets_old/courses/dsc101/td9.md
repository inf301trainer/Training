
# TD9 - Nhận diện khuôn mặt một số ca sĩ Việt Nam 

# Mô tả

Trong TD này ta tiếp tục sử dụng OpenCV trên Python để nhận diện khuôn mặt một số ca sĩ Việt Nam thông qua hình ảnh chụp chính diện của họ thông qua Support Vector Machine.

Bố cục TD như sau
- Thông qua phương pháp Haar-Cascade, trích xuất khuôn mặt của ca sĩ trong hình ảnh.
- Cũng cùng phương pháp, trích xuất các thành phần của khuôn mặt. Trong phạm vi TD, ta chỉ trích xuất đôi mắt.
- Xây dựng các tập dữ liệu dựa trên khuôn mặt/đôi mắt của người trong ảnh
- Thực hiện huấn luyện và đánh giá với SVM tuyến tính và SVM đa thức
- Chọn các biến quan trọng để giảm số chiều bài toán.

Dữ liệu của bài toán được download tại đây
https://drive.google.com/drive/u/0/folders/14qdG6Txe0osUvTEbFjnxrqqfUk6dTsxR

Trong đó bạn cần sử dụng các hình ảnh trong `RawData/face/format2`

Bạn cần hoàn thành các hàm/method trong file `SingerClassification.py`. File `Constants.py` được sử dụng để import cho `SingerClassification.py`, gồm các hằng số được định nghĩa trước có ý nghĩa như dưới đây.


```python
# -*- coding: utf-8 -*-

# Thư mục chứa các hình ảnh gốc sẽ được sử dụng
RAW_IMAGE_FOLDER = "RawData/face/format2"

# Thư mục chứa các hình ảnh được copy, tập trung và đổi tên
IMAGE_FOLDER = "ImageFolder"

# Thư mục chứa các file text lưu trích xuất khuôn mặt, đôi mắt dưới dạng vector số, output của phần 1
TEXT_DATA_FOLDER = "TextData"

# Định dạng hình ảnh
EXTENSION = ".png"

# File Haar cascade của khuôn mặt
HAARCASCADE_FRONTALFACE_DEFAULT = "Configuration/haarcascade_frontalface_default.xml"

# File Haar cascade của đôi mắt
HAARCASCADE_EYE_DEFAULT = "Configuration/haarcascade_eye.xml"

# Hằng số kiểm tra xem khuôn mặt có được chụp ở tư thế trực diện không
# Khuôn mặt được xem là có tư thế trực diện nếu tâm mắt trái và tâm mắt phải có tung độ không cách nhau quá 3 pixel
HORIZONTAL_CHECK = 3

# File text lưu trích xuất khuôn mặt, đôi mắt dưới dạng vector số, output của phần 1
FACES_DATA = "TextData/Faces.csv"
EYES_DATA = "TextData/Eyes.csv"

# Chuẩn hoá tên các ca sĩ từ thư mục hình ảnh gốc
SINGER_NAME_DICTIONARY = {
    "bao thy" : "BaoThy",
    "chi pu": "ChiPu",
    "dam vinh hung": "DamVinhHung",
    "dan truong": "DanTruong",
    "ha anh tuan": "HaAnhTuan",
    "ho ngoc ha": "HoNgocHa",
    "huong tram": "HuongTram",
    "lam truong": "LamTruong",
    "my tam": "MyTam",
    "No phuoc thing": "NooPhuocThinh",
    "son tung": "SonTung",
    "tuan hung": "TuanHung"
}

# Đánh số thứ tự tên các ca sĩ
SINGER_INDEX_DICTIONARY = {
    "BaoThy": 0,
    "ChiPu": 1,
    "DamVinhHung": 2,
    "DanTruong": 3,
    "HaAnhTuan": 4,
    "HoNgocHa": 5,
    "HuongTram": 6,
    "LamTruong": 7,
    "MyTam": 8,
    "NooPhuocThinh": 9,
    "SonTung": 10,
    "TuanHung": 11
}

# Sau khi xử lí các hình ảnh và sắp xếp theo thứ tự ABC, các hình ảnh có thứ tự 0 đến 220 sẽ thuộc ca sĩ BaoThy
# v.v.
SINGER_IMAGE_RANGE = {
    "BaoThy": range(0, 221),
    "ChiPu": range(221, 521),
    "DamVinhHung": range(521, 676),
    "DanTruong": range(676, 811),
    "HaAnhTuan": range(811, 921),
    "HoNgocHa": range(921, 1109),
    "HuongTram": range(1109, 1327),
    "LamTruong": range(1327, 1407),
    "MyTam": range(1407, 1580),
    "NooPhuocThinh": range(1580, 1820),
    "SonTung": range(1820, 2020),
    "TuanHung": range(2020, 2226)
}
```

Đoạn code dưới đây import lời giải để minh hoạ.


```python
from SingerClassification_Solution import *
```

## Phần 1 - Class `Face` - Trích xuất khuôn mặt và đôi mắt

### Bài 1. Copy và tập trung hình ảnh vào một thư mục

Thư mục chứa hình ảnh gốc nằm tại `RawData/face/format2` gồm 12 thư mục con như sau
<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/F1.png"></img>

Tên các thư mục chính là các key của dict `SINGER_NAME_DICTIONARY` đã được định nghĩa trong `Constants.py`. Mỗi thư mục này chứa các hình ảnh ca sĩ có định dạng `.png` và kích thước `96 pixel x 96 pixel`. Tên các hình ảnh được đánh số không liên tục (vì có thể một số hình ảnh đã bị xoá do chất lượng không tốt hoặc ảnh không trực diện)

Để dễ làm việc về sau với các hình ảnh, ta muốn tập trung tất cả hình ảnh về một thư mục duy nhất và đặt lại tên để tên các hình ảnh có chỉ số được liên tục.

***Hãy viết hàm `prepareImageFolder(raw_image_folder, image_folder)` copy các hình ảnh trong tất cả các thư mục con (được đặt tên bằng các key trong `SINGER_NAME_DICTIONARY`) của `raw_image_folder` sang file `image_folder` và đặt lại tên các file dưới dạng `TenCaSi_Maso.png`, trong đó `TenCaSi` là value trong `SINGER_NAME_DICTIONARY`, các chỉ số bắt đầu từ 0. Ví dụ các hình ảnh từ thư mục `bao thy` sẽ có tên `BaoThy_0.png`, ..., `BaoThy_220.png`.***

Đoạn code sau giúp test hàm của bạn.


```python
prepareImageFolder(RAW_IMAGE_FOLDER, IMAGE_FOLDER)
```


```python
import pandas as pd
l = os.listdir(IMAGE_FOLDER) # Liệt kê thư mục mới
pd.DataFrame(l)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BaoThy_0.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BaoThy_1.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BaoThy_10.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BaoThy_100.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BaoThy_101.png</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BaoThy_102.png</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BaoThy_103.png</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BaoThy_104.png</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BaoThy_105.png</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BaoThy_106.png</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BaoThy_107.png</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BaoThy_108.png</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BaoThy_109.png</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BaoThy_11.png</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BaoThy_110.png</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BaoThy_111.png</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BaoThy_112.png</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BaoThy_113.png</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BaoThy_114.png</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BaoThy_115.png</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BaoThy_116.png</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BaoThy_117.png</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BaoThy_118.png</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BaoThy_119.png</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BaoThy_12.png</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BaoThy_120.png</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BaoThy_121.png</td>
    </tr>
    <tr>
      <th>27</th>
      <td>BaoThy_122.png</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BaoThy_123.png</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BaoThy_124.png</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2197</th>
      <td>TuanHung_72.png</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>TuanHung_73.png</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>TuanHung_74.png</td>
    </tr>
    <tr>
      <th>2200</th>
      <td>TuanHung_75.png</td>
    </tr>
    <tr>
      <th>2201</th>
      <td>TuanHung_76.png</td>
    </tr>
    <tr>
      <th>2202</th>
      <td>TuanHung_77.png</td>
    </tr>
    <tr>
      <th>2203</th>
      <td>TuanHung_78.png</td>
    </tr>
    <tr>
      <th>2204</th>
      <td>TuanHung_79.png</td>
    </tr>
    <tr>
      <th>2205</th>
      <td>TuanHung_8.png</td>
    </tr>
    <tr>
      <th>2206</th>
      <td>TuanHung_80.png</td>
    </tr>
    <tr>
      <th>2207</th>
      <td>TuanHung_81.png</td>
    </tr>
    <tr>
      <th>2208</th>
      <td>TuanHung_82.png</td>
    </tr>
    <tr>
      <th>2209</th>
      <td>TuanHung_83.png</td>
    </tr>
    <tr>
      <th>2210</th>
      <td>TuanHung_84.png</td>
    </tr>
    <tr>
      <th>2211</th>
      <td>TuanHung_85.png</td>
    </tr>
    <tr>
      <th>2212</th>
      <td>TuanHung_86.png</td>
    </tr>
    <tr>
      <th>2213</th>
      <td>TuanHung_87.png</td>
    </tr>
    <tr>
      <th>2214</th>
      <td>TuanHung_88.png</td>
    </tr>
    <tr>
      <th>2215</th>
      <td>TuanHung_89.png</td>
    </tr>
    <tr>
      <th>2216</th>
      <td>TuanHung_9.png</td>
    </tr>
    <tr>
      <th>2217</th>
      <td>TuanHung_90.png</td>
    </tr>
    <tr>
      <th>2218</th>
      <td>TuanHung_91.png</td>
    </tr>
    <tr>
      <th>2219</th>
      <td>TuanHung_92.png</td>
    </tr>
    <tr>
      <th>2220</th>
      <td>TuanHung_93.png</td>
    </tr>
    <tr>
      <th>2221</th>
      <td>TuanHung_94.png</td>
    </tr>
    <tr>
      <th>2222</th>
      <td>TuanHung_95.png</td>
    </tr>
    <tr>
      <th>2223</th>
      <td>TuanHung_96.png</td>
    </tr>
    <tr>
      <th>2224</th>
      <td>TuanHung_97.png</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>TuanHung_98.png</td>
    </tr>
    <tr>
      <th>2226</th>
      <td>TuanHung_99.png</td>
    </tr>
  </tbody>
</table>
<p>2227 rows × 1 columns</p>
</div>



Kết quả cần chứa 2227 hình ảnh với định dạng tên như liệt kê trên và như thư mục dưới đây:
<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/F2.png"></img>

### Bài 2. Class `Face`

Ta xây dựng class `Face` sao cho mỗi instance của nó biểu diễn một hình ảnh, với trích xuất khuôn mặt và đôi mắt. Mỗi instance của class `Face` chứa các attribute sau:

- `image_path`: đường dẫn đến hình ảnh (ví dụ `ImageFolder/BaoThy_0.png`)
- `image_name`: tên hình ảnh, là tên file không chứa phần định dạng (ví dụ `BaoThy_0`)
- `gray_image`: numpy array biểu diễn hình ảnh dạng trắng đen (được gọi từ `cv2.imread(some_image, 0)`)
- `face_positions`, `eye_positions`: vị trí các khuôn mặt và con mắt trong hình ảnh (sẽ được tìm hiểu sau)
- `faces`, `eyes`: trích xuất các hình ảnh và con mắt, là một numpy array con của `gray_image`
- `normalized_faces`, `normalized_eyes`: numpy array biểu diễn khuôn mặt được chuẩn hoá về 64 x 64; con mắt được chuẩn hoá về 32 x 32.

Trước hết, ta xây dựng 3 attribute đầu tiên trong hàm `__init__`.

***Trong class `Face`, hãy viết method `__init__(self, image_path)` nhận đối số `image_path` là đường dẫn của hình ảnh và xây dựng instance thuộc class `Face` có các attributes `image_path`, `image_name`, `gray_image` như đã mô tả.***

Đoạn code dưới đây giúp test hàm của bạn.


```python
face = Face(IMAGE_FOLDER + "/BaoThy_0.png")
print(face.gray_image)
```

    [[204 204 203 ...  97  89  84]
     [205 204 204 ... 100  92  86]
     [205 205 205 ... 105  98  91]
     ...
     [217 216 217 ... 162 160 159]
     [215 215 215 ... 160 158 158]
     [215 215 214 ... 157 156 157]]
    


```python
face.image_path
```




    'ImageFolder/BaoThy_0.png'




```python
face.image_name
```




    'BaoThy_0'



### Bài 3. Vẽ hình bằng `matplotlib`

Phần này minh hoạ hình ảnh với `matplotlib`.

***Trong class `Face`, hãy viết method `draw(self, mode, index)` trong đó `mode` là một str, `index` là một int (mà ta chưa cần quan tâm trong bài này) sao cho nếu `mode` nhận giá trị `"full_gray"` thì khi gọi hàm `self.draw(mode = "full_gray")` sẽ trả lại dạng trắng đen của hình ảnh đang dùng.***

Đoạn code dưới đây giúp test hàm của bạn.


```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png")
face.draw(mode = "full_gray")
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_15_0.png)


### Bài 4. Xác định vị trí khuôn mặt và trích xuất khuôn mặt

Mặc dù các hình ảnh được xem là lấy khuôn mặt một cách trực diện, có những sự sai khác nhất định về vị trí tương đối của khuôn mặt trong hình ảnh, cũng như một số khuôn mặt không thực sự trực diện, chẳng hạn như hình dưới đây.


```python
face = Face(IMAGE_FOLDER + "/BaoThy_0.png")
face.draw(mode = "full_gray")
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_17_0.png)


Vì vậy, ta sử dụng phương pháp Haar Feature-based Cascade để "trích xuất" gương mặt theo phương pháp tương đồng giữa 1 zone hình ảnh với thác Haar tương ứng. Bạn có thể xem chi tiết về phương pháp và cách dùng ở đây: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale

***Trong class `Face`, hãy viết method `detectFaces(self)` tìm tất cả các bộ `(x, y, w, h)` xác định vị trí khuôn mặt như output của `detectMultiScale` trong cv2 (tức `(x, y), (x + w, y), (x, y + h), (x + w, y + h)` sẽ là toạ độ 4 đỉnh của vùng khuôn mặt) và gán vào attribute `self.face_positions` array tất cả các bộ 4 tìm được. Hình ảnh xác định được `N` khuôn mặt thì array sẽ có chiều `N x 4`***.

Bạn có thể sử dụng `scaleFactor = Face.HAAR_SCALE_FACTOR, minNeighbors = Face.HAAR_MIN_NEIGHBORS, minSize = Face.HAAR_MIN_SIZE` cho hàm `detectMultiScale` và file `Face.HAARCASCADE_FRONTALFACE` cho cascade của khuôn mặt.

Đoạn code dưới đây giúp test hàm của bạn.


```python
face = Face(IMAGE_FOLDER + "/BaoThy_10.png") # Cần ra 1 khuôn mặt, array có dạng 4 x 1
face.face_positions
```




    array([[11,  5, 74, 74]], dtype=int32)




```python
face = Face(IMAGE_FOLDER + "/BaoThy_0.png") # Cần ra 0 khuôn mặt, array có dạng 4 x 0 = 0
face.face_positions
```




    ()



***Tiếp theo, cũng trong method `detectFaces(self)`, gán các numpy array con của `self.gray_image` biểu diễn khuôn mặt tương ứng vào attribute `faces` dưới dạng 1 list. Hình ảnh xác định được bao nhiêu khuôn mặt thì list sẽ có bấy nhiêu phần tử, mỗi phần tử là một array.***

Đoạn code dưới đây giúp test hàm của bạn.


```python
face = Face(IMAGE_FOLDER + "/BaoThy_10.png") 
face.faces # List chứa 1 phần tử: 1 array 74 x 74
```




    [array([[ 29,  32,  26, ...,  99, 110, 113],
            [ 29,  31,  27, ...,  99, 114, 103],
            [ 27,  28,  29, ..., 104, 108, 107],
            ...,
            [ 15,  33,  26, ...,  14,  16,  22],
            [ 28,  44,  32, ...,  13,  15,  20],
            [ 55,  55,  46, ...,  13,  14,  18]], dtype=uint8)]




```python
face = Face(IMAGE_FOLDER + "/BaoThy_0.png") 
face.faces # List chứa 0 phần tử
```




    []



***Cuối cùng, thêm vào method `draw(self, mode, index)` đã viết ở bài 3 để khi `mode` nhận giá trị `"face"` thì kết quả nhận được là biểu diễn hình ảnh của trích xuất khuôn mặt thứ `index` trong list `self.faces`, còn nếu `mode` nhận giá trị là `face_marked` thì kết quả là toàn bộ hình ảnh, nhưng phần khuôn mặt được trích xuất được đánh dấu (chẳng hạn bằng cách đóng khung)***

Đoạn code dưới đây lặp lại bài 3 khi `mode="full_gray"`


```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png") 
face.draw(mode = "full_gray") 
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_25_0.png)


Đoạn code dưới đây minh hoạ khi `mode="face"`


```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png") 
face.draw(mode = "face", index = 0) 
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_27_0.png)


Đoạn code dưới đây đánh dấu (đóng khung) khuôn mặt trên hình gốc


```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png") 
face.draw(mode = "face_marked", index = 0) 
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_29_0.png)


### Bài 5 - Chuẩn hoá kích thước khuôn mặt

***Áp dụng `cv2.resize()` đã thực hiện ở các TD trước, chuẩn hoá kích thước các khuôn mặt trong list `self.faces` về 64 x 64 pixel và lưu chúng vào list `self.normalized_faces` trong class `Face`.***

***Tiếp tục viết thêm vào method `draw(self, mode, index)` để khi `mode` nhận giá trị `"normalized_face"` thì nhận được trích xuất khuôn mặt được vẽ bởi matplotlib với kích thước 64 x 64.*** 


```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png") 
face.normalized_faces
```




    [array([[ 28,  25,  25, ...,  37,  47, 102],
            [ 23,  22,  25, ...,  40,  44,  78],
            [ 20,  20,  26, ...,  45,  50,  64],
            ...,
            [117,  66,  40, ...,  75,  25,  57],
            [130, 101,  68, ...,  99,  21,  41],
            [171, 147, 112, ..., 147,  35,  15]], dtype=uint8)]




```python
face.normalized_faces[0].shape # Kết quả là 64 x 64
```




    (64, 64)




```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png") 
face.draw(mode = "normalized_face", index = 0) # Kết quả là 1 hình có kích thước 64 pixel x 64 pixel
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_33_0.png)


### Bài 6 - Xác định vị trí mắt và trích xuất mắt

***Trong class `Face`, hãy viết method `detectFaces(self)` tìm tất cả các bộ `(i, (x, y, w, h))` xác định vị trí mắt, trong đó `i` là số thứ tự khuôn mặt chứa mắt trong `self.faces`; còn `(x, y, w, h)` như output của `detectMultiScale` trong cv2 (tức `(x, y), (x + w, y), (x, y + h), (x + w, y + h)` sẽ là toạ độ 4 đỉnh của vùng mắt); và gán list tất cả các bộ `(i, (x, y, w, h))` tìm được vào attribute `self.eye_positions`. Hình ảnh xác định được bao nhiêu con mắt thì `self.eye_positions` sẽ là list sẽ có độ dài bấy nhiêu***.

Ví dụ với hình ảnh `DanTruong_10`, chỉ xác định được 1 mắt.


```python
face = Face(IMAGE_FOLDER + "/DanTruong_10.png")
face.eye_positions
```




    [(0, array([40, 13, 22, 22], dtype=int32))]



Với hình ảnh `DanTruong_4`, chỉ xác định được 2 mắt.


```python
face = Face(IMAGE_FOLDER + "/DanTruong_4.png")
face.eye_positions
```




    [(0, array([37, 15, 22, 22], dtype=int32)),
     (0, array([ 9, 14, 24, 24], dtype=int32))]



Với hình ảnh `DanTruong_1`, thậm chí xác định được nhiều "mắt" hơn


```python
face = Face(IMAGE_FOLDER + "/DanTruong_4.png")
face.eye_positions
```




    [(0, array([37, 15, 22, 22], dtype=int32)),
     (0, array([ 9, 14, 24, 24], dtype=int32))]



***Tiếp theo, cũng trong method `detectFaces(self)`, gán các bộ `(i, X)` vào `self.eyes` trong đó `X` là array con tương ứng các mắt đã xác định được.***

Các đoạn code dưới đây giúp test hàm của bạn


```python
face = Face(IMAGE_FOLDER + "/DanTruong_4.png")
face.eyes # Cần là list 2 phần tử, mỗi phần tử là 1 bộ 1 số nguyên và 1 array)
```




    [(0, array([[214, 216, 217, 217, 216, 214, 213, 213, 214, 213, 209, 210, 211,
              213, 214, 213, 213, 210, 208, 209, 195, 183],
             [218, 219, 218, 218, 216, 213, 209, 204, 209, 225, 209, 198, 204,
              201, 200, 197, 193, 195, 195, 194, 177, 172],
             [216, 213, 210, 210, 211, 212, 211, 207, 206, 211, 207, 192, 180,
              167, 166, 167, 155, 160, 160, 146, 131, 140],
             [210, 206, 203, 204, 206, 205, 200, 195, 182, 171, 177, 166, 142,
              122, 130, 143, 136, 121, 112, 105,  98, 105],
             [199, 198, 199, 200, 197, 185, 173, 165, 147, 124, 112, 103,  94,
               87,  89, 103, 107,  90,  80,  84,  78,  74],
             [191, 194, 197, 195, 184, 163, 140, 127, 115,  96,  67,  66,  72,
               70,  63,  72,  89,  82,  80,  78,  62,  47],
             [190, 195, 195, 185, 167, 141, 120, 107, 101,  95,  82,  89,  91,
               82,  70,  74,  87,  81,  74,  72,  75,  66],
             [194, 195, 187, 169, 148, 130, 121, 120, 120, 125, 137, 145, 132,
              116, 127, 142, 146, 133, 123, 120, 125, 125],
             [197, 189, 174, 154, 139, 135, 140, 150, 167, 177, 177, 176, 166,
              159, 174, 186, 185, 175, 183, 181, 170, 175],
             [193, 178, 161, 147, 142, 146, 158, 169, 170, 163, 149, 142, 152,
              159, 155, 159, 159, 158, 170, 179, 180, 187],
             [185, 165, 151, 146, 151, 156, 165, 163, 136, 124,  99,  66,  97,
              105,  84,  99, 118, 128, 131, 148, 163, 168],
             [186, 162, 150, 151, 158, 159, 154, 141, 120, 150, 126,  41,  46,
               41,  29,  80, 112, 110,  98, 120, 145, 148],
             [193, 167, 155, 157, 160, 154, 133, 129, 144, 165, 144,  83,  71,
               52,  50, 116, 137, 120,  99, 122, 149, 158],
             [189, 164, 153, 156, 163, 161, 142, 148, 175, 167, 163, 159, 150,
              139, 131, 136, 139, 132, 125, 149, 169, 173],
             [194, 168, 154, 155, 167, 176, 175, 177, 184, 181, 177, 175, 172,
              171, 168, 160, 157, 156, 160, 181, 188, 181],
             [205, 180, 161, 157, 169, 185, 196, 192, 184, 185, 184, 183, 182,
              182, 182, 183, 183, 180, 182, 192, 194, 189],
             [215, 193, 171, 162, 167, 183, 197, 194, 186, 188, 187, 187, 188,
              189, 190, 193, 194, 195, 198, 206, 204, 199],
             [221, 204, 182, 169, 165, 174, 185, 190, 190, 191, 191, 193, 195,
              196, 198, 202, 204, 205, 209, 215, 212, 207],
             [219, 207, 189, 174, 167, 169, 176, 186, 191, 190, 192, 194, 197,
              199, 200, 203, 206, 207, 211, 215, 212, 208],
             [213, 204, 192, 180, 175, 174, 179, 187, 191, 191, 192, 194, 197,
              198, 198, 199, 201, 201, 206, 212, 209, 205],
             [212, 204, 194, 188, 185, 185, 188, 193, 195, 195, 196, 196, 197,
              197, 195, 195, 195, 195, 201, 207, 203, 200],
             [215, 204, 196, 192, 192, 193, 197, 203, 206, 206, 207, 208, 208,
              208, 205, 202, 201, 198, 201, 204, 200, 197]], dtype=uint8)),
     (0, array([[173, 172, 185, 197, 198, 198, 198, 196, 195, 199, 205, 205, 201,
              196, 195, 202, 212, 218, 220, 217, 211, 206, 205, 205],
             [176, 175, 189, 198, 191, 184, 182, 184, 189, 197, 206, 210, 213,
              210, 201, 204, 213, 220, 222, 220, 215, 212, 212, 217],
             [186, 183, 184, 178, 166, 157, 153, 159, 169, 177, 185, 192, 196,
              204, 211, 212, 213, 214, 213, 211, 209, 209, 211, 216],
             [195, 185, 165, 144, 129, 119, 113, 118, 125, 134, 140, 147, 152,
              169, 194, 200, 202, 204, 206, 204, 202, 203, 206, 211],
             [191, 170, 140, 115,  98,  83,  78,  81,  86,  89,  94, 101, 109,
              119, 130, 141, 152, 162, 172, 182, 190, 195, 202, 206],
             [175, 137, 102,  82,  75,  68,  66,  71,  74,  72,  70,  71,  74,
               80,  88,  95, 108, 125, 141, 159, 177, 189, 197, 204],
             [155, 112,  78,  65,  66,  66,  71,  80,  84,  85,  84,  81,  79,
               77,  79,  84,  96, 113, 131, 148, 166, 179, 186, 191],
             [144, 112, 100,  99, 107, 115, 124, 136, 143, 140, 137, 132, 125,
              108,  97,  97,  97, 105, 116, 130, 150, 167, 179, 187],
             [142, 126, 139, 149, 163, 173, 181, 189, 195, 197, 195, 189, 183,
              176, 165, 157, 147, 138, 134, 134, 144, 163, 178, 193],
             [153, 155, 174, 181, 190, 193, 189, 187, 180, 169, 162, 155, 148,
              151, 161, 173, 175, 172, 164, 156, 153, 160, 171, 181],
             [170, 179, 183, 181, 185, 183, 172, 148, 140, 142, 110,  88,  79,
               70,  84, 114, 136, 156, 167, 165, 163, 163, 168, 170],
             [182, 187, 187, 181, 174, 158, 138, 103, 118, 147,  93,  78,  80,
               58,  58,  96, 143, 141, 144, 157, 163, 164, 165, 167],
             [175, 177, 180, 174, 162, 141, 119,  98, 131, 156, 104,  79,  73,
               63,  65, 108, 160, 134, 123, 143, 158, 161, 160, 167],
             [163, 164, 165, 159, 151, 140, 129, 119, 150, 170, 137, 115, 106,
              107, 111, 136, 165, 150, 138, 144, 161, 160, 161, 173],
             [174, 171, 171, 170, 169, 170, 167, 157, 167, 176, 171, 170, 172,
              173, 172, 177, 183, 182, 170, 167, 180, 169, 166, 178],
             [192, 183, 186, 194, 197, 202, 199, 186, 172, 173, 185, 201, 202,
              195, 195, 183, 182, 199, 190, 189, 199, 180, 169, 178],
             [201, 189, 197, 209, 209, 206, 201, 194, 181, 180, 190, 196, 198,
              194, 191, 173, 172, 191, 184, 185, 192, 175, 163, 172],
             [192, 187, 194, 202, 203, 203, 203, 201, 194, 194, 191, 179, 188,
              199, 199, 190, 193, 203, 194, 186, 184, 179, 174, 181],
             [193, 195, 192, 194, 199, 204, 207, 207, 203, 205, 203, 193, 198,
              203, 203, 203, 206, 209, 203, 188, 182, 190, 197, 201],
             [197, 201, 197, 199, 207, 215, 214, 209, 205, 203, 211, 216, 208,
              201, 200, 206, 207, 206, 207, 189, 181, 190, 196, 201],
             [194, 195, 202, 210, 214, 212, 205, 208, 214, 205, 212, 223, 212,
              198, 195, 202, 202, 205, 213, 203, 195, 197, 196, 198],
             [198, 198, 202, 208, 209, 204, 198, 209, 224, 212, 205, 209, 213,
              207, 205, 214, 210, 208, 211, 210, 202, 196, 195, 197],
             [201, 201, 199, 198, 199, 201, 202, 208, 215, 215, 215, 214, 213,
              212, 215, 221, 216, 207, 202, 202, 191, 181, 185, 191],
             [200, 202, 200, 199, 200, 202, 205, 208, 213, 218, 223, 220, 211,
              214, 215, 214, 210, 203, 197, 197, 194, 192, 196, 195]],
            dtype=uint8))]



***Cuối cùng, thêm vào method `draw(self, mode, index)` đã viết ở bài 3 để khi `mode` nhận giá trị `"eye"` thì kết quả nhận được là biểu diễn hình ảnh của trích xuất mắt thứ `index` trong list `self.eyes`, còn nếu `mode` nhận giá trị là `eye_marked` thì kết quả là toàn bộ hình ảnh, nhưng phần mắt được trích xuất được đánh dấu (chẳng hạn bằng cách đóng khung)***


```python
face = Face(IMAGE_FOLDER + "/DanTruong_1.png")
face.draw(mode="eye", index = 0)
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_43_0.png)



```python
face = Face(IMAGE_FOLDER + "/DanTruong_1.png")
face.draw(mode="eye_marked", index = 1)
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_44_0.png)



```python
face = Face(IMAGE_FOLDER + "/DanTruong_1.png")
face.draw(mode="eye_marked", index = 2) # Trường hợp phát hiện nhiều mắt trong cùng 1 zone
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_45_0.png)


### Bài 7 - Chuẩn hoá kích thước mắt

***Áp dụng `cv2.resize()` đã thực hiện ở các TD trước, chuẩn hoá kích thước các khuôn mặt trong list `self.eyes` về 32 x 32 pixel và lưu chúng vào attribute `self.normalized_eyes` dưới dạng 1 list `(i, X)` như `self.eyes`, nhưng `X` là các numpy array 32 x 32.***

***Tiếp tục viết thêm vào method `draw(self, mode, index)` để khi `mode` nhận giá trị `"normalized_eye"` thì nhận được trích xuất mắt thứ `index` được vẽ bởi matplotlib với kích thước 32 x 32.*** 


```python
face = Face(IMAGE_FOLDER + "/DanTruong_4.png")
face.normalized_eyes
```




    [array([[214, 215, 216, ..., 198, 189, 183],
            [216, 217, 217, ..., 189, 182, 177],
            [218, 217, 217, ..., 170, 166, 165],
            ...,
            [212, 208, 202, ..., 205, 203, 201],
            [213, 208, 202, ..., 202, 200, 199],
            [215, 209, 202, ..., 201, 199, 197]], dtype=uint8),
     array([[173, 172, 177, ..., 205, 205, 205],
            [175, 174, 179, ..., 210, 210, 212],
            [180, 179, 181, ..., 211, 213, 217],
            ...,
            [200, 200, 200, ..., 188, 190, 193],
            [201, 201, 200, ..., 187, 190, 192],
            [200, 201, 201, ..., 194, 195, 195]], dtype=uint8)]




```python
face.draw(mode = "normalized_eye", index = 0) # Mắt phải, kích thước 32 x 32
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_48_0.png)



```python
face.draw(mode = "normalized_eye", index = 1) # Mắt trái, kích thước 32 x 32
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_49_0.png)


### Bài 8 - Chuyển các khuôn mặt thành vector

Ta muốn lưu các khuôn mặt dưới dạng các vector số vào một file dữ liệu sao cho mỗi khuôn mặt ứng với một dòng của file, tức toạ độ của một vector có số chiều 4096 (=64 x 64) cách nhau bằng dấu phẩy; sau đó là một số nguyên cho biết chỉ số của ca sĩ trong từ điển `SINGER_INDEX_DICTIONARY` và tên hình ảnh tương ứng. Ví dụ file có dạng sau.

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/F3.png" width = 1000></img>

Nếu quan sát file bằng pandas, ta dễ thấy cấu trúc file.


```python
pd.read_csv("TextData/Faces.csv", sep=",", header=None) #Faces.csv là ví dụ utput của bài
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
      <th>4088</th>
      <th>4089</th>
      <th>4090</th>
      <th>4091</th>
      <th>4092</th>
      <th>4093</th>
      <th>4094</th>
      <th>4095</th>
      <th>4096</th>
      <th>4097</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>155</td>
      <td>140</td>
      <td>109</td>
      <td>62</td>
      <td>27</td>
      <td>26</td>
      <td>28</td>
      <td>30</td>
      <td>52</td>
      <td>83</td>
      <td>...</td>
      <td>114</td>
      <td>103</td>
      <td>76</td>
      <td>50</td>
      <td>56</td>
      <td>61</td>
      <td>89</td>
      <td>171</td>
      <td>2</td>
      <td>DamVinhHung_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>102</td>
      <td>10</td>
      <td>4</td>
      <td>6</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>29</td>
      <td>51</td>
      <td>...</td>
      <td>186</td>
      <td>163</td>
      <td>162</td>
      <td>203</td>
      <td>194</td>
      <td>207</td>
      <td>205</td>
      <td>214</td>
      <td>2</td>
      <td>DamVinhHung_10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>30</td>
      <td>38</td>
      <td>50</td>
      <td>60</td>
      <td>59</td>
      <td>55</td>
      <td>63</td>
      <td>80</td>
      <td>90</td>
      <td>...</td>
      <td>61</td>
      <td>63</td>
      <td>60</td>
      <td>50</td>
      <td>46</td>
      <td>46</td>
      <td>49</td>
      <td>52</td>
      <td>2</td>
      <td>DamVinhHung_100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>76</td>
      <td>118</td>
      <td>143</td>
      <td>148</td>
      <td>152</td>
      <td>158</td>
      <td>163</td>
      <td>168</td>
      <td>171</td>
      <td>...</td>
      <td>36</td>
      <td>56</td>
      <td>65</td>
      <td>65</td>
      <td>66</td>
      <td>60</td>
      <td>62</td>
      <td>75</td>
      <td>2</td>
      <td>DamVinhHung_101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>21</td>
      <td>11</td>
      <td>3</td>
      <td>6</td>
      <td>33</td>
      <td>76</td>
      <td>106</td>
      <td>128</td>
      <td>133</td>
      <td>...</td>
      <td>112</td>
      <td>77</td>
      <td>26</td>
      <td>9</td>
      <td>19</td>
      <td>19</td>
      <td>17</td>
      <td>13</td>
      <td>2</td>
      <td>DamVinhHung_102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>203</td>
      <td>209</td>
      <td>211</td>
      <td>210</td>
      <td>209</td>
      <td>208</td>
      <td>209</td>
      <td>212</td>
      <td>208</td>
      <td>191</td>
      <td>...</td>
      <td>183</td>
      <td>187</td>
      <td>187</td>
      <td>183</td>
      <td>184</td>
      <td>186</td>
      <td>186</td>
      <td>185</td>
      <td>2</td>
      <td>DamVinhHung_103</td>
    </tr>
    <tr>
      <th>6</th>
      <td>214</td>
      <td>213</td>
      <td>219</td>
      <td>213</td>
      <td>188</td>
      <td>143</td>
      <td>93</td>
      <td>58</td>
      <td>57</td>
      <td>66</td>
      <td>...</td>
      <td>175</td>
      <td>175</td>
      <td>174</td>
      <td>171</td>
      <td>170</td>
      <td>168</td>
      <td>166</td>
      <td>164</td>
      <td>2</td>
      <td>DamVinhHung_104</td>
    </tr>
    <tr>
      <th>7</th>
      <td>79</td>
      <td>90</td>
      <td>93</td>
      <td>108</td>
      <td>125</td>
      <td>132</td>
      <td>131</td>
      <td>132</td>
      <td>135</td>
      <td>141</td>
      <td>...</td>
      <td>249</td>
      <td>243</td>
      <td>237</td>
      <td>242</td>
      <td>244</td>
      <td>241</td>
      <td>240</td>
      <td>242</td>
      <td>2</td>
      <td>DamVinhHung_105</td>
    </tr>
    <tr>
      <th>8</th>
      <td>66</td>
      <td>71</td>
      <td>76</td>
      <td>81</td>
      <td>86</td>
      <td>92</td>
      <td>98</td>
      <td>105</td>
      <td>111</td>
      <td>115</td>
      <td>...</td>
      <td>165</td>
      <td>163</td>
      <td>160</td>
      <td>159</td>
      <td>161</td>
      <td>164</td>
      <td>166</td>
      <td>166</td>
      <td>2</td>
      <td>DamVinhHung_106</td>
    </tr>
    <tr>
      <th>9</th>
      <td>133</td>
      <td>138</td>
      <td>145</td>
      <td>150</td>
      <td>152</td>
      <td>152</td>
      <td>151</td>
      <td>149</td>
      <td>150</td>
      <td>154</td>
      <td>...</td>
      <td>68</td>
      <td>61</td>
      <td>63</td>
      <td>66</td>
      <td>67</td>
      <td>65</td>
      <td>61</td>
      <td>58</td>
      <td>2</td>
      <td>DamVinhHung_107</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17</td>
      <td>18</td>
      <td>17</td>
      <td>21</td>
      <td>40</td>
      <td>51</td>
      <td>53</td>
      <td>58</td>
      <td>61</td>
      <td>67</td>
      <td>...</td>
      <td>35</td>
      <td>34</td>
      <td>30</td>
      <td>24</td>
      <td>21</td>
      <td>23</td>
      <td>26</td>
      <td>27</td>
      <td>2</td>
      <td>DamVinhHung_108</td>
    </tr>
    <tr>
      <th>11</th>
      <td>197</td>
      <td>196</td>
      <td>196</td>
      <td>197</td>
      <td>197</td>
      <td>197</td>
      <td>194</td>
      <td>189</td>
      <td>183</td>
      <td>179</td>
      <td>...</td>
      <td>115</td>
      <td>119</td>
      <td>114</td>
      <td>109</td>
      <td>106</td>
      <td>122</td>
      <td>126</td>
      <td>116</td>
      <td>2</td>
      <td>DamVinhHung_109</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
      <td>9</td>
      <td>19</td>
      <td>28</td>
      <td>38</td>
      <td>48</td>
      <td>59</td>
      <td>74</td>
      <td>86</td>
      <td>102</td>
      <td>...</td>
      <td>139</td>
      <td>230</td>
      <td>219</td>
      <td>201</td>
      <td>197</td>
      <td>163</td>
      <td>116</td>
      <td>112</td>
      <td>2</td>
      <td>DamVinhHung_11</td>
    </tr>
    <tr>
      <th>13</th>
      <td>135</td>
      <td>139</td>
      <td>144</td>
      <td>152</td>
      <td>162</td>
      <td>163</td>
      <td>155</td>
      <td>140</td>
      <td>124</td>
      <td>123</td>
      <td>...</td>
      <td>244</td>
      <td>243</td>
      <td>240</td>
      <td>239</td>
      <td>237</td>
      <td>231</td>
      <td>223</td>
      <td>215</td>
      <td>2</td>
      <td>DamVinhHung_110</td>
    </tr>
    <tr>
      <th>14</th>
      <td>24</td>
      <td>18</td>
      <td>10</td>
      <td>12</td>
      <td>17</td>
      <td>24</td>
      <td>38</td>
      <td>45</td>
      <td>48</td>
      <td>54</td>
      <td>...</td>
      <td>226</td>
      <td>227</td>
      <td>228</td>
      <td>228</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>227</td>
      <td>2</td>
      <td>DamVinhHung_111</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>9</td>
      <td>7</td>
      <td>46</td>
      <td>101</td>
      <td>83</td>
      <td>14</td>
      <td>23</td>
      <td>...</td>
      <td>92</td>
      <td>110</td>
      <td>131</td>
      <td>158</td>
      <td>194</td>
      <td>210</td>
      <td>211</td>
      <td>221</td>
      <td>2</td>
      <td>DamVinhHung_113</td>
    </tr>
    <tr>
      <th>16</th>
      <td>254</td>
      <td>253</td>
      <td>251</td>
      <td>251</td>
      <td>252</td>
      <td>251</td>
      <td>243</td>
      <td>242</td>
      <td>206</td>
      <td>112</td>
      <td>...</td>
      <td>103</td>
      <td>116</td>
      <td>137</td>
      <td>132</td>
      <td>137</td>
      <td>181</td>
      <td>226</td>
      <td>233</td>
      <td>2</td>
      <td>DamVinhHung_114</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>9</td>
      <td>10</td>
      <td>20</td>
      <td>50</td>
      <td>91</td>
      <td>131</td>
      <td>...</td>
      <td>169</td>
      <td>175</td>
      <td>180</td>
      <td>189</td>
      <td>203</td>
      <td>212</td>
      <td>212</td>
      <td>215</td>
      <td>2</td>
      <td>DamVinhHung_115</td>
    </tr>
    <tr>
      <th>18</th>
      <td>21</td>
      <td>21</td>
      <td>16</td>
      <td>9</td>
      <td>15</td>
      <td>34</td>
      <td>65</td>
      <td>97</td>
      <td>129</td>
      <td>143</td>
      <td>...</td>
      <td>34</td>
      <td>31</td>
      <td>36</td>
      <td>44</td>
      <td>47</td>
      <td>44</td>
      <td>63</td>
      <td>99</td>
      <td>2</td>
      <td>DamVinhHung_116</td>
    </tr>
    <tr>
      <th>19</th>
      <td>67</td>
      <td>77</td>
      <td>86</td>
      <td>95</td>
      <td>100</td>
      <td>106</td>
      <td>113</td>
      <td>120</td>
      <td>127</td>
      <td>133</td>
      <td>...</td>
      <td>56</td>
      <td>47</td>
      <td>38</td>
      <td>33</td>
      <td>39</td>
      <td>50</td>
      <td>54</td>
      <td>54</td>
      <td>2</td>
      <td>DamVinhHung_117</td>
    </tr>
    <tr>
      <th>20</th>
      <td>13</td>
      <td>16</td>
      <td>21</td>
      <td>24</td>
      <td>32</td>
      <td>48</td>
      <td>69</td>
      <td>94</td>
      <td>114</td>
      <td>127</td>
      <td>...</td>
      <td>243</td>
      <td>206</td>
      <td>119</td>
      <td>69</td>
      <td>57</td>
      <td>62</td>
      <td>77</td>
      <td>75</td>
      <td>2</td>
      <td>DamVinhHung_118</td>
    </tr>
    <tr>
      <th>21</th>
      <td>32</td>
      <td>37</td>
      <td>44</td>
      <td>55</td>
      <td>79</td>
      <td>111</td>
      <td>135</td>
      <td>139</td>
      <td>145</td>
      <td>151</td>
      <td>...</td>
      <td>190</td>
      <td>195</td>
      <td>193</td>
      <td>190</td>
      <td>191</td>
      <td>192</td>
      <td>193</td>
      <td>194</td>
      <td>2</td>
      <td>DamVinhHung_119</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24</td>
      <td>46</td>
      <td>47</td>
      <td>43</td>
      <td>48</td>
      <td>60</td>
      <td>68</td>
      <td>103</td>
      <td>151</td>
      <td>185</td>
      <td>...</td>
      <td>215</td>
      <td>209</td>
      <td>208</td>
      <td>212</td>
      <td>209</td>
      <td>211</td>
      <td>218</td>
      <td>225</td>
      <td>2</td>
      <td>DamVinhHung_12</td>
    </tr>
    <tr>
      <th>23</th>
      <td>221</td>
      <td>218</td>
      <td>206</td>
      <td>206</td>
      <td>210</td>
      <td>178</td>
      <td>142</td>
      <td>120</td>
      <td>105</td>
      <td>101</td>
      <td>...</td>
      <td>67</td>
      <td>76</td>
      <td>84</td>
      <td>86</td>
      <td>75</td>
      <td>95</td>
      <td>159</td>
      <td>234</td>
      <td>2</td>
      <td>DamVinhHung_120</td>
    </tr>
    <tr>
      <th>24</th>
      <td>30</td>
      <td>30</td>
      <td>42</td>
      <td>61</td>
      <td>76</td>
      <td>102</td>
      <td>153</td>
      <td>173</td>
      <td>172</td>
      <td>176</td>
      <td>...</td>
      <td>20</td>
      <td>20</td>
      <td>22</td>
      <td>20</td>
      <td>22</td>
      <td>31</td>
      <td>43</td>
      <td>51</td>
      <td>2</td>
      <td>DamVinhHung_121</td>
    </tr>
    <tr>
      <th>25</th>
      <td>16</td>
      <td>16</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>21</td>
      <td>17</td>
      <td>16</td>
      <td>20</td>
      <td>22</td>
      <td>...</td>
      <td>49</td>
      <td>47</td>
      <td>45</td>
      <td>44</td>
      <td>40</td>
      <td>34</td>
      <td>29</td>
      <td>24</td>
      <td>2</td>
      <td>DamVinhHung_122</td>
    </tr>
    <tr>
      <th>26</th>
      <td>224</td>
      <td>223</td>
      <td>221</td>
      <td>218</td>
      <td>223</td>
      <td>238</td>
      <td>167</td>
      <td>143</td>
      <td>169</td>
      <td>123</td>
      <td>...</td>
      <td>210</td>
      <td>209</td>
      <td>207</td>
      <td>204</td>
      <td>203</td>
      <td>203</td>
      <td>204</td>
      <td>205</td>
      <td>2</td>
      <td>DamVinhHung_123</td>
    </tr>
    <tr>
      <th>27</th>
      <td>49</td>
      <td>47</td>
      <td>53</td>
      <td>66</td>
      <td>85</td>
      <td>107</td>
      <td>129</td>
      <td>137</td>
      <td>137</td>
      <td>141</td>
      <td>...</td>
      <td>211</td>
      <td>206</td>
      <td>191</td>
      <td>180</td>
      <td>174</td>
      <td>172</td>
      <td>167</td>
      <td>156</td>
      <td>2</td>
      <td>DamVinhHung_124</td>
    </tr>
    <tr>
      <th>28</th>
      <td>216</td>
      <td>223</td>
      <td>229</td>
      <td>231</td>
      <td>228</td>
      <td>223</td>
      <td>223</td>
      <td>227</td>
      <td>229</td>
      <td>232</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>DamVinhHung_126</td>
    </tr>
    <tr>
      <th>29</th>
      <td>80</td>
      <td>56</td>
      <td>34</td>
      <td>26</td>
      <td>24</td>
      <td>26</td>
      <td>68</td>
      <td>132</td>
      <td>141</td>
      <td>160</td>
      <td>...</td>
      <td>111</td>
      <td>110</td>
      <td>119</td>
      <td>161</td>
      <td>154</td>
      <td>111</td>
      <td>138</td>
      <td>146</td>
      <td>2</td>
      <td>DamVinhHung_127</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>541</th>
      <td>24</td>
      <td>27</td>
      <td>27</td>
      <td>24</td>
      <td>20</td>
      <td>16</td>
      <td>14</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>...</td>
      <td>90</td>
      <td>88</td>
      <td>88</td>
      <td>90</td>
      <td>92</td>
      <td>95</td>
      <td>94</td>
      <td>94</td>
      <td>9</td>
      <td>NooPhuocThinh_70</td>
    </tr>
    <tr>
      <th>542</th>
      <td>50</td>
      <td>39</td>
      <td>45</td>
      <td>38</td>
      <td>42</td>
      <td>37</td>
      <td>33</td>
      <td>24</td>
      <td>20</td>
      <td>21</td>
      <td>...</td>
      <td>83</td>
      <td>104</td>
      <td>118</td>
      <td>130</td>
      <td>139</td>
      <td>144</td>
      <td>148</td>
      <td>147</td>
      <td>9</td>
      <td>NooPhuocThinh_72</td>
    </tr>
    <tr>
      <th>543</th>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>14</td>
      <td>9</td>
      <td>5</td>
      <td>13</td>
      <td>25</td>
      <td>29</td>
      <td>23</td>
      <td>...</td>
      <td>11</td>
      <td>15</td>
      <td>14</td>
      <td>13</td>
      <td>13</td>
      <td>12</td>
      <td>12</td>
      <td>11</td>
      <td>9</td>
      <td>NooPhuocThinh_73</td>
    </tr>
    <tr>
      <th>544</th>
      <td>13</td>
      <td>16</td>
      <td>17</td>
      <td>14</td>
      <td>17</td>
      <td>28</td>
      <td>41</td>
      <td>48</td>
      <td>50</td>
      <td>50</td>
      <td>...</td>
      <td>131</td>
      <td>130</td>
      <td>128</td>
      <td>127</td>
      <td>127</td>
      <td>128</td>
      <td>128</td>
      <td>128</td>
      <td>9</td>
      <td>NooPhuocThinh_74</td>
    </tr>
    <tr>
      <th>545</th>
      <td>38</td>
      <td>47</td>
      <td>45</td>
      <td>41</td>
      <td>40</td>
      <td>45</td>
      <td>52</td>
      <td>67</td>
      <td>90</td>
      <td>128</td>
      <td>...</td>
      <td>24</td>
      <td>25</td>
      <td>22</td>
      <td>21</td>
      <td>35</td>
      <td>65</td>
      <td>91</td>
      <td>118</td>
      <td>9</td>
      <td>NooPhuocThinh_75</td>
    </tr>
    <tr>
      <th>546</th>
      <td>27</td>
      <td>29</td>
      <td>32</td>
      <td>33</td>
      <td>35</td>
      <td>34</td>
      <td>32</td>
      <td>29</td>
      <td>29</td>
      <td>30</td>
      <td>...</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>97</td>
      <td>9</td>
      <td>NooPhuocThinh_76</td>
    </tr>
    <tr>
      <th>547</th>
      <td>34</td>
      <td>26</td>
      <td>24</td>
      <td>26</td>
      <td>26</td>
      <td>25</td>
      <td>24</td>
      <td>25</td>
      <td>28</td>
      <td>38</td>
      <td>...</td>
      <td>57</td>
      <td>124</td>
      <td>213</td>
      <td>253</td>
      <td>253</td>
      <td>252</td>
      <td>241</td>
      <td>221</td>
      <td>9</td>
      <td>NooPhuocThinh_77</td>
    </tr>
    <tr>
      <th>548</th>
      <td>73</td>
      <td>78</td>
      <td>71</td>
      <td>55</td>
      <td>59</td>
      <td>70</td>
      <td>66</td>
      <td>54</td>
      <td>47</td>
      <td>44</td>
      <td>...</td>
      <td>69</td>
      <td>136</td>
      <td>180</td>
      <td>200</td>
      <td>200</td>
      <td>203</td>
      <td>213</td>
      <td>192</td>
      <td>9</td>
      <td>NooPhuocThinh_78</td>
    </tr>
    <tr>
      <th>549</th>
      <td>146</td>
      <td>114</td>
      <td>82</td>
      <td>130</td>
      <td>87</td>
      <td>21</td>
      <td>7</td>
      <td>26</td>
      <td>21</td>
      <td>14</td>
      <td>...</td>
      <td>169</td>
      <td>169</td>
      <td>169</td>
      <td>169</td>
      <td>169</td>
      <td>169</td>
      <td>169</td>
      <td>170</td>
      <td>9</td>
      <td>NooPhuocThinh_79</td>
    </tr>
    <tr>
      <th>550</th>
      <td>21</td>
      <td>22</td>
      <td>21</td>
      <td>20</td>
      <td>20</td>
      <td>23</td>
      <td>28</td>
      <td>28</td>
      <td>34</td>
      <td>44</td>
      <td>...</td>
      <td>40</td>
      <td>63</td>
      <td>119</td>
      <td>169</td>
      <td>186</td>
      <td>177</td>
      <td>177</td>
      <td>175</td>
      <td>9</td>
      <td>NooPhuocThinh_8</td>
    </tr>
    <tr>
      <th>551</th>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>120</td>
      <td>122</td>
      <td>122</td>
      <td>119</td>
      <td>117</td>
      <td>125</td>
      <td>125</td>
      <td>...</td>
      <td>65</td>
      <td>75</td>
      <td>82</td>
      <td>97</td>
      <td>96</td>
      <td>82</td>
      <td>73</td>
      <td>84</td>
      <td>9</td>
      <td>NooPhuocThinh_80</td>
    </tr>
    <tr>
      <th>552</th>
      <td>64</td>
      <td>63</td>
      <td>65</td>
      <td>73</td>
      <td>86</td>
      <td>105</td>
      <td>129</td>
      <td>155</td>
      <td>178</td>
      <td>197</td>
      <td>...</td>
      <td>184</td>
      <td>182</td>
      <td>181</td>
      <td>181</td>
      <td>182</td>
      <td>184</td>
      <td>185</td>
      <td>185</td>
      <td>9</td>
      <td>NooPhuocThinh_81</td>
    </tr>
    <tr>
      <th>553</th>
      <td>51</td>
      <td>61</td>
      <td>54</td>
      <td>42</td>
      <td>33</td>
      <td>31</td>
      <td>30</td>
      <td>30</td>
      <td>34</td>
      <td>41</td>
      <td>...</td>
      <td>244</td>
      <td>244</td>
      <td>246</td>
      <td>245</td>
      <td>243</td>
      <td>243</td>
      <td>242</td>
      <td>243</td>
      <td>9</td>
      <td>NooPhuocThinh_82</td>
    </tr>
    <tr>
      <th>554</th>
      <td>25</td>
      <td>23</td>
      <td>18</td>
      <td>12</td>
      <td>8</td>
      <td>8</td>
      <td>11</td>
      <td>16</td>
      <td>33</td>
      <td>66</td>
      <td>...</td>
      <td>95</td>
      <td>81</td>
      <td>71</td>
      <td>65</td>
      <td>64</td>
      <td>62</td>
      <td>57</td>
      <td>43</td>
      <td>9</td>
      <td>NooPhuocThinh_83</td>
    </tr>
    <tr>
      <th>555</th>
      <td>33</td>
      <td>45</td>
      <td>60</td>
      <td>76</td>
      <td>89</td>
      <td>102</td>
      <td>111</td>
      <td>120</td>
      <td>127</td>
      <td>128</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>NooPhuocThinh_84</td>
    </tr>
    <tr>
      <th>556</th>
      <td>40</td>
      <td>35</td>
      <td>35</td>
      <td>38</td>
      <td>38</td>
      <td>33</td>
      <td>25</td>
      <td>21</td>
      <td>22</td>
      <td>26</td>
      <td>...</td>
      <td>51</td>
      <td>59</td>
      <td>59</td>
      <td>58</td>
      <td>58</td>
      <td>57</td>
      <td>57</td>
      <td>57</td>
      <td>9</td>
      <td>NooPhuocThinh_85</td>
    </tr>
    <tr>
      <th>557</th>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>14</td>
      <td>19</td>
      <td>24</td>
      <td>28</td>
      <td>...</td>
      <td>32</td>
      <td>40</td>
      <td>41</td>
      <td>23</td>
      <td>14</td>
      <td>23</td>
      <td>31</td>
      <td>28</td>
      <td>9</td>
      <td>NooPhuocThinh_86</td>
    </tr>
    <tr>
      <th>558</th>
      <td>15</td>
      <td>14</td>
      <td>15</td>
      <td>15</td>
      <td>14</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>16</td>
      <td>18</td>
      <td>...</td>
      <td>105</td>
      <td>104</td>
      <td>105</td>
      <td>106</td>
      <td>108</td>
      <td>109</td>
      <td>110</td>
      <td>110</td>
      <td>9</td>
      <td>NooPhuocThinh_87</td>
    </tr>
    <tr>
      <th>559</th>
      <td>16</td>
      <td>20</td>
      <td>24</td>
      <td>22</td>
      <td>20</td>
      <td>23</td>
      <td>26</td>
      <td>28</td>
      <td>39</td>
      <td>58</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>16</td>
      <td>13</td>
      <td>11</td>
      <td>13</td>
      <td>18</td>
      <td>23</td>
      <td>9</td>
      <td>NooPhuocThinh_88</td>
    </tr>
    <tr>
      <th>560</th>
      <td>60</td>
      <td>26</td>
      <td>22</td>
      <td>17</td>
      <td>20</td>
      <td>27</td>
      <td>34</td>
      <td>44</td>
      <td>54</td>
      <td>70</td>
      <td>...</td>
      <td>149</td>
      <td>141</td>
      <td>114</td>
      <td>90</td>
      <td>82</td>
      <td>78</td>
      <td>67</td>
      <td>67</td>
      <td>9</td>
      <td>NooPhuocThinh_89</td>
    </tr>
    <tr>
      <th>561</th>
      <td>30</td>
      <td>31</td>
      <td>32</td>
      <td>33</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>36</td>
      <td>38</td>
      <td>38</td>
      <td>...</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>47</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>42</td>
      <td>9</td>
      <td>NooPhuocThinh_9</td>
    </tr>
    <tr>
      <th>562</th>
      <td>85</td>
      <td>84</td>
      <td>85</td>
      <td>87</td>
      <td>89</td>
      <td>91</td>
      <td>93</td>
      <td>94</td>
      <td>95</td>
      <td>97</td>
      <td>...</td>
      <td>150</td>
      <td>151</td>
      <td>153</td>
      <td>155</td>
      <td>156</td>
      <td>158</td>
      <td>159</td>
      <td>159</td>
      <td>9</td>
      <td>NooPhuocThinh_91</td>
    </tr>
    <tr>
      <th>563</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>18</td>
      <td>28</td>
      <td>...</td>
      <td>47</td>
      <td>46</td>
      <td>53</td>
      <td>57</td>
      <td>57</td>
      <td>56</td>
      <td>51</td>
      <td>42</td>
      <td>9</td>
      <td>NooPhuocThinh_92</td>
    </tr>
    <tr>
      <th>564</th>
      <td>8</td>
      <td>5</td>
      <td>9</td>
      <td>12</td>
      <td>14</td>
      <td>16</td>
      <td>14</td>
      <td>10</td>
      <td>5</td>
      <td>6</td>
      <td>...</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>92</td>
      <td>91</td>
      <td>91</td>
      <td>92</td>
      <td>91</td>
      <td>9</td>
      <td>NooPhuocThinh_93</td>
    </tr>
    <tr>
      <th>565</th>
      <td>190</td>
      <td>192</td>
      <td>196</td>
      <td>197</td>
      <td>194</td>
      <td>192</td>
      <td>196</td>
      <td>199</td>
      <td>203</td>
      <td>209</td>
      <td>...</td>
      <td>198</td>
      <td>115</td>
      <td>34</td>
      <td>18</td>
      <td>12</td>
      <td>23</td>
      <td>29</td>
      <td>26</td>
      <td>9</td>
      <td>NooPhuocThinh_94</td>
    </tr>
    <tr>
      <th>566</th>
      <td>95</td>
      <td>94</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>94</td>
      <td>94</td>
      <td>95</td>
      <td>94</td>
      <td>...</td>
      <td>47</td>
      <td>20</td>
      <td>20</td>
      <td>28</td>
      <td>36</td>
      <td>41</td>
      <td>37</td>
      <td>34</td>
      <td>9</td>
      <td>NooPhuocThinh_95</td>
    </tr>
    <tr>
      <th>567</th>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>9</td>
      <td>18</td>
      <td>33</td>
      <td>57</td>
      <td>87</td>
      <td>110</td>
      <td>120</td>
      <td>...</td>
      <td>44</td>
      <td>46</td>
      <td>45</td>
      <td>44</td>
      <td>42</td>
      <td>40</td>
      <td>41</td>
      <td>43</td>
      <td>9</td>
      <td>NooPhuocThinh_96</td>
    </tr>
    <tr>
      <th>568</th>
      <td>47</td>
      <td>51</td>
      <td>51</td>
      <td>44</td>
      <td>35</td>
      <td>41</td>
      <td>45</td>
      <td>60</td>
      <td>91</td>
      <td>122</td>
      <td>...</td>
      <td>213</td>
      <td>179</td>
      <td>158</td>
      <td>138</td>
      <td>126</td>
      <td>129</td>
      <td>146</td>
      <td>147</td>
      <td>9</td>
      <td>NooPhuocThinh_97</td>
    </tr>
    <tr>
      <th>569</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>9</td>
      <td>28</td>
      <td>...</td>
      <td>153</td>
      <td>155</td>
      <td>155</td>
      <td>155</td>
      <td>156</td>
      <td>158</td>
      <td>161</td>
      <td>164</td>
      <td>9</td>
      <td>NooPhuocThinh_98</td>
    </tr>
    <tr>
      <th>570</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>15</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>19</td>
      <td>22</td>
      <td>...</td>
      <td>106</td>
      <td>110</td>
      <td>111</td>
      <td>113</td>
      <td>119</td>
      <td>127</td>
      <td>138</td>
      <td>143</td>
      <td>9</td>
      <td>NooPhuocThinh_99</td>
    </tr>
  </tbody>
</table>
<p>571 rows × 4098 columns</p>
</div>



Để giới hạn thời gian, ta dùng thêm một tham số `index_list` dạng list, chẳng hạn nếu `index_list = [0, 1, 3, 4, 5]` thì ta chỉ xử lí 5 hình ảnh có số thứ tự tương ứng.

***Hãy viết hàm `transformImagesToFacesTable(source_folder, destination_folder, destination_data_file, index_list)` (ngoài class `Face`) đọc các hình ảnh có số thứ tự trong list `index_list` trong thư mục `source_folder`, rồi lưu dữ liệu vào file `destination_data_file` nằm trong thư mục `destination_folder` như mô tả trên (mỗi khuôn mặt 1 dòng gồm 4096 toạ độ, 1 chỉ số ca sĩ, 1 tên file cách nhau bởi các dấu phẩy).***

Bạn có thể test bằng đoạn code dưới đây sau đó kiểm tra cấu trúc file `output.csv`


```python
transformImagesToFacesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, "TextData/output.csv", range(20))
pd.read_csv('TextData/output.csv', sep=',', header=None)
#Dòng '10 files processed' không quan trọng, nó được viết để theo dõi tiến độ chạy của chương trình 
```

    10 files processed.
    20 files processed.
    




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
      <th>4088</th>
      <th>4089</th>
      <th>4090</th>
      <th>4091</th>
      <th>4092</th>
      <th>4093</th>
      <th>4094</th>
      <th>4095</th>
      <th>4096</th>
      <th>4097</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70</td>
      <td>78</td>
      <td>86</td>
      <td>88</td>
      <td>80</td>
      <td>80</td>
      <td>74</td>
      <td>67</td>
      <td>75</td>
      <td>59</td>
      <td>...</td>
      <td>16</td>
      <td>14</td>
      <td>11</td>
      <td>30</td>
      <td>83</td>
      <td>142</td>
      <td>198</td>
      <td>191</td>
      <td>0</td>
      <td>BaoThy_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29</td>
      <td>30</td>
      <td>27</td>
      <td>33</td>
      <td>33</td>
      <td>42</td>
      <td>63</td>
      <td>62</td>
      <td>70</td>
      <td>76</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>13</td>
      <td>14</td>
      <td>18</td>
      <td>0</td>
      <td>BaoThy_10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>76</td>
      <td>80</td>
      <td>83</td>
      <td>87</td>
      <td>90</td>
      <td>97</td>
      <td>103</td>
      <td>111</td>
      <td>115</td>
      <td>...</td>
      <td>220</td>
      <td>220</td>
      <td>220</td>
      <td>220</td>
      <td>220</td>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>0</td>
      <td>BaoThy_100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>76</td>
      <td>73</td>
      <td>65</td>
      <td>73</td>
      <td>95</td>
      <td>100</td>
      <td>93</td>
      <td>91</td>
      <td>88</td>
      <td>...</td>
      <td>231</td>
      <td>244</td>
      <td>251</td>
      <td>250</td>
      <td>251</td>
      <td>253</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>BaoThy_101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>30</td>
      <td>33</td>
      <td>36</td>
      <td>24</td>
      <td>8</td>
      <td>25</td>
      <td>17</td>
      <td>21</td>
      <td>26</td>
      <td>...</td>
      <td>20</td>
      <td>23</td>
      <td>30</td>
      <td>37</td>
      <td>38</td>
      <td>36</td>
      <td>42</td>
      <td>51</td>
      <td>0</td>
      <td>BaoThy_102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>106</td>
      <td>64</td>
      <td>79</td>
      <td>102</td>
      <td>107</td>
      <td>94</td>
      <td>94</td>
      <td>106</td>
      <td>114</td>
      <td>107</td>
      <td>...</td>
      <td>60</td>
      <td>58</td>
      <td>50</td>
      <td>61</td>
      <td>71</td>
      <td>75</td>
      <td>76</td>
      <td>74</td>
      <td>0</td>
      <td>BaoThy_103</td>
    </tr>
    <tr>
      <th>6</th>
      <td>88</td>
      <td>73</td>
      <td>64</td>
      <td>68</td>
      <td>67</td>
      <td>58</td>
      <td>45</td>
      <td>37</td>
      <td>36</td>
      <td>40</td>
      <td>...</td>
      <td>64</td>
      <td>65</td>
      <td>65</td>
      <td>68</td>
      <td>69</td>
      <td>70</td>
      <td>72</td>
      <td>76</td>
      <td>0</td>
      <td>BaoThy_104</td>
    </tr>
    <tr>
      <th>7</th>
      <td>135</td>
      <td>129</td>
      <td>130</td>
      <td>135</td>
      <td>144</td>
      <td>155</td>
      <td>169</td>
      <td>178</td>
      <td>180</td>
      <td>186</td>
      <td>...</td>
      <td>57</td>
      <td>53</td>
      <td>53</td>
      <td>57</td>
      <td>65</td>
      <td>70</td>
      <td>70</td>
      <td>68</td>
      <td>0</td>
      <td>BaoThy_105</td>
    </tr>
    <tr>
      <th>8</th>
      <td>153</td>
      <td>149</td>
      <td>151</td>
      <td>151</td>
      <td>150</td>
      <td>152</td>
      <td>153</td>
      <td>154</td>
      <td>157</td>
      <td>159</td>
      <td>...</td>
      <td>98</td>
      <td>99</td>
      <td>99</td>
      <td>100</td>
      <td>100</td>
      <td>101</td>
      <td>101</td>
      <td>102</td>
      <td>0</td>
      <td>BaoThy_106</td>
    </tr>
    <tr>
      <th>9</th>
      <td>67</td>
      <td>88</td>
      <td>111</td>
      <td>107</td>
      <td>101</td>
      <td>88</td>
      <td>74</td>
      <td>86</td>
      <td>89</td>
      <td>104</td>
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
      <td>BaoThy_107</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14</td>
      <td>15</td>
      <td>18</td>
      <td>22</td>
      <td>27</td>
      <td>34</td>
      <td>39</td>
      <td>44</td>
      <td>50</td>
      <td>60</td>
      <td>...</td>
      <td>7</td>
      <td>6</td>
      <td>49</td>
      <td>96</td>
      <td>63</td>
      <td>11</td>
      <td>5</td>
      <td>15</td>
      <td>0</td>
      <td>BaoThy_108</td>
    </tr>
    <tr>
      <th>11</th>
      <td>94</td>
      <td>95</td>
      <td>92</td>
      <td>89</td>
      <td>85</td>
      <td>72</td>
      <td>50</td>
      <td>40</td>
      <td>49</td>
      <td>57</td>
      <td>...</td>
      <td>77</td>
      <td>89</td>
      <td>92</td>
      <td>79</td>
      <td>66</td>
      <td>69</td>
      <td>81</td>
      <td>90</td>
      <td>0</td>
      <td>BaoThy_109</td>
    </tr>
    <tr>
      <th>12</th>
      <td>76</td>
      <td>67</td>
      <td>66</td>
      <td>67</td>
      <td>56</td>
      <td>45</td>
      <td>41</td>
      <td>42</td>
      <td>56</td>
      <td>39</td>
      <td>...</td>
      <td>193</td>
      <td>185</td>
      <td>183</td>
      <td>185</td>
      <td>184</td>
      <td>180</td>
      <td>172</td>
      <td>158</td>
      <td>0</td>
      <td>BaoThy_11</td>
    </tr>
    <tr>
      <th>13</th>
      <td>101</td>
      <td>105</td>
      <td>108</td>
      <td>108</td>
      <td>106</td>
      <td>104</td>
      <td>100</td>
      <td>98</td>
      <td>101</td>
      <td>104</td>
      <td>...</td>
      <td>173</td>
      <td>172</td>
      <td>170</td>
      <td>169</td>
      <td>167</td>
      <td>165</td>
      <td>163</td>
      <td>160</td>
      <td>0</td>
      <td>BaoThy_110</td>
    </tr>
    <tr>
      <th>14</th>
      <td>36</td>
      <td>42</td>
      <td>39</td>
      <td>29</td>
      <td>24</td>
      <td>26</td>
      <td>31</td>
      <td>33</td>
      <td>30</td>
      <td>23</td>
      <td>...</td>
      <td>148</td>
      <td>146</td>
      <td>143</td>
      <td>142</td>
      <td>146</td>
      <td>142</td>
      <td>124</td>
      <td>112</td>
      <td>0</td>
      <td>BaoThy_111</td>
    </tr>
    <tr>
      <th>15</th>
      <td>113</td>
      <td>113</td>
      <td>114</td>
      <td>113</td>
      <td>113</td>
      <td>112</td>
      <td>115</td>
      <td>120</td>
      <td>124</td>
      <td>119</td>
      <td>...</td>
      <td>189</td>
      <td>194</td>
      <td>195</td>
      <td>195</td>
      <td>198</td>
      <td>190</td>
      <td>173</td>
      <td>167</td>
      <td>0</td>
      <td>BaoThy_112</td>
    </tr>
    <tr>
      <th>16</th>
      <td>104</td>
      <td>101</td>
      <td>94</td>
      <td>86</td>
      <td>75</td>
      <td>61</td>
      <td>48</td>
      <td>47</td>
      <td>50</td>
      <td>53</td>
      <td>...</td>
      <td>37</td>
      <td>40</td>
      <td>41</td>
      <td>43</td>
      <td>44</td>
      <td>46</td>
      <td>60</td>
      <td>81</td>
      <td>0</td>
      <td>BaoThy_113</td>
    </tr>
    <tr>
      <th>17</th>
      <td>134</td>
      <td>134</td>
      <td>134</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>134</td>
      <td>135</td>
      <td>122</td>
      <td>...</td>
      <td>144</td>
      <td>108</td>
      <td>42</td>
      <td>10</td>
      <td>25</td>
      <td>33</td>
      <td>38</td>
      <td>40</td>
      <td>0</td>
      <td>BaoThy_114</td>
    </tr>
    <tr>
      <th>18</th>
      <td>12</td>
      <td>12</td>
      <td>11</td>
      <td>13</td>
      <td>20</td>
      <td>32</td>
      <td>42</td>
      <td>49</td>
      <td>58</td>
      <td>65</td>
      <td>...</td>
      <td>6</td>
      <td>7</td>
      <td>9</td>
      <td>12</td>
      <td>12</td>
      <td>7</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>BaoThy_115</td>
    </tr>
  </tbody>
</table>
<p>19 rows × 4098 columns</p>
</div>



Ta thấy hình ảnh `BaoThy_0` không xuất hiện trong file output vì không xác định được khuôn mặt. Do đó khi process 20 hình ảnh, ta chỉ thu được 19 khuôn mặt.

### Bài 9 - Chuyển các đôi mắt thành vector

Ta muốn lưu các đôi mắt dưới dạng các vector số vào một file dữ liệu sao cho mỗi đôi mắt ứng với một dòng của file, tức toạ độ của một vector có số chiều 2048 (=2 x 32 x 32) cách nhau bằng dấu phẩy; sau đó là một số nguyên cho biết chỉ số của ca sĩ trong từ điển `SINGER_INDEX_DICTIONARY` và tên hình ảnh tương ứng. Ví dụ file có dạng sau. Điều này chỉ thực hiện được khi hình ảnh xác định được cả 2 mắt của người trong ảnh.

Muốn vậy, ta phải tìm các ảnh sao cho attribute `self.normalized_eyes` gồm ít nhất 2 phần tử. Thông thường, hình ảnh không rõ sẽ dẫn đến nhầm lẫn miệng-mắt. Nếu các phần tử tìm được đều là mắt và được tìm theo thứ tự từ trái sang phải của hình. Như vậy, nếu cả 2 mắt đều được tìm ra thì `self.normalized_eyes[0]` sẽ là mắt phải, và `self.normalized_eyes[-1]` sẽ là mắt trái của người trong hình. Hai mắt này được xem là "chất lượng tốt" nếu các tiêu chuẩn sau được thoả mãn:

- Hoành độ 2 tâm hình chữ nhật chứa 2 mắt trong ảnh không cách nhau quá `HORIZONTAL_CHECK = 3` pixel (nếu lớn hơn, cho thấy mặt không được chụp thẳng hoặc 1 thành phần khác của khuôn mặt (như miệng) đã bị lẫn vào.)

- Tung độ tâm hình chữ nhật chứa 2 mắt trong ảnh phải ít nhất lớn hơn `VERTICAL_CHECK = 10` pixel (nếu nhỏ hơn, nó cho thấy `self.normalized_eyes[0]` và `self.normalized_eyes[-1]` là 1 mắt được tìm ra 2 lần, còn 1 mắt không tìm ra được).

***Hãy viết hàm `transformImagesToEyesTable(source_folder, destination_folder, destination_data_file, index_list)` (ngoài class `Face`) đọc các hình ảnh có số thứ tự trong list `index_list` trong thư mục `source_folder`, rồi lưu dữ liệu vào file `destination_data_file` nằm trong thư mục `destination_folder` như mô tả trên (mỗi khuôn mặt 1 dòng gồm 2048 toạ độ trong đó 1024 của mắt phải, theo bởi 1024 của mắt trái, rồi 1 chỉ số ca sĩ, 1 tên file cách nhau bởi các dấu phẩy). Nếu ít nhất một hai điều kiện kiểm tra "chất lượng tốt" trên không được thoả mãn, loại bỏ hình ảnh khỏi tập dữ liệu.***

Đoạn code dưới đây giúp kiểm tra hàm của bạn.


```python
transformImagesToEyesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, "TextData/output2.csv", range(20))
pd.read_csv('TextData/output2.csv', sep=',', header=None)
```

    10 files processed.
    20 files processed.
    




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
      <th>2040</th>
      <th>2041</th>
      <th>2042</th>
      <th>2043</th>
      <th>2044</th>
      <th>2045</th>
      <th>2046</th>
      <th>2047</th>
      <th>2048</th>
      <th>2049</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>190</td>
      <td>189</td>
      <td>188</td>
      <td>184</td>
      <td>177</td>
      <td>170</td>
      <td>163</td>
      <td>156</td>
      <td>150</td>
      <td>147</td>
      <td>...</td>
      <td>211</td>
      <td>213</td>
      <td>216</td>
      <td>217</td>
      <td>219</td>
      <td>219</td>
      <td>218</td>
      <td>218</td>
      <td>0</td>
      <td>BaoThy_100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>94</td>
      <td>93</td>
      <td>89</td>
      <td>83</td>
      <td>74</td>
      <td>75</td>
      <td>81</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>...</td>
      <td>230</td>
      <td>223</td>
      <td>223</td>
      <td>224</td>
      <td>226</td>
      <td>227</td>
      <td>224</td>
      <td>221</td>
      <td>0</td>
      <td>BaoThy_101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201</td>
      <td>199</td>
      <td>198</td>
      <td>198</td>
      <td>201</td>
      <td>205</td>
      <td>210</td>
      <td>215</td>
      <td>218</td>
      <td>220</td>
      <td>...</td>
      <td>181</td>
      <td>177</td>
      <td>174</td>
      <td>171</td>
      <td>167</td>
      <td>162</td>
      <td>154</td>
      <td>147</td>
      <td>0</td>
      <td>BaoThy_104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>209</td>
      <td>213</td>
      <td>215</td>
      <td>213</td>
      <td>208</td>
      <td>205</td>
      <td>199</td>
      <td>193</td>
      <td>186</td>
      <td>174</td>
      <td>...</td>
      <td>178</td>
      <td>162</td>
      <td>143</td>
      <td>120</td>
      <td>97</td>
      <td>76</td>
      <td>67</td>
      <td>63</td>
      <td>0</td>
      <td>BaoThy_105</td>
    </tr>
    <tr>
      <th>4</th>
      <td>181</td>
      <td>182</td>
      <td>183</td>
      <td>183</td>
      <td>181</td>
      <td>181</td>
      <td>183</td>
      <td>187</td>
      <td>190</td>
      <td>194</td>
      <td>...</td>
      <td>165</td>
      <td>162</td>
      <td>158</td>
      <td>155</td>
      <td>152</td>
      <td>150</td>
      <td>148</td>
      <td>146</td>
      <td>0</td>
      <td>BaoThy_106</td>
    </tr>
    <tr>
      <th>5</th>
      <td>111</td>
      <td>113</td>
      <td>115</td>
      <td>117</td>
      <td>115</td>
      <td>113</td>
      <td>110</td>
      <td>107</td>
      <td>104</td>
      <td>103</td>
      <td>...</td>
      <td>169</td>
      <td>167</td>
      <td>163</td>
      <td>160</td>
      <td>158</td>
      <td>159</td>
      <td>157</td>
      <td>154</td>
      <td>0</td>
      <td>BaoThy_108</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12</td>
      <td>31</td>
      <td>56</td>
      <td>78</td>
      <td>88</td>
      <td>91</td>
      <td>84</td>
      <td>82</td>
      <td>84</td>
      <td>95</td>
      <td>...</td>
      <td>221</td>
      <td>222</td>
      <td>222</td>
      <td>219</td>
      <td>216</td>
      <td>217</td>
      <td>219</td>
      <td>219</td>
      <td>0</td>
      <td>BaoThy_11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>48</td>
      <td>51</td>
      <td>56</td>
      <td>67</td>
      <td>80</td>
      <td>94</td>
      <td>106</td>
      <td>112</td>
      <td>110</td>
      <td>104</td>
      <td>...</td>
      <td>207</td>
      <td>206</td>
      <td>203</td>
      <td>199</td>
      <td>196</td>
      <td>197</td>
      <td>202</td>
      <td>207</td>
      <td>0</td>
      <td>BaoThy_110</td>
    </tr>
    <tr>
      <th>8</th>
      <td>193</td>
      <td>190</td>
      <td>188</td>
      <td>188</td>
      <td>187</td>
      <td>187</td>
      <td>187</td>
      <td>187</td>
      <td>188</td>
      <td>187</td>
      <td>...</td>
      <td>185</td>
      <td>182</td>
      <td>179</td>
      <td>178</td>
      <td>177</td>
      <td>176</td>
      <td>175</td>
      <td>172</td>
      <td>0</td>
      <td>BaoThy_113</td>
    </tr>
    <tr>
      <th>9</th>
      <td>140</td>
      <td>144</td>
      <td>149</td>
      <td>153</td>
      <td>157</td>
      <td>160</td>
      <td>168</td>
      <td>169</td>
      <td>165</td>
      <td>159</td>
      <td>...</td>
      <td>203</td>
      <td>203</td>
      <td>202</td>
      <td>199</td>
      <td>194</td>
      <td>189</td>
      <td>183</td>
      <td>177</td>
      <td>0</td>
      <td>BaoThy_115</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 2050 columns</p>
</div>



Kết quả cho thấy trong 20 hình ảnh đầu tiên của ca sí BaoThy, chỉ có 10 hình ảnh chất lượng tốt (tìm được 2 mắt)


```python
# Ví dụ 1 ảnh tốt
face = Face('ImageFolder/BaoThy_104.png')
face.draw('eye_marked', 0)
face.draw('eye_marked', -1)
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_59_0.png)



```python
# Ví dụ 1 ảnh tốt khác
face = Face('ImageFolder/BaoThy_115.png')
face.draw('eye_marked', 0)
face.draw('eye_marked', -1)
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_60_0.png)



```python
# Ví dụ 1 ảnh xấu
face = Face('ImageFolder/BaoThy_102.png')
face.draw('eye_marked', 0)
face.draw('eye_marked', -1)
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_61_0.png)



```python
# 1 ảnh "xấu" khác. Trong code của lời giải xử lí nếu index > len(self.normalized_eyes) thì in ra thông báo "Chỉ có ... mắt" được tìm ra"
face = Face('ImageFolder/BaoThy_103.png')
face.draw('eye_marked', 0)
face.draw('eye_marked', -1)
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_62_0.png)


## Phần 2 - Xây dựng class DataSet

### Bài 10 - Class `DataSet`

Ta xây dựng một class `DataSet` từ dữ liệu đã được lưu ở file output của các bài 8, 9. Mỗi instance của class này có các attribute như sau:

- `X`: một array `N x d` với mỗi dòng là một vector (số chiều bằng `d`=4096 hoặc 2048 tuỳ theo tập dữ liệu lấy từ mặt hay mắt; số chiều có thể giảm để nhỏ hơn 4096, 2048 ở các bài tiếp theo; `N` là số dòng trong file dữ liệu)

- `y`: một array có độ dài `N` tương ứng với lớp mà các khuôn mặt/đôi mắt thuộc về. Chỉ số của lớp chính là chỉ số của ca sĩ.

- `names`: một array có độ dài `N` tương ứng với tên hình ảnh.

***Trong class `DataSet`, viết method `__init__(self, data_file, selected_columns = None)` trong đó `data_file` là đường dẫn đến file dữ liệu (có dạng như output của bài 8 hoặc 9), `select_columns` là một list các chỉ số toạ độ mà ta chưa cần quan tâm ở bài này; sau đó xây dựng các attribute `X`, `y` và `names` như mô tả.***

Với file `output.csv` như đã thực hiện ở bài 8, bạn có thể kiểm tra hàm đã viết với đoạn code sau.


```python
dataSet = DataSet("TextData/output.csv")
dataSet.X
```




    array([[ 70,  78,  86, ..., 142, 198, 191],
           [ 29,  30,  27, ...,  13,  14,  18],
           [ 72,  76,  80, ..., 221, 221, 221],
           ...,
           [104, 101,  94, ...,  46,  60,  81],
           [134, 134, 134, ...,  33,  38,  40],
           [ 12,  12,  11, ...,   7,   4,   4]], dtype=int64)




```python
dataSet.X.shape # Một array 19 x 4096
```




    (19, 4096)




```python
dataSet.y, dataSet.names
```




    (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           dtype=int64),
     array(['BaoThy_1', 'BaoThy_10', 'BaoThy_100', 'BaoThy_101', 'BaoThy_102',
            'BaoThy_103', 'BaoThy_104', 'BaoThy_105', 'BaoThy_106',
            'BaoThy_107', 'BaoThy_108', 'BaoThy_109', 'BaoThy_11',
            'BaoThy_110', 'BaoThy_111', 'BaoThy_112', 'BaoThy_113',
            'BaoThy_114', 'BaoThy_115'], dtype=object))



**Chú ý:** Sau khi viết xong các hàm ở bài 8, 9, 10, để xây dựng data set gồm tất cả các khuôn mặt xác định được từ các hình ảnh của chẳng hạn 2 ca sĩ MyTam và DanTruong, bạn có thể dùng đoạn code sau.


```python
from itertools import chain
singers = ["MyTam", "DanTruong"]

# Lưu vào file MyTam_vs_DanTruong.csv
transformImagesToFacesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, "TextData/MyTam_vs_DanTruong.csv", chain.from_iterable([SINGER_IMAGE_RANGE[s] for s in singers]))
```

    10 files processed.
    20 files processed.
    30 files processed.
    40 files processed.
    50 files processed.
    60 files processed.
    70 files processed.
    80 files processed.
    90 files processed.
    100 files processed.
    110 files processed.
    120 files processed.
    130 files processed.
    140 files processed.
    150 files processed.
    160 files processed.
    170 files processed.
    180 files processed.
    190 files processed.
    200 files processed.
    210 files processed.
    220 files processed.
    230 files processed.
    240 files processed.
    250 files processed.
    260 files processed.
    270 files processed.
    280 files processed.
    290 files processed.
    300 files processed.
    


```python
dataSet = DataSet("TextData/MyTam_vs_DanTruong.csv")
dataSet.X
```




    array([[ 95,  91,  82, ...,  11,  18,  33],
           [164, 164, 143, ...,  31,  36,  39],
           [128, 127, 126, ..., 199, 202, 209],
           ...,
           [ 23,  20,  14, ...,  87,  59,  34],
           [ 30,  28,   7, ..., 247, 224, 192],
           [ 13,  13,  16, ..., 156, 166,  96]], dtype=int64)




```python
dataSet.X.shape # Kết quả cho thấy có 303 khuôn mặt được xác định
```




    (303, 4096)




```python
dataSet.y
```




    array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=int64)




```python
dataSet.names
```




    array(['MyTam_0', 'MyTam_1', 'MyTam_10', 'MyTam_100', 'MyTam_101',
           'MyTam_102', 'MyTam_103', 'MyTam_105', 'MyTam_106', 'MyTam_107',
           'MyTam_108', 'MyTam_109', 'MyTam_11', 'MyTam_110', 'MyTam_111',
           'MyTam_112', 'MyTam_113', 'MyTam_114', 'MyTam_115', 'MyTam_116',
           'MyTam_117', 'MyTam_118', 'MyTam_119', 'MyTam_12', 'MyTam_120',
           'MyTam_121', 'MyTam_122', 'MyTam_123', 'MyTam_124', 'MyTam_125',
           'MyTam_126', 'MyTam_127', 'MyTam_128', 'MyTam_129', 'MyTam_13',
           'MyTam_130', 'MyTam_131', 'MyTam_132', 'MyTam_133', 'MyTam_134',
           'MyTam_135', 'MyTam_136', 'MyTam_137', 'MyTam_138', 'MyTam_139',
           'MyTam_14', 'MyTam_140', 'MyTam_141', 'MyTam_142', 'MyTam_143',
           'MyTam_144', 'MyTam_145', 'MyTam_146', 'MyTam_147', 'MyTam_148',
           'MyTam_149', 'MyTam_15', 'MyTam_150', 'MyTam_151', 'MyTam_152',
           'MyTam_153', 'MyTam_154', 'MyTam_155', 'MyTam_156', 'MyTam_157',
           'MyTam_158', 'MyTam_159', 'MyTam_16', 'MyTam_160', 'MyTam_161',
           'MyTam_162', 'MyTam_163', 'MyTam_164', 'MyTam_165', 'MyTam_166',
           'MyTam_167', 'MyTam_168', 'MyTam_169', 'MyTam_17', 'MyTam_170',
           'MyTam_171', 'MyTam_172', 'MyTam_19', 'MyTam_2', 'MyTam_20',
           'MyTam_21', 'MyTam_22', 'MyTam_23', 'MyTam_24', 'MyTam_25',
           'MyTam_26', 'MyTam_27', 'MyTam_28', 'MyTam_29', 'MyTam_3',
           'MyTam_30', 'MyTam_31', 'MyTam_32', 'MyTam_33', 'MyTam_34',
           'MyTam_35', 'MyTam_36', 'MyTam_37', 'MyTam_38', 'MyTam_39',
           'MyTam_4', 'MyTam_40', 'MyTam_41', 'MyTam_42', 'MyTam_43',
           'MyTam_44', 'MyTam_45', 'MyTam_46', 'MyTam_47', 'MyTam_48',
           'MyTam_49', 'MyTam_5', 'MyTam_50', 'MyTam_51', 'MyTam_52',
           'MyTam_53', 'MyTam_54', 'MyTam_55', 'MyTam_56', 'MyTam_57',
           'MyTam_58', 'MyTam_59', 'MyTam_6', 'MyTam_60', 'MyTam_61',
           'MyTam_62', 'MyTam_63', 'MyTam_64', 'MyTam_65', 'MyTam_66',
           'MyTam_67', 'MyTam_68', 'MyTam_69', 'MyTam_7', 'MyTam_70',
           'MyTam_71', 'MyTam_72', 'MyTam_73', 'MyTam_75', 'MyTam_76',
           'MyTam_77', 'MyTam_78', 'MyTam_79', 'MyTam_8', 'MyTam_80',
           'MyTam_81', 'MyTam_82', 'MyTam_83', 'MyTam_84', 'MyTam_85',
           'MyTam_86', 'MyTam_87', 'MyTam_88', 'MyTam_89', 'MyTam_9',
           'MyTam_90', 'MyTam_91', 'MyTam_92', 'MyTam_93', 'MyTam_94',
           'MyTam_95', 'MyTam_96', 'MyTam_97', 'MyTam_98', 'MyTam_99',
           'DanTruong_0', 'DanTruong_1', 'DanTruong_10', 'DanTruong_100',
           'DanTruong_101', 'DanTruong_102', 'DanTruong_103', 'DanTruong_104',
           'DanTruong_105', 'DanTruong_106', 'DanTruong_107', 'DanTruong_108',
           'DanTruong_11', 'DanTruong_110', 'DanTruong_111', 'DanTruong_112',
           'DanTruong_113', 'DanTruong_114', 'DanTruong_115', 'DanTruong_116',
           'DanTruong_117', 'DanTruong_118', 'DanTruong_119', 'DanTruong_12',
           'DanTruong_120', 'DanTruong_121', 'DanTruong_122', 'DanTruong_123',
           'DanTruong_124', 'DanTruong_125', 'DanTruong_126', 'DanTruong_127',
           'DanTruong_128', 'DanTruong_129', 'DanTruong_13', 'DanTruong_130',
           'DanTruong_131', 'DanTruong_132', 'DanTruong_133', 'DanTruong_134',
           'DanTruong_14', 'DanTruong_15', 'DanTruong_16', 'DanTruong_17',
           'DanTruong_18', 'DanTruong_19', 'DanTruong_2', 'DanTruong_20',
           'DanTruong_21', 'DanTruong_22', 'DanTruong_23', 'DanTruong_24',
           'DanTruong_25', 'DanTruong_26', 'DanTruong_27', 'DanTruong_28',
           'DanTruong_29', 'DanTruong_3', 'DanTruong_30', 'DanTruong_31',
           'DanTruong_32', 'DanTruong_33', 'DanTruong_34', 'DanTruong_35',
           'DanTruong_36', 'DanTruong_37', 'DanTruong_38', 'DanTruong_39',
           'DanTruong_4', 'DanTruong_40', 'DanTruong_41', 'DanTruong_42',
           'DanTruong_43', 'DanTruong_44', 'DanTruong_45', 'DanTruong_46',
           'DanTruong_47', 'DanTruong_48', 'DanTruong_49', 'DanTruong_5',
           'DanTruong_50', 'DanTruong_51', 'DanTruong_52', 'DanTruong_53',
           'DanTruong_54', 'DanTruong_55', 'DanTruong_56', 'DanTruong_57',
           'DanTruong_58', 'DanTruong_59', 'DanTruong_6', 'DanTruong_60',
           'DanTruong_61', 'DanTruong_63', 'DanTruong_64', 'DanTruong_65',
           'DanTruong_66', 'DanTruong_67', 'DanTruong_68', 'DanTruong_69',
           'DanTruong_7', 'DanTruong_70', 'DanTruong_71', 'DanTruong_72',
           'DanTruong_73', 'DanTruong_74', 'DanTruong_75', 'DanTruong_76',
           'DanTruong_77', 'DanTruong_78', 'DanTruong_79', 'DanTruong_8',
           'DanTruong_80', 'DanTruong_81', 'DanTruong_82', 'DanTruong_83',
           'DanTruong_84', 'DanTruong_85', 'DanTruong_86', 'DanTruong_87',
           'DanTruong_88', 'DanTruong_89', 'DanTruong_9', 'DanTruong_90',
           'DanTruong_91', 'DanTruong_92', 'DanTruong_93', 'DanTruong_94',
           'DanTruong_95', 'DanTruong_96', 'DanTruong_97', 'DanTruong_98',
           'DanTruong_99'], dtype=object)



### Bài 11. Chia thành 2 phần train/test

***Trong class `DataSet`, viết method `trainTestSplit(self, test_size)` chia các array `X`, `y`, `names` thành 2 phần được gán cho `X_train`, `X_test`, `y_train`, `y_test`, `names_train`, `names_test` sao cho tỉ lệ các dòng dữ liệu của phần test bằng `test_size` (nằm trong (0, 1)). Bạn có thể dùng `model_selection.train_test_split` của scikit-learn và dùng tham số `random_state` của class này để cố định tập train và test.***

Đoạn code dưới đây giúp test hàm của bạn.


```python
dataSet = DataSet("TextData/MyTam_vs_DanTruong.csv")
dataSet.trainTestSplit(test_size = 0.5)
```


```python
dataSet.X_train
```




    array([[161, 147, 134, ...,  95,  95, 101],
           [ 67,  78,  75, ...,   0,   0,   0],
           [ 39,  50,  34, ...,  22,  35,  39],
           ...,
           [ 72,  85, 103, ...,  48,  43,  45],
           [167, 147, 131, ..., 115, 119, 124],
           [ 28,  25,  25, ..., 147,  35,  15]], dtype=int64)




```python
dataSet.X_train.shape # 151 dòng được dùng để train
```




    (151, 4096)




```python
dataSet.y_train, dataSet.names_train # Các dòng được chọn ngẫu nhiên ứng với ca sĩ MyTam và DanTruong
```




    (array([8, 3, 3, 8, 3, 3, 3, 8, 8, 8, 3, 8, 3, 8, 8, 3, 8, 8, 8, 8, 8, 3,
            3, 8, 3, 8, 3, 3, 8, 3, 3, 8, 8, 8, 3, 3, 8, 8, 3, 8, 3, 8, 3, 8,
            8, 3, 8, 8, 8, 3, 8, 3, 3, 3, 8, 8, 8, 8, 3, 8, 3, 8, 3, 8, 8, 8,
            3, 8, 3, 8, 3, 3, 3, 8, 8, 3, 8, 3, 8, 8, 3, 3, 8, 3, 8, 8, 3, 8,
            8, 8, 8, 8, 8, 3, 8, 8, 8, 3, 3, 8, 8, 8, 8, 8, 3, 8, 8, 3, 8, 8,
            3, 8, 8, 3, 3, 8, 8, 8, 3, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 8, 8, 8,
            3, 3, 3, 8, 3, 8, 8, 8, 3, 3, 3, 3, 8, 3, 3, 3, 8, 8, 3],
           dtype=int64),
     array(['MyTam_169', 'DanTruong_78', 'DanTruong_117', 'MyTam_116',
            'DanTruong_57', 'DanTruong_37', 'DanTruong_68', 'MyTam_171',
            'MyTam_145', 'MyTam_10', 'DanTruong_40', 'MyTam_39',
            'DanTruong_61', 'MyTam_22', 'MyTam_108', 'DanTruong_27',
            'MyTam_151', 'MyTam_136', 'MyTam_111', 'MyTam_86', 'MyTam_144',
            'DanTruong_19', 'DanTruong_39', 'MyTam_56', 'DanTruong_23',
            'MyTam_155', 'DanTruong_118', 'DanTruong_32', 'MyTam_62',
            'DanTruong_17', 'DanTruong_114', 'MyTam_138', 'MyTam_48',
            'MyTam_7', 'DanTruong_125', 'DanTruong_25', 'MyTam_80', 'MyTam_46',
            'DanTruong_95', 'MyTam_33', 'DanTruong_24', 'MyTam_29',
            'DanTruong_133', 'MyTam_92', 'MyTam_131', 'DanTruong_105',
            'MyTam_47', 'MyTam_0', 'MyTam_3', 'DanTruong_91', 'MyTam_30',
            'DanTruong_96', 'DanTruong_63', 'DanTruong_56', 'MyTam_161',
            'MyTam_143', 'MyTam_142', 'MyTam_21', 'DanTruong_97', 'MyTam_72',
            'DanTruong_132', 'MyTam_12', 'DanTruong_50', 'MyTam_8', 'MyTam_75',
            'MyTam_17', 'DanTruong_107', 'MyTam_35', 'DanTruong_13',
            'MyTam_63', 'DanTruong_69', 'DanTruong_98', 'DanTruong_122',
            'MyTam_103', 'MyTam_160', 'DanTruong_129', 'MyTam_20',
            'DanTruong_0', 'MyTam_54', 'MyTam_71', 'DanTruong_58',
            'DanTruong_75', 'MyTam_73', 'DanTruong_59', 'MyTam_27', 'MyTam_19',
            'DanTruong_83', 'MyTam_109', 'MyTam_52', 'MyTam_37', 'MyTam_130',
            'MyTam_150', 'MyTam_99', 'DanTruong_33', 'MyTam_158', 'MyTam_1',
            'MyTam_53', 'DanTruong_67', 'DanTruong_113', 'MyTam_137',
            'MyTam_4', 'MyTam_64', 'MyTam_170', 'MyTam_114', 'DanTruong_70',
            'MyTam_133', 'MyTam_147', 'DanTruong_6', 'MyTam_60', 'MyTam_124',
            'DanTruong_110', 'MyTam_93', 'MyTam_82', 'DanTruong_45',
            'DanTruong_128', 'MyTam_127', 'MyTam_128', 'MyTam_6',
            'DanTruong_112', 'DanTruong_79', 'DanTruong_72', 'MyTam_79',
            'DanTruong_77', 'DanTruong_104', 'MyTam_34', 'DanTruong_123',
            'DanTruong_44', 'MyTam_49', 'DanTruong_65', 'MyTam_164',
            'MyTam_121', 'MyTam_95', 'DanTruong_87', 'DanTruong_101',
            'DanTruong_89', 'MyTam_134', 'DanTruong_12', 'MyTam_24',
            'MyTam_162', 'MyTam_23', 'DanTruong_9', 'DanTruong_43',
            'DanTruong_76', 'DanTruong_15', 'MyTam_107', 'DanTruong_121',
            'DanTruong_51', 'DanTruong_119', 'MyTam_50', 'MyTam_141',
            'DanTruong_10'], dtype=object))



### Bài 12. Training

***Trong class `DataSet`, viết method `train(self, model)` nhận đối số `model` là một Classifier của `scikit-learn` (như `SVC` hay `LogisticRegression` v.v...), train model này với `X_train` và `y_train`, và trả lại `model` đã train.***

Sau khi train, bạn cần tự viết một đoạn code để test hàm này với Support Vector Machine với linear kernel. Đoạn code dưới đây chỉ dùng để tham khảo, bạn cần viết đoạn code tương tự như vậy.


```python
from sklearn.svm import SVC
dataSet = DataSet("TextData/MyTam_vs_DanTruong.csv")
dataSet.trainTestSplit(test_size = 0.5)
myModel = SVC(kernel = "linear")
dataSet.train(myModel)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
# Tương tự với LogisticRegression

from sklearn.linear_model import LogisticRegression
dataSet = DataSet("TextData/MyTam_vs_DanTruong.csv")
dataSet.trainTestSplit(test_size = 0.5)
myModel2 = LogisticRegression()
dataSet.train(myModel2)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



### Bài 13. Dự đoán

***Trong class `DataSet`, viết method `predict(self, model)` nhận đối số `model` là một classifier của scikit-learn đã được train với hàm `train` của bài trước, và trả lại một array là dự đoán của model này dành cho `X_test`.***

Bạn cần viết đoạn code để kiểm tra hàm đã viết và predict với Support Vector Machine.


```python
dataSet.predict(myModel) #Dự đoán với SVC
```




    array([3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 8, 3, 8, 8, 8, 8, 8, 3, 8, 8, 3,
           3, 3, 8, 3, 3, 3, 8, 8, 3, 3, 8, 8, 8, 3, 3, 8, 3, 8, 8, 8, 8, 3,
           8, 8, 8, 3, 3, 8, 8, 3, 8, 8, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 3, 8,
           3, 8, 8, 8, 8, 8, 8, 3, 3, 8, 3, 3, 8, 8, 8, 3, 3, 8, 3, 3, 3, 3,
           8, 3, 3, 8, 8, 8, 8, 8, 3, 3, 8, 3, 3, 8, 3, 8, 3, 3, 3, 8, 8, 8,
           3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8,
           3, 8, 8, 3, 8, 8, 3, 3, 3, 3, 8, 3, 8, 8, 3, 8, 8, 8, 8, 8],
          dtype=int64)




```python
dataSet.predict(myModel2) #Dự đoán với LogisticRegression (C=1)
```




    array([3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 8, 3, 8, 8, 8, 8, 8, 3, 8, 8, 3,
           3, 3, 8, 3, 3, 3, 8, 8, 3, 3, 8, 8, 8, 3, 3, 8, 3, 8, 8, 8, 8, 3,
           8, 8, 8, 3, 3, 8, 8, 3, 8, 8, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 3, 8,
           3, 8, 8, 8, 8, 8, 8, 3, 3, 8, 3, 3, 8, 8, 8, 3, 3, 8, 3, 3, 3, 3,
           8, 3, 3, 8, 8, 8, 8, 8, 3, 3, 8, 3, 3, 8, 3, 8, 8, 3, 3, 8, 8, 8,
           3, 3, 3, 3, 8, 3, 8, 8, 8, 3, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8,
           3, 8, 8, 3, 8, 8, 3, 3, 3, 3, 8, 3, 8, 8, 3, 8, 8, 8, 8, 8],
          dtype=int64)




```python
dataSet.y_test #Dữ liệu thực
```




    array([3, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 8, 3, 8, 8, 8, 8, 8, 3, 8, 8, 3,
           3, 3, 8, 3, 3, 3, 8, 8, 3, 3, 8, 8, 8, 3, 3, 8, 3, 8, 8, 8, 8, 3,
           8, 8, 8, 3, 3, 8, 8, 8, 8, 8, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 3, 3,
           3, 3, 8, 8, 8, 8, 8, 3, 3, 8, 3, 3, 8, 8, 8, 3, 3, 8, 3, 3, 3, 3,
           8, 3, 3, 8, 8, 3, 8, 8, 3, 3, 8, 3, 3, 8, 3, 8, 3, 3, 3, 8, 8, 8,
           3, 3, 3, 3, 8, 3, 8, 8, 8, 3, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 3,
           3, 8, 8, 3, 8, 8, 3, 3, 3, 3, 8, 3, 8, 8, 3, 8, 8, 8, 8, 8],
          dtype=int64)



### Bài 14. Đánh giá dự đoán

***Trong class `DataSet`, viết method `score(self, model)` nhận đối số `model` là model đã train ở bài 12 và trả lại kết quả là accuracy khi dự đoán trên `X_test`. Viết method `getConfusionMatrix(self, model)` trả lại kết quả là confusion matrix khi dự đoán trên `X_test` với model đã train.***


```python
dataSet.score(myModel) # Với SVC
```




    0.9539473684210527




```python
dataSet.score(myModel2) # Với LogisticRegression
```




    0.9605263157894737




```python
dataSet.getConfusionMatrix(myModel) #Có 2 hình của MyTam được gán cho DanTruong, 5 hình của DanTruong được gán cho MyTam
```




    array([[61,  5],
           [ 2, 84]], dtype=int64)




```python
dataSet.getConfusionMatrix(myModel2) #Có 1 hình của MyTam được gán cho DanTruong, 5 hình của DanTruong được gán cho MyTam
```




    array([[61,  5],
           [ 1, 85]], dtype=int64)



### Áp dụng cho nhiều lớp

Nếu viết đúng, các hàm từ bài 10 đến 14 sẽ áp dụng được cho phân loại nhiều lớp. Đoạn code dưới đây cần chạy được và cho ra kết quả.


```python
from itertools import chain
singers = ["MyTam", "DanTruong", "NooPhuocThinh"]

# Lưu vào file MyTam_vs_DanTruong.csv
transformImagesToFacesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, "TextData/MyTam_vs_DanTruong_vs_NooPhuocThinh.csv", chain.from_iterable([SINGER_IMAGE_RANGE[s] for s in singers]))
```

    10 files processed.
    20 files processed.
    30 files processed.
    40 files processed.
    50 files processed.
    60 files processed.
    70 files processed.
    80 files processed.
    90 files processed.
    100 files processed.
    110 files processed.
    120 files processed.
    130 files processed.
    140 files processed.
    150 files processed.
    160 files processed.
    170 files processed.
    180 files processed.
    190 files processed.
    200 files processed.
    210 files processed.
    220 files processed.
    230 files processed.
    240 files processed.
    250 files processed.
    260 files processed.
    270 files processed.
    280 files processed.
    290 files processed.
    300 files processed.
    310 files processed.
    320 files processed.
    330 files processed.
    340 files processed.
    350 files processed.
    360 files processed.
    370 files processed.
    380 files processed.
    390 files processed.
    400 files processed.
    410 files processed.
    420 files processed.
    430 files processed.
    440 files processed.
    450 files processed.
    460 files processed.
    470 files processed.
    480 files processed.
    490 files processed.
    500 files processed.
    510 files processed.
    520 files processed.
    530 files processed.
    540 files processed.
    


```python
dataSet = DataSet("TextData/MyTam_vs_DanTruong_vs_NooPhuocThinh.csv")
dataSet.trainTestSplit(test_size = 0.5)
myModel = SVC(kernel = "linear", decision_function_shape = "ovr")
dataSet.train(myModel)
dataSet.getConfusionMatrix(myModel) #Các hàng theo thứ tự chỉ số của ca sĩ, tức DanTruong, MyTam, NooPhuocThinh theo thứ tự
```




    array([[55,  1, 10],
           [ 3, 79,  6],
           [ 5,  7, 96]], dtype=int64)




```python
dataSet.score(myModel)
```




    0.8778625954198473



### Bài 15. Các biến quan trọng

Sau khi train với $k$ class bằng SVM tuyến tính, ta sẽ tìm ra các hệ số $w_{11}, \ldots, w_{1d}; \ldots,  w_{k1}, \ldots, w_{kd}$ và $b_1, \ldots, b_k$.

***Trong class `DataSet`, viết method `getSignificantFeatures(self, model, seuil = SEUIL)` trả lại các chỉ số toạ độ $i$ mà tại đó tổng các giá trị tuyệt đối của các giá trị $\frac{w_{i1}}{b_1}, \ldots, \frac{w_{ik}}{b_k}$ lớn hơn `seuil`. Đây được xem là các toạ độ quan trọng nhất trong hình ảnh.***

Đoạn code dưới đây giúp test hàm của bạn.


```python
significantFeatures = dataSet.getSignificantFeatures(myModel, seuil = 0.0015)
significantFeatures
```




    [18,
     19,
     20,
     21,
     22,
     23,
     908,
     909,
     910,
     972,
     973,
     974,
     1008,
     1009,
     1010,
     1035,
     1036,
     1037,
     1038,
     1070,
     1071,
     1072,
     1073,
     1074,
     1075,
     1076,
     1077,
     1099,
     1100,
     1101,
     1102,
     1103,
     1104,
     1105,
     1106,
     1107,
     1133,
     1134,
     1135,
     1136,
     1137,
     1138,
     1139,
     1140,
     1141,
     1164,
     1165,
     1166,
     1167,
     1168,
     1169,
     1170,
     1171,
     1172,
     1196,
     1197,
     1198,
     1199,
     1200,
     1201,
     1202,
     1203,
     1204,
     1205,
     1228,
     1229,
     1230,
     1231,
     1232,
     1233,
     1234,
     1235,
     1236,
     1260,
     1261,
     1262,
     1263,
     1264,
     1265,
     1266,
     1267,
     1268,
     1298,
     1324,
     1325,
     1328,
     1390,
     1391,
     1423,
     1424,
     1449,
     1453,
     1454,
     1455,
     1456,
     1457,
     1458,
     1459,
     1460,
     1487,
     1488,
     1512,
     1513,
     1517,
     1518,
     1519,
     1520,
     1521,
     1522,
     1523,
     1524,
     1525,
     1550,
     1551,
     1574,
     1575,
     1576,
     1577,
     1586,
     1587,
     1588,
     1589,
     1614,
     1615,
     1638,
     1639,
     1640,
     1650,
     1651,
     1652,
     1702,
     1703,
     1727,
     1867,
     1931,
     2027,
     2028,
     2043,
     2170,
     2233,
     2234,
     2495,
     2559,
     2613,
     2623,
     2676,
     2677,
     2740,
     2741,
     2803,
     2804,
     2805,
     2851,
     2852,
     2867,
     2868,
     2869,
     2914,
     2915,
     2916,
     2922,
     2931,
     2932,
     2933,
     2995,
     2996,
     3059,
     3060,
     3122,
     3123,
     3187,
     3331,
     3395,
     3658,
     3723,
     3776,
     3777,
     3778,
     3840,
     3841,
     3842,
     3843,
     3904,
     3905,
     3906,
     4092]



Ta có thể xem vị trí của các toạ độ này trong bố cục 64 x 64


```python
# Đưa các toạ độ về 64 x 64
featuresInSquare = np.array([[feature % 64, feature // 64] for feature in significantFeatures])
# Vẽ các toạ độ với matplotlib 
plt.scatter(featuresInSquare[:, 0], featuresInSquare[:, 1])
```




    <matplotlib.collections.PathCollection at 0xf16e4a8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/TD/output_99_1.png)


Chú ý rằng trong hình ảnh, toạ độ (0, 0) sẽ ứng với góc trên, bên trái, nên hình vẽ trên cần được đọc "ngược" từ dưới lên. Ta thấy một phần lớn vị trí các điểm quan trọng để nhận diện các ca sĩ nằm ở khu vực mắt-lông mày. (Đây là lí do ở phần trước ta chú trọng việc trích xuất mắt). Trong [2], phương pháp dùng toàn bộ khuôn mặt được gọi là *global approach*, phương pháp dùng một số thành phần của khuôn mặt được gọi là *component-based approach*.

## Phần 3 - Nhận diện ca sĩ

Đến giờ, bạn đã có công cụ để thực hiện việc train-test với một phần/toàn bộ dữ liệu. Hãy thực hiện các yêu cầu sau:

### Bài 16.
***Dùng SVM tuyến tính và một hay một số phương pháp classify khác đã biết (KNN, LDA, QDA, GNB, LogisticRegression) để train và tính score trên tập test để phân biệt:***
- *Một ca sĩ nam và một ca sĩ nữ: ví dụ BaoThy và DanTruong*
- *Hai ca sĩ nam: ví dụ DanTruong và LamTruong*
- *Hai ca sĩ nữ: ví dụ BaoThy và HuongTram*

***Với các phương pháp xác suất (LDA, QDA, GNB, LogisticRegression), tính ROC AUC trên tập test. Cặp ca sĩ nào cho kết quả tốt nhất, kém hơn?***

### Bài 17.
***Dùng SVM tuyến tính để train và tính score trên tập test để phân biệt một nhóm nhiều ca sĩ nhất có thể (tuỳ tốc độ máy của bạn, tốt nhất là toàn bộ 12 ca sĩ)***

### Bài 18.
***Dùng SVM tuyến tính để train và tính score trên tập test để phân biệt một nhóm nhiều ca sĩ, nhưng thay vì sử dụng dữ liệu của toàn bộ khuôn mặt thì chỉ sử dụng dữ liệu của đôi mắt. Kết quả tốt hay kém hơn so với bài 17?***

### Bài 19.
***Dùng SVM tuyến tính để train và tính score trên tập test để phân biệt một nhóm nhiều ca sĩ, nhưng thay vì sử dụng dữ liệu của đôi mắt thì chỉ sử dụng 1 mắt trái hoặc 1 mắt phải. Kết quả tốt hay kém hơn so với bài 18?***

Dưới đây là một cách thực hiện bài 18.

**Dùng toàn bộ khuôn mặt: 621 hình ảnh được đưa vào tập dữ liệu, sau khi huấn luyện nhận diện đúng 77.17%**


```python
from itertools import chain
singers = ["BaoThy", "HuongTram", "DanTruong", "LamTruong"]
transformImagesToFacesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, "TextData/BaoThy_vs_HuongTram_vs_DanTruong_vs_LamTruong_Faces.csv", chain.from_iterable([SINGER_IMAGE_RANGE[s] for s in singers]))
```

    10 files processed.
    20 files processed.
    30 files processed.
    40 files processed.
    50 files processed.
    60 files processed.
    70 files processed.
    80 files processed.
    90 files processed.
    100 files processed.
    110 files processed.
    120 files processed.
    130 files processed.
    140 files processed.
    150 files processed.
    160 files processed.
    170 files processed.
    180 files processed.
    190 files processed.
    200 files processed.
    210 files processed.
    220 files processed.
    230 files processed.
    240 files processed.
    250 files processed.
    260 files processed.
    270 files processed.
    280 files processed.
    290 files processed.
    300 files processed.
    310 files processed.
    320 files processed.
    330 files processed.
    340 files processed.
    350 files processed.
    360 files processed.
    370 files processed.
    380 files processed.
    390 files processed.
    400 files processed.
    410 files processed.
    420 files processed.
    430 files processed.
    440 files processed.
    450 files processed.
    460 files processed.
    470 files processed.
    480 files processed.
    490 files processed.
    500 files processed.
    510 files processed.
    520 files processed.
    530 files processed.
    540 files processed.
    550 files processed.
    560 files processed.
    570 files processed.
    580 files processed.
    590 files processed.
    600 files processed.
    610 files processed.
    620 files processed.
    630 files processed.
    640 files processed.
    650 files processed.
    


```python
dataSet = DataSet("TextData/BaoThy_vs_HuongTram_vs_DanTruong_vs_LamTruong_Faces.csv")
dataSet.X.shape # Xác định được 621 khuôn mặt
```




    (621, 4096)




```python
dataSet.trainTestSplit(test_size = 0.5)
myModel = SVC(kernel = "linear", decision_function_shape = "ovr")
dataSet.train(myModel)
dataSet.getConfusionMatrix(myModel)
```




    array([[70,  2, 26,  7],
           [ 1, 53,  1,  7],
           [17,  1, 86,  2],
           [ 2,  4,  1, 31]], dtype=int64)




```python
dataSet.score(myModel)
```




    0.7717041800643086



**Dùng đôi mắt: 402 hình ảnh được đưa vào tập dữ liệu, được xem như có chất lượng tốt, sau khi huấn luyện nhận diện đúng 71.14%**


```python

transformImagesToEyesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, "TextData/BaoThy_vs_HuongTram_vs_DanTruong_vs_LamTruong_Eyes.csv", chain.from_iterable([SINGER_IMAGE_RANGE[s] for s in singers]))
```

    10 files processed.
    20 files processed.
    30 files processed.
    40 files processed.
    50 files processed.
    60 files processed.
    70 files processed.
    80 files processed.
    90 files processed.
    100 files processed.
    110 files processed.
    120 files processed.
    130 files processed.
    140 files processed.
    150 files processed.
    160 files processed.
    170 files processed.
    180 files processed.
    190 files processed.
    200 files processed.
    210 files processed.
    220 files processed.
    230 files processed.
    240 files processed.
    250 files processed.
    260 files processed.
    270 files processed.
    280 files processed.
    290 files processed.
    300 files processed.
    310 files processed.
    320 files processed.
    330 files processed.
    340 files processed.
    350 files processed.
    360 files processed.
    370 files processed.
    380 files processed.
    390 files processed.
    400 files processed.
    410 files processed.
    420 files processed.
    430 files processed.
    440 files processed.
    450 files processed.
    460 files processed.
    470 files processed.
    480 files processed.
    490 files processed.
    500 files processed.
    510 files processed.
    520 files processed.
    530 files processed.
    540 files processed.
    550 files processed.
    560 files processed.
    570 files processed.
    580 files processed.
    590 files processed.
    600 files processed.
    610 files processed.
    620 files processed.
    630 files processed.
    640 files processed.
    650 files processed.
    


```python
dataSet = DataSet("TextData/BaoThy_vs_HuongTram_vs_DanTruong_vs_LamTruong_Eyes.csv")
dataSet.X.shape # Chỉ xác định được 402 đôi mắt
```




    (402, 2048)




```python
dataSet.trainTestSplit(test_size = 0.5)
myModel = SVC(kernel = "linear", decision_function_shape = "ovr")
dataSet.train(myModel)
dataSet.getConfusionMatrix(myModel)
```




    array([[45,  4, 18,  3],
           [ 8, 24,  1,  5],
           [12,  0, 59,  0],
           [ 4,  3,  0, 15]], dtype=int64)




```python
dataSet.score(myModel)
```




    0.7114427860696517



Như vậy chỉ với thành phần mắt không thay thế được cho toàn bộ khuôn mặt dù đã sử dụng data set chất lượng tốt.

## Tham khảo

[1] P.Jonathon Phillips, *Support Vector Machines Applied to Face Recognition, National Institute of standards and Technology* 

[2] Bernd Heisele, Purdy Ho, Tomaso Poggio, *Face Recognition with Support Vector Machines: Global versus Component-based Approach*, Masachusetts Institute of Technology, Center of Biological and Computational Learning

[3] https://drive.google.com/file/d/1szwAQHgXUR2vESBexosW0yJvvfKQfWnI/view - Link gốc dữ liệu của Thor Pham, chia sẻ trên blog Machine Learning cơ bản (https://www.facebook.com/groups/machinelearningcoban/permalink/458486681275411/)
