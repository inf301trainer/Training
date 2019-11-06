
# TD8 - Đọc phiếu chấm học bổng Đồng Hành (tiếp theo)

## Yêu cầu

Bạn cần download các thư mục tại TD8. Các thư mục tận cùng bằng '_Solution' chỉ dùng để so sánh kết quả. Dựa trên file source code `DHEvaluation.py` bạn đã hoàn thành trong TD7, hãy tiếp tục viết các đoạn code tiếp theo trong file này để hoàn thành chương trình theo hướng dẫn ở các phần sau:

- Phần 5: Sử dụng các mô hình train-test dành cho multiclass 
- Phần 6: Load mô hình và hoàn chỉnh chương trình dự đoán các dữ liệu mới với các mô hình đã được lưu
- Phần 7: Cách tiếp cận đơn giản với điểm số thập phân


```python
from DHEvaluation_Solution import *
```

## Phần 5 - Các mô hình phân loại nhiều lớp

Ở phân loại 2 lớp, ta đã loại bỏ một số mô hình có độ chính xác thấp như QDA, GNB. Ta giữ lại các mô hình dưới đây cho phân loại nhiều lớp:
- Logistic Regression, phương pháp OVR hoặc dùng multinomial distribution
- LDA, dùng multinomial distribution
- K-Nearest Neighbors với 5 điểm gần nhất hoặc 1 điểm gần nhất

### Bài 24 - Các mô hình phân loại nhiều lớp

*Hãy viết các hàm dưới đây implement các mô hình trên, theo thứ tự.* 

- **`MyLogisticRegression_Multiclass_OVR()`**
- **`MyLogisticRegression_Multiclass_MTN()`**
- **`MyLDA_Multiclass()`**
- **`MyKNN5_Multiclass()`**
- **`MyKNN1_Multiclass()`**

*Đối với LogisticRegression, hãy implement bằng class **`LogisticRegressionCV`** của scikit-learn để bước tối ưu cho regularization được thực hiện một cách tự động.*

Đoạn code dưới đây giúp test hàm của bạn cho hàm đầu tiên, bạn có thể chạy tương tự với các hàm còn lại. Khi in model, tham số `multi_class` của instance là `ovr`. 


```python
donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
donghanh_mark.filterData()
model = MyLogisticRegression_Multiclass_OVR()
model.fit(donghanh_mark.getFilteredX(), donghanh_mark.getFilteredy())
model
```




    LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
               refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)



Kết quả dự đoán cần chứa các class từ 0 đến 10.


```python
model.predict(donghanh_mark.getFilteredX()[:100]) #Predict cho 100 hình ảnh đầu tiên trong tập đã dùng để train
```




    array([10., 10.,  8.,  9., 10.,  9.,  8.,  7.,  7.,  9.,  9.,  9., 10.,
            8., 10., 10.,  9.,  7., 10.,  9.,  8.,  9.,  9., 10.,  9., 10.,
            8.,  9.,  7.,  9., 10.,  8., 10.,  7.,  9.,  9.,  9.,  9.,  8.,
           10.,  7.,  8., 10.,  9., 10., 10.,  7.,  7.,  6.,  4.,  1.,  0.,
            7.,  4.,  1.,  0.,  7.,  2.,  1.,  1.,  6.,  2.,  2.,  0.,  7.,
            3.,  1.,  0.,  6.,  3.,  1.,  0.,  7.,  3.,  2.,  1.,  6.,  2.,
            2.,  1.,  7.,  2.,  1.,  0.,  7.,  3.,  2.,  0.,  8.,  3.,  1.,
            0.,  8.,  2.,  2.,  0.,  5.,  3.,  1.,  1.])



### Bài 25 - Đánh giá các mô hình

Ở TD trước, ta đã có instance method **`getTrainTestScore(self, model, i, j)`** trong class **`Mark`** thực hiện việc tìm accuracy trung bình cho mô hình `model` khi lọc ra các hình ảnh chất lượng thuộc class `i`, `j` và dùng cross validation kiểu K-folds với $K=3$ để chia ngẫu nhiên tập dữ liệu thành 3 phần, 2 để train và 1 để test, thay đổi luân phiên.

Bây giờ, ta muốn sử dụng lại instance method này cho phân loại nhiều lớp, muốn vậy, ta cho các đối số `i, j` thành các đối số không bắt buộc (optional), và thêm một đối số **`scoring`** để có thể đánh giá mô hình bằng các phương pháp khác (f1, precision, recall...). Method trở thành

**`getTrainTestScore(self, model, i = None, j = None, scoring='accuracy')`**

*Hãy chỉnh sửa method này dựa trên code bạn đã viết cho `**getTrainTestScore**` ở TD trước, để nếu giá trị của `i`, `j` không được đề cập (bằng `None`), method sẽ thực hiện việc tính chỉ số scoring của bài toán multiclass classification (nhận 1 trong các giá trị 'accuracy', 'f1_micro', 'f1_macro', 'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro' như ở đây http://scikit-learn.org/stable/modules/model_evaluation.html) khi ngẫu nhiên chia tập dữ liệu thành 3 phần và đánh giá theo 3-fold cross-validation.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
donghanh_mark.filterData()
model = MyLogisticRegression_Multiclass_OVR()
donghanh_mark.getTrainTestScore(model) # Tính accuracy
```




    0.912780209291041




```python
donghanh_mark.getTrainTestScore(model, scoring='precision_macro') # Tính precision_macro
```




    0.9190903304854224



### Bài 26 - Đánh giá các mô hình (toàn diện)

*Hãy viết một đoạn code sử dụng hàm **`getTrainTestScore`** vừa viết để tính các chỉ số 'accuracy', 'f1_micro', 'f1_macro', 'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro' trên data set DONGHANH của 5 mô hình trên thông qua 3-folds cross-validation.*

Bạn có thể biểu diễn kết quả ở dạng array hoặc DataFrame như dưới đây.


```python
A = test_26()
A
```

    Evaluation of model LogisticRegression_OVR, using accuracy
    Evaluation of model LogisticRegression_OVR, using f1_micro
    Evaluation of model LogisticRegression_OVR, using f1_macro
    Evaluation of model LogisticRegression_OVR, using precision_micro
    Evaluation of model LogisticRegression_OVR, using precision_macro
    Evaluation of model LogisticRegression_OVR, using recall_micro
    Evaluation of model LogisticRegression_OVR, using recall_macro
    Evaluation of model LogisticRegression_Multinomial, using accuracy
    Evaluation of model LogisticRegression_Multinomial, using f1_micro
    Evaluation of model LogisticRegression_Multinomial, using f1_macro
    Evaluation of model LogisticRegression_Multinomial, using precision_micro
    Evaluation of model LogisticRegression_Multinomial, using precision_macro
    Evaluation of model LogisticRegression_Multinomial, using recall_micro
    Evaluation of model LogisticRegression_Multinomial, using recall_macro
    Evaluation of model LDA_Multinomial, using accuracy
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model LDA_Multinomial, using f1_micro
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model LDA_Multinomial, using f1_macro
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model LDA_Multinomial, using precision_micro
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model LDA_Multinomial, using precision_macro
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model LDA_Multinomial, using recall_micro
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model LDA_Multinomial, using recall_macro
    

    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    D:\Users\ndoannguyen\AppData\Local\Continuum\anaconda2\envs\Tensorflow3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")
    

    Evaluation of model KNN5, using accuracy
    Evaluation of model KNN5, using f1_micro
    Evaluation of model KNN5, using f1_macro
    Evaluation of model KNN5, using precision_micro
    Evaluation of model KNN5, using precision_macro
    Evaluation of model KNN5, using recall_micro
    Evaluation of model KNN5, using recall_macro
    Evaluation of model KNN1, using accuracy
    Evaluation of model KNN1, using f1_micro
    Evaluation of model KNN1, using f1_macro
    Evaluation of model KNN1, using precision_micro
    Evaluation of model KNN1, using precision_macro
    Evaluation of model KNN1, using recall_micro
    Evaluation of model KNN1, using recall_macro
    




    array([[0.91278021, 0.91278021, 0.90550712, 0.91278021, 0.91909033,
            0.91278021, 0.89827087],
           [0.93223133, 0.93223133, 0.92457184, 0.93223133, 0.93413081,
            0.93223133, 0.92138473],
           [0.8045219 , 0.8045219 , 0.78971981, 0.8045219 , 0.80204618,
            0.8045219 , 0.78664076],
           [0.94788586, 0.94788586, 0.94117794, 0.94788586, 0.9506969 ,
            0.94788586, 0.94003042],
           [0.95832615, 0.95832615, 0.9557867 , 0.95832615, 0.96319252,
            0.95832615, 0.95205785]])




```python
MULTICLASS_MODEL_NAMES
SCORINGS = ["accuracy", "f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]
pd.DataFrame(A, index=MULTICLASS_MODEL_NAMES, columns=SCORINGS)
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
      <th>accuracy</th>
      <th>f1_micro</th>
      <th>f1_macro</th>
      <th>precision_micro</th>
      <th>precision_macro</th>
      <th>recall_micro</th>
      <th>recall_macro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LogisticRegression_OVR</th>
      <td>0.912780</td>
      <td>0.912780</td>
      <td>0.905507</td>
      <td>0.912780</td>
      <td>0.919090</td>
      <td>0.912780</td>
      <td>0.898271</td>
    </tr>
    <tr>
      <th>LogisticRegression_Multinomial</th>
      <td>0.932231</td>
      <td>0.932231</td>
      <td>0.924572</td>
      <td>0.932231</td>
      <td>0.934131</td>
      <td>0.932231</td>
      <td>0.921385</td>
    </tr>
    <tr>
      <th>LDA_Multinomial</th>
      <td>0.804522</td>
      <td>0.804522</td>
      <td>0.789720</td>
      <td>0.804522</td>
      <td>0.802046</td>
      <td>0.804522</td>
      <td>0.786641</td>
    </tr>
    <tr>
      <th>KNN5</th>
      <td>0.947886</td>
      <td>0.947886</td>
      <td>0.941178</td>
      <td>0.947886</td>
      <td>0.950697</td>
      <td>0.947886</td>
      <td>0.940030</td>
    </tr>
    <tr>
      <th>KNN1</th>
      <td>0.958326</td>
      <td>0.958326</td>
      <td>0.955787</td>
      <td>0.958326</td>
      <td>0.963193</td>
      <td>0.958326</td>
      <td>0.952058</td>
    </tr>
  </tbody>
</table>
</div>



Như vậy, sau bước này ta đã hoàn tất train các mô hình. Giả sử ta đã hài lòng với KNN hoặc LogisticRegression (ưu tiên LogisticRegression_Multinomial) để đưa vào thực tế, bước tiếp theo, ta sẽ lưu lại mô hình. Trong trường hợp của các mô hình tham số như LogisticRegression, LDA, ta hoàn toàn có thể vứt đi các dữ liệu train và chỉ giữ lại các tham số đã train.

### Bài 27 - Lưu mô hình

*Hãy hoàn chỉnh method **`saveModelParameters(self, model, model_name, parameters_file)`** lưu các tham số cần thiết của mô hình `model` có tên `model_name` đã được train với toàn bộ hình ảnh có chất lượng tốt của dataset từ `self` vào 1 file `parameters_name`. Cấu trúc của file do bạn quyết định, nó sẽ cần có cấu trúc đơn giản để bạn load lại về sau.*

Chú ý rằng mỗi mô hình có các tham số cần thiết riêng, ví dụ với LDA, các tham số của nó được liệt kê ở mục `attributes` của <a href="http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis">trang sau</a>, bao gồm: 

- `coef_`: Các hệ số của đường thẳng đặc trưng cho biên giới các lớp (array 11x784)
- `intercept_`: Các hệ số tự do của các đường thẳng này (array 11)
- `covariance_`: Ma trận covariance chung cho các lớp ($\Sigma$ trong amphi, array 784 x 784)
- `priors_`: Tỉ lệ dữ liệu của các lớp trong tập train ($\pi_k$ trong amphi, array 11)
- `means_`: Trọng tâm của các lớp ($\mu_k$ trong amphi, array 11 x 784)
- `classes_`: Tên các class (trường hợp của ta là các số 0, 1, $\ldots$, 10)

Bạn nghiên cứu tương tự cho LogisticRegressionCV và KNN.

*Chạy đoạn code dưới đây và kiểm tra nội dung file "SavedModels/LogisticRegression_OVR.txt".*


```python
SCORE_DATA = "ScoreData/"
LABEL_DATA = "LabelData/Labels.csv"
LOGISTIC_REGRESSION_OVR_PARAMS = "SavedModels/LogisticRegression_OVR.txt"

donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
donghanh_mark.filterData()
donghanh_mark.saveModelParameters(MyLogisticRegression_Multiclass_OVR, "LogisticRegression_OVR", LOGISTIC_REGRESSION_OVR_PARAMS)
```

Trong đáp án (`SavedModels_Solution/LogisticRegression_OVR.txt`), file được lưu dưới dạng: dòng đầu tiên là tên model, sau đó là tên các tham số: coef\_ ($\mathbf w$), intercept\_ ($b$), C\_ (các hệ số regularization), tên các class và các array giá trị tương ứng, mỗi hàng của array là một dòng, các số cách nhau bởi dấu phẩy.

Đáp án không implement cho KNN.

## Phần 6 - Sử dụng mô hình để dự đoán dữ liệu mới

### Bài 28 - Load mô hình

*Hãy hoàn chỉnh method **`loadModelParameters(self, parameters_file)`** đọc file `parameters_file` và trả lại mô hình từ file đã lưu, tức một mô hình giống như `MyLogisticRegression_Multiclass_OVR, MyLogisticRegression_Multiclass_MTN, MyLDA_Multiclass, MyKNN5_Multiclass, MyKNN1_Multiclass`. Mô hình này có thể gọi được method `predict()` mà không cần train (gọi method `fit`) và có thể dùng dự đoán ngay cho dữ liệu mới.*

Nói cách khác, sau khi train xong mô hình, ta có thể quên đi những dữ liệu đã train và chỉ cần giữ lại các tham số để dự đoán dữ liệu mới sau này.

Đoạn code dưới đây giúp test hàm của bạn.


```python
SCORE_DATA = "ScoreData/"
LABEL_DATA = "LabelData/Labels.csv"
donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
model = donghanh_mark.loadModelParameters(LOGISTIC_REGRESSION_OVR_PARAMS)
model
```




    LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
               refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)




```python
model.intercept_ #Kết quả cần giống những gì đã lưu trong file thay vì None hay list rỗng
```




    array([ -9.07987045, -29.80441946,  -0.90076684, -12.30850544,
            -6.79405901,  -1.81940433, -14.05347336,   3.64873098,
           -16.12474074,  -9.82872082,  -7.67512244])



### Bài 29 - Dự đoán

*Hãy viết instance method **predict(self, model)** trong class **`Mark`** để từ mô hình `model` đã đọc từ bài 28, dự đoán kết quả cho tất cả các dữ liệu chứa trong instance dưới dạng một array.*

*Hãy viết instance method **predict(self, model)** trong class **`Mark`** để từ mô hình `model` đã đọc từ bài 28, dự đoán kết quả cho tất cả các dữ liệu chứa trong instance dưới dạng một dict với key là tên của hình (`self.name`) và value là điểm số tương ứng.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
SCORE_DATA = "ScoreData/"
LABEL_DATA = "LabelData/Labels.csv"
donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
model = donghanh_mark.loadModelParameters(LOGISTIC_REGRESSION_OVR_PARAMS)
result = donghanh_mark.predict(model)
result
```




    array([10., 10.,  8.,  9., 10.,  9.,  8.,  7.,  7.,  9.,  9.,  9., 10.,
            8., 10., 10.,  9.,  7., 10.,  9.,  8.,  9.,  9., 10.,  9., 10.,
            8.,  9.,  7.,  9., 10.,  8., 10.,  7.,  9.,  9.,  9.,  9.,  8.,
           10.,  7.,  8., 10.,  9., 10., 10.,  7.,  7.,  6.,  4.,  1.,  0.,
            7.,  4.,  1.,  0.,  7.,  2.,  1.,  1.,  6.,  2.,  2.,  0.,  7.,
            3.,  1.,  0.,  6.,  3.,  1.,  0.,  7.,  3.,  2.,  1.,  6.,  2.,
            2.,  1.,  7.,  2.,  1.,  0.,  7.,  3.,  2.,  0.,  8.,  3.,  1.,
            0.,  8.,  2.,  2.,  0.,  5.,  3.,  1.,  1.,  5.,  3.,  1.,  1.,
            5.,  3.,  1.,  1.,  5.,  3.,  1.,  1.,  4.,  2.,  1.,  1.,  4.,
            3.,  1.,  1.,  5.,  3.,  0.,  0.,  4.,  3.,  0.,  1.,  5.,  2.,
            1.,  1.,  4.,  3.,  1.,  1.,  4.,  3.,  1.,  1.,  5.,  2.,  0.,
            0., 10.,  9., 10.,  5.,  7., 10., 10.,  5.,  7.,  9., 10.,  5.,
            9.,  9., 10.,  5., 10.,  9., 10.,  5.,  7., 10., 10.,  5.,  8.,
            9., 10.,  5.,  8., 10., 10.,  5.,  7.,  9., 10.,  5.,  9., 10.,
           10.,  5.,  7.,  9., 10.,  5.,  9., 10., 10.,  5., 10.,  7.,  6.,
            9., 10.,  7.,  3.,  9.,  5.,  8.,  3.,  5.,  5.,  7.,  6.,  6.,
           10.,  8.,  6.,  7., 10.,  8.,  6.,  9., 10.,  7.,  6.,  7.,  5.,
            8.,  6.,  9., 10.,  7.,  6.,  6., 10.,  8.,  6.,  6.,  5.,  7.,
            3.,  6., 10.,  7.,  6.,  5.,  7.,  6.,  1.,  2.,  7.,  5.,  1.,
            2.,  8.,  3.,  0.,  4.,  7.,  2.,  1.,  5.,  7.,  4.,  0.,  3.,
            7.,  3.,  0.,  2.,  8.,  3.,  1.,  3.,  8.,  3.,  0.,  2.,  8.,
            6.,  1.,  7.,  7.,  3.,  1.,  5.,  7.,  5.,  0.,  6.,  7.,  6.,
            1.,  4.,  6.,  2.,  3.,  3.,  3.,  0.,  1.,  2.,  6.,  1.,  4.,
            1.,  9.,  2.,  4.,  0.,  6.,  2.,  3.,  4.,  9.,  2.,  3.,  3.,
            3.,  1.,  5.,  4.,  6.,  1.,  5.,  3.,  9.,  1.,  4.,  2.,  6.,
            1.,  4.,  1.,  9.,  1.,  4.,  0.,  3.,  1.,  4.,  0.,  3.,  3.,
            2.,  0.,  2.,  2.,  3.,  0.,  5.,  2.,  5.,  0.,  4.,  3.,  1.,
            1.,  5.,  3.,  2.,  0.,  4.,  2.,  4.,  0.,  1.,  3.,  2.,  0.,
            5.,  2.,  4.,  1.,  4.,  2.,  2.,  0.,  5.,  3.,  4.,  1.,  4.,
            2.,  2.,  0.,  1.,  3.,  4.,  1., 10.,  9.,  3.,  6., 10.,  9.,
            6.,  3.,  6.,  7.,  7.,  6.,  5.,  7.,  6.,  7.,  9.,  7.,  7.,
            7.,  9., 10.,  7.,  6.,  8., 10.,  5.,  5.,  8., 10.,  5.,  4.,
            8.,  6.,  4.,  3.,  7.,  5.,  2.,  3.,  7., 10.,  4.,  6.,  6.,
            6.,  7.,  5.,  9., 10.,  9.,  5.,  9., 10.,  7.,  5.,  8., 10.,
            9.,  2., 10.,  9.,  6.,  4.,  8., 10.,  9.,  5.,  6.,  9.,  9.,
            5.,  6.,  9., 10.,  3.,  9.,  8.,  9.,  6.,  4.,  9.,  7.,  7.,
            6.,  8.,  9.,  5.,  9.,  9., 10.,  4.,  9.,  7.,  6.,  3.,  9.,
            7.,  9., 10.,  7.,  9.,  7., 10.,  9.,  8.,  9.,  0.,  9.,  6.,
            9., 10.,  9.,  9.,  6., 10.,  6.,  6.,  6., 10.,  9.,  9.,  9.,
           10.,  7.,  7.,  7., 10.,  8.,  9.,  7., 10.,  8.,  8.,  9., 10.,
            6.,  6.,  9., 10.,  9.,  9.,  8.,  0.,  9.,  5.,  3.,  2.,  7.,
            9.,  3.,  0.,  9.,  9.,  3.,  0.,  7.,  8.,  1.,  2.,  9.,  7.,
            1.,  0.,  5.,  7.,  1.,  2.,  7.,  6.,  3.,  2.,  6.,  8.,  3.,
            2.,  9.,  8.,  1.,  1.,  7.,  4.,  1.,  1.,  9.,  2.,  3.,  2.,
            6.,  2.,  1.,  1.,  6.,  7.,  4., 10.,  7.,  1.,  4.,  0.,  1.,
            5.,  2., 10.,  6.,  5.,  3.,  0.,  6.,  6.,  4.,  0.,  1.,  6.,
            7., 10.,  6.,  9.,  6.,  5.,  9.,  1.,  9.,  5.,  6.,  6.,  7.,
           10.,  1.,  7.,  6.,  0.,  6.,  7.,  5.,  5.,  7.,  6.,  2.,  5.,
            6.,  7.,  4.,  3.,  7.,  6.,  4.,  5.,  3.,  7.,  2.,  6.,  7.,
            5.,  1.,  3.,  6.,  9.,  3.,  5.,  3.,  3.,  4.,  4.,  6.,  9.,
            4.,  6.,  9.,  7.,  2.,  5.,  7.,  6.,  2.,  3.,  6.,  6.,  4.,
            2.,  9.,  9.,  3.,  6.,  9.,  6.,  4.,  3.,  9.,  7.,  7.,  8.,
            9.,  9.,  6.,  8.,  9.,  9.,  6., 10.,  6.,  9.,  7., 10.,  7.,
            6.,  9.,  8.,  6.,  9.,  7., 10.,  9.,  7.,  6., 10.,  6.,  7.,
            9.,  8.,  9.,  6.,  7., 10.,  6.,  6.,  6., 10.,  9.,  9.,  9.,
            8.,  7.,  7.,  7.,  8.,  9.,  3.,  7.,  7.,  9.,  5.,  3.,  6.,
            9.,  6.,  9.,  5.,  8.,  6.,  7.,  9.,  6.,  3.,  7.,  7.,  5.,
            2.,  6.,  9.,  4.,  5.,  7.,  8.,  9.,  1.,  9.,  8.,  5.,  5.,
            8.,  9.,  6.,  3.,  7.,  7.,  9.,  5.,  5.,  3.,  7.,  1.,  6.,
            6.])




```python
result = donghanh_mark.predictAsDict(model)
result
```




    {'BKDN_1_0-A.jpg': 10.0,
     'BKDN_1_0-B.jpg': 10.0,
     'BKDN_1_0-C.jpg': 8.0,
     'BKDN_1_0-D.jpg': 9.0,
     'BKDN_1_1-A.jpg': 10.0,
     'BKDN_1_1-B.jpg': 9.0,
     'BKDN_1_1-C.jpg': 8.0,
     'BKDN_1_1-D.jpg': 7.0,
     'BKDN_1_10-A.jpg': 7.0,
     'BKDN_1_10-B.jpg': 9.0,
     'BKDN_1_10-C.jpg': 9.0,
     'BKDN_1_10-D.jpg': 9.0,
     'BKDN_1_11-A.jpg': 10.0,
     'BKDN_1_11-B.jpg': 8.0,
     'BKDN_1_11-C.jpg': 10.0,
     'BKDN_1_11-D.jpg': 10.0,
     'BKDN_1_2-A.jpg': 9.0,
     'BKDN_1_2-B.jpg': 7.0,
     'BKDN_1_2-C.jpg': 10.0,
     'BKDN_1_2-D.jpg': 9.0,
     'BKDN_1_3-A.jpg': 8.0,
     'BKDN_1_3-B.jpg': 9.0,
     'BKDN_1_3-C.jpg': 9.0,
     'BKDN_1_3-D.jpg': 10.0,
     'BKDN_1_4-A.jpg': 9.0,
     'BKDN_1_4-B.jpg': 10.0,
     'BKDN_1_4-C.jpg': 8.0,
     'BKDN_1_4-D.jpg': 9.0,
     'BKDN_1_5-A.jpg': 7.0,
     'BKDN_1_5-B.jpg': 9.0,
     'BKDN_1_5-C.jpg': 10.0,
     'BKDN_1_5-D.jpg': 8.0,
     'BKDN_1_6-A.jpg': 10.0,
     'BKDN_1_6-B.jpg': 7.0,
     'BKDN_1_6-C.jpg': 9.0,
     'BKDN_1_6-D.jpg': 9.0,
     'BKDN_1_7-A.jpg': 9.0,
     'BKDN_1_7-B.jpg': 9.0,
     'BKDN_1_7-C.jpg': 8.0,
     'BKDN_1_7-D.jpg': 10.0,
     'BKDN_1_8-A.jpg': 7.0,
     'BKDN_1_8-B.jpg': 8.0,
     'BKDN_1_8-C.jpg': 10.0,
     'BKDN_1_8-D.jpg': 9.0,
     'BKDN_1_9-A.jpg': 10.0,
     'BKDN_1_9-B.jpg': 10.0,
     'BKDN_1_9-C.jpg': 7.0,
     'BKDN_1_9-D.jpg': 7.0,
     'BKDN_2_0-A.jpg': 6.0,
     'BKDN_2_0-B.jpg': 4.0,
     'BKDN_2_0-C.jpg': 1.0,
     'BKDN_2_0-D.jpg': 0.0,
     'BKDN_2_1-A.jpg': 7.0,
     'BKDN_2_1-B.jpg': 4.0,
     'BKDN_2_1-C.jpg': 1.0,
     'BKDN_2_1-D.jpg': 0.0,
     'BKDN_2_10-A.jpg': 7.0,
     'BKDN_2_10-B.jpg': 2.0,
     'BKDN_2_10-C.jpg': 1.0,
     'BKDN_2_10-D.jpg': 1.0,
     'BKDN_2_11-A.jpg': 6.0,
     'BKDN_2_11-B.jpg': 2.0,
     'BKDN_2_11-C.jpg': 2.0,
     'BKDN_2_11-D.jpg': 0.0,
     'BKDN_2_2-A.jpg': 7.0,
     'BKDN_2_2-B.jpg': 3.0,
     'BKDN_2_2-C.jpg': 1.0,
     'BKDN_2_2-D.jpg': 0.0,
     'BKDN_2_3-A.jpg': 6.0,
     'BKDN_2_3-B.jpg': 3.0,
     'BKDN_2_3-C.jpg': 1.0,
     'BKDN_2_3-D.jpg': 0.0,
     'BKDN_2_4-A.jpg': 7.0,
     'BKDN_2_4-B.jpg': 3.0,
     'BKDN_2_4-C.jpg': 2.0,
     'BKDN_2_4-D.jpg': 1.0,
     'BKDN_2_5-A.jpg': 6.0,
     'BKDN_2_5-B.jpg': 2.0,
     'BKDN_2_5-C.jpg': 2.0,
     'BKDN_2_5-D.jpg': 1.0,
     'BKDN_2_6-A.jpg': 7.0,
     'BKDN_2_6-B.jpg': 2.0,
     'BKDN_2_6-C.jpg': 1.0,
     'BKDN_2_6-D.jpg': 0.0,
     'BKDN_2_7-A.jpg': 7.0,
     'BKDN_2_7-B.jpg': 3.0,
     'BKDN_2_7-C.jpg': 2.0,
     'BKDN_2_7-D.jpg': 0.0,
     'BKDN_2_8-A.jpg': 8.0,
     'BKDN_2_8-B.jpg': 3.0,
     'BKDN_2_8-C.jpg': 1.0,
     'BKDN_2_8-D.jpg': 0.0,
     'BKDN_2_9-A.jpg': 8.0,
     'BKDN_2_9-B.jpg': 2.0,
     'BKDN_2_9-C.jpg': 2.0,
     'BKDN_2_9-D.jpg': 0.0,
     'BKDN_3_0-A.jpg': 5.0,
     'BKDN_3_0-B.jpg': 3.0,
     'BKDN_3_0-C.jpg': 1.0,
     'BKDN_3_0-D.jpg': 1.0,
     'BKDN_3_1-A.jpg': 5.0,
     'BKDN_3_1-B.jpg': 3.0,
     'BKDN_3_1-C.jpg': 1.0,
     'BKDN_3_1-D.jpg': 1.0,
     'BKDN_3_10-A.jpg': 5.0,
     'BKDN_3_10-B.jpg': 3.0,
     'BKDN_3_10-C.jpg': 1.0,
     'BKDN_3_10-D.jpg': 1.0,
     'BKDN_3_11-A.jpg': 5.0,
     'BKDN_3_11-B.jpg': 3.0,
     'BKDN_3_11-C.jpg': 1.0,
     'BKDN_3_11-D.jpg': 1.0,
     'BKDN_3_2-A.jpg': 4.0,
     'BKDN_3_2-B.jpg': 2.0,
     'BKDN_3_2-C.jpg': 1.0,
     'BKDN_3_2-D.jpg': 1.0,
     'BKDN_3_3-A.jpg': 4.0,
     'BKDN_3_3-B.jpg': 3.0,
     'BKDN_3_3-C.jpg': 1.0,
     'BKDN_3_3-D.jpg': 1.0,
     'BKDN_3_4-A.jpg': 5.0,
     'BKDN_3_4-B.jpg': 3.0,
     'BKDN_3_4-C.jpg': 0.0,
     'BKDN_3_4-D.jpg': 0.0,
     'BKDN_3_5-A.jpg': 4.0,
     'BKDN_3_5-B.jpg': 3.0,
     'BKDN_3_5-C.jpg': 0.0,
     'BKDN_3_5-D.jpg': 1.0,
     'BKDN_3_6-A.jpg': 5.0,
     'BKDN_3_6-B.jpg': 2.0,
     'BKDN_3_6-C.jpg': 1.0,
     'BKDN_3_6-D.jpg': 1.0,
     'BKDN_3_7-A.jpg': 4.0,
     'BKDN_3_7-B.jpg': 3.0,
     'BKDN_3_7-C.jpg': 1.0,
     'BKDN_3_7-D.jpg': 1.0,
     'BKDN_3_8-A.jpg': 4.0,
     'BKDN_3_8-B.jpg': 3.0,
     'BKDN_3_8-C.jpg': 1.0,
     'BKDN_3_8-D.jpg': 1.0,
     'BKDN_3_9-A.jpg': 5.0,
     'BKDN_3_9-B.jpg': 2.0,
     'BKDN_3_9-C.jpg': 0.0,
     'BKDN_3_9-D.jpg': 0.0,
     'BKDN_4_0-A.jpg': 10.0,
     'BKDN_4_0-B.jpg': 9.0,
     'BKDN_4_0-C.jpg': 10.0,
     'BKDN_4_0-D.jpg': 5.0,
     'BKDN_4_1-A.jpg': 7.0,
     'BKDN_4_1-B.jpg': 10.0,
     'BKDN_4_1-C.jpg': 10.0,
     'BKDN_4_1-D.jpg': 5.0,
     'BKDN_4_10-A.jpg': 7.0,
     'BKDN_4_10-B.jpg': 9.0,
     'BKDN_4_10-C.jpg': 10.0,
     'BKDN_4_10-D.jpg': 5.0,
     'BKDN_4_11-A.jpg': 9.0,
     'BKDN_4_11-B.jpg': 9.0,
     'BKDN_4_11-C.jpg': 10.0,
     'BKDN_4_11-D.jpg': 5.0,
     'BKDN_4_2-A.jpg': 10.0,
     'BKDN_4_2-B.jpg': 9.0,
     'BKDN_4_2-C.jpg': 10.0,
     'BKDN_4_2-D.jpg': 5.0,
     'BKDN_4_3-A.jpg': 7.0,
     'BKDN_4_3-B.jpg': 10.0,
     'BKDN_4_3-C.jpg': 10.0,
     'BKDN_4_3-D.jpg': 5.0,
     'BKDN_4_4-A.jpg': 8.0,
     'BKDN_4_4-B.jpg': 9.0,
     'BKDN_4_4-C.jpg': 10.0,
     'BKDN_4_4-D.jpg': 5.0,
     'BKDN_4_5-A.jpg': 8.0,
     'BKDN_4_5-B.jpg': 10.0,
     'BKDN_4_5-C.jpg': 10.0,
     'BKDN_4_5-D.jpg': 5.0,
     'BKDN_4_6-A.jpg': 7.0,
     'BKDN_4_6-B.jpg': 9.0,
     'BKDN_4_6-C.jpg': 10.0,
     'BKDN_4_6-D.jpg': 5.0,
     'BKDN_4_7-A.jpg': 9.0,
     'BKDN_4_7-B.jpg': 10.0,
     'BKDN_4_7-C.jpg': 10.0,
     'BKDN_4_7-D.jpg': 5.0,
     'BKDN_4_8-A.jpg': 7.0,
     'BKDN_4_8-B.jpg': 9.0,
     'BKDN_4_8-C.jpg': 10.0,
     'BKDN_4_8-D.jpg': 5.0,
     'BKDN_4_9-A.jpg': 9.0,
     'BKDN_4_9-B.jpg': 10.0,
     'BKDN_4_9-C.jpg': 10.0,
     'BKDN_4_9-D.jpg': 5.0,
     'BKDN_5_0-A.jpg': 10.0,
     'BKDN_5_0-B.jpg': 7.0,
     'BKDN_5_0-C.jpg': 6.0,
     'BKDN_5_0-D.jpg': 9.0,
     'BKDN_5_1-A.jpg': 10.0,
     'BKDN_5_1-B.jpg': 7.0,
     'BKDN_5_1-C.jpg': 3.0,
     'BKDN_5_1-D.jpg': 9.0,
     'BKDN_5_10-A.jpg': 5.0,
     'BKDN_5_10-B.jpg': 8.0,
     'BKDN_5_10-C.jpg': 3.0,
     'BKDN_5_10-D.jpg': 5.0,
     'BKDN_5_11-A.jpg': 5.0,
     'BKDN_5_11-B.jpg': 7.0,
     'BKDN_5_11-C.jpg': 6.0,
     'BKDN_5_11-D.jpg': 6.0,
     'BKDN_5_2-A.jpg': 10.0,
     'BKDN_5_2-B.jpg': 8.0,
     'BKDN_5_2-C.jpg': 6.0,
     'BKDN_5_2-D.jpg': 7.0,
     'BKDN_5_3-A.jpg': 10.0,
     'BKDN_5_3-B.jpg': 8.0,
     'BKDN_5_3-C.jpg': 6.0,
     'BKDN_5_3-D.jpg': 9.0,
     'BKDN_5_4-A.jpg': 10.0,
     'BKDN_5_4-B.jpg': 7.0,
     'BKDN_5_4-C.jpg': 6.0,
     'BKDN_5_4-D.jpg': 7.0,
     'BKDN_5_5-A.jpg': 5.0,
     'BKDN_5_5-B.jpg': 8.0,
     'BKDN_5_5-C.jpg': 6.0,
     'BKDN_5_5-D.jpg': 9.0,
     'BKDN_5_6-A.jpg': 10.0,
     'BKDN_5_6-B.jpg': 7.0,
     'BKDN_5_6-C.jpg': 6.0,
     'BKDN_5_6-D.jpg': 6.0,
     'BKDN_5_7-A.jpg': 10.0,
     'BKDN_5_7-B.jpg': 8.0,
     'BKDN_5_7-C.jpg': 6.0,
     'BKDN_5_7-D.jpg': 6.0,
     'BKDN_5_8-A.jpg': 5.0,
     'BKDN_5_8-B.jpg': 7.0,
     'BKDN_5_8-C.jpg': 3.0,
     'BKDN_5_8-D.jpg': 6.0,
     'BKDN_5_9-A.jpg': 10.0,
     'BKDN_5_9-B.jpg': 7.0,
     'BKDN_5_9-C.jpg': 6.0,
     'BKDN_5_9-D.jpg': 5.0,
     'BKHCM_1_0-A.jpg': 7.0,
     'BKHCM_1_0-B.jpg': 6.0,
     'BKHCM_1_0-C.jpg': 1.0,
     'BKHCM_1_0-D.jpg': 2.0,
     'BKHCM_1_1-A.jpg': 7.0,
     'BKHCM_1_1-B.jpg': 5.0,
     'BKHCM_1_1-C.jpg': 1.0,
     'BKHCM_1_1-D.jpg': 2.0,
     'BKHCM_1_10-A.jpg': 8.0,
     'BKHCM_1_10-B.jpg': 3.0,
     'BKHCM_1_10-C.jpg': 0.0,
     'BKHCM_1_10-D.jpg': 4.0,
     'BKHCM_1_11-A.jpg': 7.0,
     'BKHCM_1_11-B.jpg': 2.0,
     'BKHCM_1_11-C.jpg': 1.0,
     'BKHCM_1_11-D.jpg': 5.0,
     'BKHCM_1_2-A.jpg': 7.0,
     'BKHCM_1_2-B.jpg': 4.0,
     'BKHCM_1_2-C.jpg': 0.0,
     'BKHCM_1_2-D.jpg': 3.0,
     'BKHCM_1_3-A.jpg': 7.0,
     'BKHCM_1_3-B.jpg': 3.0,
     'BKHCM_1_3-C.jpg': 0.0,
     'BKHCM_1_3-D.jpg': 2.0,
     'BKHCM_1_4-A.jpg': 8.0,
     'BKHCM_1_4-B.jpg': 3.0,
     'BKHCM_1_4-C.jpg': 1.0,
     'BKHCM_1_4-D.jpg': 3.0,
     'BKHCM_1_5-A.jpg': 8.0,
     'BKHCM_1_5-B.jpg': 3.0,
     'BKHCM_1_5-C.jpg': 0.0,
     'BKHCM_1_5-D.jpg': 2.0,
     'BKHCM_1_6-A.jpg': 8.0,
     'BKHCM_1_6-B.jpg': 6.0,
     'BKHCM_1_6-C.jpg': 1.0,
     'BKHCM_1_6-D.jpg': 7.0,
     'BKHCM_1_7-A.jpg': 7.0,
     'BKHCM_1_7-B.jpg': 3.0,
     'BKHCM_1_7-C.jpg': 1.0,
     'BKHCM_1_7-D.jpg': 5.0,
     'BKHCM_1_8-A.jpg': 7.0,
     'BKHCM_1_8-B.jpg': 5.0,
     'BKHCM_1_8-C.jpg': 0.0,
     'BKHCM_1_8-D.jpg': 6.0,
     'BKHCM_1_9-A.jpg': 7.0,
     'BKHCM_1_9-B.jpg': 6.0,
     'BKHCM_1_9-C.jpg': 1.0,
     'BKHCM_1_9-D.jpg': 4.0,
     'BKHCM_2_0-A.jpg': 6.0,
     'BKHCM_2_0-B.jpg': 2.0,
     'BKHCM_2_0-C.jpg': 3.0,
     'BKHCM_2_0-D.jpg': 3.0,
     'BKHCM_2_1-A.jpg': 3.0,
     'BKHCM_2_1-B.jpg': 0.0,
     'BKHCM_2_1-C.jpg': 1.0,
     'BKHCM_2_1-D.jpg': 2.0,
     'BKHCM_2_10-A.jpg': 6.0,
     'BKHCM_2_10-B.jpg': 1.0,
     'BKHCM_2_10-C.jpg': 4.0,
     'BKHCM_2_10-D.jpg': 1.0,
     'BKHCM_2_11-A.jpg': 9.0,
     'BKHCM_2_11-B.jpg': 2.0,
     'BKHCM_2_11-C.jpg': 4.0,
     'BKHCM_2_11-D.jpg': 0.0,
     'BKHCM_2_2-A.jpg': 6.0,
     'BKHCM_2_2-B.jpg': 2.0,
     'BKHCM_2_2-C.jpg': 3.0,
     'BKHCM_2_2-D.jpg': 4.0,
     'BKHCM_2_3-A.jpg': 9.0,
     'BKHCM_2_3-B.jpg': 2.0,
     'BKHCM_2_3-C.jpg': 3.0,
     'BKHCM_2_3-D.jpg': 3.0,
     'BKHCM_2_4-A.jpg': 3.0,
     'BKHCM_2_4-B.jpg': 1.0,
     'BKHCM_2_4-C.jpg': 5.0,
     'BKHCM_2_4-D.jpg': 4.0,
     'BKHCM_2_5-A.jpg': 6.0,
     'BKHCM_2_5-B.jpg': 1.0,
     'BKHCM_2_5-C.jpg': 5.0,
     'BKHCM_2_5-D.jpg': 3.0,
     'BKHCM_2_6-A.jpg': 9.0,
     'BKHCM_2_6-B.jpg': 1.0,
     'BKHCM_2_6-C.jpg': 4.0,
     'BKHCM_2_6-D.jpg': 2.0,
     'BKHCM_2_7-A.jpg': 6.0,
     'BKHCM_2_7-B.jpg': 1.0,
     'BKHCM_2_7-C.jpg': 4.0,
     'BKHCM_2_7-D.jpg': 1.0,
     'BKHCM_2_8-A.jpg': 9.0,
     'BKHCM_2_8-B.jpg': 1.0,
     'BKHCM_2_8-C.jpg': 4.0,
     'BKHCM_2_8-D.jpg': 0.0,
     'BKHCM_2_9-A.jpg': 3.0,
     'BKHCM_2_9-B.jpg': 1.0,
     'BKHCM_2_9-C.jpg': 4.0,
     'BKHCM_2_9-D.jpg': 0.0,
     'BKHCM_3_0-A.jpg': 3.0,
     'BKHCM_3_0-B.jpg': 3.0,
     'BKHCM_3_0-C.jpg': 2.0,
     'BKHCM_3_0-D.jpg': 0.0,
     'BKHCM_3_1-A.jpg': 2.0,
     'BKHCM_3_1-B.jpg': 2.0,
     'BKHCM_3_1-C.jpg': 3.0,
     'BKHCM_3_1-D.jpg': 0.0,
     'BKHCM_3_10-A.jpg': 5.0,
     'BKHCM_3_10-B.jpg': 2.0,
     'BKHCM_3_10-C.jpg': 5.0,
     'BKHCM_3_10-D.jpg': 0.0,
     'BKHCM_3_11-A.jpg': 4.0,
     'BKHCM_3_11-B.jpg': 3.0,
     'BKHCM_3_11-C.jpg': 1.0,
     'BKHCM_3_11-D.jpg': 1.0,
     'BKHCM_3_2-A.jpg': 5.0,
     'BKHCM_3_2-B.jpg': 3.0,
     'BKHCM_3_2-C.jpg': 2.0,
     'BKHCM_3_2-D.jpg': 0.0,
     'BKHCM_3_3-A.jpg': 4.0,
     'BKHCM_3_3-B.jpg': 2.0,
     'BKHCM_3_3-C.jpg': 4.0,
     'BKHCM_3_3-D.jpg': 0.0,
     'BKHCM_3_4-A.jpg': 1.0,
     'BKHCM_3_4-B.jpg': 3.0,
     'BKHCM_3_4-C.jpg': 2.0,
     'BKHCM_3_4-D.jpg': 0.0,
     'BKHCM_3_5-A.jpg': 5.0,
     'BKHCM_3_5-B.jpg': 2.0,
     'BKHCM_3_5-C.jpg': 4.0,
     'BKHCM_3_5-D.jpg': 1.0,
     'BKHCM_3_6-A.jpg': 4.0,
     'BKHCM_3_6-B.jpg': 2.0,
     'BKHCM_3_6-C.jpg': 2.0,
     'BKHCM_3_6-D.jpg': 0.0,
     'BKHCM_3_7-A.jpg': 5.0,
     'BKHCM_3_7-B.jpg': 3.0,
     'BKHCM_3_7-C.jpg': 4.0,
     'BKHCM_3_7-D.jpg': 1.0,
     'BKHCM_3_8-A.jpg': 4.0,
     'BKHCM_3_8-B.jpg': 2.0,
     'BKHCM_3_8-C.jpg': 2.0,
     'BKHCM_3_8-D.jpg': 0.0,
     'BKHCM_3_9-A.jpg': 1.0,
     'BKHCM_3_9-B.jpg': 3.0,
     'BKHCM_3_9-C.jpg': 4.0,
     'BKHCM_3_9-D.jpg': 1.0,
     'BKHCM_4_0-A.jpg': 10.0,
     'BKHCM_4_0-B.jpg': 9.0,
     'BKHCM_4_0-C.jpg': 3.0,
     'BKHCM_4_0-D.jpg': 6.0,
     'BKHCM_4_1-A.jpg': 10.0,
     'BKHCM_4_1-B.jpg': 9.0,
     'BKHCM_4_1-C.jpg': 6.0,
     'BKHCM_4_1-D.jpg': 3.0,
     'BKHCM_4_10-A.jpg': 6.0,
     'BKHCM_4_10-B.jpg': 7.0,
     'BKHCM_4_10-C.jpg': 7.0,
     'BKHCM_4_10-D.jpg': 6.0,
     'BKHCM_4_11-A.jpg': 5.0,
     'BKHCM_4_11-B.jpg': 7.0,
     'BKHCM_4_11-C.jpg': 6.0,
     'BKHCM_4_11-D.jpg': 7.0,
     'BKHCM_4_2-A.jpg': 9.0,
     'BKHCM_4_2-B.jpg': 7.0,
     'BKHCM_4_2-C.jpg': 7.0,
     'BKHCM_4_2-D.jpg': 7.0,
     'BKHCM_4_3-A.jpg': 9.0,
     'BKHCM_4_3-B.jpg': 10.0,
     'BKHCM_4_3-C.jpg': 7.0,
     'BKHCM_4_3-D.jpg': 6.0,
     'BKHCM_4_4-A.jpg': 8.0,
     'BKHCM_4_4-B.jpg': 10.0,
     'BKHCM_4_4-C.jpg': 5.0,
     'BKHCM_4_4-D.jpg': 5.0,
     'BKHCM_4_5-A.jpg': 8.0,
     'BKHCM_4_5-B.jpg': 10.0,
     'BKHCM_4_5-C.jpg': 5.0,
     'BKHCM_4_5-D.jpg': 4.0,
     'BKHCM_4_6-A.jpg': 8.0,
     'BKHCM_4_6-B.jpg': 6.0,
     'BKHCM_4_6-C.jpg': 4.0,
     'BKHCM_4_6-D.jpg': 3.0,
     'BKHCM_4_7-A.jpg': 7.0,
     'BKHCM_4_7-B.jpg': 5.0,
     'BKHCM_4_7-C.jpg': 2.0,
     'BKHCM_4_7-D.jpg': 3.0,
     'BKHCM_4_8-A.jpg': 7.0,
     'BKHCM_4_8-B.jpg': 10.0,
     'BKHCM_4_8-C.jpg': 4.0,
     'BKHCM_4_8-D.jpg': 6.0,
     'BKHCM_4_9-A.jpg': 6.0,
     'BKHCM_4_9-B.jpg': 6.0,
     'BKHCM_4_9-C.jpg': 7.0,
     'BKHCM_4_9-D.jpg': 5.0,
     'BKHCM_5_0-A.jpg': 9.0,
     'BKHCM_5_0-B.jpg': 10.0,
     'BKHCM_5_0-C.jpg': 9.0,
     'BKHCM_5_0-D.jpg': 5.0,
     'BKHCM_5_1-A.jpg': 9.0,
     'BKHCM_5_1-B.jpg': 10.0,
     'BKHCM_5_1-C.jpg': 7.0,
     'BKHCM_5_1-D.jpg': 5.0,
     'BKHCM_5_10-A.jpg': 8.0,
     'BKHCM_5_10-B.jpg': 10.0,
     'BKHCM_5_10-C.jpg': 9.0,
     'BKHCM_5_10-D.jpg': 2.0,
     'BKHCM_5_11-A.jpg': 10.0,
     'BKHCM_5_11-B.jpg': 9.0,
     'BKHCM_5_11-C.jpg': 6.0,
     'BKHCM_5_11-D.jpg': 4.0,
     'BKHCM_5_2-A.jpg': 8.0,
     'BKHCM_5_2-B.jpg': 10.0,
     'BKHCM_5_2-C.jpg': 9.0,
     'BKHCM_5_2-D.jpg': 5.0,
     'BKHCM_5_3-A.jpg': 6.0,
     'BKHCM_5_3-B.jpg': 9.0,
     'BKHCM_5_3-C.jpg': 9.0,
     'BKHCM_5_3-D.jpg': 5.0,
     'BKHCM_5_4-A.jpg': 6.0,
     'BKHCM_5_4-B.jpg': 9.0,
     'BKHCM_5_4-C.jpg': 10.0,
     'BKHCM_5_4-D.jpg': 3.0,
     'BKHCM_5_5-A.jpg': 9.0,
     'BKHCM_5_5-B.jpg': 8.0,
     'BKHCM_5_5-C.jpg': 9.0,
     'BKHCM_5_5-D.jpg': 6.0,
     'BKHCM_5_6-A.jpg': 4.0,
     'BKHCM_5_6-B.jpg': 9.0,
     'BKHCM_5_6-C.jpg': 7.0,
     'BKHCM_5_6-D.jpg': 7.0,
     'BKHCM_5_7-A.jpg': 6.0,
     'BKHCM_5_7-B.jpg': 8.0,
     'BKHCM_5_7-C.jpg': 9.0,
     'BKHCM_5_7-D.jpg': 5.0,
     'BKHCM_5_8-A.jpg': 9.0,
     'BKHCM_5_8-B.jpg': 9.0,
     'BKHCM_5_8-C.jpg': 10.0,
     'BKHCM_5_8-D.jpg': 4.0,
     'BKHCM_5_9-A.jpg': 9.0,
     'BKHCM_5_9-B.jpg': 7.0,
     'BKHCM_5_9-C.jpg': 6.0,
     'BKHCM_5_9-D.jpg': 3.0,
     'BKHN_1_0-A.jpg': 9.0,
     'BKHN_1_0-B.jpg': 7.0,
     'BKHN_1_0-C.jpg': 9.0,
     'BKHN_1_0-D.jpg': 10.0,
     'BKHN_1_1-A.jpg': 7.0,
     'BKHN_1_1-B.jpg': 9.0,
     'BKHN_1_1-C.jpg': 7.0,
     'BKHN_1_1-D.jpg': 10.0,
     'BKHN_1_10-A.jpg': 9.0,
     'BKHN_1_10-B.jpg': 8.0,
     'BKHN_1_10-C.jpg': 9.0,
     'BKHN_1_10-D.jpg': 0.0,
     'BKHN_1_11-A.jpg': 9.0,
     'BKHN_1_11-B.jpg': 6.0,
     'BKHN_1_11-C.jpg': 9.0,
     'BKHN_1_11-D.jpg': 10.0,
     'BKHN_1_2-A.jpg': 9.0,
     'BKHN_1_2-B.jpg': 9.0,
     'BKHN_1_2-C.jpg': 6.0,
     'BKHN_1_2-D.jpg': 10.0,
     'BKHN_1_3-A.jpg': 6.0,
     'BKHN_1_3-B.jpg': 6.0,
     'BKHN_1_3-C.jpg': 6.0,
     'BKHN_1_3-D.jpg': 10.0,
     'BKHN_1_4-A.jpg': 9.0,
     'BKHN_1_4-B.jpg': 9.0,
     'BKHN_1_4-C.jpg': 9.0,
     'BKHN_1_4-D.jpg': 10.0,
     'BKHN_1_5-A.jpg': 7.0,
     'BKHN_1_5-B.jpg': 7.0,
     'BKHN_1_5-C.jpg': 7.0,
     'BKHN_1_5-D.jpg': 10.0,
     'BKHN_1_6-A.jpg': 8.0,
     'BKHN_1_6-B.jpg': 9.0,
     'BKHN_1_6-C.jpg': 7.0,
     'BKHN_1_6-D.jpg': 10.0,
     'BKHN_1_7-A.jpg': 8.0,
     'BKHN_1_7-B.jpg': 8.0,
     'BKHN_1_7-C.jpg': 9.0,
     'BKHN_1_7-D.jpg': 10.0,
     'BKHN_1_8-A.jpg': 6.0,
     'BKHN_1_8-B.jpg': 6.0,
     'BKHN_1_8-C.jpg': 9.0,
     'BKHN_1_8-D.jpg': 10.0,
     'BKHN_1_9-A.jpg': 9.0,
     'BKHN_1_9-B.jpg': 9.0,
     'BKHN_1_9-C.jpg': 8.0,
     'BKHN_1_9-D.jpg': 0.0,
     'BKHN_2_0-A.jpg': 9.0,
     'BKHN_2_0-B.jpg': 5.0,
     'BKHN_2_0-C.jpg': 3.0,
     'BKHN_2_0-D.jpg': 2.0,
     'BKHN_2_1-A.jpg': 7.0,
     'BKHN_2_1-B.jpg': 9.0,
     'BKHN_2_1-C.jpg': 3.0,
     'BKHN_2_1-D.jpg': 0.0,
     'BKHN_2_10-A.jpg': 9.0,
     'BKHN_2_10-B.jpg': 9.0,
     'BKHN_2_10-C.jpg': 3.0,
     'BKHN_2_10-D.jpg': 0.0,
     'BKHN_2_11-A.jpg': 7.0,
     'BKHN_2_11-B.jpg': 8.0,
     'BKHN_2_11-C.jpg': 1.0,
     'BKHN_2_11-D.jpg': 2.0,
     'BKHN_2_2-A.jpg': 9.0,
     'BKHN_2_2-B.jpg': 7.0,
     'BKHN_2_2-C.jpg': 1.0,
     'BKHN_2_2-D.jpg': 0.0,
     'BKHN_2_3-A.jpg': 5.0,
     'BKHN_2_3-B.jpg': 7.0,
     'BKHN_2_3-C.jpg': 1.0,
     'BKHN_2_3-D.jpg': 2.0,
     'BKHN_2_4-A.jpg': 7.0,
     'BKHN_2_4-B.jpg': 6.0,
     'BKHN_2_4-C.jpg': 3.0,
     'BKHN_2_4-D.jpg': 2.0,
     'BKHN_2_5-A.jpg': 6.0,
     'BKHN_2_5-B.jpg': 8.0,
     'BKHN_2_5-C.jpg': 3.0,
     'BKHN_2_5-D.jpg': 2.0,
     'BKHN_2_6-A.jpg': 9.0,
     'BKHN_2_6-B.jpg': 8.0,
     'BKHN_2_6-C.jpg': 1.0,
     'BKHN_2_6-D.jpg': 1.0,
     'BKHN_2_7-A.jpg': 7.0,
     'BKHN_2_7-B.jpg': 4.0,
     'BKHN_2_7-C.jpg': 1.0,
     'BKHN_2_7-D.jpg': 1.0,
     'BKHN_2_8-A.jpg': 9.0,
     'BKHN_2_8-B.jpg': 2.0,
     'BKHN_2_8-C.jpg': 3.0,
     'BKHN_2_8-D.jpg': 2.0,
     'BKHN_2_9-A.jpg': 6.0,
     'BKHN_2_9-B.jpg': 2.0,
     'BKHN_2_9-C.jpg': 1.0,
     'BKHN_2_9-D.jpg': 1.0,
     'BKHN_3_0-A.jpg': 6.0,
     'BKHN_3_0-B.jpg': 7.0,
     'BKHN_3_0-C.jpg': 4.0,
     'BKHN_3_0-D.jpg': 10.0,
     'BKHN_3_1-A.jpg': 7.0,
     'BKHN_3_1-B.jpg': 1.0,
     'BKHN_3_1-C.jpg': 4.0,
     'BKHN_3_1-D.jpg': 0.0,
     'BKHN_3_10-A.jpg': 1.0,
     'BKHN_3_10-B.jpg': 5.0,
     'BKHN_3_10-C.jpg': 2.0,
     'BKHN_3_10-D.jpg': 10.0,
     'BKHN_3_11-A.jpg': 6.0,
     'BKHN_3_11-B.jpg': 5.0,
     'BKHN_3_11-C.jpg': 3.0,
     'BKHN_3_11-D.jpg': 0.0,
     'BKHN_3_2-A.jpg': 6.0,
     'BKHN_3_2-B.jpg': 6.0,
     'BKHN_3_2-C.jpg': 4.0,
     'BKHN_3_2-D.jpg': 0.0,
     'BKHN_3_3-A.jpg': 1.0,
     'BKHN_3_3-B.jpg': 6.0,
     'BKHN_3_3-C.jpg': 7.0,
     'BKHN_3_3-D.jpg': 10.0,
     'BKHN_3_4-A.jpg': 6.0,
     'BKHN_3_4-B.jpg': 9.0,
     'BKHN_3_4-C.jpg': 6.0,
     'BKHN_3_4-D.jpg': 5.0,
     'BKHN_3_5-A.jpg': 9.0,
     'BKHN_3_5-B.jpg': 1.0,
     'BKHN_3_5-C.jpg': 9.0,
     'BKHN_3_5-D.jpg': 5.0,
     'BKHN_3_6-A.jpg': 6.0,
     'BKHN_3_6-B.jpg': 6.0,
     'BKHN_3_6-C.jpg': 7.0,
     'BKHN_3_6-D.jpg': 10.0,
     'BKHN_3_7-A.jpg': 1.0,
     'BKHN_3_7-B.jpg': 7.0,
     'BKHN_3_7-C.jpg': 6.0,
     'BKHN_3_7-D.jpg': 0.0,
     'BKHN_3_8-A.jpg': 6.0,
     'BKHN_3_8-B.jpg': 7.0,
     'BKHN_3_8-C.jpg': 5.0,
     'BKHN_3_8-D.jpg': 5.0,
     'BKHN_3_9-A.jpg': 7.0,
     'BKHN_3_9-B.jpg': 6.0,
     'BKHN_3_9-C.jpg': 2.0,
     'BKHN_3_9-D.jpg': 5.0,
     'BKHN_4_0-A.jpg': 6.0,
     'BKHN_4_0-B.jpg': 7.0,
     'BKHN_4_0-C.jpg': 4.0,
     'BKHN_4_0-D.jpg': 3.0,
     'BKHN_4_1-A.jpg': 7.0,
     'BKHN_4_1-B.jpg': 6.0,
     'BKHN_4_1-C.jpg': 4.0,
     'BKHN_4_1-D.jpg': 5.0,
     'BKHN_4_10-A.jpg': 3.0,
     'BKHN_4_10-B.jpg': 7.0,
     'BKHN_4_10-C.jpg': 2.0,
     'BKHN_4_10-D.jpg': 6.0,
     'BKHN_4_11-A.jpg': 7.0,
     'BKHN_4_11-B.jpg': 5.0,
     'BKHN_4_11-C.jpg': 1.0,
     'BKHN_4_11-D.jpg': 3.0,
     'BKHN_4_2-A.jpg': 6.0,
     'BKHN_4_2-B.jpg': 9.0,
     'BKHN_4_2-C.jpg': 3.0,
     'BKHN_4_2-D.jpg': 5.0,
     'BKHN_4_3-A.jpg': 3.0,
     'BKHN_4_3-B.jpg': 3.0,
     'BKHN_4_3-C.jpg': 4.0,
     'BKHN_4_3-D.jpg': 4.0,
     'BKHN_4_4-A.jpg': 6.0,
     'BKHN_4_4-B.jpg': 9.0,
     'BKHN_4_4-C.jpg': 4.0,
     'BKHN_4_4-D.jpg': 6.0,
     'BKHN_4_5-A.jpg': 9.0,
     'BKHN_4_5-B.jpg': 7.0,
     'BKHN_4_5-C.jpg': 2.0,
     'BKHN_4_5-D.jpg': 5.0,
     'BKHN_4_6-A.jpg': 7.0,
     'BKHN_4_6-B.jpg': 6.0,
     'BKHN_4_6-C.jpg': 2.0,
     'BKHN_4_6-D.jpg': 3.0,
     'BKHN_4_7-A.jpg': 6.0,
     'BKHN_4_7-B.jpg': 6.0,
     'BKHN_4_7-C.jpg': 4.0,
     'BKHN_4_7-D.jpg': 2.0,
     'BKHN_4_8-A.jpg': 9.0,
     'BKHN_4_8-B.jpg': 9.0,
     'BKHN_4_8-C.jpg': 3.0,
     'BKHN_4_8-D.jpg': 6.0,
     'BKHN_4_9-A.jpg': 9.0,
     'BKHN_4_9-B.jpg': 6.0,
     'BKHN_4_9-C.jpg': 4.0,
     'BKHN_4_9-D.jpg': 3.0,
     'BKHN_5_0-A.jpg': 9.0,
     'BKHN_5_0-B.jpg': 7.0,
     'BKHN_5_0-C.jpg': 7.0,
     'BKHN_5_0-D.jpg': 8.0,
     'BKHN_5_1-A.jpg': 9.0,
     'BKHN_5_1-B.jpg': 9.0,
     'BKHN_5_1-C.jpg': 6.0,
     'BKHN_5_1-D.jpg': 8.0,
     'BKHN_5_10-A.jpg': 9.0,
     'BKHN_5_10-B.jpg': 9.0,
     'BKHN_5_10-C.jpg': 6.0,
     'BKHN_5_10-D.jpg': 10.0,
     'BKHN_5_11-A.jpg': 6.0,
     'BKHN_5_11-B.jpg': 9.0,
     'BKHN_5_11-C.jpg': 7.0,
     'BKHN_5_11-D.jpg': 10.0,
     'BKHN_5_2-A.jpg': 7.0,
     'BKHN_5_2-B.jpg': 6.0,
     'BKHN_5_2-C.jpg': 9.0,
     'BKHN_5_2-D.jpg': 8.0,
     'BKHN_5_3-A.jpg': 6.0,
     'BKHN_5_3-B.jpg': 9.0,
     'BKHN_5_3-C.jpg': 7.0,
     'BKHN_5_3-D.jpg': 10.0,
     'BKHN_5_4-A.jpg': 9.0,
     'BKHN_5_4-B.jpg': 7.0,
     'BKHN_5_4-C.jpg': 6.0,
     'BKHN_5_4-D.jpg': 10.0,
     'BKHN_5_5-A.jpg': 6.0,
     'BKHN_5_5-B.jpg': 7.0,
     'BKHN_5_5-C.jpg': 9.0,
     'BKHN_5_5-D.jpg': 8.0,
     'BKHN_5_6-A.jpg': 9.0,
     'BKHN_5_6-B.jpg': 6.0,
     'BKHN_5_6-C.jpg': 7.0,
     'BKHN_5_6-D.jpg': 10.0,
     'BKHN_5_7-A.jpg': 6.0,
     'BKHN_5_7-B.jpg': 6.0,
     'BKHN_5_7-C.jpg': 6.0,
     'BKHN_5_7-D.jpg': 10.0,
     'BKHN_5_8-A.jpg': 9.0,
     'BKHN_5_8-B.jpg': 9.0,
     'BKHN_5_8-C.jpg': 9.0,
     'BKHN_5_8-D.jpg': 8.0,
     'BKHN_5_9-A.jpg': 7.0,
     'BKHN_5_9-B.jpg': 7.0,
     'BKHN_5_9-C.jpg': 7.0,
     'BKHN_5_9-D.jpg': 8.0,
     'BKHN_6_0-A.jpg': 9.0,
     'BKHN_6_0-B.jpg': 3.0,
     'BKHN_6_0-C.jpg': 7.0,
     'BKHN_6_0-D.jpg': 7.0,
     'BKHN_6_1-A.jpg': 9.0,
     'BKHN_6_1-B.jpg': 5.0,
     'BKHN_6_1-C.jpg': 3.0,
     'BKHN_6_1-D.jpg': 6.0,
     'BKHN_6_10-A.jpg': 9.0,
     'BKHN_6_10-B.jpg': 6.0,
     'BKHN_6_10-C.jpg': 9.0,
     'BKHN_6_10-D.jpg': 5.0,
     'BKHN_6_11-A.jpg': 8.0,
     'BKHN_6_11-B.jpg': 6.0,
     'BKHN_6_11-C.jpg': 7.0,
     'BKHN_6_11-D.jpg': 9.0,
     'BKHN_6_2-A.jpg': 6.0,
     'BKHN_6_2-B.jpg': 3.0,
     'BKHN_6_2-C.jpg': 7.0,
     'BKHN_6_2-D.jpg': 7.0,
     'BKHN_6_3-A.jpg': 5.0,
     'BKHN_6_3-B.jpg': 2.0,
     'BKHN_6_3-C.jpg': 6.0,
     'BKHN_6_3-D.jpg': 9.0,
     'BKHN_6_4-A.jpg': 4.0,
     'BKHN_6_4-B.jpg': 5.0,
     'BKHN_6_4-C.jpg': 7.0,
     'BKHN_6_4-D.jpg': 8.0,
     'BKHN_6_5-A.jpg': 9.0,
     'BKHN_6_5-B.jpg': 1.0,
     'BKHN_6_5-C.jpg': 9.0,
     'BKHN_6_5-D.jpg': 8.0,
     'BKHN_6_6-A.jpg': 5.0,
     'BKHN_6_6-B.jpg': 5.0,
     'BKHN_6_6-C.jpg': 8.0,
     'BKHN_6_6-D.jpg': 9.0,
     'BKHN_6_7-A.jpg': 6.0,
     'BKHN_6_7-B.jpg': 3.0,
     'BKHN_6_7-C.jpg': 7.0,
     'BKHN_6_7-D.jpg': 7.0,
     'BKHN_6_8-A.jpg': 9.0,
     'BKHN_6_8-B.jpg': 5.0,
     'BKHN_6_8-C.jpg': 5.0,
     'BKHN_6_8-D.jpg': 3.0,
     'BKHN_6_9-A.jpg': 7.0,
     'BKHN_6_9-B.jpg': 1.0,
     'BKHN_6_9-C.jpg': 6.0,
     'BKHN_6_9-D.jpg': 6.0}




```python
len(result) #Kết quả cần là 768 vì dataset của ta có 768 hình ảnh 
```




    768



### Bài 30 - Dự đoán trên dataset mới

Bây giờ, ta có một dataset mới lấy từ 7 hình ảnh là các phiếu chấm của trường KTDN nằm trong thư mục "RawFormTest", không liên quan đến tập dữ liệu đã train từ "RawForm".

Ta sẽ tổng hợp toàn bộ quá trình đã làm từ TD trước đến giờ để hoàn thành hàm **`readEvaluationFormAsDict(image_folder, model_file)`**, đọc tất cả các hình ảnh trong thư mục `image_folder` (chứa các hình ảnh mới như "RawFormTest") và các tham số trong file `model_file` (có dạng như "SavedModels/LogisticRegression_OVR.txt" mà bạn đã định ra cấu trúc), rồi dự đoán kết quả dưới dạng một từ điển có dạng:

{'KTDN_1_0-A.jpg': 9.0, ...}

Nghĩa là trong hình ảnh "KTDN_1.jpg", dòng thứ nhất (chỉ số 0), cột A (hoàn cảnh) tương ứng với điểm số là 7.

*Hãy viết hàm **`readEvaluationFormAsDict(image_folder, model_file)`** thực hiện chức năng trên. Bạn cần tận dụng tất cả các hàm đã viết ở TD trước và TD này.**

Đoạn code dưới đây giúp test hàm của bạn.


```python
RAW_DATA = "RawFormTest/"
SCORE_DATA = "ScoreDataTest/"
LOGISTIC_REGRESSION_OVR_PARAMS = "SavedModels/LogisticRegression_OVR.txt"
result = readEvaluationFormAsDict(RAW_DATA, LOGISTIC_REGRESSION_OVR_PARAMS)
result
```




    {'KTDN_1_0-A.jpg': 9.0,
     'KTDN_1_0-B.jpg': 5.0,
     'KTDN_1_0-C.jpg': 4.0,
     'KTDN_1_0-D.jpg': 3.0,
     'KTDN_1_1-A.jpg': 9.0,
     'KTDN_1_1-B.jpg': 5.0,
     'KTDN_1_1-C.jpg': 4.0,
     'KTDN_1_1-D.jpg': 2.0,
     'KTDN_1_10-A.jpg': 9.0,
     'KTDN_1_10-B.jpg': 7.0,
     'KTDN_1_10-C.jpg': 1.0,
     'KTDN_1_10-D.jpg': 4.0,
     'KTDN_1_11-A.jpg': 6.0,
     'KTDN_1_11-B.jpg': 6.0,
     'KTDN_1_11-C.jpg': 2.0,
     'KTDN_1_11-D.jpg': 3.0,
     'KTDN_1_12-A.jpg': 5.0,
     'KTDN_1_12-B.jpg': 4.0,
     'KTDN_1_12-C.jpg': 1.0,
     'KTDN_1_12-D.jpg': 3.0,
     'KTDN_1_2-A.jpg': 9.0,
     'KTDN_1_2-B.jpg': 3.0,
     'KTDN_1_2-C.jpg': 4.0,
     'KTDN_1_2-D.jpg': 2.0,
     'KTDN_1_3-A.jpg': 7.0,
     'KTDN_1_3-B.jpg': 3.0,
     'KTDN_1_3-C.jpg': 2.0,
     'KTDN_1_3-D.jpg': 7.0,
     'KTDN_1_4-A.jpg': 9.0,
     'KTDN_1_4-B.jpg': 3.0,
     'KTDN_1_4-C.jpg': 2.0,
     'KTDN_1_4-D.jpg': 6.0,
     'KTDN_1_5-A.jpg': 7.0,
     'KTDN_1_5-B.jpg': 3.0,
     'KTDN_1_5-C.jpg': 2.0,
     'KTDN_1_5-D.jpg': 7.0,
     'KTDN_1_6-A.jpg': 9.0,
     'KTDN_1_6-B.jpg': 9.0,
     'KTDN_1_6-C.jpg': 1.0,
     'KTDN_1_6-D.jpg': 6.0,
     'KTDN_1_7-A.jpg': 8.0,
     'KTDN_1_7-B.jpg': 3.0,
     'KTDN_1_7-C.jpg': 1.0,
     'KTDN_1_7-D.jpg': 9.0,
     'KTDN_1_8-A.jpg': 9.0,
     'KTDN_1_8-B.jpg': 2.0,
     'KTDN_1_8-C.jpg': 1.0,
     'KTDN_1_8-D.jpg': 5.0,
     'KTDN_1_9-A.jpg': 6.0,
     'KTDN_1_9-B.jpg': 2.0,
     'KTDN_1_9-C.jpg': 0.0,
     'KTDN_1_9-D.jpg': 5.0,
     'KTDN_2_0-A.jpg': 9.0,
     'KTDN_2_0-B.jpg': 4.0,
     'KTDN_2_0-C.jpg': 7.0,
     'KTDN_2_0-D.jpg': 8.0,
     'KTDN_2_1-A.jpg': 9.0,
     'KTDN_2_1-B.jpg': 10.0,
     'KTDN_2_1-C.jpg': 7.0,
     'KTDN_2_1-D.jpg': 4.0,
     'KTDN_2_10-A.jpg': 10.0,
     'KTDN_2_10-B.jpg': 6.0,
     'KTDN_2_10-C.jpg': 7.0,
     'KTDN_2_10-D.jpg': 6.0,
     'KTDN_2_11-A.jpg': 10.0,
     'KTDN_2_11-B.jpg': 4.0,
     'KTDN_2_11-C.jpg': 7.0,
     'KTDN_2_11-D.jpg': 5.0,
     'KTDN_2_12-A.jpg': 0.0,
     'KTDN_2_12-B.jpg': 6.0,
     'KTDN_2_12-C.jpg': 7.0,
     'KTDN_2_12-D.jpg': 6.0,
     'KTDN_2_13-A.jpg': 5.0,
     'KTDN_2_13-B.jpg': 5.0,
     'KTDN_2_13-C.jpg': 7.0,
     'KTDN_2_13-D.jpg': 5.0,
     'KTDN_2_2-A.jpg': 1.0,
     'KTDN_2_2-B.jpg': 7.0,
     'KTDN_2_2-C.jpg': 7.0,
     'KTDN_2_2-D.jpg': 5.0,
     'KTDN_2_3-A.jpg': 9.0,
     'KTDN_2_3-B.jpg': 9.0,
     'KTDN_2_3-C.jpg': 7.0,
     'KTDN_2_3-D.jpg': 5.0,
     'KTDN_2_4-A.jpg': 3.0,
     'KTDN_2_4-B.jpg': 7.0,
     'KTDN_2_4-C.jpg': 7.0,
     'KTDN_2_4-D.jpg': 3.0,
     'KTDN_2_5-A.jpg': 9.0,
     'KTDN_2_5-B.jpg': 7.0,
     'KTDN_2_5-C.jpg': 7.0,
     'KTDN_2_5-D.jpg': 9.0,
     'KTDN_2_6-A.jpg': 8.0,
     'KTDN_2_6-B.jpg': 6.0,
     'KTDN_2_6-C.jpg': 7.0,
     'KTDN_2_6-D.jpg': 8.0,
     'KTDN_2_7-A.jpg': 6.0,
     'KTDN_2_7-B.jpg': 4.0,
     'KTDN_2_7-C.jpg': 7.0,
     'KTDN_2_7-D.jpg': 8.0,
     'KTDN_2_8-A.jpg': 9.0,
     'KTDN_2_8-B.jpg': 6.0,
     'KTDN_2_8-C.jpg': 7.0,
     'KTDN_2_8-D.jpg': 9.0,
     'KTDN_2_9-A.jpg': 6.0,
     'KTDN_2_9-B.jpg': 5.0,
     'KTDN_2_9-C.jpg': 7.0,
     'KTDN_2_9-D.jpg': 9.0,
     'KTDN_3_0-A.jpg': 7.0,
     'KTDN_3_0-B.jpg': 2.0,
     'KTDN_3_0-C.jpg': 7.0,
     'KTDN_3_0-D.jpg': 7.0,
     'KTDN_3_1-A.jpg': 6.0,
     'KTDN_3_1-B.jpg': 7.0,
     'KTDN_3_1-C.jpg': 6.0,
     'KTDN_3_1-D.jpg': 4.0,
     'KTDN_3_10-A.jpg': 9.0,
     'KTDN_3_10-B.jpg': 9.0,
     'KTDN_3_10-C.jpg': 3.0,
     'KTDN_3_10-D.jpg': 7.0,
     'KTDN_3_11-A.jpg': 7.0,
     'KTDN_3_11-B.jpg': 6.0,
     'KTDN_3_11-C.jpg': 6.0,
     'KTDN_3_11-D.jpg': 2.0,
     'KTDN_3_12-A.jpg': 5.0,
     'KTDN_3_12-B.jpg': 9.0,
     'KTDN_3_12-C.jpg': 5.0,
     'KTDN_3_12-D.jpg': 9.0,
     'KTDN_3_2-A.jpg': 9.0,
     'KTDN_3_2-B.jpg': 7.0,
     'KTDN_3_2-C.jpg': 9.0,
     'KTDN_3_2-D.jpg': 2.0,
     'KTDN_3_3-A.jpg': 9.0,
     'KTDN_3_3-B.jpg': 3.0,
     'KTDN_3_3-C.jpg': 8.0,
     'KTDN_3_3-D.jpg': 7.0,
     'KTDN_3_4-A.jpg': 2.0,
     'KTDN_3_4-B.jpg': 7.0,
     'KTDN_3_4-C.jpg': 6.0,
     'KTDN_3_4-D.jpg': 2.0,
     'KTDN_3_5-A.jpg': 2.0,
     'KTDN_3_5-B.jpg': 2.0,
     'KTDN_3_5-C.jpg': 5.0,
     'KTDN_3_5-D.jpg': 7.0,
     'KTDN_3_6-A.jpg': 6.0,
     'KTDN_3_6-B.jpg': 7.0,
     'KTDN_3_6-C.jpg': 4.0,
     'KTDN_3_6-D.jpg': 2.0,
     'KTDN_3_7-A.jpg': 9.0,
     'KTDN_3_7-B.jpg': 8.0,
     'KTDN_3_7-C.jpg': 4.0,
     'KTDN_3_7-D.jpg': 7.0,
     'KTDN_3_8-A.jpg': 8.0,
     'KTDN_3_8-B.jpg': 7.0,
     'KTDN_3_8-C.jpg': 3.0,
     'KTDN_3_8-D.jpg': 2.0,
     'KTDN_3_9-A.jpg': 6.0,
     'KTDN_3_9-B.jpg': 6.0,
     'KTDN_3_9-C.jpg': 3.0,
     'KTDN_3_9-D.jpg': 0.0,
     'KTDN_4_0-A.jpg': 7.0,
     'KTDN_4_0-B.jpg': 6.0,
     'KTDN_4_0-C.jpg': 3.0,
     'KTDN_4_0-D.jpg': 6.0,
     'KTDN_4_1-A.jpg': 6.0,
     'KTDN_4_1-B.jpg': 3.0,
     'KTDN_4_1-C.jpg': 2.0,
     'KTDN_4_1-D.jpg': 7.0,
     'KTDN_4_10-A.jpg': 2.0,
     'KTDN_4_10-B.jpg': 6.0,
     'KTDN_4_10-C.jpg': 2.0,
     'KTDN_4_10-D.jpg': 0.0,
     'KTDN_4_11-A.jpg': 1.0,
     'KTDN_4_11-B.jpg': 9.0,
     'KTDN_4_11-C.jpg': 1.0,
     'KTDN_4_11-D.jpg': 5.0,
     'KTDN_4_12-A.jpg': 4.0,
     'KTDN_4_12-B.jpg': 9.0,
     'KTDN_4_12-C.jpg': 0.0,
     'KTDN_4_12-D.jpg': 9.0,
     'KTDN_4_2-A.jpg': 9.0,
     'KTDN_4_2-B.jpg': 9.0,
     'KTDN_4_2-C.jpg': 5.0,
     'KTDN_4_2-D.jpg': 6.0,
     'KTDN_4_3-A.jpg': 10.0,
     'KTDN_4_3-B.jpg': 7.0,
     'KTDN_4_3-C.jpg': 5.0,
     'KTDN_4_3-D.jpg': 9.0,
     'KTDN_4_4-A.jpg': 5.0,
     'KTDN_4_4-B.jpg': 8.0,
     'KTDN_4_4-C.jpg': 7.0,
     'KTDN_4_4-D.jpg': 6.0,
     'KTDN_4_5-A.jpg': 4.0,
     'KTDN_4_5-B.jpg': 9.0,
     'KTDN_4_5-C.jpg': 0.0,
     'KTDN_4_5-D.jpg': 5.0,
     'KTDN_4_6-A.jpg': 3.0,
     'KTDN_4_6-B.jpg': 7.0,
     'KTDN_4_6-C.jpg': 5.0,
     'KTDN_4_6-D.jpg': 6.0,
     'KTDN_4_7-A.jpg': 2.0,
     'KTDN_4_7-B.jpg': 0.0,
     'KTDN_4_7-C.jpg': 2.0,
     'KTDN_4_7-D.jpg': 5.0,
     'KTDN_4_8-A.jpg': 6.0,
     'KTDN_4_8-B.jpg': 9.0,
     'KTDN_4_8-C.jpg': 0.0,
     'KTDN_4_8-D.jpg': 6.0,
     'KTDN_4_9-A.jpg': 5.0,
     'KTDN_4_9-B.jpg': 7.0,
     'KTDN_4_9-C.jpg': 2.0,
     'KTDN_4_9-D.jpg': 3.0,
     'KTDN_5_0-A.jpg': 7.0,
     'KTDN_5_0-B.jpg': 5.0,
     'KTDN_5_0-C.jpg': 3.0,
     'KTDN_5_0-D.jpg': 6.0,
     'KTDN_5_1-A.jpg': 7.0,
     'KTDN_5_1-B.jpg': 5.0,
     'KTDN_5_1-C.jpg': 2.0,
     'KTDN_5_1-D.jpg': 5.0,
     'KTDN_5_10-A.jpg': 7.0,
     'KTDN_5_10-B.jpg': 4.0,
     'KTDN_5_10-C.jpg': 2.0,
     'KTDN_5_10-D.jpg': 5.0,
     'KTDN_5_11-A.jpg': 6.0,
     'KTDN_5_11-B.jpg': 6.0,
     'KTDN_5_11-C.jpg': 3.0,
     'KTDN_5_11-D.jpg': 6.0,
     'KTDN_5_12-A.jpg': 6.0,
     'KTDN_5_12-B.jpg': 5.0,
     'KTDN_5_12-C.jpg': 2.0,
     'KTDN_5_12-D.jpg': 4.0,
     'KTDN_5_2-A.jpg': 6.0,
     'KTDN_5_2-B.jpg': 4.0,
     'KTDN_5_2-C.jpg': 3.0,
     'KTDN_5_2-D.jpg': 6.0,
     'KTDN_5_3-A.jpg': 8.0,
     'KTDN_5_3-B.jpg': 5.0,
     'KTDN_5_3-C.jpg': 2.0,
     'KTDN_5_3-D.jpg': 5.0,
     'KTDN_5_4-A.jpg': 6.0,
     'KTDN_5_4-B.jpg': 4.0,
     'KTDN_5_4-C.jpg': 3.0,
     'KTDN_5_4-D.jpg': 6.0,
     'KTDN_5_5-A.jpg': 9.0,
     'KTDN_5_5-B.jpg': 5.0,
     'KTDN_5_5-C.jpg': 2.0,
     'KTDN_5_5-D.jpg': 5.0,
     'KTDN_5_6-A.jpg': 10.0,
     'KTDN_5_6-B.jpg': 4.0,
     'KTDN_5_6-C.jpg': 2.0,
     'KTDN_5_6-D.jpg': 6.0,
     'KTDN_5_7-A.jpg': 6.0,
     'KTDN_5_7-B.jpg': 5.0,
     'KTDN_5_7-C.jpg': 3.0,
     'KTDN_5_7-D.jpg': 6.0,
     'KTDN_5_8-A.jpg': 9.0,
     'KTDN_5_8-B.jpg': 7.0,
     'KTDN_5_8-C.jpg': 2.0,
     'KTDN_5_8-D.jpg': 2.0,
     'KTDN_5_9-A.jpg': 8.0,
     'KTDN_5_9-B.jpg': 3.0,
     'KTDN_5_9-C.jpg': 3.0,
     'KTDN_5_9-D.jpg': 6.0,
     'KTDN_6_0-A.jpg': 2.0,
     'KTDN_6_0-B.jpg': 6.0,
     'KTDN_6_0-C.jpg': 7.0,
     'KTDN_6_0-D.jpg': 5.0,
     'KTDN_6_1-A.jpg': 7.0,
     'KTDN_6_1-B.jpg': 7.0,
     'KTDN_6_1-C.jpg': 6.0,
     'KTDN_6_1-D.jpg': 5.0,
     'KTDN_6_10-A.jpg': 9.0,
     'KTDN_6_10-B.jpg': 8.0,
     'KTDN_6_10-C.jpg': 7.0,
     'KTDN_6_10-D.jpg': 5.0,
     'KTDN_6_11-A.jpg': 5.0,
     'KTDN_6_11-B.jpg': 6.0,
     'KTDN_6_11-C.jpg': 6.0,
     'KTDN_6_11-D.jpg': 6.0,
     'KTDN_6_2-A.jpg': 6.0,
     'KTDN_6_2-B.jpg': 6.0,
     'KTDN_6_2-C.jpg': 6.0,
     'KTDN_6_2-D.jpg': 10.0,
     'KTDN_6_3-A.jpg': 7.0,
     'KTDN_6_3-B.jpg': 6.0,
     'KTDN_6_3-C.jpg': 5.0,
     'KTDN_6_3-D.jpg': 5.0,
     'KTDN_6_4-A.jpg': 10.0,
     'KTDN_6_4-B.jpg': 5.0,
     'KTDN_6_4-C.jpg': 7.0,
     'KTDN_6_4-D.jpg': 10.0,
     'KTDN_6_5-A.jpg': 9.0,
     'KTDN_6_5-B.jpg': 6.0,
     'KTDN_6_5-C.jpg': 6.0,
     'KTDN_6_5-D.jpg': 5.0,
     'KTDN_6_6-A.jpg': 7.0,
     'KTDN_6_6-B.jpg': 6.0,
     'KTDN_6_6-C.jpg': 8.0,
     'KTDN_6_6-D.jpg': 5.0,
     'KTDN_6_7-A.jpg': 5.0,
     'KTDN_6_7-B.jpg': 5.0,
     'KTDN_6_7-C.jpg': 6.0,
     'KTDN_6_7-D.jpg': 5.0,
     'KTDN_6_8-A.jpg': 9.0,
     'KTDN_6_8-B.jpg': 8.0,
     'KTDN_6_8-C.jpg': 7.0,
     'KTDN_6_8-D.jpg': 5.0,
     'KTDN_6_9-A.jpg': 7.0,
     'KTDN_6_9-B.jpg': 6.0,
     'KTDN_6_9-C.jpg': 6.0,
     'KTDN_6_9-D.jpg': 5.0,
     'KTDN_7_0-A.jpg': 6.0,
     'KTDN_7_0-B.jpg': 4.0,
     'KTDN_7_0-C.jpg': 9.0,
     'KTDN_7_0-D.jpg': 9.0,
     'KTDN_7_1-A.jpg': 7.0,
     'KTDN_7_1-B.jpg': 5.0,
     'KTDN_7_1-C.jpg': 9.0,
     'KTDN_7_1-D.jpg': 0.0,
     'KTDN_7_10-A.jpg': 6.0,
     'KTDN_7_10-B.jpg': 2.0,
     'KTDN_7_10-C.jpg': 2.0,
     'KTDN_7_10-D.jpg': 7.0,
     'KTDN_7_11-A.jpg': 8.0,
     'KTDN_7_11-B.jpg': 9.0,
     'KTDN_7_11-C.jpg': 9.0,
     'KTDN_7_11-D.jpg': 0.0,
     'KTDN_7_2-A.jpg': 6.0,
     'KTDN_7_2-B.jpg': 4.0,
     'KTDN_7_2-C.jpg': 9.0,
     'KTDN_7_2-D.jpg': 0.0,
     'KTDN_7_3-A.jpg': 7.0,
     'KTDN_7_3-B.jpg': 3.0,
     'KTDN_7_3-C.jpg': 9.0,
     'KTDN_7_3-D.jpg': 9.0,
     'KTDN_7_4-A.jpg': 6.0,
     'KTDN_7_4-B.jpg': 3.0,
     'KTDN_7_4-C.jpg': 9.0,
     'KTDN_7_4-D.jpg': 7.0,
     'KTDN_7_5-A.jpg': 2.0,
     'KTDN_7_5-B.jpg': 2.0,
     'KTDN_7_5-C.jpg': 10.0,
     'KTDN_7_5-D.jpg': 0.0,
     'KTDN_7_6-A.jpg': 6.0,
     'KTDN_7_6-B.jpg': 9.0,
     'KTDN_7_6-C.jpg': 9.0,
     'KTDN_7_6-D.jpg': 1.0,
     'KTDN_7_7-A.jpg': 1.0,
     'KTDN_7_7-B.jpg': 4.0,
     'KTDN_7_7-C.jpg': 7.0,
     'KTDN_7_7-D.jpg': 7.0,
     'KTDN_7_8-A.jpg': 6.0,
     'KTDN_7_8-B.jpg': 3.0,
     'KTDN_7_8-C.jpg': 9.0,
     'KTDN_7_8-D.jpg': 7.0,
     'KTDN_7_9-A.jpg': 7.0,
     'KTDN_7_9-B.jpg': 5.0,
     'KTDN_7_9-C.jpg': 9.0,
     'KTDN_7_9-D.jpg': 0.0}



## Phần 7 - Xử lí ảnh điểm thập phân - Class ComplexImage

Bây giờ ta có một phiếu chấm với điểm thập phân trong thư mục 'ComplexRawForm'.

### Bài 31 - Cắt ảnh gốc

*Sử dụng những gì đã thực hiện ở TD7, hãy cắt ảnh này thành các ô chứa điểm (số thập phân), nhưng thay vì với kích thước 28 x 28, thì dùng kích thước 100 x 100*.

(Ta muốn làm việc với các ảnh to hơn vì các số bây giờ dài hơn).

Bạn có thể lưu kết quả trong "ComplexDataScore". Kết quả cần gồm 48 hình ảnh như hình dưới đây.

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson8/TD/F1.png" width=800></img>

### Bài 32 - Class ComplexImage

Mỗi hình ảnh như `ComplexScoreData/BKHCM_3B_0-A.jpg` chứa 1 số thập phân sẽ được biểu diễn bằng một instance của class  `ComplexImage`. Class này gồm attribute:

- `img`: Một numpy array 100 x 100 biểu diễn nó: mỗi entry của array này nhận giá trị 0 hay 255 tuỳ theo pixel là đen hay trắng.
- `name`: Tên của instance, ví dụ với hình `ComplexScoreData/BKHCM_3B_0-A.jpg` thì tên là `BKHCM_3B_0-A.jpg`. Attribute này sẽ được dùng để sau này kết nối hình ảnh với dữ liệu.

*Hãy viết method **`__init__(self, source_image)`** để từ hình ảnh ở đường dẫn `source_img` xây dựng instance với 2 attributes trên.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
my_image = ComplexImage("ComplexScoreData/BKHCM_3B_0-A.jpg")
my_image.name
```




    'BKHCM_3B_0-A.jpg'




```python
my_image.img
```




    array([[255, 255, 255, ...,   0, 255, 255],
           [255, 255, 255, ...,   0, 255, 255],
           [255, 255, 255, ...,   0,   0, 255],
           ...,
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)



### Bài 33 - Các getter

*Trong class ComplexImage, viết các method **`getImage()`, `getWidth()`, `getHeight()`** trả lại array mô tả hình ảnh, chiều rộng, chiều cao của hình ảnh.*

*Viết method **`draw()`** cho phép "vẽ" hình ảnh dưới dạng dễ nhìn, ví dụ như đoạn code test dưới đây.*


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
my_image = ComplexImage("ComplexScoreData/BKHCM_3B_0-A.jpg")
my_image.getImage()
```




    array([[255, 255, 255, ...,   0, 255, 255],
           [255, 255, 255, ...,   0, 255, 255],
           [255, 255, 255, ...,   0,   0, 255],
           ...,
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)




```python
my_image.getWidth(), my_image.getHeight()
```




    (100, 100)




```python
my_image.draw()
```
```
    ----------------------------------------------------------------------------------------------------
    |                                                                                           XXXXXXX  |
    |                                                                                           XXXXXXX  |
    |                                                                                         XXXXXXXXXX |
    |                                                                                         XXXXXXXXXX |
    |                                                                                    XXXXXXXXXXXXXXXX|
    |                                                                                    XXXXXXXXXXXXXXXX|
    |                XXX                                                               XXXXXXXXXXXXXXXXXX|
    |               XXXX                                                               XXXXXXXXXXXXXXXXXX|
    |               XXXXXX                                                           XXXXXXXXXXXXXXXXXXXX|
    |              XXXXXXX                                                           XXXXXXXXXXXXXXXXXX  |
    |              XXXXXXXX                                                        XXXXXXXXXXXXXXXXXXXX  |
    |              XXXXXXXX                                                        XXXXXXXXXXXXX         |
    |         XXXXXXXXXXXXX                                                       XXXXXXXXXXXXXX         |
    |         XXXXXXXXXXXXX                                                      XXXXXXXXXXXX            |
    |        XXXXXXXXXXXXXXX                                                XXXXXXXXXXXXXXXXX            |
    |        XXXXXXXXXXXXXXX                                                XXXXXXXXXXXXXXX              |
    |        XXXXXXXXXXXXXXXX                                             XXXXXXXXXXXXXXXX               |
    |        XXXXXXX     XXXX                                             XXXXXXXXXXXXX                  |
    |        XXXXXX      XXXX                                           XXXXXXXXXXXXXXX                  |
    |        XXXXXX      XXXX                                           XXXXXXXXXXXXX                    |
    |       XXXXXXX      XXXX                                          XXXXXXXXXXXXXX                    |
    |       XXXXXXX      XXXX                                        XXXXXXXXXXXXXXX                     |
    |       XXXXXXX      XXXX                                        XXXXXXXXXXXXXXX                     |
    |       XXXXXXX      XXXX                                       XXXXXXXXX XX                         |
    |        XXXXX       XXXX                                       XXXXXXXX  X                          |
    |        XXXXX       XXXX                                      XXXXXX                                |
    |         XXX        XXXX                                      XXXXXX                                |
    |          XX        XXXX                                      XXXXX                                 |
    |          X         XXX                                       XXXXX                                 |
    |                   XXXX                                       XXXXX                                 |
    |                   XXX                                        XXXXX                                 |
    |                   XXX                                        XXXXX                                 |
    |                   XXX                                        XXXXX                                 |
    |                   XXX                                       XXXXX                                  |
    |                   XXX                                       XXXXX                                  |
    |                  XXXX                                       XXXXX                                  |
    |                  XXX                                        XXXXX                                  |
    |                  XXX                                       XXXXXX                                  |
    |                  XX                                        XXXXXX                                  |
    |                  XX                                        XXXXX                                   |
    |                                                            XXXXX                                   |
    |                                                            XXXXXXX                                 |
    |                                                             XXXXXXX                                |
    |        XXXXX                                                XXXXXXXXXXXX                           |
    |        XXXXX                                                XXXXXXXXXXXX                           |
    |       XXXXXXX                                               XXXXXXXXXXXX                           |
    |       XXXXXXXX                                              XXXXXXXXXXXXX                          |
    |      XXXXXXXXX                                              XXXXXXXXXXXXX                          |
    |      XXXXXXXXX   XXX                                        XXXXXXXXXXXXXXXX                       |
    |       XXXXXXX   XXXXX                                       XXXXXXXXXXXXXXXX                       |
    |       XXX       XXXXXX                                            XXXXXXXXXXXX                     |
    |        XX       XXXXXXX                                            XXXXXXXXXXX                     |
    |                 XXXXXXX                                               XXXXXXXXX                    |
    |                  XXXXXXX                                              XXXXXXXXX                    |
    |                  XXXXXXX                                               XXXXXXXXX                   |
    |                   XXXXXX                                                XXXXXXXX                   |
    |                   XXXXXXXX                                               XXXXXXX                   |
    |                    XXXXXXX                                                XXXXXX                   |
    |                    XXXXXXXX                                               XXXXXXX                  |
    |                     XXXXXXX                                               XXXXXXX                  |
    |                     XXXXXXX                                                XXXXXX                  |
    |                       XXXXX                                                 XXXXX                  |
    |                        XXXX                                                 XXXXX                  |
    |                        XXXX                                                  XXXX                  |
    |                        XXXX                                                  XXXXX                 |
    |                        XXXX                                                  XXXXX                 |
    |                        XXXX         XXXX                                     XXXXXX                |
    |                        XXXX         XXXX                                     XXXXXX                |
    |                        XXXX        XXXXX                                     XXXXXX                |
    | X                      XXXX       XXXXXX                                     XXXXXX                |
    |XX                      XXXX        XXXXX                                    XXXXXX                 |
    |XX                     XXXXX        XXX                    XX              XXXXXXXX                 |
    |XXX                    XXXXX        XXX                   XXX              XXXXXXX                  |
    |XXX                    XXXX         XXX                   XXX              XXXXXXX                  |
    |XXXX                   XXXX         XXX                   XXX              XXXXXXX                  |
    |XXXX                  XXXX          XXX                   XX             XXXXXXXXX                  |
    |XXXXX                 XXXX           XX                    X             XXXXXXXX                   |
    | XXXXXX             XXXXX                 XX            XX      XXXXXXXXXXXXXXXXX                   |
    | XXXXXX             XXXXX                 XXX          XXX      XXXXXXXXXXXXXXXX                    |
    | XXXXXXX           XXXXXX               XXXXX         XXXXXX    XXXXXXXXXXXXXXX                     |
    | XXXXXXX           XXXXXX               XXXXX         XXXXXX    XXXXXXXXXXXXXXX                     |
    |  XXXXXXXX        XXXXXX               XXXXX          XXXXXX   XXXXXXXXXXXXXXX                      |
    |   XXXXXXX        XXXXX                XXX            XXXXXX   XXXXXXXXXXXXXX                       |
    |    XXXXXXXXXXXXXXXXXXX                XXX             XXXXX  XXXXXXXXXXXXXX                        |
    |    XXXXXXXXXXXXXXXXX                  XXX             XXXXX  XXXXXXXXXXXXXX                        |
    |    XXXXXXXXXXXXXXXXX                  XXX             XXXXXXXXXXXXXXXXXXXXX                        |
    |    XX    XXXXXXX                 XX    X              XXXXXXXXXXXXXXXXXXXXX                        |
    |          XXXXXX                        X              XXXXXXXXXXXXXXXXXXXX                         |
    |                                                       XXXXXXXXXXXXXXXXXXX                          |
    |                                                       XXXXXXXXXXXXXXXXXX                           |
    |                                                       XXXXXXXXXXXXXXXX                             |
    |                                                        XXXXXXXXXXXXXXX                             |
    |                                                        XXXXXXXXXXXXXX                              |
    |                                                      X XXXXXXXXXXXXXX                              |
    |                                                     XXXXXXXXXXXXXX                                 |
    |                                                     XXXXXXXXXXXXXX                                 |
    |                                                    XXXXXXXXXXXXX                                   |
    |                                                    XXXXXXXXXXXX                                    |
    |                                                     XXXXXXXXX                                      |
    |                                                     XXXXXXXX                                       |
    ----------------------------------------------------------------------------------------------------

```    

### Bài 34 - Lấy hình chiếu theo phương ngang

Ta dùng một phương pháp đơn giản để cắt các số thập phân thành các chữ số. Phương pháp này thường không được sử dụng với các ảnh viết tay nhưng được sử dụng với các ảnh mà các đoạn chữ số được viết dưới dạng in, đánh máy. Theo đó, ta sẽ chiếu ảnh theo phương thẳng đứng lên một đường ngang. Mỗi điểm trên đường ngang này sẽ có màu đen nếu nó là hình chiếu cuả 1 pixel đen theo phương thẳng đứng, và sẽ có màu trắng nếu không phải là hình chiếu của pixel đen nào.

Ví dụ khi chiếu hình ảnh trên lên phương ngang, ta sẽ được hình có dạng sau:

`xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx        xxxxxxxxxxxxx        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

Tức hình chiếu có 3 vùng đen, xen giữa là 2 vùng trắng. 3 vùng đen này tương ứng với vị trí của số 3, dấu phẩy (chấm), và số 5.

Giả sử mọi hình ảnh đều có dạng này.

*Hãy viết method **`getProjection()`** trong class **`ComplexImage`** trả lại hình chiếu trên phương ngang của hình ảnh dưới dạng một list các list $[[a_1, b_1], \ldots, [a_k, b_k]]$, trong đó $[a_1, b_1[$ là vùng đen thứ nhất trên hình chiếu, $[a_2, b_2[$ là vùng đen thứ hai v.v...*

Đoạn code dưới đây giúp test hàm của bạn.


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
my_image = ComplexImage("ComplexScoreData/BKHCM_3B_0-A.jpg")
my_image.getProjection()
```




    [[0, 28], [34, 45], [52, 100]]



Kết quả trên nói rằng khi chiếu lên phương ngang (theo phương thẳng đứng), ta được list 100 pixel có các pixel từ 0 đến 27, từ 34 đến 44, từ 52 đến 99 màu đen, còn lại màu trắng.

### Bài 35 - Cắt ra các hình ảnh nhỏ hơn

*Từ kết quả bài 34, viết method **`getSmallerImages()`** trong class **`ComplexImage`** cắt ra các hình con theo từng nhóm đen một đã xác định ở bài 34. Với mỗi hình đen này, xoá tất cả các hàng gồm toàn pixel trắng. Trả lại kết quả dưới dạng 1 list các numpy array, mỗi *

Đoạn code dưới đây giúp test hàm của bạn.


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
my_image = ComplexImage("ComplexScoreData/BKHCM_3B_0-A.jpg")
smaller_images = my_image.getSmallerImages() #Cần có 3 array, 1 ứng với số 3, 1 ứng với dấu chấm, 1 ứng với số 5
smaller_images
```




    [array([[255, 255, 255, ..., 255, 255, 255],
            [255, 255, 255, ..., 255, 255, 255],
            [255, 255, 255, ..., 255, 255, 255],
            ...,
            [255, 255, 255, ..., 255, 255, 255],
            [255, 255, 255, ..., 255, 255, 255],
            [255, 255, 255, ..., 255, 255, 255]], dtype=uint8),
     array([[255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255],
            [255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255],
            [255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255],
            [255,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255],
            [255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255],
            [255, 255,   0,   0,   0, 255, 255, 255, 255, 255, 255],
            [255, 255,   0,   0,   0, 255, 255, 255, 255, 255, 255],
            [255, 255,   0,   0,   0, 255, 255, 255, 255, 255, 255],
            [255, 255,   0,   0,   0, 255, 255, 255, 255, 255, 255],
            [255, 255,   0,   0,   0, 255, 255, 255, 255, 255, 255],
            [255, 255, 255,   0,   0, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255,   0,   0, 255],
            [255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0],
            [255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
            [255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0],
            [255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255],
            [255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255],
            [255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255],
            [255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255],
            [255, 255, 255, 255, 255,   0,   0,   0, 255, 255, 255],
            [  0,   0, 255, 255, 255, 255,   0, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255,   0, 255, 255, 255, 255]],
           dtype=uint8),
     array([[255, 255, 255, ...,   0, 255, 255],
            [255, 255, 255, ...,   0, 255, 255],
            [255, 255, 255, ...,   0,   0, 255],
            ...,
            [  0,   0,   0, ..., 255, 255, 255],
            [255,   0,   0, ..., 255, 255, 255],
            [255,   0,   0, ..., 255, 255, 255]], dtype=uint8)]



Đoạn code dưới đây giúp kiểm tra hình dạng 3 hình vừa cắt ra


```python
def draw_array(array):
    S = "-" * array.shape[1] + "\n"
    for i in range(array.shape[0]):
        S += "|"
        for j in range(array.shape[1]):
            if array[i, j] == 0:
                S += "X"
            else:
                S += " "
        S += "|\n"        
    S += ("-" * array.shape[1])
    print(S)

draw_array(smaller_images[0])
```

```
    ----------------------------
    |                XXX         |
    |               XXXX         |
    |               XXXXXX       |
    |              XXXXXXX       |
    |              XXXXXXXX      |
    |              XXXXXXXX      |
    |         XXXXXXXXXXXXX      |
    |         XXXXXXXXXXXXX      |
    |        XXXXXXXXXXXXXXX     |
    |        XXXXXXXXXXXXXXX     |
    |        XXXXXXXXXXXXXXXX    |
    |        XXXXXXX     XXXX    |
    |        XXXXXX      XXXX    |
    |        XXXXXX      XXXX    |
    |       XXXXXXX      XXXX    |
    |       XXXXXXX      XXXX    |
    |       XXXXXXX      XXXX    |
    |       XXXXXXX      XXXX    |
    |        XXXXX       XXXX    |
    |        XXXXX       XXXX    |
    |         XXX        XXXX    |
    |          XX        XXXX    |
    |          X         XXX     |
    |                   XXXX     |
    |                   XXX      |
    |                   XXX      |
    |                   XXX      |
    |                   XXX      |
    |                   XXX      |
    |                  XXXX      |
    |                  XXX       |
    |                  XXX       |
    |                  XX        |
    |                  XX        |
    |                            |
    |                            |
    |                            |
    |        XXXXX               |
    |        XXXXX               |
    |       XXXXXXX              |
    |       XXXXXXXX             |
    |      XXXXXXXXX             |
    |      XXXXXXXXX   XXX       |
    |       XXXXXXX   XXXXX      |
    |       XXX       XXXXXX     |
    |        XX       XXXXXXX    |
    |                 XXXXXXX    |
    |                  XXXXXXX   |
    |                  XXXXXXX   |
    |                   XXXXXX   |
    |                   XXXXXXXX |
    |                    XXXXXXX |
    |                    XXXXXXXX|
    |                     XXXXXXX|
    |                     XXXXXXX|
    |                       XXXXX|
    |                        XXXX|
    |                        XXXX|
    |                        XXXX|
    |                        XXXX|
    |                        XXXX|
    |                        XXXX|
    |                        XXXX|
    | X                      XXXX|
    |XX                      XXXX|
    |XX                     XXXXX|
    |XXX                    XXXXX|
    |XXX                    XXXX |
    |XXXX                   XXXX |
    |XXXX                  XXXX  |
    |XXXXX                 XXXX  |
    | XXXXXX             XXXXX   |
    | XXXXXX             XXXXX   |
    | XXXXXXX           XXXXXX   |
    | XXXXXXX           XXXXXX   |
    |  XXXXXXXX        XXXXXX    |
    |   XXXXXXX        XXXXX     |
    |    XXXXXXXXXXXXXXXXXXX     |
    |    XXXXXXXXXXXXXXXXX       |
    |    XXXXXXXXXXXXXXXXX       |
    |    XX    XXXXXXX           |
    |          XXXXXX            |
    ----------------------------
``` 


```python
draw_array(smaller_images[1])
```

    -----------
    |   XXXX    |
    |   XXXX    |
    |  XXXXX    |
    | XXXXXX    |
    |  XXXXX    |
    |  XXX      |
    |  XXX      |
    |  XXX      |
    |  XXX      |
    |  XXX      |
    |   XX      |
    |        XX |
    |        XXX|
    |      XXXXX|
    |      XXXXX|
    |     XXXXX |
    |     XXX   |
    |     XXX   |
    |     XXX   |
    |     XXX   |
    |XX    X    |
    |      X    |
    -----------
    


```python
draw_array(smaller_images[2])
```

```
    ------------------------------------------------
    |                                       XXXXXXX  |
    |                                       XXXXXXX  |
    |                                     XXXXXXXXXX |
    |                                     XXXXXXXXXX |
    |                                XXXXXXXXXXXXXXXX|
    |                                XXXXXXXXXXXXXXXX|
    |                              XXXXXXXXXXXXXXXXXX|
    |                              XXXXXXXXXXXXXXXXXX|
    |                            XXXXXXXXXXXXXXXXXXXX|
    |                            XXXXXXXXXXXXXXXXXX  |
    |                          XXXXXXXXXXXXXXXXXXXX  |
    |                          XXXXXXXXXXXXX         |
    |                         XXXXXXXXXXXXXX         |
    |                        XXXXXXXXXXXX            |
    |                   XXXXXXXXXXXXXXXXX            |
    |                   XXXXXXXXXXXXXXX              |
    |                 XXXXXXXXXXXXXXXX               |
    |                 XXXXXXXXXXXXX                  |
    |               XXXXXXXXXXXXXXX                  |
    |               XXXXXXXXXXXXX                    |
    |              XXXXXXXXXXXXXX                    |
    |            XXXXXXXXXXXXXXX                     |
    |            XXXXXXXXXXXXXXX                     |
    |           XXXXXXXXX XX                         |
    |           XXXXXXXX  X                          |
    |          XXXXXX                                |
    |          XXXXXX                                |
    |          XXXXX                                 |
    |          XXXXX                                 |
    |          XXXXX                                 |
    |          XXXXX                                 |
    |          XXXXX                                 |
    |          XXXXX                                 |
    |         XXXXX                                  |
    |         XXXXX                                  |
    |         XXXXX                                  |
    |         XXXXX                                  |
    |        XXXXXX                                  |
    |        XXXXXX                                  |
    |        XXXXX                                   |
    |        XXXXX                                   |
    |        XXXXXXX                                 |
    |         XXXXXXX                                |
    |         XXXXXXXXXXXX                           |
    |         XXXXXXXXXXXX                           |
    |         XXXXXXXXXXXX                           |
    |         XXXXXXXXXXXXX                          |
    |         XXXXXXXXXXXXX                          |
    |         XXXXXXXXXXXXXXXX                       |
    |         XXXXXXXXXXXXXXXX                       |
    |               XXXXXXXXXXXX                     |
    |                XXXXXXXXXXX                     |
    |                   XXXXXXXXX                    |
    |                   XXXXXXXXX                    |
    |                    XXXXXXXXX                   |
    |                     XXXXXXXX                   |
    |                      XXXXXXX                   |
    |                       XXXXXX                   |
    |                       XXXXXXX                  |
    |                       XXXXXXX                  |
    |                        XXXXXX                  |
    |                         XXXXX                  |
    |                         XXXXX                  |
    |                          XXXX                  |
    |                          XXXXX                 |
    |                          XXXXX                 |
    |                          XXXXXX                |
    |                          XXXXXX                |
    |                          XXXXXX                |
    |                          XXXXXX                |
    |                         XXXXXX                 |
    |       XX              XXXXXXXX                 |
    |      XXX              XXXXXXX                  |
    |      XXX              XXXXXXX                  |
    |      XXX              XXXXXXX                  |
    |      XX             XXXXXXXXX                  |
    |       X             XXXXXXXX                   |
    |    XX      XXXXXXXXXXXXXXXXX                   |
    |   XXX      XXXXXXXXXXXXXXXX                    |
    |  XXXXXX    XXXXXXXXXXXXXXX                     |
    |  XXXXXX    XXXXXXXXXXXXXXX                     |
    |  XXXXXX   XXXXXXXXXXXXXXX                      |
    |  XXXXXX   XXXXXXXXXXXXXX                       |
    |   XXXXX  XXXXXXXXXXXXXX                        |
    |   XXXXX  XXXXXXXXXXXXXX                        |
    |   XXXXXXXXXXXXXXXXXXXXX                        |
    |   XXXXXXXXXXXXXXXXXXXXX                        |
    |   XXXXXXXXXXXXXXXXXXXX                         |
    |   XXXXXXXXXXXXXXXXXXX                          |
    |   XXXXXXXXXXXXXXXXXX                           |
    |   XXXXXXXXXXXXXXXX                             |
    |    XXXXXXXXXXXXXXX                             |
    |    XXXXXXXXXXXXXX                              |
    |  X XXXXXXXXXXXXXX                              |
    | XXXXXXXXXXXXXX                                 |
    | XXXXXXXXXXXXXX                                 |
    |XXXXXXXXXXXXX                                   |
    |XXXXXXXXXXXX                                    |
    | XXXXXXXXX                                      |
    | XXXXXXXX                                       |
    ------------------------------------------------
```
    

### Bài 36 - Đâu là dấu phẩy, đâu là số?

Ta sẽ chấp nhận trong các hình nhỏ đã cắt ra, một hình là số nếu nó có số pixel đen ít nhất bằng 400, và một hình là dấu phẩy nếu số pixel đen nằm giữa 20 và 300, còn lại được xem là nhiễu (bụi). Các tham số này được định nghĩa bằng 

``` python
    DECIMAL_POINT_MIN_SIZE = 20
    DECIMAL_POINT_MAX_SIZE = 300
```

trong cùng class.

*Viết các static method **`isDecimalPoint(img)`** và  **`isDigit(img)`** trả lại `True` nếu numpy array `img` thoả mãn các điều kiện trên và `False` nếu không.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
my_image = ComplexImage("ComplexScoreData/BKHCM_3B_0-A.jpg")
smaller_images = my_image.getSmallerImages()
ComplexImage.isDigit(smaller_images[0]), ComplexImage.isDigit(smaller_images[1]), ComplexImage.isDigit(smaller_images[2])
```




    (True, False, True)




```python
ComplexImage.isDecimalPoint(smaller_images[0]), ComplexImage.isDecimalPoint(smaller_images[1]), ComplexImage.isDecimalPoint(smaller_images[2])
```




    (False, True, False)



### Bài 37 - Lưu các hình ảnh là số

Sau khi xác định đâu là hình ảnh đâu là số, ta sẽ lưu các file hình ảnh với kích thước 28 x 28 để phục vụ cho bài toán phân loại chữ số. Thư mục sẽ lưu file là

`SCORE_DATA_SPLITTED = "ComplexScoreData_Splitted/"`

Ta sẽ thực hiện như sau:
- Cắt các ảnh thành các ảnh con (bài 35)
- Dùng bài 36 để phân biệt đâu là dấu phẩy, đâu là số, đâu là nhiễu
- Căn cứ vào đó, ta sẽ lưu các ảnh là số với tên dạng như sau `"ComplexScoreData/BKHCM_3B_0-A_Position_0.jpg"`, tức là thêm `"_Position_i"` vào trước ".jpg", trong đó i = 0 nếu là chữ số hàng đơn vị của điểm, -1 nếu là chữ số hàng phần chục, 1 nếu là chữ số hàng chục, -2 nếu là chữ số hàng phần trăm...

*Hãy viết method **`saveSmallerImages()`** trong cùng class thực hiện quy trình trên.*

Đoạn code dưới đây giúp test hàm của bạn. Hãy xoá các file trong thư mục "ComplexScoreData_Splitted/" trước


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
SCORE_DATA_SPLITTED = "ComplexScoreData_Splitted/"
my_image = ComplexImage("ComplexScoreData/BKHCM_3B_0-A.jpg")
my_image.saveSmallerImages()
```

Sau khi chạy xong, thư mục "ComplexScoreData_Splitted" sẽ chứa 2 file như sau:

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson8/TD/F2.png" width=800></img>

Trong đó `"_Position_0"` nói rằng ảnh thứ nhất là chữ số hàng đơn vị, `"_Position_-1"` nói rằng ảnh thứ hai là chữ số hàng phần mười.

### Bài 38 - Dự đoán trên dataset

Ở bài 30 ta đã có hàm **`readEvaluationFormAsDict`** đọc tất cả các phiếu chấm từ 1 thư mục và trả lại 1 từ điển gồm các điểm được cho, áp dụng cho số nguyên.

*Bây giờ, hãy viết hàm **`readComplexEvaluationFormAsDict(image_folder, model_file)`** (nằm ngoài class `ComplexMark`) thực hiện tương tự nhưng có xét đến các điểm thập phân, trả lại kết quả dưới dạng một từ điển như đoạn code test dưới đây.* 


```python
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
SCORE_DATA_SPLITTED = "ComplexScoreData_Splitted/"
readComplexEvaluationFormAsDict(RAW_DATA, LOGISTIC_REGRESSION_MTN_PARAMS)
```




    {'BKHCM_3B_0-A.jpg': 3.5,
     'BKHCM_3B_0-B.jpg': 3.2,
     'BKHCM_3B_0-C.jpg': 2.0,
     'BKHCM_3B_0-D.jpg': 0.5,
     'BKHCM_3B_1-A.jpg': 2.8,
     'BKHCM_3B_1-B.jpg': 2.1,
     'BKHCM_3B_1-C.jpg': 3.0,
     'BKHCM_3B_1-D.jpg': 5.0,
     'BKHCM_3B_10-A.jpg': 5.5,
     'BKHCM_3B_10-B.jpg': 2.7,
     'BKHCM_3B_10-C.jpg': 5.9,
     'BKHCM_3B_10-D.jpg': 0.5,
     'BKHCM_3B_11-A.jpg': 4.0,
     'BKHCM_3B_11-B.jpg': 3.6,
     'BKHCM_3B_11-C.jpg': 1.7000000000000002,
     'BKHCM_3B_11-D.jpg': 1.5,
     'BKHCM_3B_2-A.jpg': 5.1,
     'BKHCM_3B_2-B.jpg': 3.3,
     'BKHCM_3B_2-C.jpg': 2.5,
     'BKHCM_3B_2-D.jpg': 0.5,
     'BKHCM_3B_3-A.jpg': 4.7,
     'BKHCM_3B_3-B.jpg': 2.2,
     'BKHCM_3B_3-C.jpg': 4.5,
     'BKHCM_3B_3-D.jpg': 0.6000000000000001,
     'BKHCM_3B_4-A.jpg': 1.6,
     'BKHCM_3B_4-B.jpg': 3.5,
     'BKHCM_3B_4-C.jpg': 2.5,
     'BKHCM_3B_4-D.jpg': 0.5,
     'BKHCM_3B_5-A.jpg': 9.4,
     'BKHCM_3B_5-B.jpg': 2.1,
     'BKHCM_3B_5-C.jpg': 4.0,
     'BKHCM_3B_5-D.jpg': 1.5,
     'BKHCM_3B_6-A.jpg': 4.3,
     'BKHCM_3B_6-B.jpg': 2.1,
     'BKHCM_3B_6-C.jpg': 2.5,
     'BKHCM_3B_6-D.jpg': 0.1,
     'BKHCM_3B_7-A.jpg': 5.2,
     'BKHCM_3B_7-B.jpg': 3.4,
     'BKHCM_3B_7-C.jpg': 4.5,
     'BKHCM_3B_7-D.jpg': 1.5,
     'BKHCM_3B_8-A.jpg': 4.6,
     'BKHCM_3B_8-B.jpg': 2.9,
     'BKHCM_3B_8-C.jpg': 2.5,
     'BKHCM_3B_8-D.jpg': 0.5,
     'BKHCM_3B_9-A.jpg': 1.9,
     'BKHCM_3B_9-B.jpg': 3.3,
     'BKHCM_3B_9-C.jpg': 4.4,
     'BKHCM_3B_9-D.jpg': 1.5}



(Lỗi ở "BKHCM_3B_11" hay "BKHCM_3B_3" là do sai số của phép cộng trong Python, không phải do hình ảnh chứa số 2 ở hàng phần trăm tỉ...)
