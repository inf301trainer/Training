
# TD6 - Dự đoán giá Bitcoin

## Mô tả

Bitcoin là đồng tiền điện tử có giá cả và khối lượng giao dịch lớn nhất thế giới tính đến đầu năm 2018. Bitcoin được giao dịch ở các sàn thuộc nhiều quốc gia khác nhau, với giá cả xấp xỉ nhau. 

Trong TD này ta dựa vào dữ liệu về giá cả Bitcoin tại sàn giao dịch Bitfinex (trụ sở tại Anh, Đài Loan và Hongkong), được xem là một trong các sàn giao dịch tiền điện tử lớn nhất thế giới, cùng với một số dữ liệu khác, để xây dựng một mô hình đơn giản dự đoán giá của Bitcoin (BTC) theo đơn vị USD; cùng với sự thay đổi của nó trong thời gian gần. Giả sử ta là một nhà đầu tư ngắn hạn, thời điểm giá ta muốn dự đoán là 1 tuần (7 ngày) sau khi có dữ liệu hiện tại.


## Dữ liệu

Trong bài này, bạn sẽ cần làm việc với các dữ liệu sau:

- `Data/BTCPrice.csv`: Tập tin mô tả giá của BTC theo ngày từ 31/12/2015 đến hết 31/12/2017, gồm các cột: Ngày, Giá mở cửa, Giá cao nhất ngày, Giá thấp nhất ngày, Giá đóng cửa, Khối lượng giao dịch theo số BTC, Khối lượng giao dịch theo USD, và giá trung bình trong ngày. Như các loại sản phẩm tài chính khác, ta quy ước **giá BTC của ngày là giá đóng cửa (Close) của ngày đó**, và sẽ làm việc chủ yếu với cột "Close" (thay vì cột "Weighted Price"). 

 Dữ liệu này được download tại https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv


```python
import pandas as pd
price_data = pd.read_csv('Data/BTCPrice.csv')
price_data.head()
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
      <th>Timestamp</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume (BTC)</th>
      <th>Volume (Currency)</th>
      <th>Weighted Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31/12/2015 00:00</td>
      <td>426.09</td>
      <td>433.89</td>
      <td>419.99</td>
      <td>430.89</td>
      <td>6634.86</td>
      <td>2833319.70</td>
      <td>427.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/01/2016 00:00</td>
      <td>430.89</td>
      <td>436.00</td>
      <td>427.20</td>
      <td>433.82</td>
      <td>3788.11</td>
      <td>1640577.93</td>
      <td>433.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02/01/2016 00:00</td>
      <td>434.87</td>
      <td>435.99</td>
      <td>430.42</td>
      <td>433.55</td>
      <td>2972.06</td>
      <td>1287773.39</td>
      <td>433.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>03/01/2016 00:00</td>
      <td>433.20</td>
      <td>434.09</td>
      <td>424.06</td>
      <td>431.04</td>
      <td>4571.10</td>
      <td>1959152.60</td>
      <td>428.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04/01/2016 00:00</td>
      <td>431.54</td>
      <td>435.86</td>
      <td>428.44</td>
      <td>434.17</td>
      <td>5717.60</td>
      <td>2474772.50</td>
      <td>432.83</td>
    </tr>
  </tbody>
</table>
</div>



- `Data/BTCTrend.csv`: Tập tin mô tả xu hướng tìm kiếm từ "bitcoin" (sai khác một phép viết hoa) trên google trong năm 2016-2017.
- `Data/BTCNews.csv`: Tập tin liệt kê những sự kiện được cho là ảnh hưởng tích cực hay tiêu cực đến Bitcoin trong năm 2016-2017.


```python
from BTCPriceForecast_Solution import *
```

## Yêu cầu

Bạn cần hoàn thành các hàm trong `BTCPriceForecast.py`.

Lưu ý: Mục đích của TD là làm quen với các mô hình hồi quy tuyến tính (+suy rộng) đã học như Linear Regression, Polynomial Regression, Ridge, Lasso. Do đó một số biến (liên quan đến các chỉ số kĩ thuật trong tài chính, và mô tả hiệu ứng của thông tin) sẽ được giới thiệu một cách không tự nhiên. Việc giải thích vì sao giới thiệu các biến này không thuộc phạm vi machine learning.

Bạn chạy đoạn code sau trước khi test các hàm của mình


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split #replaced by model_selection in sklearn 0.19+
from bs4 import BeautifulSoup
import statsmodels.formula.api as sm
from datetime import datetime
import time
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SHIFT_NUMBER = 7

COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'rsv_9', 'kdjk_9', 'kdjj_9', 'macd', 'macds', 'macdh', 'rs_6', 'rsi_6', 'rs_12', 'rsi_12', 'wr_6', 'wr_10', 'cci', 'cci_20', 'tr', 'atr', 'dma', 'high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14', 'pdi_14', 'mdm_14', 'mdi_14', 'dx_14', 'adx', 'adxr', 'trix', 'change', 'vr', 'vr_6_sma']
REFINED_COLUMNS = [ 'cr-ma3', 'kdjk_9', 'macds', 'macdh', 'rs_6', 'rsi_6', 'wr_6', 'atr', 'dma', 'high_delta', 'um', 'dm', 'pdm_14', 'mdm_14', 'mdi_14', 'trix', 'change']

PRICE_FILE = "Data/BTCPrice.csv"
EVENT_FILE = "Data/BTCNews.csv"
TREND_FILE = "Data/BTCTrend.csv"
EVENT_PAGE = "https://99bitcoins.com/price-chart-history/"

# COLUMNS
TIMESTAMP = "Timestamp"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
VOLUME_BTC = "Volume (BTC)"
VOLUME_CURRENCY = "Volume (Currency)"
WEIGHTED_PRICE = "Weighted Price"

#TEST_SIZE
TEST_SIZE = 0.5
MAX_PREVIOUS_DAYS = 30

#PARAMETERS
ALPHA = np.log(2)
```

## Phần 1. Khởi động

### Bài 1. Chuẩn bị target

Ta cần dự đoán giá của BTC sau 1 tuần. Muốn vậy, ta xây dựng một mô hình đoán $y$ từ $X$, trong đó $X$ là ma trận tạo thành từ một số cột trong file dữ liệu, $y$ là cột "Close" bị đẩy lệch về sau 7 ngày.

1. *Hãy viết hàm **`readData(filename)`** nhận đối số là đường dẫn của một file dữ liệu dạng csv như 3 file trên và trả lại DataFrame tương ứng.*

2. *Hãy viết hàm **`getTarget(data, shift_number)`** nhận đối số **`data`** là một DataFrame có chứa cột "Close" (như dataframe ứng với file **`Data/BTCPrice.csv`**), **`shift_number`** là số ngày cần đẩy lệch, và trả lại kết quả là cột "Close" bị đẩy lệch **`shift_number`** ngày về tương lai. Kết quả này có dạng một list hoặc array tuỳ bạn chọn, có độ dài bằng số hàng của **`data`**, riêng các phần tử cuối cùng bằng 0 (vì chưa biết kết quả trong tương lai).*

Hàm của bạn cần chạy được đoạn code sau.


```python
price_data = readData(PRICE_FILE) # File BTCPrice.csv 
price_data.head(10)
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
      <th>Timestamp</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume (BTC)</th>
      <th>Volume (Currency)</th>
      <th>Weighted Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31/12/2015 00:00</td>
      <td>426.09</td>
      <td>433.89</td>
      <td>419.99</td>
      <td>430.89</td>
      <td>6634.86</td>
      <td>2833319.70</td>
      <td>427.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/01/2016 00:00</td>
      <td>430.89</td>
      <td>436.00</td>
      <td>427.20</td>
      <td>433.82</td>
      <td>3788.11</td>
      <td>1640577.93</td>
      <td>433.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02/01/2016 00:00</td>
      <td>434.87</td>
      <td>435.99</td>
      <td>430.42</td>
      <td>433.55</td>
      <td>2972.06</td>
      <td>1287773.39</td>
      <td>433.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>03/01/2016 00:00</td>
      <td>433.20</td>
      <td>434.09</td>
      <td>424.06</td>
      <td>431.04</td>
      <td>4571.10</td>
      <td>1959152.60</td>
      <td>428.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04/01/2016 00:00</td>
      <td>431.54</td>
      <td>435.86</td>
      <td>428.44</td>
      <td>434.17</td>
      <td>5717.60</td>
      <td>2474772.50</td>
      <td>432.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>05/01/2016 00:00</td>
      <td>433.31</td>
      <td>435.39</td>
      <td>429.50</td>
      <td>432.43</td>
      <td>3881.72</td>
      <td>1677111.44</td>
      <td>432.05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>06/01/2016 00:00</td>
      <td>432.45</td>
      <td>432.67</td>
      <td>426.54</td>
      <td>429.56</td>
      <td>5507.61</td>
      <td>2369036.46</td>
      <td>430.14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>07/01/2016 00:00</td>
      <td>429.99</td>
      <td>459.16</td>
      <td>429.25</td>
      <td>457.00</td>
      <td>16833.42</td>
      <td>7536413.29</td>
      <td>447.71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>08/01/2016 00:00</td>
      <td>456.87</td>
      <td>465.00</td>
      <td>444.51</td>
      <td>452.70</td>
      <td>10258.99</td>
      <td>4654977.99</td>
      <td>453.75</td>
    </tr>
    <tr>
      <th>9</th>
      <td>09/01/2016 00:00</td>
      <td>452.02</td>
      <td>454.00</td>
      <td>446.68</td>
      <td>448.84</td>
      <td>4396.05</td>
      <td>1976929.79</td>
      <td>449.71</td>
    </tr>
  </tbody>
</table>
</div>




```python
trend_data = readData(TREND_FILE) # File BTCTrend.csv
trend_data.head()
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
      <th>Timestamp</th>
      <th>Trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/01/2016</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02/01/2016</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03/01/2016</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04/01/2016</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05/01/2016</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>




```python
target = getTarget(price_data, SHIFT_NUMBER) # Kết quả là một array bắt đầu bởi giá của ngày thứ 7, kết thúc bởi 7 số 0
pd.DataFrame(getTarget(price_data, SHIFT_NUMBER)).T # Biểu diễn dưới dạng frame để dễ nhìn
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
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>457.0</td>
      <td>452.7</td>
      <td>448.84</td>
      <td>448.79</td>
      <td>448.0</td>
      <td>434.85</td>
      <td>432.64</td>
      <td>429.55</td>
      <td>360.0</td>
      <td>386.91</td>
      <td>...</td>
      <td>14340.0</td>
      <td>12640.0</td>
      <td>13880.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 732 columns</p>
</div>



### Bài 2. Mô hình AR

Mô hình AR giả thuyết rằng giá của một sản phẩm tài chính ở ngày thứ $n$ phụ thuộc vào giá của nó ở $p$ ngày trước đó ($n-1, \ldots, n-p$) theo quan hệ

$$
X_n = a_0 + \sum_{i=1}^p a_i X_{n-i} + \epsilon_n
$$

trong đó $X_n$ là giá đóng  $\epsilon_n$ là một bụi Gaussian (kì vọng 0) iid với tất cả các $n$. Đây là một mô hình hồi quy tuyến tính.

Ta thử dùng mô hình này để dự đoán không chỉ giá của ngày tiếp theo $X_n$, mà còn là giá của tương lai $X_{n+7}$. Mô hình sẽ trở thành:

$$
X_{n + SHIFT} = a_0 + a_1 X_n + a_2 X_{n-1} + \ldots + a_p X_{n-p+1} + \epsilon_n
$$

trong đó $SHIFT = 7$ là số ngày trong tương lai mà ta quan tâm.

1. *Hãy viết hàm **`getpRecentPrices(current_price, p)`** nhận đối số **`current_price`** là một array biểu thị giá hiện tại (như cột "Close") của file giá cả; **`p`** là số nguyên biểu thị số ngày liên tiếp cho đến hiện tại; và trả lại một array gồm **`p`** cột, cột 0 là giá hiện tại, cột 1 là giá ngày hôm trước, ..., cột $i$ là giá cách đây $i$ hôm. Số hàng của array bằng số hàng của **current_price**, dữ liệu nào khuyết thay bằng 0.*

    Đoạn code sau giúp test hàm của bạn.


```python
price_data = readData(PRICE_FILE)
target = getTarget(price_data, SHIFT_NUMBER)
current_price = price_data.loc[:, CLOSE]
X1 = getpRecentPrices(current_price, 10)
pd.DataFrame(X1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
      <td>430.89</td>
    </tr>
    <tr>
      <th>10</th>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
      <td>433.82</td>
    </tr>
    <tr>
      <th>11</th>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
      <td>433.55</td>
    </tr>
    <tr>
      <th>12</th>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
      <td>431.04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
      <td>434.17</td>
    </tr>
    <tr>
      <th>14</th>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
      <td>432.43</td>
    </tr>
    <tr>
      <th>15</th>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
      <td>429.56</td>
    </tr>
    <tr>
      <th>16</th>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
      <td>457.00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
      <td>452.70</td>
    </tr>
    <tr>
      <th>18</th>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
      <td>448.84</td>
    </tr>
    <tr>
      <th>19</th>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
      <td>448.79</td>
    </tr>
    <tr>
      <th>20</th>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
      <td>448.00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
      <td>434.85</td>
    </tr>
    <tr>
      <th>22</th>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
      <td>432.64</td>
    </tr>
    <tr>
      <th>23</th>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
      <td>429.55</td>
    </tr>
    <tr>
      <th>24</th>
      <td>402.16</td>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
      <td>360.00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>390.90</td>
      <td>402.16</td>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
      <td>386.91</td>
    </tr>
    <tr>
      <th>26</th>
      <td>391.77</td>
      <td>390.90</td>
      <td>402.16</td>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
      <td>380.16</td>
    </tr>
    <tr>
      <th>27</th>
      <td>394.95</td>
      <td>391.77</td>
      <td>390.90</td>
      <td>402.16</td>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
      <td>385.49</td>
    </tr>
    <tr>
      <th>28</th>
      <td>379.75</td>
      <td>394.95</td>
      <td>391.77</td>
      <td>390.90</td>
      <td>402.16</td>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
      <td>376.76</td>
    </tr>
    <tr>
      <th>29</th>
      <td>378.22</td>
      <td>379.75</td>
      <td>394.95</td>
      <td>391.77</td>
      <td>390.90</td>
      <td>402.16</td>
      <td>386.40</td>
      <td>382.55</td>
      <td>409.59</td>
      <td>416.32</td>
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
    </tr>
    <tr>
      <th>702</th>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
      <td>9868.82</td>
      <td>9708.07</td>
      <td>9271.06</td>
      <td>8717.99</td>
      <td>8199.19</td>
      <td>7989.00</td>
    </tr>
    <tr>
      <th>703</th>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
      <td>9868.82</td>
      <td>9708.07</td>
      <td>9271.06</td>
      <td>8717.99</td>
      <td>8199.19</td>
    </tr>
    <tr>
      <th>704</th>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
      <td>9868.82</td>
      <td>9708.07</td>
      <td>9271.06</td>
      <td>8717.99</td>
    </tr>
    <tr>
      <th>705</th>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
      <td>9868.82</td>
      <td>9708.07</td>
      <td>9271.06</td>
    </tr>
    <tr>
      <th>706</th>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
      <td>9868.82</td>
      <td>9708.07</td>
    </tr>
    <tr>
      <th>707</th>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
      <td>9868.82</td>
    </tr>
    <tr>
      <th>708</th>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
      <td>9824.68</td>
    </tr>
    <tr>
      <th>709</th>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
      <td>9947.67</td>
    </tr>
    <tr>
      <th>710</th>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
      <td>10840.45</td>
    </tr>
    <tr>
      <th>711</th>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
      <td>10872.00</td>
    </tr>
    <tr>
      <th>712</th>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
      <td>11250.00</td>
    </tr>
    <tr>
      <th>713</th>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
      <td>11613.07</td>
    </tr>
    <tr>
      <th>714</th>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
      <td>11677.00</td>
    </tr>
    <tr>
      <th>715</th>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
      <td>13623.50</td>
    </tr>
    <tr>
      <th>716</th>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
      <td>16599.99</td>
    </tr>
    <tr>
      <th>717</th>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
      <td>15800.00</td>
    </tr>
    <tr>
      <th>718</th>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
      <td>14607.49</td>
    </tr>
    <tr>
      <th>719</th>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
      <td>14691.00</td>
    </tr>
    <tr>
      <th>720</th>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
      <td>16470.00</td>
    </tr>
    <tr>
      <th>721</th>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
      <td>16650.01</td>
    </tr>
    <tr>
      <th>722</th>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
      <td>16250.00</td>
    </tr>
    <tr>
      <th>723</th>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
      <td>16404.99</td>
    </tr>
    <tr>
      <th>724</th>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
      <td>17471.50</td>
    </tr>
    <tr>
      <th>725</th>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
      <td>19187.78</td>
    </tr>
    <tr>
      <th>726</th>
      <td>15764.44</td>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
      <td>18953.00</td>
    </tr>
    <tr>
      <th>727</th>
      <td>15364.93</td>
      <td>15764.44</td>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
      <td>18940.57</td>
    </tr>
    <tr>
      <th>728</th>
      <td>14470.07</td>
      <td>15364.93</td>
      <td>15764.44</td>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
      <td>17700.00</td>
    </tr>
    <tr>
      <th>729</th>
      <td>14340.00</td>
      <td>14470.07</td>
      <td>15364.93</td>
      <td>15764.44</td>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
      <td>16466.98</td>
    </tr>
    <tr>
      <th>730</th>
      <td>12640.00</td>
      <td>14340.00</td>
      <td>14470.07</td>
      <td>15364.93</td>
      <td>15764.44</td>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
      <td>15600.01</td>
    </tr>
    <tr>
      <th>731</th>
      <td>13880.00</td>
      <td>12640.00</td>
      <td>14340.00</td>
      <td>14470.07</td>
      <td>15364.93</td>
      <td>15764.44</td>
      <td>13911.28</td>
      <td>14157.87</td>
      <td>14619.00</td>
      <td>14009.79</td>
    </tr>
  </tbody>
</table>
<p>732 rows × 10 columns</p>
</div>



<ol start="2">
<li> *Hãy viết hàm **`buildModel1(X_train, y_train, X_test, y_test)`** train một LinearRegression dự đoán giá 7 ngày sau bằng giá 10 ngày gần nhất tính đến hiện tại, sử dụng output của bài 1 và của câu trên, để tìm các hệ số $a_i$ trong mô hình AR nêu trên. Hàm nhận các đối số **`X_train, y_train, X_test, y_test`** là một cách chia dữ liệu **`X, y`** thành tập train/test, trong đó **`y`** có dạng như output của bài 1, **`X`** có dạng như output câu trên của bài 2. Hàm trả lại kết quả là một tuple 6 thành phần: 
    <ol>
        <li>Thành phần thứ nhất là model đã được train</li>
    
        <li>Thành phần thứ hai là các hệ số của tổ hợp tuyến tính (một array có số phần tử bằng số cột của **`X`**.</li>
    
        <li>Thành phần thứ ba là hệ số tự do của mô hình (một số thực)</li>
    
        <li>Thành phần thứ tư là kết quả dự đoán **`y`** trên tập **`X_test`** áp dụng mô hình này</li>
    
        <li>Thành phần thứ năm là RMSE của kết quả dự đoán trên tập **`X_test`**, sai khác so với **`y_test`**.</li>
        
        <li>Thành phần thứ sáu là R2 score của kết quả dự đoán trên tập **`X_test`**, sai khác so với **`y_test`**.</li>
    </ol>
    
</li>
</ol>

Đoạn code dưới đây giúp test hàm của bạn. Chú ý rằng trong thực tế, bạn sẽ cần phải tự viết đoạn code test cho phần preprocessing, nên hãy thử viết đoạn code test một cách thành thạo.


```python
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
X1 = getpRecentPrices(current_price, 10)
y1 = getTarget(price_data, SHIFT_NUMBER)
X1 = X1[9: len(X1) - SHIFT_NUMBER] #Bỏ các hàng khuyết dữ liệu: 9 hàng đầu và 7 hàng cuối cùng
y1 = y1[9: len(y1) - SHIFT_NUMBER]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=TEST_SIZE, random_state=0)
model1, coefs1, intercept1, y1_predict, RMSE1, score1 = buildModel1(X1_train, y1_train, X1_test, y1_test)
```


```python
coefs1
```




    array([ 1.17955694, -0.63209418,  0.62283975, -0.62161257,  0.68271052,
           -0.2004747 , -0.13905007,  0.9099273 , -0.35754852, -0.37324748])




```python
intercept1
```




    -2.905756155375002




```python
pd.DataFrame(y1_predict).T #y1_predict là một array, ở đây in ở dạng DataFrame để dễ quan sát
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
      <th>348</th>
      <th>349</th>
      <th>350</th>
      <th>351</th>
      <th>352</th>
      <th>353</th>
      <th>354</th>
      <th>355</th>
      <th>356</th>
      <th>357</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>822.859363</td>
      <td>559.906945</td>
      <td>663.855157</td>
      <td>611.051013</td>
      <td>1308.48615</td>
      <td>19988.100478</td>
      <td>7647.833786</td>
      <td>450.567504</td>
      <td>962.840367</td>
      <td>450.691943</td>
      <td>...</td>
      <td>649.588168</td>
      <td>449.661297</td>
      <td>397.460989</td>
      <td>977.29124</td>
      <td>4955.061842</td>
      <td>457.462637</td>
      <td>1105.466942</td>
      <td>4523.601558</td>
      <td>7812.272819</td>
      <td>609.792923</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 358 columns</p>
</div>




```python
RMSE1
```




    935.109303130576




```python
score1
```




    0.9234069639086381



### Bài 3. Chọn biến quan trọng trong mô hình AR bằng RMSE

Các tính toán ở bài 2 cho thấy:
- R2 tương đối gần với 1, tức phương sai của dự đoán gần với phương sai của quan sát.
- Tuy vậy, root-mean-square error lại vào khoảng 935 (chênh lệch trung bình giữa giá dự đoán và giá quan sát xấp xỉ 1000). Chênh lệch này dễ dàng được quan sát dưới đây:


```python
pd.DataFrame([y1_predict, y1_test])
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
      <th>348</th>
      <th>349</th>
      <th>350</th>
      <th>351</th>
      <th>352</th>
      <th>353</th>
      <th>354</th>
      <th>355</th>
      <th>356</th>
      <th>357</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>822.859363</td>
      <td>559.906945</td>
      <td>663.855157</td>
      <td>611.051013</td>
      <td>1308.48615</td>
      <td>19988.100478</td>
      <td>7647.833786</td>
      <td>450.567504</td>
      <td>962.840367</td>
      <td>450.691943</td>
      <td>...</td>
      <td>649.588168</td>
      <td>449.661297</td>
      <td>397.460989</td>
      <td>977.29124</td>
      <td>4955.061842</td>
      <td>457.462637</td>
      <td>1105.466942</td>
      <td>4523.601558</td>
      <td>7812.272819</td>
      <td>609.792923</td>
    </tr>
    <tr>
      <th>1</th>
      <td>798.650000</td>
      <td>584.500000</td>
      <td>606.710000</td>
      <td>625.670000</td>
      <td>1350.21000</td>
      <td>15600.010000</td>
      <td>6355.130000</td>
      <td>423.510000</td>
      <td>905.990000</td>
      <td>420.040000</td>
      <td>...</td>
      <td>745.220000</td>
      <td>416.000000</td>
      <td>399.280000</td>
      <td>1011.07000</td>
      <td>5679.700000</td>
      <td>462.330000</td>
      <td>1130.010000</td>
      <td>4333.380000</td>
      <td>8717.990000</td>
      <td>622.010000</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 358 columns</p>
</div>



Ta muốn chọn một giá trị $p$ tối ưu trong mô hình AR.

*Hãy viết hàm **`getRMSEList(current_price, future_price, max_p)`** nhận đối số **`current_price`** và **`future_price`** là 2 array ứng với giá hiện tại và giá trong $SHIFT$ ngày như mô tả ở bài 2, **`max_p`** là một số nguyên dương, rồi thực hiện việc train và test với mô hình AR với $p$ chạy từ 1 đến **`max_p`** bằng các linear regressor, cuối cùng trả lại root-mean-square error của $p$ linear regressors dưới dạng một list hay array. Hàm này cũng vẽ đồ thị $RMSE$ theo $p$.*

Đoạn code dưới đây giúp test hàm của bạn. 


```python
%matplotlib inline
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price = getTarget(price_data, SHIFT_NUMBER)
RMSEList = getRMSEList(current_price, future_price, 20)
pd.DataFrame(RMSEList).T # Phần tử 0: RMSE khi chọn p=1, etc.
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
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>787.195736</td>
      <td>873.849764</td>
      <td>867.24436</td>
      <td>913.486686</td>
      <td>736.225409</td>
      <td>759.586408</td>
      <td>771.618451</td>
      <td>964.663182</td>
      <td>865.095662</td>
      <td>935.109303</td>
      <td>1029.863682</td>
      <td>865.220378</td>
      <td>1109.875205</td>
      <td>1149.552349</td>
      <td>954.206685</td>
      <td>1096.882721</td>
      <td>1348.192087</td>
      <td>1028.080528</td>
      <td>1028.104548</td>
      <td>1144.320144</td>
    </tr>
  </tbody>
</table>
</div>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_25_1.png)


Kết quả chạy cho thấy mô hình đề xuất $p=5$, đồng thời cũng cho thấy $p=1, 6, 7$ cho $RMSE$ không khác quá xa với kết quả tối ưu. Tuy nhiên, đề xuất $p=5$ là kết quả của cách chia ngẫu nhiên tập train/test, do đó bạn hoàn toàn có thể gặp kết quả khác (thay random_state trong hàm `train_test_split`). Trong mọi cách chia tập train/test, trường hợp $p=1$ luôn nằm trong nhóm có RMSE nhỏ nhất. Ta có thể chọn $p=1$.

### Bài 4. Chọn biến quan trọng trong mô hình AR bằng Backward Elimination

Bài này đề xuất một phương pháp khác để chọn biến quan trọng, bằng thuật toán Backward Elimination.

1. *Hãy viết hàm **`backwardEliminationOnPrice(current_price, future_price, max_p, significant_level = 0.05)`** thực hiện việc chọn biến quan trọng (significant level) bằng cách một mô hình gồm **`max_p`** giá gần nhất, loại bỏ các biến không quan trọng cho đến khi tất cả các biến đều có p-value không vượt quá **`significant_level`**; rồi trả lại danh sách các biến dưới dạng một list.*


```python
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price = getTarget(price_data, SHIFT_NUMBER)
significant_variables = backwardEliminationOnPrice(current_price, future_price, 20)
significant_variables
```




    [1, 7, 9, 11, 13, 14, 16, 19, 20]



Ở đây, 0 là chỉ số của cột dummy (hệ số tự do), 1 là chỉ số của cột giá hiện tại, ..., 20 là hệ số của giá 19 ngày trước đó. Tương tự như bài 3, khi thay cách chia tập train/test, ta có những list khác nhau và các chỉ số có thể bị thay đổi.

<ol start="2">
<li>*Hãy chỉnh sửa hàm **`backwardEliminationOnPrice(current_price, future_price, max_p, significant_level = 0.05)`** thành hàm **`backwardEliminationOnPrice(current_price, future_price, max_p, significant_level = 0.05, nb_random_states = 1)`** nhận thêm một đối số **`nb_random_states`** là số cách ngẫu nhiên chia dữ liệu thành các tập train/test khác nhau. Ứng với mỗi cách chia ta sẽ có một list chỉ số các biến quan trọng (\[1, 7, 9, 11, 13, 14, 16, 19, 20\]). Hàm này sẽ trả lại list các chỉ số có mặt trong tất cả các kết quả ứng với mọi cách chia train/test.* </li>
</ol>


```python
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price = getTarget(price_data, SHIFT_NUMBER)
significant_variables = backwardEliminationOnPrice(current_price, future_price, 20, significant_level = 0.01, nb_random_states = 20)
significant_variables
```




    [1]



Như vậy, tương tự như cách chọn biến quan trọng ở bài 3 khi xét đến yếu tố ngẫu nhiên của tập train/test, ta thấy chỉ có biến có chỉ số 1 (ứng với giá hiện tại) được chọn trong mọi (hay hầu hết) các cách chia dữ liệu 50% train/50% test ngẫu nhiên. 

## Phần 2. Độ biến thiên của giá BTC

Ở phần 1 ta đã chọn giá của BTC ở 7 ngày sau là target variable (biến cần dự đoán). Thực ra, khi đầu tư, điều ta quan tâm là giá BTC sẽ tăng hay giảm với độ chênh lệch bao nhiêu, tức quan tâm đến $X_{n+1}/X_{n} - 1$ với $X_n$ là giá ở ngày thứ $n$. Thông thường nếu tỉ lệ này nhỏ so với 1, nên ta có thể dùng xấp xỉ $\ln (1+x) \approx x$ và quan tâm đến $\log(X_{n+1}) - \log(X_n)$. Ở BTC, độ tăng giảm này khá lớn, có thể đạt đến 1 trong 1 tuần, nên ta vẫn chọn $X_{n+1} / X_{n} - 1$ là biến cần dự đoán. Ta nhân tỉ số này với 100 để dự đoán số phần trăm tăng giảm.

### Bài 5. Tính target mới
*Hãy viết hàm **`getPriceDiff(data, shift_number)`** nhận đối số **`data`** là một DataFrame có chứa cột "Close" (như dataframe ứng với file **`Data/BTCPrice.csv`**), **`shift_number`** là số ngày trong tương lai ta quan tâm đến giá, và trả lại kết quả là tỉ lệ phần trăm chênh lệch $\Delta_{n, SHIFT} = (X_{n+SHIFT}/X_n - 1) * 100$ giữa ngày đó và hiện tại. Kết quả được trả dưới dạng một list hoặc array có độ dài bằng số hàng của **`data`**, những chỗ khuyết dữ liệu thay bằng 0.*


```python
price_data = readData(PRICE_FILE) # File BTCPrice.csv 
price_diff = getPriceDiff(price_data, SHIFT_NUMBER) # Kết quả là một array
pd.DataFrame(price_diff).T # Biểu diễn ở dạng DataFrame
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
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.059551</td>
      <td>4.352035</td>
      <td>3.526698</td>
      <td>4.117947</td>
      <td>3.185388</td>
      <td>0.559628</td>
      <td>0.717013</td>
      <td>-6.006565</td>
      <td>-20.477137</td>
      <td>-13.79779</td>
      <td>...</td>
      <td>2.356995</td>
      <td>-13.537178</td>
      <td>-1.962654</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 732 columns</p>
</div>



### Bài 6. Mô hình AR cho độ chênh lệch?

Giả sử tỉ lệ phần trăm $\Delta_{n, SHIFT}$ có thể được biểu diễn bằng:

$$
\Delta_{n, SHIFT} = b_0 + \sum_{i=1}^p b_i X_{n-i} + \epsilon_n
$$

*Hãy thực hiện các bước như ở phần 1 để xây dựng một mô hình Linear Regression tìm các hệ số $b_i$ với các $p$ khác nhau và tìm $p$ thích hợp nhất cho mô hình.*

*Hãy viết hàm **`buildModel2(X_train, y_train, X_test, y_test)`** train một LinearRegression dự đoán tỉ lệ phần trăm tăng giá ở $SHIFT$ ngày sau bằng giá $p$ ngày gần nhất tính đến hiện tại, để tìm các hệ số $n_i$ trong mô hình AR nêu trên. Hàm nhận các đối số **`X_train, y_train, X_test, y_test`** là một cách chia dữ liệu **`X, y`** thành tập train/test, trong đó **`y`** có dạng như output của bài 5, **`X`** có dạng như output câu đầu của bài 2. Hàm trả lại kết quả là một tuple 6 thành phần như ở bài 2:*
    
<ol>
<li>Thành phần thứ nhất là model đã được train</li>
<li>Thành phần thứ hai là các hệ số của tổ hợp tuyến tính (một array có số phần tử bằng số cột của **`X`**.</li>
<li>Thành phần thứ ba là hệ số tự do của mô hình (một số thực)</li>
<li>Thành phần thứ tư là kết quả dự đoán **`y`** trên tập **`X_test`** áp dụng mô hình này</li>
<li>Thành phần thứ năm là RMSE của kết quả dự đoán trên tập **`X_test`**, sai khác so với **`y_test`**.</li>
<li>Thành phần thứ sáu là R2 score của kết quả dự đoán trên tập **`X_test`**, sai khác so với **`y_test`**.</li>
</ol>


```python
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X2 = getpRecentPrices(current_price, 10)
X2 = X2[9: len(X2) - SHIFT_NUMBER] #Bỏ các hàng khuyết dữ liệu: 9 hàng đầu và 7 hàng cuối cùng
y2 = future_price_diff[9: len(future_price_diff) - SHIFT_NUMBER]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=TEST_SIZE, random_state=0)
model2, coefs2, intercept2, y1_predict2, RMSE2, score2 = buildModel2(X2_train, y2_train, X2_test, y2_test)
```


```python
coefs2, intercept2
```




    (array([-6.64159095e-05, -5.27816074e-03,  3.24835349e-03, -3.82102995e-03,
             4.39697186e-03,  1.37427035e-04, -2.19771796e-03,  8.72902618e-03,
            -3.39276416e-03, -5.00090892e-04]), 2.4006036430958897)




```python
RMSE2
```




    12.134049361902981



Nếu so sánh với dự đoán ngây thơ: luôn dự đoán $y$ bằng trung bình của $y$ trên tập train, thậm chí ta thấy sai số này còn lớn hơn.


```python
y2_naif = np.array([np.mean(y2_train)] * len(X2_test))
np.sqrt(mean_squared_error(y2_test, y2_naif))
```




    10.997379938806352



**Biểu đồ giá 200 ngày gần nhất**

*Hãy vẽ biểu đồ độ chênh giá bitcoin sau 7 ngày tính cho 200 dữ liệu gần nhất trên **`X (= X_train union X_test`)** cùng với kết quả dự đoán trong mô hình đang sử dụng.


```python
plt.plot(y2[-200:], label="price_diff")
plt.plot(model2.predict(X2)[-200:], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0xcff5748>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_43_1.png)


*Hãy viết hàm **`backwardEliminationOnPriceDiff(current_price, future_price, max_p, significant_level = 0.05)`** thực hiện việc chọn biến quan trọng (significant level) bằng cách một mô hình gồm max_p giá gần nhất, loại bỏ các biến không quan trọng cho đến khi tất cả các biến đều có p-value không vượt quá significant_level; rồi trả lại danh sách các biến dưới dạng một list.*


```python
backwardEliminationOnPriceDiff(current_price, future_price_diff, 20, significant_level = 0.01, nb_random_states = 20)
```




    []



Kết quả cho thấy không có biến nào quan trọng ở mức dộ 0.01, cho phép nhận xét mô hình AR không dự đoán được độ chênh lệch giá.

## Phần 3. Mô hình kĩ thuật

Hai phương pháp thường được sử dụng để dự đoán giá các sản phẩm tài chính là fundamental analysis và technical analysis. Đối với phương pháp kĩ thuật, người ta tính một số chỉ số (indicator) và thông qua chúng để mô tả xu hướng giá trong tương lai.

### Bài 7. Thêm các chỉ số kĩ thuật vào dữ liệu

Để sử dụng các chỉ số kĩ thuật, bạn cần cài đặt thư viện **`statmodels`**. Thư viện này làm việc với các DataFrame có chứa các cột có tên "open", "high", "low", "close", "volume". Do đó, để sử dụng với dữ liệu của chúng ta bạn cần đổi tên các cột.

*1. Hãy viết hàm **`readAsStockDataFrame(filename)`** nhận đối số là đường dẫn đến file dữ liệu có các cột như trong file `Data/BTCPrice.csv` và trả lại kết quả là một DataFrame sao cho tên các cột tương ứng được đổi thành "open", "high", "low", "close", "volume" (volume là khối lượng BTC được giao dịch); và chỉ giữ lại 5 cột này cùng với cột "timestamp".*


```python
data = readAsStockDataFrame(PRICE_FILE)
data.head()
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
      <th>timestamp</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31/12/2015 00:00</td>
      <td>426.09</td>
      <td>433.89</td>
      <td>419.99</td>
      <td>430.89</td>
      <td>6634.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/01/2016 00:00</td>
      <td>430.89</td>
      <td>436.00</td>
      <td>427.20</td>
      <td>433.82</td>
      <td>3788.11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02/01/2016 00:00</td>
      <td>434.87</td>
      <td>435.99</td>
      <td>430.42</td>
      <td>433.55</td>
      <td>2972.06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>03/01/2016 00:00</td>
      <td>433.20</td>
      <td>434.09</td>
      <td>424.06</td>
      <td>431.04</td>
      <td>4571.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04/01/2016 00:00</td>
      <td>431.54</td>
      <td>435.86</td>
      <td>428.44</td>
      <td>434.17</td>
      <td>5717.60</td>
    </tr>
  </tbody>
</table>
</div>



Khi đã có DataFrame trên, áp dụng hàm sau để thêm các cột chỉ số kĩ thuật và chuẩn hoá để đưa tất cả các cột về xấp xỉ phân phối Gaussian (0, 1).


```python
COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'rsv_9', 'kdjk_9', 'kdjj_9', 'macd', 'macds', 'macdh', 'rs_6', 'rsi_6', 'rs_12', 'rsi_12', 'wr_6', 'wr_10', 'cci', 'cci_20', 'tr', 'atr', 'dma', 'high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14', 'pdi_14', 'mdm_14', 'mdi_14', 'dx_14', 'adx', 'adxr', 'trix', 'change', 'vr', 'vr_6_sma']

import stockstats as sts

def addTechnicalIndicators(simple_data):
    stock = sts.StockDataFrame(simple_data)
    stock['cr']
    stock['kdjk']
    stock['kdjd']
    stock['kdjj']
    stock['close_10_sma']
    stock['macd']
    stock['boll']
    stock['rsi_6']
    stock['rsi_12']
    stock['wr_6']
    stock['wr_10']
    stock['cci']
    stock['cci_20']
    stock['tr']
    stock['atr']
    stock['dma']
    stock['adxr']
    stock['close_12_ema']
    stock['trix']
    stock['trix_9_sma']
    stock['vr']
    stock['vr_6_sma']
    new_dataframe = pd.DataFrame(stock).loc[:, COLUMNS]
    transformed_dataframe = new_dataframe.iloc[10: len(new_dataframe) - SHIFT_NUMBER] # Bỏ các hàng khuyết dữ liệu
    scaler = StandardScaler()
    scaler.fit(transformed_dataframe)
    return pd.DataFrame(scaler.transform(transformed_dataframe), columns=COLUMNS)
```

<ol start="2">
<li>*Thêm đoạn gọi hàm trên vào **`readAsStockDataFrame(filename)`** để bổ sung các cột chỉ số kĩ thuật vào DataFrame và chuẩn hoá các cột dữ liệu về trung bình 0, phương sai 1.*</li>
</ol>


```python
data = readAsStockDataFrame(PRICE_FILE)
data.head()
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
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>cr</th>
      <th>cr-ma1</th>
      <th>cr-ma2</th>
      <th>cr-ma3</th>
      <th>rsv_9</th>
      <th>...</th>
      <th>pdi_14</th>
      <th>mdm_14</th>
      <th>mdi_14</th>
      <th>dx_14</th>
      <th>adx</th>
      <th>adxr</th>
      <th>trix</th>
      <th>change</th>
      <th>vr</th>
      <th>vr_6_sma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.557662</td>
      <td>-0.555577</td>
      <td>-0.567160</td>
      <td>-0.558149</td>
      <td>-0.684977</td>
      <td>0.254511</td>
      <td>-0.054549</td>
      <td>2.535630</td>
      <td>9.440731</td>
      <td>-0.223739</td>
      <td>...</td>
      <td>0.512639</td>
      <td>-0.389301</td>
      <td>-0.390922</td>
      <td>-0.017700</td>
      <td>0.761042</td>
      <td>1.047989</td>
      <td>-0.435186</td>
      <td>-0.146963</td>
      <td>-0.961677</td>
      <td>-0.787207</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.557872</td>
      <td>-0.555192</td>
      <td>-0.568139</td>
      <td>-0.558405</td>
      <td>-0.154990</td>
      <td>0.013675</td>
      <td>0.406567</td>
      <td>1.990468</td>
      <td>6.653909</td>
      <td>-0.299411</td>
      <td>...</td>
      <td>0.109721</td>
      <td>-0.386393</td>
      <td>-0.291621</td>
      <td>-0.491732</td>
      <td>0.349655</td>
      <td>0.856236</td>
      <td>-0.416393</td>
      <td>-0.189552</td>
      <td>-1.131634</td>
      <td>-0.838256</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.557937</td>
      <td>-0.556164</td>
      <td>-0.569415</td>
      <td>-0.562664</td>
      <td>-0.298147</td>
      <td>-0.298446</td>
      <td>0.591809</td>
      <td>1.894016</td>
      <td>4.221087</td>
      <td>-1.745237</td>
      <td>...</td>
      <td>-0.216358</td>
      <td>-0.381915</td>
      <td>-0.136020</td>
      <td>-1.063830</td>
      <td>-0.155752</td>
      <td>0.554994</td>
      <td>-0.428084</td>
      <td>-0.902234</td>
      <td>-1.235631</td>
      <td>-0.847340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.562264</td>
      <td>-0.559159</td>
      <td>-0.572859</td>
      <td>-0.563380</td>
      <td>0.127178</td>
      <td>-0.620990</td>
      <td>0.687280</td>
      <td>1.880327</td>
      <td>3.147333</td>
      <td>-1.804377</td>
      <td>...</td>
      <td>-0.477820</td>
      <td>-0.363203</td>
      <td>0.573056</td>
      <td>-0.839978</td>
      <td>-0.428006</td>
      <td>0.253150</td>
      <td>-0.457949</td>
      <td>-0.275354</td>
      <td>-1.344351</td>
      <td>-1.097693</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.562965</td>
      <td>-0.560509</td>
      <td>-0.571985</td>
      <td>-0.564381</td>
      <td>-0.638554</td>
      <td>-0.664228</td>
      <td>0.828384</td>
      <td>1.773829</td>
      <td>2.466648</td>
      <td>-2.103575</td>
      <td>...</td>
      <td>-0.617778</td>
      <td>-0.371092</td>
      <td>0.349826</td>
      <td>-0.839978</td>
      <td>-0.620778</td>
      <td>-0.023577</td>
      <td>-0.499683</td>
      <td>-0.328561</td>
      <td>-1.381465</td>
      <td>-1.223595</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



### Bài 8. Linear Regression trên dữ liệu kĩ thuật

*Lấy $X$ là dữ liệu kĩ thuật output của bài 7 và $y$ là độ chênh lệch giá của 7 ngày sau, viết hàm **`buildModel3(X_train, y_train, X_test, y_test)`** train một LinearRegression theo **`X_train`, `y_train`** rồi dự đoán giá **`X_test`** kiểm tra bằng **`y_test`** *

Lưu ý rằng X có số hàng giảm đi sau khi bỏ các hàng khuyết dữ liệu, nên ở đoạn code sau ta bỏ các hàng tương ứng của $y$. Kiểm tra hàm của bạn bằng đoạn code test.


```python
technical_data = readAsStockDataFrame(PRICE_FILE)
price_data = readData(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X3 = technical_data
y3 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=TEST_SIZE, random_state=0)
model3, coefs3, intercept3, y3_predict, RMSE3, score3 = buildModel3(X3_train, y3_train, X3_test, y3_test)
```


```python
coefs3, intercept3
```




    (array([-8.95687615e+01, -3.30204735e+03,  3.06912711e+03, -4.65246059e+01,
            -2.29734999e+00, -9.18862442e-01, -1.21752036e+00,  2.15990590e+00,
            -2.02055527e+00, -1.87639294e+00, -4.86511827e+00, -6.41731505e-02,
            -1.92423252e+00, -2.93017689e+00,  3.36614062e+00,  1.39141887e+00,
             1.39077432e+01, -1.07249119e+00, -6.10122770e+00,  3.10729644e+00,
            -2.09425508e+00,  3.55980511e+00, -3.80140550e+00,  4.58495658e+02,
            -4.45405840e+00, -8.05719310e+00, -8.60893298e+00,  3.18297846e+01,
            -9.61027916e-01, -2.48817087e+00, -2.92514352e+01, -2.49799093e+00,
             1.30093413e-01,  3.25346436e+00,  1.53855039e+00,  2.35884068e+00,
            -4.33218512e+00,  3.51929872e+00,  2.71819164e+00, -1.35978671e+00,
            -2.96215545e-01,  1.72408476e+00]), 4.375715086380031)




```python
RMSE3
```




    10.177908401176442




```python
score3
```




    0.11700720586575208



*Vẽ đồ thị chênh lệch giá tuần của 200 ngày gần nhất theo thực tế và dự đoán với mô hình 3.*


```python
plt.plot(y3[-200:], label="price_diff")
plt.plot(model3.predict(X3)[-200:], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0x14cf35f8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_59_1.png)


### Bài 9. Giả sử mô hình đang overfitting...

Bài này minh hoạ việc sử dụng feature selection trên một mô hình không overfitting sẽ nhìn chung không cải thiện được kết quả.

1. *Giả sử mô hình đang overfitting, dùng backward elimination để chọn các biến kĩ thuật quan trọng trong mô hình trên. Bạn có thể viết hàm **`backwardEliminationOnTechnicalData`** tương tự như ở phần 1, 2 để chọn các biến quan trọng theo một ngưỡng xác định trước. Chú ý rằng khi cách chia train/test thay đổi, sẽ không có biến nào xuất hiện trong tất cả các kết quả. Bạn có thể chọn những biến xuất hiện trong nhiều kết quả nhất. Ví dụ, cho 30 phép chia train/test và chọn các biến xuất hiện trong >= 20 kết quả.*

Kiểm tra kết quả bằng đoạn code sau (các biến lựa chọn có thể khác)


```python
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
variable_selection = backwardEliminationOnTechnicalData(technical_data, price_diff, 0.1, 50)
variable_selection
```




    [0, 1, 8, 9, 11, 16, 17, 22, 27, 40]



Dưới đây, hàm **`getColumnNameFromIndices(indices)`** được viết sẵn trả lại tên các cột tương ứng với chỉ số. Nó cho phép người khảo sát hiểu các biến có ý nghĩa gì.


```python
REFINED_COLUMNS = getColumnNameFromIndices(variable_selection)
REFINED_COLUMNS
```




    ['vr_6_sma',
     'open',
     'cr-ma2',
     'cr-ma3',
     'kdjk_9',
     'rs_6',
     'rsi_6',
     'cci',
     'high_delta',
     'change']



<ol start="2">
<li>*Dựa trên kết quả chọn biến, viết hàm **`buildModel4(X_train, y_train, X_test, y_test)`** train lại model LinearRegression nhưng chỉ trên các biến đã được chọn và tính các hệ số, RMSE trên tập test.*</li>
</ol>


```python
technical_data = readAsStockDataFrame(PRICE_FILE)
REFINED_COLUMNS = getColumnNameFromIndices(variable_selection)
refined_technical_data = technical_data.loc[:, REFINED_COLUMNS]
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X4 = refined_technical_data
y4 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=TEST_SIZE, random_state=0)
model4, coefs4, intercept4, y4_predict, RMSE4, score4 = buildModel4(X4_train, y4_train, X4_test, y4_test)
```


```python
coefs4, intercept4
```




    (array([-1.50081231,  1.80297088,  1.36029255, -1.87375089, -3.38263938,
             1.20351048,  3.28764352,  0.82167238, -1.4776895 , -0.60175674]),
     4.148188742602574)




```python
RMSE4
```




    10.546203009098091



Như vậy, việc chọn biến quan trọng không cho phép giảm sai số. Điều này minh hoạ việc mô hình đang không overfitting. Bạn có thể so sánh với sai số trên tập train và thấy chúng có cùng mức độ sai số. Sai số lớn như nhau trên cả tập train và test chứng tỏ mô hình đang không overfitting.


```python
y4_predict_train = model4.predict(X4_train)
np.sqrt(mean_squared_error(y4_train, y4_predict_train))
```




    10.211131942337394



### Bài 10. Polynomial Regression

*Lấy $X$ là dữ liệu kĩ thuật output của bài 7 và $y$ là độ chênh lệch giá của 7 ngày sau,  viết hàm **`buildModel5(X_train, y_train, X_test, y_test)`** sử dụng Polynomial Regression bậc 2 để xây dựng mô hình dự đoán $y$ từ $X$. Tính RMSE, R2 score với mô hình này trên tập test. So sánh với sai số trên tập train để phát hiện mô hình có đang overfitting không. Vẽ đồ thị giá 200 ngày gần nhất (thực tế, dự đoán) để minh hoạ.*


```python
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X5 = technical_data[:]
y5 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=TEST_SIZE, random_state=0)
model5, coefs5, intercept5, y5_predict, RMSE5, score5 = buildModel5(X5_train, y5_train, X5_test, y5_test)
```


```python
RMSE5
```




    86.93831016933417



RMSE quá cao, và sự bất thường của đồ thị đường dự đoán cho phép nhận xét mô hình đang overfitting. Thật vậy, nếu sử dụng tập train, sai số sẽ nhỏ hơn nhiều, xấp xỉ 0. Như vậy, mô hình bị overfitting trên tập train đang sử dụng.


```python
# Đoạn code này ví dụ việc tính RMSE trên tập train. Đoạn code của bạn có thể khác
y5_predict_train = model5.predict(PolynomialFeatures(2).fit_transform(X5_train))
np.sqrt(mean_squared_error(y5_train, y5_predict_train))
```




    5.452984231312599e-13



Đồ thị minh hoạ: ta thấy sự cao bất thường của đường dự đoán ở đoạn dữ liệu 190-200.


```python
plt.plot(y5[-200:], label="price_diff")
plt.plot(model5.predict(PolynomialFeatures(2).fit_transform(X5))[-200:], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0x14fceeb8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_77_1.png)


Thậm chí nếu giảm size của tập test, sai số giữa tập train và tập test thậm chí còn có khoảng cách lớn hơn


```python
TEST_SIZE_2 = 0.1
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=TEST_SIZE_2, random_state=0)
model5, coefs5, intercept5, y5_predict, RMSE5, score5 = buildModel5(X5_train, y5_train, X5_test, y5_test)
print "Error on test set: ", np.sqrt(mean_squared_error(y5_test, y5_predict))
y5_predict_train = model5.predict(PolynomialFeatures(2).fit_transform(X5_train))
print "Error on training set: ", np.sqrt(mean_squared_error(y5_train, y5_predict_train))
```

    Error on test set:  205.23785803532968
    Error on training set:  5.8461524435091435e-12
    

Mô hình đang overfitting do đó việc giảm số biến là cần thiết. Thay vì sử dụng backward elimination, ta có thể sử dụng Lasso.

### Bài 11. Lasso

*Lấy $X$ là dữ liệu kĩ thuật output của bài 7 và $y$ là độ chênh lệch giá của 7 ngày sau, viết hàm **`buildModel6(X_train, y_train, X_test, y_test, alpha)`** sử dụng Lasso trên Polynomial Regression bậc 2 để xây dựng mô hình dự đoán $y$ từ $X$, **`alpha`** là hệ số phạt. Vẫn trả kết quả ở dạng bộ 6 phần tử như các mô hình trước. Chọn hệ số phạt $\alpha$ tốt nhất cho cách chọn train/test hiện tại. Tính RMSE, R2 score với mô hình này trên tập test. Vẽ đồ thị giá 200 ngày gần nhất (thực tế, dự đoán) để minh hoạ. Cẩn thận: khi dùng Lasso, chú ý số bước lặp để bài toán optimization hội tụ.*


```python
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X6 = technical_data[:]
y6 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=TEST_SIZE, random_state=0)
alpha6 = 0.11
model6, coefs6, intercept6, y6_predict, RMSE6, score6 = buildModel6(X6_train, y6_train, X6_test, y6_test, alpha6)
RMSE6
```

    Nb_iterations used:  12313
    




    9.342546946384736



Các biến quan trọng (có 104 biến có hệ số lớn hơn 0.01)


```python
important_variables = filter(lambda i: abs(coefs6[i]) > 1e-2, range(len(model6.coef_)))
pd.DataFrame(important_variables).T
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
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>9</td>
      <td>11</td>
      <td>12</td>
      <td>33</td>
      <td>35</td>
      <td>41</td>
      <td>42</td>
      <td>126</td>
      <td>128</td>
      <td>...</td>
      <td>930</td>
      <td>931</td>
      <td>932</td>
      <td>934</td>
      <td>935</td>
      <td>936</td>
      <td>937</td>
      <td>939</td>
      <td>940</td>
      <td>941</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 103 columns</p>
</div>



*Hãy tìm lại ý nghĩa các biến này bằng cách viết hàm **`indToNameOnPolynomialRegressionDegree2(initial_columns, selected_indices)`** nhận đối số **`initial_columns`** là list tên các cột ban đầu (có dạng như `COLUMNS`), và **`selected_indices`** là list chỉ số các biến được chọn (có dạng như `important_variables` trong ví dụ trên), trả lại một từ điển ứng mỗi chỉ số ý nghĩa biến (chỉ số đó chỉ biến nào, hoặc chỉ tích của 2 biến nào trong đa thức bậc 2)*


```python
important_variables_dict = indToNameOnPolynomialRegressionDegree2(COLUMNS, important_variables) # Một từ điển
pd.DataFrame([important_variables_dict.keys(), important_variables_dict.values()]).T
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
      <td>5</td>
      <td>1 * volume</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>1 * cr-ma3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>1 * kdjk_9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>1 * kdjj_9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>526</td>
      <td>macds * low_delta</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>1 * pdi_14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>35</td>
      <td>1 * mdi_14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>550</td>
      <td>macdh * atr</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>1 * vr</td>
    </tr>
    <tr>
      <th>9</th>
      <td>42</td>
      <td>1 * vr_6_sma</td>
    </tr>
    <tr>
      <th>10</th>
      <td>559</td>
      <td>macdh * mdm_14</td>
    </tr>
    <tr>
      <th>11</th>
      <td>563</td>
      <td>macdh * adxr</td>
    </tr>
    <tr>
      <th>12</th>
      <td>568</td>
      <td>rs_6 * rs_6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>578</td>
      <td>rs_6 * dma</td>
    </tr>
    <tr>
      <th>14</th>
      <td>580</td>
      <td>rs_6 * um</td>
    </tr>
    <tr>
      <th>15</th>
      <td>126</td>
      <td>low * low</td>
    </tr>
    <tr>
      <th>16</th>
      <td>128</td>
      <td>low * volume</td>
    </tr>
    <tr>
      <th>17</th>
      <td>641</td>
      <td>rs_12 * adxr</td>
    </tr>
    <tr>
      <th>18</th>
      <td>875</td>
      <td>pdm * adxr</td>
    </tr>
    <tr>
      <th>19</th>
      <td>132</td>
      <td>low * cr-ma3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>646</td>
      <td>rsi_12 * rsi_12</td>
    </tr>
    <tr>
      <th>21</th>
      <td>791</td>
      <td>atr * vr</td>
    </tr>
    <tr>
      <th>22</th>
      <td>145</td>
      <td>low * cci</td>
    </tr>
    <tr>
      <th>23</th>
      <td>663</td>
      <td>rsi_12 * dx_14</td>
    </tr>
    <tr>
      <th>24</th>
      <td>665</td>
      <td>rsi_12 * adxr</td>
    </tr>
    <tr>
      <th>25</th>
      <td>670</td>
      <td>wr_6 * wr_6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>671</td>
      <td>wr_6 * wr_10</td>
    </tr>
    <tr>
      <th>27</th>
      <td>676</td>
      <td>wr_6 * dma</td>
    </tr>
    <tr>
      <th>28</th>
      <td>165</td>
      <td>low * vr_6_sma</td>
    </tr>
    <tr>
      <th>29</th>
      <td>684</td>
      <td>wr_6 * mdm_14</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>865</td>
      <td>dm * change</td>
    </tr>
    <tr>
      <th>74</th>
      <td>357</td>
      <td>cr-ma3 * macdh</td>
    </tr>
    <tr>
      <th>75</th>
      <td>363</td>
      <td>cr-ma3 * wr_10</td>
    </tr>
    <tr>
      <th>76</th>
      <td>364</td>
      <td>cr-ma3 * cci</td>
    </tr>
    <tr>
      <th>77</th>
      <td>377</td>
      <td>cr-ma3 * mdi_14</td>
    </tr>
    <tr>
      <th>78</th>
      <td>378</td>
      <td>cr-ma3 * dx_14</td>
    </tr>
    <tr>
      <th>79</th>
      <td>383</td>
      <td>cr-ma3 * vr</td>
    </tr>
    <tr>
      <th>80</th>
      <td>896</td>
      <td>pdi_14 * adxr</td>
    </tr>
    <tr>
      <th>81</th>
      <td>900</td>
      <td>pdi_14 * vr_6_sma</td>
    </tr>
    <tr>
      <th>82</th>
      <td>905</td>
      <td>mdm_14 * adxr</td>
    </tr>
    <tr>
      <th>83</th>
      <td>910</td>
      <td>mdi_14 * mdi_14</td>
    </tr>
    <tr>
      <th>84</th>
      <td>911</td>
      <td>mdi_14 * dx_14</td>
    </tr>
    <tr>
      <th>85</th>
      <td>924</td>
      <td>dx_14 * vr_6_sma</td>
    </tr>
    <tr>
      <th>86</th>
      <td>927</td>
      <td>adx * trix</td>
    </tr>
    <tr>
      <th>87</th>
      <td>928</td>
      <td>adx * change</td>
    </tr>
    <tr>
      <th>88</th>
      <td>418</td>
      <td>kdjk_9 * kdjk_9</td>
    </tr>
    <tr>
      <th>89</th>
      <td>931</td>
      <td>adxr * adxr</td>
    </tr>
    <tr>
      <th>90</th>
      <td>932</td>
      <td>adxr * trix</td>
    </tr>
    <tr>
      <th>91</th>
      <td>934</td>
      <td>adxr * vr</td>
    </tr>
    <tr>
      <th>92</th>
      <td>935</td>
      <td>adxr * vr_6_sma</td>
    </tr>
    <tr>
      <th>93</th>
      <td>936</td>
      <td>trix * trix</td>
    </tr>
    <tr>
      <th>94</th>
      <td>937</td>
      <td>trix * change</td>
    </tr>
    <tr>
      <th>95</th>
      <td>939</td>
      <td>trix * vr_6_sma</td>
    </tr>
    <tr>
      <th>96</th>
      <td>940</td>
      <td>change * change</td>
    </tr>
    <tr>
      <th>97</th>
      <td>941</td>
      <td>change * vr</td>
    </tr>
    <tr>
      <th>98</th>
      <td>443</td>
      <td>kdjk_9 * dx_14</td>
    </tr>
    <tr>
      <th>99</th>
      <td>445</td>
      <td>kdjk_9 * adxr</td>
    </tr>
    <tr>
      <th>100</th>
      <td>450</td>
      <td>kdjj_9 * kdjj_9</td>
    </tr>
    <tr>
      <th>101</th>
      <td>930</td>
      <td>adx * vr_6_sma</td>
    </tr>
    <tr>
      <th>102</th>
      <td>472</td>
      <td>kdjj_9 * mdm_14</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 2 columns</p>
</div>



Sau khi hàm trên được viết, ta có thể in ra dạng "dễ đọc" của mô hình như sau:


```python
def getModelReadableForm(coefs, important_variables_dict):
    S = ""
    for k, V in sorted(important_variables_dict.items(), key = lambda x: -abs(coefs[x[0]])):
        S += " + (%.2f) * %s " % (coefs[k], V)
    return S

getModelReadableForm(model6.coef_, important_variables_dict)
```




    ' + (5.60) * low * low  + (-4.34) * macdh * adxr  + (-3.80) * dma * dma  + (-3.79) * low * cr-ma3  + (-3.42) * kdjk_9 * adxr  + (3.02) * cr-ma1 * tr  + (-2.61) * low * cci  + (-2.58) * 1 * kdjj_9  + (2.43) * cci_20 * trix  + (2.42) * adxr * vr_6_sma  + (2.40) * rsi_12 * adxr  + (-2.26) * pdi_14 * vr_6_sma  + (-2.00) * atr * vr  + (-2.00) * 1 * cr-ma3  + (1.93) * trix * trix  + (1.86) * adxr * vr  + (-1.80) * cr * vr_6_sma  + (1.71) * mdm_14 * adxr  + (-1.70) * 1 * kdjk_9  + (-1.66) * low * vr_6_sma  + (1.65) * cr-ma3 * vr  + (-1.60) * volume * kdjk_9  + (-1.56) * 1 * mdi_14  + (-1.56) * adx * change  + (-1.51) * dma * vr  + (-1.45) * volume * high_delta  + (-1.41) * volume * mdi_14  + (1.38) * 1 * pdi_14  + (-1.36) * cr * adxr  + (1.31) * rsi_12 * rsi_12  + (-1.27) * wr_6 * dma  + (-1.26) * pdi_14 * adxr  + (1.20) * 1 * vr_6_sma  + (1.12) * rs_6 * dma  + (1.11) * mdi_14 * dx_14  + (1.04) * cr-ma3 * wr_10  + (-1.04) * cr-ma2 * adxr  + (0.99) * rsi_12 * dx_14  + (0.96) * adxr * adxr  + (0.88) * tr * tr  + (-0.86) * kdjj_9 * mdm_14  + (-0.86) * cr-ma1 * cr-ma1  + (0.86) * cr-ma2 * change  + (0.82) * cr-ma2 * rs_6  + (0.81) * wr_6 * vr_6_sma  + (0.79) * low_delta * dx_14  + (-0.71) * 1 * volume  + (0.71) * wr_6 * wr_6  + (0.69) * high_delta * adxr  + (-0.68) * cr * macdh  + (-0.66) * adx * trix  + (-0.65) * wr_6 * mdi_14  + (0.63) * rs_12 * adxr  + (0.58) * volume * tr  + (-0.57) * high_delta * pdm_14  + (0.57) * cr * wr_6  + (0.57) * change * change  + (0.53) * wr_10 * vr  + (0.52) * macdh * atr  + (-0.52) * wr_10 * wr_10  + (-0.47) * volume * cr-ma2  + (-0.45) * cr-ma1 * adx  + (0.45) * cr-ma3 * macdh  + (0.42) * cr-ma3 * dx_14  + (0.41) * dx_14 * vr_6_sma  + (0.41) * cr-ma2 * cr-ma2  + (-0.40) * dm * dm  + (-0.40) * cr * adx  + (0.37) * wr_10 * pdm  + (-0.37) * low * volume  + (0.36) * cr * cci_20  + (0.35) * volume * volume  + (0.35) * trix * change  + (0.34) * cci_20 * dx_14  + (0.28) * adx * vr_6_sma  + (-0.27) * rs_6 * um  + (-0.26) * low_delta * change  + (-0.24) * kdjk_9 * dx_14  + (0.24) * volume * wr_6  + (0.22) * cr-ma3 * mdi_14  + (-0.21) * cr-ma3 * cci  + (0.21) * cci * cci  + (0.20) * wr_6 * adxr  + (-0.19) * cr-ma3 * cr-ma3  + (-0.19) * wr_6 * wr_10  + (0.16) * high_delta * high_delta  + (0.14) * macdh * mdm_14  + (0.14) * 1 * vr  + (0.13) * kdjk_9 * kdjk_9  + (-0.13) * cr-ma1 * mdi_14  + (-0.13) * adxr * trix  + (0.12) * volume * pdi_14  + (0.10) * wr_6 * mdm_14  + (0.08) * mdi_14 * mdi_14  + (0.06) * dm * change  + (0.06) * kdjj_9 * kdjj_9  + (0.06) * pdm * adxr  + (-0.05) * change * vr  + (0.05) * high_delta * adx  + (-0.04) * macds * low_delta  + (-0.03) * trix * vr_6_sma  + (0.03) * rs_6 * rs_6  + (0.03) * cci * pdi_14 '



Kết quả dự đoán cho 200 ngày gần nhất:


```python
plt.plot(y6[-200:], label="price_diff")
plt.plot(model6.predict(PolynomialFeatures(2).fit_transform(X6)[-200:]), label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0xd175278>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_90_1.png)


Như vậy, đến giờ ta đã đưa ra một mô hình mà sai số dự đoán trung bình (đo bằng RMSE) giảm từ 12.13% xuống 9.34%. Sai số này nhỏ hơn sai số của mô hình ngây thơ (luôn đoán độ tăng giá bằng độ tăng trung bình trên tập train)

## Phần 4 - Mô hình dựa vào thông tin

### Bài 12. Tính các chỉ số thông tin

Bây giờ ta dựa vào hai nguồn thông tin mới là google trend (xu hướng tìm kiếm) từ khoá "bitcoin" trên google theo thời gian và dữ liệu một số thông tin tích cực và tiêu cực liên quan đến bitcoin. File `Data/BTCTrend.csv` cho ta chỉ số tìm kiếm theo ngày, các chỉ số này tỉ lệ thuận với lượng tìm kiếm trên google. File `Data/BTCNews.csv` cho ta các thông tin sắp xếp theo thời gian. Ta chỉ quan tâm đến cột cuối cùng: 1 cho biết thông tin là tích cực, có khả năng khiến BTC tăng giá; -1 là tiêu cực, có khả năng khiến BTC giảm giá. Ngày xảy ra các sự kiện này không trùng nhau.

Gọi $X_n$ là giá của bitcoin tại ngày thứ $n$, $T_n$ là chỉ số xu hướng tìm kiếm ngày đó. Gọi $s_m$ là dấu của ngày xảy ra sự kiện, bằng 1 hoặc -1 nếu ngày đó xảy ra sự kiện tích cực hay tiêu cực, bằng 0 nếu không xảy ra sự kiện gì. Gọi $S$ là tập hợp tất cả các ngày có xảy ra một sự kiện tích cực hoặc tiêu cực. Ta xây dựng thêm một chỉ số, được gọi là hệ số thông tin của ngày, chỉ số này được tính như sau:

$$
H_n = \sum_{m \in S, n \geq m} \exp(-\alpha/(n - m)) * s_m
$$

Công thức này có nghĩa là: hệ số thông tin của một ngày bằng tổng độ ảnh hưởng của các thông tin diễn ra trước đó. Độ ảnh hưởng của thông tin giảm theo hàm mũ trên số ngày đã diễn ra thông tin: nếu thông tin diễn ra hôm nay, nó có độ ảnh hưởng bằng 1 hoặc -1, nếu diễn ra hôm qua, nó có độ ảnh hưởng $\pm e^{-\alpha}$; nếu diễn ra hôm trước, nó có độ ảnh hưởng bằng $$.

*Hãy viết hàm **`getTCoefficient(price_data, trend_data)`** nhận đối số là 2 DataFrame **`price_data`** đọc từ `Data/BTCPrice.csv` và **`trend_data`** đọc từ `Data/BTCTrend.csv`* và trả lại $T_n$, một list hay array có độ dài bằng độ dài **`price_data`**, tính có các ngày tương ứng với các ngày trong **`price_data`**. Hãy chuẩn hoá kết quả này bằng biến đổi tuyến tính để có trung bình 0, phương sai 1. (Khi chuẩn hoá, bỏ qua ngày 31/12/2015 vì không có dữ liệu, chỉ tính trung bình và độ lệch chuẩn trên những ngày còn lại)*

*Hãy viết hàm **`getHCoefficient(price_data, news_data, alpha)`** nhận đối số là 2 DataFrame **`price_data`** đọc từ `Data/BTCPrice.csv`, **`news_data`** đọc từ `Data/BTCNews.csv` cùng một đối số **`alpha`** là một số thực dương; và trả lại $H_n$ như mô tả trên, dưới dạng một list hay array có độ dài bằng độ dài **`price_data`**, tính có các ngày tương ứng với các ngày trong **`price_data`**. Hãy chuẩn hoá kết quả này để có trung bình 0, phương sai 1.*

*Hãy viết hàm **`getLogTCoefficient(price_data, trend_data)`** nhận đối số là 2 DataFrame **`price_data`** đọc từ `Data/BTCPrice.csv` và **`trend_data`** đọc từ `Data/BTCTrend.csv` và trả lại $\ln T_n$, một list hay array có độ dài bằng độ dài **`price_data`**, tính có các ngày tương ứng với các ngày trong **`price_data`**. Hãy chuẩn hoá kết quả này bằng biến đổi tuyến tính để có trung bình 0, phương sai 1. (Khi chuẩn hoá, bỏ qua ngày 31/12/2015 vì không có dữ liệu, chỉ tính trung bình và độ lệch chuẩn trên những ngày còn lại)*



```python
trend_data = readData(TREND_FILE)
trend_data.head()
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
      <th>Timestamp</th>
      <th>Trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/01/2016</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02/01/2016</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03/01/2016</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04/01/2016</td>
      <td>65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05/01/2016</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>




```python
news_data = readData(EVENT_FILE)
news_data.head()
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
      <th>Timestamp</th>
      <th>Title</th>
      <th>Reference</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11/01/2016</td>
      <td>The US-based exchange Cryptsy declared bankrup...</td>
      <td>"Cryptsy CEO Stole Millions From Exchange, Cou...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13/01/2016</td>
      <td>Mike Hearn Quits Bitcoin (a.k.a The Hearnia) -...</td>
      <td>https://www.cryptocoinsnews.com/mike-hearn-say...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16/01/2016</td>
      <td>� Le�Bitcoin�est un �chec �. C'est un de ses d...</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20/01/2016</td>
      <td>Bitcoin : les Pays-Bas arr�tent 10 hommes susp...</td>
      <td>http://www.clubic.com/antivirus-securite-infor...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17/02/2016</td>
      <td>IBM wants to move blockchain tech beyond Bitco...</td>
      <td>https://arstechnica.com/information-technology...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
price_data = readData(PRICE_FILE)
trend_data = readData(TREND_FILE)
TCoefficient = getTCoefficient(price_data, trend_data)
pd.DataFrame(TCoefficient).T 
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
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.607179</td>
      <td>-0.459056</td>
      <td>-0.46611</td>
      <td>-0.46611</td>
      <td>-0.454354</td>
      <td>-0.4473</td>
      <td>-0.456705</td>
      <td>-0.428491</td>
      <td>-0.423789</td>
      <td>-0.452003</td>
      <td>...</td>
      <td>8.463554</td>
      <td>5.01677</td>
      <td>3.474416</td>
      <td>3.112339</td>
      <td>3.020644</td>
      <td>2.839606</td>
      <td>2.750262</td>
      <td>2.477529</td>
      <td>2.385834</td>
      <td>1.751024</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 732 columns</p>
</div>




```python
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
pd.DataFrame(HCoefficient).T # Các giá trị đầu tiên bằng nhau vì không có sự kiện
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
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>...</td>
      <td>-0.278343</td>
      <td>-0.171978</td>
      <td>-0.118796</td>
      <td>-0.092204</td>
      <td>-0.078909</td>
      <td>-3.487617</td>
      <td>-1.776615</td>
      <td>-0.921114</td>
      <td>-0.493364</td>
      <td>-0.279488</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 732 columns</p>
</div>




```python
# kiểm tra ngày có sự kiện đầu tiên là ngày 11/1 (ngày có chỉ số 11, vì ngày 31/12/2015 có chỉ số 0)
pd.DataFrame(HCoefficient).head(15).T
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
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-0.065613</td>
      <td>-3.480969</td>
      <td>-1.773291</td>
      <td>-4.334808</td>
      <td>-2.20021</td>
    </tr>
  </tbody>
</table>
</div>




```python
price_data = readData(PRICE_FILE)
trend_data = readData(TREND_FILE)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
pd.DataFrame(logTCoefficient).T 
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
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-5.417552</td>
      <td>-0.913431</td>
      <td>-0.966472</td>
      <td>-0.966472</td>
      <td>-0.879455</td>
      <td>-0.830403</td>
      <td>-0.89631</td>
      <td>-0.709487</td>
      <td>-0.681248</td>
      <td>-0.862857</td>
      <td>...</td>
      <td>3.559854</td>
      <td>3.040186</td>
      <td>2.691711</td>
      <td>2.590723</td>
      <td>2.563587</td>
      <td>2.507936</td>
      <td>2.479385</td>
      <td>2.387281</td>
      <td>2.354475</td>
      <td>2.095325</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 732 columns</p>
</div>



### Bài 13. Mô hình dựa vào thông tin

Ta dự đoán tỉ lệ phần trăm tăng giá BTC sau 7 ngày là một đa thức theo các biến $H_n, T_n, \ln T_n$. 

*Lấy $X$ là dữ liệu tạo thành từ 3 cột $H_n$, $T_n$, $\ln T_n$, và $y$ là độ chênh lệch giá của 7 ngày sau, viết hàm **`buildModel7(X_train, y_train, X_test, y_test)`** xây dựng Polynomial Regression bậc 2 dự đoán $y$ từ $X$, vẫn trả kết quả ở dạng bộ 6 phần tử như các mô hình trước. Tính RMSE, vẽ đồ thị giá 200 ngày gần nhất.*


```python
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
trend_data = readData(TREND_FILE)
TCoefficient = getTCoefficient(price_data, trend_data)
TCoefficient = np.array(TCoefficient).reshape(len(TCoefficient), 1)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
HCoefficient = np.array(HCoefficient).reshape(len(HCoefficient), 1)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
logTCoefficient = np.array(logTCoefficient).reshape(len(logTCoefficient), 1)
information_data = np.concatenate([TCoefficient, logTCoefficient, HCoefficient], axis = 1)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X7 = information_data[10: len(price_diff) - SHIFT_NUMBER] #Lấy từ ngày đầu tiên có thông tin
y7 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=TEST_SIZE, random_state=0)

model7, coefs7, intercept7, y7_predict, RMSE7, score7 = buildModel7(X7_train, y7_train, X7_test, y7_test)
RMSE7
```




    10.544458029012162




```python
plt.plot(y7[-200:], label="price_diff")
plt.plot(model7.predict(PolynomialFeatures(2).fit_transform(X7))[-200:], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0x15aea6d8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_101_1.png)



```python
coefs7
```




    array([ 7.00311546e+01,  2.50207816e+02, -9.46881928e+01,  1.65967921e+00,
            4.41411437e+00, -6.49611002e+01, -2.79800934e-01, -1.60091536e+01,
            2.22393493e-01, -4.61889215e-01])



Sử dụng mô hình thông tin với cách xử lí của ta cho kết quả tốt hơn nhẹ so với mô hình ngây thơ, nhưng kém mô hình kĩ thuật.

## Phần 5 - Mô hình kết hợp

### Bài 14. Kết hợp mô hình kĩ thuật và mô hình thông tin

Ta dự đoán tỉ lệ phần trăm tăng giá BTC sau 7 ngày là một đa thức bậc 2 theo cả các biến kĩ thuật trong phần 3 và 3 biến thông tin ở cuối bài 13.

*Lấy $X$ là dữ liệu tạo thành từ tất cả các cột trong mô hình kĩ thuật và thêm 3 cột $H_n$, $T_n$, $\ln T_n$, $y$ là độ chênh lệch giá của 7 ngày sau, viết hàm **`buildModel8(X_train, y_train, X_test, y_test)`** xây dựng Polynomial Regression bậc 2 dự đoán $y$ từ $X$, vẫn trả kết quả ở dạng bộ 6 phần tử như các mô hình trước. Tính RMSE, vẽ đồ thị giá 200 ngày gần nhất.*

Đoạn code dưới đây dùng để test. Nhắc lại, bạn cần có khả năng tự viết lại đoạn code này.


```python
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
trend_data = readData(TREND_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)

TCoefficient = getTCoefficient(price_data, trend_data)
TCoefficient = np.array(TCoefficient).reshape(len(TCoefficient), 1)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
HCoefficient = np.array(HCoefficient).reshape(len(HCoefficient), 1)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
logTCoefficient = np.array(logTCoefficient).reshape(len(logTCoefficient), 1)
information_data = np.concatenate([TCoefficient, logTCoefficient, HCoefficient], axis = 1)
X8_info = information_data[10: len(price_diff) - SHIFT_NUMBER] #Lấy từ ngày đầu tiên có thông tin

technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X8_tech = technical_data[:]

X8 = np.concatenate([X8_info, X8_tech], axis = 1)

FULL_COLUMNS = COLUMNS + ["T_n", "Log_T_n", "H_n"]

y8 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=TEST_SIZE, random_state=0)
model8, coefs8, intercept8, y8_predict, RMSE8, score8 = buildModel8(X8_train, y8_train, X8_test, y8_test)
```


```python
RMSE8
```




    76.542897674416




```python
plt.plot(y8[-200:], label="price_diff")
plt.plot(model8.predict(PolynomialFeatures(2).fit_transform(X8))[-200:], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0x17c48748>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_108_1.png)


RMSE lớn và khác xa so với tập train chứng tỏ mô hình overfit trên tập train. Ta dùng Lasso để điều hoà.

*Lấy $X$ là dữ liệu tạo thành từ tất cả các cột trong mô hình kĩ thuật và thêm 3 cột $H_n$, $T_n$, $\ln T_n$, $y$ là độ chênh lệch giá của 7 ngày sau, viết hàm **`buildModel9(X_train, y_train, X_test, y_test, alpha)`** xây dựng mô hình Lasso trên Polynomial Regression bậc 2 dự đoán $y$ từ $X$, vẫn trả kết quả ở dạng bộ 6 phần tử như các mô hình trước. Chọn $\alpha$ tối ưu cho tập test. Tính RMSE, vẽ đồ thị giá 200 ngày gần nhất.*


```python
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
trend_data = readData(TREND_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)

TCoefficient = getTCoefficient(price_data, trend_data)
TCoefficient = np.array(TCoefficient).reshape(len(TCoefficient), 1)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
HCoefficient = np.array(HCoefficient).reshape(len(HCoefficient), 1)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
logTCoefficient = np.array(logTCoefficient).reshape(len(logTCoefficient), 1)
information_data = np.concatenate([TCoefficient, logTCoefficient, HCoefficient], axis = 1)
X9_info = information_data[10: len(price_diff) - SHIFT_NUMBER] 

technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X9_tech = technical_data

X9 = np.concatenate([X9_tech, X9_info], axis = 1)

FULL_COLUMNS = COLUMNS + ["T_n", "Log_T_n", "H_n"]

y9 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, test_size=TEST_SIZE, random_state=0)

model9, coefs9, intercept9, y9_predict, RMSE9, score9 = buildModel9(X9_train, y9_train, X9_test, y9_test, 0.3)
RMSE9
```

    Nb_iterations used:  6021
    




    9.199798639276453



Cuối cùng, bằng cách thay đổi giá trị của `ALPHA` (hệ số giảm ảnh hưởng của thông tin) và alpha (hệ số phạt trong Lasso), ta có thể cải thiện nhẹ mô hình.


```python
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
trend_data = readData(TREND_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)

ALPHA = 0.1

TCoefficient = getTCoefficient(price_data, trend_data)
TCoefficient = np.array(TCoefficient).reshape(len(TCoefficient), 1)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
HCoefficient = np.array(HCoefficient).reshape(len(HCoefficient), 1)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
logTCoefficient = np.array(logTCoefficient).reshape(len(logTCoefficient), 1)
information_data = np.concatenate([TCoefficient, logTCoefficient, HCoefficient], axis = 1)
X9_info = information_data[10: len(price_diff) - SHIFT_NUMBER] 

technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X9_tech = technical_data

X9 = np.concatenate([X9_tech, X9_info], axis = 1)

FULL_COLUMNS = COLUMNS + ["T_n", "Log_T_n", "H_n"]

y9 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, test_size=TEST_SIZE, random_state=0)

model9, coefs9, intercept9, y9_predict, RMSE9, score9 = buildModel9(X9_train, y9_train, X9_test, y9_test, 0.1)
RMSE9
```

    Nb_iterations used:  42529
    




    8.928270351686729



Như vậy, việc đưa thêm các biến thông tin cải thiện RMSE từ 9.34% xuống 8.93%. Độ giảm này không quá đáng kể so với mô hình thuần kĩ thuật. Có thể giải thích hiện tượng này: hoặc chất lượng của dữ liệu thông tin không tốt dẫn đến mô hình  hoặc cách số hoá thông tin của ta chưa hợp lí, hoặc mô hình kĩ thuật chế ngự (dominate) và đã phần nào bao hàm mô hình thông tin. 

Trong khuôn khổ khảo sát các mô hình đơn giản ở TD này, ta có thể dừng tại mô hình này. Phần còn lại khảo sát các biến quan trọng và vẽ đồ thị mô hình.


```python
important_variables = filter(lambda i: abs(model9.coef_[i]) > 1e-2, range(len(model9.coef_)))
pd.DataFrame(important_variables).T
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
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>114</th>
      <th>115</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>9</td>
      <td>11</td>
      <td>12</td>
      <td>22</td>
      <td>35</td>
      <td>42</td>
      <td>45</td>
      <td>111</td>
      <td>135</td>
      <td>...</td>
      <td>1048</td>
      <td>1049</td>
      <td>1053</td>
      <td>1060</td>
      <td>1061</td>
      <td>1062</td>
      <td>1065</td>
      <td>1066</td>
      <td>1076</td>
      <td>1080</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 116 columns</p>
</div>




```python
important_variables_dict = indToNameOnPolynomialRegressionDegree2(FULL_COLUMNS, important_variables) # Một từ điển
pd.DataFrame([important_variables_dict.keys(), important_variables_dict.values(), ]).T
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
      <td>1024</td>
      <td>mdi_14 * Log_T_n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1026</td>
      <td>dx_14 * dx_14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>1 * volume</td>
    </tr>
    <tr>
      <th>3</th>
      <td>518</td>
      <td>kdjj_9 * Log_T_n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>519</td>
      <td>kdjj_9 * H_n</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1032</td>
      <td>dx_14 * vr_6_sma</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>1 * cr-ma3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>1 * kdjk_9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>1 * kdjj_9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1039</td>
      <td>adx * change</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1041</td>
      <td>adx * vr_6_sma</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1044</td>
      <td>adx * H_n</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1045</td>
      <td>adxr * adxr</td>
    </tr>
    <tr>
      <th>13</th>
      <td>22</td>
      <td>1 * cci</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1047</td>
      <td>adxr * change</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1048</td>
      <td>adxr * vr</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1049</td>
      <td>adxr * vr_6_sma</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1053</td>
      <td>trix * trix</td>
    </tr>
    <tr>
      <th>18</th>
      <td>35</td>
      <td>1 * mdi_14</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1060</td>
      <td>change * change</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1061</td>
      <td>change * vr</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1062</td>
      <td>change * vr_6_sma</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1065</td>
      <td>change * H_n</td>
    </tr>
    <tr>
      <th>23</th>
      <td>42</td>
      <td>1 * vr_6_sma</td>
    </tr>
    <tr>
      <th>24</th>
      <td>45</td>
      <td>1 * H_n</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1076</td>
      <td>T_n * Log_T_n</td>
    </tr>
    <tr>
      <th>26</th>
      <td>568</td>
      <td>macds * low_delta</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1035</td>
      <td>dx_14 * H_n</td>
    </tr>
    <tr>
      <th>28</th>
      <td>600</td>
      <td>macdh * dm</td>
    </tr>
    <tr>
      <th>29</th>
      <td>608</td>
      <td>macdh * adxr</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>368</td>
      <td>cr-ma2 * dx_14</td>
    </tr>
    <tr>
      <th>87</th>
      <td>370</td>
      <td>cr-ma2 * adxr</td>
    </tr>
    <tr>
      <th>88</th>
      <td>372</td>
      <td>cr-ma2 * change</td>
    </tr>
    <tr>
      <th>89</th>
      <td>886</td>
      <td>dma * vr</td>
    </tr>
    <tr>
      <th>90</th>
      <td>380</td>
      <td>cr-ma3 * kdjk_9</td>
    </tr>
    <tr>
      <th>91</th>
      <td>390</td>
      <td>cr-ma3 * wr_10</td>
    </tr>
    <tr>
      <th>92</th>
      <td>391</td>
      <td>cr-ma3 * cci</td>
    </tr>
    <tr>
      <th>93</th>
      <td>913</td>
      <td>um * pdm</td>
    </tr>
    <tr>
      <th>94</th>
      <td>405</td>
      <td>cr-ma3 * dx_14</td>
    </tr>
    <tr>
      <th>95</th>
      <td>410</td>
      <td>cr-ma3 * vr</td>
    </tr>
    <tr>
      <th>96</th>
      <td>413</td>
      <td>cr-ma3 * Log_T_n</td>
    </tr>
    <tr>
      <th>97</th>
      <td>414</td>
      <td>cr-ma3 * H_n</td>
    </tr>
    <tr>
      <th>98</th>
      <td>928</td>
      <td>low_delta * low_delta</td>
    </tr>
    <tr>
      <th>99</th>
      <td>931</td>
      <td>low_delta * pdm_14</td>
    </tr>
    <tr>
      <th>100</th>
      <td>935</td>
      <td>low_delta * dx_14</td>
    </tr>
    <tr>
      <th>101</th>
      <td>754</td>
      <td>wr_6 * Log_T_n</td>
    </tr>
    <tr>
      <th>102</th>
      <td>945</td>
      <td>dm * dm</td>
    </tr>
    <tr>
      <th>103</th>
      <td>440</td>
      <td>rsv_9 * mdi_14</td>
    </tr>
    <tr>
      <th>104</th>
      <td>244</td>
      <td>volume * low_delta</td>
    </tr>
    <tr>
      <th>105</th>
      <td>444</td>
      <td>rsv_9 * trix</td>
    </tr>
    <tr>
      <th>106</th>
      <td>451</td>
      <td>kdjk_9 * kdjk_9</td>
    </tr>
    <tr>
      <th>107</th>
      <td>968</td>
      <td>pdm * adxr</td>
    </tr>
    <tr>
      <th>108</th>
      <td>976</td>
      <td>pdm_14 * pdm_14</td>
    </tr>
    <tr>
      <th>109</th>
      <td>985</td>
      <td>pdm_14 * vr</td>
    </tr>
    <tr>
      <th>110</th>
      <td>476</td>
      <td>kdjk_9 * dx_14</td>
    </tr>
    <tr>
      <th>111</th>
      <td>478</td>
      <td>kdjk_9 * adxr</td>
    </tr>
    <tr>
      <th>112</th>
      <td>480</td>
      <td>kdjk_9 * change</td>
    </tr>
    <tr>
      <th>113</th>
      <td>995</td>
      <td>pdi_14 * adxr</td>
    </tr>
    <tr>
      <th>114</th>
      <td>999</td>
      <td>pdi_14 * vr_6_sma</td>
    </tr>
    <tr>
      <th>115</th>
      <td>1016</td>
      <td>mdi_14 * dx_14</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 2 columns</p>
</div>




```python
getModelReadableForm(coefs9, important_variables_dict)
```




    ' + (4.37) * low * low  + (-4.28) * pdm_14 * vr  + (4.19) * T_n * Log_T_n  + (-3.69) * macdh * adxr  + (-3.66) * kdjk_9 * adxr  + (-3.47) * dma * dma  + (2.63) * cci_20 * trix  + (2.57) * cr-ma1 * tr  + (2.53) * adxr * vr  + (-2.37) * 1 * kdjj_9  + (-2.32) * cr-ma3 * Log_T_n  + (-2.32) * 1 * kdjk_9  + (-2.28) * 1 * mdi_14  + (-2.28) * pdi_14 * vr_6_sma  + (-2.12) * 1 * cr-ma3  + (-2.09) * low * volume  + (2.08) * trix * trix  + (-1.89) * rsi_12 * Log_T_n  + (-1.80) * cr-ma1 * adx  + (1.73) * adxr * vr_6_sma  + (-1.70) * wr_6 * dma  + (-1.68) * low * cr-ma3  + (1.66) * 1 * vr_6_sma  + (-1.62) * cr * vr_6_sma  + (1.60) * pdm * adxr  + (-1.49) * adx * change  + (-1.34) * macdh * H_n  + (1.32) * rsi_12 * rsi_12  + (1.25) * 1 * H_n  + (1.23) * mdi_14 * dx_14  + (-1.18) * high * cci  + (1.16) * cr-ma2 * dx_14  + (1.15) * volume * H_n  + (1.13) * wr_6 * vr_6_sma  + (-1.10) * volume * kdjk_9  + (-1.10) * wr_10 * wr_10  + (1.09) * rsi_12 * adx  + (-1.07) * volume * mdi_14  + (-1.00) * pdi_14 * adxr  + (-0.97) * kdjj_9 * Log_T_n  + (0.96) * cr-ma2 * rs_6  + (-0.96) * adxr * trix  + (0.95) * cr-ma2 * change  + (0.92) * rsi_12 * adxr  + (0.90) * cr-ma3 * H_n  + (0.86) * low_delta * dx_14  + (-0.84) * cr * adx  + (0.79) * rs_12 * adxr  + (-0.77) * volume * high_delta  + (0.72) * adxr * adxr  + (0.72) * kdjk_9 * kdjk_9  + (0.67) * cr * cci_20  + (-0.67) * dma * vr  + (0.67) * wr_10 * vr  + (0.67) * cr-ma3 * vr  + (-0.67) * cr-ma3 * kdjk_9  + (-0.65) * cr * dx_14  + (0.65) * cr-ma2 * cr-ma2  + (-0.62) * cr-ma3 * cci  + (0.61) * wr_6 * wr_6  + (0.60) * adx * vr_6_sma  + (0.54) * change * change  + (0.54) * 1 * cci  + (0.53) * dx_14 * vr_6_sma  + (-0.53) * cr-ma1 * cr-ma1  + (-0.50) * mdi_14 * Log_T_n  + (0.49) * rsi_12 * dx_14  + (-0.48) * tr * low_delta  + (0.47) * wr_10 * Log_T_n  + (-0.47) * low * vr_6_sma  + (0.44) * volume * volume  + (-0.39) * 1 * volume  + (-0.39) * cr-ma2 * adxr  + (0.37) * kdjk_9 * change  + (-0.36) * volume * low_delta  + (0.36) * cci_20 * dx_14  + (0.36) * cr-ma3 * wr_10  + (-0.35) * dm * dm  + (0.33) * macdh * dm  + (0.32) * change * H_n  + (-0.30) * rs_6 * um  + (-0.29) * vr * vr  + (-0.28) * wr_6 * mdi_14  + (0.25) * wr_6 * Log_T_n  + (0.23) * rs_6 * dma  + (-0.23) * dx_14 * dx_14  + (-0.23) * change * vr  + (-0.22) * dx_14 * H_n  + (0.19) * tr * tr  + (0.18) * cr * wr_6  + (0.17) * high_delta * adxr  + (0.17) * cci * cci  + (0.17) * volume * change  + (-0.16) * change * vr_6_sma  + (-0.15) * cr-ma1 * pdi_14  + (0.15) * rsv_9 * trix  + (-0.12) * low_delta * low_delta  + (-0.11) * wr_6 * H_n  + (0.10) * cr-ma1 * kdjk_9  + (0.08) * cr-ma3 * dx_14  + (0.08) * wr_6 * adxr  + (-0.08) * kdjk_9 * dx_14  + (-0.08) * um * pdm  + (-0.07) * volume * cr-ma2  + (-0.07) * adx * H_n  + (-0.07) * low * pdi_14  + (-0.05) * pdm_14 * pdm_14  + (-0.05) * H_n * H_n  + (-0.05) * tr * high_delta  + (-0.05) * adxr * change  + (-0.04) * macds * low_delta  + (-0.03) * low_delta * pdm_14  + (0.03) * kdjj_9 * H_n  + (0.03) * rs_6 * rs_6  + (0.03) * cci_20 * H_n  + (0.02) * rsv_9 * mdi_14 '




```python
# 200 ngày đầu tiên
plt.plot(y9[:200], label="price_diff")
plt.plot(model9.predict(PolynomialFeatures(2).fit_transform(X9))[:200], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0x19e8b400>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_118_1.png)



```python
# 200 ngày cuối cùng
plt.plot(y9[-200:], label="price_diff")
plt.plot(model9.predict(PolynomialFeatures(2).fit_transform(X9))[-200:], label="predicted_price_diff")
plt.legend()
```




    <matplotlib.legend.Legend at 0x19f4ce48>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_119_1.png)


## References

[1] https://bitcoincharts.com/charts/bitstampUSD#rg60ztgSzm1g10zm2g25zv

[2] http://www.empirical.net/wp-content/uploads/2014/12/Fama-Random-Walks-in-Stock-Market-Prices.pdf

[3] http://ijssst.info/Vol-15/No-4/data/4923a105.pdf

[4] http://www.statsmodels.org/dev/tsa.html

[5] http://www.econ.yale.edu/~shiller/behfin/2001-05-11/chan.pdf
