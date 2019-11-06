
# TD2 - Univariate Polynomials - Đa thức một biến


## Mô tả

Đa thức một biến được biểu diễn dưới dạng 
$a_0+a_1 X+a_2 X^2+⋯+a_n X^n$
trong đó $a_n \neq 0$.

Trong TD này ta làm việc với các đa thức với hệ số **thực**, tức $a_0, a_1, \ldots, a_n \in \mathbf R$.

Ta muốn xây dựng một class Polynomial trong Python thực hiện những phép toán cơ bản trên đa thức như cộng, trừ, nhân, chia, luỹ thừa, tìm ước chung lớn nhất và bội chung nhỏ nhất, tìm nghiệm của đa thức và tìm cực trị của đa thức.


## Yêu cầu

Bạn cần viết code trong file <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson2/TD/Polynomial.py">Polynomial.py</a> và chạy các test trong <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson2/TD/TestPolynomial.py">TestPolynomial.py</a>. Trong file Polynomial.py, bạn cần hoàn thành các method trong class **Polynomial**, **LinearPolynomial**, **QuadraticPolynomial** và **CubicPolynomial**. Ngoài ra đã có một class **PolynomialError** được viết sẵn mà bạn có thể sử dụng. Bạn có thể viết các hàm phụ khác ngoài các hàm đã yêu cầu trong đề.

Bạn có thể sử dụng các hàm từ thư viện khác miễn là hàm đó không trực tiếp trả lời câu hỏi trong bài 

Việc sử dụng file **TestPolynomial.py** để test được thực hiện như bài 1. (trong Spyder, Run -> Configure và gõ tên test tương ứng với bài tập (test_1, test_2, test_3, …, test_13)

Bạn cũng có thể tự copy các đoạn test phía dưới vào một file Python để chạy, nhớ import module **Polynomial.py** với các hàm hoàn chỉnh.


```python
from Polynomial_Solutions import *
```

### Bài 1. Khởi tạo đa thức.

Ta biết rằng một đa thức hoàn toàn được xác định nếu các hệ số của nó được xác định. Giả sử ta biết các hệ số của nó và biểu diễn nó trong một list của Python.

Ví dụ, với đa thức $1+2X+3X^2+4X^5$, list hệ số tương ứng là **[1, 2, 3, 0, 0, 4]**. 

*1. Hãy viết instance method **\_\_init\_\_(self, coefficients)** trong class **Polynomial** để khởi tạo đa thức từ đối số **coefficients** là list các hệ số. Việc chọn attribute cho các instance của class **Polynomial** hoàn toàn do bạn quyết định.*

*2. Hãy viết instance method **getCoefficients(self)** trong class Polynomial để trả lại list các hệ số của đa thức **self** ở dạng đơn giản nhất, tức là hệ số cao nhất (phần tử cuối cùng của list trả lại) phải khác 0. (Theo quy ước này, đa thức không sẽ trả lại list rỗng [])*
    
Sau khi viết các instance method ở câu 1, 2, đoạn code dưới đây giúp test các hàm của bạn.


```python
P = Polynomial([1, -2, 3, 0, 1, 4, 0, 0])
print("Coefficients of P: ")
print(P.getCoefficients())
Q = Polynomial([0, 0, 0])
print("Coefficients of Q: ")
print(Q.getCoefficients())
```

    Coefficients of P: 
    [1, -2, 3, 0, 1, 4]
    Coefficients of Q: 
    []
    

*3. Hãy viết instance method **\_\_str\_\_(self)** để in ra **self** một cách "đẹp mắt." Cách biểu diễn do bạn tuỳ ý quyết định (ví dụ biểu diễn đa thức theo các số hạng có bậc giảm dần hoặc tăng dần, sử dụng biến "X" hay "x" cho biến số). Hàm này không cần được test và không có ảnh hưởng đến các bài tập sau.*

Dưới đây là một lời giải về một hàm (method) như vậy. Bạn có thể sử dụng hàm này hoặc tự viết hàm cho mình.


```python
def __str__(self):
        """
            Exercise 1:
            Print a polynomial
        """
        if self.__coefficients == []:
            return "0"
        
        expression = ""
        for i, coef in enumerate(self.__coefficients):
            if coef > 0:
                expression += "+ "
                if i == 0:
                    expression += str(coef)
                elif coef != 1:
                    expression += str(coef) + "*"
                if i > 0:
                    expression += "X"
                    if i > 1:
                        expression += "^" + str(i) 
                expression += " "
            if coef < 0:
                expression += "- "
                if i == 0:
                    expression += str(-coef)
                elif coef != -1:
                    expression += str(-coef) + "*"
                if i > 0:
                    expression += "X"
                    if i > 1:
                        expression += "^" + str(i) 
                expression += " "
                
        if expression[0] == "+":
            expression = expression[2:] #Remove "+ " if the first coefficient is >0
        
        return expression

P = Polynomial([1, -1, 1, 2, 0, 0, 3.5, 1])
print(P) #Print a human-readable form of P
```

    1 - X + X^2 + 2*X^3 + 3.5*X^6 + X^7 
    

### Bài 2. Bậc của đa thức

*Viết instance method **getDegree(self)** trong class **Polynomial** trả lại bậc của **self** dưới dạng một số nguyên.*

*Ta quy ước bậc của đa thức không là -1.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, -1, 1, 2, 0, 0, 3.5, 1])
Q = Polynomial([1, 0, 0])
R = Polynomial([0])
print(P.getDegree(), Q.getDegree(), R.getDegree())
```

    (7, 0, -1)
    

### Bài 3. Cộng đa thức

*Viết instance method **add(self, P)** trong class **Polynomial** nhận đối số **P** là một đa thức khác (tức một instance của class **Polynomial**), và trả lại kết quả là một đa thức (tức một instance thuộc class **Polynomial**) bằng tổng của **self** và **P**.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 2, 2, 0, 2, 1])
print(P)
Q = Polynomial([-1, 2, 1, 0, -1, -1])
print(Q)
R = P.add(Q)
print("Their sum: ")
print(R)
```

    1 + 2*X + 2*X^2 + 2*X^4 + X^5 
    - 1 + 2*X + X^2 - X^4 - X^5 
    Their sum: 
    4*X + 3*X^2 + X^4 
    

### Bài 4. Trừ đa thức

*Viết instance method **substract(self, P)** trong class **Polynomial** nhận đối số **P** là một đa thức khác (tức một instance của class **Polynomial**), và trả lại kết quả là một đa thức (tức một instance thuộc class **Polynomial**) bằng **self - P**. *

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 2, 2, 0, 2, 1])
print(P)
Q = Polynomial([-1, 2, 1, 0, -1, 1])
print(Q)
R = P.substract(Q)
print("Their difference: ")
print(R)
```

    1 + 2*X + 2*X^2 + 2*X^4 + X^5 
    - 1 + 2*X + X^2 - X^4 + X^5 
    Their difference: 
    2 + X^2 + 3*X^4 
    

### Bài 5. Nhân đa thức

*Viết instance method **multiply(self, P)** trong class **Polynomial** nhận đối số **P** là một đa thức khác (tức một instance của class **Polynomial**), và trả lại kết quả là một đa thức (tức một instance thuộc class **Polynomial**) bằng tích của **self** và **P**. *

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 1])
print("P = " + str(P))
Q = Polynomial([1, 2])
print("Q = " + str(Q))
R = P.multiply(Q)
print("P * Q = " + str(R))
print("P * P * P * P * P = " + str(P.multiply(P).multiply(P).multiply(P).multiply(P)))
S = Polynomial([0])
print("P * 0 = " + str(P.multiply(S)))
```

    P = 1 + X 
    Q = 1 + 2*X 
    P * Q = 1 + 3*X + 2*X^2 
    P * P * P * P * P = 1 + 5*X + 10*X^2 + 10*X^3 + 5*X^4 + X^5 
    P * 0 = 0
    

### Bài 6. Luỹ thừa một đa thức

*Viết instance method **power(self, a)** trong class **Polynomial** nhận đối số **a** là một số nguyên không âm (type **int**) và trả lại kết quả là một đa thức bằng luỹ thừa bậc **a** của **self**. Lưu ý: $P^0=1$ với mọi đa thức khác không $P$. Với trường hợp $0^0$, bạn có thể quy ước kết quả bằng 0 hoặc báo lỗi.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 1])
print("P = " + str(P))
Q = P.power(10)
print("P^10 = " + str(Q))
```

    P = 1 + X 
    P^10 = 1 + 10*X + 45*X^2 + 120*X^3 + 210*X^4 + 252*X^5 + 210*X^6 + 120*X^7 + 45*X^8 + 10*X^9 + X^10 
    

Ngoài ra, bạn có thể tự test bằng cách tính $(X+1)^{1000}$. Nếu xảy ra lỗi **RuntimeError: maximum recursion depth exceeded**, hay thử implement bằng một thuật toán khác có số bước lặp ít hơn.

### Bài 7. Chia đa thức

*Viết instance method **divide(self, P)** trong class **Polynomial** nhận đối số là một đa thức **P**, và trả lại kết quả là một tuple **(Q, R)**, trong đó **Q** là đa thức thương và **R** là đa thức dư trong phép chia. Nếu **P** bằng 0, raise một Exception thuộc type PolynomialError.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 0, 1, 0, 2, 1])
print("P = " + str(P))
Q = Polynomial([1, -3, 1])
print("Q = " + str(Q))

Division = P.divide(Q)
R = Division[0]
S = Division[1]

print("P/Q = " + str(R))
print("P mod Q = " + str(S))

print("Verification: R * Q + S = " + str(R.multiply(Q).add(S)))

print("----------")

try:
    P.divide(Polynomial([0]))
except PolynomialError as e:
    print(e)
```

    P = 1 + X^2 + 2*X^4 + X^5 
    Q = 1 - 3*X + X^2 
    P/Q = 38.0 + 14.0*X + 5.0*X^2 + X^3 
    P mod Q = - 37.0 + 100.0*X 
    Verification: R * Q + S = 1.0 + X^2 + 2.0*X^4 + X^5 
    ----------
    'Impossible to divide by 0'
    

### Bài 8. Tính chia hết của đa thức

*Viết instance method **isDivisor(self, P)** trong class **Polynomial** nhận đối số là đa thức **P** và trả lại kết quả **True** nếu **self** là ước của **P**, False nếu không. Nếu **self** là đa thức không, quy ước kết quả là **False**.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 1])
Q = Polynomial([1, 0, 0, 0, 0, 0, -1])
R = Polynomial([1, 0, 0, 0, 0, 0, 1])
print(P.isDivisor(Q))
print(P.isDivisor(R))
```

    True
    False
    

### Bài 9. Ước chung lớn nhất

*Viết instance method **getGcd(self, P**) trong class **Polynomial** nhận đối số là đa thức P và trả lại kết quả là đa thức ước chung lớn nhất của chúng. (Nhắc lại rằng ước chung lớn nhất có hệ số cao nhất là 1).*

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([1, 0, -1])
Q = Polynomial([1, 0, 0, -1])
R = P.getGcd(Q)
print("gcd(" + str(P) + ", " + str(Q) + ") = " + str(R))
```

    gcd(1 - X^2 , 1 - X^3 ) = - 1.0 + X 
    

### Bài 10. Đa thức bậc nhất, bậc hai và bậc ba

Bây giờ ta làm việc với các đa thức đơn giản nhất: bậc nhất, bậc hai và bậc ba thông qua các class **LinearPolynomial, QuadraticPolynomial, CubicPolynomial**.

*Hãy hoàn thiện instance method **\_\_init\_\_** trong các class **LinearPolynomial, QuadraticPolynomial, CubicPolynomial** thừa kế class Polynomial, sao cho nếu nó khởi tạo một đa thức không phải bậc nhất, bậc hai, bậc ba tương ứng thì chương trình sẽ báo lỗi.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
try:
    P = LinearPolynomial([1, 2])
    print(P)
    Q = QuadraticPolynomial([1, 2, -1])
    print(Q)
    R = CubicPolynomial([1, 2, 3, -1, 2])
    print(R)
except PolynomialError as e:
    print("Error: " + str(e))

```

    1 + 2*X 
    1 + 2*X - X^2 
    Error: 'Not a cubic polynomial.'
    

### Bài 11. Nghiệm của đa thức bậc nhất và bậc hai

Bài toán tiếp theo là tìm nghiệm của mọi đa thức.

*Viết instance method **getRoots()** trong class **Polynomial** trả lại list rỗng [] (by default) cho mọi đa thức.*

*Sau đó, trong các class kế thừa **LinearPolynomial, QuadraticPolynomial**, viết lại method **getRoots()** trả lại một list gồm tất cả các nghiệm **phức** của các đa thức bậc nhất và bậc hai.* 

Lưu ý, list trả về cần được sắp xếp phần thực tăng dần. Nếu phần thực bằng nhau thì xếp phần ảo tăng dần.

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = LinearPolynomial([1, 2])
Q = QuadraticPolynomial([1, 0, 1])
R = CubicPolynomial([1, 2, 3, -1])
print(P.getRoots())
print(Q.getRoots())
#print(R.getRoots()) #should print []
```

    [-0.5]
    [-1j, 1j]
    

### Bài 12. Nghiệm phức của đa thức

*Viết lại hàm **getRoots()** cho class **Polynomial** bằng cách implement thuật toán Muller, giúp tìm tất cả các nghiệm phức của mọi đa thức với hệ số thực.*

Bạn có thể tham khảo trang 42, 43 của tài liệu sau: http://www.math.niu.edu/~dattab/MATH435.2013/ROOT_FINDING.pdf

Cũng như trên, list nghiệm của bạn cần được sắp xếp phần thực tăng dần. Nếu phần thực bằng nhau thì xếp phần ảo tăng dần.

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([-16, 0, 0, 0, 0, 0, 0, 0, 1]) #x^8 – 16 = 0
Roots = P.getRoots()
print("Roots of " + str(P) + ": ")
for root in Roots:
    print("%.4f + %.4fj" % (root.real, root.imag))
```

    Roots of - 16 + X^8 : 
    -1.4142 + 0.0000j
    -1.0000 + -1.0000j
    -1.0000 + 1.0000j
    -0.0000 + -1.4142j
    -0.0000 + 1.4142j
    1.0000 + -1.0000j
    1.0000 + 1.0000j
    1.4142 + 0.0000j
    

### Bài 13. Cực trị của đa thức

*Từ kết quả của bài 12, hãy viết instance method **getLocalMin()** và **getLocalMax()** trong class **Polynomial** để trả về list tất cả các số thực là điểm cực tiểu (tương ứng, cực đại) của một đa thức, xếp theo thứ tự tăng dần. *

Bạn có thể viết một hàm phụ **getDerivative()** để tính đạo hàm (và đạo hàm cấp cao, nếu cần) của đa thức.

Đoạn code dưới đây giúp test hàm của bạn.


```python
P = Polynomial([-16, 0, 0, 0, 0, 0, 0, 0, 1]) 
print(P.getLocalMin())
print(P.getLocalMax())
Q = Polynomial([-30, 11, 34, -12, -4, 1]) #x^5 - 4 x^4 - 12 x^3 + 34 x^2 + 11 x - 30
print(Q.getRoots())
print(Q.getLocalMin())
print(Q.getLocalMax())
```

    [0]
    []
    [-3.0000000000000373, -0.99999999999993672, 0.99999999999995959, 2.0000000000000289, 4.999999999999986]
    [-0.15059909416438733, 4.114969050072768]
    [-2.304711819773494, 1.5403418638656496]
    

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson2/TD/Illustration.png"/>
