
# TD4 - Noughts and Crosses - Carô

## Mô tả
Trò chơi caro (Noughts and Crosses) được mô tả như sau: Có một bảng $m \times n$. Hai người chơi lần lượt đánh kí hiệu của riêng mình (**X**) hoặc (**O**) vào một ô còn trống của bảng. Người đầu tiên đánh được kí hiệu của mình ở $k$ ô liên tiếp theo hàng ngang, cột dọc hoặc đường chéo là người thắng cuộc. Nếu tất cả các ô của bảng đều đã được điền mà không ai thực hiện được nhiệm vụ trên thì kết quả được xem là hoà.

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson4/TD/Fig1.png"/>
<center>*Trong hình vẽ, $m = n = 3$. Nếu chọn $k = 3$, người có kí hiệu **X** thắng vì đánh được $k$ kí hiệu **X** liên tiếp trên một đường chéo.*</center>

Trong TD này ta sẽ implement trò chơi này một cách đơn giản, theo các bước sau:

-	Thiết lập một class **Board** để mô tả trạng thái của một bàn cờ

-	Thiết lập một class **Game** để mô tả trạng thái của ván chơi

-	Sau đó, tìm cách để xây dựng chiến thuật khi máy tính chơi với con người sao cho khả năng máy thắng là cao nhất. Ta cũng tìm cách implement chiến thuật sao cho tốn ít thời gian chạy của máy. Trong phạm vi của TD, với phần này, ta cũng chỉ cần thực hiện cho các trường hợp $m, n, k$ nhỏ (không quá 5). 

Bạn cần viết code trong file <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson4/TD/NoughtsAndCrosses.py">NoughtsAndCrosses.py</a>. Trong file này, một số hàm đã được viết sẵn và một số khác đã được đặt tên sẵn. Bạn không sửa tên các hàm này và không thay đổi code đã viết, trừ trường hợp đổi syntax giữa Python 2 và Python 3. 

Bạn có thể viết các hàm phụ khác ngoài các hàm đã yêu cầu trong đề. Bạn có thể sử dụng các hàm từ thư viện khác miễn là hàm đó không trực tiếp trả lời câu hỏi trong bài.

Để test, bạn có thể sử dụng file <a href="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson4/TD/TestNoughtsAndCrosses.py">TestNoughtsAndCrosses.py</a> hoặc sử dụng các đoạn test nằm trong hướng dẫn này.



```python
from NoughtsAndCrosses_Solution import *
```

## Phần 1. Mô tả bàn cờ - Class **Board**

Một bàn cờ là một hình chữ nhật được chia thành m x n ô bằng nhau. Toạ độ được đánh số như sau:
<center>
(0,0)	(0,1)	(0,2)	(0,3)   ...
</center>
<center>
(1,0)	(1,1)	(1,2)	(1,3)   ...
</center>
<center>
(2,0)	(2,1)	(2,2)	(2,3)   ...
</center>
<center>
(3,0)	(3,1)	(3,2)	(3,3)   ...
</center>
<center>
...
</center>
theo quy ước: chỉ số hàng viết trước, chỉ số cột viết sau.

Ta mô tả bàn cờ này bằng class **Board**. Một instance của class **Board**, tức một bàn cờ, được xác định bởi chiều cao, chiều rộng, và trạng thái của tất cả các ô của nó (đã được đánh dấu hay chưa). Ban đầu, tất cả các ô của bàn cờ đều trống. Do đó, hàm __init__ trong class được viết như sau:



```python
  def __init__(self, height, width):
        """
            Create a board with defined height and width
            Fill 0 (empty) to every cell of the board
        """
        self.__height = height
        self.__width = width
        self.__cells = []
        for i in range(height):
            self.__cells.append([])
            for j in range(width):
                self.__cells[i].append(Board.EMPTY)
```

Theo đó, chiều cao và chiều rộng của bàn cờ được biểu diễn bằng các attributes **\_\_height** và **\_\_width**. Trạng thái tất cả các ô của bàn cờ sẽ được biểu diễn bởi một list **self.\_\_cells** của $m$ list con của $n$ số nguyên (**int**). Nếu ô **(i, j)** đang trống, **self.\_\_cells[i][j]** sẽ mang giá trị 0 (được kí hiệu là **EMPTY** trong code). Nếu sau này ô **(i, j)** được đánh bới người chơi 1 (kí hiệu là **X**) hay 2 (kí hiệu là **O**), **self.\_\_cells[i][j]** sẽ mang giá trị tương ứng 1 (**X**) hoặc 2 (**O**).

Khi khởi tạo bàn cờ ở đầu trò chơi, tất cả các ô đều trống, do đó trong hàm **\_\_init** trên, tất cả các ô **self.\_\_cells[i][j]** đều có giá trị 0 (**EMPTY**). 

Trong code đã viết sẵn **draw()** cho phép phác hoạ trạng thái của bàn cờ. Ví dụ ban đầu, bàn cờ rỗng, do đó khi in bàn cờ, ta có trạng thái sau:


```python
board = Board(5, 5)
print board.draw()
```

```
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    

Bây giờ ta viết code mô tả bàn cờ.

### Bài 1. Các getter, setter cơ bản của Board
*Hãy viết các getter, setter sau trong class **Board**:*

*-	**getHeight()** : trả lại chiều cao $m$ của bảng* (type **int**)

*-	**getWidth()** : trả lại chiều rộng $n$ của bảng* (type **int**)

*-	**getBoardStatus()** : trả lại trạng thái của tất cả các ô, (một list m list con, mỗi list con gồm n số nguyên mang giá trị 0, 1 hoặc 2.)*

*-	**setBoardStatus(cells)** : nhận đối số **cells** là một list các list các **int** và gán nó cho **self.__cells***

Đoạn code dưới đây giúp test hàm của bạn


```python
board = Board(5, 4)
print(board.draw())

# Test getHeight()
print("Height: %d" % board.getHeight())

# Test getWidth()
print("Width: %d" % board.getWidth())

# Test getBoardStatus()
print("Status of cells: %s" % board.getBoardStatus())
```

```
    ---------
    | | | | |
    ---------
    | | | | |
    ---------
    | | | | |
    ---------
    | | | | |
    ---------
    | | | | |
    ---------
```
    
    Height: 5
    Width: 4
    Status of cells: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    


```python
board = Board(5, 4)

# Test setBoardStatus()
board.setBoardStatus([[0, 2, 0, 0], [0, 1, 0, 2], [0, 1, 2, 0], [0, 0, 2, 1], [0, 0, 1, 0]])
print(board.draw())
print(board.getBoardStatus())
```

```
    ---------
    | |O| | |
    ---------
    | |X| |O|
    ---------
    | |X|O| |
    ---------
    | | |O|X|
    ---------
    | | |X| |
    ---------
```
    
    [[0, 2, 0, 0], [0, 1, 0, 2], [0, 1, 2, 0], [0, 0, 2, 1], [0, 0, 1, 0]]
    

### Bài 2. Làm việc với từng ô của bàn cờ

Ta sẽ biểu diễn mỗi ô của bàn cờ bằng toạ độ của nó, một tuple **(i, j)** trong Python, trong đó **i** là chỉ số hàng, **j** là chỉ số cột.

*Hãy viết các instant method sau:*

*- **getCellStatus(cell)**: nhận đối số **cell** là toạ độ của ô dưới dạng một tuple **(i, j)** và trả lại trạng thái của ô đó (một trong các số nguyên **EMPTY** = 0, **X** = 1 hay **O** = 2)*

*- **isEmptyCell(cell)**: nhận đối số **cell** là toạ độ của ô dưới dạng một tuple **(i, j)** và trả lại **True** nếu ô đó đang trống (tức có trạng thái 0) hay **False** nếu ô đó đã có người đánh (có trạng thái 1 hoặc 2). 

*- **mark(value, cell)** : nhận đối số **cell** là toạ độ của ô dưới dạng một tuple **(i, j)**, đối số **value** là một giá trị **EMPTY** = 0, **X** = 1 hay **O** = 2, và gán **value** cho trạng thái **_cells[i][j]** của ô đó.*

*- **getEmptyCells()** : trả lại list toạ độ tất cả các ô trống trên bàn cờ dưới dạng list các tuple, ví dụ [(1, 2), (1, 3)]**

Đoạn code dưới đây giúp test hàm của bạn.


```python
# Test setBoardStatus 
board = Board(5, 5)
board.setBoardStatus([[0,0,0,0,0], [0,1,0,0,0], [0,1,2,0,0], [0,2,0,0,0], [0,0,2,0,0]])
print(board.draw())

# Then test getCellStatus
board.getCellStatus((2, 2))
```

```
    -----------
    | | | | | |
    -----------
    | |X| | | |
    -----------
    | |X|O| | |
    -----------
    | |O| | | |
    -----------
    | | |O| | |
    -----------
 ```   
    




    2




```python
# Then test isEmptyCell
board.isEmptyCell((3, 2))
```




    True




```python
# Then test mark
board.mark(1, (1, 2))
print(board.draw())
```

```
    -----------
    | | | | | |
    -----------
    | |X|X| | |
    -----------
    | |X|O| | |
    -----------
    | |O| | | |
    -----------
    | | |O| | |
    -----------
```    
    
    


```python
# Then test getEmptyCells
board = Board(5, 5)
board.setBoardStatus([[1,2,1,1,0], [0,1,2,0,2], [0,1,2,1,0], [0,2,2,0,1], [1,2,2,0,2]])
print(board.draw())

board.getEmptyCells()
```

```
    -----------
    |X|O|X|X| |
    -----------
    | |X|O| |O|
    -----------
    | |X|O|X| |
    -----------
    | |O|O| |X|
    -----------
    |X|O|O| |O|
    -----------
 ```   
    




    [(0, 4), (1, 0), (1, 3), (2, 0), (2, 4), (3, 0), (3, 3), (4, 3)]



### Bài 3. Tìm các lân cận của một ô trên bàn cờ
Một ô **(x, y)** trên bàn cờ được gọi là lân cận của ô **(i, j)** nếu toạ độ của nó khác **(i, j)**,  **x** nằm trong khoảng đóng **[i-1, i+1]** và **y** nằm trong khoảng đóng **[j-1, j+1]**. Mỗi ô trên bàn cờ có tối thiểu 3 lân cận (trường hợp của các ô ở góc), tối đa 8 lân cận (trường hợp của các ô không ở biên).

*Viết instant method **getNeighbors(cell)** nhận đối số **cell** là toạ độ **(i, j)** của một ô trên bàn cờ, và trả lại tất cả các lân cận của nó dưới dạng một list các tuple toạ độ.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
board = Board(5, 5)
board.getNeighbors((2, 3))
```




    [(1, 3), (2, 2), (1, 2), (3, 2), (3, 3), (2, 4), (1, 4), (3, 4)]




```python
board.getNeighbors((4, 3))
```




    [(3, 3), (4, 2), (3, 2), (4, 4), (3, 4)]



## Phần 2 - Mô tả ván chơi - Class Game

Một ván chơi được tiến hành như sau: Hai người chơi 1 (**X**) và 2 (**O**) thống nhất về kích thước bàn cờ, luật chơi (tức về $m, n, k$), và ai là người đi trước. Sau đó họ tiến hành đánh cờ lần lượt vào các ô trống cho đến khi xác định được người chiến thắng hoặc khi mọi ô của bàn cờ được đánh hết.

Như vậy, các tham số cần cho thời điểm ban đầu:

1.	$m, n, k$ (khác nhau với các ván chơi khác nhau)

2.	Người đánh trước là ai.

Ở mỗi thời điểm (bất kì), trạng thái của một ván chơi sẽ hoàn toàn được xác định bởi các thông tin sau:

1.	Trạng thái của bàn cờ như thế nào (ô nào đang trống hay đang có kí hiệu gì)

2.	Trò chơi đã kết thúc chưa.

3.	Nếu trò chơi đã kết thúc, ai là người thắng, hay kết quả là hoà.

4.	Nếu trò chơi chưa kết thúc, ai có quyền đánh ở nước tiếp theo.

Ta diễn dịch các thông tin này thành các attribute của các instance trong class **Game**. Ta viết hàm **\_\_init\_\_** như sau.


```python
def __init__(self, height, width, firstTurn, winNumber):
    """
    Begin a game and define which attributes are necessary to describe the game status at some moment
    """
    self.__winNumber = winNumber
    self.__board = Board(height, width)
    self.__turn = firstTurn
    self.__status = Game.ACTIVE
    self.__winner = Game.NOWINNER 
```

Các tham số cần cho **thời điểm ban đầu** là $m, n, k$ và ai là người đi trước được lần lượt diễn dịch thành các đối số **height**, **width**, **winNumber**, **firstTurn** cung cấp cho **\_\_init\_\_**.

Các tham số cần cho **mỗi thời điểm bất kì** ứng với các điểm 1, 2, 3, 4 nêu trên được diễn dịch thành các attribute **\_\_board**, **\_\_status**, **\_\_winner**, **\_\_turn** của class. Trong đó, 

- **\_\_board** thuộc class **Board**, ban đầu là một bàn cờ trống; 
- **\_\_turn** nhận giá trị 1 (**X**) hoặc 2 (**O**), ban đầu nhận giá trị **firstTurn** được cung cấp từ tham số; 
- **\_\_status** nhận giá trị 1 (**ACTIVE**) hoặc 0 (**INACTIVE**), ban đầu **ACTIVE**; 
- **\_\_winner** nhận giá trị 0 (**NOWINNER**, chưa có người thắng hoặc kết quả hoà), 1 (**X**) hoặc 2 (**O**), ban đầu là **NOWINNER**.

$k$ sẽ được sử dụng về sau, do đó ta cũng lưu $k$ trong attribute **\_\_winNumber**. $m, n$ cũng được sử dụng về sau nhưng đã được lưu trong **\_\_board** (có thể gọi bằng **\_\_board.getHeight()**, **\_\_board.getWidth()**) nên không cần lưu bằng attribute riêng.

Một hàm **draw** đã được viết sẵn để phác hoạ bàn cờ (tương tự phần 1).

Bây giờ ta viết phần code cần cho một ván chơi ở các bài 4, 5, 6, 7, 8.

### Bài 4. Các getter, setter cơ bản của Game
Trong class **Game**, hãy viết các getter, setter dưới đây.

*1.	Hãy viết hàm **getBoard()** trả lại một instance thuộc class **Board** mô tả bàn cờ tại một thời điểm bất kì.*

*2.	Hãy viết hàm **isActive()** trả lại **True** nếu ván chơi chưa kết thúc và **False** nếu ván chơi đã kết thúc.*

*3.	Hãy viết hàm **getTurn()** trả lại số nguyên 1 (**X**) hoặc 2 (**O**) cho biết ai có quyền đi nước tiếp theo nếu trò chơi vẫn tiếp diễn.*

*4.	Hãy viết hàm **getWinner()** trả lại số nguyên 1 (**X**), 2 (**O**) cho biết người thắng, hoặc 0 (**NOWINNER**) nếu chưa có người thắng hay kết quả hoà.*

*5.	Hãy viết hàm **deactivate()** để chuyển trạng thái của trò chơi thành kết thúc (**INACTIVE = 0**). Sau khi thực hiện hàm này, nếu gọi lại **isActive()** thì kết quả là **False**. Bạn có thể raise error nếu trò chơi đã kết thúc (tức trạng thái đã là **INACTIVE**) mà hàm **deactivate()** vẫn được gọi (nếu muốn).*

*6.	Hãy viết hàm **activate()** để chuyển trạng thái của trò chơi thành tiếp diễn (**ACTIVE = 1**). Sau khi thực hiện hàm này, nếu gọi lại **isActive()** thì kết quả là **True**. Bạn có thể raise error nếu trò chơi đang tiếp diễn (tức trạng thái đã là **ACTIVE** mà hàm **activate()** vẫn được gọi.*

*7.	Hãy viết hàm **switchTurn()** để gán cho attribute **\_\_turn** giá trị **X** (1) nếu attribute này đang là **O** (2) và ngược lại.*

*8.	Hãy viết hàm **declareWinner(player)** để tuyên bố người chơi **player** đã thắng (tức gán **player** cho **\_\_winner**).*

Các đoạn dưới đây giúp test 8 hàm trên (hãy chạy lần lượt)


```python
# Create a game on a 5x5 board, O (player 2) goes first, a player wins when he marks k = 4 consecutive signs
game = Game(5, 5, 2, 4)
game.draw()
```

```
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
# Test getBoard
game.getBoard().getWidth(), game.getBoard().getHeight(), game.getBoard().getBoardStatus()
```




    (5,
     5,
     [[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]])




```python
# Test isActve()
game.isActive()
```




    True




```python
# Test getTurn()
game.getTurn()
```




    2




```python
# Test getWinner()
game.getWinner()
```




    0




```python
# Test deactivate()
game.deactivate()
game.isActive()
```




    False




```python
# Test activate()
game.activate()
game.isActive()
```




    True




```python
# Test switchTurn()
game.switchTurn()
game.getTurn()
```




    1




```python
# Test declareWinner()
game.declareWinner(2)
game.getWinner()
```




    2



### Bài 5. Đánh cờ
Bây giờ ta viết hàm **mark** biểu diễn một nước cờ. Giả sử người chơi player muốn đánh kí hiệu của mình vào ô có toạ độ là **cell** (nhắc lại rằng toạ độ một ô được biểu diễn bởi 1 tuple **(i, j)** trong Python). 

*Hãy viết instant method **mark(player, cell)** lấy đối số **player** là một số nguyên (người chơi **X** = 1 hoặc **O** = 2) và **cell** (toạ độ **(i, j)** của ô muốn đánh), kiểm tra xem **player** có quyền đánh vào ô **cell** không. Người chơi **player** được quyền đánh vào ô **cell** nếu tất cả các điều kiện sau được thoả mãn:*

*- Trò chơi đang ở trạng thái **ACTIVE** (chưa kết thúc).*

*- Lượt hiện tại đang là của người chơi **player**.*

*- Ô **cell** còn đang trống.*

*Nếu tất cả các điều kiện trên được thoả mãn, đánh kí hiệu của **player** vào ô **cell** của bàn cờ (bằng cách dùng hàm **mark** của class **Board** đã viết ở bài 2). Nếu không có quyền, hàm báo một lỗi cho biết vì sao người chơi không có quyền đánh cờ.*

Dùng các đoạn code dưới đây để kiểm tra hàm của bạn.

(Lưu ý rằng ta sẽ tiếp tục hoàn chỉnh hàm **mark** trong bài 8.)


```python
game = Game(5, 5, 2, 4)
try:
    game.mark(1, (0, 0)) # Đoạn code không được thực hiện vì người chơi 1 không có quyền đánh trước. Lỗi được báo.
except Exception as e:
    print e
game.draw()
```

    "It's not 1's turn."
```
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
game = Game(5, 5, 2, 4)
game.mark(2, (1, 1)) # Đoạn code được thực hiện vì mọi điều kiện đều thoả mãn
game.draw()
```

```
    -----------
    | | | | | |
    -----------
    | |O| | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
game = Game(5, 5, 2, 4)
game.mark(2, (1, 1)) # Đoạn code được thực hiện vì mọi điều kiện đều thoả mãn.
try:
    game.mark(1, (1, 1)) # Đoạn code không được thực hiện vì ô (1, 1) đã bị đánh. Lỗi được báo
except Exception as e:
    print e
game.draw()
```

    'Cell (1, 1) is not empty.'
```
    -----------
    | | | | | |
    -----------
    | |O| | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
game = Game(5, 5, 2, 4)
game.deactivate()
try:
    game.mark(1, (3, 2)) # Đoạn code không được thực hiện vì trò chơi đã inactive, hàm deactivate đã được gọi
except Exception as e:
    print e
game.draw()
```

    'Game has finished'
```
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    

### Bài 6. Người chơi đã thắng?

Giả sử người chơi **player** vừa đánh vào ô **cell**. Nước đánh này có thể làm cho người chơi **player** thắng cuộc nếu nó tạo được với các ô đã đánh trước đó $k$ kí hiệu liên tiếp trên một hàng ngang, cột dọc hoặc đường chéo.

*Hãy viết instant method **isVictoryCell(cell)** nhận tham số là toạ độ **cell** của một ô và trả lại **True** nếu nó làm cho người đánh tại ô đó thắng, và **False** nếu nó chưa làm cho người đánh tại ô đó thắng. Nói cách khác, trả lại **True** nếu **cell** nằm trên một hàng ngang, cột dọc hoặc đường chéo gồm $k$ kí hiệu liên tiếp do một trong 2 người chơi đánh.*


```python
game = Game(5, 5, 2, 4)
checker = 1
game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [2, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, 0, 2]])
game.draw()
game.isVictoryCell((3, 1)) # Trả lại True vì ô (3, 1) nằm trên 1 cột dọc gồm 4 chữ X liên tiếp.
```

```
    -----------
    | | |O| | |
    -----------
    |O|X| | | |
    -----------
    | |X| | | |
    -----------
    | |X| |O| |
    -----------
    | |X| | |O|
    -----------
 ```   
    




    True




```python
game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [2, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
game.draw()
game.isVictoryCell((3, 1)) # Trả lại False vì ô (3, 1) không nằm trên 1 cột dọc, hàng ngang, đường chéo nào gồm 4 chữ X hay O.
```

```
    -----------
    | | |O| | |
    -----------
    |O|X|X|X|X|
    -----------
    | | | | | |
    -----------
    | | | |O| |
    -----------
    | | | | |O|
    -----------
 ```   
    




    False




```python
game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [2, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
game.draw()
game.isVictoryCell((1, 4)) # Trả lại True vì ô (1, 4) nằm trên 1 hàng ngang gồm 4 chữ X liên tiếp.
```

```
    -----------
    | | |O| | |
    -----------
    |O|X|X|X|X|
    -----------
    | | | | | |
    -----------
    | | | |O| |
    -----------
    | | | | |O|
    -----------
 ```   
    




    True




```python
game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [1, 2, 1, 1, 1], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
game.draw()
game.isVictoryCell((4, 4))  # Trả lại True vì ô (4, 4) nằm trên 1 đường chéo Tây Bắc - Đông Nam gồm 4 chữ O liên tiếp.
```

```
    -----------
    | | |O| | |
    -----------
    |X|O|X|X|X|
    -----------
    | | |O| | |
    -----------
    | | | |O| |
    -----------
    | | | | |O|
    -----------
 ```   
    




    True




```python
game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 2, 1, 0], [0, 2, 1, 0, 0]])
game.draw()
game.isVictoryCell((4, 1))  # Trả lại True vì ô (4, 1) nằm trên 1 đường chéo Đông Bắc - Tây Nam gồm 4 chữ O liên tiếp.
```

```
    -----------
    | | | | | |
    -----------
    | | |X|X|O|
    -----------
    | | | |O| |
    -----------
    | | |O|X| |
    -----------
    | |O|X| | |
    -----------
 ```   
    




    True



### Bài 7. Bàn cờ đã được đánh hết?

*Trong class **Game**, hãy viết instant method **isFull()** trả lại **True** nếu tất cả các ô của bàn cờ đã được đánh hết (tức là mọi ô của bàn cờ đều có giá trị **X**=1 hoặc **O**=2) và **False** nếu chưa.*

Đoạn code dưới đây giúp test hàm của bạn.


```python
game = Game(5, 5, 2, 4)
game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 2, 1, 0], [0, 2, 1, 0, 0]])
game.draw()
game.isFull()
```

```
    -----------
    | | | | | |
    -----------
    | | |X|X|O|
    -----------
    | | | |O| |
    -----------
    | | |O|X| |
    -----------
    | |O|X| | |
    -----------
 ```   
    




    False




```python
game = Game(5, 5, 2, 4)
game.getBoard().setBoardStatus([[1, 2, 1, 2, 2], [2, 1, 1, 1, 2], [2, 1, 1, 2, 1], [2, 2, 2, 1, 1], [1, 2, 1, 1, 2]])
game.draw()
game.isFull()
```

```
    -----------
    |X|O|X|O|O|
    -----------
    |O|X|X|X|O|
    -----------
    |O|X|X|O|X|
    -----------
    |O|O|O|X|X|
    -----------
    |X|O|X|X|O|
    -----------
 ```   
    




    True



### Bài 8. Kiểm tra sau mỗi nước đánh

Ngay sau khi người chơi **player** đánh xong, máy tính sẽ thực hiện các thao tác sau:

*- Kiểm tra xem nước đánh đó có giúp người chơi thắng không (bằng cách gọi hàm isVictoryCell đã viết ở bài 6) hoặc bàn cờ đã được đánh hết chưa (dùng hàm isFull đã viết ở bài 7). *

 * *Nếu người chơi **player** thắng, máy sẽ tuyên bố người thắng cuộc (hàm **declareWinner(player)** đã viết ở bài 4), chuyển trạng thái ván chơi về kết thúc (hàm  **deactivate()** đã viết ở bài 4). *

 * *Nếu không, nếu bàn cờ đã đầy, máy không tuyên bố người thắng cuộc nhưng chuyển trạng thái ván chơi về kết thúc (hàm  **deactivate()** đã viết ở bài 4). *

*- Nếu cả trường hợp trên đều không xảy ra, máy sẽ trao quyền đi tiếp cho người tiếp theo (hàm switchTurn()đã viết ở bài 4).*

*Hãy bổ sung các thao tác này vào cuối instant method **mark(player, cell)** đã viết ở bài 5 để hoàn thiện method **mark**.*

Test các khả năng xảy ra sau khi một người đánh bằng các đoạn test dưới đây.


```python
# Kịch bản 1
game = Game(5, 5, 2, 4)

# Giả sử bàn cờ đã được đánh đến trạng thái này
game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 0, 1, 0], [0, 2, 1, 0, 0]])

game.mark(2, (3, 2)) #Tại đây người chơi 2 sẽ thắng, do đó khi gọi isActive() và getWinner() cần cho kết quả False và 2.
game.draw()

print(game.isActive()) #Cần bằng False
print(game.getWinner()) #Cần bằng 2
```

```
    -----------
    | | | | | |
    -----------
    | | |X|X|O|
    -----------
    | | | |O| |
    -----------
    | | |O|X| |
    -----------
    | |O|X| | |
    -----------
 ```   
    False
    2
    


```python
# Kịch bản 2
game = Game(5, 5, 1, 4)

# Giả sử bàn cờ đã được đánh đến trạng thái này
game.getBoard().setBoardStatus([[0, 2, 1, 2, 2], [2, 1, 1, 1, 2], [1, 1, 2, 2, 1], [2, 2, 1, 1, 1], [1, 2, 1, 1, 2]])

game.mark(1, (0, 0)) #Tại đây không ai thắng nhưng bàn cờ full, do đó gọi isActive() và getWinner() cần cho kết quả False và 0.

game.draw()
print(game.isActive()) #Cần bằng False
print(game.getWinner()) #Cần bằng 0
```

```
    -----------
    |X|O|X|O|O|
    -----------
    |O|X|X|X|O|
    -----------
    |X|X|O|O|X|
    -----------
    |O|O|X|X|X|
    -----------
    |X|O|X|X|O|
    -----------
 ```   
    False
    0
    


```python
# Kịch bản 3
game = Game(5, 5, 2, 4)

# Giả sử bàn cờ đã được đánh đến trạng thái này
game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 0, 1, 0], [0, 2, 1, 0, 0]])

game.mark(2, (3, 4)) #Tại đây không ai thắng, bàn cờ chưa đầy, do đó khi gọi isActive() và getWinner() các giá trị là True, 0
game.draw()

print(game.isActive()) #Cần bằng True
print(game.getWinner()) #Cần bằng 0

# Ngoài ra khi gọi getTurn() cần nhận giá trị là 1, vì player 2 vừa mới đánh xong
print(game.getTurn()) #Cần bằng 1
```

```
    -----------
    | | | | | |
    -----------
    | | |X|X|O|
    -----------
    | | | |O| |
    -----------
    | | | |X|O|
    -----------
    | |O|X| | |
    -----------
 ```   
    True
    0
    1
    

### Một ván chơi

Đến đây việc implement luật chơi vào class **Game** đã kết thúc. Bạn có thể tự test một ván chơi đơn giản bằng cách gõ các dòng sau vào iPython console lần lượt. (Thay toạ độ các ô bằng nước đi bạn muốn đi)


```python
X = 1
O = 2
game = Game(5, 5, O, 4)
game.mark(O, (2, 1))
game.draw()
```

```
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | |O| | | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
game.mark(X, (2, 2))
game.draw()
```

```
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
    | |O|X| | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
# Ván chơi tiếp tục
game.mark(O, (1, 2))
game.mark(X, (0, 3))
game.mark(O, (1, 1))
game.mark(X, (3, 1))
game.mark(O, (1, 3))
game.mark(X, (1, 4))
game.draw()
```

```
    -----------
    | | | |X| |
    -----------
    | |O|O|O|X|
    -----------
    | |O|X| | |
    -----------
    | |X| | | |
    -----------
    | | | | | |
    -----------
 ```   
    


```python
# Ván chơi kết thúc
game.mark(O, (1, 0))
print(game.getWinner()) #Cần bằng 2, tức người chơi O
```

    2
    

## Phần 3 - Máy tính học chơi cờ - Thuật toán thứ nhất

Ta tiếp tục làm việc với class **Game**. Thay vì kí hiệu 2 người chơi là **X**, **O** ta sẽ xem 1 là con người (**HUMAN**) và 2 là máy tính (**MACHINE**). Các hằng số này đã được định nghĩa trong class **Game**.

Giả sử ván cờ đang ở một trạng thái nhất định, chưa kết thúc và **MACHINE** sẽ có quyền đánh tiếp. Việc tiếp theo của **MACHINE** là phải quyết định nên đánh vào ô nào để khả năng thắng lớn nhất.

Chiến thuật của **MACHINE** như sau:

**Thuật toán 1**

**Input:** Trạng thái hiện tại của ván chơi (tức instant thuộc class Game đang chứa hàm implement thuật toán)

**Output:** Toạ độ **(i, j)** "tốt nhất" để thắng

- Đối với từng ô **(i, j)** trong tất cả các ô trống của bàn cờ:

- Tạo ra $N$ (= 10, 100 hay 1000, …) bản sao như trạng thái hiện tại của ván chơi.

- Với mỗi bản sao:
 - Đánh vào ô **(i, j)**

 - Tiếp tục ván chơi một cách ngẫu nhiên (đóng vai **HUMAN** đánh vào một ô trống ngẫu nhiên, rồi trở lại vai **MACHINE** đánh vào một ô trống ngẫu nhiên khác…) cho đến khi ván chơi kết thúc
 
 - Tính hiệu suất chiến thắng của việc đánh vào **(i, j)** trên tập hợp tất cả các bản sao đã tạo ra. Hiệu suất có thể được tính như sau:
 
<center>**H(i, j)** =  Số ván thắng – số ván thua</center>
<center>(chú ý số ván thắng + hoà + thua = $N$, nên -$N \leq $ **H(i,j)** $\leq N$.)</center>
 - Sau khi có tất cả **H(i,j)**, chọn **(i,j)** làm **H** lớn nhất.
 
Ta sẽ viết code cho phần này trong các bài 9, 10, 11


### Bài 9. Đánh giá nước vừa đi

*1.	Trong class **Board** (không phải class **Game**), hãy viết instance method **getARandomEmptyCell()** trả lại toạ độ của một ô đang trống một cách ngẫu nhiên đều (các ô trống có xác suất được chọn như nhau). Nhắc lại toạ độ ô là một tuple của Python.*

*2.	Giả sử trò chơi đang ở trạng thái chưa kết thúc và **player** là người vừa đánh nước gần nhất. Trong class **Game**, hãy viết instant method **evaluateLastStepRandomly(player)** thực hiện việc cho 2 người chơi chơi ngẫu nhiên bằng cách liên tục chọn một ô trống ngẫu nhiên (dùng hàm đã viết ở phần 1 bài này.) rồi lần lượt đánh trên chính ván chơi đó cho đến khi ván chơi kết thúc, sau đó trả lại 1 nếu **player** thắng, -1 nếu **player** thua và 0 nếu hoà.*

Test các hàm vừa viết bằng các đoạn code dưới đây.


```python
# Test getARandomEmptyCell() 
X, O = 1, 2
game = Game(5, 5, O, 4)
game.mark(O, (2, 1))
game.mark(X, (2, 2))
game.mark(O, (1, 1))
game.mark(X, (0, 3))
game.mark(O, (3, 1))
game.getBoard().getARandomEmptyCell() # Cần có kết quả khác nhau khi chạy và không trùng với (2, 1), (2, 2), (1, 1), (0, 3), (3, 1)
```




    (3, 4)




```python
# Test evaluateLastStepRandomly(player)
X, O = 1, 2
game = Game(5, 5, O, 4)
game.mark(O, (2, 1))
game.mark(X, (2, 2))
game.mark(O, (1, 1))
game.mark(X, (0, 3))
game.mark(O, (3, 1))
game.mark(X, (4, 0))
game.mark(O, (4, 1))
game.draw()
game.evaluateLastStepRandomly(O) # Cần có kết quả 1 vì người chơi O đã thắng
```

```
    -----------
    | | | |X| |
    -----------
    | |O| | | |
    -----------
    | |O|X| | |
    -----------
    | |O| | | |
    -----------
    |X|O| | | |
    -----------
 ```   
    




    1




```python
# Test evaluateLastStepRandomly(player)
X, O = 1, 2
game = Game(5, 5, X, 4)

# Giả sử bàn cờ đã đánh đến đây, và O là người vừa đánh.
game.getBoard().setBoardStatus([[0, 2, 0, 2, 2], [2, 1, 1, 1, 2], [1, 1, 2, 2, 2], [2, 2, 1, 1, 1], [1, 2, 1, 1, 2]])
game.mark(X, (0, 0))
game.draw()
game.evaluateLastStepRandomly(X) # Cần có kết quả -1 vì ở nước cuối cùng O sẽ thắng, và ta đang đánh giá nước đi của X
```

```
    -----------
    |X|O| |O|O|
    -----------
    |O|X|X|X|O|
    -----------
    |X|X|O|O|O|
    -----------
    |O|O|X|X|X|
    -----------
    |X|O|X|X|O|
    -----------
 ```   
    




    -1




```python
# Test evaluateLastStepRandomly(player)
X, O = 1, 2
game = Game(5, 5, O, 4)
# Giả sử bàn cờ đã đánh đến đây, và O là người vừa đánh.
game.getBoard().setBoardStatus([[0, 0, 0, 2, 2], [2, 1, 1, 1, 2], [1, 1, 2, 2, 2], [2, 2, 1, 1, 1], [1, 2, 1, 1, 2]])
game.mark(O, (0, 1))
game.draw()
game.evaluateLastStepRandomly(O) # Cần có kết quả 0 hoặc 1 vì ở các nước còn lại O sẽ thắng hoặc hoà, và ta đang đánh giá nước vừa đi của O
```

```
    -----------
    | |O| |O|O|
    -----------
    |O|X|X|X|O|
    -----------
    |X|X|O|O|O|
    -----------
    |O|O|X|X|X|
    -----------
    |X|O|X|X|O|
    -----------
 ```   
    




    1



### Bài 10. Tạo một bản sao trạng thái hiện tại của ván chơi

Trong class **Game**, hãy viết instant method **generateGameCopy()** trả lại một instance mới thuộc class **Game** có trạng thái (tức **giá trị** của mọi attributes) giống với trạng thái hiện tại của instance ứng với ván đang chơi. (Lưu ý: với attribute **\_\_board**, hãy chỉ copy các giá trị, đừng copy nguyên instance **\_\_board** của ván chơi chính) 

Bạn cần test với đoạn code dưới đây.


```python
X, O = 1, 2
game = Game(5, 5, O, 4)
checker = 1
game.mark(O, (2, 1))
game.mark(X, (2, 2))
game.mark(O, (1, 2))
game.mark(X, (0, 3))
game.mark(O, (1, 1))
    
copy_game = game.generateGameCopy()

# Các giá trị của ván copy cần giống hệt các giá trị của ván chính, tức là các giá trị sau phải giống nhau
print("SHOULD BE THE SAME")
print (copy_game.getTurn(), game.getTurn())
print (copy_game.getWinner(), game.getWinner())
print (copy_game.getBoard().getBoardStatus(), game.getBoard().getBoardStatus())

# Tuy nhiên, instance __board của ván copy phải khác instance __board của ván chính, tức 2 giá trị sau phải khác nhau. 
# Nếu chúng giống nhau, mọi thay đổi sau đó trên ván copy sẽ thay đổi ngay trên ván chính. Ta không muốn điều này vì chỉ muốn
# chơi thử trên ván copy chứ không làm thay đổi trạng thái ván chính khi chơi thử.

print("SHOULD BE DIFFERENT")
print (game.getBoard(), copy_game.getBoard())
```

    SHOULD BE THE SAME
    (1, 1)
    (0, 0)
    ([[0, 0, 0, 1, 0], [0, 2, 2, 0, 0], [0, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0], [0, 2, 2, 0, 0], [0, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    SHOULD BE DIFFERENT
    (<NoughtsAndCrosses_Solution.Board instance at 0x00000000040F5048>, <NoughtsAndCrosses_Solution.Board instance at 0x00000000040F5708>)
    

### Bài 11. Cho máy tính tự chơi với chính nó $N$ lần.

Ván chơi đang ở trạng thái chưa kết thúc và **player** sẽ là người đánh nước tiếp theo. *Trong class **Game**, hãy viết instance method **algo1(player)** implement thuật toán 1, nhận tham số là **player** và thực hiện các bước của thuật toán, rồi trả lại một tuple là ô cần đánh.* Nhắc lại, các bước của thuật toán là:

- Tìm tất cả các ô trống đang có trên bàn cờ
- Với mỗi ô trống **(i, j)**:
 - Tạo một vòng lặp $N$ lần (nên bắt đầu với $N$ nhỏ, khoảng bằng 10 rồi 100, để kiểm tra tính chính xác của code), ở mỗi lần:
  - Tạo ra một bản sao giống hệt trạng thái ván chơi hiện tại.
  - Đánh vào **(i, j)** trong bản sao vừa tạo ra (đừng đánh vào game chính!)
  - Đánh giá ngẫu nhiên nước vừa đi bằng điểm 1, 0 hoặc -1 bằng cách dùng **evaluateLastStepRandomly(player)** cho bản sao.
 - Tính tổng điểm của tất cả các ván chơi bản sao trong vòng lặp, đó chính là **H(i,j)**.
- Trả lại (return) toạ độ **(i, j)** làm **H** lớn nhất.

Sử dụng đoạn code dưới đây để test.


```python
if __name__ == "__main__":
    HUMAN = 1
    MACHINE = 2
    game = Game(3, 3, MACHINE, 3)
    print game.decideNextStep(MACHINE, Game.algo1)
```

    (1, 1)
    

Trong đoạn code để test trên, một method **decideNextStep(game, player, algo)** đã được viết sẵn với mục đích sử dụng các thuật toán khác nhau giải quyết bài toán. Ở đây dòng **game.decideNextStep(MACHINE, Game.algo1)** nghĩa là ta dùng **algo1** cho tham số algo của hàm để quyết định nước đi tiếp theo. Một số thuật toán khác (**algo2**, **algo3**) sẽ được giới thiệu ở phần sau.

Kết quả test có hợp lí không? Nếu là người chơi được đi trước trong ván cờ 3x3, bạn sẽ đánh vào ô nào, có giống như lựa chọn của máy không?


## Phần 4. Cải thiện thuật toán

Thử lập một ván chơi trên bàn cờ 5 x 5 với $k$ = 4, **MACHINE** đi trước. Bạn có thể đánh giá độ nhanh của thuật toán bằng đoạn code sau (trong lời giải, $N = 1000$, bạn có thể chọn $N = 100$ để đánh giá).


```python
import time
if __name__ == "__main__":
    HUMAN = 1
    MACHINE = 2
    game = Game(5, 5, MACHINE, 4)
    game.mark(MACHINE, (2, 1))
    game.mark(HUMAN, (2, 2))
    game.mark(MACHINE, (1, 2))
    game.mark(HUMAN, (0, 3))
    game.mark(MACHINE, (1, 1))
    game.mark(HUMAN, (1, 0))
    game.draw()
    t = time.time() 
    print game.decideNextStep(MACHINE, Game.algo1)
    print "Algo 1 took ", time.time() - t, " second."

```

```
    -----------
    | | | |X| |
    -----------
    |X|O|O| | |
    -----------
    | |O|X| | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    (3, 1)
    Algo 1 took  18.6069998741  second.
    

Trên thực tế, thuật toán này có tốc độ không tốt.

Ta thay đổi chiến thuật một chút. Ta cảm giác rằng cả **HUMAN** và **MACHINE** đều không nên đánh ở những ô quá xa với những ô đã được đánh, do đó trong thuật toán 1, thay vì tìm tất cả các ô trống trên bàn cờ, ta chỉ tìm những ô trống có ít nhất một lân cận đã được đánh.

Ta gọi các ô trống nằm cạnh một ô không trống của bàn cờ là một ô **tiềm năng**.

Ta có thuật toán dưới đây.

#### Thuật toán 2 #### 

**Input:** Trạng thái hiện tại của ván chơi (tức instant thuộc class Game đang chứa hàm implement thuật toán)

**Output:** Toạ độ (i, j) “tốt nhất” để thắng

- Đối với từng ô **(i, j)** trong tất cả các ô tiềm năng của bàn cờ:
 - Tạo ra $N = 1000$ bản sao như trạng thái hiện tại của ván chơi.
 - Với mỗi bản sao:
  - Đánh vào ô **(i, j)**
  - Tiếp tục ván chơi một cách ngẫu nhiên (đóng vai **HUMAN** đánh vào một ô **tiềm năng**, rồi trở lại vai **MACHINE** đánh vào một ô tiềm **năng khác**…) cho đến khi ván chơi kết thúc
 - Tính hiệu suất chiến thắng của việc đánh vào **(i, j)** trên tất cả các bản sao đã tạo ra. Hiệu suất có thể được tính như cũ:
 
<center>**H(i, j)** =  Số ván thắng – số ván thua</center>
<center>(chú ý số ván thắng + hoà + thua = $N$, nên -$N \leq $ **H(i,j)** $\leq N$.)</center>
 - Sau khi có tất cả **H(i,j)**, chọn **(i,j)** làm **H** lớn nhất.


### Bài 12. Thuật toán 2.

*Viết hàm **algo2()** implement thuật toán 2.*

Chạy lại đoạn code dưới đây để test. (Thời gian chạy có thể tốt hơn thuật toán 1 hoặc không)


```python
import time
if __name__ == "__main__":
    HUMAN = 1
    MACHINE = 2
    game = Game(5, 5, MACHINE, 4)
    game.mark(MACHINE, (2, 1))
    game.mark(HUMAN, (2, 2))
    game.mark(MACHINE, (1, 2))
    game.mark(HUMAN, (0, 3))
    game.mark(MACHINE, (1, 1))
    game.mark(HUMAN, (1, 0))
    game.draw()
    t = time.time() 
    print game.decideNextStep(MACHINE, Game.algo2)
    print "Algo 2 took ", time.time() - t, " second."
```

```
    -----------
    | | | |X| |
    -----------
    |X|O|O| | |
    -----------
    | |O|X| | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    (3, 1)
    Algo 2 took  18.7060000896  second.
    

### Bài 13. Thuật toán 3.

*Viết hàm **algo3()** implement thuật toán 2 ở dạng cho nhiều CPU core chạy đồng thời (dùng multiprocessing.Pool). Mỗi pool đảm nhận một ô trống (là lân cận của một ô đã được đánh).*

Test bằng đoạn code sau. (Thời gian chạy thường tốt hơn nhiều so với thuật toán 1 và 2).


```python
import time
if __name__ == "__main__":
    HUMAN = 1
    MACHINE = 2
    game = Game(5, 5, MACHINE, 4)
    game.mark(MACHINE, (2, 1))
    game.mark(HUMAN, (2, 2))
    game.mark(MACHINE, (1, 2))
    game.mark(HUMAN, (0, 3))
    game.mark(MACHINE, (1, 1))
    game.mark(HUMAN, (1, 0))
    game.draw()
    t = time.time() 
    print game.decideNextStep(MACHINE, Game.algo3)
    print "Algo 3 took ", time.time() - t, " second."
```

```
    -----------
    | | | |X| |
    -----------
    |X|O|O| | |
    -----------
    | |O|X| | |
    -----------
    | | | | | |
    -----------
    | | | | | |
    -----------
 ```   
    (3, 1)
    Algo 3 took  7.24599981308  second.
    

Bạn có thể xây dựng thêm những thuật toán cải thiện hơn để khả năng thắng cao hơn và thời gian chạy nhanh hơn. Thử implement **algo4, 5, 6**… nếu có ý tưởng.
