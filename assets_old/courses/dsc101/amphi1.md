
# Amphi 1 - Introduction to programming in Python

# 1. Numbers

Number in Python can be integers, real numbers and complex numbers.

## 1.1 Integers: int and long 

**int** is used for integers between -2\*\*31 and 2\*\*31 - 1.
**long** may be used for integers out of this range. We can specify "L" at the end of an **int** to force it as **long**.


```python
a = 10
b = 2**32
c = 10L
type(a), type(b), type(c)
```




    (int, long, long)



The largest integer of type **int** can be called by using the constant **maxint** in library **sys**.


```python
import sys
a = sys.maxint
a, type(a)
```




    (2147483647, int)




```python
a == 2**31 - 1
```




    True



An integer operation exceeding the range of **int** will return a **long**. 


```python
a = 2
b = 32
c = a**b
type(a), type(b), type(c)
```




    (int, int, long)



## 1.2 Real numbers: float

A **float** represents a real number.


```python
a = 1.0
b = 6e-5
a, b, type(a), type(b)
```




    (1.0, 6e-05, float, float)



The syntax "%.2f" can be used to print a float rounded upto 2 decimal digits.


```python
print "%.5f, %.4f" % (b, b)
```

    0.00006, 0.0001
    

## 1.3 Complex numbers: complex

Type **complex** represents complex numbers.


```python
a = 2 + 1j
print a, type(a)
```

    (2+1j) <type 'complex'>
    

Some basic attribute and functions of a **complex**.


```python
a.real, a.imag, a.conjugate(), abs(a)
```




    (2.0, 1.0, (2-1j), 2.23606797749979)



For more interesting functions, use library **cmath**.


```python
import cmath
b = 1/2 + cmath.sqrt(3)/2*1j
cmath.phase(b)/cmath.pi
```




    0.5



## 1.4 Boolean: bool

A **bool** can takes 2 possible values **True**, **False**. As a number, they can be considered as 1, 0 respectively.


```python
a = True
b = 2>3
a, b
```




    (True, False)




```python
True + True, False - True, (True + True) * 10, True + 3j
```




    (2, -1, 20, (1+3j))



## 1.5 Type conversion

**bool**, **int**, **long**, **float**, **complex** are numbers in Python. Number of one type can be converted to another type by the following rules:

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/Amphi/Figure1.png" />

Some other functions to convert:


```python
import math
a = round(1.256, 2)
b = math.floor(1.256)
c = math.ceil(1.256)
a, b, c
```




    (1.26, 1.0, 2.0)



## 1.6 Operations on numbers

Type inclusion: **bool** $\subset$ **int** $\subset$ **long** $\subset$ **float** $\subset$ **complex**.

General rule: Operations (+, -, \*, /, \*\*) on 2 numbers return a result of the "bigger" type.


```python
2 + 3.0 - 5L, type(2 + 3.0 - 5L), (3.0 + 2j) ** 2.0, type((3.0 + 2j) ** 2.0), True + 15L, type(True + 15L)
```




    (0.0, float, (5+12j), complex, 16L, long)



**Exception**: Some functions can change the type of variables (like $\log$, $\sin$, $\mathrm{abs}$)


```python
type(abs(2+3j)), type(math.sin(1)), type(math.log(4))
```




    (float, float, float)



**Exception**: Division of integers in Python 2 is integer division, and in Python is float division.


```python
3/2
```




    1



% is used for taking remainder in an integer division


```python
103 % 5
```




    3



**Operations on boolean**


```python
True and False, True or (2 > 3), not (1 == 1)
```




    (False, True, False)



## 1.7 Comparison between numbers

In Python, to compare numbers of different types, we compare them as values of the "bigger" type.


```python
2 == 2.0, True == 1 + 0j, True > 4.0, 3/(2+1j)*(2+1j) == True + True + True
```




    (True, True, False, True)



Attention: Some error may affect the comparison


```python
3/(2+3j)*(2+3j), 3/(2+3j)*(2+3j) == 3
```




    ((3-2.220446049250313e-16j), False)



# 2. String

## 2.1 String: str

A **str** in Python is delimited by "" or ''


```python
a = "Python"
b = 'python'
a, b
```




    ('Python', 'python')



**Get length**


```python
len(a), len(b)
```




    (6, 6)



**Get a character of strings by specifying its index in []. Characters are indexed from 0 to length-1**


```python
a[0], a[1], a[2], b[3], b[4], b[5], b[len(b) - 1]
```




    ('P', 'y', 't', 'h', 'o', 'n', 'n')



**Characters can also be indexed from -length to -1**


```python
a[-6], a[-5], a[-4], b[-3], b[-2], b[-1]
```




    ('P', 'y', 't', 'h', 'o', 'n')



**Get substring**


```python
a[1:4], a[3:], b[:2], b[:-1]
```




    ('yth', 'hon', 'py', 'pytho')



**A str is immutable. We can't modify characters of it.**


```python
#a[0] = "Q"
#TypeError: 'str' object does not support item assignment
```

## 2.2 Operations on strings

**Concatenation**


```python
a = "Python"
b = "python"
c = a + " " + b
c
```




    'Python python'



**Split string to words**


```python
c.split(" ")
```




    ['Python', 'python']



**Split string to characters.**


```python
list(c)
```




    ['P', 'y', 't', 'h', 'o', 'n', ' ', 'p', 'y', 't', 'h', 'o', 'n']



**Find the first position of a substring in a string (if not found, return -1)**


```python
c.find("tho"), c.find("THO"), c.rfind("tho")
```




    (2, -1, 9)



**Find number of occurence of a character or substring**


```python
c.count("ho"), c.count("H")
```




    (2, 0)



**Capitalization**


```python
d = c.lower()
e = d.upper()
f = d.capitalize()
d, e, f
```




    ('python python', 'PYTHON PYTHON', 'Python python')



**Replace a substring**


```python
c.replace("ho", "HO")
```




    'PytHOn pytHOn'



**Join a list to one string, using delimitor**


```python
"--".join(c)
```




    'P--y--t--h--o--n-- --p--y--t--h--o--n'



## 2.3 Special string literals

"\t", "\n" are used for tab and newline. 


```python
s = "1\t2\t3\n4\t5\t6\n"
s
```




    '1\t2\t3\n4\t5\t6\n'




```python
print s
```

    1	2	3
    4	5	6
    
    

Other characters: 

| Escape Sequence  | Meaning                              |
|------------------|--------------------------------------|
| \newline         | Ignored                              |
| \\               | Backslash (\)                        |
| \'               | Single quote (')                     |
| \"               | Double quote (")                     |
| \a               | ASCII Bell (BEL)                     |
| \b               | ASCII Backspace (BS)                 |
| \f               | ASCII Formfeed (FF)                  |
| \n               | ASCII Linefeed (LF)                  |
| \r               | ASCII Carriage Return (CR)           |
| \t               | ASCII Horizontal Tab (TAB)           |
| \v               | ASCII Vertical Tab (VT)              |
| \ooo             | ASCII character with octal value ooo |
| \xhh...          | ASCII character with hex value hh... |

## 2.4 Read from and write to files

**Read everything as a string**


```python
myfile = open("text.txt", 'r')
text = myfile.read()
print text
myfile.close()
```

    Line 1, Column 1, Column 2
    Line 2, Column 1, Column 2
    Line 3, Column 1, Column 2
    Line 4, Column 1, Column 2
    

**Read everything as a list of lines**


```python
myfile = open("text.txt", 'r')
text = myfile.readlines()
print text
myfile.close()
```

    ['Line 1, Column 1, Column 2\n', 'Line 2, Column 1, Column 2\n', 'Line 3, Column 1, Column 2\n', 'Line 4, Column 1, Column 2']
    

**Read line by line**


```python
myfile = open("text.txt", 'r')
line = myfile.readline()
print line
```

    Line 1, Column 1, Column 2
    
    


```python
line = myfile.readline()
print line
```

    Line 2, Column 1, Column 2
    
    


```python
myfile.close()
```

**Write to file**


```python
myfile = open("new_text.txt", 'w')
myfile.write('Line 1, Column 1, Column 2\n')
myfile.write('Line 2, Column 1, Column 2\n')
myfile.close()
```


```python
myfile = open("new_text.txt", 'r')
text = myfile.read()
print text
myfile.close()
```

    Line 1, Column 1, Column 2
    Line 2, Column 1, Column 2
    
    

**Add to the end of file, without overriding**


```python
myfile = open("new_text.txt", 'a')
myfile.write('Line 3, Column 1, Column 2\n')
myfile.write('Line 4, Column 1, Column 2\n')
myfile.close()
```


```python
myfile = open("new_text.txt", 'r')
text = myfile.read()
print text
myfile.close()
```

    Line 1, Column 1, Column 2
    Line 2, Column 1, Column 2
    Line 3, Column 1, Column 2
    Line 4, Column 1, Column 2
    
    

# 3. Iterables: list, tuple, set and dict

An iterable type represents object with several similar components.

The most popular iterable types in Python are **list**, **tuple**, **set** and **dict**.

If **w** is an iterable object, we can retrieve its component using a **for** loop.


```python
w = [1, 2, 3, 4, 5, 6, "last"]
print type(w)
for x in w:
    print x
```

    <type 'list'>
    1
    2
    3
    4
    5
    6
    last
    

## 3.1 list


```python
#list, indices, mutability, operation on list, sort a list, range, loop in a list
```

Python recognizes an object between "[" and "]" size as a list.


```python
my_list = [1, 'P', 5.0, 6-2j, "abc"]
type(my_list)
```




    list



Elements (components) are indexed like in **str** (from 0 to length-1, or from -length to -1)


```python
my_list[0], my_list[-len(my_list) + 1], my_list[1:3], my_list[2:]
```




    (1, 'P', ['P', 5.0], [5.0, (6-2j), 'abc'])



### 3.1.1 list is mutable

That means we can modify their elements without creating a new copy of the list.


```python
L = [0, 1, 2, 3, 4, 5]
L[0] = -1
L
```




    [-1, 1, 2, 3, 4, 5]



This is very useful for deleting elements.


```python
L = [0, 1, 2, 3, 4, 5]
L[1:4] = []
L
```




    [0, 4, 5]



or for insertion


```python
L = [0, 1, 2, 3, 4, 5]
L[1:2] = [-1, L[1]]
L
```




    [0, -1, 1, 2, 3, 4, 5]




```python
L =  [0, 1, 2, 3, 4, 5]
L[len(L):] = [6]
L
```




    [0, 1, 2, 3, 4, 5, 6]



But pay attention to your code if there is any copy of the list.


```python
L = [0, 1, 2, 3, 4, 5]
M = [0, 1, 2, 3, 4, 5]
L[0] = -1
M
```




    [0, 1, 2, 3, 4, 5]




```python
L = [0, 1, 2, 3, 4, 5]
M = L
L[0] = -1
M
```




    [-1, 1, 2, 3, 4, 5]



### 3.1.2 Operations on list

**Concatenation**


```python
L = [1, 2, 3, 4, 5, 6]
M = [7, 8]
K = L + M
K
```




    [1, 2, 3, 4, 5, 6, 7, 8]



**Append an element to the end**


```python
L = [1, 2, 3, 4, 5, 6]
L.append(7) #Like L = L + [7]
L
```




    [1, 2, 3, 4, 5, 6, 7]



**Insertion**


```python
L = [1, 2, 3, 4, 5, 6]
L.insert(2, 7) #Like L = L[:2] + [7] + L[2:]
L
```




    [1, 2, 7, 3, 4, 5, 6]



**Extension**


```python
L = [1, 2, 3, 4, 5, 6]
L.extend([7, 8]) #Like L = L + [7, 8]
L
```




    [1, 2, 3, 4, 5, 6, 7, 8]



**Sorting in ascending order**


```python
L = [2, 3, 1, 6, 4, 5]
L.sort()
L
```




    [1, 2, 3, 4, 5, 6]



**Another way to sort**


```python
L = [2, 3, 1, 6, 4, 5]
sorted(L)
```




    [1, 2, 3, 4, 5, 6]



**Reverse the list**


```python
L = [2, 3, 1, 6, 4, 5]
L.reverse()
L
```




    [5, 4, 6, 1, 3, 2]



**Delete an element**


```python
L = [1, 2, 3, 4, 5, 6]
a = L.pop(2)
L, a
```




    ([1, 2, 4, 5, 6], 3)



**Delete the last element**


```python
L = [1, 2, 3, 4, 5, 6]
a = L.pop()
L, a
```




    ([1, 2, 3, 4, 5], 6)



**max, min, sum of a list**


```python
L = [1, 2, 3, 4, 5, 6]
max(L), min(L), sum(L)
```




    (6, 1, 21)



**Count the number of ocurrences**


```python
L = ['a', 'ab', ['a'], 'a', ['a'], 'aa', 'a']
L.count('a'), L.count(['a'])
```




    (3, 2)



**Duplicate elements**


```python
L = [1, 2]
L*3
```




    [1, 2, 1, 2, 1, 2]



**Check if element in list**


```python
L = ['a', 'ab', ['a'], 'a', ['a'], 'aa', 'a']
['a'] in L, 'A' in L #Like L.count(['a']) > 0, L.count('A') > 0
```




    (True, False)



### 3.1.3 List of list

Elements of a **list** can be any object, eventually a **list**. In this case, **a\[i\]** is a list. To access to the **j**-th element of the **i**-th element of **a**, we can use **a\[i\]\[j\]**.


```python
L = [[1, 2, 3], [4, 5, 6]]
L[0][1]
```




    2



We can use list of list to describe a table of $m$ rows and $n$ columns. Row and column can be access as follows: 


```python
L = [[1, 2, 3], [4, 5, 6]]
L[0]
```




    [1, 2, 3]




```python
[row[0] for row in L]
```




    [1, 4]



## 3.2 tuple

Python recognize an object between "\(" and "\)" as a tuple. Indexing of tuple is like list.


```python
T = (1, 2, 3, 4)
type(T), len(T), T[-3], T[1:]
```




    (tuple, 4, 2, (2, 3, 4))



When we print several objects to console delimited by a ",", the tuple of these objects is printed.


```python
1, [2, 3], True
```




    (1, [2, 3], True)



### 3.2.1 tuple is immutable


```python
T = (1, 2, 3, 4)
#The following line should raise an error: 'tuple' object does not support item assignment
#T[2] = 2
```

### 3.2.2 Operations on tuple

**Concatenation**


```python
T = (1, 2, 3, 4)
print T + T
print T * 0
print T * 3
```

    (1, 2, 3, 4, 1, 2, 3, 4)
    ()
    (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)
    

**Get max, min, sum**


```python
T = (1, 2, 3, 4)
max(T), min(T), sum(T)
```




    (4, 1, 10)



**Check element in tuple**


```python
T = (1, 2, 3, 4)
2 in T
```




    True



## 3.3 set

Python recognizes a nonempty object delimited by {} which does not contain ":" as a **set**.


```python
S = {1, 2, 'a', 3, 3}
print type(S)
print S
```

    <type 'set'>
    set(['a', 1, 2, 3])
    

**Comparison of set in mathematical sense** (with number comparison)


```python
S = {1, 2, 'a', 4.0}
R = {'a', 4, True, (2+0j)}
S == R
```




    True



### 3.3.1 Set is mutable. Operations


```python
R = {1, 2, 3, 4, 5}
S = {1, 2, 6}
R.update(S)
print R
```

    set([1, 2, 3, 4, 5, 6])
    

**Add an element to a set**


```python
R = {1, 2, 3, 4, 5}
R.add(10)
R
```




    {1, 2, 3, 4, 5, 10}



**Remove an existing element from the set (raise error if not existing)**


```python
R = {1, 2, 3, 4, 5}
R.remove(5)
R
# The following line should raise an error KeyError
# R.remove(6)
```




    {1, 2, 3, 4}



**Remove an element from a set, if it is in the set**


```python
R = {1, 2, 3, 4, 5}
R.discard(5)
R
# The following line raises no error 
R.discard(6)
```

**Clear all elements of a set**


```python
R = set([1, 2, 3, 4, 5]) #like {1, 2, 3, 4, 5}
R.clear()
R
```




    set()



**Get size of a set**


```python
R = {1, 2, 3, 4, 5, 4}
len(R)
```




    5



**Check subset relation**


```python
R = set([1, 2, 3, 4, 5])
S = {1, 4}
S.issubset(R), R.issuperset(S), R.issubset(S)
```




    (True, True, False)



**Check appartenance relation**


```python
R = set([1, 2, 3, 4, 5])
2 in R, 0 in R
```




    (True, False)



**Union, Intersection, Difference, Symmetric difference**


```python
R = {1, 2, 3, 4, 5}
S = {1, 6}
R | S, R & S, R - S, R ^ S
```




    ({1, 2, 3, 4, 5, 6}, {1}, {2, 3, 4, 5}, {2, 3, 4, 5, 6})



### 3.3.2 frozenset

An immutable version of **set** is **frozenset**. Operations are the same as **set** but operations that can changes elements of the set are not allowed.


```python
R = frozenset([1, 2, 3])
S = frozenset([1, 4])
R|S
# The following line should raise an error.
# R.add(4)
```




    frozenset({1, 2, 3, 4})



## 3.4 dict

Python recognizes an object delimited by "{", "}" containing ":", or an empty {}, as a **dict** (**dict** stands for dictionary). A **dict** is described as several $key$:$value$ pairs wrapped in {}, where $key$ is a **str** and $value$ may be of any type.


```python
A = {}
B = {"x": 1, "y": [2, (3, 5)]}
C = {"x"}
type(A), type(B), type(C)
```




    (dict, dict, set)



Key duplication is not allow. The last information on a key will be taken into account.


```python
A = {"x": 1, "y": [2, (3, 5)], "x": 2.0}
A
```




    {'x': 2.0, 'y': [2, (3, 5)]}



Access to values of **dict** by using the key.


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1["name"], person1["family"]["brothers"], person1["age"]
```




    ('Peter', ['Fred', 'Raoul'], 17)



### 3.4.1 dict is mutable

We can change the value corresponding to a key.


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1["family"]["brothers"] = ["Fred", "Raoul", "Bob"]
person1
```




    {'age': 17,
     'family': {'brothers': ['Fred', 'Raoul', 'Bob'],
      'dad': 'John',
      'mum': 'Daisy'},
     'name': 'Peter'}




```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1["family"]["brothers"].append("David")
person1
```




    {'age': 17,
     'family': {'brothers': ['Fred', 'Raoul', 'David'],
      'dad': 'John',
      'mum': 'Daisy'},
     'name': 'Peter'}



### 3.4.2 Operations on dict

**Add a new (key, value) pair or update value to key**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1["nationality"] = "USA"
person1["age"] = 18 
person1
```




    {'age': 18,
     'family': {'brothers': ['Fred', 'Raoul'], 'dad': 'John', 'mum': 'Daisy'},
     'name': 'Peter',
     'nationality': 'USA'}



**Remove a (key, value) pair**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
del person1["age"]
person1
```




    {'family': {'brothers': ['Fred', 'Raoul'], 'dad': 'John', 'mum': 'Daisy'},
     'name': 'Peter'}



**Size of the dictionary**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
len(person1), len(person1["family"])
```




    (3, 3)



**Get all keys into a list**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1.keys()
```




    ['age', 'name', 'family']



**Get all values into a list**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1.values()
```




    [17, 'Peter', {'brothers': ['Fred', 'Raoul'], 'dad': 'John', 'mum': 'Daisy'}]



**Get all (key, value) pair into a list**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
person1.items()
```




    [('age', 17),
     ('name', 'Peter'),
     ('family', {'brothers': ['Fred', 'Raoul'], 'dad': 'John', 'mum': 'Daisy'})]



**Looping in a dictionary = Looping in its key list**


```python
person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
for x in person1:
    print x
print "-----------"
for x in person1:
    print person1[x]
```

    age
    name
    family
    -----------
    17
    Peter
    {'dad': 'John', 'mum': 'Daisy', 'brothers': ['Fred', 'Raoul']}
    

### 3.4.3 Hashing

The list representation equivalence of dictionary is $[(key1, value1), (key2, value2), ..., (keyN, valueN)]$. 

It is not optimized for searching.

For example, if
person1 = [("name": "Peter"), ..., ("age": 17), ...] 
has lots of components and we need to find the age of person1. Then we have to scan many elements before reaching "age".


**Idea for dictionary mechanism**
We compute a function $f$ of each key$ k, f(k)$, where
$f$ is a kind of "random" function so that it is quasi-injective on all possible values of $k$.

When the dictionary is created, we associate $f(k)$ to a position in the memory, and store (key, value) in that position.
Example:

- $f$("name") = 5034981 -> store ("name", "Peter") to memory zone 0x1111 and associate 0x1111 to 15034981. If 0x1111 has been occupied, it will try 0x1112, 0x1113 (not so long to find a free zone)

- ...

- $f$("age") = -1597925398 -> store ("age", 17) to memory zone 0x1263 and associate 0x1263 to -1597925398. If 0x1263 has been occupied, it will try 0x1263, 0x1264

The association table will be:

| f()         |Zone    |
|-------------|--------|
| -1597925398 | 0x1111 |
| -998372342 | 0x1812 |
|...        |...     |
| 15034981 | 0x1263 |
| 27738428 | 0x1374 |

Searching in this association table is fast (take $O(\log N)$ instead of $O(N)$) because the table is sorted and we works with integer instead of string.

So when **person1["age"**] is called, $f$("age") is computed, then the association table gives 0x1263 as the position in the memory. If $f$ is injective, the value of age is found immediately and return to **person1["age"]**. If $f$ is quasi-injective, some further scan to the following position like 0x1264, 0x1265 may be needed but it will not take long.

An example for a hashing mechanism in Python is the function **hash**.


```python
hash("name"), hash("age")
```




    (15034981, -1597925398)



## 3.5 Type conversion

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson1/Amphi/Figure2.png" />

## 3.6 Comparison

*This paragraph is copied from the official python documentation page* (https://docs.python.org/2/tutorial/datastructures.html#comparing-sequences-and-other-types)

Sequence objects may be compared to other objects with the same sequence type. The comparison uses lexicographical ordering: first the first two items are compared, and if they differ this determines the outcome of the comparison; if they are equal, the next two items are compared, and so on, until either sequence is exhausted. 

If two items to be compared are themselves sequences of the same type, the lexicographical comparison is carried out recursively. 

If all items of two sequences compare equal, the sequences are considered equal. 

If one sequence is an initial sub-sequence of the other, the shorter sequence is the smaller (lesser) one. Lexicographical ordering for strings uses the ASCII ordering for individual characters. Some examples of comparisons between sequences of the same type:


```python
a = (1, 2, 3)              < (1, 2, 4)
b = [1, 2, 3]              < [1, 2, 4]
c1 = 'ABC' < 'C' 
c2 = 'C' < 'Pascal' 
c3 = 'Pascal'< 'Python'
d = (1, 2, 3, 4)           < (1, 2, 4)
e = (1, 2)                 < (1, 2, -1)
f = (1, 2, 3)             == (1.0, 2.0, 3.0)
g = (1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4)
a, b, c1, c2, c3, d, e, f, g
```




    (True, True, True, True, True, True, True, True, True)



# 4. Functions

## 4.1 Define a function

A function may return something.


```python
def square(x):
    return x**2

square((1j))
```




    (-1+0j)



It may also modify its arguments (if the argument is mutable). Without **return**, a function will return **None**.


```python
def add_1_to_the_end(x):
    x.append(1)

L = [2, 3, []]
print add_1_to_the_end(L)
print L
```

    None
    [2, 3, [], 1]
    

## 4.2 lambda

Sometimes we want to write a function in a shorter way. For example, we want to sort a list of dictionaries by the value of the key "age" in ascending order.


```python
def get_age(my_dict):
    return my_dict["age"]

person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
get_age(person1)
```




    17



It can be simplified by:


```python
get_age = lambda my_dict: my_dict["age"]

person1 = {"name": "Peter", "age": 17, "family": {"dad": "John", "mum": "Daisy", "brothers": ["Fred", "Raoul"]}}
get_age(person1)
```




    17



## 4.3 filter, map, reduce

**filter** is used to collect all elements of a list satisfying a condition.


```python
filter(lambda x: int(math.sqrt(x))**2 == x, range(1, 100))
```




    [1, 4, 9, 16, 25, 36, 49, 64, 81]



**map** is used to apply a mapping to all elements of a list.


```python
map(lambda x: x**2, range(1, 10))
```




    [1, 4, 9, 16, 25, 36, 49, 64, 81]



**reduce** apply a binary operation to 2 first elements of a list, then the result with the 3rd, then the result with the 4th.


```python
reduce(lambda x, y: x + y**2, [0, 1, 2, 3, 4, 5])
```




    55



## 4.4 sorted

We can use the keyword **key** to specify which function is used to determine the order in a list, then sort in that order.


```python
D = [{"name": "Robert", "age": 15}, {"name": "Benoit", "age": 19}, {"name": "Loic", "age": 18}]
sorted(D, key=lambda my_dict: my_dict["age"])
```




    [{'age': 15, 'name': 'Robert'},
     {'age': 18, 'name': 'Loic'},
     {'age': 19, 'name': 'Benoit'}]



# 5. Decision and looping

## 5.1 if

**if**, **elif** and **else** are used to take decision.


```python
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -sign(-x)

sign(-1.5)
```




    -1



## 5.2 for

**for** is used when we know how many iterations are needed in a loop


```python
def isPrime(a):
    result = True
    if a == 1:
        result = False
    for i in range(2, a - 1):
        if a % i == 0:
            result = False
    return result          

def find_prime(a, b):
    for i in range(a, b):
        if isPrime(i):
            print i
    
find_prime(1000, 1020)
```

    1009
    1013
    1019
    

## 5.3 while

**while** is used when we know the condition to end the loop.


```python
def isPrime(a):
    result = True
    if a == 1:
        result = False
    for i in range(2, a - 1):
        if a % i == 0:
            result = False
    return result    

def find_prime_2(a, b):
    i = a
    while i < b:
        if isPrime(i):
            print i
        i += 1
            
find_prime_2(1000, 1020)
```

    1009
    1013
    1019
    

## 5.4 return

**return** can be used in the middle of a function to force it to finish.


```python
def isPrime(a):
    result = True
    if a == 1:
        result = False
    for i in range(2, a - 1):
        if a % i == 0:
            result = False
    return result          

def find_one_prime(a, b):
    for i in range(a, b):
        if isPrime(i):
            print i
            return i
    
find_one_prime(1000, 1020)
```

    1009
    




    1009



## 5.5 break

**break** is used to force a loop to end.


```python
def isPrime(a):
    result = True
    if a == 1:
        result = False
    for i in range(2, a - 1):
        if a % i == 0:
            print a, " is divisible by ", i
            result = False
            break
    return result          

def find_one_prime(a, b):
    for i in range(a, b):
        if isPrime(i):
            print i
            return i
    
find_one_prime(1000, 1020)
```

    1000  is divisible by  2
    1001  is divisible by  7
    1002  is divisible by  2
    1003  is divisible by  17
    1004  is divisible by  2
    1005  is divisible by  3
    1006  is divisible by  2
    1007  is divisible by  19
    1008  is divisible by  2
    1009
    




    1009



It only affects the smallest loop


```python
def find_one_prime(a, b):
    for i in range(a, b):
        isPrime = True
        for j in range(2, a):
            if i % j == 0:
                print i, " is divisible by ", j
                isPrime = False
                break
        if isPrime:
            print i
            return
    
find_one_prime(1000, 1020)
```

    1000  is divisible by  2
    1001  is divisible by  7
    1002  is divisible by  2
    1003  is divisible by  17
    1004  is divisible by  2
    1005  is divisible by  3
    1006  is divisible by  2
    1007  is divisible by  19
    1008  is divisible by  2
    1009
    

## 5.6 continue

**continue** is used to end the current iteration and turn to the next one.


```python
def find_one_prime(a, b):
    for i in range(a, b):
        if i % 2 == 0:  
            print i, " is divisible by 2"
            continue
        if i % 3 == 0:
            print i, " is divisible by 3"
            continue
        if i % 5 == 0:
            print i, " is divisible by 5"
            continue
        if i % 7 == 0:
            print i, " is divisible by 7"
            continue
        if i % 17 == 0:
            print i, " is divisible by 17"
            continue
        if i % 19 == 0:
            print i, " is divisible by 19"
            continue
        print i
        return
    
find_one_prime(1000, 1020)
#But it is wrong
```

    1000  is divisible by 2
    1001  is divisible by 7
    1002  is divisible by 2
    1003  is divisible by 17
    1004  is divisible by 2
    1005  is divisible by 3
    1006  is divisible by 2
    1007  is divisible by 19
    1008  is divisible by 2
    1009
    

## 5.7 enumerate

**enumerate** is used when the index of the list is also important.


```python
data = ['A', 'B', 'C']
for i, x in enumerate(data):
    print i, x
```

    0 A
    1 B
    2 C
    

## 5.8 zip

**zip** is used to pair elements of the same index in 2 list


```python
A = [1, 2, 3]
B = [5, 6, 7]
for (x, y) in zip(A, B):
    print x + y
```

    6
    8
    10
    

# 6. Modules

When you write your code to a file Myfife.py, it becomes a module. To use it for coding in another file, import it by:


```python
import Myfile
```

or if you want to import just some functions, constants,...


```python
from Myfile import myfunction, MYCONSTANT
```
