# Python科学计算基础

## 1.1 numpy库入门

``NumPy`` 扩展库  定义了由同类型的元素组成的多维数组``ndarray``及其常用运算，``ndarray`` 是科学计算中最常用的数据类型。NumPy 数组相对列表的优势是运算速度更快和占用内存更少。``ndarray``是一个类，它的别名是``array``。它的主要属性包括``ndim``(维数)、``shape``(形状，即由每个维度的长度构成的元组) 、``size``(元素数量) 和``dtype``(元素类型：可以是 Python 的内置类型，也可以是 NumPy定义的类型，例如``numpy.int32``、``numpy.int16`` 和 ``numpy.float64`` 等)。

### 1.1.1 ``ndarry``对象

#### - 数组的创建及属性

+ 数组的创建使用``array()``函数，创建过程中向函数传递序列对象，如果传递多层嵌套序列则创建多维数组
  
  ```python
  a=np.array([1,2,3,4])
  b=np.array((1,2,3,4))
  c=np.array([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
  ```

+ 数组的形状可以通过``shape``(描述数组各个轴的长度的元组)属性获得；同时还可以通过修改数组的``shape``,在保持数组元素个数不变的情况下改变数组每个轴的长度(不是转置);当设置``shape``中某个轴的元素个数为-1时，将自动计算此轴的长度
  
  ```python
  --snip--
  print(a.shape,b.shape,c.shape) # (4,) (4,) (3,4)
  c.shape=4,3
  print(c)
  c.shape=2,-1
  print(c)
  """
  (4,) (4,) (3, 4)
  [[1 2 3]
   [4 4 5]
   [6 7 7]
   [8 9 0]]
  [[1 2 3 4 4 5]
   [6 7 7 8 9 0]]
  """
  ```

+ 数组的``reshape()``方法可以返回新的指定形状的数组，且新数组与原数组共享存储空间，同步变化

+ 数组的元素类型可以通过``detype``属性获得，也可以通过``dtype``参数在创建数组吋指定元素类型(在需要指定``dtype``参数时，也可以传递字符串来表示元素的数值类型。``NumPy``中的每个数值类型都有几种字符串表示方式，字符串和类型之间的对应关系都存储在``typeDict``字典中),使用``astype()``方法可以对数组的元素类型进行转换
  
  ```python
  --snip--
  
  print(c.dtype)
  
  e=np.array([1,2,3,4],dtype=float)
  
  f=e.astype(np.int32)
  
  print(e.dtype,f.dtype)
  
  print(e,f)
  
  """
  
  int32
  
  float64 int32
  
  [1. 2. 3. 4.] [1 2 3 4]
  
  """
  ```

+ ``NumPy``有自己的数据类型，可以通过其数据类型建立对象，``NumPy``的数值对象的运算速度比Python的内置类型的运算速度慢很多，如果程序中需要大量地对单个数值运算，应当尽量避免使用``NumPy``的数值对象
  
  ```python
  import numpy as np 
  import time 
  a = 3.14 
  b = np.float64(a) 
  print(a, b) 
  starttime = time.time() 
  print(pow(a * 5, 55)) 
  endtime = time.time() 
  print(endtime - starttime) 
  starttime = time.time() 
  print(pow(b * 5, 55)) 
  endtime = time.time() 
  print(endtime - starttime) 
  """ 
  3.14159 3.14159 
  6.117487080542922e+65 
  0.0009829998016357422 
  6.117487080542922e+65 
  0.0 
  """ 
  ```

#### - 自动生成数组

+ ``arange()``函数通过指定开始值、终值和步长来创建表示等差数列的一维数组，所得到的结果中不包含终值

+ ``linspace()``通过指定幵始值、终值和元素个数来创建表示等差数列的一维数组，可以通过``endpoint``参数指定是否包含终值，默认值为``True``(包含终值)

+ ``logspace()``可以通过指定初始值、终值、元素个数创建等比数列，参数``base``(默认为10)和``endpoint``分别指定底数和是否包含终值 

+ ``zeros()``、``ones()``、``empty()``、``full()``可以创建指定形状和类型的数组。其中``empty()``只分配数组所使用的内存，不对数组元素进行初始化操作；``zeros()``将数组元素初始化为0, ``ones()``将数组元素初始化为1；``full()``将元素初始化为指定的值
  
  ```python
  print(np.arange(1, 10, 1)) 
  print(np.linspace(1, 10, 10, endpoint=True)) 
  print(np.linspace(1, 10, 10, endpoint=False)) 
  print(np.logspace(1, 5, 5, base=3, endpoint=True)) 
  print(np.zeros((2, 3), np.float32)) 
  print(np.ones((2, 3), np.float32)) 
  print(np.empty((2, 3), np.float32)) 
  print(np.full((2, 3), 1)) 
  """ 
  [1 2 3 4 5 6 7 8 9] 
  [ 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.] 
  [1. 1.9 2.8 3.7 4.6 5.5 6.4 7.3 8.2 9.1] 
  [ 3. 9. 27. 81. 243.] 
  [[0. 0. 0.] [0. 0. 0.]] 
  [[1. 1. 1.] [1. 1. 1.]] 
  [[1. 1. 1.] [1. 1. 1.]] 
  [[1 1 1] [1 1 1]]
  """
  ```

+ ``zeros_like()``、``ones_like()``、``empty_like()``、``full_like()``等函数创建与参数数组的形状和类型相同的数组，因此``zeroslike(a)``和 ``zeros(a.shape，a.dtype)``的效果相同

+ ``frombuffer()``、``fromstring()``、``fromfile()``等函数可以从字节序列或文件创建数组，此时数组内的元素则按照元素类型有所不同：Python的字符串是一个字节序列，每个字符占一个字节，因此如果从字符串创建一个8位的整数数组，所得到的数组正好就是字符串中每个字符的ASCII编码([字符、位关系详见该文档](https://blog.csdn.net/qq_41675254/article/details/86481615))；如果从字符串创建16位的整数数组，那么两个相邻的字节就表示一个整数，且16位的整数是以低位字节在前的方式保存在内存中的([关于位和大小端的概念见该文档](https://zhuanlan.zhihu.com/p/510962688))；``fromstring()``会对字符串的字节序列进行复制，而使用``frombuffer()``创建的数组与原始字符串共享内存。由于字符串是只读的，因此无法修改所创建的数组的内容
  
  ```python
  a = np.fromstring(s, dtype=np.int8) 
  b = np.fromstring(s, dtype=np.int16) 
  c = np.fromstring(s, dtype=np.float64) 
  print(a,b,c) 
  """ [ 97 98 99 100 101 102 103 104] 
  [25185 25699 26213 26727] 
  [8.54088322e+194] 
  """
  ```

+ ``fromfunction()``通过函数返回值创建数组，其第一个参数是计算每个数组元素的函数，第二个参数指定数组的形状。它支持多维数组，第二个参数必须是一个序列，在调用函数时并没有``shape``的各个坐标迭代多次调用函数，**实际上只调用了一次函数**([关于传入数组问题详见该文章](https://www.zhihu.com/question/33561391))
  
  ```python
  def fun_1(i, j):
      return (i + 1) * (j + 1)
  
  a = np.fromfunction(fun_1, (9, 9))
  print(a)
  
  """
  [[ 1.  2.  3.  4.  5.  6.  7.  8.  9.]
   [ 2.  4.  6.  8. 10. 12. 14. 16. 18.]
   [ 3.  6.  9. 12. 15. 18. 21. 24. 27.]
   [ 4.  8. 12. 16. 20. 24. 28. 32. 36.]
   [ 5. 10. 15. 20. 25. 30. 35. 40. 45.]
   [ 6. 12. 18. 24. 30. 36. 42. 48. 54.]
   [ 7. 14. 21. 28. 35. 42. 49. 56. 63.]
   [ 8. 16. 24. 32. 40. 48. 56. 64. 72.]
   [ 9. 18. 27. 36. 45. 54. 63. 72. 81.]]
  """
  ```

#### - 数组元素存取

+ ``NumPy``数组的简单存取与列表相似，可以使用下标存取、修改和切片；切片使用``a[a1:a2:a3]``语言，其中``a1`` 表示开始下标，``a2``表示结束下标(切片时不包括) ，``a3``表示步长；下标可负，表示从末尾开始技术；步长可负，表示数组首尾颠倒；省略切片的开始下标和结束下标且步长为-1 ,可使整个数组头尾颠倒；和列表不同的是，通过切片获取的新的数组是原始数组的一个视图。它与原始数组共享同一块数椐存储空间 ，修改其中一个会修改另一个
  
  ```python
  a = np.array([0, 1, 2, 3, 4, 5, 6])
  print(a[2])
  print(a[0:3])
  print(a[:5])
  print(a[:-2])
  print(a[1:5:2])
  print(a[::-1])
  a[2:4] = 3, 4
  print(a)
  b = a[2:4]
  print(b)
  b[0:2] = 2, 3
  print(a, b)
  """
  2
  [0 1 2]
  [0 1 2 3 4]
  [0 1 2 3 4]
  [1 3]
  [6 5 4 3 2 1 0]
  [0 1 3 4 4 5 6]
  [3 4]
  [0 1 2 3 4 5 6] [2 3]
  """
  ```

+ ``NumPy``可以使用整数列表对数组元素进行存取，此时将使用列表中的每个元素(可为负)作为下标。使用列表作为下标得到的数组不和原始数组共享数据，同样具有下标法的方法。

+ ``NumPy``也可以使用整数数组作为数组下标，得到形状和下标数组相同的新数组，新数组的每个元素都是用下标数组中对应位置的值作为下标从原数组获得的值。当下标数组是一维数组时，结果和用列表作为下标的结果相同；当下标是多维数组时，得到的也是多维数组
  
  ```python
  a = np.array([0, 1, 2, 3, 4, 5, 6])
  b = a[[1, 2, 3, 4]]
  print(b)
  a[[0, 1, 2]] = 1, 2, 3
  c = a[np.array([1, 1, 3, 2, 4])]
  print(c)
  d = a[np.array([[1, 2, 3], [2, 3, 4]])]
  print(d)
  """
  [1 2 3 4]
  [2 2 3 3 4]
  [[2 3 3]
   [3 3 4]]
  """
  ```

+ ``NumPy``可以使用布尔数组作为下标存取和修改数组中的元素，此时获得原数组中与布尔数组中``True``对应的元素；使用布尔数组作为下标获得的数组不和原始数组共享数据内存，注意这种方式在``NumPy 1.10``之前版本只对应于布尔数组，不能使用布尔列表，如果是布尔列表，就把``True``当作1, 把``False``当作0 , 按照整数序列方式获取元素，在``NumPy 1.10``之后布尔列表与布尔数组输出相同
  
  ```python
  a = np.array([0, 1, 2])
  
  print(a[np.array([True, True, False])])
  c = a[[True, True, False]]
  print(c)
  
  b = np.logspace(1, 10, 5, base=3)
  print(b[b > 3.5])
  
  """
  [0 1]
  [0 1]
  [3.55339983e+01 4.20888346e+02 4.98528193e+03 5.90490000e+04]
  """
  ```

#### -数组的堆叠与分割

``NumPy``中``np.hstack``函数沿第一轴将两个数组堆叠在一起形成新的数组，``np.vstack``函数沿第零轴将两个数组堆叠在一起形成新的数组。``np.r_``函数将多个数组(数值)堆叠在一起形成新的数组

```python
#堆叠
a = np.arange(1, 7).reshape(2, -1)
b = np.arange(7, 13).reshape(2, -1)

c = np.hstack((a, b))
d = np.vstack((a, b))

print(c)
print(d)

"""
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
"""
#分割
a = np.r_[np.array([1, 3, 7]), 0, 8:2:-2, 0] #start:stop:step 等同于 np.arange(start, stop,step)
print(a)
# [1 3 7 0 8 6 4 0]
a = np.r_[-1:2:6j, [1] * 2, 5] # start:stop:numj 等同于 np.linspace(start, stop, num)
print(a)
# [-1.  -0.4  0.2  0.8  1.4  2.   1.   1.   5. ]
```

``np.hsplit``函数沿第一轴将一个数组分割成为多个数组，可以指定一个正整数表示均匀分割得到的数组的数量或指定一个元组表示各分割点的索引值。``np.vsplit`` 函数沿第零轴将一个数组分割成为多个数组，参数和``hsplit``类似

```python
c = np.arange(1, 25).reshape(2, 12)
print(c)
print(np.hsplit(c, 4))
print(np.hsplit(c, (4,7,9)))

"""
[[ 1  2  3  4  5  6  7  8  9 10 11 12]
 [13 14 15 16 17 18 19 20 21 22 23 24]]
[array([[ 1,  2,  3],
       [13, 14, 15]]), array([[ 4,  5,  6],
       [16, 17, 18]]), array([[ 7,  8,  9],
       [19, 20, 21]]), array([[10, 11, 12],
       [22, 23,24]])]
[array([[ 1,  2,  3,  4],
       [13, 14, 15, 16]]), array([[ 5,  6,  7],
       [17, 18, 19]]), array([[ 8,  9],
       [20, 21]]), array([[10, 11, 12],
       [22, 23, 24]])]
"""
c = np.arange(1, 25).reshape(6, 4)
print(c)
print(np.vsplit(c, (2, 3, 5)))

"""
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]]
[array([[1, 2, 3, 4],
       [5, 6, 7, 8]]), array([[ 9, 10, 11, 12]]), array([[13, 14, 15, 16],
       [17, 18, 19, 20]]), array([[21, 22, 23, 24]])]
"""
```

#### - 多维数组

+ 多维数组的存取、切片、修改和一维数组类似，因为多维数组有多个轴，所以它的下标需要用多个值来表示，``NumPy``采用元组作为数组的下标，元组中的每个元素和数组的每个轴对应；多维数组的切片同样采用下标法，与一位数组方法相同，如果下标元组中只包含整数和切片，那么得到的数组和原始数组共享数据，它是原数组的视图
  
  ```python
  a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
  print(a)
  print(a[1:3, 2:5])
  print(a[::2, 2:3])
  """
  [[ 0  1  2  3  4  5]
   [10 11 12 13 14 15]
   [20 21 22 23 24 25]
   [30 31 32 33 34 35]
   [40 41 42 43 44 45]
   [50 51 52 53 54 55]]
  [[12 13 14]
   [22 23 24]]
  [[ 2]
   [22]
   [42]]
  """
  ```

+ 因为数组的下标是一个元组，所以我们可以将下标元组保存起来，用同一个元组存取多个数组，创建这样的数组时使用切片(slice)对象或``s_``对象。``slice()``有三个参数，分别为开始值、结束值和间隔步长，当这些值需要省略时可以使用``None``；``s_``是``NumPy``提供的一个对象，用来创建数组下标
  
  ```python
  a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
  idx_1 = slice(None, None, 2), slice(2, None)
  print(a[idx_1])
  idx_2=np.s_[::2, 2:]
  print(a[idx_2])
  
  """
  [[ 2  3  4  5]
   [22 23 24 25]
   [42 43 44 45]]
  [[ 2  3  4  5]
   [22 23 24 25]
   [42 43 44 45]]
  """
  ```
  
  <img title="" src="file:///C:/Users/jiani/Pictures/Typedown/ab88be08-40d7-4e58-a5d8-a5bdc6609594.png" alt="ab88be08-40d7-4e58-a5d8-a5bdc6609594" data-align="center" style="zoom:100%;">

+ 在多维数组的下标元组中，也可以使用整数元组或列表、整数数组和布尔数组，当下标中使用这些对象时，所获得的数椐是原始数据的副木，因此修改结果数组不会改变原始数组。
  
  ```python
  a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
  print(a[[0, 1, 2, 3], (2, 3, 4, 5)])
  print(a[3:, (0, 2, 5)])
  mask1 = np.array([1, 0, 1, 0, 0, 1], dtype=np.bool_)
  print(a[mask1, 2])
  mask2 = np.array([1, 0, 1, 0, 0, 1])
  print(a[mask2, 2])
  x = np.array([[0, 1], [-1, 3]])
  y = np.array([[1, 2], [2, -3]])
  print(a[x, y])
  print(a[x])
  """
  [ 2 13 24 35]
  [[30 32 35]
   [40 42 45]
   [50 52 55]]
  [ 2 22 52]
  [12  2 12  2  2 12]
  [[ 1 12]
   [52 33]]
  [[[ 0  1  2  3  4  5]
    [10 11 12 13 14 15]]
  
   [[50 51 52 53 54 55]
    [30 31 32 33 34 35]]]
  """
  ```
  
  

#### - 结构数组

类似于C语言可以通过``struct``关键字定义结构类型，``NumPy``也提供了结构数组，其定义方式如下
首先创建一个``dtype``对象``persontype``，它的参数是一个描述结构类型的各个字段的字典。字典有两个键：``names``和``formats``， 每个键对应的值都是一个列表。``name``定义结构中每个字段的名称，``formats``定义每个字段的类型：

+ ``'S30'``:长度为30个字节的字符串类型，由于结构中的每个元素的大小必须同定，因此需要指定字符串的长度。

+ ``'i'``:32位的整数类型，相当于``np.int32``

+ ``'f'``:32位的单精度浮点数类型，相当于``np.float32``

```python
persontype = np.dtype(
{'names': ['name', 'age', 'weight'], 'formats': ['S30', 'i', 'f']}, align=True
)

a = np.array([('Petre', 32, 60), ("Lee", 20, 61)], dtype=persontype)

print(a.dtype)
print(a)

"""
{'names': ['name', 'age', 'weight'], 'formats': ['S30', '<i4', '<f4'], 'offsets': [0, 32, 36], 'itemsize': 40, 'aligned': True}
[(b'Petre', 32, 60.) (b'Lee', 20, 61.)]
"""
```

同时还可以用包含多个元组的列表来描述结构的类型，类型字符串前面的``'|'``(忽略字节顺序)、``'<’``(小端模式)、``'>'``(大端模式) 等字符表示字段值的字节顺序

```python
persontype = np.dtype([('name', '|S30'), ('age', '<i4'), ('weight', '<f4')])

a = np.array([('Petre', 32, 60), ("Lee", 20, 61)], dtype=persontype)

print(a.dtype)
print(a)

"""
{'names': ['name', 'age', 'weight'], 'formats': ['S30', '<i4', '<f4'], 'offsets': [0, 32, 36], 'itemsize': 40, 'aligned': True}
[(b'Petre', 32, 60.) (b'Lee', 20, 61.)]
"""
```

结构数组的存取方式和一般数组相同，通过下标能够取得其中的元素，同时可以使用字段名作为下标获取对应的字段值；通过下标法返回的内容是原结构体的视图，与其共用内存

```python
--snip--
print(a[0])
print(a[0]['name'])

"""
(b'Petre', 32, 60.)
b'Petre'
"""
```

当结构体数组作为实参传入函数时对其处理时会对数组进行改变

```python
#test1.py
def func(data):
    a = data['age']
    a[0] = 22

--snip--
print(a['age'])

test1.func(a)

print(a['age'])


"""
[32 20]
[22 20]
"""
```

通过``tostring()``或``tofile()``方法可以将数组以二进制的方式转换成字符串或写入文件，如``a.tofile(test.bin)``

```python
print(a.tostring())

"""
b'Petre\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00\x00\x00pBLee\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00\x00\x00tB'
"""
```

### 1.1.2 ``ufunc``函数

+ ``ufunc``是``universal function``的缩写，它是一种能对数组的每个元素进行运算的函数。``NumPy``内置的许多``ufunc``函数都是用C语言实现的，因此它们的计算速度要快于内置的``math``库函数

+ ``ufunc``函数对数组计算得到的结果会新创建一个数组来保存；也可以通过``out``参数指定保存计算结果的数组。因此如果希望直接在原数组中保存结果，可以将它传递给out参数

#### - 四则运算

数组运算支持常规运算符(``+``、``-``、``*``、``/``等)，但计算速度较慢，``NumPy``为数组定义了各种数学运算操作符，如下表:

![31a01c93-d6dd-45fc-a828-26587a3fbd9d](file:///C:/Users/jiani/Pictures/Typedown/31a01c93-d6dd-45fc-a828-26587a3fbd9d.png)

```python
a = np.arange(1, 18, 4)
b = np.arange(1, 10, 2)
print(a, b)
print(np.add(a, b))
print(a, b)
np.add(a, b, out=a)
print(a, b)

"""
[ 1  5  9 13 17] [1 3 5 7 9]
[ 2  8 14 20 26]
[ 1  5  9 13 17] [1 3 5 7 9]
[ 2  8 14 20 26] [1 3 5 7 9]
"""
```

#### - 比较运算和布尔运算

使用``=``、``>``等比较运算符对两个数组进行比较，将返回一个布尔数组，它的每个元素值都是两个数组对应元素的比较结果，同样的，每个运算符也对应一个函数

![6eb37a6c-b6b2-46f8-886a-26122645fa2c](file:///C:/Users/jiani/Pictures/Typedown/6eb37a6c-b6b2-46f8-886a-26122645fa2c.png)

```python
a = np.arange(1, 18, 4)
b = np.arange(1, 10, 2)
print(a == b)

"""
[ True False False False False]
[ True False False False False]
"""
```

由于Python中的布尔运算使用的``and``、``or``和``not``等关键字无法被重载，因此数组的布尔运算只能通过相应的``ufunc``函数进行，这些函数名都以``logical_``幵头，对两个布尔数组使用``and``、``or``和``not``等进行布尔运算，将抛出``ValueError``异常

```python
a = np.arange(1, 18, 4)
b = np.arange(1, 10, 2)
print(a == b)
print(a > b)
print(np.logical_and(a == b, a > b))


"""
[ True False False False False]
[False  True  True  True  True]
[False False False False False]
"""
```

在``NumPy``中定义了``any()``和``all()``函数，只要数组中有一个元素值为``True``,`` any()``就返回``True``; 而只有当数组的全部元素都为``True``时，``all()``才返回``True``

```python
a = np.arange(1, 18, 4)
b = np.arange(1, 10, 2)
print(np.any(a == b))
print(np.any(a > b))
print(np.any(a == b) and np.any(a > b))

"""
True
True
True
"""
```

以``bitwise_``开头的函数是[位运算](https://www.runoob.com/w3cnote/bit-operation.html)函数，包括``bitwise_and``、``bitwise_not``、``bitwise_or`` 和``bitwise_xor``等。也可以使用``&`` 、``~``、``丨``和``^``等操作符进行计算

```python
a = np.arange(1, 18, 4)
b = np.arange(1, 10, 2)
print(~a)
print((a == b) | (a > b))

"""
[ -2  -6 -10 -14 -18]
[ True  True  True  True  True]
"""
```

#### - 自定义``ufunc``函数

通过``NumPy``提供的标准``ufunc``函数可以组合出复杂的表达式，在C语言级別对数组的每个元素进行计算。但有时这种表达式不易编写，而对每个元素进行计算的程序却很容易用Python实现，这时可以用``frompyfunc()``将计算单个元素的函数转换成``ufunc``函数，这样就可以方便地将所产生的``ufunc``函数对数组进行计算。``frompyfunc()``的调用格式为``frompyfunc(func, nin, nout)``，其中``func``是计算单个元素的函数，``nin``是``func``的输入参数的个数，``nout``是``func``的返回值的个数。这种函数所返回的数组的元素类型是``object``, 因此还需要调用数组的``astype()``方法将其转换为所需要的类型。同时，使用``vectorize()``也可以实现和``frompyfunc()``类似的功能，但它可以通过``otypes``参数指定返回的数组的元素类型。``otypes``参数可以是一个表示元素类型的字符串，也可以是一个类型列表，
使用列表可以描述多个返回数组的元素类型

#### - 广播

当使用``ufunc``函数对两个数组进行计算时,``ufunc``函数会对这两个数组的对应元素进行计算,因此它要求这两个数组的形状相同。如果形状不同，会进行如下广播(``broadcasting``)处理

+ 让所有输入数组都向其中维数最多的数组看齐，``shape``属性中不足的部分都通过在前面加1补齐

+ 输出数组的``shape``属性是输入数组的``shape``属性的各个轴上的最大值

+ 如果输入数组的某个轴的长度为1或与输出数组的对应轴的长度相同，这个数组能够用来计算，否则出错

+ 当输入数组的某个轴的长度为1吋，沿着此轴运算时都用此轴上的第一组值

以创建一个形状为(6,1)的二维数组和一个形状为(5,)的一维数组并计算二者之和为例：

+ 计算a与b的和会得到一个加法表，它相当于计算两个数组中所有元素对的和，得到一个形状为(6,5)的数组
  
  ```python
  a = np.arange(0, 60, 10).reshape(6, 1)
  b = np.arange(0, 5)
  c = a + b
  print(c)
  
  """
  [[ 0  1  2  3  4]
   [10 11 12 13 14]
   [20 21 22 23 24]
   [30 31 32 33 34]
   [40 41 42 43 44]
   [50 51 52 53 54]]
  """
  ```

+ 由于a和b的维数不同，根据规则(1)，需要让b的``shape``属性向a对齐，于是在b的``shape``属性前加1 , 补齐为(1,5)，即相当于``b.shape=1,5``

+ 根据规则(2),输出数组的各个轴的长度为输入数组各个轴的长度的最大值，可知输出数组的``shape``属性为(6,5)

+ 由于b的第0轴的长度为1，而a的第0轴的长度为6，为了让它们在第0轴上能够相加，需要将b的第0轴的长度扩展为6，即相当于
  
  ```python
  b.shape = 1, 5
  b = b.repeat(6, axis=0)
  print(b)
  
  """
  [[0 1 2 3 4]
   [0 1 2 3 4]
   [0 1 2 3 4]
   [0 1 2 3 4]
   [0 1 2 3 4]
   [0 1 2 3 4]]
  """
  ```

+ 同样的，由于a的第1轴的长度为1，而b的第1轴的长度为5，为了让它们在第1轴上能够相加，需要将a的第1轴的长度扩展为5，即相当于``a=a.repeat(5, axis=1)``

由于这种广播计算很常用，因此``NumPy``提供了``ogrid``对象，用于创建广播运算用的数组``x,y = np.ogrid[:5, :5]``，还提供了``mgrid``对象，返回进行广播之后的数组``x,y = np.mgrid[:5, :5]``，``ogrid``它像多维数组一样，用切片元组作为下标

```python
a = np.arange(4)
print(a)
b = a[None, :]
c = a[:, None]
print(b)
print(c)

"""
[0 1 2 3]
[[0 1 2 3]]
[[0]
 [1]
 [2]
 [3]]
"""
```

#### - ``ufunc``方法

``ufunc``函数对象本身还有一些方法函数，这些方法只对两个输入、一个输出的``ufunc``函数有效，其他的``ufunc``对象调用这些方法时会抛出``ValueError``异常

+ ``reduce()``方法沿着``axis``参数指定的轴对数组进行操作，相当于将``<op>``运算符插入到沿``axis``轴的所有元素之间:``<op>.reduce(array,axis=0,dtype=None)``
  
  ```python
  a = [[0, 1, 2, 3], [1, 2, 3, 4]]
  r1 = np.add.reduce(a[0])
  r2 = np.add.reduce(a, axis=1)
  print(r1, r2)
  
  """
  6 [ 6 10]
  """
  ```

+ ``accumulate()``方法和``reduce()``类似，它返回的数组和输入数组的形状相同，保存所有的中间计算结果
  
  ```python
  a = [[0, 1, 2, 3], [1, 2, 3, 4]]
  r1 = np.add.accumulate(a[0])
  r2 = np.add.accumulate(a, axis=1)
  print(r1)
  print(r2)
  
  """
  [0 1 3 6]
  [[ 0  1  3  6] 
   [ 1  3  6 10]]
  """
  ```

+ ``reduceat()``方法计算多组``reduce()``的结果，通过``indices``参数指定一系列的起始和终止位置，以下为例
  
  ```python
  a = [[0, 1, 2, 3], [1, 2, 3, 4]]
  r1 = np.add.reduceat(a[0], indices=[0, 1, 0, 2, 0, 3, 0])
  print(r1)
  
  """ 
  [0 1 1 2 3 3 6] 
  """
  ```
  
  其结果中除最后一个元素之外，都按照如下计算得出

```python
if indices[i] < indices[i+1]:
    result[i] = <op>.reduce(a[indices[i]:indices[i+1]])
else:
    result[i] = a[indices[i]]
```

最后一个元素计算方法为

```python
if indices[-1] == 0:
    result[i] = <op>.reduce(a[indices[-1]:])
else:
    result[i] = a[indices[i]]
```
