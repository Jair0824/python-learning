# Python科学计算基础

## 1.1 numpy库入门

``NumPy`` 扩展库  定义了由同类型的元素组成的多维数组``ndarray``及其常用运算，``ndarray`` 是科学计算中最常用的数据类型。NumPy 数组相对列表的优势是运算速度更快和占用内存更少。``ndarray``是一个类，它的别名是``array``。它的主要属性包括``ndim``(维数)、``shape``(形状，即由每个维度的长度构成的元组) 、``size``(元素数量) 和``dtype``(元素类型：可以是 Python 的内置类型，也可以是 NumPy
定义的类型，例如 numpy.int32、numpy.int16 和 numpy.float64 等)。

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
  c.shape=6,-1
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
+ ``logspace()``可以通过指定初始值、终值创建等比数列，参数``base``(默认为10)和``endpoint``分别指定底数和是否包含终值 
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
