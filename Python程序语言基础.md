# Python程序语言基础

## 1.1 基本命令

### 1.1.1 基本运算

+ \+ - \* / //(整数商) %(mod) **(幂)

### 1.1.2 赋值语句

+ 同步赋值语句``` <a>,<b>,<c> = <A>,<B>,<C>``` 赋值时首先运算右侧N个表达式，将结果赋给左侧N个变量，如变量的互换
  
  ```python
  ##Ex1
  
  t=x x=y y=t
  
  ##Ex2
  
  x,y = y,x
  ```

### 1.1.3 条件与循环

+ 条件语句
  
  ```python
  #Python使用对齐缩进作为语句的层次标记，同一层次的缩进量要一一对应
  if T1 :
     A1
  elif T2 :
     A2
  else T3 :
     A3
  ```

+ 循环语句
  
  ```python
  #while循环
  s,k = 0
  while k < 101:
    k = k + 1
    s = s + k
  print s
  ```
  
  ```python
  #for循环
  s = 0
  for k in range(101):
    s = s + k
  print s
  ```
  
  + Python提供```break```和```continue```循环保留字来控制循环的进行
    ```break```用来跳出最内层```for```及```while```循环，每个     ```break```语句只有跳出当前循环的能力，脱离循环从循环代码后继续执行；```continue```语句用来结束当次循环，但不跳出当前循环
    
    ```python
    #break语句
    for s in "PYTHON":
         if s=="T":
           break
         print(s,end="")
    
    #continue语句
    for s in "PYTHON":
     if s=="T":
           continue
         print(s,end="")          
    ```

### 1.1.4 推导式与穷举

+ 推导式：``list``、``dice``和``set``等容器提供推导式型紧凑语法，可以通过迭代从已有容器创建新的容器

+ 穷举：穷举 (Exhaustive Search) 一种解决问题的基本方法。穷举法的基本思想是：当问题的解属于一个规模较小的有限集合时，可以通过逐一列举和检查集合中的所有元素找到解
  
  ```python
  nums={1,2,3,5,33,0}
  multiplier_of_3=[n for n in nums if n % 3==0] #创建元素为集合num中3的倍数的值的列表
  square_of_odds={n*n for n in nums if n % 2 ==1} #创建元素为集合num中奇数的平方的集合
  
  s=[25,30,66,34,23]
  sr={n:n%2 for n in set(s)} #创建元素为列表s转化的集合中的元素与其与2的余数组成的键值对的字典
  tr={n:r for (n,r) in sr.items() if r==0} #创建元素为sr中值为0的键值对组成的字典
  
  ```

## 1.2 数据结构

### 1.2.1 基本数字类型

+ Python数字有整形、浮点型和复数型等

+ Python数字类型之间可以相互转换
  
  | 函数               | 描述                                        |
  |:----------------:|:-----------------------------------------:|
  | int(x)           | 将x转换为整数，x可以是浮点数或字符串                       |
  | float(x)         | 将x转换为浮点数，x可以是浮点数或字符串                      |
  | complex(re[,im]) | 生成一个复数，实部为re(可以是整数、浮点数、字符串)，虚部为im(不能是字符串) |

+ Python有内置的数字处理函数
  
  | 函数                 | 描述                   |
  |:------------------:|:--------------------:|
  | abs(x)             | x的绝对值                |
  | divmod(x,y)        | （x//y,x%y)，输出为元组类型   |
  | pow(x,y[,z])       | （x**y）%z，[..]表示参数可省略 |
  | round(x[,ndigits]) | 对x四舍五入，保留ndigits位小数  |
  | max($x_1,x_2,...$) | 求最大值                 |
  | min($x_1,x_2,...$) | 求最小值                 |

### 1.2.2 字符串

- 字符串在Python中用""或''隔开，若字符串内包含换行符```\n```,则使用```""""""```

- 字符串包括两种序列

- Python字符串可区间访问，采用[N:M]格式，表示从N到M(不包括)的字符串

- 使用双引号(单引号)时单引号(双引号)可以作为字符串的一部分，三引号中可以使用单引号、双引号、换行

- ```input()```函数将用户输入内容当作字符串类型，```print()```函数直接打印字符串 

- Python提供了5个字符串操作符
  
  | 操作符      | 描述                       |
  |:--------:|:------------------------:|
  | x+y      | 连接字符串x和y                 |
  | x*n或n\*x | 复制n次x                    |
  | x in s   | 如果x是s的字串返回True，否则返回False |
  | str[i]   | 索引，返回第i个字符               |
  | str[N:M] | 返回第N到M字串(不包括M)           |

- Python提供了内置的字符串处理函数
  
  | 函数     | 描述                   |
  |:------:|:--------------------:|
  | len(x) | 返回x的长度               |
  | str(x) | 返回任意类型x对应的字符串形式      |
  | chr(x) | 返回Unicode编码x对应的单字符   |
  | ord(x) | 返回单字符表示的Unicode编码    |
  | hex(x) | 返回整数x对应十六进制数的小写形式字符串 |
  | oct(x) | 返回整数x对应八进制数的小写形式字符串  |

- Python提供了内置的字符串处理方法
  
  | 方法            | 描述                       |
  |:-------------:|:------------------------:|
  | str.lower()   | 返回字符串str的全部字母小写的副本       |
  | str.upper()   | 返回字符串str的全部字母大写的副本       |
  | str.islower() | 当所有字母为小写时返回TRUE，否则为FALSE |
  
  其余非常用方法见下图
  ![输入图片说明](https://img2.imgtp.com/2024/03/12/4D6WvGYk.png)

- 字符串的format()方法格式化处理
  
  - 基本格式:```<模板字符串>.format(<逗号分隔的参数>)```
  
  - 模板字符串由一系列槽组成，用来控制修改字符串中嵌入值出现的位置，其基本思想是将format()方法中逗号分隔的参数按照序号关系替换到模板字符串的槽中。槽用大括号表示，如果大括号中没有序号，则按照出现顺序替换;如果大括号中指定了使用参数的序号，按照序号对应参数替换。
  
  ![输入图片说明](https://img2.imgtp.com/2024/03/12/QVXuURVD.png)
  
  - format()的槽还可以包含格式控制信息，即```{<参数序号>:<格式控制标记>}```
    ![输入图片说明](https://img2.imgtp.com/2024/03/12/XdbGlblV.png)
    ```<宽度>```指当前槽的设定输出字符宽度，如果该槽对应的 format()参数长度比<宽度>设定值大，则使用参数实际长度;如果该值的实际位数小于指定宽度，则位数将被默认以空格字符补充；
    ```<对齐>```分别用```<^>```表示左对齐、居中对齐、右对齐；
    ```<填充>```指宽度内除了字符其他的填充内容；
    ```<.精度>```对于浮点数表示小数的输出位数，对于字符串表示输出的最大长度；
    ```<类型>```表示输出浮点数和整数的规则

```python
a = 'abcde'
b = '12345'

print(a[1]) #输出b
c = a*3 
print(c[6:8],len(c)) #输出bc,15

"{}{}{}".format("圆周率","是",3.1415926) #格式化输出

#参数控制信息
s="PYTHON"
"{0:10}".format(s) #默认左对齐，宽度为30，输出'PYTHON    '
"{0:>10}".format(s) #右对齐，宽度为10，输出'    PYTHON'
"{0:*^10}".format(s) #居中对齐，宽度为10，用*填充，输出'**PYTHON**'
"{0:-^10,}".format(1234.56) #居中对齐，宽度为20，用-填充，带分隔符，输出'-1,234.56-'
"{0:.2f}".format(123.4567) #输出'123.46'，注意四舍五入
"{0:b},{0:c},{0:d},{0:o},{0:x},{0:X}".format(125) #输出'1111101,},125,175,7d,7D'
"{0:e},{0:E},{0:f},{0:%},{0:.2e},{0:.2E},{0:.2f},{0:.2%}".format(3.14) #输出'3.140000e+00,3.140000E+00,3.140000,314.000000%,3.14e+00,3.14E+00,3.14,314.00%'
```

### 1.2.3 列表及元组

+ 列表用```[]```标记，元组用```()```标记，二者的访问方式相同，内部数据类型不限

+ 列表可以被修改，元组则不可修改

+ 创建列表/元组时需要注意加```[]/()```

+ 复制列表的方式为```b=a[:]```,而不是```b=a```

+ ```list()```函数将对象转换为列表，```tuple()```函数将对象转换为元组

+ 列表解析可以简化对列表内元素逐一进行操作的代码

+ 常见与列表/元组相关的函数
  
  | 函数       | 功能          | 函数        | 功能    |
  |:--------:|:-----------:|:---------:|:-----:|
  | cmp(a,b) | 比较列表/元组内的元素 | min(a)    | 返回最小值 |
  | len(a)   | 元素个数        | sum(a)    | 求和    |
  | max(a)   | 返回最大值       | sorted(a) | 升序排序  |

+ 常见对列表的处理方法
  
  | 函数              | 功能                   |
  |:---------------:|:--------------------:|
  | a.append(1)     | 将1添加到列表a末尾           |
  | a.account(1)    | 统计列表a中元素1出现的次数       |
  | a.extend([1,2]) | 将列表[1,2]的内容追加到列表a的末尾 |
  | a.index(1)      | 从列表a中找出第一个1的索引位置     |
  | a.insert(2,1)   | 将1插入列表中索引为2的位置       |
  | a.pop(1)        | 移除列表a中索引为1的元素        |

```python
a = [1,2,3]
b = (4,5,6)
c = [1,'abc',[1,2]]

print(a[1],b[2],c[2][1]) #输出结果为2，6，2

a[1] = 3

print(a[1],b[2],c[2][1]) #输出结果为3，6，2
```

### 1.2.4 集合

+ 集合(set)是包含0个或多个数据的无序组合，集合中的元素类型只能是固定数据类型(非列表、字典、集合)且不可重复

+ 集合用``{}``表示，是无序组合，没有索引和位置的概念，不能分片，集合中的元素可以动态增加或删除

+ ```set(x)```也可用来生成集合，输入的参数可以是任何组合的数据类型，返回一个无重复且任意排序的集合

+ 集合类型常见的操作符如下图
  ![输入图片说明](https://img2.imgtp.com/2024/03/12/fjkYUU9I.png)

+ 集合类型常见的处理函数如下图
  ![输入图片说明](https://img2.imgtp.com/2024/03/12/KCrRY0uB.png)

+ 集合的应用主要在成员关系测试、元素去重和删除数据项，由于其具有元素不重复特性，故需要对一维数据去重或进行数据重复处理时一般使用集合

```python
#集合的生成与动态变化
s={1,2,(1,2,3),'abc'}
print(s，type(s)) #输出{1, 2, (1, 2, 3), 'abc'} <class 'set'>
#由于集合是无序的，故其打印效果可能与定义有不同

set("12345") #输出{'3', '1', '5', '2', '4'}

#集合的操作与函数处理
s={1,2,3,4,"abcd",'a'}
t={1,3,5,7,"bd",'c'}
s-t #输出{'abcd', 'a', 2, 4}
t-s #输出{'bd', 'c', 5, 7}
s&t #输出{1, 3}
s^t #输出{2, 'c', 4, 5, 7, 'bd', 'a', 'abcd'}
s|t #输出{1, 2, 3, 4, 'c', 5, 7, 'bd', 'a', 'abcd'}
s<=t #输出False
t<=s #输出False
s.add(5)
s #输出{'abcd', 1, 2, 3, 4, 5, 'a'}
s.copy() #输出{'abcd', 1, 2, 3, 4, 5, 'a'}
s.pop() #输出'abcd'(结果不唯一)
s.discard(5)
s #输出{1, 2, 3, 4, 'a'}
len(s) #输出5
5 in s #输出False

#集合的应用
tup=("python","123",123,"abc",123)
set(tup) #去重，输出{'abc', 123, '123', 'python'}
newtup=tuple(set(tup)-{"python"}) #去重同时删除数据项，输出(123, 'abc', '123')
```

### 1.2.5字典

+ 字典(dict)的基础是键值对关系，在程序语言中，根据一个信息查找到了另一个信息的方式构成了键值对，表示索引用的键和对应的值构成的成对的关系

+ 字典是python语言中实现映射(通过任意键信息查找任意值信息)的方法，可以通过``{}``建立(字典可视作集合的延伸，同样没有顺序也不能重复)，建立模式为```{<键1>:<值1>,<键2>:<值2>,...,<键n>:<值n>}```

+ 字典的键必须为可哈希对象(一个对象的哈希值在生命周期内不改变，就被成为可哈希,可变容器如列表或字典都不可哈希)
  
  + 字典的建立直接使用``{}``，而集合的建立需使用函数``set()``
  
  + 字典中键值对的访问模式采用``[]``模式：```<值>=<字典变量>[<键>]```
  
  + 字典中键值的修改采用``[]``访问和赋值的方式实现
  
  + 字典在python中同样可以进行面向对象处理，对应方法采用```<a>.<b>()```格式
    ![输入图片说明](https://img2.imgtp.com/2024/03/12/D22UUfxI.png)
  
  + 字典可以通过for-in语句对其元素进行遍历
    
    ```python
    for <变量名> in <字典名>:
    <语句块>
    ```

```python
#建立字典
DC={"a1":1,"b1":2,"c1":"3"}
print(DC) #输出{'a1': 1, 'b1': 2, 'c1': '3'}，顺序可能不同

#键值对访问
DC['a1'] #输出1

#字典的修改
DC['a1']='a'
DC #输出{'a1': 'a', 'b1': 2, 'c1': '3'}

#字典函数处理
DC.keys() #输出dict_keys(['a1', 'b1', 'c1'])
DC.values() #输出dict_values(['a', 2, '3'])
DC.items() #输出dict_items([('a1', 'a'), ('b1', 2), ('c1', '3')])
DC.get('a1') #输出'a'
DC.popitem() #输出('c1', '3')

#字典遍历处理
for key in DC:
   print(key)
#输出a1 \n b1 \n c1
```

## 1.3函数式编程

### 1.3.1 函数简介

+ 函数是一段具有特定功能的、可重用的语句组，用函数名表示及调用，包括自定义函数和库函数

+ Python使用``def``保留字自定义函数，
  
  ```python
  def <函数名>(<参数列表>):
    <函数体>
    return <返回值列表>
  ```

+ 函数名可以是任何有效的Python标识符；参数列表是调用该函数时传递的值，可以有一个或多个，括号内的参数为“形参”(形式参数，可以不和调用时的参数名相等)；需要返回值时使用``return``和返回值列表

+ Python支持用``lambda``对简单的功能定义“行内函数”
  
  ```python
  f = lambda x : x + 2 #f(x)=x+2
  g = lambda x,y: x + y #g(x,y)=x+y
  ```

### 1.3.2 函数调用与参数传递

+ 函数的调用过程可分为如下步骤
  
  + 程序在调用处暂停执行
  
  + 在调用时将实参复制到函数的形参
  
  + 执行函数语句
  
  + 调用结束给出返回值，程序继续执行
    
    ```python
    #函数调用示例
    def put(a):
    
        b = "{} is a wonderful subiect!".format(a)
    
        return b
    name = ["A", "B", "C"]
    
    for i in name: 
        print(put(i))
    ```
  
  + 可选参数：在定义函数时如果有些参数存在默认值，则可在定义时直接将这些参数指定默认值，当函数被调用时如果没有传入对应的参数则直接使用默认值代替；由于函数调用时需按顺序输入参数，可选参数必须定义在非可选参数后面
  
  + 可变数量参数：在函数定义时可通过在变量名前加``*``定义可变数量参数，调用时这些变量被当作元组传递，可变数量参数必须定义在参数列表最后
  
  + 参数的位置传递与名称传递：函数调用时实参默认按照位置顺序传入函数，为位置传递；同时在调用函数时可以指定参数名称进行传递，为名称传递
  
  + Python的函数返回值可以是各种形式，如列表、多个值等，同时函数也可以没有返回值
  
  ```python
  #可选参数
  def dup(str,times=2):
      print(str*times)
  dup("a") #输出aa
  dup("a",4) #输出aaaa
  
  #可变数量参数
  def fun(a,*b):
      for i in b:
          a += b
      return a
  fun(1,2,3,4) #输出15
  
  #参数的位置传递
  a=fun(1,2,3,4)
  
  #参数名称传递
  a=fun(b=2,a=3)
  
  #多值返回
  def add2(x=0,y=0):
    return [x+2,y+2] #返回列表
  
  def add3(x,y):
    return x+3,y+3 #双重返回
  ```

### 1.3.3 函数与变量

+ 一般程序中包含两种变量：全局变量与局部变量，全局变量指在函数外定义的变量，一般没有缩进，在程序执行时全程有效；局部变量指在函数内部使用的变量，仅在函数内部有效

+ 在函数内部使用变量有如下情形
  
  + 在函数内部处理局部变量，函数执行结束后变量将被释放；
  
  + 在函数内部处理与全局变量同名的整型、浮点型等简单数据类型的“全局变量”并改变其值，全局变量值并不发生变化，是因为函数在自己的内存空间中会新建一个变量，在函数中处理的该变量仍为局部变量，故不改变对应的值。如果想处理对应的局部变量，需在函数中使用``global xxx``显式声明；
  
  + 在函数内部处理列表等组合数据类型全局变量并改变其值，调用后全局变量会发生变化，因为列表类型变量的创建和调用语言不一样，对此类变量进行改变时只能调用已存在的变量，而不会创建一个新变量。如果函数内部也创建了该类型的同名的变量，则函数仍处理局部变量
    
    ```python
    n=1
    s=[]
    def func_1(a,b): 
        n=a+b 
        return nfunc_1(1,2)
    print(n) #输出1 函数仅处理局部变量
    
    def func_2(a,b): 
        global n #声明为全局变量 
        n=a+b 
        return nfunc_2(1,2)
    print(n) #输出3
    
    def func_3(a) 
        a="str"
        s.append(a)
    print(s) #输出str
    ```

### 1.3.4  函数作为实参及返回值

#### -函数作为实参

+ 在程序语言中函数也可以作为实参进行调用，用法与普通实参类似，以排序函数``sorted()``进行演示

+ 内置函数 ``sorted()`` 可以对一个可遍历对象 (iterable) 中的元素进行排序，排序的结果存储在一个新创建的列表中。关键字实参 ``key`` 指定一个函数，它从每个元素生成用于排序的比较值。关键字实参 ``reverse`` 的默认值为 False，若设为 True 则表示从大到小的次序排序。
  
  ```python
  n[1]: animals = ["elephant", "tiger", "rabbit", "goat", "dog","penguin"]
  In[2]: sorted(animals)
  Out[2]: ['dog', 'elephant', 'goat', 'penguin', 'rabbit', 'tiger']
  #len()函数作为实参做关键字
  In[3]: sorted(animals, key=len) 
  Out[3]: ['dog', 'goat', 'tiger', 'rabbit', 'penguin', 'elephant']
  In[4]: sorted(animals, key=len, reverse=True) 
  Out[4]: ['elephant', 'penguin', 'rabbit', 'tiger', 'goat', 'dog']
  In[5]: def m1(s): return ord(min(s)) 
  #自定义函数m1作为实参做关键字,按字符串中最小字符Unicode排序
  In[6]: sorted(animals, key=m1) 
  Out[6]: ['elephant', 'rabbit', 'goat', 'dog', 'tiger', 'penguin']
  In[7]: def m2(s): return ord(min(s)), len(s)
  #自定义函数作为实参做关键字，先按照最小编码值从小到大排序，若两个字符串具有相同的最小编码值，则按照长度从小到大排序
  In[8]: sorted(animals, key=m2) 
  Out[8]: ['goat', 'rabbit', 'elephant', 'dog', 'tiger', 'penguin']
  ```

#### -函数作为返回值

+ 在自定义函数及库函数中均可使用函数作为函数的返回值，用法与正常返回值相同，以下为例

+ 定义一个函数``key_fun``，其中定义了两个用于字符串排序的函数``m1``和``m2``。列表``ms``存储了``None``、``len`` 和这些函数。``key_fun``以实参为索引值返回``ms``中的对应函数，即该函数的返回值是一个函数。第 9 行至第 10 行的循环依次使用这些函数对字符串进行排序，其中 None 表示默认的排序方式。
  
  ```python
  def key_fun(n):
      def m1(s): return ord(min(s))
      def m2(s): return ord(min(s)), len(s)
  
      ms = [None, len, m1, m2]
      return ms[n]
  animals = ["elephant", "tiger", "rabbit", "goat", "dog", "penguin"] 
  for i in range(4): 
      print(sorted(animals, key=key_fun(i))) 
  """ 
  输出： 
  ['dog', 'elephant', 'goat', 'penguin', 'rabbit', 'tiger'] 
  ['dog', 'goat', 'tiger', 'rabbit', 'penguin', 'elephant'] 
  ['elephant', 'rabbit', 'goat', 'dog', 'tiger', 'penguin'] 
  ['goat', 'rabbit', 'elephant', 'dog', 'tiger', 'penguin'] 
  """
  ```

### 1.3.5 代码复用与模块化设计

#### -面向过程与面向对象

+ 现阶段的编程语言从代码层面采用函数和对象两种抽象方式，分别对应面向过程和面向对象两种编程思想
+ 面向过程是一种以过程描述为主的编程思想，要求列出所有解决问题所需的步骤并通过函数一一实现步骤，使用时依次建立并调用函数，函数通过将步骤或子功能封装实现代码复用并降低编程难度
+ 对象将程序代码组织为更高级别的类，对象包括表示对象属性的类和代表对象操作的方法，在程序设计中，对于对象``<a>``，获取其属性``<b>``采用``<a>.<b>``，调用其方法``<c>``采用``<a>.<c>()``，如对一个列表``ls``，使用``append()``方法``ls.append(1)``在列表末尾添加元素``1``；对象是一种高级别抽象，它包括一组静态值(属性)和一组函数(方法)，对象和函数都使用了一个抽象逻辑，但对象可以凝聚更多的代码，更适合代码量大，交互逻辑复杂的程序

#### -代码复用及递归

+ 函数可以被内部代码调用，这种方法称为递归，递归需要满足的条件有：
+ 原问题可以分解为一个或多个结构类似但规模更小的子问题
+ 当子问题规模足够小时可以直接求解，称为递归终止条件
+ 原问题的解可以由子问题的解合并而成
+ 递归解决问题的重要基础是基于问题提出递归公式，以阶乘为例，阶乘公式为

$$
\left. n!=\left\{\begin{array}{ll}1&\text{if} \quad n=1\\n*(n-1)!&\text{if} \quad n>1\end{array}\right.\right.
$$

```python
def func(n):
    if n==1:
        return 1
    else:
        return n * func(n-1)
```

#### -模块化设计

+ 将一个程序分成多段，每段实现一个功能，对程序合理划分功能模块并基于模块设计程序的方法称为模块化设计；通过将函数存储在独立的文件中，可隐藏程序代码的细节，将重点放在程序的高层逻辑上；将函数存储在独立文件中后，可与其他程序员共享这些文件而不是整个程序

+ 创建模块：模块是扩展名为.py的文件，包含要导入到程序中的代码，每个模块都有一个全局变量``__name__``，模块中应包含模块的文档，解释模块的用法

+ 模块的使用方式有两种
  
  + 模块作为一个独立的程序运行，此时变量``__name__``的值为``'__main__'``
  + 被其他程序导入以后调用其中的函数，此时变量``__name__``的值为函数的名称

+ 导入模块：导入模块的方法有多种，可根据实际需要选择
  
  + 导入整个模块：导入整个模块使用``import module_name``，Python读取这个文件时，代码行``import module_name``打开文件``module_name.py``，并将其中的所有函数都复制到这个程序中；要调用被导入的模块中的函数，可指定导入的模块的名称``module_name``和函数名``func_name``，并用句点分隔它们，即``module_name.func_name()``
  + 导入特定函数：导入模块中特定函数的语法为``from module_name import function_name``;通过用逗号分隔函数名，可根据需要从模块中导入任意数量的函数，即``from module_name import function_0, function_1, function_2``；若使用这种语法，调用函数时就无需使用句点，只需指定其名称
  + 为函数指定别名：如果要导入的函数的名称可能与程序中现有的名称冲突，或者函数的名称太长，可在导入时指定别名，语法为``from module_name import function_name as fn``，此时需要调用``function_name``时只需调用``fn``即可
  + 为模块指定别名：在调用模块时同样可以用``as``指定别名，即``import module_name as mn``，此时调用模块中的函数时可以通过``nm.func_name()``进行简化
  + 导入模块中所有的函数：使用星号``*``运算符可让Python导入模块中的所有函数，即``from module_name import *``；import语句中的星号让Python将模块中的每个函数都复制到这个程序文件中。由于导入了每个函数，可通过名称来调用每个函数，而无需使用句点表示法；使用并非自己编写的大型模块时，最好不要采用这种导入方法，Python可能遇到多个名称相同的函数或变量，进而覆盖函数，而不是分别导入所有的函数
  + 使用模块的程序和模块文件不在同一个目录下时，使用 import 语句导入模块会报错。此时需要将模块所在目录插入到列表 sys.path 中，然后可以导入模块，即``import sys; sys.path.insert(0,’D:\Python\src’)``

+ 在软件交付使用前应通过充分测试尽可能查找和改正错误。模块中可使用以``test____``为名称前缀的测试函数使用已知正确答案的数据测试程序中的一些关键函数

+ 在操作系统的命令行运行模块时首先进入模块文件``module_name.py``所在目录，然后输入命令``python module_name.py --name_1 --name_2``，其中``name_1``、``name_2``分别为两个实参
  
  ```python
  #模块示例
    """
    Module for printing the monthly calendar for the year and
    the month specified by the user.
  
      For example, given year 2022 and month 9, the module prints
      the monthly calendar of September 2022.
  
      > > > run month_calendar.py --year 2022 --month 9
  
      2022  9
      ---------------------------
  
      Sun Mon Tue Wed Thu Fri Sat
                  1   2   3
      4   5   6   7   8   9   10
      11  12  13  14  15  16  17
      18  19  20  21  22  23  24
      25  26  27  28  29  30
      """
  
  import sys, math
  
  def is_leap(year): 
      #若year是闰年返回True，否则返回False
      return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
  
  def test____is_leap():
      d = {1900:False, 2000:True, 2020:True, 2022:False}
      #用已知正确答案的数据测试is_leap函数,若计算结果不正确则报错
      for y in d.keys():  
          if is_leap(y) != d[y]:  
              print("test failed: is_leap(%d) != %s" % (y, d[y]))
  
  def get_0101_in_week(year):
      return (year + math.floor((year - 1) / 4) -
              math.floor((year - 1) / 100) +
              math.floor((year - 1) / 400)) % 7
  
  def test____get_0101_in_week():
      d = {2008:2, 2014:3, 2021:5, 2022:6}
      for y in d.keys():
          if d[y] != get_0101_in_week(y):
              print("test failed: get_0101_in_week(%d) != %s"
                    % (y, d[y]))
  
  month_days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
  def get_num_days_in_month(year, month):
      n = month_days[month]
      if month == 2 and is_leap(year):
          return n + 1
      return n
  
  def get_num_days_from_0101_to_m01(year, month):
      n = 0
      for i in range(1, month):
          n += get_num_days_in_month(year, i)
      return n
  
  def get_m01_in_week(year, month):
      n1 = get_0101_in_week(year)
      n2 = get_num_days_from_0101_to_m01(year, month)
      n = (n1 + n2) % 7
      return n
  
  def test____get_m01_in_week():
      d = {(2022, 6):3, (2019, 10):2, (2016, 5):0, (2011, 7):5}
      for y in d.keys():
          if d[y] != get_m01_in_week(y[0], y[1]):
              print("test failed: get_m01_in_week(%s) != %s"
                    % (y, d[y]))
  
  def print_header(year, month):
      print("%d  %d " % (year, month))
      print("---------------------------")
      print("Sun Mon Tue Wed Thu Fri Sat")
  
  def print_body(year, month):
      n = get_m01_in_week(year, month)
      print(n * 4 * ' ', end='')
      for i in range(1, get_num_days_in_month(year, month) + 1):
          print('%-04d' % i, end='')
          if (i + n) % 7 == 0: print()
  
  def print_monthly_calendar(year, month):
      print_header(year, month)
      print_body(year, month)
  
  def test_all_functions():
      #调用所有以“test____”为名称前缀的测试函数。
      test____is_leap()
      test____get_0101_in_week()
      test____get_m01_in_week()
  
  if __name__ == '__main__':
      #判断模块是否作为一个独立的程序运行
      #sys.argv是一个记录了用户在命令行输入的所有参数的列表。列表的第一个元素是模块名称，列表的其余元素(若存在)是用户依次输入的所有参数  
      if len(sys.argv) == 1:
          print(__doc__)  #用户未输入参数，输出模块的文档
      elif len(sys.argv) == 2 and sys.argv[1] == '-h':
          print(__doc__)  #用户输入了’−h’，输出模块的文档
      elif len(sys.argv) == 2 and sys.argv[1] == 'test':
          test_all_functions()#用户输入了’test’，测试模块中的函数  
      else:
          import argparse #此模块获取用户在命令行输入的年份和月份
          parser = argparse.ArgumentParser()
          #在每个参数(年份或月份)的输入值前面都要用”−−参数名称”的格式指定对应的参数名称，参数的顺序无关紧要
          parser.add_argument('--year', type=int, default=2022)
          parser.add_argument('--month', type=int, default=1)
          args = parser.parse_args()
          year = args.year; month = args.month
          print_monthly_calendar(year, month) 
  
  ```

## 1.4 类与继承

### 1.4.1 类与实例

+ 面向对象编程(Object Oriented Programming)，简称OOP，是一种程序设计思想。OOP把对象作为程序的基本单元，一个对象包含了数据和操作数据的函数。面向对象的程序设计把计算机程序视为一组对象的集合，每个对象都可以接收其他对象发过来的消息，并处理这些消息，计算机程序的执行就是一系列消息在各个对象之间传递。在Python中，所有数据类型都可以视为对象，当然也可以自定义对象。自定义的对象数据类型就是面向对象中的类(Class)的概念
+ 类(Class)与实例(Instance)是面向对象编程中的重要概念，类是抽象的模板，类中的函数称为方法；实例是根据类创建出的具体的对象，每个对象拥有相同的方法，但各自的数据可能不同

#### -类的创建与使用

+ 定义类使用``class``关键字，以创建``Student``类为例
  
  ```python
  class Student():
        def __init__(self,name,age):
            self.name=name
            self.age=age
  ```

+ ``class``后为类名，通常首字母大写

+ ``__init__()``为一种方法，即类中的函数，用于初始化对象的属性；在Python前后中加``__``的方法名为Python默认方法；在类中还可以定义普通方法，方法名前不需要加``__``，定义方法相同

+ 一个类中所有方法的第一个形参始终为``slef``，是一个特殊的对象名，表示当前对象；在调用类中每一个方法时都会自动传入``self``的值，当通过类创建实例的时候都只需要按照顺序给后面的参数提供值
  
  + 对于一个对象调用其所属类的的方法的语法是``参数名.方法名()``，括号内可为空 (表示无实参) 或包含一些实参。如果实参的数量超过一个，它们之间用逗号分隔。这些实参必须和该方法中除了 self 以外的那些形参在数量上相同，并且在顺序上一一对应。

#### -实例的创建与使用

+ 定义类后即可根据该类创建实例，创建方法为``类名(参数1，参数2,...)``，如
  
  ```python
  class Student(): 
        def __init__(self,name,age): 
            self.name=name 
            self.age=age
  student_1=Student(‘Peter',23)
  ```

+ Python使用提供的实参调用所创建类中的方法``__init__()``；方法``__init__()``创建一个特定的实例，并使用提供的值来设置属性；方法``__init__()``并未显式地包含``return``语句，但Python自动返回一个实例

+ 根据命名约定，通常可以认为首字母大写的名称指的是类，而小写的名称指的是根据类创建的实例

+ 要访问实例的属性，可使用句点表示法，即``实例名.属性名``，同样也可以用句点语句调用实例中的方法，即``实例名.方法名(参数)``

#### -属性的处理方法

+ 为属性指定默认值可以直接在``__init__()``方法中直接进行赋值指定，之后调用时实参无需包含该属性

+ 修改属性的值有以下方法
  
  + 直接通过句点语言法访问并赋值修改，如
    
    ```python
    class Student(): 
              def __init__(self,name,age): 
                  self.name=name 
                  self.age=age 
    
          student_1=Student(‘Peter',23)
    
          student_1.age=20
    ```
  
  + 通过自定义方法修改属性的值，如
    
    ```python
    class Student(): 
              def __init__(self,name,age): 
                  self.name=name 
                  self.age=age
    
              #自定义方法
              def update_age(self,age_1):
                  self.age=age_1
    
    student_1=Student(‘Peter',2)
    student_1.age=student.update_age(20)
    ```

### 1.4.2 继承和多态

编写类时，如果你要编写的类是另一个现成类的特殊版本，可使用继承。一个类继承另一个类时，它将自动获得另一个类的所有属性和方法；原有的类称为父类或基类，而新类称为子类或派生类。子类继承了其父类的所有属性和方法，同时还可以定义自己的属性和方法。

#### -子类创建与使用

+ 以上文``Student()``类为例，当需要创建一个新的与``Studeng()``类有类似方法的``Un_student()``类时，可使用继承。需注意在编写子类时文件中必须包含父类，需使用``super()``函数从``__init__()``方法初始化父类的属性
  
  ```python
  class Student():
        def __init__(self,name,age): 
            self.name=name 
            self.age=age
  
        def out(self):
            print("%s is %d years old" %(self.name,self.age))
  
    class Un_student(Student):
        def __init__(self,name,age):
            super().__init__(self,name,age) #从父类中继承属性
  ```

+ 当创建一个子类时，可以为该子类创建新的不同于父类的属性和方法，仍以``Student()``类为例，可在创建子类``Un_student()``时为其添加新的属性和方法，根据子类创建的实例都将包含新的属性和方法，而根据父类创建的实例则没有
  
  ```python
  class Student():
        def __init__(self,name,age): 
            self.name=name 
            self.age=age
  
        def out(self):
            print("%s is %d years old" %(self.name,self.age))
  
  class Un_student(Student):
  
        def __init__(self,name,age,sex):
            super().__init__(self,name,age) #从父类中继承属性
            self.sex=sex #添加新的属性
  
        #添加新的方法
        def describe_information(self):
            print("%s is a %s, and is %d years old" %(self.name,self.sex,self.age))
  ```

+ 当父类中的方法不符合子类的要求时可以进行重写，形成一个与父类中的某个方法在名称、形参列表和返回类型上都相同，但方法体不同的子类，称为覆盖 (overriding)
  
  ```python
  class Un_student(Student):
        --snip--
        def out(self):
            print("He is %d years old" %self.age)
  ```

+ 在创建子类的过程中随着添加属性和方法的增多文件会很长，可以将类的一部分作为一个独立的类提取出来，将大型类拆分成多个协同工作的小类，以``Un_student``类为例
  
  ```python
  class Student:
  
      class Base_Information:
  
          def __init__(self, name, age, sex, num):
              self.name = name
              self.age = age
              self.sex = sex
              self.num = num
  
      class Un_student(Student):
  
          def __init__(self, name, age, sex):
              super().__init__(self, name, age)
              self.base_info = Base_Information()
  
          def describe_information(self):
              print("%s is a %s, and is %d years old" % (self.name, self.sex, self.age))
  
  ```

#### -多态

+ 覆盖的存在产生了继承的优点——多态，多态的特点在于当将类作为参数传向新的方法或函数时可以实现父类中的任意方法

+ 多态的基础是相同的数据类型，即从同一类父类发源；当在面向对象编程中定义一个``class``时即定义了一个数据类型，判断一个变量是否是某个类型可以用``isinstance()``函数进行判断；同时子类的数据类型可视作与父类相同，反之不成立
  
  ```python
  lass Student():
        --snip--
  
    class Un_Student(Student):
        --snip--
  
    a=Student()
    b=Un_Student()
  
    print(isinstance(a,Student)) #True
    print(isinstance(a,Un_Student)) #True
  ```

+ 以下文代码为例，多态的意义在于对于一个可以以父类``Student()``实例为形参调用的函数``put_age()``，子类也可以作为形参调用，不用深究子类是派生出的何种类型，即只要来自于同一个父类``Student``，调用方只管调用，不管细节，而当我们新增一种子类时，只要确保原方法编写正确，不用管原来的代码是如何调用的，即“开闭原则”
  
  ```python
  class Student():
        --snip--
  
    class Un_Student(Student):
        --snip--
  
    def put_age(student):
        print(student.age())
  
    put_age(Student)
    put_age(Un_Student) #输出对应的数
    #创建新的子类
    class PhD(Student):
        --snip--
  
    put_age(PhD) #同样可以正常输出对应的数据
  ```

+ 开闭原则即对扩展开放(允许新增子类)；对修改封闭(不需要修改依赖父类的函数)

+ 继承还可以一级一级地继承下来，任何类最终都可以追溯到根类object
    <img title="" src="https://img2.imgtp.com/2024/04/16/3XQ1fNy8.png" alt="继承树" data-align="inline"> 

### 1.4.3 类的导入

Python允许你将类存储在模块中，然后在主程序中导入所需的模块，导入类的方法包括多种，与函数模块化设计相关联

+ 导入单个类：在编写大规模程序时可以在一个模块中包含单个类，调用类中的方法时仍使用句点语法，如在只有一个模块的``student.py``文件中导入``Student``类并创建实例
  
  ```python
  from student import Student
  
    S_1=Student("Peter",23) #创建实例并传入参数
  
    S_1.out()
  ```

+ 同一模块存储多个类：在同一模块中可以存储任意数量的类，导入时仍使用``from -- import --``语句，调用其中的方法时仍使用句点语法，如包含了``Un_Student``和``PhD``类的``student1.py``文件
  
  ```python
  from student1 import Un_Student
  
    --snip--
  ```

+ 从同一模块导入指定类：可根据实际需求从存储了多个类的一个模块中导入多个类，导入方式同函数，如从存储了``Student``、``Un_Student``、``PhD``等多个类的``student.py``文件中导入``Un_Student``、``PhD``
  
  ```python
  from student import Un_Student,PhD
    --snip--
  ```

+ 导入整个模块：导入整个模块会使代码具有更大的易读性，导入方式简洁，引用时同样使用句点法``module_name.class_name.func_name()``
  
  ```python
  import student
  
    S_1=student.Student("Peter",23) #创建实例并传入参数
  
    S_1.out()
  ```

+ 导入模块中所有的类：导入模块中所有的类与函数的导入方法相同，即``from moudle_name import *``

+ 在一个模块中导入另一个模块：有时候，需要将类分散到多个模块中，以免模块太大，或在同一个模块中存储不相关的类。将类存储在多个模块中时，可能会存在一个模块中的类依赖于另一个模块中的类的情况。在这种情况下，可在前一个模块中导入必要的类。如若将``Student``存储在``s_1.py``模块中，将 ``Un_Student``、``PhD``存储在``s_2.py``模块中，由于``Un_Student``、``PhD``依赖于父类``Student``，需要在
    ``s_2.py``模块中导入``Student``，导入方式与正常情况相同

## 1.5 文件处理

### 1.5.1 文件数据读取

#### -文件路径

文件路径包括绝对路径和相对路径，当Python程序需要读取文件时，若程序与文件在同一文件夹，则无需指定文件具体路径，仅需提供文件名，但当二者在不同文件夹时则需要根据具体情况传入文件路径参数

+ 相对路径：当所要操作的文件位于程序所在目录的其他文件夹时，可以使用相对文件路径来打开文件
  
  ```python
  #Linux/OS X系统
  with open('text_files/filename.txt') as file_object:
  --snip--
  
  #Windows系统
  with open('text_files\filename.txt') as file_object:
  --snip--
  ```

+ 绝对路径：在一般情况下，可以直接将文件在计算机中的准确位置(即绝对路径)提供给程序，绝对路径较长时可存储在变量中，再将变量传递给程序中读取文件的函数
  
  ```python
  #Linux/OS X系统
  file_path = '/home/other_files/text_files/filename.txt'
  with open(file_path) as file_object:
  
  #Windows系统
  file_path = 'C:\Users\other_files\text_files\filename.txt'
  with open(file_path) as file_object:
  ```

#### -文件的整体读取与逐行读取

+ 文件的读取首先需要打开文件，常用的打开文件方式有两种
  
  + 直接使用``open()``函数打开文件
    
    ```python
    f=open(file_name,mode)
    ```
    
     该语句打开文件名为``file_name``的文件并创建对象``f``，其中``mode``包括``'r'``(读)、``'w'``(写)、``'a'``(追加)、``'r+'``(读写)，代表打开方式，若含``b``则表示以二进制方式打开，文件处理结束之后需使用``f.close()``      语句关闭文件
  
  + 使用``with``关键字打开文件
    
    ```python
    with open(filename, mode) as f:
    ```
    
      关键字with在不需要访问文件后将其关闭。在调用``open()``和``close()``函数来打开和关闭文件时，如果程序存在bug，导致``close()``语句未执行，文件将不会关闭，可能会导致数据丢失或受损；如果在程序中过早地调用close()，则需要使用文件时已无法访问，会导致更多的错误。通过使用``with``关键字，Python可直接确定合适的时间将文件关闭。

+ 文件的读取
  
  + 整体读取文件时直接使用``read``方法，将文件内容存储在变量中，通过处理变量对文件进行处理，``read()``在读取到文件末尾时返回一个空字符串作为结束的标志，可使用``rstrip()``方法对其进行处理
    
    ```python
    with open(filename, mode) as f:
        contents=f.read()
        --snip--
    ```
  
  + 要对文件进行逐行读取，可以使用循环语句实现，``read()``在读取到每行末尾时返回一个空字符串作为结束的标志，可使用``rstrip()``方法对其进行处理
    
    ```python
    file_name = 'filename.txt'
    
    with open(file_name) as file_object:
        for line in file_object:
        --snip--
    ```

+ 文件读取中可以使用``readline()``方法读取文件的首行，同时可以使用``readlines()``方法读取全部文件的行并返回一个列表
  
  ```python
  with open(file_name) as file_object:
      lines=file_objects.readlines() #使用readlines方法创建一个包含所有行的列表lines
      --snip--
  ```

### 1.5.2 文件写入

#### -写入空文件

+ 通过在调用``open()``函数并向其提供提供第二个参数``'w'``可进行文件的写入，若打开的文件不存在则会自动创建；若打开的文件中已有内容则会清空原有的内容

+ Python只能将字符串写入文本文件，要将数值数据存储到文本文件中时必须先使用函数``str()``将其转换为字符串格式

+ 如果想在文件中写入多行内容，则需在字符串中添加换行符``\n``
  
  ```python
  filename=file_name
  with open(filename,'r') as file_project:
      file_project.write("USTC\n")
      file_project.write("1958\n")
  ```

#### -追加文件内容

+ 如果要向文件添加内容而不覆盖原有的内容，可以通过传入第二个参数``'a'``将内容添加到文件的末尾
  
  ```python
  filename=file_name
  with open(filename,'r') as file_project:
      file_project.write("USTC\n")
      file_project.write("1958\n")
  ```

### 1.5.3 存储数据

+ 很多程序要求将用户输入的信息或程序运行中产生的信息进行存储，模块``json``能够将简单的Python数据结构转储到文件中，并在程序再次运行时加载该文件中的数据，也使用``json``在Python程序之间分享数据。同时JSON(JavaScript Object Notation)是一种常用的应用程序间数据交换格式，这种数据格式具有通用性，能够将以JSON格式存储的数据与使用其他编程语言的人分享

+ ``josn``模块中最基础的两个函数为``josn.dump()``和``josn.load``，``json.dump()``用来存储数据，``json.load()``用来调用数据
  
  + 存储数据 
    
    ```python
    import json
    
    numbers = [2, 3, 5, 7, 11, 13]
    filename = 'numbers.json'
    with open(filename, 'w') as f_obj:
      json.dump(numbers, f_obj)    
    ```
  
  + 调用数据
    
    ```python
    import json
    
    filename = 'numbers.json'
    with open(filename) as f_obj:
      numbers = json.load(f_obj)
    
    print(numbers)
    ```

## 1.6 异常

### 1.6.1 错误的分类

程序发生的错误可分为三大类：语法错误、逻辑错误和运行时错误

+ 语法错误是指程序违反了程序设计语言的语法规则，例如语句``if 3>2print(’3>2’)``因冒号缺失导致语法解析器报错``SyntaxError:invalid syntax``

+ 逻辑错误是指程序可以正常运行，但结果不正确

+ 运行时错误也称为异常 (exception), 是指程序在运行过程中发生了意外情形而无法继续运行。每当发生错误时，Python都会创建一个异常对象。如果程序中有处理该异常的代码，程序将继续运行；如果未对异常进行处理，程序将停止并显示一个``traceback``，其中包含有关异常的报告
  
  #### - 常见异常

+ ``ZeroDivisionError``异常，即除0异常
  
  ```python
  print(5/0)
  
  >>>Traceback (most recent call last):
         File "division.py", line 1, in <module>
             print(5/0)
       ZeroDivisionError: division by zero
  ```

+ ``FileNotFoundError``异常，即文件路径异常
  
  ```python
  filename = 'file_name.txt'
  
  with open(filename) as f_obj:
      contents = f_obj.read()
  
  >>>Traceback (most recent call last):
         File "file_name.py", line 3, in <module>
             with open(filename) as f_obj:
       FileNotFoundError: [Errno 2] No such file or directory: 'file_name.txt'
  ```

#### - ``try-except``代码块处理异常

+ 当程序可能发生了错误时，可编写一个``try-except``代码块来处理可能引发的异常。Python尝试运行程序代码，这些代码中包含如果代码引发了指定的异常该怎么办。

+ 对于处理异常的代码，将导致错误的代码行放在``try``代码块中。如果``try``代码块中的代码运行正常，Python将跳过``except``代码块；如果``try``代码块中的代码导致了错误，Python将查找对应的``except``代码块并运行其中的代码
  
  ```python
  try:
      print(5/0)
  except ZeroDivisionError:
      print("You can't divide by zero!")
  
  >>>You can't divide by zero!
  ```

+ 在``try-except``代码块中还可包含``else``代码块，包含依赖于``try``代码块成功执行的代码
  
  ```python
  try:
      answer=5/1
      print(ansewr)
  except ZeroDivisionError:
      print("You can't divide by zero!")
  else:
      print(ansewr)
  
  >>>5
  ```

### 1.6.2 自定义异常类

用户在程序中可以使用内置异常类，也可以根据需要自定义异常类，自定义的异常类以``Exception``作为父类。以自定义异常类``InputRangeError``为例,``gcd``函数判断用户输入的两个整数中是否存在负数，若是则抛出``InputRangeError``异常，因为这种情形下``while``循环不会终止。异常导致``gcd``函数返回，该异常对象被 ``except`` 语句块捕获，然后输出出错信息。``sys.argv``是一个列表，存储了用户在命令行输入的所有字符串。索引值为0的字符串是程序的名称，其余字符串是用户输入的参数。如果输入的参数个数少于两个，则读取``sys.argv[2]``导致``IndexError``，如果某个参数不是整数，则``int``函数报错``ValueError``

```python
class InputRangeError(Exception):
    """Raised when an input is not in suitable range
    Attributes:
        message -- explanation of suitable range
    """

    def __init__(self, message):
        self.message = message


def gcd(a, b):
    if a <= 0 or b <= 0:
        raise InputRangeError('Each integer should be positive')
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
    return a


import sys

try:
    x = int(sys.argv[1])
    y = int(sys.argv[2])
    print('The greatest common divisor of %d and %d is %d' % (x, y, gcd(x, y)))
except IndexError:
    print('Two arguments must be supplied on the command line')
except ValueError:
    print('Each argument should be an integer.')
except InputRangeError as ex:
    print(ex.message)
finally:
    print("executing finally clause")

```

``finally``语句块是可选的，必须位于所有其他语句块之后。无论是否发生异常``finally``语句块都会运行，通常用于回收系统资源等善后工作。语句块的语义规则如下

+ 如果``try``语句块在运行过程中抛出了异常，且未被任何``except``语句块处理，则运行``finally``语句块后会重新抛出该异常

+ 如果``except``语句块和``else``语句块在运行过程中抛出了异常，则运行``finally``语句块后会重新抛出该异常

+ 如果``try``语句块中即将运行``break``、``continue``或``return``等跳转语句，则会先运行``finally``语句块再运行跳转语句

## 1.7 测试

### 1.7.1  测试函数

+ Python的``unittest``模块提供了代码测试工具，测试有单元测试、测试用例、全覆盖式测试等形式；单元测试用于核实函数的某个方面没有问题；测试用例是一组单元测试，这些单元测试一起核实函数在各种情形下的行为都符合要求。全覆盖式测试用例包含一整套单元测试，涵盖了各种可能的函数使用方式，以对``calc.py``文件中的``add()``函数进行测试为例
  
  ```python
  #calc.py
  def add(a,b):
      return a+b
  ```

+ 要为函数编写测试用例，可先导入模块``unittest``以及要测试的函数，再创建一个继承``unittest.TestCase``的类，并编写一系列方法对函数行为的不同方面进行测试；类的命名最好与要测试的函数相关，并包含字样``Test``；类中包含的方法用于对被测试函数的各方面进行测试，方法命名时以``test_``打头，运测试代码时，所有以``test_``打头的方法都将自动运行；在测试方法中一般调用``unittest``中的断言方法``assertEqual()``用来核实得到的结果是否与期望的结果一致
  
  ```python
  #test_calc.py
  import unittest
  from calc import add
  class Cal_TestCase(unittest.TestCase):
      def test_add(self):
          out_put=add(1,2)
          self.assertEqual(out_put,3)
  unittest.main()
  ```
  
     当运行测试文件后若测试通过，将输出对应的测试结果及测试时间，第一行的``.``表示通过一个测试用例
  
  ```
  .
  --------------------------------------------------------
  Ran 1 test in 0.000s
  OK
  ```
  
    当测试用例未通过时会返回相应的错误信息，将在下文指出

+ 当对``add()``函数进行修改后原测试代码可能将测试失败，如更新函数为
  
  ```python
  #calc.py
  def add(a,b,c):
      return a+b+c
  ```
  
    此时调用测试函数时会输出失败
  
  ```
  E
  ==================================================================
  ERROR: test_add (__main__.Cal_TestCase.test_add)
  ----------------------------------------------------------------------
  Traceback (most recent call last):
    File "c:\Users\jair\Desktop\Python\test\test.py", line 7, in test_add
      out_put = add(1, 2)
            ^^^^^^^^^
  TypeError: add() missing 1 required positional argument: 'c'
  
  ----------------------------------------------------------------------
  Ran 1 test in 0.001s
  
  FAILED (errors=1)
  ```
  
    在失败输出中，第1行输出一个字母E，它指出测试用例中有一个单元测试导致了错误，之后一行指出产生``ERROR``的函数，并在之后指出详细位置及具体错误，最后一行指出了测试用例未通过的个数    
    测试未通过时应检查未通过函数的代码部分并作出修改

+ 在需要添加测试时只需要在定义的类中加入新的方法，以测试修改后的``add()``函数为例，使用可变参数``c``，此时函数既可以计算两数又可以计算三数，此时需要对测试用例进行修改
  
  ```python
  #calc.py
  def add(a,b,c=NULL):
      if c:
          out_put=a+b+c
      else:
          out_put=a+b
      return out_put
  ```
  
  ```python
  #test_calc.py
  import unittest
  from calc import add
  class Cal_TestCase(unittest.TestCase):
      def test_add_2(self):
          out_put=add(1,2)
          self.assertEqual(out_put,3)
      def test_add_3(self):
          out_put=add(1,2,3)
          self.assertEqual(out_put,6)
  unittest.main()
  ```
  
    此时会输出正确测试结果
  
  ```
  ..
  ----------------------------------------------------------------------
  Ran 2 tests in 0.000s
  
  OK
  ```

### 1.7.2  测试类

#### -断言方法

Python在``unittest.TestCase``类中提供了很多断言方法，其用法如下
| 方法 | 含义 |
|:--:|:--:|
| assertEqual(a,b) | 核实a==b  |
| assertNotEqual(a,b) | 核实a!=b |
| assertTrue(a) | 核实a为True |
| assertFalse(a) | 核实a为False |
| assertIn(item,list) | 核实item在list中 |
| assertNotIn(item,list) | 核实item不在list中 |

#### -类的测试

类的测试与函数的测试相似，所做的大部分工作都是测试类中方法的行为，以对一个学生信息管理类的测试为例

```python
#Information.py
class information():
    def __init__(self,name):
        self.name=name
        self.skills=[]

    def out_put(self):
        print(self.name + "Please enter your skills")

    def store_skill(self,new_skill):
        self.skills.append(new_skill)

    def show_skills(self):
        print(self.name + ":\n")
        for skill in self.skills:
          print("\t -"+skill)
```

首先对输入一个信息这一个方面进行测试

```python
#test_information.py
import unittest
from calc import information

class Test_information(unittest.TestCase):
    def test_store_angle_information(self):
        name = 'Peter'
        test_student = information(name)
        test_student.store_skill("Ball")
        self.assertIn('Ball', test_student.skills)

unittest.main()
```

测试成功，输出对应的结果

```
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

同时可以对输入多个信息进行测试

```python
#test_information.py
import unittest
from calc import information

class Test_information(unittest.TestCase):
    def test_store_angle_information(self):
        name = 'Peter'
        test_student = information(name)
        test_student.store_skill("Ball")
        self.assertIn('Ball', test_student.skills)

    def test_store_three_information(self):
        name = 'peter'
        test_student = information(name)
        skills = ['ball', 'run', 'sing']

        for skill in skills:
            test_student.store_skill(skill)

        for skill in skills:
            self.assertIn(skill, test_student.skills)

unittest.main()
```

测试成功，输出对应结果

```
..
----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK
```

#### - ``setUp()``方法

在前面的``test_information.py``中，在每个测试方法中都创建了一个``test_student``实例，并在每个方法中都创建了答案。``unittest.TestCase``类包含的方法``setUp()``可以实现创建一次对象，并在每个测试方法中使用它们。如果在``TestCase``类中包含了方法``setUp()``，Python将先运行它，再运行各个以``test_``打头的方法

```python
# test_information_2.py
import unittest
from calc import information

class Test_information(unittest.TestCase):
    def setUp(self):
        name = 'peter'
        self.test_student = information(name)
        self.responses = ['ball', 'run', 'sing']

    def test_store_angle_information(self):
        self.test_student.store_skill(self.responses[0])
        self.assertIn(self.responses[0], self.test_student.skills)

    def test_store_three_information(self):
        for skill in self.responses:
            self.test_student.store_skill(skill)

        for skill in self.responses:
            self.assertIn(skill, self.test_student.skills)

unittest.main()
```

## 1.8 程序运行时间分析与测量

### 1.8.1 算法和时间性能分析

算法是求解问题的一系列计算步骤，用来将输入数据转换成输出结果，评价算法优劣的一个重要指标是时间性能，即在给定的问题规模下运行算法所消耗的时间

程序是使用某种程序设计语言对一个算法的实现。算法的时间性能是决定程序的运行时间的关键因素。时间性能的分析对算法在计算机上的运行过程进行各种简
化和抽象，把算法的运行时间定义为问题规模的函数，称为时间复杂度。分析的主要结果是时间复杂度的增长阶(order of growth)，即时间复杂度的增长速度有多快。当问题规模充分大时，增长阶决定了算法的时间性能。有些算法的运行时间受问题输入数据的影响很大，例如排序算法。此时需要对最坏、平均和最好三种情形进行分析

时间复杂度通常使用三种记号描述

+ $O$记号：设函数$f(n)$和$g(n)$是定义在非负整数集合上的正函数，如果存在两个正整数$c$和$n_0$，使得当$n \ge n_0$时，$f(n)\le cg(n)$成立，则记为$f(n) = O(g(n))$，即当$n$增长到充分大以后，$g(n)$是$f(n)$的一个上界

+ $\Omega$记号：$Omega$记号与$O$记号对称，即$f(n)=\Omega(g(n))$当且仅当$g(n)=O(f(n))$

+ $\Theta$记号：$f(n)=\Theta(g(n))$当且仅当$f(n) = O(g(n))$并且$g(n) = O(f(n))$

## 1.8.2 算法的时间复杂度

算法由一些不同类别的的基本运算组成，包括算术运算、关系运算、逻辑运算、数组 (列表) 元素的访问和流程控制等。这些基本运算的运行时间都是常数。算法的时间复杂度等于每种基本运算的运行时间和其对应运行次数的乘积的总和，其中运行次数是问题规模的函数

为了简化分析，一般只考虑运行次数最多的基本运算，因为当问题规模较大时它们的运行时间是时间复杂度中增长阶最高的项。对于单层循环或嵌套的多层循环，
运行次数最多的基本运算位于最内层循环，循环的时间复杂度的增长阶由最内层循环的运行次数决定

对于由递归结构构成的算法，根据递归公式可以得到时间复杂度满足的递归方程$T(n) = aT(n/b) + f(n)$，$T(n)$ 表示求解规模为非负整数$n$的原问题的时间复杂度，原问题被分解成$a$个规模为$n/b$的与原问题结构类似的子问题，其中$a \ge 1$和$b \ge 1$是常数，$n/b$等于$⌊n/b⌋$或$⌈n/b⌉$。原问题的解可由这些子问题的解合并
而成，合并过程的时间复杂度是一个函数$f(n)$，该方程可由Master定理求解

+ 若$f(n)=O(n^{log_b{a}-\varepsilon})$对于常数 $\varepsilon>0$成立，则$T(n) =\Theta(n^{log_ba})$；

+ 若$f(n)=\Theta(n^{log_b a})$成立，则$T(n)=\Theta(n^{log_b a} log_2 n)$；

+ 若$f(n)=\Omega(n^{log_b a+ \varepsilon })$ 对于常数 $ \varepsilon> 0$ 成立，并且当$n$充分大时$af(n/b) \le cf(n)$对于常数$c < 1$成立，则$T(n) = \Theta(f(n))$

大多数算法的时间复杂度属于以下七类，按照增长阶从低到高的次序依次为：$O(1)，O(log n)，O(n)，O(n log n)，O(n^2)，O(n^3) 和 O(a^n)(a > 1)$，一般认为增长阶为$O(n^k)$(k 是一个常数) 的算法是可行的。增长阶为$O(a^n) (a > 1)$ 的算法只适用于规模较小的问题

常见的各复杂度的算法有

+ $O(1)$:时间复杂度属于$O(1)$的算法与问题的规模无关，包括算术运算、逻辑运算、关系运算、读写简单类型的变量(int、float 、bool等)、读写数组中某个索引值的元素等

+ $O(log(n))$:二分查找算法的时间复杂度是$O(log n)$

+ $O(n)$:线性查找算法的时间复杂度是$O(n)$，循环在列表 s 中查找指定元素k是否出现，若出现则返回其索引值，否则返回-1表示未找到。循环在最坏情形下的运行次数是列表s的长度n

+ $O(n\,log\,n)$:归并排序的时间复杂度是$O(n\,log\,n)$，归并排序的运行时间包括三部分，即递归调用左子列表、递归调用右子列表和归并排好序的两个子列表。对于长度为 n 的列表，时间复杂度$T(n)$满足方程$T(n) = 2T(n/2) + \Theta(n)$

+ $O(n^2)$:插入排序算法的时间复杂度是$O(n^2)$,对于长度为$n$的列表s，内层循环在最坏情况下(待排序列表是从大到小的顺序)的运行次数为$1 + 2 + ... + (n − 1) = n(n − 1)/2$

+ $O(n^3)$:

+ $O(a^n)(a>1)$:


