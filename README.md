# Deep_Learning_with_Tensorflow_notes :spiral_notepad::pencil2::pushpin: #

## Tensörler :heartbeat: ## 

TensorFlow, tf.Tensor nesneleri olarak temsil edilen çok boyutlu diziler veya tensörler üzerinde çalışır. İşte üç boyutlu bir tensor :point_right::

```
import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.],
                 [8.,9.,10.]],name = 'constant_x')
                 
```

constant_x adında bir tf.Operation oluşturur ve bir tensor döndürür.
```
print(x.shape)
print(x.dtype)
```
Tensor.shape : her bir ekseni boyunca tensörün boyutunu size söyler. :point_right: yukarıdaki tensor için (3,3)
Tensor.dtype : tensördeki tüm öğelerin türünü söyler. :point_right: yukarıdaki tensor için <dtype: 'float32'>

TensorFlow, tensörler üzerinde standart matematiksel işlemlerin yanı sıra makine öğrenimi için özelleştirilmiş birçok işlemi uygular.

tf.matMul():  işlevi, A * B olmak üzere iki matrisin iç çarpımını hesaplamak için kullanılır.

'@' sembolü Matris çarpımı olarak kullanılır ve tf.matmul()'a eşittir.:woman_technologist:

```
x @ tf.transpose(x)
----------------------
c = tf.matmul(x,tf.transpose(x)) 
```
Bu iki ifade aynı sonucu verir.

tf.add(): iki tf.Tensor nesnesinin birbirine eklenmesi

tf.function, işlev için bir TensorFlow statik yürütme(static execution graph) grafiği oluşturmak üzere TensorFlow Autograph'ı kullanır.

```
@tf.function
def add(a,b):
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms
    print(c)
    return c
```

```
@tf.function
def mathmul(a,b):
  return tf.matmul(a, b)
  
 ```

TensorFlow kodu iki modda çalıştırılabilir: eager mode ve graph mode. Eager modu, kod çalıştırmanın standart, etkileşimli yoludur: bir işlevi her çağırdığınızda yürütülür.Bununla birlikte, grafik modu(graph mode) biraz farklıdır. Grafik modunda, işlevi yürütmeden önce TensorFlow, işlevi yürütmek için gerekli işlemleri içeren bir veri yapısı olan bir hesaplama grafiği(computation graph) oluşturur.

```
a = tf.constant(np.array[1. , 2. , 3. ])
b = tf.constant(np.array[4. , 5. , 6. ])
```
a ve b tensörünü oluşturuyoruz
```
c = tf.tensordot(a,b,1)
```
dot product(iç çarpım) hesaplar ve sonucu c'ye atarız.
Ancak şu ana kadar herhangi bir hesaplama yapılmadı. c yalnızca henüz bir değeri olmayan execution graph(yürütme grafiği) temsil eder.

```
session = tf.Session()
output = session.run(c)
session.close()
```

'session' bir kez yürütüldüğünde, bir hesaplama gerçekleşir ve c'ye bir değer atanır. Bu ara sonuçlara erişmeyi zorlaştırdığı için hata ayıklamayı(debug) zorlaştırır.

```
a = tf.constant(np.array[1. , 2. , 3. ])
b = tf.constant(np.array[4. , 5. , 6. ])

c = tf.tersordot(a,b,1)

output = c.numpy()
```
Eager execution etkinleştirildiğinde, kod satır satır yürütülür ve ara sonuçlar anında kullanılabilir. Tensorflow kodunun sıradan python kodu gibi görünmesini sağlar.


![image](https://upload.wikimedia.org/wikipedia/commons/4/45/Dimension_levels.svg)

Sıfır boyut(zero dimension) tek bir nesne/öge, skaler bir değer olarak görülebilir. Bir boyut(one dimension) ise bir çizgi veya bir vektör olarak görülebilir. 2-boyut(2-dimesion) bir yüzey gibi düşünebiliriz. Matrisleri buna örnek olarak verebiliriz. 3-boyutta ise tensorleri örnek verebiliz.

```
Scalar = tf.constant(8)
Vector = tf.constant([2,6,3])
Matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
Tensor = tf.constant( [ [[1,2,3],[2,3,1],[3,4,5]] , [[4,5,6],[5,8,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
```

![image](https://miro.medium.com/max/900/1*AB3CIu1s6LllkcXy4ZpYMQ.png)

[image source](https://towardsdatascience.com/how-convolution-neural-networks-interpret-images-1f99913070b2)

Görüntüleri düşündüğümüzde genişliğe ve yüksekliğe sahip olduklarını biliriz. Görüntülerin renklere de sahip olduğunu düşünürsek bize tek boyut yeterli olmayacaktır.Görüntüler renk kanallarıyla gösterilir. RGB, Red (kırmızı):red_circle:, Green (Yeşil):green_circle: ve Blue (Maviden):large_blue_circle: oluşur. 

tf.Variable(): Değişkenler tf.Variable sınıfı aracılığıyla oluşturulur ve izlenir. Bir tf.Variable , değeri değiştirilebilen bir tensörü temsil eder. Belirli işlemler, bu tensörün değerlerini okumanıza ve değiştirmenize izin verir. 

 Medical Cost Personal Datasets [Source](https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial/data)
 
 ### Linear Regression with TensorFlow ###
 
 ```
df = pd.read_csv("insurance.csv")
 ```
 Vücut Kitle Endeksi (BMI) ile charge(ücret)'i tahmin etmek için doğrusal regresyonu(Linear Regression) kullandığımızı varsayalım.
 
  ```
 train_x = np.asanyarray(df[['bmi']])
train_y = np.asanyarray(df[['charges']])
 ```
restgele bir şekilde a ve b değişkenlerini başlatıyoruz.
 
  ```
a = tf.Variable(20.0)
b = tf.Variable(30.2)
 
  ```
  
 lineer fonksiyonu tanımlıyoruz. Y = aX + b -->> Burada Y bağımlı değiken, a 'eğim(slope)' veya 'gradient', X bağımsiz değişken ve b 'intercept' olarak adlandırılır.
  
  
 ```
 def h(x):
   y = a*x + b
   return y
    
 ```
 
 
Loss Fonksiyonunu tanımlıyoruz. Tahmin edilen değerler ile hedef değerler(sahip olduğumuz değerler) arasındaki farkın kare hatasını(squared error) minimize etmeyi hedefliyoruz. 


   ```
def loss_object(y,train_y) :
    return tf.reduce_mean(tf.square(y - train_y))

   ```
Geriye yayılım(backpropagation) ile parametreler güncellenmektedir. 
   
    ```
   learning_rate = 0.01
train_data = []
loss_values =[]
a_values = []
b_values = []
# steps of looping through all your data to update the parameters
training_epochs = 200

# train model
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_predicted = h(train_x)
        loss_value = loss_object(train_y,y_predicted)
        loss_values.append(loss_value)

        # get gradients
        gradients = tape.gradient(loss_value, [b,a])
        
        # compute and adjust weights
        a_values.append(a.numpy())
        b_values.append(b.numpy())
        b.assign_sub(gradients[0]*learning_rate)
        a.assign_sub(gradients[1]*learning_rate)
        if epoch % 5 == 0:
            train_data.append([a.numpy(), b.numpy()])
   
    ```
