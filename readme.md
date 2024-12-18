# Метод Нумерова для решения краевой задачи

### Описание метода

Метод Нумерова используется для численного решения краевых задач второго порядка. Он позволяет учитывать нелинейную правую часть с высокой точностью благодаря введению дополнительных поправок в аппроксимацию.

### Матрица СЛАУ

В процессе численного решения формируется трёхдиагональная матрица СЛАУ

```math
A =
begin{pmatrix}
-2 + alpha_1 & 1 + beta_1 & 0 & 0 & dots & 0 
1 + beta_1 & -2 + alpha_2 & 1 + beta_2 & 0 & dots & 0 
0 & 1 + beta_2 & -2 + alpha_3 & 1 + beta_3 & dots & 0 
vdots & vdots & vdots & vdots & ddots & vdots 
0 & 0 & dots & 1 + beta_{N-3} & -2 + alpha_{N-2} & 1 + beta_{N-2} 
0 & 0 & dots & 0 & 1 + beta_{N-2} & -2 + alpha_{N-1}
end{pmatrix}.
```

#### Элементы матрицы
- Главная диагональ
  ```math
  -2 + alpha_i, quad alpha_i = -frac{5h^2}{6} cdot e^{y_i}.
  ```
- Верхняя и нижняя диагонали
  ```math
  1 + beta_i, quad beta_i = -frac{h^2}{12} cdot e^{y_i}.
  ```

Эти формулы учитывают нелинейные поправки, связанные с правой частью уравнения.

### Постановка задачи

На каждой итерации метода Нумерова решается следующая СЛАУ

```math
A cdot begin{pmatrix}
Delta y_1 
Delta y_2 
Delta y_3 
vdots 
Delta y_{N-1}
end{pmatrix} =
begin{pmatrix}
R_1 
R_2 
R_3 
vdots 
R_{N-1}
end{pmatrix},
```

где
- (Delta y_i) — исправления к текущим значениям (y_i) на итерации.
- (R_i) — вектор невязки, вычисляемый как
  ```math
  R_i = h^2 cdot frac{1}{12} left(f_{i+1} + 10f_i + f_{i-1}right) - left(y_{i+1} - 2y_i + y_{i-1}right),
  ```
  где (f_i = f(x_i, y_i) = e^{y_i}).

### Итерационный процесс

1. Инициализация
   - На первом шаге задаётся начальное приближение (y^{(0)}) как линейная интерполяция между граничными условиями.

2. Построение матрицы
   - Вычисляются элементы матрицы (A) (главная, верхняя и нижняя диагонали) на основе текущего приближения (y^{(k)}).

3. Вычисление невязки
   - Формируется правая часть (R), зависящая от текущих приближений.

4. Решение СЛАУ
   - Решается система (A cdot Delta y = R) методом прогонки (или Гаусса).

5. Обновление решения
   - Новые значения вычисляются как
     ```math
     y_i^{(k+1)} = y_i^{(k)} + Delta y_i.
     ```

6. Проверка сходимости
   - Процесс повторяется до тех пор, пока норма невязки (R) не станет меньше заданного порога.


### Компиляция и запуск:
`g++ lab2.c -o lab2 -fopenmp -lm`
`lab2.exe 1000 1 4`
