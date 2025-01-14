## Аннотация

Работа посвящена исследованию влияния выбора функций активации на решение PINN для нахождения функции Эйри.

_Дифференциальное уравнение Эйри:_

$$\frac{d^2y}{dx^2} - xy = 0, \quad x \in (a, b)$$

_Граничные условия:_

$$\begin{equation}
  \begin{cases}
    y(0) = \frac1{3^{2/3}\Gamma(2/3)},\\
    \frac{dy}{dx}(0) = -\frac1{3^{1/3}\Gamma(1/3)}.
  \end{cases}
\end{equation}$$

_Задача:_ Исследовать влияние различных функций активации (Sigmoid, Tanh,
Softplus, Sin) на точность и эффективность нахождения функции Эйри с помощью PINN.

## Дополнение к параметрам и условиям задачи

Для использования граничных условий в качестве данных требуется провести аппроксимацию. По определению, $$\Gamma(x) = \int\limits_0^{+∞}t^{x-1}e^{-t}dt$$.

Таким образом, $$(\ast)$$

$$\begin{equation}
  \begin{cases}
    y(0) \approx \frac1{3^{2/3}\cdot(1.354)} \approx 0.355,\\
    y^{\prime} \approx - \frac1{3^{1/3} \cdot (2.679)} \approx -0.259.
  \end{cases}
\end{equation}$$

$$(\ast)$$ функция $$\Gamma$$ была посчитана с помощью math.gamma библиотеки math.

В ходе решения поставленной задачи будем считать, что под $$(a, b)$$ подразумевается интервал $$(-10, 3)$$, так как он являяется достаточным для наблюдений за функцией, а также именно на $$(-10, 3)$$ функция Эйри с обозначенными граничными условиями демонстрирует смену характера поведения.

## Сбор датасета посредством численных методов

Для получения качетсвенного датасета с большим набором данных воспользуемся численным методом Рунге-Кутта 4-го порядка точности по отношению к нашей задаче.

Реализация метода и его применение построены на функции solve_ivp (библиотека scipy, модуль integrate).

## Функция потерь (Loss)

$$MSE \textrm{-} PINN(y^{true}, y^{pred}) = G_{eq} + \lambda \cdot G_{b} = \frac1{N} \sum\limits_{i=1}^N (y_{i}^{\prime\prime} - x_{i}*y_{i})^2 + \lambda \cdot [(y(0)- f(0))^2 + (y^{\prime}(0) - f^{\prime}(0))^2]$$

Полькольку PINN-нейросеть, обрабатывая данные, в качестве результата возвращает $$y_{i} = f(x_{i})$$ величина $$y_{i}^{\prime} = f_{i}^{\prime}(x_{i})$$ может быть посчитана с помощью автоматического дифференцирования, поддерживаемого модулем torch.

## Вычисляемые метрики качества

$$MSE(y^{true},y^{pred}) = \frac1{N} \sum\limits_{i=1}^N (y_{i} - f(x_{i}))^2$$

$$RMSE(y^{true},y^{pred}) = \sqrt{MSE} = \sqrt{ \frac1{N} \sum\limits_{i=1}^N (y_{i} - f(x_{i}))^2}$$

$$R^{2}(y^{true},y^{pred}) = 1 - \frac{\sum_{i=1}^N (y_{i} - f(x_{i}))^2}{\sum_{i=1}^N (y_{i} - \overline{y})^2}$$

$$MAE(y^{true},y^{pred}) = \frac1{N} \sum\limits_{i=1}^N | y_{i} - f(x_{i})|$$

$$MAPE(y^{true},y^{pred}) = \frac1{N} \sum\limits_{i=1}^{N} \frac{|y_{i}-f(x_{i})|}{|y_{i}|}$$

$$WAPE(y^{true},y^{pred}) = \frac{\sum_{i=1}^{N}|y_{i} - f(x_{i})|}{\sum_{i=1}^{N}|y_{i}|}$$
