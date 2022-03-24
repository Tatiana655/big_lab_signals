# Intelligent Placer

## Постановка задачи
### Общее
Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник.  

### Ввод / вывод
Ввод: фотография предметов с многоугольником в формате jpg

Вывод: ответ "y" / "n" в фаил <ans_img_name>.txt

### Требования 
#### Фотометрические
- Высота съемки 30 - 45 см
- Фотографировать предметы видом сверху (параллельно поверхности), допускается угол наклона камеры меньше 5 градусов
- При одинаковом равномерном освещении
- на фотографии отсутствуют пересвеченные и серо-черные области
- Без цветовой коррекции
- Без сжатия
- разрешение фотографии 2592x4608

#### По расположению объектов на фотографии
- На фоне фотографии не должно быть лишних предметов, кроме тех, которые заранее были известны
- Объекты не могут перекрывать друг друга (расстояние между каждыми не менее 10 px)
- Объекты целиком помещаются на листе бумаги
- Границы объектов должны четко выделяться на фоне 
- Один объект может присутствовать на  одной фотографии ровно один раз, кроме star (их максимум может быть три штуки)

#### Фон
- белый лист бумаги размером около А4, допускается погрешность 1 см по каждой оси листа, полностью помешающийся на фотографии 
- Ориентация листа не важна
- В пространстве находится в горизонтальном положении на ровной поверхности (без выпуклостей и впадин)
- На одной половинке листа A4 (размер половины - примерно A5) изображён многоугольник, на другой лежат объекты.

#### Многоугольник
- Многоугольник задается фигурой, нарисованной темным маркером на белом листе бумаги, сфотографированной вместе с предметами.
- Многоугольник является простым: две его стороны не могут пересекаться как крест.
- Толщина нарисованной линии не превышает 5мм

### Датасет с объектами
Фотографии объектов: https://drive.google.com/drive/folders/1rrVlvPRfrt-whINtHcryGTQC_jkgdkBm?usp=sharing

#### Замечание к реализации
- Если на изображении есть star и loupe, то одну звезду можно вложить внутрь круга loupe и поместить в многоугольник.
- При расположени объектов в многоугольнике расстояние между объектами должно быть больше или равно 1px.

## Сбор данных
На гугл диске фотографии и таблица с ответами и комменариями про особенные случаи: https://drive.google.com/drive/folders/1FsFPt8-jVN2nXhVPMfr8zcz-5BrAyhLl?usp=sharing

## План
1) Нахождение многоугольника и листа бумаги
Применяется цветовой фильтр. Находим контуры. Самый большой контур - лист А4. Два контура с близкими центрами - многоугольник. Также определеряется позиция многоугольника относительно листа в зависимости от ориентации фотографии (соотношения сторон фотографии). Функция find_polygon_paper()
2) Определение объектов по особым точкам
Выбирается та сторона листа на которой лежат объекты. Пля определния их числа используется цветовой фильтр. Далее идентификация объектов: 
loupe - объект с вложенным контуром, который определяется с помощью иерархии контуров. 
star - определяется из геометрических соображений - отношение площади звезды к площади описанной окружности - константа, допускается некоторая вариация константы.
Остальные объекты находятся с помощью sift.
to-do: 
- нахождение стикера с помощью бинаризации по оттенкам желтого.
- вместо UNKNOWN выбирать ближайший по площади объект.
3) Задача упаковки объектов в многоугольник
Планировался генетический алгоритм, но он требует перебора слишком большого количества конфигураций. Ведутся размышления по поводу разности Минковского и индекса Жаккара или методов нелинейной оптимизации

