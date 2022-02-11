# Intelligent Placer

## Постановка задачи
### Общее
Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. При расположени объектов в многоугольнике расстояние между объектами должно быть больше или равно 1px. 

### Требования 
#### Фотометрические
- Фотографии сделаны с одного устройства
- Фотогравировать предметы видом сверху (параллельно поверхности), допускается угол наклона камеры в меньше 3 градусов
- При одинаковом равномерном освещении
- на фотографии отсутствуют пересвеченные и серо-черные области
- Без цветовой коррекции

#### По расположению объектов на фотографии
- На фотографии не должно быть лишних предметов, кроме тех, которые заранее были известны
- Предметы не могут перекрывать друг друга, короме loupe и star (их взаимодействие написано ниже)
> Внутрь круга loupe можно вложить другой объект в одном экземпляре (из них только star помещается)
- Границы объектов должны четко выделяться на фоне 
- Объекты должны помещаться целиком
- Один объект может присутствовать на  одной фотографии ровно один раз, кроме star (их максимум может быть три штуки)

#### Фон
- белый лист бумаги размером около А4, допускается погрешность 1 см по каждой оси листа, полностью или больй частью помешающийся на фотографии. Ориентация листа не важна
- Один для всех фотографий
- В пространстве находится в горизонтальном положении
- на одной стороне листа изображён многоугольник, на другой лежат предметы

#### Многоугольник
- Многоугольник задается фигурой, нарисованной темным маркером на белом листе бумаги, сфотографированной вместе с предметами.

### Датасет с объектами
Фотографии объектов: https://drive.google.com/drive/folders/1rrVlvPRfrt-whINtHcryGTQC_jkgdkBm?usp=sharing

## Сбор данных
На гугл диске фотографии и таблица с отвеами и комменариями про особенные случаи: https://drive.google.com/drive/folders/1FsFPt8-jVN2nXhVPMfr8zcz-5BrAyhLl?usp=sharing
