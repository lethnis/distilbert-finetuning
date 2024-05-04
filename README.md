# Дообучение модели distilbert для классификации новостей
[Датасет](https://www.kaggle.com/datasets/kishanyadav/inshort-news) состоит из коротких новостей семи разных категорий.  
![image](https://github.com/lethnis/distilbert-finetuning/assets/88483002/6db6ce2b-d339-4dcb-ac55-32eea7681201)  

Для обучения использовалась модель [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).  
Было проведено 3 эксперимента с разными обучаемыми слоями.  
1. Обучался только финальный классификатор.
2. Обучался классификатор и пре-классификатор.
3. Обучалась вся модель.  
![image](https://github.com/lethnis/distilbert-finetuning/assets/88483002/3214526c-688e-48bd-b347-159efd1bd2d3)  

# Результаты обучения
![image](https://github.com/lethnis/distilbert-finetuning/assets/88483002/3db29f99-85d3-4286-85d2-742fed6483f2)  
Полностью обучаемая модель достигла лучших результатов - 96%.  
Распределение предсказаний модели на проверочных данных.  
![image](https://github.com/lethnis/distilbert-finetuning/assets/88483002/f3c28b7a-7410-41da-9851-904d0b9a690f)  

# Проверка работы модели на новых данных
Я скопировал несколько новостей из разных источников и проверил как модель предсказывает их.  
![image](https://github.com/lethnis/distilbert-finetuning/assets/88483002/6e4645ed-3d6a-47f1-9a6f-5fe2edbaab91)  

