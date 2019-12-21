# Deep fashion deep learning sequential classification

## Install requirements
```bash
pip install -r requirements.txt
```

## Preparing dataset
0. Make csv fie by following form  
| image_file_name | color | style | part | season | category |  
|----------------|-------|--------|--------|--------|------|
| jean_0001.jpg	 | blue  | casual | bottom | winter | jean |  
| jean_0002.jpg	 | white | casual | bottom | winter | jean |  
...
0. Make directory like this and add file  
```
dataset/
    images/
        jean_0001.jpg
        jean_0002.jpg
        ...
        vest_0049.jpg
    labels/
        labels.csv
```
0. Run split_csv.py to make label for each classification category
```bash
python split_scv.py
```

## Train model
You need two terminal

#### terminal 1  
To see the loss plot you need to run visdom
```bash
python -m visdom.server
```

#### terminal 2
You can choose backbone network  

##### supporting backbone network
0. mobilenet_v2
0. vgg11_bn
```bash
python main.py --backbone mobilenet_v2
```
