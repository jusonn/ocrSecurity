## Finetune detector

```python
python finetuning_detector.py --batch-size 4 --output model
```

## Finetune recognizer
```python
python finetuning_recognizer.py --batch-size 4 --output model
```

## OCR 데이터셋 변환
특수문자_맑은고딕_8_001.png
특수문자_맑은고딕_8_001.xml 
...

```python
python xml2txt.py
```