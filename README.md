## Bert Question Answering
- Chinese Question Answering using Bert pretrained model
- Use BertForMultipleChoice to predict relevant context
- Then use BertForQuestionAnswering to do QA

## Download bert pretrained models and QA models
```shell
# models will be downloaded as ckpt/slot/best.pt and ckpt/intent/best.pt
bash download.sh
```

## Predict public.json / private.json
```shell
python predict.py --context_dir /context.json --data_dir /data.json --result_path /result.json
```

## Context Selection Training
```shell
python train_Choice_strong.py
```

## Question Answering Training
```shell
python train_QA_strong.py
```