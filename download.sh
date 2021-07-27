python3.8 -c "from transformers import BertConfig, BertTokenizerFast, BertForQuestionAnswering, BertForMultipleChoice;
pretrain_mc = 'hfl/chinese-roberta-wwm-ext';pretrain_qa='hfl/chinese-roberta-wwm-ext-large';
tokenizer_mc = BertTokenizerFast.from_pretrained(pretrain_mc);config_mc=BertConfig.from_pretrained(pretrain_mc);
model_mc=BertForMultipleChoice.from_pretrained(pretrain_mc);tokenizer_qa = BertTokenizerFast.from_pretrained(pretrain_qa);
config_qa= BertConfig.from_pretrained(pretrain_qa);model_qa=BertForQuestionAnswering.from_pretrained(pretrain_qa)"

wget https://www.dropbox.com/s/cpxhgtx8qmnov7w/Choice_strong.pt?dl=1 -O ckpt/choice.pt
wget https://www.dropbox.com/s/vouexdtmbs3fjt1/QA_strong.pt?dl=1 -O ckpt/qa.pt