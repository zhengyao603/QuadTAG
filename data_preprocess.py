import json
import torch
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from collections import defaultdict


class QAMDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        self.data_dir = data_dir
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.max_sent_num = 32

        self.inputs = []
        self.targets = []
        self.tags = get_table_tags(self.data_path, self.max_sent_num)
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        source_ids = torch.LongTensor(self.inputs[item]["input_ids"])
        target_ids = self.targets[item]["input_ids"].squeeze()

        source_mask = torch.LongTensor(self.inputs[item]["attention_mask"])
        target_mask = self.targets[item]["attention_mask"].squeeze()

        sent_mask = torch.LongTensor(self.inputs[item]["sent_mask"])
        sent_first_mask = torch.LongTensor(self.inputs[item]["sent_first_mask"])

        return {"source_ids": source_ids, "source_mask": source_mask, "sent_mask": sent_mask, "tags": self.tags[item],
                "target_ids": target_ids, "target_mask": target_mask, 'target_sent': self.targets[item]["target_sent"],
                'sent_num': self.sent_nums[item], "sent_first_mask": sent_first_mask}

    def _build_examples(self):
        inputs, targets, self.sent_nums = get_transformed_io(self.data_path, self.data_dir)

        for i in range(len(inputs)):
            # inputs[i] -> document
            tokenized_input = {'input_ids': [], 'attention_mask': [], 'sent_mask': [], 'sent_first_mask': []}
            for j in range(len(inputs[i])):
                # inputs[i][j] -> sentence
                sent = ' '.join(inputs[i][j])
                # if it is last sentence
                if j == len(inputs[i]) - 1:
                    sent_input_ids = self.tokenizer.encode(sent)
                else:
                    sent_input_ids = self.tokenizer.encode(sent)[:-1]
                tokenized_input['input_ids'].extend(sent_input_ids)
                tokenized_input['sent_mask'].extend([0] * len(sent_input_ids))
                tokenized_input['sent_mask'][-1] = 1

                tokenized_input['sent_first_mask'].append(1)
                tokenized_input['sent_first_mask'].extend([0] * (len(sent_input_ids) - 1))

            tokenized_input['attention_mask'].extend([1] * len(tokenized_input['input_ids']))
            data_len = len(tokenized_input['input_ids'])
            assert len(tokenized_input['input_ids']) == len(tokenized_input['sent_mask']) == len(tokenized_input['attention_mask'])

            if data_len >= self.max_len:
                tokenized_input['input_ids'] = tokenized_input['input_ids'][:self.max_len]
                tokenized_input['attention_mask'] = tokenized_input['attention_mask'][:self.max_len]
                tokenized_input['sent_mask'] = tokenized_input['sent_mask'][:self.max_len]
                tokenized_input['sent_first_mask'] = tokenized_input['sent_first_mask'][:self.max_len]

            else:
                tokenized_input['input_ids'] += [self.tokenizer.pad_token_id] * (self.max_len - data_len)
                tokenized_input['attention_mask'] += [0] * (self.max_len - data_len)
                tokenized_input['sent_mask'] += [-1] * (self.max_len - data_len)
                tokenized_input['sent_first_mask'] += [0] * (self.max_len - data_len)

            target = targets[i]
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target['target_sent'] = targets[i]
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def get_transformed_io(data_path, max_sent_num):
    # inputs -> [[['<SS>', 'Topic', ':', 'Will', 'artificial', 'intelligence', 'replace', 'humans', '?', '<SE>'], xxx], xxx]
    # inputs -> every sentence of every document of every token

    # targets -> ['#2 supports the topic : #3 Explanation [SEP] #13 supports the topic : #10 Explanation | #11 Explanation', xxx]
    # Equation 2.

    inputs = list()
    targets = list()
    tokenizer = TweetTokenizer()

    with open(data_path[:-4] + ".json", 'r') as f:
        data = json.load(f)

    sent_nums = list()
    stance = {"1": "supports the topic", "-1": "is against the topic"}

    for example in data:
        sent_nums.append(len(example["sents"]) + 1)

        # sents -> [['<SS>', '#1', 'It', 'is', 'xxxx', '<SE>'], ['<SS>', '#2', 'xx', '<SE>'], xxx]
        sents = [['<SS>', f"#{idx + 1}:"] + tokenizer.tokenize(s) + [f"<SE>"] for idx, s in enumerate(example['sents'])]

        # label_dict -> {(12, '1'): [(9, 'Explanation'), (10, 'Explanation'), ...], ...}
        label_dict = defaultdict(list)
        for cid, eid, st, evi_type in example["labels"]:
            label_dict[(cid, st)].append((eid, evi_type))

        temp = list()
        for key, value in label_dict.items():
            assert len(value) == len(set(value))
            value = sorted(value, key=lambda x:x[0])
            temp.append(f"#{key[0] + 1} {stance[key[1]]} : " + " | ".join([f'#{evi + 1} {evi_ty}' for evi, evi_ty in value]))
        targets.append(' [SEP] '.join(temp))

        topic = f'Topic: {example["topic"][:-1] if example["topic"][-1] == "?" else example["topic"]} ? <SE>'
        inputs.append([['<SS>'] + tokenizer.tokenize(topic)] + sents)

    return inputs, targets, sent_nums


def get_table_tags(data_path, max_sent_num):
    # data_path[:-4] -> get rid of ".txt"

    # data -> [
    #           {'labels':[[1, 2, '1', 'Explanation'], [...]]
    #           'sents': ["sentence1", [...]]
    #           'doc_id': '5_2'
    #           'topic': 'Will artificial intelligence replace humans'},
    #           {....}
    #          ]

    # labels -> claim_id, evidence_id, stance, evidence_type

    with open(data_path[:-4] + ".json", 'r') as f:
        data = json.load(f)

    label_map = {'Research': 1, 'Expert': 2, 'Case': 3, 'Explanation': 4, 'Others': 5, '1': 6, "-1": 7}
    # Figure 3. table
    tags = torch.full((len(data), max_sent_num, max_sent_num), -1)

    for example_idx, example in enumerate(data):
        tags[example_idx, :len(example['sents']) + 1, :len(example['sents']) + 1] = 0
        for cid, eid, st, evi_type in example['labels']:
            tags[example_idx, cid + 1, eid + 1] = label_map[evi_type]
            tags[example_idx, cid + 1, 0] = label_map[st]

    return tags


if __name__ == "__main__":
    from transformers import T5Tokenizer

    data_type = "train"
    max_seq_length = 2496
    dataloader = QAMDataset(tokenizer=T5Tokenizer.from_pretrained("t5-base"), data_dir="QAM", data_type="train", max_len=max_seq_length)

    for data in dataloader:
        print(data)
