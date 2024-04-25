import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import json

from torch.nn import functional
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from modules import Biaffine
from data_preprocess import QAMDataset
from tqdm import tqdm


class T5FineTuner(nn.Module):
    def __init__(self, hparams, tfm_model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.biaffine_layer = Biaffine(self.hparams.embed_dim, self.hparams.hidden_embed_dim, self.hparams.table_label_num)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, output_hidden_states=False, sent_mask=None, sent_first_mask=None):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

    def _step(self, batch, device):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels.to(device),
            decoder_attention_mask=batch['target_mask'].to(device),
            output_hidden_states=True,
        )
        last_hidden_states = outputs.encoder_hidden_states[-1]

        sent_embeds, sent_token_masks = transform_doc_tokens_to_sent_tokens(last_hidden_states, batch['sent_mask'],
                                                                            self.hparams.max_sent_num,
                                                                            self.hparams.max_sent_len)
        updated_sent_embeds = sent_embeds[:, :, 0]
        sent_num = (sent_token_masks.sum(-1) != 0).sum(-1)

        truncated_dim = sent_num[0]
        table_logits = self.biaffine_layer(updated_sent_embeds, updated_sent_embeds, sent_num[0])

        logits_flatten = table_logits.reshape(-1, table_logits.size()[-1])
        tags_flatten = batch['tags'][:, :truncated_dim, :truncated_dim].reshape(-1, 1).to(device)

        assert (tags_flatten == -1).sum() == 0

        logits_mask = (tags_flatten != 0).bool().to(table_logits.device)
        neg_idx = (logits_mask == 0).nonzero(as_tuple=False)
        negative_samples_num = (tags_flatten > 0).sum() * args.negative_ratio
        if negative_samples_num > neg_idx.size(0):
            negative_samples_num = torch.tensor(neg_idx.size(0))

        choice = torch.LongTensor(np.random.choice(neg_idx.size(0), negative_samples_num.item(), replace=False)).to(
            table_logits.device)
        logits_mask[neg_idx[choice][:, 0], neg_idx[choice][:, 1]] = True

        logits_flatten = torch.masked_select(logits_flatten, logits_mask).reshape(-1, self.hparams.table_label_num)
        tags_flatten = torch.masked_select(tags_flatten, logits_mask).reshape(-1)

        table_loss = functional.cross_entropy(logits_flatten, tags_flatten, ignore_index=-1, reduction='mean')

        loss = outputs[0] + table_loss

        return loss


def train(model, train_dataloader, val_dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # configure optimizer and scheduler
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = ((len(train_dataloader.dataset) // (args.train_batch_size * max(1, args.n_gpu)))
               // args.gradient_accumulation_steps
               * float(args.num_train_epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # training step
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0
        batch_count = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model._step(batch, device)
            train_loss += loss.item()
            print(f"Training Loss: {loss}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_count % 200 == 0:
                # save the current model for batches
                model.model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                torch.save(model.state_dict(), f'{args.output_dir}/model.pt')
                print("Epoch " + str((epoch) + 1) + ": Finish batches and saving the model!", args.output_dir)
            batch_count += 1

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Training Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                val_loss += model._step(batch, device).item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Val Loss: {val_loss:.4f}")

        # save the current model
        model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(model.state_dict(), f'{args.output_dir}/model.pt')
        print("Finish training epoch" + str((epoch) + 1) + " and saving the model!", args.output_dir)

    return model


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='QAM', type=str, required=False,
                        help="The name of the dataset, the default is [QAM]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_dev", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--n_gpu", default=0, type=int,
                        help="GPU device")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU for evaluation.")
    parser.add_argument("--max_seq_length", default=1280, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir", default=f"./checkpoints/", type=str)
    parser.add_argument("--result_dir", default=f"./results/", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--negative_ratio", default=5, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--max_generate_len", default=512, type=int)
    parser.add_argument("--max_sent_num", default=32, type=int)
    parser.add_argument("--max_sent_len", default=400, type=int)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--hidden_embed_dim", default=256, type=int)
    parser.add_argument("--table_label_num", default=8, type=int)
    parser.add_argument('--seed', type=int, default=68,
                        help="random seed for initialization")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    return args


def transform_doc_tokens_to_sent_tokens(hidden_states, sent_mask, max_sent_num, max_sent_len):
    bsz, _, embed_dim = hidden_states.size()
    sent_embeds = torch.zeros(bsz, max_sent_num, max_sent_len, embed_dim).to(hidden_states.device)
    sent_attn_masks = torch.zeros(bsz, max_sent_num, max_sent_len).to(hidden_states.device)
    for i in range(bsz):
        st = 0
        sent_id = 0
        for idx, ind in enumerate(sent_mask[i]):
            if ind == 1:
                sent_embeds[i][sent_id][:idx+1-st] = hidden_states[i][st:idx+1]
                sent_attn_masks[i][sent_id][:idx+1-st] = 1
                st = idx + 1
                sent_id += 1

    return sent_embeds, sent_attn_masks


def evaluate(data_loader, model, mode=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.model.eval()

    outputs, targets = [], []
    for batch in tqdm(data_loader):
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=args.max_generate_len,
                                    )

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(dec)
        targets.extend(target)

    # return scores
    print("saving generation results...")
    with open(f"./results/{mode}[pred]{args.output_dir.split('/')[-1]}.txt", 'w') as fw:
        json.dump(outputs, fw)
    with open(f"./results/{mode}[gold]{args.output_dir.split('/')[-1]}.txt", 'w') as fw:
        json.dump(targets, fw)


if __name__ == "__main__":
    # arguments initialization
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP on {args.dataset}", "=" * 30, "\n")

    # tokenizer initialization
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    special_tokens = ['<SS>', '<SE>', '[SEP]']
    tokenizer.add_tokens(special_tokens)

    # example
    print(f"\nHere is an example (from the dev set):")
    dataset = QAMDataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev', max_len=args.max_seq_length)
    data_sample = dataset[7]
    example_input = tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True)
    example_output = tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True)
    print('\n-> Input :', example_input)
    print('')
    print('-> Output:', example_output)
    print('')
    print('-> dataset:  ', args.dataset)

    # example
    if args.do_train:
        with open(f"./io_example/[example]{args.output_dir.split('/')[-1]}.txt", 'w') as fw:
            fw.write(example_input + '\n')
            fw.write(example_output + '\n')
            fw.write("\t".join(special_tokens))
    print('\n')
    print("output model dir", args.output_dir)

    # training
    if args.do_train:
        torch.manual_seed(args.seed)
        print("\n****** Conduct Training ******")
        # initialize the T5 model
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer)
        model.model.resize_token_embeddings(len(tokenizer))

        if args.model_name_or_path != 't5-base':
            print("Reload other model paramters")
            model.load_state_dict(torch.load(f'{args.output_dir}/model.pt'))

            print("Reload pretrained model")
            model.model.from_pretrained(args.output_dir)
            model.tokenizer.from_pretrained(args.output_dir)

        # initialize dataloader
        train_dataset = QAMDataset(tokenizer, "QAM", "train", args.max_seq_length)
        val_dataset = QAMDataset(tokenizer, "QAM", "dev", args.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size)

        # train model
        trained_model = train(model, train_dataloader, val_dataloader, args)

        # save the final model
        model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Finish training and saving the model!", args.output_dir)
        torch.save(model.state_dict(), f'{args.output_dir}/model.pt')

    # evaluation
    if args.do_test:
        print("\n****** Conduct Evaluating with the last state ******")

        print(f"Load trained model from {args.output_dir}")
        print('Note that a pretrained model is required and `do_true` should be False')

        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

        model = T5FineTuner(args, tfm_model, tokenizer)
        print("Reload other model parameters")
        model.load_state_dict(torch.load(f'{args.output_dir}/model.pt'))

        print("Reload pretrained model")
        model.model.from_pretrained(args.output_dir)
        model.tokenizer.from_pretrained(args.output_dir)

        test_dataset = QAMDataset(tokenizer, data_dir=args.dataset, data_type='test', max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False)

        scores = evaluate(test_loader, model)

    # evaluation
    if args.do_dev:
        print("\n****** Conduct Evaluating with the last state ******")

        print(f"Load trained model from {args.output_dir}")
        print('Note that a pretrained model is required and `do_true` should be False')

        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

        model = T5FineTuner(args, tfm_model, tokenizer)
        print("Reload other model parameters")
        model.load_state_dict(torch.load(f'{args.output_dir}/model.pt'))

        print("Reload pretrained model")
        model.model.from_pretrained(args.output_dir)
        model.tokenizer.from_pretrained(args.output_dir)

        test_dataset = QAMDataset(tokenizer, data_dir=args.dataset, data_type='dev', max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False)

        scores = evaluate(test_loader, model, 'dev')
