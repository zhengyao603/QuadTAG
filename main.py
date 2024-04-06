import argparse
import os

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='QAM', type=str, required=True,
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
    parser.add_argument("--max_seq_length", default=2496, type=int)
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