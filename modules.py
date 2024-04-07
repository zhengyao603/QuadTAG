import torch
import torch.nn as nn


class Bilinear(nn.Module):
    def __init__(self, input_size1, input_size2, output_size, bias=True):
        super(Bilinear, self).__init__()

        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.output_size = output_size

        # weights -> input_size1 * input_size2 * output_size
        # bias -> output_size
        self.weights = nn.Parameter(torch.Tensor(input_size1, input_size2, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else None

        # init params
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, input1, input2):
        input_size1 = list(input1.size())
        input_size2 = list(input2.size())

        # matrix multiplication between input1 and weights
        intermediate = torch.mm(input1.view(-1, input_size1[-1]), self.weights.view(-1, self.input_size2 * self.output_size))

        input2 = input2.transpose(1, 2)
        output = intermediate.view(input_size1[0], input_size1[1] * self.output_size, input_size2[2]).bmm(input2)
        output = output.view(input_size1[0], input_size1[1], self.output_size, input_size2[1]).transpose(2, 3)

        # output -> [i, j, k, l] -> k-th dimension bilinear of i-th batch, j-th tensor of input1, l-th tensor of input2

        if self.bias is not None:
            output += self.bias

        return output


class Biaffine(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True, bias_init=None):
        super(Biaffine, self).__init__()

        self.trans_1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim, bias=True)])
        self.trans_2 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim, bias=True)])

        self.linear_1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.bilinear = Bilinear(hidden_dim, hidden_dim, output_dim, bias=bias)
        if bias_init is not None:
            self.bilinear.bias.data = bias_init

        self.trans_1.apply(self.initialize_weights)
        self.trans_2.apply(self.initialize_weights)
        self.linear_1.apply(self.initialize_weights)
        self.linear_2.apply(self.initialize_weights)

    def forward(self, x, y, sent_num=None):
        x, y = self.trans_1(x), self.trans_2(y)
        res = self.bilinear(x, y) + self.linear_1(x).unsqueeze(2) + self.linear_2(y).unsqueeze(1)
        if sent_num:
            return res[:, :sent_num, :sent_num]
        else:
            return res

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)


if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    biaffine = Biaffine(input_dim, hidden_dim, output_dim)

    batch_size = 2
    seq_len = 3
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, seq_len, input_dim)

    output = biaffine(x, y)
    print("Output shape:", output.shape)

    sent_num = 2
    output_sent_num = biaffine(x, y, sent_num=sent_num)
    print("Output shape with sent_num:", output_sent_num.shape)

    print("Output:")
    print(output)
    print("\nOutput with sent_num:")
    print(output_sent_num)