import torch
import torch.nn as nn

class QA_RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, model="lstm", n_layers=1):
        super(QA_RNN, self).__init__()
        self.model = model.lower()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.n_layers = n_layers

        # TODO-Explain, why we need to separate encoder and decoder
        # Find the encoder-decoder architecture in lecture note
        # Answer:
        # Encoder: To encode the question state to hidden state
        # Decoder: To decode the hidden state to answer state
        # TODO: Draw the task computational graph of our word-based question-answering
        self.embed = nn.Embedding(vocab_size, hidden_size) # 词嵌入层
        if self.model == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
            self.decoder = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
        elif self.model == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
            self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_question, target_answer, hidden):
        batch_size = input_question.size(0)
        assert batch_size == 1
        question_embed = self.embed(input_question)  # [B=1, Question_len, hidden_size]
        answer_embed = self.embed(target_answer)  # [B=1, Answer_len, hidden_size]
        output_enc, hidden_enc = self.encoder(question_embed, hidden)
        output_dec, hidden_dec = self.decoder(answer_embed, hidden_enc)

        pred_word_dec = self.fc(output_dec)
        return pred_word_dec

    def generate(self, input_question, answer_init_token, pos_end_token, max_predict_len, device):

        hidden = self.init_hidden(1, device)
        
        # 先把问题进行词嵌入，得到词嵌入后的向量并增加一个维度
        question_embed = self.embed(input_question)  # [B=1, Question_len, hidden_size]
        question_embed = question_embed.unsqueeze(0)

        # 将答案预处理，假如 [BEG] 并进行词嵌入
        # In inference, the input is [BEG] only, so its size is [1,1,D]
        answer_init_embed = self.embed(answer_init_token)
        answer_init_embed = answer_init_embed.unsqueeze(0)
        ########################################
        #### Stage-1: Encoding Query String  ###
        ########################################
        # print(question_embed.device, hidden[0].device)
        # 把问题的嵌入和隐藏状态传入编码器
        output_enc, hidden_enc = self.encoder(question_embed, hidden)

        #######################################
        #### Stage-2 decoding as response  ####
        #######################################
        # TODO: Explain stage-2 as answer generation
        # draw a pipeline as in lecture note about inference-stage
        # draw encoder and decoder
        # note that we pass encoder hidden state to decoder as init hidden state
        # 注意：我们需要把编码器的隐藏状态传给解码器，作为解码器的初始隐藏状态。
        # note that we should start from BEG token as first input (x0) to decoder
        # 注意：我们需要从 [BEG] token 开始，作为解码器的第一个输入。
        token_cur = answer_init_embed   # 从 [BEG] token 开始
        hidden_cur = hidden_enc # 把编码器的隐藏状态传给解码器，作为解码器的初始隐藏状态。
        pred_tokens = []
        for p in range(max_predict_len):
            feat, hidden_cur = self.decoder(token_cur, hidden_cur) # 解码器解码
            pred_token = self.fc(feat)  # 预测下一个 token

            # The predicted word is from argmax the output probability of each word in vocab
            # 从预测的概率分布中找到概率最大的 token
            top_i = torch.argmax(pred_token, dim=2)
            token_cur = self.embed(top_i) # 把预测的 token 嵌入

            word_index = top_i.item() # 获取 token 的索引
            pred_tokens.append(word_index) # 把 token 加入到预测的 token 列表中
            if word_index == pos_end_token:
                break

        return pred_tokens

    def init_hidden(self, batch_size, device):
        if self.model == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))
        # GRU
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
