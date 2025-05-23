import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModel import BasicModel

class REMIuseremb(BasicModel):
    
    def __init__(self, item_num, user_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0, args=None, device=None):
        super(REMIuseremb, self).__init__(item_num, user_num, hidden_size, batch_size, seq_len, beta)
        self.interest_num = interest_num
        self.num_heads = interest_num
        self.interest_num = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        self.seq_len = seq_len
        self.alpha = args.alpha if args is not None else 0.5
#        print(self.hidden_size)
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.reset_parameters()

    def get_embeddings(self, item_list, user_list):
        """获取物品和用户的嵌入"""
        item_eb = self.item_embeddings(item_list)    # 物品嵌入
        user_eb = self.user_embeddings(user_list)    # 用户嵌入
        return item_eb, user_eb

    def forwardLogits(self, item_eb, mask):
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))
        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
            # item_eb_add_pos = item_eb + self.position_embedding[:, -1, :].repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()


        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0

        # print(item_eb.size(), item_att_w.size(), atten_mask.size(), mask.size())

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # 矩阵A，shape=(batch_size, num_heads, maxlen)
        return item_att_w

    def forward(self, item_list, user_list, label_list, mask, times, device, train=True):
        item_eb, user_eb = self.get_embeddings(item_list, user_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.item_embeddings(label_list)

        # 历史物品嵌入序列，shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w  = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous()
        user_hidden = user_eb.repeat(1, 4)  # [batch_size, hidden_size*4]
        # 计算用户注意力权重
        user_att_w = torch.matmul(item_hidden, user_hidden.unsqueeze(2))  # [batch_size, seq_len, 1]
        user_att_w = user_att_w.transpose(2,1)  # [batch_size, 1, seq_len]
        user_att_w = user_att_w.repeat(1, self.num_heads, 1)  # [batch_size, num_heads, seq_len]
        

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1) # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        user_att_w = torch.where(torch.eq(atten_mask, 0), paddings, user_att_w)

        item_att_w = F.softmax(item_att_w, dim=-1) # 矩阵A，shape=(batch_size, num_heads, maxlen)
        user_att_w = F.softmax(user_att_w, dim=-1) # 矩阵A，shape=(batch_size, num_heads, maxlen)

        # 融合两种注意力
        att_w = (1 - self.alpha) * item_att_w + self.alpha * user_att_w

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(att_w, # shape=(batch_size, num_heads, maxlen)
                                item_eb # shape=(batch_size, maxlen, embedding_dim)
                                ) # shape=(batch_size, num_heads, embedding_dim)

        # 用户多兴趣向量
        user_eb = interest_emb # shape=(batch_size, num_heads, embedding_dim)

        if not train:
            return user_eb, item_att_w

        readout, selection = self.read_out(user_eb, label_eb)
        scores = None if self.is_sampler else self.calculate_score(readout)

        return user_eb, scores, att_w, readout, selection
    
    def calculate_atten_loss(self, attention):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        # C_reg = C_reg.matmul(C_reg.transpose(1,2)) / self.hidden_size
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.hidden_size
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return n2.sum()
    
    def read_little_out(self, user_eb, label_eb):
        print(user_eb.shape)
        print(label_eb.shape)

        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        atten = torch.matmul(user_eb, # shape=(batch_size, interest_num, hidden_size)
                        torch.reshape(label_eb, (-1, self.hidden_size, 1)) # shape=(batch_size, hidden_size, 1)
                        ) # shape=(batch_size, interest_num, 1)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1), dim=-1) # shape=(batch_size, interest_num)

        if self.hard_readout: # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                        (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0], device=user_eb.device) * self.interest_num).long()]
        else: # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), # shape=(batch_size, 1, interest_num)
                                user_eb # shape=(batch_size, interest_num, hidden_size)
                                ) # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size)) # shape=(batch_size, hidden_size)
        # readout是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
        selection = torch.argmax(atten, dim=-1)
        return readout, selection