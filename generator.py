import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma

        return weight_sn, u, sigma

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)
        module.register_buffer(name + '_sv', torch.ones(1).squeeze())

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u, sigma = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)
        setattr(module, self.name + '_sv', sigma)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)
    return module

def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    return spectral_norm(module)

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, condition_dim):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.linear1 = nn.Linear(condition_dim, in_channel)
        self.linear2 = nn.Linear(condition_dim, in_channel)

    def forward(self, input, condition):
        out = self.bn(input)
        gamma, beta = self.linear1(condition), self.linear2(condition)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, condition_dim=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        gain = 2 ** 0.5

        self.conv1 = spectral_init(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)
        self.conv2 = spectral_init(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_skip = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                     1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn

        if bn:
            self.norm1 = ConditionalNorm(in_channel, condition_dim)
            self.norm2 = ConditionalNorm(out_channel, condition_dim)

    def forward(self, input, condition=None, condition1=None):
        out = input

        if self.bn:
            out = self.norm1(out, condition)
        out = self.activation(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv1(out)
        if self.bn:
            out = self.norm2(out, condition)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='nearest')
            skip = self.conv_skip(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input

        return out + skip
    
class SelfAttention(nn.Module):
    def __init__(self, in_channel, embed_dim, gain=2 ** 0.5):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, embed_dim, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, embed_dim, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input): # [bsz, channel, freq, time]
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1) # [bsz, channel, freq*time]
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key) # [bsz, freq*time, freq*time]
        attention_map = F.softmax(query_key, 1)
        out = torch.bmm(value, attention_map)
        out = out.view(*shape)
        out = self.gamma * out + input

        return (out, attention_map)

class CrossAttention(nn.Module):
    def __init__(self, in_channel, cond_channel, embed_dim, gain=2 ** 0.5):
        super().__init__()

        self.key = spectral_init(nn.Conv1d(cond_channel, embed_dim, 1),
                                   gain=gain)
        self.value = spectral_init(nn.Conv1d(cond_channel, in_channel, 1),
                                   gain=gain)
        self.query = spectral_init(nn.Conv1d(in_channel, embed_dim, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input, condition, sequence_lengths=None):
        # input : mel [bsz, channel, freq, time] or sentence [bsz, channel]
        # condition : sentence [bsz, channel] or word [bsz, word_num, channel]
        input_shape = input.shape
        if len(input.shape) == 4: # mel [bsz, channel, freq, time]
            batch_size, c, w, h = input.shape
            num = w * h
            x = input.reshape([batch_size, c, num]) #[bsz, channel, input_num]
        elif len(input.shape) == 2: # sentence [bsz, channel]
            batch_size, c = input.shape
            num = 1
            x = input.unsqueeze(2) # [bsz, channel, input_num]

        if len(condition.shape) == 2: # sentence [bsz, channel]
            condition = condition.unsqueeze(2) # [bsz, channel, cond_num]
        else: # word [bsz, word_num, channel]
            condition = condition.permute(0, 2, 1) # [bsz, channel, cond_num]

        query = self.query(x).permute(0, 2, 1) # [bsz, input_num, channel]
        key = self.key(condition) # [bsz, channel, cond_num]
        value = self.value(condition).permute(0, 2, 1) # [bsz, cond_num, channel]
        attention_map = torch.bmm(query, key) # [bsz, input_num, cond_num]

        if sequence_lengths is not None: # condition is word embedding
            total_len = condition.shape[2]

            mask = torch.tile(torch.arange(total_len), [batch_size, num, 1]).to(condition.device)
            for i in range(batch_size):
                sequence_lengths_i = sequence_lengths[i]
                mask[i,:,:] = mask[i,:,:] >= sequence_lengths_i.item()
            attention_map = attention_map + mask * (-1e9)
        
        attention_map = F.softmax(attention_map, dim=-1) # [bsz, input_num, cond_num]
        out = torch.bmm(attention_map, value).permute(0, 2, 1) # [bsz, input_num, channel]
        out = out.permute(0, 2, 1).reshape(input_shape).squeeze()
        out = self.gamma * out + input

        return out, attention_map

class Spec_Attention(nn.Module):
    def __init__(self, in_channel, cond_channel=None, embed_dim=64, gain=2 ** 0.5):
        super().__init__()
        if cond_channel is None:
            cond_channel = in_channel

        self.f_query = spectral_init(nn.Conv1d(in_channel, embed_dim, 1),
                                   gain=gain)
        self.t_key = spectral_init(nn.Conv1d(cond_channel, embed_dim, 1),
                                   gain=gain)

        self.t_query = spectral_init(nn.Conv1d(in_channel, embed_dim, 1),
                                   gain=gain)
        self.f_key = spectral_init(nn.Conv1d(cond_channel, embed_dim, 1),
                                   gain=gain)

        self.value = spectral_init(nn.Conv1d(cond_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input, condition=None, sequence_lengths=None):
        # input : mel [bsz, channel, freq, time]
        # condition : sentence [bsz, channel] or word [bsz, word_num, channel]
        
        batch_size, c, f, t = input.shape

        freq_embedding = input.mean(dim=3) # [bsz, channel, freq]
        time_embedding = input.mean(dim=2) # [bsz, channel, time]

        if condition is not None:
            if len(condition.shape) == 2: # sentence [bsz, channel]
                condition = condition.unsqueeze(2) # [bsz, channel, 1]
            else: # word [bsz, word_num, channel]
                condition = condition.permute(0, 2, 1) # [bsz, channel, cond_num]
            t_condition = condition
            f_condition = condition
        else:
            t_condition = time_embedding
            f_condition = freq_embedding

        f_query = self.f_query(freq_embedding).permute(0, 2, 1) # [bsz, freq, channel]
        t_key = self.t_key(t_condition) # [bsz, channel, time] or [bsz, channel, cond_num]
        freq_cond_map = torch.bmm(f_query, t_key) # [bsz, freq, time] or [bsz, freq, cond_num]

        t_query = self.t_query(time_embedding).permute(0, 2, 1) # [bsz, time, channel]
        f_key = self.f_key(f_condition) # [bsz, channel, freq] or [bsz, channel, cond_num]
        time_cond_map = torch.bmm(t_query, f_key) # [bsz, time, freq] or [bsz, time, cond_num]

        if sequence_lengths is not None: # condition is word embedding
            total_len = condition.shape[2]

            mask = torch.arange(total_len, device=condition.device)[None, None, :]
            mask = mask >= sequence_lengths[:, None, None]

            freq_cond_map = freq_cond_map + mask * (-1e9)
            time_cond_map = time_cond_map + mask * (-1e9)

        freq_cond_map = F.softmax(freq_cond_map, dim=-1) # [bsz, freq, time] or [bsz, freq, cond_num]
        time_cond_map = F.softmax(time_cond_map, dim=-1) # [bsz, time, freq] or [bsz, time, cond_num]
        
        if condition is None:
            freq_time_embedding = input.reshape([batch_size, c, f*t]) # [bsz, channel, freq*time]
            weight_map = torch.add(freq_cond_map, time_cond_map.permute(0, 2, 1)).reshape([batch_size, f*t]).unsqueeze(-1) # [bsz, freq*time, 1]
            value = self.value(freq_time_embedding).permute(0, 2, 1) # [bsz, freq*time, channel]
            out = torch.mul(value, weight_map).permute(0, 2, 1).reshape(batch_size, c, f, t) # [bsz, channel, freq, time]
        else:
            freq_cond_map = torch.tile(freq_cond_map.unsqueeze(2), [1, 1, t, 1]) # [bsz, freq, time, cond_num]
            time_cond_map = torch.tile(time_cond_map.unsqueeze(1), [1, f, 1, 1]) # [bsz, freq, time, cond_num]
            weight_map = torch.add(freq_cond_map, time_cond_map).reshape([batch_size, f*t, -1]) # [bsz, freq*time, cond_num]
            value = self.value(condition).permute(0, 2, 1) # [bsz, cond_num, channel]
            out = torch.bmm(weight_map, value).permute(0, 2, 1).reshape(batch_size, c, f, t) # [bsz, channel, freq, time]

        out = self.gamma * out + input

        return out, weight_map

class Multi_Triple_Attention(nn.Module):
    def __init__(self, in_channel, sentence_embed_dim=768, word_embed_dim=768, embed_dim=64, reverse=False, gain=2 ** 0.5, n_heads=2, attention_list="self,word,sentence", spec_attention=False):
        super().__init__()
        self.reverse = reverse
        self.n_heads = n_heads
        self.attention_list = attention_list.split(",")
        
        if "self" in self.attention_list:
            if spec_attention:
                self.self_attention_modules = nn.ModuleList([Spec_Attention(in_channel, embed_dim=embed_dim) for _ in range(self.n_heads)])
            else:
                self.self_attention_modules = nn.ModuleList([SelfAttention(in_channel, embed_dim=embed_dim) for _ in range(self.n_heads)])
        
        if "word" in self.attention_list:
            if spec_attention:
                self.cross_attention_for_word_modules = nn.ModuleList([Spec_Attention(in_channel, cond_channel=word_embed_dim, embed_dim=embed_dim) for _ in range(self.n_heads)])
            else:
                self.cross_attention_for_word_modules = nn.ModuleList([CrossAttention(in_channel, cond_channel=word_embed_dim, embed_dim=embed_dim) for _ in range(self.n_heads)])
        
        if "sentence" in self.attention_list:
            if spec_attention:
                self.cross_attention_for_sent_modules = nn.ModuleList([Spec_Attention(in_channel, cond_channel=sentence_embed_dim, embed_dim=embed_dim) for _ in range(self.n_heads)])
            else:
                self.cross_attention_for_sent_modules = nn.ModuleList([CrossAttention(in_channel, cond_channel=sentence_embed_dim, embed_dim=embed_dim) for _ in range(self.n_heads)])
        
        self.gamma = [nn.Parameter(torch.tensor(0.0)) for _ in range(self.n_heads)]

        self.conv_for_attention = spectral_init(nn.Conv1d(in_channel * len(self.attention_list), in_channel, 1), gain=gain)
            
        self.out = spectral_init(nn.Conv1d(in_channel * self.n_heads, in_channel, 1), gain=gain)

    def forward(self, input, sentence_embedding, word_embedding, sequence_lengths):
        batch_size, c, f, t = input.shape
        x = input

        result = []
        for head in range(self.n_heads):
            out_list = []
            if "self" in self.attention_list:
                x_self, attention_map = self.self_attention_modules[head](x)
                out_list.append(x_self)
            if "word" in self.attention_list:
                x_word, attention_map = self.cross_attention_for_word_modules[head](x, word_embedding, sequence_lengths)
                out_list.append(x_word)
            if "sentence" in self.attention_list:
                x_sent, attention_map = self.cross_attention_for_sent_modules[head](x, sentence_embedding)
                out_list.append(x_sent)
            out = torch.cat(out_list, dim=1)
            out = self.conv_for_attention(out.reshape([batch_size, c*len(out_list), f*t])).reshape([batch_size, c, f, t])
            out = self.gamma[head] * out + x
            result.append(out)
        x = torch.cat(result, dim=1)
        x = self.out(x.reshape([batch_size, c * self.n_heads, f*t])).reshape([batch_size, c, f, t])

        x = input + x

        return x

class Generator(nn.Module):
    def __init__(self, model_config=None):
        super().__init__()

        if model_config is None:
            model_config = {
                "noise_dim":128,
                "g_chaneel":128,
                "n_heads":10,
                "sentence_embed_dim":512,
                "word_embed_dim":768,
                "attention_list":["self,word,sentence", "word,sentence", "sentence"],
                "spec_attention":True,
            }

        self.noise_dim = model_config['noise_dim']
        self.channel = model_config['g_chaneel']
        self.n_heads = model_config['n_heads']
        
        self.sentence_embed_dim = model_config['sentence_embed_dim']
        self.word_embed_dim = model_config['word_embed_dim']

        self.attention_list = model_config['attention_list']
        self.spec_attention = model_config['spec_attention']

        channel_list = [self.channel, self.channel, self.channel//2, self.channel//2, self.channel//4, self.channel//4, self.channel//4, self.channel//8, self.channel//8]

        self.lin_code = spectral_init(nn.Linear(self.noise_dim, channel_list[0] * 2 * 32))

        self.conv1 = ConvBlock(channel_list[0], channel_list[1], condition_dim=self.sentence_embed_dim)
        self.conv2 = ConvBlock(channel_list[1], channel_list[2], condition_dim=self.sentence_embed_dim)
        self.multi_triple_attention_1 = Multi_Triple_Attention(channel_list[2],
                                                   sentence_embed_dim=self.sentence_embed_dim,
                                                   word_embed_dim=self.word_embed_dim,
                                                   embed_dim=channel_list[2],
                                                   reverse=False,
                                                   n_heads=self.n_heads,
                                                   attention_list=self.attention_list[0],
                                                   spec_attention=self.spec_attention)

        self.conv3 = ConvBlock(channel_list[2], channel_list[3], condition_dim=self.sentence_embed_dim)
        self.conv4 = ConvBlock(channel_list[3], channel_list[4], condition_dim=self.sentence_embed_dim, upsample=False)
        self.multi_triple_attention_2 = Multi_Triple_Attention(channel_list[4],
                                                   sentence_embed_dim=self.sentence_embed_dim,
                                                   word_embed_dim=self.word_embed_dim,
                                                   embed_dim=channel_list[4],
                                                   reverse=False,
                                                   n_heads=self.n_heads,
                                                   attention_list=self.attention_list[1],
                                                   spec_attention=self.spec_attention)

        self.conv5 = ConvBlock(channel_list[4], channel_list[5], condition_dim=self.sentence_embed_dim)
        self.conv6 = ConvBlock(channel_list[5], channel_list[6], condition_dim=self.sentence_embed_dim, upsample=False)
        self.multi_triple_attention_3 = Multi_Triple_Attention(channel_list[6],
                                                   sentence_embed_dim=self.sentence_embed_dim,
                                                   word_embed_dim=self.word_embed_dim,
                                                   embed_dim=channel_list[6],
                                                   reverse=False,
                                                   n_heads=self.n_heads,
                                                   attention_list=self.attention_list[2],
                                                   spec_attention=self.spec_attention)

        self.conv7 = ConvBlock(channel_list[6], channel_list[7], condition_dim=self.sentence_embed_dim)
        self.bn = nn.BatchNorm2d(channel_list[8])
        self.colorize = spectral_init(nn.Conv1d(channel_list[8], 1, 1))

    def forward(self, z, sentence_embedding, word_embedding, sequence_lengths):
        batch_size = z.shape[0]

        x = self.lin_code(z)
        x = x.view(-1, self.channel, 2, 32) # [bsz, c, 2, 32]

        x = self.conv1(x, sentence_embedding) # [bsz, c, 4, 64]
        x = self.conv2(x, sentence_embedding) # [bsz, c, 8, 128]
        x = self.multi_triple_attention_1(x, sentence_embedding, word_embedding, sequence_lengths) # [bsz, c, 8, 128]

        x = self.conv3(x, sentence_embedding) # [bsz, c, 16, 256]
        x = self.conv4(x, sentence_embedding) # [bsz, c, 16, 256]
        x = self.multi_triple_attention_2(x, sentence_embedding, word_embedding, sequence_lengths) # [bsz, c, 16, 256]

        x = self.conv5(x, sentence_embedding) # [bsz, c, 32, 512]
        x = self.conv6(x, sentence_embedding) # [bsz, c, 32, 512]
        x = self.multi_triple_attention_3(x, sentence_embedding, word_embedding, sequence_lengths) # [bsz, c, 32, 512]

        x = self.conv7(x, sentence_embedding) # [bsz, c, 64, 1024]
        x = self.bn(x) # [bsz, c // 8, 64, 1024]
        x = F.relu(x)
        x = self.colorize(x.reshape([batch_size, -1, 64*1024])).reshape([batch_size, 1, 64, 1024]) # [bsz, 1, 64, 1024]

        return x
