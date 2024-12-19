from collections import OrderedDict
from typing import Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from . import clip
from .clip.model import LayerNorm, QuickGELU
from .torch_utils import activation
from torch.nn import functional as F

class Adapter(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0):
        super().__init__()
        self.temporal_modeling_type = 'expand_temporal_view_step2'
        
        self.attn = activation.MultiheadAttention(d_model, n_head, temporal_shift=self.temporal_modeling_type, T=T)
        self.ln1 = LayerNorm(d_model)
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        
    def forward(self, x):
        x = self.attention(self.ln1(x)) + x
        ln_x = self.ln2(x)      
        output = self.mlp1(ln_x)
        
        x = x + output
        return x

class AdapterAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, temporal_modeling_type='expand_temporal_view', adapter=False, need_mask=False):
        super().__init__()
        self.T = T
        self.has_adapter =adapter
        self.need_mask = need_mask
        self.n_head = n_head
        
        # type: channel_shift or expand_temporal_view
        if self.has_adapter:
            self.adapter = Adapter(d_model, n_head, attn_mask,T=self.T)
            self.adapter1 = Adapter(d_model, n_head, attn_mask,T=self.T)

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor=None, mask: torch.Tensor=None):
        if not self.need_mask:
            if self.has_adapter:
                if y is None:
                    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
                    return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]
                else:
                    # mask = mask.to(dtype=x.dtype, device=x.device)
                    return self.attn(y, x, x, need_weights=True, attn_mask=mask)[0]
            else:
                self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
                return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]
        else:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def myforward(self, x, mask=None):   
        l, bt, d = x.size()
        b = bt // self.T

        n=mask.shape[1]
        if not self.need_mask:
            if self.has_adapter:
                x12 = self.attention(self.ln_1(x)) + x
                x12 = x12 + self.mlp(self.ln_2(x12))
                x_temp : torch.Tensor = self.adapter(x12)#（l+1,bt,c)
                
                x_temp2 : torch.Tensor = self.adapter1(x12)#（l,bt,c)
                
                x_temp =x_temp.permute(1,0,2)#(bt,l+1,c)
                
                x_temp_cls=x_temp[:,0,:].unsqueeze(1) #(bt,1,c)
                x_temp = x_temp[:,1:,:]#bt,l,c
                mask1=mask.permute(0,2,1)#bt,l,n
                # mask1=mask1.contiguous().view(-1,n)
                
                mask1_cls=mask1.sum(1)>0#bt,n
                x_temp_cls= x_temp_cls.expand(-1,n,-1)#bt,n,c
                x_temp_cls = x_temp_cls.permute(1,0,2)
                mask1_cls=mask1_cls.permute(1,0)
                mask_cls_num = mask1_cls.sum(1)#n
                x_temp_cls= x_temp_cls[mask1_cls]
                x_temp_cls = x_temp_cls.split(mask_cls_num.tolist(),dim=0)
                idx = torch.where(mask1.contiguous().view(bt*(l-1),-1).permute(1,0)>0)[1]
                x_temp=x_temp.contiguous().view(-1,d)#(bt*(l-1),c)
                x_1 = x_temp.index_select(dim=0,index=idx)
                idx2 = mask1.contiguous().view(bt*(l-1),-1).sum(0).long()
                idx_mask = idx2>0
                idx2 = idx2[idx_mask]
                x_2 = x_1.split(idx2.tolist(),dim=0)
                feat = []
                for list_cls,list in zip(x_temp_cls,x_2):
                    feat.append(torch.cat([list_cls,list],dim=0).mean(0,keepdim=True))
                
                feat= torch.cat(feat)
                feat=feat.unsqueeze(0).expand(bt,-1,-1)#(bt,n,c)
                y=feat.permute(1,0,2)#(n,bt,c)
                mask = mask[:,idx_mask,:]
                mask_temp = torch.cat([torch.ones(mask.shape[0], mask.shape[1],1, dtype=mask.dtype, device=x.device), mask], dim=-1)#(bt,n,l)
                mask=mask_temp.unsqueeze(1).expand(bt,self.n_head,-1,-1)#(bt,nhead,n,l)
                mask=1*mask.reshape(bt*self.n_head,-1, l).float() #(bt*nhead,n,l)
                
                x_temp1=self.attention(self.ln_1(x),self.ln_1(y), mask)+y
                x_temp1 = self.mlp(self.ln_2(x_temp1)) + x_temp1
                
                y = y.permute(1,0,2).contiguous().view(b,-1,d)#（b,tn,c)
                mask= mask_temp.permute(0,2,1).sum(1).view(b,-1)#(bt,n,l)->(bt,l,n)->(bt,n)->(b,tn)
                mask=(mask>1)
                num = mask.sum(-1)#b
                num = num[num>0]
                y = y[mask,:]
                y = y.split(num.long().tolist(),dim=0)
                new_y_temp= []
                for y_1 in y:
                    new_y_temp.append(y_1.mean(0,keepdim=True))
                y = torch.cat(new_y_temp,dim=0).view(b,1,1,-1).repeat(1,self.T,1,1).view(-1,1,d).permute(1,0,2)
                            
                x = x_temp1
            else:
                x = self.attention(self.ln_1(x)) + x
                x = x + self.mlp(self.ln_2(x))
            return [x, y, x_temp2]
        else:
            x_temp, atten_mask = self.attention(self.ln_1(x))
            x = x + x_temp
            x = x + self.mlp(self.ln_2(x))
            return x, atten_mask
        
    def forward(self, x1):
        
        if isinstance(x1, list):
            x,mask = x1
        if not self.has_adapter:
            l, bt, d = x.size()
            b = bt // self.T
            if not self.need_mask:
                x = self.attention(self.ln_1(x),mask=mask) + x
                x = x + self.mlp(self.ln_2(x))
                return [x,mask]
            else:
                x_temp, atten_mask = self.attention(self.ln_1(x),mask=mask)
                x = x + x_temp
                x = x + self.mlp(self.ln_2(x))
                return x, atten_mask
        else:
            return self.myforward(x, mask)
        

# TYPE 1: expand temporal attention view
class TimesAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, temporal_modeling_type='expand_temporal_view',need_mask=False):
        super().__init__()
        self.T = T
        
        # type: channel_shift or expand_temporal_view
        self.attn = activation.MultiheadAttention(d_model, n_head, temporal_shift=temporal_modeling_type, T=T)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.need_mask = need_mask
        
    def attention(self, x: torch.Tensor, mask: torch.Tensor):
        if not self.need_mask:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        else:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)


    def forward(self, x, mask):
        l, bt, d = x.size()
        b = bt // self.T
        # x = x.view(l, b, self.T, d)
        if not self.need_mask:
            x = x + self.attention(self.ln_1(x), mask)
            x = x + self.mlp(self.ln_2(x))
            return x
        else:
            x_temp, atten_mask = self.attention(self.ln_1(x),mask)
            x = x + x_temp
            x = x + self.mlp(self.ln_2(x))
            return x, atten_mask        

# Type 2: temporal shift, same as space-time mixing paper
class ChannelShiftAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, ):
        super().__init__()
        self.T = T
        
        # type: channel_shift or expand_temporal_view
        self.attn = activation.MultiheadAttention(d_model, n_head, temporal_shift='channel_shift')
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
         
        return x

# TYPE 3: additional parameter, do temporal cls tokens attention
class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, num_experts=0, record_routing=False):
        super().__init__()
        self.T = T
        
        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)

        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.num_experts = num_experts
        self.record_routing = record_routing
        if num_experts > 0:
            self.experts_head = nn.Sequential(*[nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
            ])) for _ in range(num_experts)])
            
            self.experts_tail = nn.Sequential(*[nn.Sequential(OrderedDict([
                ("c_proj", nn.Linear(d_model * 4, d_model))
                ])) for _ in range(num_experts)])

            self.routing1 = nn.Linear(d_model, self.num_experts + 1)
            self.routing2 = nn.Linear(d_model*4, self.num_experts + 1)

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        if type(x) == tuple:
            x, routing_state = x
        else:
            routing_state = None

        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d)

        msg_token = self.message_fc(x[0,:,:,:])
        msg_token = msg_token.view(b, self.T, 1, d)

        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d)
        msg_token = msg_token + self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0]
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)

        x = torch.cat([x, msg_token], dim=0)

        x = x.view(l+1, -1, d)
        x = x + self.attention(self.ln_1(x))
        x = x[:l,:,:]

        ln_x = self.ln_2(x)
        # x = x + self.mlp(self.ln_2(x))
        if self.num_experts > 0:
            output_head = [self.mlp[0](ln_x)]
            [output_head.append(self.experts_head[i][0](ln_x)) for i in range(self.num_experts)]
            rout1 = torch.nn.functional.softmax(self.routing1(ln_x), -1).unsqueeze(-1)
            output_head = torch.stack(output_head, 0).permute(1,2,0,3)
            output_head = (rout1 * output_head).sum(-2)
            output_head = self.mlp[1](output_head)

            output = [self.mlp[2](output_head)]
            [output.append(self.experts_tail[i](output_head)) for i in range(self.num_experts)]
            rout2 = torch.nn.functional.softmax(self.routing2(output_head), -1).unsqueeze(-1)
            output = torch.stack(output, 0).permute(1,2,0,3)
            output = (rout2 * output).sum(-2)
        else:
            output = self.mlp(ln_x)
        
        x = x + output
        
        if self.record_routing:
            if self.num_experts > 0:
                current_rout = torch.stack([rout1.squeeze(-1), rout2.squeeze(-1)], 0)
                if routing_state == None:
                    routing_state = current_rout
                else:
                    routing_state = torch.cat([routing_state, current_rout], 0)

            return x, routing_state

        return x

# TYPE 4: additional parameter, STadapter
class STAdaptAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, ):
        super().__init__()
        self.T = T
         
        self.stadapt_down_1 = nn.Conv3d(d_model, d_model//2, kernel_size=(1,1,1),
                padding='same', groups=1,)
        self.stadapt_up_1 = nn.Conv3d(d_model//2, d_model, kernel_size=(1,1,1),
                padding='same', groups=1,)
        self.stadapt_conv3d_1 = nn.Conv3d(d_model//2, d_model//2, kernel_size=(3,1,1),
                padding='same', groups=d_model//2,)
        
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        
        x = x.view(l, b, self.T, d)
        cls_token = x[0:1,:,:,:]
        x_img = x[1:, :, :, :]
        x_raw = x_img
        # x_img (l, b, T, d) -> (b, d, T, h, w) 
        h = int(torch.tensor(x_img.shape[0]).sqrt())
        w = int(torch.tensor(x_img.shape[0]).sqrt())
        assert h*w == x_img.shape[0]
        x_img = x_img.permute(1, 3, 2, 0).view(b, d, self.T, h, w)
        # down - conv3d - up
        
        x_img = self.stadapt_up_1(self.stadapt_conv3d_1(self.stadapt_down_1(x_img)))
        x_img = x_img.view(b,d,self.T,h*w).permute(3, 0, 2, 1)
        x_img = x_raw + x_img
        
        x = torch.cat([cls_token, x_img], dim=0)
         
        x = x.view(l, -1, d)
        x = x + self.attention(self.ln_1(x))
        x = x[:l,:,:]
        x = x + self.mlp(self.ln_2(x))
        return x

# TYPE 5: additional parameter, STadapter, with zero initialization
class STAdaptZeroInitAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, num_experts=0, record_routing=False):
        super().__init__()
        self.T = T
         
        self.stadapt_down_1 = nn.Conv3d(d_model, d_model//2, kernel_size=(1,1,1),
                padding='same', groups=1,)
        self.stadapt_up_1 = nn.Conv3d(d_model//2, d_model, kernel_size=(1,1,1),
                padding='same', groups=1,)
        self.stadapt_conv3d_1 = nn.Conv3d(d_model//2, d_model//2, kernel_size=(3,1,1),
                padding='same', groups=d_model//2,)
        
        nn.init.constant_(self.stadapt_up_1.weight, 0)
        nn.init.constant_(self.stadapt_up_1.bias, 0)
 
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.num_experts = num_experts
        self.record_routing = record_routing
        if num_experts > 0:
            self.experts_head = nn.Sequential(*[nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
            ])) for _ in range(num_experts)])

            self.experts_tail = nn.Sequential(*[nn.Sequential(OrderedDict([
                ("c_proj", nn.Linear(d_model * 4, d_model))
                ])) for _ in range(num_experts)])
            
            self.routing1 = nn.Linear(d_model, self.num_experts + 1)
            self.routing2 = nn.Linear(d_model*4, self.num_experts + 1)

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        if type(x) == tuple:
            x, routing_state = x
        else:
            routing_state = None

        l, bt, d = x.size()
        b = bt // self.T
        
        x = x.view(l, b, self.T, d)
        cls_token = x[0:1,:,:,:]
        x_img = x[1:, :, :, :]
        x_raw = x_img
        # x_img (l, b, T, d) -> (b, d, T, h, w) 
        h = int(torch.tensor(x_img.shape[0]).sqrt())
        w = int(torch.tensor(x_img.shape[0]).sqrt())
        assert h*w == x_img.shape[0]
        x_img = x_img.permute(1, 3, 2, 0).view(b, d, self.T, h, w)
        # down - conv3d - up
        
        x_img = self.stadapt_up_1(self.stadapt_conv3d_1(self.stadapt_down_1(x_img)))
        x_img = x_img.view(b,d,self.T,h*w).permute(3, 0, 2, 1)
        x_img = x_raw + x_img
        
        x = torch.cat([cls_token, x_img], dim=0)
         
        x = x.view(l, -1, d)
        x = x + self.attention(self.ln_1(x))
        x = x[:l,:,:]
        
        ln_x = self.ln_2(x)
        # x = x + self.mlp(self.ln_2(x))
        if self.num_experts > 0:
            output_head = [self.mlp[0](ln_x)]
            [output_head.append(self.experts_head[i][0](ln_x)) for i in range(self.num_experts)]
            rout1 = torch.nn.functional.softmax(self.routing1(ln_x), -1).unsqueeze(-1)
            output_head = torch.stack(output_head, 0).permute(1,2,0,3)
            output_head = (rout1 * output_head).sum(-2)
            output_head = self.mlp[1](output_head)

            output = [self.mlp[2](output_head)]
            [output.append(self.experts_tail[i](output_head)) for i in range(self.num_experts)]
            rout2 = torch.nn.functional.softmax(self.routing2(output_head), -1).unsqueeze(-1)
            output = torch.stack(output, 0).permute(1,2,0,3)
            output = (rout2 * output).sum(-2)
        else:
            output = self.mlp(ln_x)

        x = x + output

        if self.record_routing:
            if self.num_experts > 0:
                current_rout = torch.stack([rout1.squeeze(-1), rout2.squeeze(-1)], 0)
                if routing_state == None:
                    routing_state = current_rout
                else:
                    routing_state = torch.cat([routing_state, current_rout], 0)

            return x, routing_state

        return x





# ORIGIN Type
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, num_experts=0, record_routing=False, routing_type='patch-level', need_mask=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.num_experts = num_experts
        self.record_routing = record_routing
        self.routing_type = routing_type
        self.need_mask = need_mask

        if num_experts > 0:    
            self.experts_head = nn.Sequential(*[nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    # ("c_proj", nn.Linear(d_model * 4, d_model))
                ])) for _ in range(num_experts)])
            
            self.experts_tail = nn.Sequential(*[nn.Sequential(OrderedDict([
                    ("c_proj", nn.Linear(d_model * 4, d_model))
                ])) for _ in range(num_experts)])
            
            self.routing1 = nn.Linear(d_model, self.num_experts + 1)
            self.routing2 = nn.Linear(d_model*4, self.num_experts + 1)
        
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, mask=None):
        if not self.need_mask:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        else:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
            

    def forward(self, x, mask=None):
        if type(x) == tuple:
            x, routing_state = x
        else:
            routing_state = None

        if not self.need_mask:
            x = x + self.attention(self.ln_1(x),mask)
            
        else:
            x_temp, atten_mask = self.attention(self.ln_1(x),mask)
            x = x + x_temp
        ln_x = self.ln_2(x)
            
        # x = x + self.mlp(self.ln_2(x))
        if self.num_experts > 0:
            # output = self.experts_tail[0](self.experts_head[0][1](self.experts_head[0][0](ln_x)))
             
            output_head = [self.mlp[0](ln_x)]
            [output_head.append(self.experts_head[i][0](ln_x)) for i in range(self.num_experts)]
            
            if self.routing_type == 'patch-level':
                rout1 = torch.nn.functional.softmax(self.routing1(ln_x), -1).unsqueeze(-1)
            elif self.routing_type == 'image-level':
                rout1 = torch.nn.functional.softmax(self.routing1(ln_x[0].unsqueeze(0)), -1).unsqueeze(-1)

            output_head = torch.stack(output_head, 0).permute(1,2,0,3)
            output_head = (rout1 * output_head).sum(-2)
            output_head = self.mlp[1](output_head)
            
            """
            output_head = [self.mlp[1](self.mlp[0](ln_x))]
            [output_head.append(self.experts_head[i](ln_x)) for i in range(self.num_experts)]
            rout1 = torch.nn.functional.softmax(self.routing1(ln_x), -1).unsqueeze(-1)
            output_head = torch.stack(output_head, 0).permute(1,2,0,3)
            output_head = (rout1 * output_head).sum(-2)
            """    
            
            output = [self.mlp[2](output_head)]
            [output.append(self.experts_tail[i](output_head)) for i in range(self.num_experts)]
            # rout2 = torch.nn.functional.softmax(self.routing2(output_head), -1).unsqueeze(-1)
            if self.routing_type == 'patch-level':
                rout2 = torch.nn.functional.softmax(self.routing2(output_head), -1).unsqueeze(-1)
            elif self.routing_type == 'image-level':
                rout2 = torch.nn.functional.softmax(self.routing2(output_head[0].unsqueeze(0)), -1).unsqueeze(-1)
            output = torch.stack(output, 0).permute(1,2,0,3)
            output = (rout2 * output).sum(-2)
            
        else:
            output = self.mlp(ln_x)
        
        x = x + output
        # x = x + self.experts[0](self.ln_2(x))
        if self.record_routing:
            if self.num_experts > 0:
                current_rout = torch.stack([rout1.squeeze(-1), rout2.squeeze(-1)], 0)    
                if routing_state == None:
                    routing_state = current_rout
                else:
                    routing_state = torch.cat([routing_state, current_rout], 0)
             
            return x, routing_state
        
        if not self.need_mask:
            return x
        else:
            return x, atten_mask

# ORIGIN
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# TYPE
class TSTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_checkpoint=False, T=8, temporal_modeling_type=None, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level', need_mask=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.width = width
        self.layers = layers
        self.use_checkpoint = use_checkpoint
        self.T = T
        self.temporal_modeling_type = temporal_modeling_type
        self.record_routing = record_routing
        self.routing_type = routing_type
        self.need_mask = need_mask

        if self.temporal_modeling_type == None:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, num_experts, record_routing, routing_type,need_mask=self.need_mask) if layer_id in expert_insert_layers else ResidualAttentionBlock(width, heads, attn_mask, record_routing=record_routing, routing_type=routing_type,need_mask=self.need_mask) for layer_id in range(layers)])
        elif self.temporal_modeling_type == 'expand_temporal_view' or self.temporal_modeling_type == 'expand_temporal_view_step2' or self.temporal_modeling_type == 'expand_temporal_view_step3':
            self.resblocks = nn.Sequential(*[TimesAttentionBlock(width, heads, attn_mask, T=T, temporal_modeling_type=self.temporal_modeling_type, need_mask=self.need_mask) for _ in range(layers)])
            # TimesAttentionBlock
        elif self.temporal_modeling_type == 'channel_shift':
            self.resblocks = nn.Sequential(*[ChannelShiftAttentionBlock(width, heads, attn_mask, T=T) for _ in range(layers)])
            # ChannelShiftAttentionBlock
        elif self.temporal_modeling_type == 'cross_frame_attend':
            self.resblocks = nn.Sequential(*[CrossFramelAttentionBlock(width, heads, attn_mask, T=T, num_experts=num_experts, record_routing=record_routing) if layer_id in expert_insert_layers else CrossFramelAttentionBlock(width, heads, attn_mask, T=T, record_routing=record_routing) for layer_id in range(layers)])
            # CrossFramelAttentionBlock
        elif self.temporal_modeling_type == 'stadapt_zeroinit':
            self.resblocks = nn.Sequential(*[STAdaptZeroInitAttentionBlock(width, heads, attn_mask, T=T, num_experts=num_experts, record_routing=record_routing) if layer_id in expert_insert_layers else STAdaptZeroInitAttentionBlock(width, heads, attn_mask, T=T, record_routing=record_routing) for layer_id in range(layers)])
            # STAdapter
        elif self.temporal_modeling_type == "adapter":
            self.resblocks = nn.Sequential(*[AdapterAttentionBlock(width, heads, attn_mask, T=T, temporal_modeling_type=self.temporal_modeling_type, adapter=True, need_mask=self.need_mask) if layer_id in expert_insert_layers else AdapterAttentionBlock(width, heads, attn_mask, T=T, temporal_modeling_type=self.temporal_modeling_type, adapter=False, need_mask=self.need_mask) for layer_id in range(layers)])
        else:
            raise NotImplementedError


    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        x1 =[x,mask]
        if not self.use_checkpoint:
            if not self.need_mask:              
                if not self.record_routing:
                    return self.resblocks(x1)
                else:
                    return self.resblocks(x1)
            else:
                if not self.record_routing:
                    for block in self.resblocks:
                        x,attn_mask = block(x, mask)
                    return x,attn_mask
                else:
                    for block in self.resblocks:
                        x,attn_mask = block(x, mask)
                    return x,attn_mask
                """
                for layer_id, resblock in enumerate(self.resblocks):
                    middle_output = resblock(x)
                    if type(middle_output) == tuple and len(middle_output) == 3:
                        x, rout1, rout2 = middle_output
                        routing_state.append(rout1)
                        routing_state.append(rout2)
                    else:
                        x = middle_output

                routing_state = torch.stack(routing_state, 0) 
                return x, routing_state
                """
        else:
            return checkpoint_sequential(self.resblocks, 3, x)
        

# TYPE
class TemporalVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T = 8, temporal_modeling_type = None, use_checkpoint = False, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level', need_mask=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.temporal_modeling_type = temporal_modeling_type
        self.T = T
        self.use_checkpoint = use_checkpoint
        self.record_routing = record_routing
        self.routing_type = routing_type
        self.need_mask = need_mask

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        self.transformer = TSTransformer(width, layers, heads, use_checkpoint=self.use_checkpoint, T=self.T, temporal_modeling_type=self.temporal_modeling_type, num_experts=num_experts, expert_insert_layers=expert_insert_layers, record_routing=record_routing, routing_type=routing_type, need_mask=self.need_mask)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if not self.need_mask:
            if self.record_routing:
                x, routing_state = self.transformer(x, mask)
            else:
                x = self.transformer(x, mask) 
        else:
            if self.record_routing:
                x, atten_mask, routing_state = self.transformer(x, mask)
            else:
                x, atten_mask = self.transformer(x, mask)
        
        if isinstance(x, (list,)):
            x,y,x_a =x
            x = x.permute(1, 0, 2)  # LND -> NLD
            x_a = x_a.permute(1, 0, 2)  # LND -> NLD
            y = y.permute(1, 0, 2)  # LND -> NLD
        else:
            x_a=None
            y=None
            x = x.permute(1, 0, 2)  # LND -> NLD
            
        
        if self.need_mask:
            cls_token = self.ln_post(x[:,0,:]).unsqueeze(1)
            dense_token = self.ln_post(x[:,1:,:])
            x = torch.cat([cls_token,dense_token],dim=1)
        else:
            x = self.ln_post(x).mean(1)
            if x_a is not None:
                y = y.mean(1)
                y = y @ self.proj
                x_a = x_a.mean(1)
                x_a = x_a @ self.proj
            


        if self.proj is not None:
            x = x @ self.proj
        
        if not self.need_mask:
            if self.record_routing:
                return x, routing_state
            else:
                return [x, y, x_a]
        else:
            if self.record_routing:
                return x, atten_mask, routing_state
            else:
                return x, atten_mask
            

        
