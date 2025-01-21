import torch
import torch.nn as nn
from transformers import ResNetModel
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertLayer

device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
class AttentionAddModel(BertPreTrainedModel):
    def __init__(self, config):
        super(AttentionAddModel, self).__init__(config)
        self.bert = BertModel(config)
        self.resnet = ResNetModel.from_pretrained("./pretrained_model/resnet-50")
        self.comb_attention = BertLayer(config)  
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.image_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.classifier_single = nn.Linear(in_features=config.hidden_size, out_features=3)

    def forward(self, image_input=None, text_input=None):
        if (image_input is not None) and (text_input is not None):
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state

            t = self.resnet(**image_input).last_hidden_state
            image_features = t.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)

            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

            attention_mask = text_input.attention_mask
            image_attention_mask = torch.ones((attention_mask.size(0), 1)).to(device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], 1).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000

            image_text_attention_state = self.comb_attention(image_text_hidden_state, attention_mask)[0]

            image_pooled_output = self.image_pool(image_text_attention_state[:, 0, :])
            text_pooled_output = self.text_pool(image_text_attention_state[:, 1, :])
            final_output = torch.add(image_pooled_output, text_pooled_output)

            out = self.classifier_single(final_output)
            return out

class CatModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CatModel, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.resnet = ResNetModel.from_pretrained("./pretrained_model/resnet-50")
        self.comb_attention = BertLayer(config)
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.image_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential (
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(in_features=config.hidden_size * 2, out_features=3)
        self.classifier_single = nn.Linear(in_features=config.hidden_size, out_features=3)

    def forward(self, image_input = None, text_input = None):
        if (image_input is not None) and (text_input is not None):
            text_features = self.bert(**text_input)
            text_hidden_state, _ = text_features.last_hidden_state.max(1)
            
            t = self.resnet(**image_input).last_hidden_state
            image_features = t.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1).view(-1, self.hidden_size)
 
            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

            out = self.classifier(image_text_hidden_state)
            return out

class AttentionCatModel(BertPreTrainedModel):
    def __init__(self, config):
        super(AttentionCatModel, self).__init__(config)
        self.bert = BertModel(config)
        self.resnet = ResNetModel.from_pretrained("./pretrained_model/resnet-50")
        self.comb_attention = BertLayer(config)
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.image_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential (
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(in_features=config.hidden_size * 2, out_features=3)
        self.classifier_single = nn.Linear(in_features=config.hidden_size, out_features=3)

    def forward(self, image_input = None, text_input = None):
        if (image_input is not None) and (text_input is not None):
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state

            t = self.resnet(**image_input).last_hidden_state
            image_features = t.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)
            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

            attention_mask = text_input.attention_mask
            image_attention_mask = torch.ones((attention_mask.size(0), 1)).to(device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], 1).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000
 
            image_text_attention_state = self.comb_attention(image_text_hidden_state, attention_mask)[0]
            image_pooled_output = self.image_pool(image_text_attention_state[:, 0, :])
            text_pooled_output = self.text_pool(image_text_attention_state[:, 1, :])
            final_output = torch.cat([image_pooled_output, text_pooled_output], 1)

            out = self.classifier(final_output)
            return out
