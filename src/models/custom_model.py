import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel

class CustomSentimentModel(PreTrainedModel):
    """
    This custom model uses a pretrained backbone (such as RoBERTa)
    and adds a task-specific classification head consisting of:
      - A dropout layer,
      - A dense intermediate layer with a non-linear activation,
      - Another dropout,
      - And a final classification layer.
    
    The model is designed for three classes (negative, neutral, positive).
    """

    config_class = AutoConfig
    base_model_prefix = "custom_sentiment_model"

    def __init__(self, config):
        super().__init__(config)
        # Load pretrained backbone, e.g., from a model checkpoint such as "roberta-base"
        self.backbone = AutoModel.from_pretrained(config._name_or_path, config=config)
        
        # Set hidden size from the backbone configuration (usually config.hidden_size)
        hidden_size = config.hidden_size

        # Define the custom classification head.
        self.dropout = nn.Dropout(p=0.3)
        self.classifier_dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()  # You can experiment with other activations.
        self.dropout2 = nn.Dropout(p=0.3)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

        # Initialize weights for the new layers
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        
        # Remove any unexpected keyword arguments that the backbone does not accept.
        # In particular, remove "num_items_in_batch" if present.
        kwargs.pop("num_items_in_batch", None)

        # Get outputs from the backbone.
        # Depending on the model (e.g., RoBERTa does not use token_type_ids)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        
        # The backbone returns a tuple. The first element is the sequence output,
        # and the second is the pooled output (if available). Some models (like RoBERTa)
        # don't have a pooled output, so we may take the representation of the CLS token.
        # Here we assume that outputs[0] has shape (batch_size, seq_length, hidden_size),
        # and we take the first token ([CLS]) as the pooled representation.
        pooled_output = outputs[0][:,0]  # shape: (batch_size, hidden_size)
        
        # Pass through our custom classification head.
        x = self.dropout(pooled_output)
        x = self.classifier_dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        logits = self.out_proj(x)
        
        # If labels are provided, compute loss (using CrossEntropyLoss).
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # Return a dictionary; you can also return a tuple if you prefer.
        return {"loss": loss, "logits": logits}
