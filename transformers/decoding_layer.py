seq_length = 3

# Create a Boolean matrix to mask future tokens
tgt_mask = (1 - torch.triu(
  torch.ones(1, seq_length, seq_length), diagonal=1)
).bool()

print(tgt_mask)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        # Perform the attention calculation
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # Apply dropout and the first layer normalization
        x = self.norm1(x + self.dropout(attn_output))
        # Pass through the feed-forward sublayer
        ff_output = self.ff_sublayer(x)
        # Apply dropout and the second layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # Define the list of decoder layers and linear layer
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # Define a linear layer to project hidden states to likelihoods
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt_mask):
        # Complete the forward pass
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


# Instantiate a decoder transformer and apply it to input_tokens and tgt_mask
transformer_decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)
output = transformer_decoder(input_tokens, tgt_mask)
print(output)
print(output.shape)