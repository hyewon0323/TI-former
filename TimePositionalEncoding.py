class TimePositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, batch_size):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.dim_model = dim_model
        self.dropout_p = dropout_p
        self.batch_size = batch_size


    def forward(self, token_embedding: torch.tensor, timestep) -> torch.tensor:
        input_len = len(token_embedding[0])
        pos_encoding = torch.zeros(self.batch_size, input_len, self.dim_model)
        timestep = timestep.float()
        arr = list(range(22)) * 16
        arr = np.array(arr)
        positions_list = np.split(arr, 16)
        positions_list = torch.Tensor(positions_list)
        positions_list = positions_list.unsqueeze(2) 
        division_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0)) / self.dim_model) # 1 / 1000^(2i/dim_model)
        division_term = torch.as_tensor(division_term, device = 'cuda')
        a = len(division_term)
        division_term = division_term.unsqueeze(0)
        division_term = division_term.unsqueeze(0)
        division_term = division_term.expand(self.batch_size, 1, a) 
        
        pos_encoding[:, :, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, :, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = torch.as_tensor(pos_encoding, device = 'cuda')

        self.register_buffer("pos_encoding",pos_encoding) 
        token_embedding = torch.as_tensor(token_embedding, device = 'cuda')
        
        return self.dropout(token_embedding + pos_encoding)
