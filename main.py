device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(num_tokens=22, dim_model=128, num_heads=2, num_encoder_layers=2, num_decoder_layers=2, dropout_p=0.1).to(device)
model_time = nn.Linear(in_features=128, out_features=1, bias=True).to(device)
softmax = nn.Softmax(dim=2)
linear = nn.Linear(in_features=128, out_features=1, bias=True).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
opt_time = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = SoftDTW(use_cuda=True, gamma=0.1)
loss_fn_time = nn.MSELoss()
time_parameter = 0.01

train_loss_list, validation_loss_list = fit(model, model_time, opt, loss_fn, loss_fn_time, train_dataloader, val_dataloader, 100)
