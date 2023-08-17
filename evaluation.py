def evaluation(model, model_time, loss_fn, loss_fn_time, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, X_time = batch[:, 0], batch[:, 1]
            X_timestep = batch[:, 2]
            y, y_time = batch[:, 3], batch[:, 4]
            y_timestep = batch[:, 5]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
            X_time, y_time = torch.tensor(X_time).to(device), torch.tensor(y_time).to(device)
            X_timestep, y_timestep = torch.tensor(X_timestep).to(device), torch.tensor(y_timestep).to(device)

            y_input = y[:,:-1]
            y_input_time = y_time[:,:-1]
            y_expected = y[:,1:]
            y_input_time_expected = y_input_time[:,1:]
            y_input_timestep_expected = y_timestep[:,1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, X_time, y_input, y_input_time, tgt_mask)

            pred = pred.permute(1, 0, 2)
            pred_time = model_time(pred)
            pred_time_sum = torch.sum(pred_time, 1).unsqueeze(2)
            pred_time = pred_time * time_parameter

            pred = softmax(pred)
            #원소표현 + 시간차
            final_pred = torch.cat((pred, pred_time), 2)

            #y를 one-hot encoding(loss계산을 위해)
            one_hot_y = nn.functional.one_hot(y_expected, num_classes = 12)
            y_input_timestep_expected = y_input_timestep_expected.unsqueeze(2) * time_parameter
            final_y = torch.cat((one_hot_y, y_input_timestep_expected), 2)

            # loss 계산
            loss = loss_fn(final_pred, final_y).mean()
            
    return total_loss / len(dataloader)
