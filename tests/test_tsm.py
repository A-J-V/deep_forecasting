# Test whether training reduces loss
def test_train():
    import pandas as pd
    from pathlib import Path
    import torch
    from torch.optim import Adam
    from deep_forecasting import HTSM, TSManager, get_tsds, get_dataloaders

    lookback = 6
    forecast = 6
    num_aux = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = pd.read_csv(
        Path(__file__).parent / "synthetic_ts_data.csv",
    )
    manager = TSManager(data)
    processed_data = manager.transform_all(data)

    train_ds, test_ds = get_tsds(processed_data,
                                 lookback=lookback,
                                 forecast=forecast,
                                 train_prop=0.7,
                                 )
    train_loader, test_loader = get_dataloaders(train_ds, test_ds, train_batch_size=4, test_batch_size=4)

    model = HTSM(lookback=lookback,
                 dataset=train_ds,
                 hidden_features=12,
                 forecast=forecast,
                 blocks=1,
                 dropout=0.75,
                 num_aux=num_aux,
                 device=device,
                 )

    optimizer = Adam(params=model.model.parameters(),
                     lr=0.001,
                     )

    model.train(train_dataloader=train_loader,
                test_dataloader=test_loader,
                epochs=50,
                loss_fn=torch.nn.L1Loss(),
                optimizer=optimizer,
                verbose=False,
                )

    assert model.logs['train_loss'][-1] < model.logs['train_loss'][
        0], "Final train loss isn't less than initial train loss!"
