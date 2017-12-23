class Config:
    """feel free to play with these hyperparameters during training"""
    dataset = "resource"  # change this to the right data name
    data_path = "../%s" % dataset
    checkpoint_dir = "checkpoint"
    decay_rate = 0.95
    decay_step = 1000
    n_topics = 50
    learning_rate = 0.00002
    vocab_size = 619
    n_stops = 22 
    lda_vocab_size = vocab_size - n_stops
    n_hidden = 200
    n_layers = 2
    projector_embed_dim = 100
    generator_embed_dim = n_hidden
    dropout = 1.0
    max_grad_norm = 1.0 #for gradient clipping
    total_epoch = 5
    init_scale = 0.075
    threshold = 0.5 #probability cut-off for predicting label to be 1
    forward_only = False #indicates whether we are in testing or training mode

    log_dir = '../logs'
