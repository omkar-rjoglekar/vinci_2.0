
class hyper_parameters():
    def __init__(self):
        self.gamma = 0.99
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.total_episodes = 5000
        self.img_dims = (48, 48)
        self.state_dims = (48, 48)
        self.action_dimension = 3
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.tau = 0.01
        self.batch_size = 256
        self.max_seq_length = 40
        self.std_dev = 1
        self.latent_dim = 261

HP = hyper_parameters()
