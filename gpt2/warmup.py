from jittor.lr_scheduler import MultiStepLR


class LambdaLR(object):
    def __init__(self, optimizer, lr_lambda,  last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.lr_lambda = lr_lambda
        #TODO set last_epoch is not ready
    
    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambda')}
        state_dict['lr_lambda'] = None

    def load_state_dict(self, state_dict):
        lr_lambda = state_dict.pop('lr_lambda')
        self.__dict__.update(state_dict)
        state_dict['lr_lambda'] = lr_lambda
        self.lr_lambda.__dict__.update(lr_lambda)

    def get_gamma(self):
        return self.lr_lambda(self.last_epoch)

    def get_lr(self):
        now_lr = self.optimizer.lr
        return now_lr * self.get_gamma()

    def step(self):
        self.last_epoch += 1
        self.update_lr()
            
    def update_lr(self):
        gamma = self.get_gamma()
        if gamma != 1.0:
            self.optimizer.lr = self.optimizer.lr * gamma
            for i, param_group in enumerate(self.optimizer.param_groups):
                if param_group.get("lr") != None:
                    param_group["lr"] = param_group["lr"] * gamma


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)