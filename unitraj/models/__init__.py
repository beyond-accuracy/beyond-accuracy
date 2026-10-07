from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.fmae.trainer_mae import TrainerMAE
from unitraj.models.fmae.trainer_forecast import TrainerForecast
from unitraj.models.demo.trainer import DeMoTrainer

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'MAE': TrainerMAE,          # ForecastMAE pre-training
    'forecast': TrainerForecast,  # ForecastMAE fine-tuning
    'DeMo': DeMoTrainer,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
