from hybrid_model.hybrid import HybridConfig
from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models

# Default static config we found works well on ML100K
hybrid_config = HybridConfig(
    model_type_cf=models.SVDpp,
    model_config_cf={'n_factors': 40, 'reg_bias': 0.00005, 'reg_latent': 0.00003,
                     'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.75, 'optimizer': 'adagrad'},
    model_type_md=models.AttributeBiasAdvanced,
    model_config_md={'reg_bias': 0.0002, 'reg_att_bias': 0.0003, 'optimizer': 'adagrad'},
    batch_size_cf=256,
    batch_size_md=1024,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': 0.15, 'min_ratings_user': 30, 'f_user': 3.0, 'min_ratings_item': 10, 'f_item': 3.0},
    xtrain_epochs=4,
    xtrain_data_shuffle=True,
    cutoff_user=10,
    cutoff_item=7,
    transformation=transform.TransformationLinear
)

hybrid_config_new = HybridConfig(
    model_type_cf=models.SigmoidUserAsymFactoring,
    model_config_cf={'implicit_thresh': 3.0, 'implicit_thresh_crosstrain': 4.5, 'n_factors': 87,
                     'reg_bias': 5.182106083688767e-07, 'reg_latent': 2.3859821034039756e-05},
    model_type_md=models.AttributeBiasAdvanced,
    model_config_md={'reg_bias': 0.0002, 'reg_att_bias': 0.0004, 'optimizer': 'adagrad'},
    batch_size_cf=256,
    batch_size_md=1024,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': 0.15, 'min_ratings_user': 30, 'f_user': 3.0, 'min_ratings_item': 10, 'f_item': 3.0},
    xtrain_epochs=4,
    xtrain_data_shuffle=True,
    cutoff_user=10,
    cutoff_item=7,
    transformation=transform.TransformationLinear
)
