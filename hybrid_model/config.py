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
    model_config_cf={'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.58158909923149, 'n_factors': 79,
                     'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05},
    model_type_md=models.AttributeBiasAdvanced,
    model_config_md={'reg_att_bias': 6.578729437598415e-07, 'reg_bias': 6.842025959062749e-07},
    batch_size_cf=1024,
    batch_size_md=2048,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': 0.1928720053014314, 'f_item': 0.5082244606009562,
                          'f_user': 0.9913654219276606, 'min_ratings_item': 8, 'min_ratings_user': 24},
    xtrain_epochs=5,
    xtrain_data_shuffle=True,
    cutoff_user=10,
    cutoff_item=1,
    transformation=transform.TransformationLinear
)

hybrid_config_new2 = HybridConfig(
    model_type_cf=models.SigmoidUserAsymFactoring,
    model_config_cf={'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.58158909923149, 'n_factors': 79,
                     'reg_bias': 0.004770353622067247, 'reg_latent': 2.3618479038250382e-05},
    model_type_md=models.AttributeBiasLight,
    model_config_md={'reg_att_bias': 4.3518131605624814e-05, 'reg_bias': 6.936520853421938e-05},
    batch_size_cf=1024,
    batch_size_md=2048,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': 0.1928720053014314, 'f_item': 0.5082244606009562,
                          'f_user': 0.9913654219276606, 'min_ratings_item': 8, 'min_ratings_user': 24},
    xtrain_epochs=5,
    xtrain_data_shuffle=True,
    cutoff_user=10,
    cutoff_item=1,
    transformation=transform.TransformationLinear
)
