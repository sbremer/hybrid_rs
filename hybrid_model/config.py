from hybrid_model.hybrid import HybridConfig
from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models

# Default static config we found works well
hybrid_config = HybridConfig(
    model_type_cf=models.SVDpp,
    model_config_cf={'n_factors': 40, 'reg_bias': 0.00005, 'reg_latent': 0.00003, 'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.75},
    model_type_md=models.AttributeBiasExperimental,
    model_config_md={'reg_bias': 0.0002, 'reg_att_bias': 0.0003},
    opt_cf_init='adagrad',
    opt_md_init='adagrad',
    opt_cf_xtrain='adagrad',
    opt_md_xtrain='adagrad',
    batch_size_init_cf=256,
    batch_size_init_md=1024,
    batch_size_xtrain_cf=256,
    batch_size_xtrain_md=1024,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserItembased,
    index_sampler_config={'f_cf': 0.15, 'min_ratings_user': 35, 'f_user': 3.0, 'min_ratings_item': 10, 'f_item': 3.0},
    xtrain_epochs=4,
    xtrain_data_shuffle=True,
    transformation=transform.TransformationLinear
)

# hybrid_config_best = HybridConfig(
#     model_type_cf=models.SVDpp,
#     model_config_cf={'n_factors': 40, 'reg_bias': 0.00005, 'reg_latent': 0.00003, 'implicit_thresh': 4.0, 'implicit_thresh_crosstrain': 4.75},
#     model_type_md=models.AttributeBiasExperimental,
#     model_config_md={'reg_bias': 0.0002, 'reg_att_bias': 0.0003},
#     opt_cf_init='adagrad',
#     opt_md_init='adagrad',
#     opt_cf_xtrain='adagrad',
#     opt_md_xtrain='adagrad',
#     batch_size_init_cf=256,
#     batch_size_init_md=1024,
#     batch_size_xtrain_cf=256,
#     batch_size_xtrain_md=1024,
#     val_split_init=0.05,
#     val_split_xtrain=0.05,
#     index_sampler=index_sampler.IndexSamplerUserItembased,
#     xtrain_epochs=4,
#     xtrain_data_shuffle=False,
#     transformation=transform.TransformationLinear
# )
#
# hybrid_config_old = HybridConfig(
#     model_type_cf=models.SVDpp,
#     model_config_cf={'n_factors': 40, 'reg_bias': 0.00001, 'reg_latent': 0.00002},
#     model_type_md=models.AttributeBiasExperimental,
#     model_config_md={'reg_bias': 0.00005, 'reg_att_bias': 0.00015},
#     opt_cf_init='nadam',
#     opt_md_init='nadam',
#     opt_cf_xtrain='adadelta',
#     opt_md_xtrain='adadelta',
#     batch_size_init_cf=512,
#     batch_size_init_md=1024,
#     batch_size_xtrain_cf=256,
#     batch_size_xtrain_md=1024,
#     val_split_init=0.05,
#     val_split_xtrain=0.05,
#     index_sampler=index_sampler.IndexSamplerUserItembased,
#     xtrain_epochs=5,
#     xtrain_data_shuffle=False,
#     transformation=transform.TransformationLinear
# )

