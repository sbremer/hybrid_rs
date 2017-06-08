from hybrid_model.hybrid import HybridConfig
from hybrid_model import index_sampler
from hybrid_model import transform
from hybrid_model import models

# Default static config we found works well
hybrid_config = HybridConfig(
    model_type_cf=models.SVDpp,
    model_config_cf={},
    model_type_md=models.AttributeBiasExperimental,
    model_config_md={},
    opt_cf_init='nadam',
    opt_md_init='nadam',
    opt_cf_xtrain='adadelta',
    opt_md_xtrain='adadelta',
    batch_size_init_cf=512,
    batch_size_init_md=512,
    batch_size_xtrain_cf=256,
    batch_size_xtrain_md=512,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=index_sampler.IndexSamplerUserbased,
    xtrain_epochs=6,
    xtrain_data_shuffle=False,
    transformation=transform.TransformationLinear
)
