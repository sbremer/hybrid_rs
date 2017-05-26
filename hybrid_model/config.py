from hybrid_model.hybrid import HybridConfig
from hybrid_model.index_sampler import IndexSamplerUserbased
from hybrid_model import transform

hybrid_config = HybridConfig(
    n_factors=40,
    reg_bias_mf=0.00005,
    reg_latent=0.00004,#2
    reg_bias_cs=0.0001,
    reg_att_bias=0.0015,
    implicit_thresh_init=0.7,
    implicit_thresh_xtrain=0.8,
    opt_mf_init='nadam',
    opt_cs_init='nadam',
    opt_mf_xtrain='adadelta',
    opt_cs_xtrain='adadelta',
    batch_size_init_mf=512,
    batch_size_init_cs=1024,
    batch_size_xtrain_mf=256,
    batch_size_xtrain_cs=1024,
    val_split_init=0.05,
    val_split_xtrain=0.05,
    index_sampler=IndexSamplerUserbased,
    xtrain_patience=5,
    xtrain_max_epochs=10,
    xtrain_data_shuffle=False,
    transformation=transform.TransformationLinear
)