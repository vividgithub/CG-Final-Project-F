# PointSynthesis

Currently under development

## Changelog

- 2019.09.02 `Jianwei Jiang`: Add `git`, unit test and `__init__.py`. Complete unittest for `transform_clip_feature` and `transform_scaling`.
- 2019.09.03 `Jianwei Jiang`: Add rotation and sampling transform test
- 2019.09.04 `Jianwei Jiang`: Add `XConvLayer` for x-convolution in PointCNN
- 2019.09.06 `Jianwei Jiang`: Refine `XConvLayer`, add several computation utility
- 2019.09.09 `Jianwei Jiang`: Add `FeatureReshape` layer, initialize the function of generating complete network from 
single configuration file: `modelutil.net_from_conf` (not yet completed)
- 2019.09.10 `Jianwei Jiang`: Add function to create and train network from configuration
- 2019.09.11 - 2019.09.13 `Jianwei Jiang`: Make the PointCNN works in Keras 🌙 
- 2019.09.19 `Jianwei Jiang`: Reset a few days, add evaluation and lastest checkpoint in callback
- 2019.09.28 `Jianwei Jiang`: Add save best pattern and beautify ouptut
- 2019.10.26 `Jianwei Jiang`: Add weight decay
- 2019.10.29 `Jianwei Jiang`: Make the neighborhood aggregation method works in GPU for acceleration
- 2019.10.30 `Jianwei Jiang`: Add mode to resume previous task
- 2019.10.31 `Jianwei Jiang`: Generate network from graph configuration in pyconf