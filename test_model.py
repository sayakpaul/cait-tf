from email.mime import base

import tensorflow as tf

from cait.model_configs import base_config
from cait.models import CaiT

batch_size = 2
config = base_config.get_config()


random_tensor = tf.random.normal(
    (batch_size, config.image_size, config.image_size, 3)
)
cait_model = CaiT(config, name=config.model_name)
outputs = cait_model(random_tensor)
print(outputs[0].shape)
print(cait_model.summary(expand_nested=True))
