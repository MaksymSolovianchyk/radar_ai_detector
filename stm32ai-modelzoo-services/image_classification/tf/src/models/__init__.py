# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .st_mnistv1 import get_st_mnistv1
from .st_fdmobilenetv1 import get_st_fdmobilenetv1
from .st_efficientnetlcv1 import get_st_efficientnetlcv1
from .squeezenetv11 import get_squeezenetv11
from .resnet import get_resnet
from .resnet50v2 import get_resnet50v2
from .mobilenetv2 import get_mobilenetv2
from .mobilenetv1 import get_mobilenetv1
from .fdmobilenet import get_fdmobilenet
from .efficientnetv2 import get_efficientnetv2
from .custom_model import get_custom_model
from .utils import prepare_kwargs_for_model
