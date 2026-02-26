# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .resnet import get_resnet
from .resnet50v2 import get_resnet50v2
from .mobilenetv2 import get_mobilenetv2
from .mobilenetv1 import get_mobilenetv1
from .efficientnetv2 import get_efficientnetv2
from .osnet import get_osnet
from .custom_model import get_custom_model
from .model_utils import prepare_kwargs_for_model
