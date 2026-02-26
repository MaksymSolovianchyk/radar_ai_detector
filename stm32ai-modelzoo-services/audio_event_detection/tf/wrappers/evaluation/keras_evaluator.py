# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
from common.registries.evaluator_registry import EVALUATOR_WRAPPER_REGISTRY

from audio_event_detection.tf.src.evaluation import AEDKerasEvaluator

__all__ = ['AEDKerasEvaluator']


EVALUATOR_WRAPPER_REGISTRY.register(
    evaluator_name="keras_evaluator",
    framework="tf",
    use_case="audio_event_detection"
)(AEDKerasEvaluator)

