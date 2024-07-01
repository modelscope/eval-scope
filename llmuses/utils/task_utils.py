# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum


class EvalBackend(Enum):
    # Use native evaluation pipeline of Eval-Scope
    NATIVE = 'Native'

    # Use OpenCompass framework as the evaluation backend
    OPEN_COMPASS = 'OpenCompass'

    # Use third-party evaluation backend/modules
    THIRD_PARTY = 'ThirdParty'

