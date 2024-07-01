# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, Any, List

"""
The API meta template for OpenCompass.

See more details in the OpenCompass documentation: https://opencompass.org.cn/doc
Search for `meta template` in the documentation.
"""


class MetaTemplateType:

    default_api_meta_template_oc = 'default-api-meta-template-oc'

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_template_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


def register_template(name: str,
                      template: Dict[str, Any],
                      exists_ok: bool = False):
    if not exists_ok and name in TEMPLATE_MAPPING:
        raise ValueError(f"The `{name}` has already been registered in the TEMPLATE_MAPPING.")

    TEMPLATE_MAPPING[name] = template


def get_template(name: str) -> Dict[str, Any]:
    if name not in TEMPLATE_MAPPING:
        raise ValueError(f"The `{name}` has not been registered in the TEMPLATE_MAPPING.")

    return TEMPLATE_MAPPING[name]


# Default API meta template for OpenCompass
register_template(
    name=MetaTemplateType.default_api_meta_template_oc,
    template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True)
        ],
        reserved_roles=[
            dict(role='SYSTEM', api_role='SYSTEM'),
        ],
    )
)


if __name__ == '__main__':
    res = MetaTemplateType.get_template_name_list()
    print(res)

    print(get_template(MetaTemplateType.default_api_meta_template_oc))
