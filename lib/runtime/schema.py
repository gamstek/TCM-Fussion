"""组件内部 JSON Schema 校验。拒绝加载外部引用，统一规范化 JSON 输入与失败类型。
由「lib/tools/tool_registry.py」、「lib/vision/presence_localization.py」调用，结果按调用方的任务阶段消费。

主要定义：
- object_schema。
- local_refs。
- validate_arguments。

关联：
- 调用方 「lib/tools/tool_registry.py」、「lib/vision/presence_localization.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import json
from jsonschema import Draft202012Validator
from TCM_Fusion.lib.runtime.storage import IntegrationError


def object_schema(fields):
    return {'type':'object','properties':{k:{'type':v} for k,v in fields.items()},
            'required':list(fields),'additionalProperties':False}


def local_refs(schema):
    if isinstance(schema,dict):
        for k,v in schema.items():
            if k in {'$ref','$dynamicRef'} and (not isinstance(v,str) or not v.startswith('#')):
                raise IntegrationError('SCHEMA_EXTERNAL_REF','禁止外部schema引用')
            local_refs(v)
    elif isinstance(schema,list):
        for v in schema: local_refs(v)


def validate_arguments(schema,args):
    try:
        local_refs(schema); Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(json.loads(json.dumps(args,allow_nan=False)))
    except IntegrationError: raise
    except Exception as exc: raise IntegrationError('ACTION_DENIED','工具schema校验失败') from exc


# Optional presence hints are diagnostic metadata, not a CropMeta delivery contract.
# Keep their response schema here with the tool's other schema validation.
PRESENCE_BOX_SCHEMA = {
    'type': 'object', 'required': ['bbox', 'coordinate_space'],
    'properties': {
        'bbox': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                 'items': {'type': 'number'}},
        'coordinate_space': {'const': 'model_input_pixels'},
    }, 'additionalProperties': False,
}
