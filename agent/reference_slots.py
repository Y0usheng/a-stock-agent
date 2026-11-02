"""Short transport references; the host still validates canonical evidence identities."""
from copy import deepcopy

PROTOCOL = 'catalog-reference-slots-v1'


def reference_slots(catalog):
    return {r['evidence_id']: f'E{i+1}' for i, r in enumerate(catalog['records'])}


def rewrite_references(value, mapping, key=None):
    if isinstance(value, dict):
        return {k: rewrite_references(v, mapping, k) for k, v in value.items()}
    if key in {'supporting_ids', 'contradicting_ids'} and isinstance(value, list):
        return [mapping.get(item, item) if isinstance(item, str) else deepcopy(item) for item in value]
    if key == 'evidence_id' and isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [rewrite_references(v, mapping) for v in value]
    return deepcopy(value)


def restore_visible_references(value, slots, visible):
    restored = rewrite_references(value, {slot:ref for ref,slot in slots.items() if ref in visible})
    def check(node, key=None):
        if key in {'supporting_ids','contradicting_ids'} and isinstance(node,list):
            if any(not isinstance(ref,str) or ref not in visible for ref in node):
                raise ValueError('模型引用了本角色未展示的证据')
        elif isinstance(node,dict):
            for name,child in node.items():check(child,name)
        elif isinstance(node,list):
            for child in node:check(child)
    check(restored)
    return restored
