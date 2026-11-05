"""独立不可变证据快照：CAS提交、依赖失效传播；原B工件永不覆写。"""
import copy
from pathlib import Path
from .ledger_lock import acquire, release
from .storage import IntegrationError, digest, read_json, write_json, file_hash


class EvidenceStore:
    def __init__(self, root, identity):
        self.root=Path(root);self.identity=identity;self.head=self.root/'head.json'
        handle=acquire(self.head)
        try:
            if not self.head.exists():
                self._write({'identity':identity,'parent':None,'artifacts':{},'records':[]})
            if self.current()['identity']!=identity:
                raise IntegrationError('EVIDENCE_SCOPE','证据库样品/配置不匹配')
        finally:release(handle)

    def _write(self, value):
        version=digest(value);target=self.root/'versions'/(version+'.json')
        if not target.exists():write_json(target,{**value,'version':version})
        write_json(self.head,{'version':version})
        return {**value,'version':version}

    def current(self):
        ref=read_json(self.head)['version'];v=read_json(self.root/'versions'/(ref+'.json'))
        if v['version']!=digest({k:x for k,x in v.items() if k!='version'}):
            raise IntegrationError('EVIDENCE_CORRUPT','证据快照哈希错误')
        return v

    def commit(self, parent, artifacts, records=(), replace_scopes=()):
        handle=acquire(self.head)
        try:
            old=self.current()
            if old['version']!=parent:raise IntegrationError('EVIDENCE_STALE','父版本已改变')
            new=copy.deepcopy(old);new.pop('version');new['parent']=parent
            changed=set()
            if len({a['id'] for a in artifacts})!=len(artifacts):
                raise IntegrationError('ARTIFACT_COLLISION','同一提交包含重复工件ID')
            for item in artifacts:
                p=Path(item['path']).resolve()
                if not p.is_file() or file_hash(p)!=item['sha256']:
                    raise IntegrationError('ARTIFACT_CHANGED','工件缺失或哈希变化')
                previous=old['artifacts'].get(item['id'],{})
                core=lambda v:{k:x for k,x in v.items() if k not in {'path','valid'}}
                if core(previous)!=core(item):changed.add(item['id'])
                new['artifacts'][item['id']]={**item,'valid':True}
            supplied={a['id'] for a in artifacts}
            visiting=set();visited=set()
            def visit(k):
                if k in visiting:raise IntegrationError('EVIDENCE_CYCLE','证据依赖出现环')
                if k in visited:return
                visiting.add(k)
                for dep in new['artifacts'][k].get('parents',[]):
                    if dep in new['artifacts']:visit(dep)
                visiting.remove(k);visited.add(k)
            for k in new['artifacts']:visit(k)
            invalid=set(changed)
            while True:
                # 失效必须穿过本批重新提交的中间节点，继续触达未重算的旧下游。
                more={k for k,v in new['artifacts'].items() if set(v.get('parents',[])) & invalid}
                if more <= invalid:break
                invalid |= more
            for k in invalid-supplied:new['artifacts'][k]['valid']=False
            # 新产物依赖必须仍有效；禁止将新mask配旧尺度/旧尺寸。
            for k in supplied:
                deps=new['artifacts'][k].get('parents',[])
                if any(d not in new['artifacts'] or not new['artifacts'][d]['valid'] for d in deps):
                    raise IntegrationError('EVIDENCE_DEPENDENCY','新产物依赖失效')
            def usable(r):
                return all(new['artifacts'].get(d,{}).get('valid') for d in r['depends_on'])
            new['records']=[r for r in new['records'] if r['scope'] not in replace_scopes and usable(r)]
            for row in records:
                if not row.get('depends_on') or not usable(row):
                    raise IntegrationError('EVIDENCE_DEPENDENCY','测量缺少有效来源')
                new['records'].append(copy.deepcopy(row))
            ids=[(r['data'].get('image_stem'),r['data'].get('instance_id')) for r in new['records']]
            if len(ids)!=len(set(ids)):raise IntegrationError('INSTANCE_COLLISION','物理实例主键冲突')
            return self._write(new)
        finally:release(handle)

    def verified_records(self):
        v=self.current()
        for item in v['artifacts'].values():
            if item['valid'] and (not Path(item['path']).is_file() or file_hash(item['path'])!=item['sha256']):
                raise IntegrationError('ARTIFACT_CHANGED','已提交工件被改写')
        return [copy.deepcopy(r['data']) for r in v['records']]
