"""Bounded CNINFO implementation notices, with document/page evidence.

Only explicitly supported ordinary A-share implementation notices are accepted.
Ambiguous units/dates, corrections, incomplete result pages and fractional stock
allocations stop settlement; announcement proposals never create account cash.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from time import monotonic
import unicodedata

from agent.corporate_actions import VERSION, number, validate_action
from agent.disclosures import controlled_pdf, download_pdf, extract_pages
from agent.trading_clock import now
from config import DATA_DIR,MARKETS
from evals.experiments import canonical_hash
from servers.data_quality import event_time

HOME = DATA_DIR/'corporate_actions'
PARSER = 'cn-implementation-notice-v2'
DATE = r'(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}(?=\d{4}[/年-]|\D|$)|\d{4}-\d{2}-\d{2}(?=\d{4}[/年-]|\D|$))'


def day(value):
    parts = re.findall(r'\d+',value)
    return datetime(*(int(p) for p in parts[:3])).date().isoformat()


def extract_notice(path):
    import pymupdf
    extracted=extract_pages(path)
    if extracted['page_count']>20:raise ValueError('权益实施公告页数超出专用上限')
    tables=[]
    with pymupdf.open(path) as reader:
        for index,page in enumerate(reader):
            for table in page.find_tables().tables:
                rows=table.extract()
                if rows and '股权登记日' in re.sub(r'\s+','',''.join(str(c or '') for c in rows[0])):
                    tables.append({'pdf_page':index+1,'bbox':list(table.bbox),'rows':rows,'rows_hash':canonical_hash(rows)})
    return {**extracted,'tables':tables}


def parse_notice(document, extracted, known_at):
    pages = extracted['pages']
    text = '\n'.join(p['text'] for p in pages)
    compact = unicodedata.normalize('NFKC',re.sub(r'\s+','',text))
    symbol = document['symbol']
    if event_time(document['published_at'],'cn')>event_time(known_at,'cn'):
        raise ValueError('不能使用尚未披露的权益实施公告')
    identity=re.search(r'(?:证券|股票)代码[:：]?(\d{6})',re.sub(r'\s+','',pages[0]['text']))
    if not identity or identity[1]!=symbol or not controlled_pdf(document['url']):
        raise ValueError('实施公告股票代码不符')
    title = document['title']
    if any(s in title for s in ('更正','修订','取消','延期','优先股','B股')) or ('H股' in title and 'A股' not in title):
        raise ValueError('实施公告需要复核，不能自动代替既有事件')
    if any(s in compact[:900] for s in ('配股缴款','缩股','吸收合并','证券转换')):
        raise ValueError('不支持的公司行为，停止自动权益处理')
    # Shanghai's important notice lists actual per-share amounts. Shenzhen's
    # section 二 is the implemented scheme, after any earlier proposal figures.
    split=bool(re.search(r'股份(?:分拆|拆细)实施公告',title))
    if split:
        scheme=compact[:3500]
    elif any(k in compact[:1200] for k in ('每股分配比例','每股转增比例')) and '相关日期' in compact[:1500]:
        begin=min(compact.index(k) for k in ('每股分配比例','每股转增比例') if k in compact[:1200])
        scheme = compact[begin:compact.index('相关日期')]
    else:
        matches = list(re.finditer(r'二[、.](?:本次实施的.{0,12}方案|(?:A股)?权益分派方案|本次(?:A股)?权益分派方案)',compact))
        if not matches:raise ValueError('未找到明确的本次实施方案')
        scheme = compact[matches[-1].end():]
        scheme = re.split(r'三[、.]',scheme,maxsplit=1)[0]
    amounts = []
    for m in re.finditer(r'每(?:(10|1)股|股)(?:派发)?现金(?:红利|分红|股利)(?:人民币)?([0-9]+(?:\.[0-9]+)?)元',scheme):
        amounts.append(number(m[2])/number(m[1] or 1))
    for m in re.finditer(r'每(?:(10|1)股|股)派([0-9]+(?:\.[0-9]+)?)元(?:人民币)?现金\(含税',scheme):
        amounts.append(number(m[2])/number(m[1] or 1))
    if split:amounts=[Decimal(0)]
    if not amounts and ('每股转增' in scheme or '每10股转增' in scheme) and '现金' not in scheme:
        amounts=[Decimal(0)]  # The complete per-share allocation summary contains only stock.
    if len(set(amounts))!=1:raise ValueError('现金分红金额或每股/每10股单位不唯一')
    bonus = Decimal(0)
    if split:
        ratios={number(m[1]) for m in re.finditer(r'每1股(?:普通股)?拆(?:分|细)为([0-9]+(?:\.[0-9]+)?)股',scheme)}
        if len(ratios)!=1 or next(iter(ratios))<=1:raise ValueError('股份拆细比例缺失或不是正向拆股')
        bonus=next(iter(ratios))-1
    for pattern in (r'每(?:(10|1)股|股)送(?:红股)?([0-9]+(?:\.[0-9]+)?)股',
                    r'每(?:(10|1)股|股)(?:以资本公积金)?转增([0-9]+(?:\.[0-9]+)?)股'):
        values = {number(m[2])/number(m[1] or 1) for m in re.finditer(pattern,scheme)}
        if len(values)>1:raise ValueError('送转比例不唯一')
        bonus += next(iter(values),Decimal(0))
    if any(s in scheme for s in ('送股','送红股','转增')) and not bonus:
        negative = re.sub(r'不(?:实施资本公积金|进行资本公积|进行资本公积金|以公积金|以资本公积金|以资本公积)?转增股本|不送红股|不送股','',scheme)
        if any(s in negative for s in ('送股','送红股','转增')):
            raise ValueError('存在未解析的送转方案')
    dates = {k:set() for k in ('record_date','ex_date','pay_date','listing_date')}
    for table in extracted.get('tables',[]):
        rows=table['rows']
        if table.get('rows_hash')!=canonical_hash(rows):raise ValueError('权益日期表指纹不符')
        labels=[unicodedata.normalize('NFKC',re.sub(r'\s+','',str(c or ''))) for c in rows[0]]
        for row in rows[1:]:
            cells=[unicodedata.normalize('NFKC',re.sub(r'\s+','',str(c or ''))) for c in row]
            if len(cells)!=len(labels):raise ValueError('权益日期表列数不符')
            if '股份类别' in labels and cells[labels.index('股份类别')]!='A股':continue
            for label,cell in zip(labels,cells):
                key='record_date' if label=='股权登记日' else 'ex_date' if label.startswith('除权') else 'pay_date' if label=='现金红利发放日' else 'listing_date' if '上市' in label else None
                if key and cell not in {'','-','—','－'}:
                    if not re.fullmatch(DATE,cell):raise ValueError('权益日期单元格无法核验')
                    dates[key].add(day(cell))
    for m in ([] if extracted.get('tables') else re.finditer(r'(?P<header>股权登记日.{0,160}?)(?P<row>'+DATE+r'(?:'+DATE+r'|[—－-])+)',compact)):
        header=m['header']
        if '除权' not in header:continue
        if '股份类别' in compact[max(0,m.start()-10):m.start()] and not header.endswith('A股'):continue
        columns=re.findall(r'股权登记日|最后交易日|除权\(?息\)?日|新增无限售条件流通股(?:份)?上市(?:流通)?日|现金红利发放日',header)
        tokens=re.findall(DATE+r'|[—－-]',m['row'])
        if len(tokens)!=len(columns) or '除权' not in ''.join(columns):
            raise ValueError('实施公告日期表不能可靠解析')
        for label,token in zip(columns,tokens):
            if not re.fullmatch(DATE,token):continue
            key='record_date' if label=='股权登记日' else 'ex_date' if label.startswith('除权') else 'pay_date' if label=='现金红利发放日' else 'listing_date' if '上市' in label else None
            if key:dates[key].add(day(token))
    labels = {'股权登记日':'record_date','除权除息日':'ex_date','现金红利发放日':'pay_date',
              '红利发放日':'pay_date','新增无限售条件流通股份上市日':'listing_date','本次所送(转)股的起始交易日':'listing_date',
              '股份拆细生效日':'ex_date','股份分拆生效日':'ex_date','拆分股份上市日':'listing_date'}
    for label,key in labels.items():
        for m in re.finditer(re.escape(label)+r'(?:为)?[:：]?('+DATE+r')',compact):dates[key].add(day(m[1]))
    for m in re.finditer(r'除权除息日及现金红利发放日(?:为)?[:：]?('+DATE+r')',compact):
        dates['ex_date'].add(day(m[1]));dates['pay_date'].add(day(m[1]))
    for m in re.finditer(r'(?:A股股东现金红利|公司此次委托|本公司此次委托).{0,55}?将于('+DATE+r').{0,65}?直接划入其资金账户',compact):
        dates['pay_date'].add(day(m[1]))
    if any(len(v)>1 for v in dates.values()):raise ValueError('实施公告日期冲突')
    values = {k:next(iter(v),None) for k,v in dates.items()}
    result = {'event_id':'cninfo:'+document['announcement_id'],'symbol':symbol,'kind':'forward_split' if split else 'distribution',
        'cash_per_share':str(amounts[0]),'bonus_per_share':str(bonus),**values,'known_at':known_at,
        'source':{**document,'pdf_sha256':extracted['pdf_sha256'],'parser':PARSER,
                  'evidence':[{'pdf_page':p['pdf_page'],'text_hash':p['text_hash']} for p in pages],
                  'date_tables':[{k:t[k] for k in ('pdf_page','bbox','rows_hash')} for t in extracted.get('tables',[])],
                  'implemented_scheme':scheme[:1600]}}
    return validate_action(result)


def query_notices(session, symbol, since, cutoff, deadline):
    def post(url,data):
        if monotonic()>=deadline:raise TimeoutError('权益采集预算到期')
        r=session.post(url,data=data,timeout=(5,max(1,min(12,deadline-monotonic()))))
        r.raise_for_status();return r.json()
    companies = post('https://www.cninfo.com.cn/new/information/topSearch/query',{'keyWord':symbol,'maxNum':10})
    company = next(r for r in companies if r.get('code')==symbol)
    found = {}
    scan_from=max(since,(event_time(cutoff,'cn')-timedelta(days=90)).date().isoformat())
    # Implemented notices cover older dividends; the full recent title
    # scan catches ordinary-share splits, rights issues and implementation edits.
    for keyword,begin in [('实施',since),('',scan_from)]:
        total=None;seen=set()
        for page in range(1,5):
            params={'pageNum':str(page),'pageSize':'30','column':'szse','tabName':'fulltext',
                'stock':symbol+','+company['orgId'],'searchkey':keyword,'category':'',
                'seDate':begin+'~'+str(cutoff)[:10],'sortName':'time','sortType':'desc','isHLtitle':'false'}
            response=post('https://www.cninfo.com.cn/new/hisAnnouncement/query',params)
            if not isinstance(response.get('announcements'),list) or response.get('totalAnnouncement') is None:
                raise ValueError('权益公告目录结构不完整')
            count=response['totalAnnouncement']
            if isinstance(count,bool) or not isinstance(count,int) or count<0 or (total is not None and total!=count):
                raise ValueError('权益目录总数无效或在分页期间发生变化')
            total=count
            for row in response['announcements']:
                identifier=row.get('announcementId') or row.get('adjunctUrl')
                if not identifier or identifier in seen:raise ValueError('权益目录重复页或缺少身份')
                seen.add(identifier)
                title=re.sub(r'<[^>]*>','',row['announcementTitle'])
                if row.get('secCode')!=symbol:raise ValueError('权益公告目录公司身份不符')
                published=datetime.fromtimestamp(float(row['announcementTime'])/1000,timezone.utc).astimezone(event_time(cutoff,'cn').tzinfo)
                stamp=published.date().isoformat() if published.hour==published.minute==0 else published.isoformat()
                if event_time(stamp,'cn')>event_time(cutoff,'cn'):continue
                if any(k in title for k in ('优先股','B股')) or ('H股' in title and 'A股' not in title):continue
                related=any(k in title for k in ('权益分派','分红','派息','股份分拆','股份拆细','配股','缩股','换股实施','股份合并'))
                if related and any(k in title for k in ('更正','取消','延期','配股','缩股','换股实施','股份合并')):
                    raise ValueError('权益目录含修订或不支持的公司行为，须核对后恢复')
                if not re.search(r'(?:权益分派|分红(?:派息)?|派息|股份分拆|股份拆细).{0,5}实施公告',title):continue
                path=row['adjunctUrl']
                if not re.fullmatch(r'finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]',path):
                    raise ValueError('实施公告PDF路径不受控')
                found[path]={'symbol':symbol,'title':title,'announcement_id':Path(path).stem,
                             'published_at':stamp,'url':'https://static.cninfo.com.cn/'+path}
            # CNINFO hasMore can remain true on its last page; reconcile the
            # unique entries against totalAnnouncement instead of guessing.
            if len(seen)==total:break
            if len(seen)>total or not response['announcements']:
                raise ValueError('权益公告目录数量不能对账')
        else:raise ValueError('权益公告分页超限，不以截断目录证明无分红')
    return list(found.values())


class CorporateBook:
    def __init__(self,root=HOME):self.root=Path(root)

    def read(self):
        path=self.root/'catalog.json'
        if not path.exists():return {'version':VERSION,'actions':[],'coverage':{}}
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('fingerprint')!=canonical_hash({k:v for k,v in r.items() if k!='fingerprint'}) or r['version']!=VERSION:
            raise ValueError('权益目录指纹或口径不符')
        return r

    def collect(self,starts,cutoff,budget_seconds=60):
        import requests
        from agent.analysis import write_log
        allowed={r['symbol'] for r in MARKETS['cn']['candidate_universe']}
        if not set(starts)<=allowed:raise ValueError('权益采集仅允许受控A股股票池')
        prior=self.read();actions={r['event_id']:r for r in prior['actions']}
        coverage=dict(prior['coverage']);deadline=monotonic()+budget_seconds
        with requests.Session() as session:
            for symbol,start in starts.items():
                previous=coverage.get(symbol,{})
                covered=previous.get('through_date',start)
                if (event_time(cutoff,'cn').date()-datetime.fromisoformat(covered).date()).days>90:
                    raise ValueError('公司行为目录覆盖中断超过90日，须先核对历史')
                since=(datetime.fromisoformat(start)-timedelta(days=370)).date().isoformat()
                documents=query_notices(session,symbol,since,cutoff,deadline)
                if len(documents)>16:raise ValueError('实施公告数量超出单轮受控上限')
                for document in documents:
                    identifier='cninfo:'+document['announcement_id']
                    # Re-fetch official bytes to detect revisions under an existing URL.
                    data=download_pdf(document['url'],session,deadline)
                    if len(data)>5_000_000:raise ValueError('权益实施公告超出专用大小上限')
                    pdf_hash=hashlib.sha256(data).hexdigest()
                    if identifier in actions and actions[identifier]['source']['pdf_sha256']!=pdf_hash:
                        raise ValueError('实施公告原件发生修订，停止改写权益')
                    if identifier in actions:continue
                    path=self.root/'pdfs'/(document['announcement_id']+'.pdf');path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(data)
                    extracted={**extract_notice(path),'pdf_sha256':pdf_hash}
                    write_log(path.with_suffix('.json'),extracted)
                    actions[identifier]=parse_notice(document,extracted,now().isoformat())
                coverage[symbol]={'from_date':start,'through_date':str(cutoff)[:10],
                                  'checked_at':now().isoformat(),'source':'CNINFO implemented notices + recent full title scan; reconciled page totals',
                                  'announcement_count':len(documents),'scope':'ordinary_A_distributions_and_forward_splits; full_title_scan_90d'}
        result={'version':VERSION,'actions':list(actions.values()),'coverage':coverage}
        if monotonic()>=deadline:raise TimeoutError('权益采集预算到期，未提交目录')
        write_log(self.root/'catalog.json',{**result,'fingerprint':canonical_hash(result)})
        return result
