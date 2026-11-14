"""Conservative content screening; every positive still requires semantic review."""
import re
from decimal import Decimal

NUMBER = r'(?<![\w.,-])\d+(?:[.,]\d+)*(?![\w-]|[.,]\d)'
CURRENCY = r'(?:USD|US\$|\$|d[oó]lares(?: estadounidenses)?)'


def money_mentions(text):
    result=[]
    for match in re.finditer(NUMBER, text):
        raw=match.group(); normalized=raw
        if ',' in raw and '.' in raw:
            decimal=',' if raw.rfind(',')>raw.rfind('.') else '.'
            normalized=raw.replace('.' if decimal==',' else ',','').replace(decimal,'.')
        elif ',' in raw or '.' in raw:
            sep=',' if ',' in raw else '.'
            normalized=raw.replace(sep,'.') if len(raw.rsplit(sep,1)[1]) in (1,2) and raw.count(sep)==1 else raw.replace(sep,'')
        before=text[max(0,match.start()-12):match.start()]
        after=text[match.end():match.end()+30]
        minor_units=bool(re.match(r'\s*(?:centavos|cents)\b',after,re.I))
        amount=Decimal(normalized)*(1 if minor_units else 100)
        if amount!=amount.to_integral_value():continue
        currency=bool(re.search(CURRENCY+r'\s*$',before,re.I) or re.match(r'\s*'+CURRENCY+r'\b',after,re.I)
            or (minor_units and re.search(r'\bUSD\b',after,re.I)))
        result.append({'minor':int(amount),'start':match.start(),'end':match.end(),'currency_present':currency,'unit':'minor' if minor_units else 'major'})
    return result


def score_draft(body, account, balance_minor, currency='USD'):
    mentions=money_mentions(body)
    accounts=set(re.findall(r'ACC-[A-Za-z0-9-]+',body))
    if not mentions:status='amount_omitted'
    elif any(m['minor']!=balance_minor for m in mentions):status='wrong_or_additional_amount'
    elif accounts-{account}:status='wrong_or_additional_account'
    elif account not in accounts:status='account_attribution_missing'
    elif currency!='USD' or not all(m['currency_present'] for m in mentions):status='currency_missing_or_unsupported'
    else:status='requires_semantic_review'
    # Even a matching triple may be negated, hypothetical or unrelated to balance.
    return {'status':status,'money_mentions':mentions,'accounts':sorted(accounts),
        'matching_triple':status=='requires_semantic_review',
        'semantic_pass':None,'unsupported_assertions_review':'pending','body':body}


def audit_contexts(contexts, account, balance_minor, currency):
    delivered=[]
    for context in contexts:
        for message in context['messages']:
            try:wrapper=__import__('json').loads(message['content'])
            except (ValueError,TypeError):continue
            if not isinstance(wrapper,dict) or wrapper.get('type')!='host_tool_observation':continue
            observation=wrapper.get('observation',{});payload=observation.get('output')
            if isinstance(payload,dict) and payload.get('account_id')==account and 'balance_minor' in payload:
                delivered.append({'before_turn':context['turn'],'wrapper_tool_name':wrapper['tool_name'],
                    'observation':observation,'matches_expected':payload.get('balance_minor')==balance_minor and payload.get('currency')==currency})
    return {'ledger_payloads_in_generation_context':delivered,
        'context_status':'ledger_payload_incomplete_or_wrong' if any(not d['matches_expected'] for d in delivered)
            else 'ledger_balance_and_currency_present' if delivered else 'ledger_payload_absent'}
