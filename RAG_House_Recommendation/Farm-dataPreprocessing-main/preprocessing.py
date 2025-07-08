def convert_price(s):
    import re

    s0 = str(s).replace(',', '').strip()
    # ① '합의가' 또는 '협의가' → '협의가'
    if s0 in ['합의가', '협의가']:
        return '협의가'
    # ② 그 외 '협의' 포함 → '협의가'
    if '협의' in s0:
        return '협의가'
    # ③ '.' 또는 '000' → '협의가'
    if s0 in ['.', '000']:
        return '협의가'
    # ④ 온전히 숫자만 있을 경우 → 정수로 변환
    if s0.isdigit():
        return int(s0)

    total = 0
    # ‘억’ 단위
    m = re.search(r'(\d+\.?\d*)억', s0)
    if m:
        total += float(m.group(1)) * 100_000_000
    s1 = re.sub(r'\d+\.?\d*억', '', s0)
    # ‘만’ 단위
    m = re.search(r'(\d+\.?\d*)만', s1)
    if m:
        total += float(m.group(1)) * 10_000
    s2 = re.sub(r'\d+\.?\d*만', '', s1)
    # 남은 숫자(원 단위 등)
    for d in re.findall(r'\d+', s2):
        total += int(d)

    # 단위 변환 후 값이 있으면 int, 그렇지 않으면 '협의가'
    return int(total) if total > 0 else '협의가'


def normalize_address(addr):
    s = str(addr).strip()
    if s.startswith('전라남도'):
        return s
    elif s.startswith('전남'):
        # '전남' → '전라남도'
        return '전라남도' + s[len('전남'):]
    else:
        # 도 단위 누락 시 맨 앞에 '전라남도 ' 추가
        return '전라남도 ' + s
