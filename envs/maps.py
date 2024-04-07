# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apple spawn point
# ' ' is empty space

HARVEST_MAP = [
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
    "@ P   P      A    P AAAAA    P  A P  @",
    "@  P     A P AA    P    AAA    A  A  @",
    "@     A AAA  AAA    A    A AA AAAA   @",
    "@ A  AAA A    A  A AAA  A  A   A A   @",
    "@AAA  A A    A  AAA A  AAA        A P@",
    "@ A A  AAA  AAA  A A    A AA   AA AA @",
    "@  A A  AAA    A A  AAA    AAA  A    @",
    "@   AAA  A      AAA  A    AAAA       @",
    "@ P  A       A  A AAA    A  A      P @",
    "@A  AAA  A  A  AAA A    AAAA     P   @",
    "@    A A   AAA  A A      A AA   A  P @",
    "@     AAA   A A  AAA      AA   AAA P @",
    "@ A    A     AAA  A  P          A    @",
    "@       P     A         P  P P     P @",
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
]

CLEANUP_MAP = [
    "@@@@@@@@@@@@@@@@@@",
    "@RRRRRR     BBBBB@",
    "@HHHHHH      BBBB@",
    "@RRRRRR     BBBBB@",
    "@RRRRR  P    BBBB@",
    "@RRRRR    P BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR      BBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@RRRRR   P P BBBB@",
    "@HHHHH   P  BBBBB@",
    "@RRRRRR    P BBBB@",
    "@HHHHHH P   BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH    P  BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH  P P BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH      BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@@@@@@@@@@@@@@@@@@",
]

CLEANUP_10x10_SYM = [
    '@@@@@@@@@@',
    '@HH   P B@',
    '@RR    BB@',
    '@HH     B@',
    '@RR    BB@',
    '@HH P   B@',
    '@RR    BB@',
    '@HH     B@',
    '@RRP   BB@',
    '@@@@@@@@@@']

# 7x7 map: Agent 0 on river side, Agent 1 on apple side
CLEANUP_SMALL_SYM = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']
