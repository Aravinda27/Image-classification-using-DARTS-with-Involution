from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'invo2d_3x3',
    'invo2d_5x5',
    'dil_invo2d_3x3',
    'dil_invo2d_5x5'
]

NASNet = Genotype(
  normal = [
    ('invo2d_5x5', 1),
    ('invo2d_3x3', 0),
    ('invo2d_5x5', 0),
    ('invo2d_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('invo2d_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('invo2d_5x5', 1),
    ('invo2d_7x7', 0),
    ('max_pool_3x3', 1),
    ('invo2d_7x7', 0),
    ('avg_pool_3x3', 1),
    ('invo2d_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('invo2d_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('invo2d_3x3', 0),
    ('invo2d_5x5', 2),
    ('invo2d_3x3', 0),
    ('avg_pool_3x3', 3),
    ('invo2d_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('invo2d_3x3', 1),
    ('max_pool_3x3', 0),
    ('invo2d_7x7', 2),
    ('invo2d_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('invo2d_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('invo2d_3x3', 1), ('invo2d_3x3', 0), ('skip_connect', 0), ('invo2d_3x3', 1), ('skip_connect', 0), ('invo2d_3x3', 1), ('invo2d_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('invo2d_3x3', 0), ('invo2d_3x3', 1), ('invo2d_3x3', 0), ('invo2d_3x3', 1), ('invo2d_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_invo2d_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
DARTS_V3 = Genotype(normal=[('invo2d_3x3', 0), ('invo2d_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('invo2d_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('invo2d_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_96 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_81 = Genotype(normal=[('invo2d_3x3', 0), ('invo2d_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('invo2d_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('invo2d_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_86 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
DARTS = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
