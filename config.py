"""配置参数"""

# 数据配置
VOCAB_SIZE = 10000
MAX_LENGTH = 128
TRAIN_SIZE = 20000  # 增加数据量
TEST_SIZE = 2000

# 模型配置
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
NUM_CLASSES = 2

# 训练配置
BATCH_SIZE = 32
EPOCHS = 5  # 增加训练轮数
LR = 3e-4   # 提高学习率
