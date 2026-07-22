"""Constants for Ornith 35B MoE parameter optimization.

All magic numbers and configuration values are defined here for easy tuning.
"""

# Model Architecture
N_LAYERS = 28
N_EXPERTS_PER_LAYER = 256
N_ACTIVE_EXPERTS = 8
TOTAL_EXPERTS = N_LAYERS * N_EXPERTS_PER_LAYER  # 7168

# Expert Sizes (MB)
# Based on ~13M params per expert with GGUF overhead
EXPERT_SIZE_INT4_MB = 62.0
EXPERT_SIZE_INT8_MB = 124.0

# Memory Configuration
MAX_RAM_USAGE_RATIO = 0.5  # Use up to 50% of RAM for cache
MIN_LRU_SIZE_MB = 512  # Minimum LRU cache size (was 1024, reduced for flexibility)
MB_TO_BYTES = 1024 * 1024
GB_TO_MB = 1024

# Performance Thresholds
MIN_USABLE_TPS = 3.0  # Minimum tokens/sec for "usable" experience
TARGET_TPS_M2 = 1.0  # Target for M2
TARGET_TPS_PC = 2.0  # Target for PC

# Quantization Quality Factors
QUALITY_INT4 = 0.98  # ~2% accuracy loss
QUALITY_INT8 = 0.995  # ~0.5% accuracy loss

# Simulation Parameters
DEFAULT_PROMPT_LEN = 128
DEFAULT_GEN_LEN = 50
DEFAULT_POWER_LAW = 1.5  # Realistic MoE routing distribution

# Realistic Hit Rate Assumptions (from test results)
REALISTIC_HIT_RATE_M2_COLD = 0.05  # 5% cold start
REALISTIC_HIT_RATE_M2_WARM = 0.12  # 12% with hot-store
REALISTIC_HIT_RATE_PC_COLD = 0.06  # 6% cold start
REALISTIC_HIT_RATE_PC_WARM = 0.15  # 15% with hot-store

# Power-Law Target Distribution (empirical MoE data)
TARGET_TOP10_SHARE = 0.25  # Top 10 experts should handle ~25% of traffic
TARGET_TOP50_SHARE = 0.125  # Top 50 experts should handle ~12.5% of traffic

# Test Trace Sizes
TRACE_SIZE_SMALL = 2000  # For quick cache tests
TRACE_SIZE_MEDIUM = 3000  # For power-law tests
TRACE_SIZE_LARGE = 5000  # For comprehensive tests

# Hot-Store Sizes to Test
HOT_STORE_SIZES = [0, 8, 16, 32, 50, 64]

# Power-Law Exponents to Test
POWER_LAW_EXPONENTS = [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

# SSD Bandwidths to Test (GB/s)
SSD_BANDWIDTHS_GBS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]

# Cache Sizes to Test (MB)
CACHE_SIZES_MB = [2048, 3072, 4096, 5120, 6144, 8192]
