/*****************************************************************************
 * Copyright (C) 2018 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Keren Luzon <kluzon@gaudilabs.com>
 ******************************************************************************
 */

// clang-format off

/*
 * NOTE: This file is parsed automatically. Please use the following structure:
 *
 *  Kernel name: kernel1, kernel2, kernel3, etc..
 *
 * IMPORTANT:Do not write anything after "Kernel name: " that is not a kernel guid.
 *
 * namespace ns_someNamespace
 * {
 *      struct Params
 *      {
 *          ...
 *      };
 * }
 *
 *  Only use 1 struct per namespace. Inheritance is ok, but not multiple inheritance.
 */

#ifndef PERF_LIB_LAYER_PARAMS_H
#define PERF_LIB_LAYER_PARAMS_H

#define ASSERT_SIZE_IS_GREATER(P1, P2) \
static_assert(sizeof(P1) > sizeof(P2), "sizeof(" #P1 ") > sizeof(" #P2 ")")

typedef union
{
    float f;
    int i;
} fint_t;

enum EPoolingConvention
{
    POOLING_CONVENTION_VALID = 0,
    POOLING_CONVENTION_FULL = 1,
    POOLING_CONVENTION_SAME = 2,
    POOLING_CONVENTION_FULL_PYTORCH = 3
};
typedef enum _UpsampleType_t
{
    UPSAMPLE_TYPE_BILINEAR = 0,
    UPSAMPLE_TYPE_NEAREST_NEIGHBOR = 1,
    UPSAMPLE_TYPE_TRILINEAR = 2
} EUpsampleType_t;

typedef enum _SparseLengthsSumType_t
{
    EMBEDDED_SC_ZP = 0,
    SEPARATE_SC_ZP
} ESparseLengthsSumType_t;

typedef enum _GridGeneratorType_t
{
    GRID_GENERATOR_TYPE_AFFINE = 0,
    GRID_GENERATOR_TYPE_WARP
} EGridGeneratorType_t;

typedef enum _CrossEntropyMode_t
{
    CROSS_ENTROPY_MODE_SUM = 0,
    CROSS_ENTROPY_MODE_MEAN = 1,
    // CROSS_ENTROPY_MODE_NO_REDUCTION mode is only applicable in fwd non-sparse
    // (dense) version
    CROSS_ENTROPY_MODE_NO_REDUCTION = 2
} ECrossEntropyMode_t;

typedef enum _CastF32RoundMode_t
{
    CAST_ROUND_HALF_NE = 0,
    CAST_ROUND_DOWN = 1,
    CAST_ROUND_UP = 2,
    CAST_ROUND_SR = 3,
    CAST_ROUND_ZERO = 4,
    // CAST_ROUND_DEFAULT = 5, /* depricated */
    CAST_ROUND_HALF_AZ = 6,
    // Place holder for SR_RNE since it is supported in HW.
    //CAST_ROUND_SR_RNE = 7,
    CAST_ROUND_SFTZ = 8
} CastF32RoundMode_t;

typedef enum _CastSatMode_t
{
    CAST_NO_CLIP_NO_SAT = 0,
    CAST_CLIP = 1,
    CAST_SAT = 2,
    CAST_TRUNC = 3
} CastSatMode_t;

typedef enum _ResizeInterpolationMode_t
{
    RESIZE_INTER_NEAREST = 0,
    RESIZE_INTER_LINEAR = 1,
    RESIZE_INTER_CUBIC = 2,
    RESIZE_INTER_AREA = 3,
    RESIZE_INTER_GAUSSIAN = 4
} ResizeInterpolationMode_t;

typedef enum _GridSampleInterpolation_t
{
    SAMPLE_NEAREST = 0,
    SAMPLE_BILINEAR = 1,
    SAMPLE_CUBIC = 2
} GridSampleInterpolation_t;

typedef enum _GridSamplePad_t
{
    PAD_ZEROS = 0,
    PAD_BORDER = 1,
    PAD_REFLECTION = 2
} GridSamplePad_t;

typedef enum _ResizeNearestMode_t
{
    // versions older than 11 did not have nearest_mode attr.
    // ROUND_DEFAULT is used to maintain backward compatibility
    ROUND_DEFAULT = 0,
    ROUND_PREFER_FLOOR = 1,
    ROUND_PREFER_CEIL = 2,
    FLOOR = 3,
    CEIL = 4
} ResizeNearestMode_t;

typedef enum _ResizeCoordinateTransformationMode_t
{
    HALF_PIXEL_MODE = 0,
    PYTORCH_HALF_PIXEL_MODE = 1,
    ALIGN_CORNERS_MODE = 2,
    ASYMMETRIC_MODE = 3,
    TF_HALF_PIXEL_FOR_NN_MODE = 4
} ResizeCoordinateTransformationMode_t;

typedef enum _RoiAlignMode_t
{
    ROI_ALIGN_AVG = 0,
    ROI_ALIGN_MAX = 1
} RoiAlignMode_t;

typedef enum _SortDirection_t
{
    SORT_ASCENDING = 0,
    SORT_DESCENDING = 1,
} SortDirection_t;

typedef enum _CropAndResizeMode_t
{
    CROP_AND_RESIZE_MODE_BILINEAR = 0,
    CROP_AND_RESIZE_MODE_NEAREST = 1
} CropAndResizeMode_t;

typedef enum
{
    EMBEDDING_BAG_MODE_SUM = 0,
    EMBEDDING_BAG_MODE_MEAN
} EmbeddingBagMode_t;

typedef enum
{
    BINARY_WITH_ALPHA_MODE_ADD = 0,
    BINARY_WITH_ALPHA_MODE_SUB = 1,
    BINARY_WITH_ALPHA_MODE_RSUB = 2,
    BINARY_WITH_ALPHA_MODE_CMUL = 3,
    BINARY_WITH_ALPHA_MODE_CDIV = 4
} BinaryWithAlphaMode_t;

typedef enum
{
    LEFT = 0,
    RIGHT = 1
} ShiftDir_t;

typedef enum
{
    ARITHMETIC = 0,
    LOGICAL
} ShiftMode_t;

typedef enum
{
    NLL_LOSS_MODE_MEAN = 0,
    NLL_LOSS_MODE_SUM,
    NLL_LOSS_MODE_NONE
} NLLLossMode_t;

typedef enum
{
    MSE_LOSS_REDUCTION_MODE_MEAN = 0,
    MSE_LOSS_REDUCTION_MODE_SUM,
    MSE_LOSS_REDUCTION_MODE_NONE
} MSELossMode_t;

typedef enum
{
    LOSS_REDUCTION_MODE_MEAN = 0,
    LOSS_REDUCTION_MODE_SUM,
    LOSS_REDUCTION_MODE_NONE
} LossMode_t;

typedef enum _RoundMode_t
{
    ROUND_HALF_AWAY_FROM_ZERO = 0,
    ROUND_HALF_NEAREST_EVEN = 1,
} RoundMode_t;

typedef enum {
  DIV_ROUND_NONE = 0,
  DIV_ROUND_TRUNC = 1,
  DIV_ROUND_FLOOR = 2
} DivRoundMode_t;

typedef enum _PadMode_t
{
    PAD_MODE_CONSTANT = 0,
    PAD_MODE_REFLECT,
    PAD_MODE_EDGE,
    PAD_MODE_SYMMETRIC
} PadMode_t;

typedef enum _ColorSpaceMode_t
{
    RGB_TO_YCBCR = 0,
    RGB_TO_BGR,
    YCBCR_TO_RGB,
    YCBCR_TO_BGR,
    BGR_TO_RGB,
    BGR_TO_YCBCR,
    GRAY_TO_RGB,
    GRAY_TO_BGR,
    GRAY_TO_YCBCR,
    RGB_TO_GRAY,
    YCBCR_TO_GRAY,
    BGR_TO_GRAY,
} ColorSpaceMode_t;

typedef enum _SpatialCorrelation_t
{
    SPATIAL_CORRELATION_TYPE_NEAREST_NEIGHBOR = 0,
    SPATIAL_CORRELATION_TYPE_BILINEAR,
} SpatialCorrelation_t;

typedef enum _ExpandIntoJaggedIndices_t
{
    EXPAND_JAGGED_INDICES_WITH_NO_INCREMENT = 0,
    EXPAND_JAGGED_INDICES_WITH_INCREMENT,
} ExpandIntoJaggedIndices_t;

typedef enum
{
    RIGHT_RIGHT = 0,
    RIGHT_LEFT = 1,
    LEFT_RIGHT = 2,
    LEFT_LEFT = 3,
} DiagAlign_t;

typedef enum _RadixSortKeyType_t
{
    RADIX_SORT_KEY_UNSIGNED_INT_ASCENDING = 0,
    RADIX_SORT_KEY_UNSIGNED_INT_DESCENDING,
    RADIX_SORT_KEY_SIGNED_INT_ASCENDING,
    RADIX_SORT_KEY_SIGNED_INT_DESCENDING,
    RADIX_SORT_KEY_FLOAT_ASCENDING,
    RADIX_SORT_KEY_FLOAT_DESCENDING,
    RADIX_SORT_KEY_BF16_ASCENDING,
    RADIX_SORT_KEY_BF16_DESCENDING,
    RADIX_SORT_KEY_F16_ASCENDING,
    RADIX_SORT_KEY_F16_DESCENDING,
    RADIX_SORT_KEY_I16_ASCENDING,
    RADIX_SORT_KEY_I16_DESCENDING,
    RADIX_SORT_KEY_U16_ASCENDING,
    RADIX_SORT_KEY_U16_DESCENDING,
} RadixSortKeyType_t;

typedef enum _RadixSortFlavor_t
{
    KEY_METADATA_IN_KEY_METADATA_OUT = 0,
    KEY_IN_KEY_METADATA_OUT,
    KEY_IN_KEY_OUT,
    KEY_METADATA_IN_KEY_METADATA_OUT_VALID_COUNT,
    KEY_IN_KEY_METADATA_OUT_VALID_COUNT,
    KEY_IN_KEY_OUT_VALID_COUNT,
    KEY_METADATA_IN_KEY_METADATA_OUT_TOPK,
    KEY_IN_KEY_METADATA_OUT_TOPK,
    KEY_IN_KEY_OUT_TOPK,
    KEY_METADATA_IN_KEY_METADATA_OUT_VALID_COUNT_TOPK,
    KEY_IN_KEY_METADATA_OUT_VALID_COUNT_TOPK,
    KEY_IN_KEY_OUT_VALID_COUNT_TOPK,
} RadixSortFlavor_t;

typedef enum _KTensorType_t
{
    K_TENSOR_NONE = 0,
    K_TENSOR_SHAPE
} KTensorType_t;
typedef enum _ScatterNdUpdateMode_t
{
    // behaviour is undefined in case of duplicate indices
    SCATTER_ND_UPDATE_MODE_NON_DETERMINISTIC = 0,
    // behaviour is defined in case of duplicate indices (last update is applied)
    SCATTER_ND_UPDATE_MODE_DETERMINISTIC,
} ScatterNdUpdateMode_t;

typedef enum _PosWeightMode_t
{
    POS_WEIGHT_DISABLE = 0,
    POS_WEIGHT_ENABLE,
} PosWeightMode_t;

typedef enum _InterpolationMethod_t
{
    NEAREST_NEIGHBOR = 0,
    BILINEAR
} InterpolationMethod_t;

typedef enum _FillMode_t
{
    CONSTANT = 0,
    REFLECT,
    WRAP,
    NEAREST
} FillMode_t;

typedef enum _BatchNormMode_t
{
    BN_MODE_NON_DETERMINISTIC = 0,
    BN_MODE_DETERMINISTIC = 1,
    BN_MODE_DETERMINISTIC_TPC_ID = 2
} BatchNormMode_t;

typedef enum _ShrinkMode_t
{
    HARD_SHRINK = 0,
    SOFT_SHRINK = 1
} ShrinkMode_t;

typedef enum _BinCountMode_t
{
    //bit0 -> "size" tensor, bit1 -> "weights" tensor, bit2 ->"shape tensor"
    NO_WEIGHT = 1,
    USE_WEIGHT = 3,
    NO_WEIGHT_DSHAPE = 4,
    USE_WEIGHT_DSHAPE = 6
} BinCountMode_t;

typedef enum _KLDivMode_t
{
    KLDIV_LOSS_MODE_SUM = 0,
    KLDIV_LOSS_MODE_MEAN =  1,
    KLDIV_LOSS_MODE_NONE = 2
} KLDivMode_t;

typedef enum _RandomPoissonFlavor_t
{
    WITHOUT_DIST    = 0,
    WITH_DIST       = 1
} RandomPoissonFlavor_t;

typedef enum _BatchNormRoundingMode_t
{
    BN_ROUND_MODE_RHNE = 0,
    BN_ROUND_MODE_RZ   = 1,
    BN_ROUND_MODE_RU   = 2,
    BN_ROUND_MODE_RD   = 3,
    BN_ROUND_MODE_SR   = 4,
    BN_ROUND_MODE_RHAZ = 6
} BatchNormRoundingMode_t;

typedef enum _DataTypeSize_t
{
    DATA_8BIT  = 0,
    DATA_16BIT = 1,
    DATA_32BIT = 2
} DataTypeSize_t;

typedef enum _TypeOfP_t
{
    TYPE_P_IS_INT  = 0,
    TYPE_P_IS_FLOAT = 1
} TypeOfP_t;

typedef enum _ScatterIndicesSizeControl_t
{
    CONTROL_IND_SIZE = 0,
    NO_CONTROL_IND_SIZE = 1,
} ScatterIndicesSizeControl_t;

typedef enum _ScatterWrapIndices_t
{
    IGNORE_NEG_INDICES = 0,
    WRAP_NEG_INDICES = 1
} ScatterWrapIndices_t;

typedef enum _ScatterReduceMode_t
{
    SCATTER_REDUCE_SUM,
    SCATTER_REDUCE_PROD,
    SCATTER_REDUCE_MEAN,
    SCATTER_REDUCE_AMAX,
    SCATTER_REDUCE_AMIN
} ScatterReduceMode_t;

typedef enum _BoundsCheckMode_t
{
    // Raise an exception (CPU) or device-side assert (HPU)
    BCM_FATAL = 0,
    // Log the first out-of-bounds instance per kernel, and set to zero.
    BCM_WARNING = 1,
    // Set to zero.
    BCM_IGNORE = 2,
    // No bounds checks.
    BCM_NONE = 3
} BoundsCheckMode_t;

typedef enum _QRMode_t {
    REDUCED = 0,
    COMPLETE = 1,
    R = 2
} QRMode_t;

typedef enum _RotaryPosEmbeddingMode_t
{
    // LLaMA (First version)
    ROTARY_POS_EMBEDDING_MODE_BLOCKWISE = 0,
    // GPT - J
    ROTARY_POS_EMBEDDING_MODE_PAIRWISE = 1
} RotaryPosEmbeddingMode_t;

typedef enum _ScaledMaskedSoftmaxExpMode
{
    USE_LUT = 0,
    NO_LUT = 1,
    EXPERIMENTAL = 15
} ScaledMaskedSoftmaxExpMode_t;

typedef enum _ScaledMaskedTriangularSoftmaxMode
{
    DEFAULT_MODE                =   0,
    SLICED_MODE                 =   0x1 << 0,
    HF8_1B_MODE                 =   0x1 << 1,
    HF8_1C_MODE                 =   0x1 << 2,
    FUSED_MULT_INV_ATTN         =   0x1 << 3,
    SQUEEZED_FCD_RETAINED       =   0x1 << 4,
    CL_ALIGNED_PADDED_RETAINED  =   0x1 << 5,
    ACTUAL_SEQ_LEN_LEFT_PADDED  =   0x1 << 6,
    ACTUAL_SEQ_LEN_RIGHT_PADDED =   0x1 << 7,
    SINGLE_SLICE_MODE           =   0x1 << 8,
    FUSED_MULT_INV_ATTN_PARAM   =   0x1 << 9,
    SMT_FUSED_ADD               =   0x1 << 10
} ScaledMaskedTriangularSoftmaxMode_t;

typedef enum _ScaledMaskedTriangularSoftmaxBwdMode
{
    SMT_BWD_DEFAULT_MODE                =   0,
    SMT_BWD_SLICED_MODE                 =   0x1 << 0,
    SMT_BWD_GRAD_IN_FUSED_MULT          =   0x1 << 1,
    SMT_BWD_FA_V2                       =   0x1 << 2,
    SMT_BWD_FA_V2_FULL_CALC             =   0x1 << 3
} ScaledMaskedTriangularSoftmaxBwdMode_t;

typedef enum _CdistComputeMode
{
    USE_MM_FOR_EUCLID_DIST_IF_NECESSARY = 0,
    USE_MM_FOR_EUCLID_DIST = 1,
    DONOT_USE_MM_FOR_EUCLID_DIST = 2
} CdistComputeMode_t;

typedef enum _RmsNormBwdMode
{
    DEFAULT_RMS_NORM_BWD_MODE         = 0,
    STATIC_CASE_WIDTH_PARTITIONING    = 0x1 << 0,
    SQUASH_BATCH_DIMS                 = 0x1 << 1
} RmsNormBwdMode_t;

typedef enum _RetainInvRmsMode
{
    DEFAULT_RETAIN_INV_RMS              =   0,
    NO_RETAIN_INV_RMS                   =   1,
    SQUEEZED_RETAIN_INV_RMS             =   2,
    CL_ALIGNED_SQUEEZED_RETAIN_INV_RMS  =   3
} RetainInvRmsMode_t;

typedef enum _RepeatScalarAPIMode
{
    VPU_LD_FCD_REPEAT = 0,
    SCALAR_LD_FCD_REPEAT_NO_PARTIAL_WR = 1
} RepeatMode_t;

typedef enum _SoftmaxMode_t
{
    DEFAULT_SOFTMAX             =   0,
    SOFTMAX_HF8_1B              =   0x1 << 0,
    SOFTMAX_HF8_1C              =   0x1 << 1,
    SQUEEZED_RETAINED           =   0x1 << 4,
    CL_ALIGNED_RETAINED         =   0x1 << 5,
    FUSED_ADD                   =   0x1 << 6,
    PREFER_RECIP_NO_LUT         =   0x1 << 7,
    SOFTMAX_HF8_2B              =   0x1 << 8,
    RETAINED_RECIP_SINGLE       =   0x1 << 9,
    RETAINED_RECIP_DUAL         =   0x1 << 10,
    FUSED_MUL_INV_ATTN_PARAM    =   0x1 << 11
} SoftmaxMode_t;

typedef enum _SoftmaxFA2Mode_t
{
    FA_DEFAULT_SOFTMAX             =   0,
    FA_SOFTMAX_HF8_1B              =   0x1 << 0,
    FA_SOFTMAX_HF8_1C              =   0x1 << 1,
    FA_SQUEEZED_RETAINED           =   0x1 << 2, // to be removed and in its place "FA_RESERVED"
    FA_CL_ALIGNED_RETAINED         =   0x1 << 3,
    FA_FUSED_ADD                   =   0x1 << 4,
    FA_SOFTMAX_HF8_2B              =   0x1 << 5,
    FA_RETAINED_RECIP_SINGLE       =   0x1 << 6,
    FA_RETAINED_RECIP_DUAL         =   0x1 << 7,
    FA_FUSED_MUL_INV_ATTN_PARAM    =   0x1 << 8,
    FA_ACTUAL_SEQ_LEN_LEFT_PADDED  =   0x1 << 9,
    FA_ACTUAL_SEQ_LEN_RIGHT_PADDED =   0x1 << 10,
    FA_FUSED_MUL_INV_ATTN_TNSR     =   0x1 << 11,
    FA_ASL_5D_ALIGNED              =   0x1 << 12,
    FA_NO_IMPLICIT_SCALE           =   0x1 << 13,
    FA_NO_M_L_INPUTS               =   0x1 << 14,
} SoftmaxFA2Mode_t;

/*
 * Separate softmax mode definition for SDPA to keep it independent of
 * _SoftmaxMode_t so that any other mode can be added.
 */
typedef enum _SdpaSoftmaxMode_t
{
    SDPA_DEFAULT_SOFTMAX = 0,
    SDPA_SOFTMAX_HF8_1B = 1,
    SDPA_SOFTMAX_HF8_1C = 2,
    SDPA_FAST_SOFTMAX   = 3,
    SDPA_FP32_SOFTMAX   = 4,
} SdpaSoftmaxMode_t;

typedef enum _SdpaFlags_t
{
    SDPA_FLAGS_AMAX_S = 1<<0,
    SDPA_FLAGS_AMAX_O = 1<<1,
    SDPA_FLAGS_AMAX_dS = 1<<2,

    SDPA_FLAGS_D_SCALE_Q = 1<<8,
    SDPA_FLAGS_D_SCALE_K = 1<<9,
    SDPA_FLAGS_D_SCALE_V = 1<<10,
    SDPA_FLAGS_Q_SCALE_S = 1<<11,
    SDPA_FLAGS_Q_SCALE_O = 1<<12,
    SDPA_FLAGS_D_SCALE_S = 1<<13,
    SDPA_FLAGS_D_SCALE_dO = 1<<14,
    SDPA_FLAGS_D_SCALE_dS = 1<<15,
    SDPA_FLAGS_Q_SCALE_dS = 1<<16,
    SDPA_FLAGS_FINAL_BLOCK = 1<<23,
    SDPA_FLAGS_VALID_SEQ_LEN_PRESENT = 1 << 24,
    SDPA_FLAGS_SEQ_PADDING_LEFT = 1 << 25,
    SDPA_FLAGS_SEQ_PADDING_RIGHT = 1 << 26,

    SDPA_FLAGS_Q_SLICING_ON = 1 << 28,
} SdpaFlags_t;

typedef enum _UpperTriangularMaskArea_t
{
    DEFAULT_LOWER_TRIANGLE_MASK_OUT = 0,
    UPPER_TRIANGLE_MASK_OUT = 1,
} UpperTriangularMaskArea_t;

typedef enum _MemcpyMode
{
    MEMCPY_DEFAULT = 0,
    MEMCPY_1D = 1,
    MEMCPY_IRF44 = 2,
} MemcpyMode_t;

typedef enum _SigmoidFlavor
{
    SIGMOID_DEFAULT = 0,
    NO_SATURATION_SIGMOID = 1,
} SigmoidFlavor_t;

typedef enum _PositiveReciprocalFlavor
{
    POS_RECIPROCAL_FAST_DEFAULT = 0,
    POS_RECIPROCAL_ACCURATE = 1,
} PositiveReciprocalFlavor_t;

typedef enum _FakeQuantizeFp4Round
{
    FP4_SR_AND_SFTZ = 0,
    FP4_RNE = 1,
}FakeQuantizeFp4Round;

typedef enum _FakeQuantizeFp4IntermediateDtype
{
    FP4_130 = 0,
    FP4_121 = 1,
}FakeQuantizeFp4IntermediateDtype;

typedef enum
{
    MOE_ACTIVATION_MODE_SELU = 0,
    MOE_ACTIVATION_MODE_GELU = 1,
    MOE_ACTIVATION_MODE_RELU = 2,
    MOE_ACTIVATION_MODE_SILU = 3,
    MOE_ACTIVATION_MODE_RELU2 = 4,
} MoeActivationMode_t;

typedef enum _MoeFlags_t
{
    MOE_FLAGS_FUSED_GEMM = 1<<0,
    MOE_FLAGS_PERMUTED_WEIGHTS = 1<<1,
    MOE_FLAGS_CALC_AMAX = 1<<2,
    MOE_FLAGS_DYNAMIC_SCALE = 1<<3,
    MOE_FLAGS_BLOCKWISE_WEIGHT_QUANTIZATION = 1<<4,
    MOE_FLAGS_CALC_AMAX_FIRST_GEMM_INPUT = 1<<5,
    MOE_FLAGS_FP8_HYBRID = 1<<6,
    MOE_FLAGS_SCALED_SWIGLU = 1<<7,
    MOE_FLAGS_GPT_SWIGLU = 1<<8,
    MOE_FLAGS_PERMUTED_WEIGHT_SCALES = 1<<9,
    MOE_FLAGS_NON_GATED = 1<<10,
    MOE_FLAGS_GROUP_INDEX_QUANTIZATION = 1<<11,
} MoeFlags_t;

typedef enum _RouterMode_t
{
    ROUTER_DEFAULT              = 1,
    ROUTER_WITH_SCALES          = 2,
    ROUTER_WITH_WEIGHTS         = 3,
    ROUTER_METADATA_ONLY        = 4,
    ROUTER_METADATA_AND_SCALES  = 5,
    ROUTER_METADATA_AND_WEIGHTS = 6,
    ROUTER_MAX_MODE                 //! ILLEGAL mode, used ONLY for BOUND CHECKING
} RouterMode_t;

typedef enum _IndexReduceMode
{
    INDEX_REDUCE_AMAX = 0,
    INDEX_REDUCE_AMIN = 1,
    INDEX_REDUCE_MEAN = 2,
    INDEX_REDUCE_PROD = 3,
} IndexReduceMode_t;

typedef enum _BlockSoftmaxAdjusterStage_t
{
    BLOCK_SOFTMAX_ADJUSTER_STAGE1 = 1,       // G3: max stage 1, G2: max stage
    BLOCK_SOFTMAX_ADJUSTER_STAGE2 = 2,       // G3: max stage 2, G2: sum stage
    BLOCK_SOFTMAX_ADJUSTER_STAGE3 = 3,       // G3: sum stage 1, G2: adjustment stage
    BLOCK_SOFTMAX_ADJUSTER_STAGE4 = 4,       // G3: sum stage 2, G2: Unused
    BLOCK_SOFTMAX_ADJUSTER_STAGE5 = 5,       // G3: adjustment,  G2: Unused
} BlockSoftmaxAdjusterStage_t;

typedef enum _BlockSoftmaxAdjustmentMode_t
{
    BLOCK_SOFTMAX_ADJUSTMENT_MODE_DEFAULT_MODE     = 0,        // run all stages and output in the same type as input
    BLOCK_SOFTMAX_ADJUSTMENT_MODE_HF8_OUT          = 0x1 << 0,
    BLOCK_SOFTMAX_ADJUSTMENT_MODE_FUSED_MULT_ATTN  = 0x1 << 1,
    BLOCK_SOFTMAX_ADJUSTMENT_MODE_MAX_SUM_STAGES   = 0x1 << 2, //
    BLOCK_SOFTMAX_ADJUSTMENT_MODE_ADJUSTMENT_STAGE = 0x1 << 3, // adjustment stage
} BlockSoftmaxAdjustmentMode_t;

typedef enum _BlockSoftmaxAdjusterMode_t
{
    BLOCK_SOFTMAX_ADJUSTER_MODE_DEFAULT_MODE    = 0,
    BLOCK_SOFTMAX_ADJUSTER_MODE_HF8_OUT         = 0x1 << 0,
    BLOCK_SOFTMAX_ADJUSTER_MODE_FUSED_MULT_ATTN = 0x1 << 1,
} BlockSoftmaxAdjusterMode_t;

typedef enum _BlockSoftmaxConstantMaxerStage_t
{
    BLOCK_SOFTMAX_CONSTANT_MAXER_STAGE1 = 1,       // G3: block softmax
    BLOCK_SOFTMAX_CONSTANT_MAXER_STAGE2 = 2,       // G3: reduce group sum
    BLOCK_SOFTMAX_CONSTANT_MAXER_STAGE3 = 3,       // G3: adjustment

} BlockSoftmaxConstantMaxerStage_t;

typedef enum _BlockSoftmaxConstantMaxMode_t
{
    BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_BF16_OUT = 0x1 << 0,
    BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_HF8_OUT  = 0x1 << 1,
    BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_USE_HF8_EXP_LUT = 0x1 << 2

} BlockSoftmaxConstantMaxMode_t;

typedef enum _BlockSoftmaxMode_t
{
    BLOCK_SOFTMAX_MODE_DEFAULT = 0, // Default output type is the attn type
    BLOCK_SOFTMAX_MODE_HF8_OUT = 0x1 << 0,
    BLOCK_SOFTMAX_MODE_USE_HF8_EXP_LUT = 0x1 << 1

} BlockSoftmaxMode_t;
typedef enum _BlockSoftmaxerMode_t
{
    BLOCK_SOFTMAXER_MODE_DEFAULT = 0, // Default output type is the attn type
    BLOCK_SOFTMAXER_MODE_HF8_OUT = 0x1 << 0,
    BLOCK_SOFTMAXER_MODE_USE_HF8_EXP_LUT = 0x1 << 1
} BlockSoftmaxerMode_t;

typedef enum _ScalarMergeInputs_t
{
    SCALAR_MERGE_INDEX_TENSOR       = 1 << 0,
    SCALAR_MERGE_VALID_COUNT_TENSOR = 1 << 1,
    SCALAR_MERGE_TOPK_TENSOR        = 1 << 2,
} ScalarMergeInputs_t;

typedef enum _IndexTypeMode_t
{
    LOCAL_INDEX = 0,
    GLOBAL_INDEX = 1
} IndexTypeMode_t;
/////////////////////////////// GENERAL_KERNELS ///////////////////////////////

/*
 * Kernel name: long_loop
 */
namespace ns_LongLoop
{
    struct Params
    {
        unsigned int count;
    };
} // namespace ns_LongLoop

/*
 * Kernel name: l2norm_i8, l2norm_i16
 */
namespace ns_L2normKernel
{
    struct Params
    {
        float epsilon;
    };
} // namespace ns_L2normKernel

/*
 * Kernel name: l2norm_stage1_f32, l2norm_stage1_bf16
 */
namespace ns_L2NormStage1Kernel
{
    struct Params
    {
        int dim1ChunksSize;
    };
} // namespace ns_L2NormStage1Kernel

/*
 * Kernel name: l2norm_stage2_f32, l2norm_stage2_bf16
 */
namespace ns_L2NormStage2Kernel
{
    struct Params
    {
        float epsilon;
        int dim1ChunksSize;
    };
} // namespace ns_L2NormStage2Kernel

/*
 * Kernel name: printf_test
 */
namespace ns_PrintfTestKernel
{
    struct Params
    {
        int int_val;
        unsigned int uint_val;
        float float_val;
        short short_val __attribute__((aligned(4)));
        unsigned short ushort_val __attribute__((aligned(4)));
        char char_val __attribute__((aligned(4)));
        unsigned char uchar_val __attribute__((aligned(4)));
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_PrintfTestKernel

/*
 * Kernel name:  printf_vpu_demo_i8, printf_vpu_demo_u8, printf_vpu_demo_i16,
 *               printf_vpu_demo_i32, printf_vpu_demo_f32, printf_vpu_demo_bf16
 */
namespace ns_PrintfVpuDemoKernel
{
    struct Params
    {
        fint_t val_0;
        int pos_0;
        fint_t val_1;
        int pos_1;
        fint_t val_2;
        int pos_2;
    };
} // namespace ns_PrintfVpuDemoKernel

/*
 * Kernel name: bincount_f32, bincount_i32
 */
namespace ns_BinCountKernel
{
    struct Params
    {
        BinCountMode_t bincount_mode;
    };
} // namespace ns_BinCountKernel

/*
 * Kernel name: bounds_check_indices_fwd_i32
 */
namespace ns_BoundsCheckIndicesKernel
{
    struct Params
    {
        BoundsCheckMode_t mode;
    };
} // namespace ns_BoundsCheckIndicesKernel

///////////////////////////// ELEMENTWISE_KERNELS /////////////////////////////

/*
 * Kernel name: binary_with_alpha_fwd_f32, binary_with_alpha_f32
 */
namespace ns_BinaryWithAlphaKernel
{
    struct Params
    {
        fint_t alpha;
        BinaryWithAlphaMode_t mode;
    };
} // namespace ns_BinaryWithAlphaKernel


/*
 * Kernel name: linear_bwd_f32
 */
namespace ns_LinearBwdKernel
{
    struct Params
    {
        bool gradBias; // compute gradient for bias or not

    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LinearBwdKernel


/*
 * Kernel name: embedding_dense_pt_bwd_f32
 */
namespace ns_EmbeddingDensePtBwdKernel
{
    struct Params
    {
        int num_weights;
        int padding_idx;
        bool scaleGradByFreq;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_EmbeddingDensePtBwdKernel


/*
 * Kernel name: embedding_renorm_fwd_f32
 */
namespace ns_EmbeddingRenormFwdKernel
{
    struct Params
    {
        double max_norm;
        double norm_type;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_EmbeddingRenormFwdKernel

/*
 * Kernel name: renorm
 */
namespace ns_RenormKernel
{
    struct Params
    {
        double p;
        int dim;
        double max_norm;
    };
} // namespace ns_RenormKernel


/*
 * Kernel name: rsqrt_fwd_f32
 */
namespace ns_RsqrtKernel
{
    struct Params
    {
        bool useNonLut; // there is NonLut based flavor for rsqrt_fwd_f32
                        // pros: VLM will be free
                        // cons: accuaracy and performance degraded
                        //default is false
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RsqrtKernel

/*
 * Kernel name: reciprocal_fwd_bf16
 */
namespace ns_ReciprocalKernel
{
    struct Params
    {
        bool useNonLut; // there is NonLut based flavor for reciprocal_fwd_bf16
                        // pros: when the user doesn't want to use LUT because of memory issues
                        // cons: performance degrades incomparison to LUT
                        //default is false
    };
    enum QuakeType_t
    {
        RECIPROCAL_QUAKE_POSITIVE,
        RECIPROCAL_QUAKE_NO_ZERO
    };
    struct ParamsV2: public Params
    {
        QuakeType_t quakeType;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_ReciprocalKernel

/*
 * Kernel name: positive_reciprocal_f32
 */
namespace ns_PositiveReciprocalKernel
{
    struct Params
    {
        PositiveReciprocalFlavor_t positiveRecipMode;
    };
} // namespace ns_PositiveReciprocalKernel

/*
 * Kernel name: div_mod_fwd_i16, div_mod_fwd_i8, div_mod_fwd_i32,
 *              div_mod_i8, div_mod_i16, div_mod_i32
 */
namespace ns_DivModKernel
{
    struct Params
    {
        bool isPyCompatible; // python div_mod requires remainder with the same sign of
                             // divisor except of zero remainder, unlike C++
                             // for backward compatibility, default is false
    };
    struct ParamsV2 : public Params
    {
        bool isTruncRoundingMode;  // Boolean to check whether the rounding_mode is 'trunc' or 'floor'
                                   // default is false, using floor mode
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_DivModKernel

/*
 * Kernel name: clamp_i8, clamp_i16, clamp_f32
 */
namespace ns_ClampKernel
{
    struct Params
    {
        fint_t upperBound;
        fint_t lowerBound;
    };
} // namespace ns_ClampKernel

/*
 * Kernel name: hardtanh_f32
 */
namespace ns_HardTanhKernel
{
    struct Params
    {
        fint_t upperBound;
        fint_t lowerBound;
    };
} // namespace ns_HardTanhKernel

/*
 * Kernel name: hard_sigmoid_f32
 */
namespace ns_HardSigmoidKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_HardSigmoidKernel

/*
 * Kernel name: relu_i8, relu_u8, relu_i16, relu_f32, relu_bf16
 */
namespace ns_ReluKernel
{
    struct Params
    {
        fint_t threshold;
    };
    struct ParamsV2 : public Params
    {
        fint_t replacementValue;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_ReluKernel

/*
 * Kernel name: selu_f32, selu_i16
 */
namespace ns_SeluKernel
{
    struct Params
    {
        float gamma;
        float alpha;
    };
    struct ParamsV2 : public Params
    {
        bool isInputFeaturemap;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_SeluKernel

/*
 * Kernel name: elu_i8, elu_i16, elu_f32
 */
namespace ns_EluKernel
{
    struct Params
    {
        float alpha;
    };
    struct ParamsV2 : public Params
    {
        bool isInputFeaturemap;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        float scale;
        float input_scale;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_EluKernel

/*
 * Kernel name: leakyrelu_i8, leakyrelu_i16, leakyrelu_f32
 */
namespace ns_LeakyReluKernel
{
    struct Params
    {
        double alpha;
    };
} // namespace ns_LeakyReluKernel

/*
 * Kernel name: batch_norm_i8,  batch_norm_relu_i8,
 *              batch_norm_i16, batch_norm_relu_i16,
 *              batch_norm_f32, batch_norm_relu_f32,
 */
namespace ns_BatchNormKernel
{
    struct Params
    {
        fint_t threshold; // threshold.i to use with .._i8 and .._i16 kernels
                          // threshold.f to use with .._f32 kernel
        float momentum;
        float epsilon;
    };
    struct ParamsV2 : public Params
    {
        bool isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BatchNormKernel

/*
 * Kernel name: batch_norm_variance_f32,  batch_norm_variance_bf16
 */
namespace ns_BatchNormVarienceKernel
{
    struct Params
    {
        fint_t threshold; // threshold.i to use with .._i8 and .._i16 kernels
                          // threshold.f to use with .._f32 kernel
        float momentum;
        float epsilon;
        float N;
    };
} // namespace ns_BatchNormVarienceKernel

/*
 * Kernel name: batch_norm_stage1_bwd_bf16
 */
namespace ns_BatchNormStage1Kernel
{
    struct Params
    {
        int disable_beta_gamma_update;
        int N;
    };
    struct ParamsV2 : public Params
    {
        bool isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        BatchNormMode_t deterministic_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        BatchNormRoundingMode_t rounding_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_BatchNormStage1Kernel

/*
 * Kernel name: batch_norm_stage2_relu_bwd_bf16,        batch_norm_stage2_relu_bwd_f32
 *              batch_norm_stage2_relu_fwd_bf16,        batch_norm_stage2_relu_fwd_f32
 *              batch_norm_stage2_add_relu_bwd_bf16,    batch_norm_stage2_add_relu_bwd_f32
 *              batch_norm_stage2_add_relu_fwd_bf16,    batch_norm_stage2_add_relu_fwd_f32
 */
namespace ns_BatchNormStage2Kernel
{
    struct Params
    {
        float momentum;
        int disable_runnings_update;
        int N;
        float epsilon;
    };
    struct ParamsV2 : public Params
    {
        bool isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        BatchNormMode_t deterministic_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        BatchNormRoundingMode_t rounding_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_BatchNormStage2Kernel

namespace ns_BatchnormStage15Kernel
{
    struct Params
    {
        float momentum;
        float eps;
        int   disable_runnings_update;
        int   kernelType;
        int   useSigmaXk;
        int   enConcat;
    };
} // namespace ns_BatchnormStage15Kernel

/*
 * Kernel name: instance_norm_f32, instance_norm_relu_f32,
 */
namespace ns_InstanceNormKernel
{
    struct Params
    {
        fint_t threshold; // threshold.f to use with .._f32 kernel
    };
} // namespace ns_InstanceNormKernel

/*
 * Kernel name: instance_norm_fwd_f32
 */
namespace ns_InstanceNormTrainingKernel
{
    struct Params
    {
        float momentum;
        float eps;
    };
    struct ParamsRelu : Params
    {
        int enRelu;
    };
    ASSERT_SIZE_IS_GREATER(ParamsRelu, Params);
} // namespace ns_InstanceNormTrainingKernel

/*
 * Kernel name: div_round_fwd_f32, div_round_fwd_bf16, div_round_fwd_i32
 */
namespace ns_DivRoundKernel
{
    struct Params
    {
        DivRoundMode_t mode;
    };
} // namespace ns_DivRoundKernel

/*
 * Kernel name: layer_norm_relu_f32,
 */
namespace ns_LayerNormKernelRelu
{
    struct Params
    {
        fint_t threshold; // threshold.f to use with .._f32 kernel
        bool epsValid;
        fint_t eps; // threshold.f to use with .._f32 kernel
    };
    struct ParamsNorm : Params
    {
        int NormAxisBmp;  // A bit-map for CWHN. Set res bit for the dim to be normalized
        int ParamAxisBmp; // A bit-map for CWHN. Set res bit for the dim to be normalized
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
} // namespace ns_LayerNormKernelRelu

/*
 * Kernel name: layer_norm_f32,
 *              layer_norm_fwd_f32, layer_norm_fwd_bf16
 */
namespace ns_LayerNormKernel
{
    struct Params
    {
        bool epsValid;
        float eps;
    };
    struct ParamsNorm : Params
    {
        int NormAxisBmp;  // A bit-map for CWHN. Set res bit for the dim to be normalized
        int ParamAxisBmp; // A bit-map for CWHN. Set res bit for the dim to be normalized
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
    // It should derive after Params as ParamsNorm contents are meaningless in ParamsPt case
    // but only linear params hierarchy is currently supported
    struct ParamsPt : ParamsNorm
    {
        unsigned normalizedShapeDims;
    };
    ASSERT_SIZE_IS_GREATER(ParamsPt, ParamsNorm);
    struct ParamsRmsNorm : public ParamsPt
    {
        bool fastMath;
    };
    ASSERT_SIZE_IS_GREATER(ParamsRmsNorm, ParamsPt);
    struct ParamsRmsNormV2 : public ParamsRmsNorm
    {
        RetainInvRmsMode_t retainInvRmsMode;
        unsigned           CLAlignedPackSize;
    };
    ASSERT_SIZE_IS_GREATER(ParamsRmsNormV2, ParamsRmsNorm);
    struct ParamsRmsNormV3 : public ParamsRmsNormV2
    {
        bool hasResidual;    // If true, expect residual input tensor (x = x + residual before norm)
        bool addOneToWeight; // If true, apply (gamma + 1.0) instead of gamma
    };
    ASSERT_SIZE_IS_GREATER(ParamsRmsNormV3, ParamsRmsNormV2);
} // namespace ns_LayerNormKernel

/*
 * Kernel name: layer_norm_bwd_stage1_f32,
 *              layer_norm_bwd_stage1_bf16
 */
namespace ns_LayerNormStage1Kernel
{
    struct Params
    {
        int chunkSize; // size of chunk to be accumulated for dBeta and dGamma along width
    };
} // namespace ns_LayerNormStage1Kernel

/*
 * Kernel name: lrn_f32
 */
namespace ns_LrnKernel
{
    struct Params
    {
        float alpha;
        float beta;
        float knorm;
        int nsize;
    };
} // namespace ns_LrnKernel

/*
 * Kernel name: rrelu_with_noise_fwd_f32, rrelu_with_noise_fwd_bf16
 */
namespace ns_RreluWithNoiseKernel
{
    struct Params
    {
        float lower;
        float upper;
        bool training;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RreluWithNoiseKernel


/*
 * Kernel name: lpnorm_f32
 */
namespace ns_LpNormKernel
{
    struct Params
    {
        float p;
        float eps;
        int dim;
    };
    struct ParamsNorm : Params
    {
        bool epsMax;  // Performs 'max(denominator,eps)' instead of 'denominator += eps'
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LpNormKernel

/*
 * Kernel name: lpnorm_frobenius_stage1_f32,
 *              lpnorm_frobenius_stage1_bf16
 */
namespace ns_LpNormFroStage1Kernel
{
    struct Params
    {
        int chunkSize; // size of chunk to be accumulated for norm along width
    };
}

/*
 * Kernel name: norm_moments_f32,
 *              norm_moments_bf16
 */
namespace ns_NormMomentsKernel
{
    struct Params
    {
        unsigned int NormAxisBmp; // bit-map (bit0: FCD). Set bit for the dim to be normalized
    };
    struct ParamsV2 : public Params
    {
        bool useEstMean; // use MeanEstimation tensor (K) in calculation.
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_NormMomentsKernel

/*
 * Kernel name: norm_stage2_f32,
 *              norm_stage2_bf16
 */
namespace ns_NormStage2
{
    struct Params
    {
        float momentum;
        int disable_runnings_update;
        float epsilon;
    };
} // namespace ns_NormStage2

/*
 * Kernel name: softmax_f32, logsoftmax_f32,
 *              softmax_i16, logsoftmax_i16
 */
namespace ns_Softmax
{
    struct Params
    {
        int dim;
    };
    struct ParamsV3 : public Params
    {
        DataTypeSize_t dataTypeSize;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);
    struct ParamsV4 : public ParamsV3
    {
        int validCountAxis;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        int is_approximation;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    struct ParamsV6 : public ParamsV5
    {
        int triangularMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV6, ParamsV5);
    struct ParamsV7 : public ParamsV6
    {
        SoftmaxMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV7, ParamsV6);
    struct ParamsV8 : public ParamsV7
    {
        bool experimentalMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV8, ParamsV7);
    struct ParamsV9 : public ParamsV8
    {
        bool safeSoftmax;
        int dummy;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV9, ParamsV8);
    struct ParamsV10 : public ParamsV9
    {
        float invAttnHeads;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV10, ParamsV9);
} // namespace ns_Softmax

 /*
 * Kernel name: softmax_fa2
 */
namespace ns_SoftmaxFA2
{
    struct Params
    {
        int            dim;
        SoftmaxMode_t  mode;
        bool           experimentalMode;
        int            dummy;
    };
    struct ParamsV2 : public Params
    {
        float invAttnHeads;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    struct ParamsV3 : public ParamsV2
    {
        SoftmaxFA2Mode_t  modeV2;
        int boxGlobalDepthSize;
        int boxGlobalDepthStart;
        int boxGlobalWidthStart;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_SoftmaxFA2

/*
 * Kernel name: unpack_multiply
 */
namespace ns_UnpackMultiply
{
    struct Params
    {
        bool isHf8IEEE;
    };
} // namespace ns_UnpackMultiply

/*
 * Kernel name: mult_bf16
 */
namespace ns_MultSwizzled
{
    struct Params
    {
        DataTypeSize_t dataTypeSize;
    };
} // namespace ns_MultSwizzled

namespace ns_Mult = ns_MultSwizzled;

/*
* Kernel name: constant_i8, constant_i16, constant_f32
*/
namespace ns_ConstantKernel
{
    struct Params
    {
        fint_t constant;
    };

    struct Params_v2
    {
        int const_low;
        int const_high;
    };
    struct ParamsV3 : public Params
    {
        int const_low;
        int const_high;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);
} // namespace ns_ConstantKernel

/*
 * Kernel name: random_uniform_f32
 */
namespace ns_RandomUniform
{
    struct Params
    {
        float low;
        float high;
        unsigned int seed;
    };

    struct ParamsV2
    {
        fint_t low;
        fint_t high;
        unsigned int seed;
        int dummy; // to avoid Params size ambiguity
    };

    struct ParamsV3 : public ParamsV2
    {
        int low_low_32_Bit;
        int low_high_32_Bit;
        int high_low_32_Bit;
        int high_high_32_Bit;
    };
} // namespace ns_RandomUniform

namespace ns_PhiloxRandomUniform
{
    struct Params
    {
        float low;
        float high;

        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        unsigned int counterInitial;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        int low_i;
        int high_i;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_PhiloxRandomUniform


/*
 * Kernel name: random_normal_f32
 */
namespace ns_RandomNormal
{
    struct Params
    {
        float mean;
        float stddev;
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        int usePhilox;
    };
} // namespace ns_RandomNormal

/*
 * Kernel name: random_multinomial_f32,
 *              random_multinomial_log_f32
 */
namespace ns_RandomMultinomial
{
    struct Params
    {
        int outcomes;
        int shape;
        int get_prob;
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        int num_samples;
        bool replacement;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        int isTfMode;
        bool useRandomSeed;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RandomMultinomial

/*
 * Kernel name: exp_fast_fwd_hf8
 */
namespace ns_ExponentFast
{
    struct Params
    {
        bool experimentalMode;
    };
} // namespace ns_ExponentFast

/*
 * Kernel name: random_exponential_f32
 */
namespace ns_RandomExponential
{
    struct Params
    {
        float beta;
        unsigned int seed;
    };
} // namespace ns_RandomExponential

/*
* Kernel name : random_gamma_f32
*/
namespace ns_RandomGamma
{
    struct Params
    {
        float alpha;
        float beta;
        unsigned int seed;
    };
} // namespace ns_RandomGamma

/*
* Kernel name: random_negative_binomial_f32
*/
namespace ns_RandomNegativeBinomial
{
    struct Params
    {
        float k;
        float p;
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        bool isAdditionEnable;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RandomNegativeBinomial

/*
* Kernel name: random_bernoulli_fwd_f32, random_bernoulli_fwd_bf16
*/
namespace ns_RandomBernoulli
{
    struct Params
    {
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        float probability;
    };
} // namespace ns_RandomBernoulli

/*
 * Kernel name: random_uniform_fwd_f32, random_uniform_fwd_bf16
 */
namespace ns_XavierFill
{
    struct Params
    {
        unsigned int seed;
    };
} // namespace ns_XavierFill

/*
* Kernel name : pick_f32
*/
namespace ns_Pick
{
    struct Params
    {
        int axis;
    };
} // namespace ns_Pick

/*
 * Kernel name : random_poisson_f32
 */
namespace ns_RandomPoisson
{
    struct Params
    {
        float lambda;
        unsigned int seed;
        RandomPoissonFlavor_t poissonFlavor;
    };
} // namespace ns_RandomPoisson

/*
 * Kernel name : pow_f32
 */
namespace ns_Power
{
    struct Params
    {
        float exp_val;
    };
} // namespace ns_Power

/*
 * Kernel name: bitonic_sort_f32
 */
namespace ns_BitonicSort
{
    struct Params
    {
        int chunkSize;        // Size of each chunk to be sorted (Same for input/output)
        int axis;             // axis to be sorted
        SortDirection_t dir;  // ascending/descending
        bool isInputSqueezed; // Is the input squeezed
    };
    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BitonicSort

/*
 * Kernel name: search_sorted_fwd_f32 search_sorted_fwd_i32 search_sorted_fwd_bf16
*/
namespace ns_SearchSorted
{
    struct Params
    {
        int right;
    };
} // namespace ns_SearchSorted

/*
 * Kernel name: sort_step_f32
 */

/*
 * Multiple kernel invocations lead to sorted output
 *
 * ( i) totalIterations - number of times the kernel has to be invoked
 *
 *                 - (n * ( n + 1 ) ) / 2
 *
 *             where n = log2 of size of dimension to be sorted
 *
 *       Example : Input : 4D Tensor of size {4,16,5,5}
 *                         axis = 1
 *                         is_ascend = 1
 *
 *           Size of dimension 1 = 16
 *           log2(16) = 4
 *
 *           totalIterations = ( 4 * ( 4 + 1 ) ) / 2 = 10
 *
 *           totalIterations = 10
 *
 *           for(int i = 0; i < totalIterations; i++)
 *           {
 *               iterationNumber = i;
 *
 *               // Kernel Invocations
 *
 *           }
 */
namespace ns_SortStep
{
    struct Params
    {
        // python layer params
        int totalIterations;
        int iterationNumber; // current iteration number

        // user params
        int axis;       // axis to be sorted
        bool is_ascend; // ascending/descending
    };
    struct ParamsV2 : public Params
    {
        int isStable; // 1: stable 0: not stable
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_SortStep

/*
 * Kernel name : filter_and_squeeze_f32
 */
namespace ns_FilterAndSqueeze
{
    struct Params
    {
        fint_t threshold;
    };
} // namespace ns_FilterAndSqueeze

/*
 * Kernel name: scalar_merge_f32
 */

namespace ns_ScalarMerge
{
    struct Params
    {
        int inputChunkSize;
        int topK; // 0 < topK <= IFM dim[axis] size anything beyond that is an error.
        SortDirection_t sortDirection;
        int axis;
        bool isInputSqueezed;
    };

    struct ParamsV2 : public Params
    {
        int currIterNum; // Static iteration number to enable early exit in DS case
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    struct ParamsV3 : public ParamsV2
    {
        ScalarMergeInputs_t ScalarMergeInputs;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_ScalarMerge

/*
 * Kernel name : nantonum_fwd_f32, nantonum_fwd_bf16
 */
namespace ns_NanToNumKernel
{
    struct Params
    {
        double nan;
        double posInf;
        double negInf;
    };
} // namespace ns_NanToNumKernel

/*
 * Kernel name: sort_bwd_f32
 */
namespace ns_SortBwd
{
    struct Params
    {
        int topK;
        int axis;
    };
} // namespace ns_SortBwd

/*
 * Kernel name: sort_pre_process_f32, sort_pre_process_i32
 */
namespace ns_SortPreProcess
{
    struct Params
    {
        int staticChunkSize;     // Size of each chunk to be sorted according to the graph.
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
    };

    struct ParamsV2 : public Params
    {
        bool hasActualValidCount;   // Flag to generate actual VC for better performance
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    /// disable size check
    struct ParamsV3 : public ParamsV2
    {
        bool vcShapeDataTensor;   // Flag to indicate if VC received is a shape data tensor
    };
    // ******************************************
    // Oops: sizeof(ParamsV3) == sizeof(ParamsV2)
    // ******************************************

    struct ParamsV4 : public ParamsV3
    {
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_SortPreProcess

/*
 * Kernel name: generate_bitonic_chunks_f32, generate_bitonic_chunks_i32
 */
namespace ns_GenerateBitonicChunks
{
    struct Params
    {
        int staticChunkSize;     // Size of each chunk to be sorted according to the graph.
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
    };

    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GenerateBitonicChunks
/*
* Kernel name: large_bitonic_merge_f32, large_bitonic_merge_i32
*/
namespace ns_LargeBitonicMerge
{
    struct Params
    {
        int staticChunkSize;     // Chunk size at the GBC stage
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
        int blockId;
        int stageId;
    };

    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LargeBitonicMerge

/*
 * Kernel name: small_bitonic_merge_f32, small_bitonic_merge_i32
 */
namespace ns_SmallBitonicMerge
{
    struct Params
    {
        int staticChunkSize;     // Chunk size at the GBC stage
        int blockId;             // Block number corresponding to required SBM stage
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
    };

    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_SmallBitonicMerge

//////////////////////////////// GNMT_KERNELS /////////////////////////////////

/*
 * Kernel name: top_k_st1_i8,   top_k_st2_i8,
 *              top_k_st1_i16,  top_k_st2_i16,
 *              top_k_st1_i32,  top_k_st2_i32,
 *              top_k_st1_f32,  top_k_st2_f32,
 */
namespace ns_TopK
{
    struct Params
    {
        unsigned int kSize;
        unsigned int axis;
        unsigned int chunkSize;
    };

    struct ParamsV2 : public Params
    {
        unsigned int bottomK;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_TopK

/*
 * Kernel name: gru_st1_i8, gru_st2_i8,
 */
namespace ns_GruKernel
{
    struct Params
    {
        int expX;
        int expCt;
    };
} // namespace ns_GruKernel
/*
 * Kernel name: lstm_i16, lstm_f32
 */
namespace ns_LstmKernel
{
    struct Params
    {
        int timeStamp;            // begins with 0, must be LT B vector length
        int expGatesIntermediate; // must be GE to expGatesIn extracted from Tensors
    };
} // namespace ns_LstmKernel

/*
 * Kernel name: bs_st0_f32,
 *              bs_st1_i16,
 *              bs_st2_i16,
 *              bs_st3_i16,
 *              bs_st4_i16,
 *              bs_st5_i16,
 *              bs_st6_i16,
 *              bs_st7_i16,
 *              bs_st8_i16,
 *              bs_st8_i8,
 *              bs_st9_i16
 */
namespace ns_BsKernel
{
    struct Params
    {
        unsigned int k;       //  bsw;
        unsigned int b;       //  bdec;
        unsigned int vocSize; // vocabularySize;
        unsigned int maxOutputLen;
        unsigned int decoderOutputSize;
        float scaleFix;
        float scaleGem;
        int zp;
    };
} // namespace ns_BsKernel

////////////////////////////// REDUCTION_KERNELS //////////////////////////////

/*
 * Kernel name: reduce_sum_f32,
 *              reduce_sum_i8,
 *              reduce_sum_i16,
 *              reduce_prod_f32,
 *              reduce_L1_f32,
 *              reduce_L2_f32,
 *              reduce_log_sum_f32,
 *              reduce_log_sum_exp_f32,
 *              reduce_sum_square_f32,
 *              reduce_max_f32,
 *              reduce_min_f32,
 *              reduce_mean_f32,
 *              argmin_i8,
 *              argmin_i16,
 *              argmin_f32,
 *              argmax_i8,
 *              argmax_i16,
 *              argmax_f32,
 *              hardmax_f32
 */
namespace ns_Reduction
{
    struct Params
    {
        unsigned int reductionDimension;
    };

    struct ParamsV2
    {
        unsigned int reductionDimensionMask; // Bit mask representing dimensions(axes) to be reduced.
                                             // A '1' at Bit_n means dimension 'n' should be reduced
                                             // (with Bit_0 being LSB).
                                             // Eg: mask with binary value 1011 means
                                             // dimensions 3, 1 and 0 are to be reduced(TPC order).
        bool keepDim;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_Reduction

/*
 * Kernel name: reduce_Lp_f32
 */
namespace ns_ReduceLp
{
    struct Params : public ns_Reduction::Params
    {
        unsigned int p;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Reduction::Params);
} // namespace ns_ReduceLp

namespace ns_ReduceLpV2
{
struct Params : public ns_Reduction::Params {
        fint_t p;
        TypeOfP_t typeOfP;
};
    ASSERT_SIZE_IS_GREATER(Params, ns_Reduction::Params);

    struct ParamsV2 : public ns_Reduction::ParamsV2 {
        fint_t p;
        TypeOfP_t typeOfP;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, ns_Reduction::ParamsV2);
} // namespace ns_ReduceLpV2

/*
 * Kernel name : kthvalue_fwd_f32, kthvalue_fwd_f16,
 *               kthvalue_fwd_bf16, kthvalue_fwd_i32
 */
namespace ns_Kthvalue
{
    struct Params
    {
        unsigned int k_value;
        unsigned int axis;
        bool keep_dims;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_Kthvalue

/////////////////////////// SPATIAL_POOLING_KERNELS ///////////////////////////

/*
 * Kernel name: filter_2d_f32,
 *              filter_2d_i8,       filter_2d_i16,
 *              filter_2d_relu_i8,  filter_2d_relu_i16,
 *              maxpool_2d_i8,      maxpool_2d_i16
 */
namespace ns_SpatialReduction
{
    struct Params
    {
        int pad_w_begin;
        int pad_w_end;
        int pad_h_begin;
        int pad_h_end;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int dilation_w;
        int dilation_h;
        EPoolingConvention pooling_convention;
    };
    struct ParamsV2 : public Params
    {
        IndexTypeMode_t index_type_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_SpatialReduction

/*
 * Kernel name: filter_3d_f32,
 *              maxpool_3d_f32
 */
namespace ns_SpatialReduction3D
{
    struct Params : public ns_SpatialReduction::Params
    {
        int pad_d_begin;
        int pad_d_end;
        int kernel_d;
        int stride_d;
        int dilation_d;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction::Params);
} // namespace ns_SpatialReduction3D

/*
 * Kernel name: avg_pool_2d_i8,         avg_pool_2d_i16,
 *              avg_pool_2d_relu_i8,    avg_pool_2d_relu_i16
 */
namespace ns_AveragePooling
{
    struct Params : public ns_SpatialReduction::Params
    {
        int includePadding;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction::Params);
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
} // namespace ns_AveragePooling

/* Kernel name: avg_pool_2d_f32
                avg_pool_2d_bf16 */
namespace ns_AveragePoolingWithDivisorOverride
{
    struct Params : public ns_AveragePooling::Params
    {
        int divisorOverride;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_AveragePooling::Params);
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
} // namespace ns_AveragePoolingWithDivisorOverride

/*
 * Kernel name : avg_pool_3d_f32
 */
namespace ns_AveragePooling3D
{
    struct Params : public ns_SpatialReduction3D::Params
    {
        int includePadding;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction3D::Params);
} // namespace ns_AveragePooling3D

/* Kernel name: avg_pool_3d_f32
*               avg_pool_3d_bf16
*               avg_pool_3d_f16
*/
namespace ns_AveragePooling3DWithDivisorOverride
{
    struct Params : public ns_AveragePooling3D::Params
    {
        int divisorOverride;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_AveragePooling3D::Params);
} // namespace ns_AveragePooling3DWithDivisorOverride

/*
 * Kernel name: maxpool_roi_i8,
                maxpool_roi_i16,
                maxpool_roi_bwd_f32,
                maxpool_roi_bwd_f16,
                maxpool_roi_bwd_bf16,
 */
namespace ns_MaxPoolRoiKernel
{
    struct Params
    {
        unsigned int num_channels;
        unsigned int pooled_shapeX;
        unsigned int pooled_shapeY;
        unsigned int num_roi;
    };
    struct ParamsSegmentsPerAxis : public Params
    {
        int segmentsX;
        int segmentsY;
    };
} // namespace ns_MaxPoolRoiKernel

/*
 * CGUID name: max_unpool_2d, max_unpool_2d_fwd, max_unpool_3d, max_unpool_3d_fwd
 */
namespace ns_MaxUnpool
{
    struct Params
    {
        unsigned int output_h;
        unsigned int output_w;
    };
    struct ParamsV2 : public Params
    {
        unsigned int output_d;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_MaxUnpool

/*
 * The struct is deprecated, ns_PadKernelEx should be used instead
 */
namespace ns_PadKernel
{
    struct Params
    {
        fint_t value;
        unsigned int pads[10];
    };
} // namespace ns_PadKernel

/*
 * Kernel name: pad_i8, pad_i16, pad_f32, pad_i32
 */
namespace ns_PadKernelEx
{
    struct Params : public ns_PadKernel::Params
    {
        PadMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_PadKernel::Params);
    struct ParamsWithAlign : public Params
    {
        unsigned int output_size_alignment[5];
    };
    ASSERT_SIZE_IS_GREATER(ParamsWithAlign, Params);
} // namespace ns_PadKernelEx

/*
 * Kernel name: memset_from_vc_i8, memset_from_vc_i16, memset_from_vc_f32, memset_from_vc_i32
 */
namespace ns_MemsetFromVC
{
    struct Params
    {
        int axis;
    };
} // namespace ns_MemsetFromVC


/*
 * Kernel name: broadcast_non_fcd_u8,
 *              broadcast_non_fcd_i8,
 *              broadcast_non_fcd_i16,
 *              broadcast_non_fcd_u16,
 *              broadcast_non_fcd_i32,
 *              broadcast_non_fcd_f32,
 *              broadcast_non_fcd_f16,
 *              broadcast_non_fcd_bf16
 */
namespace ns_BroadcastNonFcd
{
    struct Params
    {
        int axis;
    };

    struct ParamsV2 : public Params
    {
        //this param is used in BroadcastNonFcd cguid as an indicator for a second pass
        int cguidAfterTile;//default 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_BroadcastNonFcd

/*
 * Kernel name : lpPooling_3d_f32, lpPooling_2d_f32
 */
namespace ns_LpPooling3D
{
    struct Params : public ns_SpatialReduction3D::Params
    {
        int p;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction3D::Params);
} // namespace ns_LpPooling3D

/*
 * Kernel name: sequence_reverse_i32,
 *              sequence_reverse_i16,
 *              sequence_reverse_i8,
 *              sequence_reverse_f32
 */
namespace ns_SequenceLength
{
    struct Params
    {
        int use_sequence_length; // To use variable-length sequences, set use_sequence_length to
                                 // True, otherwise each example in the batch is assumed
                                 //  to have the max sequence length.
        int batch_axis;
        int time_axis;
    };
} // namespace ns_SequenceLength

/*
 * Kernel name : tile_f32
 */
namespace ns_TileKernel
{
    struct Params
    {
        int repeat[4];
    };
    struct ParamsV2
    {
        int repeat[5];
    };
} // namespace ns_TileKernel

/*
 * Kernel name :sequence_mask_var_len_f32,
 *              sequence_mask_var_len_i8,
 *              sequence_mask_var_len_i16
 */
namespace ns_SequenceMask
{
    struct Params
    {
        bool use_sequence_length;
        float mask_value;
    };
} // namespace ns_SequenceMask

/*
 * Kernel name : dropout_fwd_f32, dropout_bwd_f32
 */
namespace ns_DropoutKernel
{
    struct Params
    {
        float ratio;
        unsigned int seed;
    };
    struct ParamsOptionalMaskOut : public Params
    {
        bool disableMaskOut;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalMaskOut, Params);
    struct ParamsOptionalTrain : public ParamsOptionalMaskOut
    {
        int isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalTrain, ParamsOptionalMaskOut);
} // namespace ns_DropoutKernel

/*
 * Kernel name : scaled_triangular_dropout_mask_fwd_f32,
 *               scaled_triangular_dropout_mask_fwd_f16,
 *               scaled_triangular_dropout_mask_fwd_bf16
 */
namespace ns_ScaledTriangularDropoutMaskKernel
{
    struct Params
    {
        float ratio;
        unsigned int seed;
        unsigned int allSlicesBase;
        ScaledMaskedTriangularSoftmaxMode_t mode;
        bool isSeedH2D;
        bool isAllSlicesBaseH2D;
        bool isResetUpperTriangle;
    };
} // namespace ns_ScaledTriangularDropoutMaskKernel

/*
 * Kernel name : repeat_f32
 */
namespace ns_RepeatKernel
{
    struct Params
    {
        int repeats;
        int axis;
    };
    struct ParamsV2 : public Params
    {
        RepeatMode_t scalarAPIMode;
    };
} // namespace ns_RepeatKernel

/*
 * Kernel name : repeat_fwd_f32, repeat_bwd_f32
 */
namespace ns_RepeatKernelGaudiTF
{
    struct Params
    {
        int axis;
    };
} // namespace ns_RepeatKernelGaudiTF

/*
 * CGUID name : repeat_pt
 */
namespace ns_RepeatPt
{
    struct Params
    {
        int repeat[8];
        unsigned int size;
    };
} // namespace ns_RepeatPt

/*
 * Kernel name : roll_fwd_f32, roll_fwd_bf16, roll_fwd_i32
 */
namespace ns_RollKernel
{
    struct Params
    {
        int shifts[8];
        int dims[8];
        unsigned int num_dims;
    };
} // namespace ns_RollKernel

/*
 * Kernel name : take_f32
 */
namespace ns_TakeKernel
{
    struct Params
    {
        int axis;
        int mode;
    };
} // namespace ns_TakeKernel

/*
 * Kernel name : gather_f32
 */
namespace ns_GatherKernel
{
    struct Params
    {
        int axis;
    };

    struct ParamsV2 : public Params
    {
        int batchDims;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_GatherKernel


/*
 * Kernel name : gather_with_assert
 */
namespace ns_GatherWithAssertKernel
{
    struct Params : public ns_GatherKernel::ParamsV2
    {
        int isDebugMode;
    };

} // namespace ns_GatherWithAssertKernel

/*
 * Kernel name : flip_f32
 */
namespace ns_FlipKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2
    {
        int dims[5];
        unsigned int num_dims;
    };
} // namespace ns_FlipKernel

/*
 * Kernel name : compress_f32
 */
namespace ns_CompressKernel
{
    struct Params
    {
        int axis;
    };
} // namespace ns_CompressKernel

/*
 * Kernel name : ravel_multi_index_i32
 */
namespace ns_RavelKernel
{
    struct Params
    {
        int shape[5];
    };
    struct ParamsV2
    {
        long shape[5];
    };
} // namespace ns_RavelKernel

/*
 * Kernel name : one_hot_f32
 */
namespace ns_OneHotKernel
{
    struct Params
    {
        int axis;
        int depth;
        float on_value;
        float off_value;
    };
    struct ParamsV2 : public Params
    {
        int is_out_reshaped; //defualt 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_OneHotKernel

/*
 * Kernel name : correlation_f32
 */
namespace ns_CorrelationKernel
{
    struct Params
    {
        int kernel_size;
        int max_displacement;
        int stride1;
        int stride2;
        int pad_size;
        int is_multiply;
    };
} // namespace ns_CorrelationKernel

/*
 * Kernel name : spatial_correlation_f32
 */
namespace ns_SpatialCorrelationKernel
{
    struct Params
    {
        SpatialCorrelation_t interpolation;
    };
} // namespace ns_SpatialCorrelationKernel

/*
 * Kernel name : expand_jagged_indices_fwd_dim0_i32
 */
namespace ns_ExpandIntoJaggedIndicesKernel
{
    struct Params
    {
        ExpandIntoJaggedIndices_t increment;
    };
} // namespace ns_ExpandIntoJaggedIndicesKernel

/*
 * Kernel name : upsample_f32, upsample_i8, upsample_u8
 */
namespace ns_UpsampleKernel
{
    struct Params
    {
        EUpsampleType_t mode;
        int scale;
    };
} // namespace ns_UpsampleKernel

/*
 * Kernel name : sparse_lengths_sum_i8, sparse_lengths_sum_i16
 */
namespace ns_SparseLengthsSum
{
    struct Params
    {
        ESparseLengthsSumType_t mode;
    };
} // namespace ns_SparseLengthsSum

/*
 * Kernel name : deformation_transformation_f32, deformation_transformation_i16
 */
namespace ns_DeformationTransformation
{
    struct Params
    {
        int groupCount;
        int kernelWidth;
        int kernelHeight;
        int strideW;
        int strideH;
        int dilationW;
        int dilationH;
        int padW;
        int padH;
        int isMaskUsed;
    };
} // namespace ns_DeformationTransformation

/*
 * Kernel name : fully_connected_f32
 */
namespace ns_FullyConnected
{
    struct Params
    {
        int is_relu;
    };
} // namespace ns_FullyConnected

/*
 * Kernel name : grid_generator
 */
namespace ns_GridGeneratorKernel
{
    struct Params
    {
        EGridGeneratorType_t mode;
    };
} // namespace ns_GridGeneratorKernel

/*
 * Kernel name : smooth_l1_f32
 */
namespace ns_SmoothL1Kernel
{
    struct Params
    {
        float sigma;
    };
} // namespace ns_SmoothL1Kernel

/*
 * Kernel name : shrink_f32, shrink_fwd_f32, shrink_fwd_bf16,
 *               shrink_bwd_f32, shrink_bwd_bf16
 */
namespace ns_ShrinkKernel
{
    struct Params
    {
        float lambda;
        float bias;
    };
    struct ParamsV2 : public Params
    {
        float lowerBound;
        float upperBound;
        ShrinkMode_t mode;
    };
} // namespace ns_ShrinkKernel

/*
 * Kernel name : optimizer_sgd_bwd_bf16, optimizer_sgd_bwd_f32
 */
namespace ns_OptimizerSGD
{
    struct Params
    {
        float wd;
        float mom;
        float damp;
        bool nesterov;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_OptimizerSGD

/*
 * Kernel name : optimizer_sparse_sgd_bf16
 */
namespace ns_OptimizerSparseSGD
{
    struct Params
    {
        float mom;
        bool nesterov;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_OptimizerSparseSGD

/*
 * Kernel name : optimizer_adagrad_bwd_bf16, optimizer_adagrad_bwd_f32,
 *               optimizer_adagrad_bf16, optimizer_adagrad_f32
 */
namespace ns_OptimizerAdagrad
{
    struct Params
    {
        float wd;
        float lrd;
        float eps;
    };
} // namespace ns_OptimizerAdagrad

/*
 * Kernel name : optimizer_sparse_adagrad_bf16,
 *               optimizer_sparse_adagrad_with_valid_count_2d_f32,
 *               optimizer_sparse_adagrad_with_valid_count_2d_bf16
 */
namespace ns_OptimizerSparseAdagrad
{
    struct Params
    {
        float decay;
        float eps;
    };
} // namespace ns_OptimizerSparseAdagrad

/*
 * Kernel name : optimizer_sparse_rowwise_adagrad_with_valid_count_2d_bf16,
 *               optimizer_sparse_rowwise_adagrad_with_valid_count_2d_f32
 */
namespace ns_OptimizerRowWiseAdagrad
{
    struct Params
    {
        float eps;
    };
    struct ParamsV2 : public Params
    {
        float decay;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_OptimizerRowWiseAdagrad

/*
 * Kernel name : range_f32,
 *               range_i16,
 *               range_i8
 */
namespace ns_RangeKernel
{
    struct Params
    {
        fint_t start;
        fint_t limit;
        fint_t delta;
    };
} // namespace ns_RangeKernel

/*
 * Kernel name : linspace_f32,
 *               linspace_f16,
 *               linspace_bf16
 */
namespace ns_LinspaceKernel
{
    struct Params
    {
        float start;
        float end;
        int steps;
    };
} // namespace ns_LinspaceKernel

/*
 * Kernel name : nms_f32
 */
namespace ns_Nms
{
    struct Params
    {
        float nms_threshold;
    };
} // namespace ns_Nms

/*
 * Kernel name : post_nms_f32
 */
namespace ns_PostNms
{
    struct Params
    {
        int max_output_size;
    };
    struct ParamsV2 : public Params
    {
        int max_size_per_batch;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    enum PostNms_type_t
    {
        ONNX_FRMT = 0,
        TF_COMBINED_NON_MAX_SUPPRESSION_FRMT,
        TF_NON_MAX_SUPPRESSION_FRMT
    };
    struct ParamsV3 : public ParamsV2
    {
        int pad_to_max_output_size; // meset
        PostNms_type_t  frmt;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_PostNms

/*
 * Kernel name : complex_nms_f32
 */
namespace ns_ComplexNMS
{
    struct Params
    {
        ns_FilterAndSqueeze::Params filterAndSqueeze;
        ns_BitonicSort::Params bitonicSort;
        ns_ScalarMerge::Params mergeSort;
        ns_GatherKernel::Params gather;
        ns_Nms::Params nms;
        ns_PostNms::Params postNms;
    };
} // namespace ns_ComplexNMS

/*
 * Kernel name : softmax_cross_entropy_f32, softmax_cross_entropy_bwd_bf16,
 *               softmax_cross_entropy_bwd_f32, softmax_cross_entropy_fwd_bf16,
 *               softmax_cross_entropy_fwd_f32
 */
namespace ns_SoftmaxCrossEntropy
{
    struct Params
    {
        ECrossEntropyMode_t mode;
        int batchSize;
    };

    struct ParamsV2 : public Params
    {
        int axis;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    struct ParamsV3 : public Params
    {
        bool isTfMode; //To imitate TF at BWD
        int reserved;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);

    struct ParamsV4 : public Params
    {
        bool sparse;
        int reserved[2];
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, Params);
    struct ParamsV5 : public Params
    {
        int is_approximation;
        int padding[3];
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, Params);
} // namespace ns_SoftmaxCrossEntropy

/*
 * Kernel name : binary_cross_entropy_fwd_f32, binary_cross_entropy_bwd_f32
 */
namespace ns_BinaryCrossEntropy
{
    struct Params
    {
        bool isWeightsUsed;
        ECrossEntropyMode_t mode;
    };

    struct ParamsOptionalSigmoid : public Params
    {
        bool binaryCrossEntropyWithoutSigmoid;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalSigmoid, Params);
    struct ParamsOptionalPosWeight : public ParamsOptionalSigmoid
    {
        PosWeightMode_t posMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalPosWeight, ParamsOptionalSigmoid);
    struct ParamsOptionalNormalize : public ParamsOptionalPosWeight
    {
        int isNormalizeWeights;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalNormalize, ParamsOptionalPosWeight);
} // namespace ns_BinaryCrossEntropy

/*
 * Kernel name : eye_like_f32
 */
namespace ns_EyeLikeKernel
{
    struct Params
    {
        int k;
    };
} // namespace ns_EyeLikeKernel

/*
 * Kernel name : mediandim_f32, mediandim_bf16, mediandim_f16, mediandim_i32
 */
namespace ns_MediandimKernel
{
    struct Params
    {
        int reduction_dim;
        bool keep_dim;
    };
} // namespace ns_MediandimKernel

/*
 * Kernel name: max_unpool_2d_f32
 */
namespace ns_MaxUnpoolKernel
{
    struct Params
    {
        int pad_w_begin;
        int pad_w_end;
        int pad_h_begin;
        int pad_h_end;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int out_width;
        int out_height;
    };
} // namespace ns_MaxUnpoolKernel

/*
 * Kernel name : logit_f32
 */
namespace ns_LogitKernel
{
    struct Params
    {
        float epsilon;
    };
} // namespace ns_LogitKernel

/*
 * Kernel name : cast_f32_to_i16,
 *               cast_f32_to_i32,
 *               cast_f32_to_i8,
 *               cast_f32_to_u8,
 *               cast_i16_to_f32,
 *               cast_i32_to_f32,
 *               cast_i8_to_f32,
 *               cast_u8_to_f32
 */
namespace ns_CastKernel
{
    struct Params
    {
        CastF32RoundMode_t round_mode;
    };
    // ParamV2 'seed' argument is relevant only to stochastic round
    struct ParamsV2 : public Params
    {
        unsigned int seed;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // ParamsV3 is relevant only to fp8
    struct ParamsV3 : public ParamsV2
    {
        CastSatMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool safeCastCheck;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        int use_scalar_seed;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_CastKernel

/*
 * Kernel name : scatter_f32
 */
namespace ns_ScatterKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsIndices : public Params
    {
        /*
            0 - control indices tenor size, i.e. if size of indices tensor grt
                than size of ofm tensor return ERR
            1 - ignore indices tenor size
        */
        ScatterIndicesSizeControl_t isIndicesLarger;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIndices, Params);

    struct ParamsV2 : public ParamsIndices
    {
        int dim;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, ParamsIndices);
    struct ParamsReduce : public ParamsV2
    {
        int include_self;
    };
    ASSERT_SIZE_IS_GREATER(ParamsReduce, ParamsV2);

    struct ParamsV3 : public ParamsReduce
    {
        ScatterWrapIndices_t controlIndxVal;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsReduce);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_ScatterKernel

/*
 * Kernel name : scatter_nd_fwd_f32, scatter_nd_fwd_bf16
 */
namespace ns_ScatterNDKernel
{
    struct Params
    {
        int origIndicesDims;
        int origIndicesShape[5];
    };
    struct ParamsV2 : public Params
    {
        bool isUnsorted; // sort = 1
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_ScatterNDKernel

/*
 * Kernel name : scatter_nd_update_fwd_f32, scatter_nd_update_fwd_bf16
 */
namespace ns_ScatterNdUpdateKernel
{
    struct Params
    {
        ScatterNdUpdateMode_t mode;
    };
} // namespace ns_ScatterNdUpdateKernel

/*
 * Kernel name : scatter_reduce_fwd_f32, scatter_reduce_fwd_f16, scatter_reduce_fwd_bf16
 */
namespace ns_ScatterReduceKernel
{
    struct Params
    {
        int dim;
        bool include_self;
        ScatterReduceMode_t mode;
    };
} // namespace ns_ScatterReduceKernel

/*
 * Kernel name : gather_elements_f32, gather_elements_i32
 */
namespace ns_GatherElementsKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2 : public Params
    {
        bool isSort; // sort = 1
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GatherElementsKernel

/*
 * Kernel name : resize_u8, resize_i8, resize_f32
 */
namespace ns_ResizeKernel
{
    struct Params
    {
        ResizeInterpolationMode_t mode;
        float scaleDim1;
        float scaleDim2;
        float scaleDim3;
        ResizeCoordinateTransformationMode_t coordTransMode;
        ResizeNearestMode_t nearestMode;
        bool excludeOutside;
        bool useScales; // bool value to select between scales and output sizes
        float cubicCoeffA;
        int size1;
        int size2;
        int size3;
    };

    struct ParamsGauss : Params
    {
        int filterWidth;
        float sigma;
    };
    ASSERT_SIZE_IS_GREATER(ParamsGauss, Params);

    struct ParamsAA : ParamsGauss
    {
        bool useAntialiasing;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAA, ParamsGauss);
} // namespace ns_ResizeKernel

/*
 * Kernel name : grid_sample_bf16,
 *               grid_sample_f32
 */
namespace ns_GridSample
{
    struct Params
    {
        GridSampleInterpolation_t interp;
        GridSamplePad_t pad;
        bool alignCorners;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GridSample

namespace ns_TopkNode
{
    struct Params
    {
        unsigned int bsw; // K value for the topk algorithm.
        unsigned int axis; // Axis to search on for the top K values.
        unsigned int bottomK; // If false, K largest elements will be selected, else, K smallest
    };
} // namespace ns_TopkNode

namespace ns_TopkNodeV2
{
    struct Params
    {
        unsigned int bsw;     // K value for the topk algorithm
    };

    struct ParamsV2 : public Params
    {
    unsigned int axis;    // Axis to search on for the top K values.
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    struct ParamsV3 : public ParamsV2
    {
        bool bottomK; // If false, K largest elements will be selected, else K smallest
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
        bool isVcData; // was rank - 1 data tensor provided for VC
        KTensorType_t kType; // was K tensor provided, at which format, consumed when we have
                           // 2 input tensors
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        bool avoidHandleNans; // for floating point, default 0, backward compatible
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_TopkNode

namespace ns_FullBitonicSortNode
{
    struct Params
    {
        int staticChunkSize;     // Size of each chunk to be sorted according to the graph.
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_FullBitonicSortNode

namespace ns_BitonicScalarMergeNode
{
    struct Params
    {
        unsigned int bsw;     // TopK/BottomK sorted size
        unsigned int axis;                // axis to be sorted
        bool bottomK; // ascending/descending
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BitonicScalarMergeNode

namespace ns_BitonicSortStepNode
{
    struct Params
    {
        int axis;     // axis to be sorted
        bool is_ascend; // ascending/descending
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BitonicSortStepNode

namespace ns_TwoStageTopKNode
{
    struct Params
    {
        unsigned int kSize;     // Size of each chunk to be sorted according to the graph.
        unsigned int axis;      // axis to be sorted
    };

    struct ParamsV2 : public Params
    {
        unsigned int chunkSize;
        bool bottomK;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_TwoStageTopKNode

namespace ns_LogicalSifOpShapeToH2d
{
    struct Params
    {
        int axis;
    };
} // namespace ns_LogicalSifOpShapeToH2d

namespace ns_Sort
{
    struct Params
    {
        int axis;
        bool isDescending;
        RadixSortFlavor_t sortFlavor;
    };
} // namespace ns_Sort

namespace ns_FullRadixSort
{
    struct Params
    {
        int axis;
        bool isDescending;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_FullRadixSort

/*
 * Kernel name : radix_sort_stg1_i32,
 *               radix_sort_stg2_i32,
 */
namespace ns_RadixSort
{
    struct Params
    {
        int stg; // current iteration
        bool sortDescending; // 0: ascending 1: descending
        RadixSortKeyType_t keyType;
        RadixSortFlavor_t sortFlavor; // sort api
        int numTpc; // number of TPCs to work on
        bool isNansHandlingBsCompatible; // handle NaNs same as bitonic sort (replace with +/-INF)
    };

    struct ParamsV2 : public Params
    {
        int maxNumTpc; // Maximum number of available TPCs on device (Required for SIF)
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RadixSort

namespace ns_StoreTnsrSqueeze
{
    struct Params
    {
        int stg;
        int maxBktLen;
        int sSignBit;
    };
} // namespace ns_StoreTnsrSqueeze

/*
 * Kernel name : roialign_f32, roialign_fwd_f32, roialign_fwd_bf16
 */
namespace ns_RoiAlignKernel
{
    struct Params
    {
        RoiAlignMode_t mode;
        int sampling_ratio;
        float spatial_scale;
    };
    struct ParamsAlignment : public Params
    {
        bool aligned;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAlignment, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RoiAlignKernel

/*
 * Kernel name : roialign_bwd_f32, roialign_bwd_bf16
 */
namespace ns_RoiAlignBwdKernel
{
    struct Params
    {
        RoiAlignMode_t mode;
        int sampling_ratio;
        float spatial_scale;
        int aligned;
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsSegmentsPerAxis : public ParamsIsValidCount
    {
        int segmentsX;
        int segmentsY;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSegmentsPerAxis, ParamsIsValidCount);
} // namespace ns_RoiAlignBwdKernel

/*
 * Kernel name : isinf_f32
 */
namespace ns_IsInfKernel
{
    struct Params
    {
        int detect_negative;
        int detect_positive;
    };
} // namespace ns_IsInfKernel

/*
 * Kernel name : quad_tree_fwd_f32,
 *               quad_tree_fwd_bf16,
 *               quad_tree_torch_fwd_f32
 */
namespace ns_QuadTree
{
    struct Params
    {
        int segments; // should be pow(4, x)
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsAbsoluteCoords: ParamsIsValidCount
    {
        bool enableAbsoluteCoords;
        float levelScalarFactor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAbsoluteCoords, ParamsIsValidCount);
    struct ParamsTorchVersion: public ParamsAbsoluteCoords
    {
        int enableTorchVersion;
    };
    ASSERT_SIZE_IS_GREATER(ParamsTorchVersion, ParamsAbsoluteCoords);
    struct ParamsAligned: public ParamsTorchVersion
    {
        int enableAlignment;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAligned, ParamsTorchVersion);
    struct ParamsSegmentsPerAxis : public ParamsAligned
    {
        int segmentsX;
        int segmentsY;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSegmentsPerAxis, ParamsAligned);
} // namespace ns_QuadTree

/*
* Kernel name: addr_fwd_f32, addr_fwd_bf16
*/
namespace ns_AddrKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_AddrKernel

/*
* Kernel name: addmm_fwd_f32, addmm_fwd_bf16
*/
namespace ns_AddmmKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_AddmmKernel

/*
* Kernel name: addmv_fwd_f32, addmv_fwd_bf16
*/
namespace ns_AddmvKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_AddmvKernel

/*
* Kernel name: combined_nms_fwd_f32
*/
namespace ns_CombinedNmsKernel
{
    struct Params
    {
       float score_threshold;
       float nms_threshold;
       int max_total_size;
       int max_output_size_per_class;
    };
} // namespace ns_CombinedNmsKernel
/*
 * Kernel name : crop_and_resize_f32, crop_and_resize_fwd_f32,
 *               crop_and_resize_fwd_bf16
 */
namespace ns_CropAndResizeKernel
{
    struct Params
    {
        CropAndResizeMode_t mode;
        float extrapolationValue;
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsIsOptionalCropsSize : public ParamsIsValidCount
    {
        int isOptionalCropsSize;
        int dummy;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsOptionalCropsSize, ParamsIsValidCount);
    struct ParamsBorderReplicate : public ParamsIsOptionalCropsSize
    {
        bool borderReplicate;
    };
    ASSERT_SIZE_IS_GREATER(ParamsBorderReplicate, ParamsIsOptionalCropsSize);
    struct ParamsAbsoluteCoords : ParamsBorderReplicate
    {
        bool enableAbsoluteCoords;
        float levelScalarFactor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAbsoluteCoords, ParamsBorderReplicate);
} // namespace ns_CropAndResizeKernel

/*
 * Kernel name : crop_and_resize_bwd_f32, crop_and_resize_bwd_bf16
 */
namespace ns_CropAndResizeBwdKernel
{
    struct Params
    {
        CropAndResizeMode_t mode;
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsIsOptionalCropsSize : public ParamsIsValidCount
    {
        int isOptionalCropsSize;
        int dummy;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsOptionalCropsSize, ParamsIsValidCount);
    // relevant for Reducible version only. For quad-tree based offsets value will be ignored
    struct ParamsSplitedOutput : public ParamsIsOptionalCropsSize
    {
        int outputWidthOffset;
        int outputHeightOffset;
        int outputBatchOffset;
        int outputChannelOffset;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSplitedOutput, ParamsIsOptionalCropsSize);
    struct ParamsBorderReplicate : public ParamsSplitedOutput
    {
        bool borderReplicate;
    };
    ASSERT_SIZE_IS_GREATER(ParamsBorderReplicate, ParamsSplitedOutput);
    struct ParamsAbsoluteCoords : ParamsBorderReplicate
    {
        bool enableAbsoluteCoords;
        float levelScalarFactor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAbsoluteCoords, ParamsBorderReplicate);
    struct ParamsSegmentsPerAxis : public ParamsAbsoluteCoords
    {
        int segmentsX;
        int segmentsY;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSegmentsPerAxis, ParamsAbsoluteCoords);
} // namespace ns_CropAndResizeBwdKernel

/*
 * Kernel name : complex_pyramid_roi_align_fwd_f32
 */
namespace ns_PyramidRoiAlignKernel
{
    struct Params
    {
        int num_levels;
        int down_scale_factor;
        ns_CropAndResizeKernel::ParamsIsValidCount cropAndResize_params;
        ns_AveragePoolingWithDivisorOverride::ParamsIsValidCount avgPool_params;
    };
} // namespace ns_PyramidRoiAlignKernel

/*
 * Kernel name : complex_pyramid_roi_align_bwd_f32
 */
namespace ns_PyramidRoiAlignBwdKernel
{
    struct Params
    {
        int num_levels;
        int down_scale_factor;
        ns_QuadTree::ParamsIsValidCount quadTree_params;
        ns_CropAndResizeBwdKernel::ParamsIsValidCount cropAndResize_params;
        ns_AveragePoolingWithDivisorOverride::ParamsIsValidCount avgPool_params;
    };
} // namespace ns_PyramidRoiAlignBwdKernel

/*
 * Kernel name : embedding_sgd_fwd_f32, embedding_sgd_bwd_f32
 */
namespace ns_EmbeddingWithSgdKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerSGD::Params sgd;
    };
} // namespace ns_EmbeddingWithSgdKernel

namespace ns_EmbeddingWithAdagradKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerAdagrad::Params adagrad;
    };
} // namespace ns_EmbeddingWithAdagradKernel

// TODO: remove structures with 'bag' after renaming
namespace ns_EmbeddingBagWithSgdKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerSGD::Params sgd;
    };
} // namespace ns_EmbeddingBagWithSgdKernel

namespace ns_EmbeddingBagWithAdagradKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerAdagrad::Params adagrad;
    };
} // namespace ns_EmbeddingBagWithAdagradKernel

namespace ns_EmbeddingBagSumStage2
{
    struct Params
    {
        float learningRate;
        float eps;
    };
}// namespace ns_EmbeddingBagSumStage2

/*
 * CGUID name : embedding_bag
 */
namespace ns_EmbeddingBag
{
    struct Params
    {
        bool scale_grad_by_freq;
        int mode;
        bool sparse;
        bool has_per_sample_weights;
        bool include_last_offset;
        int padding_idx;
    };
} // namespace ns_EmbeddingBag

/*
 * CGUID name : embedding_bag_bwd
 */
namespace ns_EmbeddingBagBwd
{
    struct Params
    {
        bool scale_grad_by_freq;
        bool has_per_sample_weights;
        int mode;
        int num_weights;
        int padding_idx;
    };
} // namespace ns_EmbeddingBagBwd

/*
 * Kernel name : embedding_bag_sum
 */
namespace ns_EmbeddingBagSum
{
    struct Params
    {
        int padding_idx;
    };
} // namespace ns_EmbeddingBagSum

/*
 * CGUID name : index
 */
namespace ns_IndexKernel
{
    struct Params
    {
        bool advanced_indexing_dims[5];
        int self_permute_dims[5];
        int num_index_tensors;
    };
} // namespace ns_IndexKernel

/*
 * Kernel name : cummax_f32/bf16/i32
 */
namespace ns_CumMaxKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2 : public Params
    {
        int numTpc;
    };
} // namespace ns_CumMaxKernel

/*
 * Kernel name : cumsum_f32
 */
namespace ns_CumSumKernel
{
    struct Params
    {
        int axis;
        int exclusive;
        int reverse;
    };
} // namespace ns_CumSumKernel

/*
 * Kernel name : cumprod_f32
 */
namespace ns_CumProdKernel
{
    struct Params
    {
        int axis;
        int exclusive;
        int reverse;
    };
} // namespace ns_CumProdKernel

/*
 * Kernel name : bitshift_f32
 */
namespace ns_BitShiftKernel
{
    struct Params
    {
        ShiftDir_t direction;
    };
    struct ParamsV2 : public Params
    {
        ShiftMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_BitShiftKernel

/*
 * Kernel name : matmul_bwd_f32, matmul_bwd_bf16
 */
namespace ns_MatmulBwdKernel
{
    struct Params
    {
        bool skip_other_transpose;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_MatmulBwdKernel

/*
 * Kernel name : matrix_band_part_fwd_f32, matrix_band_part_fwd_bf16,
 *               matrix_band_part_bwd_f32, matrix_band_part_bwd_bf16
 */
namespace ns_MatrixBandPartKernel
{
    struct Params
    {
        int numLower;
        int numUpper;
    };
    struct triParams: public Params
    {
        int excludeDiag; // if it is 1 numLower is  first diagonal of the band and numUpper is upper diagonal
    };
    ASSERT_SIZE_IS_GREATER(triParams, Params);
} // namespace ns_MatrixBandPartKernel

/*
 * Kernel name : transpose_f32
 */
namespace ns_TransposeKernel
{
    struct Params
    {
        int axes[4];
    };
} // namespace ns_TransposeKernel

/*
 * Kernel name : bn_get_moments_stage2_f32/bf16
 */
namespace ns_BatchnormGetMomentsKernel
{
    struct Params
    {
        float N;
    };
} // namespace ns_BatchnormGetMomentsKernel

/*
 * Kernel name : unique_fwd_f32
 */

namespace ns_UniqueKernel
{
    struct Params
    {
        int returnInverse;
        int returnCounts;
        int dim;
    };
    struct ParamsV2 : public Params
    {
        int sorted;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_UniqueKernel

/*
 * Kernel name : nll_loss_f32/bf16
 */
namespace ns_NLLLossKernel
{
    struct Params
    {
        NLLLossMode_t mode;
    };
    struct ParamsOptionalIgnoreIndex : public Params
    {
        int ignoreIndexValue;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalIgnoreIndex, Params);
} // namespace ns_NLLLossKernel

/*
 * Kernel name : matrix_diag_f32/bf16
 */
namespace ns_MatrixDiag
{
    struct Params
    {
        int kMin;
        int kMax;
        int rows;
        int cols;
        float pad;
        DiagAlign_t align;
    };
} // namespace ns_MatrixDiag

/*
 * Kernel name : matrix_set_diag_f32/bf16
 */
namespace ns_MatrixSetDiag
{
    struct Params
    {
        int kMin;
        int kMax;
        DiagAlign_t align;
    };
} // namespace ns_MatrixSetDiag

/*
 * Kernel name : smooth_l1loss_fwd_f32, smooth_l1loss_fwd_bf16
 */
namespace ns_SmoothL1LossKernel
{
    struct Params
    {
        LossMode_t mode;
        double beta;
    };
} // namespace ns_SmoothL1LossKernel

/*
 * Kernel name : mse_loss_f32, mse_loss_bf16
 */
namespace ns_MSELossKernel
{
    struct Params
    {
        MSELossMode_t mode;
    };
} // namespace ns_MSELossKernel

/*
 * Kernel name: eye_fwd_f32, eye_fwd_bf16
 */
namespace ns_Eye
{
    struct Params
    {
        int rows;
        int cols;
    };
} // namespace ns_Eye

/*
 * Kernel name : l1_loss_fwd_f32, l1_loss_fwd_bf16
 */
namespace ns_L1LossKernel
{
    struct Params
    {
        LossMode_t mode;
    };
} // namespace ns_L1LossKernel

/*
 * Kernel name : round_f32
 */

namespace ns_RoundKernel
{
    struct Params
    {
        RoundMode_t roundMode;
    };
    struct ParamsV2 : Params
    {
        int num_decimal_round;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RoundKernel

/*
 * Kernel name : huber_loss_fwd_f32, huber_loss_fwd_bf16
 */
namespace ns_HuberLossKernel
{
    struct Params
    {
        LossMode_t mode;
        double delta;
    };
} // namespace ns_HuberLossKernel

/*
 * Kernel name : adaptive_avg_pool_3d_fwd_f32,
 *               adaptive_avg_pool_3d_fwd_bf16,
 *               adaptive_max_pool_3d_fwd_f32,
 *               adaptive_max_pool_3d_fwd_bf16
 */
namespace ns_AdaptiveAvgPool3D
{
    struct Params
    {
        int outputWidth;
        int outputHeight;
        int outputBatch;
    };
} // namespace ns_AdaptiveAvgPool3D

/*
 * Kernel name : adaptive_avg_pool_2d_fwd_f32,
 *               adaptive_avg_pool_2d_fwd_bf16,
 *               adaptive_max_pool_2d_fwd_f32,
 *               adaptive_max_pool_2d_fwd_bf16
 */
namespace ns_AdaptiveAvgPool
{
    struct Params
    {
        int outputWidth;
        int outputHeight;
    };
} // namespace ns_AdaptiveAvgPool

/*
 * Kernel name : addcmul_fwd_f32,  addcmul_fwd_bf16
 */
namespace ns_AddcmulKernel
{
    struct Params
    {
        float value;
    };
} // namespace ns_AddcmulKernel

/*
 * Kernel name : brightness_u8,
 *               brightness_u16
 */
namespace ns_Brightness
{
    struct Params
    {
        float brightness_scale;
    };
} // namespace ns_Brightness

/*
 * Kernel name : affine_transform_u8
 */
namespace ns_AffineTransform
{
    struct Params
    {
        int resample;
        int fill;
    };
    struct ShearParams : Params
    {
        float shearX;
        float shearY;
        int isAngle;
        float centerX;
        float centerY;
    };
    ASSERT_SIZE_IS_GREATER(ShearParams, Params);
    struct TranslateParams : Params
    {
        float offsetX;
        float offsetY;
    };
    ASSERT_SIZE_IS_GREATER(TranslateParams, Params);
} // namespace ns_AffineTransform

/*
 * Kernel name : hue_u8,
 *               hue_u16
 */
namespace ns_Hue
{
    struct Params
    {
        float degree;
    };
}

/*
 * Kernel name : random_hue_f32, random_hue_f16, random_hue_bf16
 *               random_hue_i32, random_hue_i16, random_hue_i8
 *               random_hue_ui32, random_hue_ui16, random_hue_ui8
 *
 */
namespace ns_RandomHue
{
    struct Params
    {
        float max_delta;
        int seed;
    };
} // namespace ns_Hue

/*
 * Kernel name : random_saturation_f32, random_saturation_f16, random_saturation_bf16
 *               random_saturation_i32, random_saturation_i16, random_saturation_i8
 *               random_saturation_ui32, random_saturation_ui16, random_saturation_ui8
 *
 */
namespace ns_RandomSaturation
{
    struct Params
    {
        float low;
        float high;
        int seed;
    };
} // namespace ns_RandomSaturation

/*
 * Kernel name : random_contrast_f32, random_contrast_f16, random_contrast_bf16
 *               random_contrast_i32, random_contrast_i16, random_contrast_i8
 *               random_contrast_ui32, random_contrast_ui16, random_contrast_ui8
 *
 */
namespace ns_RandomContrast
{
    struct Params
    {
        float low;
        float high;
        int seed;
    };
} // namespace ns_RandomContrast

/*
 * Kernel name : crop_u8,
 *               crop_u16
 */
namespace ns_Crop
{
    struct Params
    {
        int crop_w;
        int crop_h;
        int crop_d;
        float crop_pos_x;
        float crop_pos_y;
        float crop_pos_z;
        int pad_val;
    };
} // namespace ns_Crop

/*
 * Kernel name : crop_mirror_i8,
 *               crop_mirror_u8,
 *               crop_mirror_u16
 */
namespace ns_CropMirror
{
    struct Params : public ns_Crop::Params
    {
        int mirror;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Crop::Params);
} // namespace ns_CropMirror

/*
 * Kernel name : crop_mirror_norm_u8,
 *               crop_mirror_norm_u16
 */
namespace ns_CropMirrorNorm
{
    struct Params : public ns_Crop::Params
    {
        int mirror;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Crop::Params);
} // namespace ns_CropMirrorNorm

/*
 * Kernel name : saturation_u8,
 *               saturation_u16
 */
namespace ns_Saturation
{
    struct Params
    {
        float saturation_level;
    };
} // namespace ns_Saturation

/*
 * Kernel name : contrast_u8,
 *               contrast_u16
 */
namespace ns_Contrast
{
    struct Params
    {
        float contrast_scale;
    };
} // namespace ns_Contrast

/*
 * Kernel name : color_space_conversion_u8,
 *               color_space_conversion_u16
 */
namespace ns_ColorSpaceConversion
{
    struct Params
    {
        ColorSpaceMode_t colorSpaceMode;
    };
} // namespace ns_ColorSpaceConversion

/*
 * Kernel name : bbflip_f32
 */
namespace ns_BbFlip
{
    struct Params
    {
        int horizontal;
        int vertical;
        bool ltrb;
        bool isHorizontalTensor;
        bool isVerticalTensor;
    };
} // namespace ns_BbFlip

/*
 * Kernel name : flip_3d_u8,
 *               flip_3d_u16
 */
namespace ns_Flip3D
{
    struct Params
    {
        int horizontal;
        int vertical;
        int depthwise;
    };
} // namespace ns_Flip3D

namespace ns_ImageProjectiveTransform
{
    struct Params
    {
        InterpolationMethod_t interpolation;
    };
    struct ParamsV3 : Params
    {
        float      fill_value;
        FillMode_t fill_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);
} // namespace ns_ImageProjectiveTransform

/*
 * Kernel name: where_stage1_i8,  where_stage2_i32
 */
namespace ns_TfWhere
{
    struct Params
    {
        unsigned int tpcCount;
    };
} // namespace ns_TfWhere

/*
 * Kernel name : normalize_u8,
 *               normalize_u16
 */
namespace ns_Normalize
{
    struct Params
    {
        float scale;
        float shift;
        int axis;
        bool batch;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_Normalize

/*
 * Kernel name : gconv_fwd_bf16, gconv_fwd_f32
 */
namespace ns_GConv
{
    struct Params
    {
        int k;
    };
} // namespace ns_GConv

/*
 * Kernel name : conv_weight_packing_fwd_f32, conv_weight_packing_fwd_bf16
 */
namespace ns_WtPack
{
    struct Params
    {
        int packDegree;
        int stride;
    };
} // namespace ns_WtPack

/*
 * Kernel name : unsorted_f32,
 *               unsorted_bf16
 */
namespace ns_UnsortedSegmentSum
{
    struct Params
    {
        int numSegments;
    };
} // namespace ns_UnsortedSegmentSum

/*
 * Kernel name : color_fwd_u8,
 */
namespace ns_Color
{
    struct Params
    {
        float colorFactor;
    };
}

/*
 * Kernel name : solarize_fwd_u8,
 */
namespace ns_Solarize
{
    struct Params
    {
        float threshold;
    };
}

/*
 * Kernel name: reverse_f32
 */
namespace ns_ReverseKernel
{
    struct Params
    {
        int axis;
    };
} // namespace ns_PadKernelEx



/*
 * Kernel name : sharpness_fwd_u8,
 */
namespace ns_Sharpness
{
    struct Params
    {
        float factor;
    };
}

/*
 * Kernel name : histogram_f32
 */
namespace ns_Histogram
{
    struct Params
    {
        int bins;
        int density;
        float min;
        float max;
    };

    struct ParamsV2 : public Params
    {
        int has_weights;
    };
}

/*
 * Kernel name : autocontrast_lut_u8
 */
namespace ns_HistogramCutOff
{
    struct Params
    {
        float cutOffLowPercent;
        float cutOffHighPercent;
    };
}

/*
 * Kernel name : ctc_loss_fwd_f32, ctc_loss_bwd_f32
 */
namespace ns_CTCLoss
{
    struct Params
    {
        int  blankIndex;
        LossMode_t reductionMode;
        int zeroInfinity;        // Whether to zero infinite losses and the associated gradients.
    };
}

/*
 * Kernel name : kl_div_fwd_f32, kl_div_fwd_bf16, kl_div_bwd_f32, kl_div_bwd_bf16
 */
namespace ns_KLDivParams
{
    struct Params
    {
        KLDivMode_t mode;
        int log_target;
    };
} // namespace ns_KLDivParams

/*
 * Kernel name: gelu_fwd_f32, gelu_fwd_bf16
 */
namespace ns_GeluKernel
{
    struct Params
    {
        bool approximation;
    };

    typedef enum _GeluAccuracyMode_t
    {
        DefaultAccuracyMode = 0,
        FastFP8AccuracyLUT  = 1,
        FastFP8AccuracyCalc = 2
    } GeluAccuracyMode_t;

    struct ParamsV2 : public Params
    {
        GeluAccuracyMode_t accuracy_mode;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GeluKernel


/*
 * Kernel name: grouped_non_zero_stage1_fwd_i8,
 *              grouped_non_zero_stage2_fwd_i32
 */
namespace ns_GroupedNonZero
{
    struct Params
    {
        unsigned int tpcCount;
    };
} // namespace ns_GroupedNonZero

namespace ns_SoftmaxDropout
{
    struct Params
    {
        ns_Softmax::Params softmax;
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
    };
    struct ParamsV2
    {
        ns_Softmax::ParamsV4 softmax;
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
    };
} // namespace ns_SoftmaxDropout

/*
 * Kernel name: batched_nms_fwd_f32,
 */
namespace ns_BatchedNmsKernel
{
    struct Params
    {
        float nms_threshold;
        int max_num_classes;
    };
} // namespace ns_BatchedNmsKernel
/*
* Kernel name: probe_nan_f32
*/
namespace ns_ProbAny
{
    struct Params
    {
        float probValue;
        int cmp; // 0 - eq, 1 - neq, 2 - geq, 3 - grt, 4 - leq, 5 - less
        unsigned int id;
    };
} // namespace ns_ProbAny


/*
 * Kernel name: non_zero_eager_fwd_i8, non_zero_eager_fwd_i32,
 *              non_zero_eager_fwd_f32, non_zero_eager_fwd_bf16
 */
namespace ns_Nonzero
{
    struct Params
    {
        unsigned int dims;
        bool force_long;
    };
} // namespace ns_NonZeroParams

/*
 * Kernel name: non_zero_v2_fwd_i8, non_zero_v2_fwd_i32,
 *              non_zero_v2_fwd_f32, non_zero_v2_fwd_bf16
 */
namespace ns_NonzeroV2
{
    struct Params
    {
        unsigned int pads[10];
        unsigned int max_chunks;
        unsigned int group_size;
    };
} // namespace ns_NonZeroV2Params

/*
 * Kernel name: assert_async_u8
 */
namespace ns_AssertAsync
{
    struct Params
    {
        unsigned int node_id;
        unsigned int msg_id;
    };
    struct ParamsV2 : public Params
    {
        unsigned int msi_interrupt;
    };
    struct ParamsV3 : public ParamsV2
    {
        unsigned int test_mode; // 1 - enabled, 0 - disabled
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_AssertAsync

/*
 * Kernel name : erfinv_fwd_f32
 */
namespace ns_erfinv
{
    struct Params
    {
        bool isTfMode; //To imitate TF
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_erfinv

/*
 * Kernel name: softplus_fwd_f32, softplus_fwd_bf16,
 *              softplus_fwd_f32, softplus_bwd_bf16
 */
namespace ns_Softplus
{
    struct Params
    {
        float beta;
        float threshold;
    };

    typedef enum _SoftplusAccuracyMode_t
    {
        DEFAULT_EXP_LOG1P = 0,
        APPROX_EXP_LOG1P = 1,
    } SoftplusAccuracyMode_t;

    struct ParamsV2 : public Params
    {
        SoftplusAccuracyMode_t accuracyMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_Softplus

/////////////////////////// RESOURCE_APPLY_KERNELS ////////////////////////////

/*
 * Kernel name: resource_apply_adam_f32, resource_apply_keras_momentum_f32,
 *              resource_sparse_apply_keras_momentum_f32,
 *              resource_apply_adam_bf16, resource_apply_keras_momentum_bf16,
 *              resource_sparse_apply_keras_momentum_bf16
 */
namespace ns_ResourceApplyNesterov
{
    struct Params
    {
        bool use_nesterov;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_ResourceApplyNesterov

/*
 * Kernel name: resource_apply_adagrad_f32, resource_apply_adagrad_v2_f32,
 *              resource_sparse_apply_adagrad_f32, resource_sparse_apply_adagrad_v2_f32,
 *              resource_apply_adagrad_bf16, resource_apply_adagrad_v2_bf16,
 *              resource_sparse_apply_adagrad_bf16, resource_sparse_apply_adagrad_v2_bf16
 */
namespace ns_ResourceApplyUpdateSlots
{
    struct Params
    {
        bool update_slots;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_ResourceApplyUpdateSlots

///////////////////////////// LOGICAL_SIF_KERNELS //////////////////////////////
/*
 * Kernel name: logical_sif_op_flatten_shape_f32
 */
namespace ns_LogicalSifOpFlattenShape
{
    struct Params
    {
        int start;
        int end;
        bool keepDims;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LogicalSifOpFlattenShape

/*
 * Kernel name: logical_sif_op_unflatten_shape_f32
 */
namespace ns_LogicalSifOpUnflattenShape
{
    struct Params
    {
        int start;
        int end;
        bool recoverDims; // useful if flatten was done with keepDim = false
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LogicalSifOpUnflattenShape

/*
 * Kernel name: logical_sif_op_calc_tile_f32
 */
namespace ns_LogicalSifOpCalcTile
{
    struct Params
    {
        bool isDimTiled[5];
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace logical_sif_op_calc_tile

/*
 * Kernel name: logical_sif_op_calc_broadcast_non_fcd_f32
 */
namespace ns_LogicalSifOpCalcBroadcastNonFcd
{
    struct Params
    {
        int vecLen;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace logical_sif_op_calc_broadcast_non_fcd

/*
 * Kernel name: logical_sif_op_moe_shape
 */
namespace ns_LogicalSifOpMoeShape
{
    struct Params
    {
        unsigned int bt;
        unsigned int k;
        unsigned int e;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace logical_sif_op_moe_shape

/////////////////////////// LINEAR_ALGEBRA_KERNELS ////////////////////////////

/*
 * Kernel name: qr_fwd_f32, qr_fwd_bf16
 */
namespace ns_Qr
{
    struct Params
    {
        QRMode_t mode;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_Qr

/*
* Kernel name: ormqr_f32, ormqr_bf16
*/
namespace ns_Ormqr
{
    struct Params
    {
        bool left;
        bool transpose;
    };
} // namespace ns_Ormqr

/*
 * Kernel name : rsub_f32, rsub_fwd_f32
 */
namespace ns_Rsub
{
    struct Params
    {
        float alpha;
    };
} // namespace ns_Rsub

/*
 * Kernel name : layer_norm_fp8_f32, layer_norm_fp8_bf16
 */
namespace ns_LayerNormFp8
{
    struct Params : public ns_CastKernel::Params
    {
        float eps;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_CastKernel::Params);
} // namespace ns_LayerNormFp8

/*
 * Kernel name : count_non_zero_fwd_f32, count_non_zero_fwd_bf16,
                 count_non_zero_fwd_f16, count_non_zero_fwd_i32
 */
namespace ns_CountNonZero
{
    struct Params
    {
        int dims;
    };
} // namespace ns_CountNonZero

/*
 * Kernel name : split_permute_cat_fwd_f32, split_permute_cat_fwd_bf16
 */
namespace ns_SplitPermuteCat
{
    struct Params
    {
        int batchSize;
        int dims;
        int numFeatures;
    };
} // namespace ns_SplitPermuteCat

/*
 * Kernel name : dropout_fp8_f32, dropout_fp8_bf16
 */
namespace ns_DropoutFp8
{
    struct Params : public ns_CastKernel::Params
    {
        float ratio;
        unsigned int seed;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_CastKernel::Params);
} // namespace ns_DropoutFp8

/*
 * Kernel name : custom_softmax_fwd_bf16
 */
namespace ns_CustomSoftmax
{
    struct Params
    {
        int flavor;
    };
} // namespace ns_CustomSoftmax

//  For RoPE CGUID
namespace ns_RoPE
{
    struct Params
    {
        unsigned int base;
        unsigned int offset;
    };
} // namespace ns_RoPE

/*
 * Kernel name : rope_st2_fwd_f32, rope_st2_fwd_f16, rope_st2_fwd_bf16
 */
namespace ns_RoPESt2
{
    struct Params
    {
        unsigned int offset;
    };
    struct ParamsV2 : public Params
    {
        RotaryPosEmbeddingMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RoPESt2

////////////// Scaled Dot Product Attention(SDPA) Kernels /////////////////////
namespace ns_Sdpa
{
    struct Params
    {
        float scale;    // Softmax scale, typ. 1.0/sqrt(head dim)
        bool is_causal; // is attention mask a lower triangular matrix of 1s
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
        bool is_inference;
    };
    struct ParamsV2 : public Params
    {
        SdpaSoftmaxMode_t softmax_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        unsigned int flags; // Flags to convey different operating
                            // modes like fp8 measurement
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        unsigned int q_slice_offset;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5: public ParamsV4
    {
        bool is_hw_aligned;
        unsigned scale_method_hash_id;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    struct ParamsV6: public ParamsV5
    {
        unsigned int block_start_row;
        unsigned int block_start_col;
        unsigned int global_ns;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV6, ParamsV5);
    struct ParamsV7: public ParamsV6
    {
        int wl; // Left (below diagonal) window
        int wr; // Right (above diagonal) window
    };
    ASSERT_SIZE_IS_GREATER(ParamsV7, ParamsV6);

} // namespace ns_Sdpa

/*
 * Kernel name : ind2ptr_i32
 */
namespace ns_Ind2ptr
{
    struct Params
    {
        int size;
    };
} // namespace ns_Ind2ptr

/*
 * Kernel name : index_add_fwd_f32
 */
namespace ns_IndexAdd
{
    struct Params
    {
        int axis;
        double alpha;
    };
} // namespace ns_IndexAdd


/*
 * Kernel name : index_copy_fwd_f32, index_copy_fwd_f16, index_copy_fwd_bf16,
                 index_copy_fwd_i32, index_copy_fwd_u16,
                 index_copy_fwd_i16, index_copy_fwd_u8, index_copy_fwd_i8
 */
namespace ns_IndexCopy
{
    struct Params
    {
        int axis;   //  Axis is in python order
    };
} // namespace ns_IndexCopy

/*
 * Kernel name : index_fill_fwd_f32, index_fill_fwd_f16, index_fill_fwd_bf16,
                 index_fill_fwd_i32, index_fill_fwd_u16,
                 index_fill_fwd_i16, index_fill_fwd_u8, index_fill_fwd_i8
 */
namespace ns_IndexFill
{
    struct Params
    {
        int dim;
        bool isValueScalar;
        float value;
    };
} // namespace ns_IndexFill

/*
 * Kernel name : rms_norm_bwd_f32, rms_norm_bwd_f16, rms_norm_bwd_bf16
 */
namespace ns_RmsNorm
{
    struct Params
    {
        unsigned int bwdStage;
        int numTpc; // number of TPCs to work on
    };

    struct ParamsV2 : public Params
    {
        RmsNormBwdMode_t bwdMode;
    };

    struct ParamsV3 : public ParamsV2
    {
        bool useStages;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool fastMath;
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        RetainInvRmsMode_t retainInvRmsMode;
        unsigned           CLAlignedPackSize;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
} // namespace ns_RmsNorm

namespace ns_ScaledMaskedSoftmax
{
    struct Params
    {
        float invScaleAttn;
        unsigned int groupedBatchSize;
        unsigned int isUseMax;
        ScaledMaskedSoftmaxExpMode_t expMode;
    };
    struct ParamsV2 : public Params
    {
        ScaledMaskedTriangularSoftmaxMode_t mode;
        int allSlicesBase;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        int isToZeroUpperTriangle;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool experimentalMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        unsigned int boxGlobalDepthSize;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    struct ParamsV6 : public ParamsV5
    {
        float minRowResetValue;
        bool forceRowReset;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV6, ParamsV5);
} // namespace ns_ScaledMaskedSoftmax

namespace ns_ScaledMaskedSoftmaxBwd
{
    struct Params
    {
        ScaledMaskedTriangularSoftmaxBwdMode_t mode;
        int allSlicesBase;
    };
    struct ParamsV2 : public Params
    {
        bool experimentalMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
}// namespace ns_ScaledMaskedSoftmaxBwd

/*
 * Kernel name : pnorm_dist_fwd_f32, pnorm_dist_fwd_f16, pnorm_dist_fwd_bf16
 */
namespace ns_PnormDist
{
    struct Params
    {
        float p;
        unsigned int axis;
    };
} // namespace ns_PnormDist

/*
 * Kernel name : pdist_fwd_f32, pdist_fwd_f16, pdist_fwd_bf16
 */
namespace ns_Pdist
{
    struct Params
    {
        float p;
    };
} // namespace ns_Pdist

namespace ns_Cdist
{
    struct Params
    {
        float p;
        CdistComputeMode_t compute_mode;
    };
} // namespace ns_Cdist

/*
 * Kernel  name : logical_sif_op_calc_pdist
 */
namespace ns_IterativeCguid
{
    struct Params
    {
        unsigned int iteration;
    };
} // namespace ns_IterativeCguid

/*
 * Kernel name : scale_f32, scale_bf16
 */
namespace ns_Scale
{
    struct Params
    {
        float scale;
    };
} // namespace ns_Scale

/*
 * Kernel name : narrow_copy_f32, narrow_copy_bf16
 */
namespace ns_NarrowCopy {
    struct Params {
        int dim;
        int start;
        int length;
    };
} // namespace ns_NarrowCopy

/*
* Kernel name : native_group_norm_fwd_f32, native_group_norm_fwd_bf16,
*               native_group_norm_fwd_f16
*/
namespace ns_NativeGroupNorm
{
    struct Params
    {
        unsigned int N;
        unsigned int G;
        float epsilon;
    };
} // namespace ns_NativeGroupNorm

/*
* Kernel name : quantize_per_channel_f32, quantize_per_channel_bf16
*               dequantize_per_channel_f32, dequantize_per_channel_bf16
*/
namespace ns_QuantizationPerChannel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2 : public Params
    {
        int quant_min;
        int quant_max;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_QuantizationPerChannel

/*
* Kernel name : dequantize_nf4_to_f32, dequantize_nf4_to_bf16
*/
namespace ns_DequantizeNF4
{
    struct Params
    {
        unsigned int blockSize;
    };
} // namespace ns_DequantizeNF4

/*
* Kernel name : upper_triangular_fwd_f32, upper_triangular_fwd_bf16, upper_triangular_fwd_f16
*/
namespace ns_UpperTriangularKernel
{
    struct Params
    {
        UpperTriangularMaskArea_t maskArea;
        unsigned int allSlicesBase;
        ScaledMaskedTriangularSoftmaxMode_t slicedMode;
    };
} // namespace ns_UpperTriangularKernel

/*
* Kernel name : mult_triangular_fwd_f32, mult_triangular_fwd_bf16
*/
namespace ns_MultTriangularKernel
{
    struct Params
    {
        ScaledMaskedTriangularSoftmaxMode_t multTriangularSlicedMode;
        unsigned int allSlicesBase;
    };
} // namespace ns_MultTriangularKernel::Params

/*
 * CGUID name : channel_shuffle
 */
namespace ns_ChannelShuffle
{
    struct Params
    {
        unsigned int groups;
    };
} // namespace ns_ChannelShuffle

/*
 * Kernel name : pixel_shuffle_fwd_f32, pixel_shuffle_fwd_f16, pixel_shuffle_fwd_bf16,
 *               pixel_shuffle_fwd_i32, pixel_shuffle_fwd_u16, pixel_shuffle_fwd_i16,
 *               pixel_shuffle_fwd_u8, pixel_shuffle_fwd_i8
 */
namespace ns_PixelShuffleKernel
{
    struct Params
    {
        unsigned int upscale_factor;
    };
} // namespace ns_PixelShuffleKernel

/*
* Kernel name : memcpy_f32
*/
namespace ns_Memcpy
{
    struct Params
    {
        MemcpyMode_t mode;
    };
} // namespace ns_Memcpy

/*
 * Kernel name: sigmoid_fwd_f32, sigmoid_fwd_i16, sigmoid_fwd_bf16, sigmoid_fwd_f16
 */
namespace ns_SigmoidKernel
{
    struct Params
    {
        SigmoidFlavor_t flavor;
    };
} // namespace ns_SigmoidKernel

namespace ns_Cholesky
{
    struct Params
    {
        int start_w;
        int start_h;
        int end_w;
        int end_h;
    };
} // namespace ns_Cholesky

namespace ns_CholeskyKernel
{
    struct Params
    {
        bool upper;
    };
    struct ParamsV2Internal : public Params
    {
        /* This params are params used internally by the CGUID  */
        int start_w;
        int start_h;
        int end_w;
        int end_h;
    };
} // namespace ns_CholeskyKernel

/*
 * Kernel name: cholesky_pt_fwd_f32, cholesky_pt_f32
 */
namespace ns_CholeskyPt
{
    struct Params
    {
        bool upper;
    };
}

/*
 * Kernel name: put_fwd_f32, put_fwd_bf16, put_fwd_f16
 */
namespace ns_PutKernel
{
    struct Params
    {
        bool accumulate;
    };
} // namespace ns_PutKernel

/*
 * Kernel name: trilu_indices_i64, trilu_indices_i32
 */
namespace ns_TriluIndicesKernel {
    struct Params {
        int row;
        int col;
        int offset;
        bool lowerTriangle;
    };
} // namespace ns_TriluIndicesKernel

/*
 * Kernel name : expand_multi_dims
 */
namespace ns_ExpandMultiDimsKernel
{
    struct Params
    {
        /* Bit mask representing axes to be expanded.
           '0' at Bit_n means axis 'n' shoudn't be expanded
           '1' at Bit_n means axis 'n' should be expanded
           Bit_0 is LSB.
           Eg: input with shape=[2,3] (TPC order) and mask with
           binary value 10011 means:
           - axis 0 should be expanded, intermediate shape is now [1,2,3]
           - axis 1 should be expanded, intermediate shape is now [1,1,2,3]
           - axis 2 shodun't be expanded, intermediate shape remains the same
           - axis 3 shodun't be expanded, intermediate shape remains the same
           - axis 4 should be expanded, output shape is now [1,1,2,3,1]
        */
        unsigned int expand_axes_mask;
    };
} // namespace ns_ExpandMultiDimsKernel

/*
 * Kernel name : isin_fwd_f32, isin_fwd_f16, isin_fwd_bf16,
 *               isin_fwd_i32, isin_fwd_u32, isin_fwd_i16,
 *               isin_fwd_u16, isin_fwd_i8, isin_fwd_u8,
 */

namespace ns_Isin
{
    struct Params
    {
        bool invert;
    };
} // namespace ns_Isin

/*
 * Kernel name: router_f32
 */
namespace ns_RouterKernel
{
    struct Params
    {
        unsigned int experts_min;
        unsigned int experts_max;
    };
    struct ParamsStage1Internal : public Params
    {
        unsigned int avoidAso;
    };
    struct ParamsV2Internal : public Params
    {
        unsigned int cs;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2Internal, Params);
    struct ParamsV3 : public ParamsV2Internal
    {
        unsigned int should_memset;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2Internal);
    struct ParamsV4 : public ParamsV3
    {
        RouterMode_t flavor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        unsigned int total_experts;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
} //namespace ns_RouterKernel

/*
 * Kernel name: scatter_value_i8, scatter_value_u8, scatter_value_u16,
 * scatter_value_i16, scatter_value_u32, scatter_value_i32, scatter_value_f32,
 * scatter_value_bf16, scatter_value_f16,
 */
namespace ns_ScatterValueKernel
{
    struct Params
    {
        int dim;
        double value;
    };
} // namespace ns_ScatterValueKernel

/*
 * CGUID name fake_quantize_fp4
 */
namespace ns_FakeQuantizeFp4Kernel
{
    struct Params
    {
        FakeQuantizeFp4Round round;
        unsigned int axis;
        FakeQuantizeFp4IntermediateDtype inter_dt;
    };
    struct ParamsV2 : public Params
    {
        int ScaleMode;// This value is documented in the FakeQuantizeFp4 Doc
        unsigned int seed;
        int use_scalar_seed;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_FakeQuantizeFp4

/*
 * CGUID name : masked_fill, masked_fill_fwd
 */
namespace ns_MaskedFill
{
    struct Params
    {
        fint_t value;
    };
    struct ParamsV2 : public Params {
        int value_low;
        int value_high;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

} // namespace ns_MaskedFill

/*
 * CGUID name : multilabel_margin_loss, multilabel_margin_loss_fwd
*/
namespace ns_MultilabelMarginLoss
{
    struct Params
    {
        LossMode_t mode;
    };
} // namespace ns_MultilabelMarginLoss

/*
 * CGUID name : multi_margin_loss, multi_margin_loss_fwd
*/
namespace ns_MultiMarginLoss
{
    struct Params
    {
        int p;
        float margin;
        LossMode_t mode;
    };
} // namespace ns_MultiMarginLoss

/*
 * Kernel name: moe_experts
 */
namespace ns_MoeExpertsKernel
{
    struct Params
    {
        MoeActivationMode_t activation;
    };
    struct ParamsV2 : public Params {
        unsigned int flags;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2 {
        unsigned int block_size;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3 {
        unsigned int num_experts;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4 {
        float alpha;
        float limit;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
} // namespace ns_MoeExpertsKernel

/*
 * Kernel name: moe
 */
namespace ns_MoeKernel
{
    struct Params
    {
        ns_MoeExpertsKernel::Params experts;
        ns_RouterKernel::Params router;
    };
    struct ParamsV2 : public Params {
        unsigned int flags;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2 {
        unsigned int block_size;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3 {
        unsigned int total_experts;
        unsigned int chunk_size;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4 {
        float alpha;
        float limit;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
} // namespace ns_MoeKernel

/*
 * CGUID name : deform_conv, deform_conv_fwd
*/
namespace ns_DeformConv
{
    struct Params
    {
        int strideW;
        int strideH;
        int padW;
        int padH;
        int dilationW;
        int dilationH;
    };
} // namespace ns_DeformConv

/*
 * Kernel name: equal_pt, equal_pt_fwd
 */
namespace ns_EqualPt
{
    struct Params
    {
        bool forceFalse;
    };
} // namespace ns_EqualPt

/*
 * Kernel name: index_reduce, index_reduce_fwd
 */
namespace ns_IndexReduce
{
    struct Params
    {
        IndexReduceMode_t mode;
        bool include_self;
        int axis;
    };
} // namespace ns_IndexReduce

namespace ns_Hadamard
{
    struct Params
    {
        int pass_to_perflib;
    };
}

/*
 * CGUID name: copy, copy_fwd
 */
namespace ns_Copy
{
    struct Params
    {
        bool isOutputBool;
    };
} // namespace ns_Copy

namespace ns_Im2Col
{
    struct Params
    {
        int kernel_w;
        int kernel_h;
        int dilation_w;
        int dilation_h;
        int pad_w;
        int pad_h;
        int stride_w;
        int stride_h;
    };
}

/*
 * CGUID name: fp8_gemm
 */
namespace ns_Fp8Gemm
{
    struct Params
    {
        bool transpose_a;
        bool transpose_b;
    };
    struct ParamsV2: public Params
    {
        bool is_hw_aligned;
        unsigned scale_method_hash_id;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_Fp8Gemm

/*
 * CGUID name: calculate_scale_for_cast_f32,
 *             calculate_scale_for_cast_bf16,
 *             calculate_scale_for_cast_f16
 */
namespace ns_CalculateScaleForCast
{
    typedef enum _CalculateScaleForCastMaxMode_t
    {
        CALCULATE_SCALE_FOR_CAST_NO_MAX = 0,
        CALCULATE_SCALE_FOR_CAST_MAX_ABS_PTS = 1,
        CALCULATE_SCALE_FOR_CAST_MAX_ABS_PCS = 2,
    } CalculateScaleForCastMaxMode_t;

    typedef enum _CalculateScaleForCastScaleMode_t
    {
        CALCULATE_SCALE_FOR_CAST_NO_SCALE = 0,
        CALCULATE_SCALE_FOR_CAST_SCALE_TO_POW2 = 1,
    } CalculateScaleForCastScaleMode_t;

    struct Params
    {
        CalculateScaleForCastMaxMode_t maxMode;
        int reduceAxis;
        bool reduceKeepdim;
        float maxAbsInputScale;
        CalculateScaleForCastScaleMode_t scaleMode;
    };
}

/*
 * CGUID name: cast_packed_nf4_to_f32, cast_packed_nf4_to_bf16
 */
namespace ns_CastNF4Kernel
{
    struct Params
    {
        int group_size;
    };
    struct ParamsV2: public Params
    {
        int big_endian;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_CastNF4

namespace ns_BlockSoftmaxAdjustment
{
    struct Params
    {
        int batchSize;
    };
    struct ParamsV2: public Params
    {
        float                        outputScale;
        BlockSoftmaxAdjustmentMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_BlockSoftmaxAdjustment

namespace ns_BlockSoftmaxAdjuster
{
    struct Params
    {
        int                         batchSize;
        BlockSoftmaxAdjusterStage_t stage;
    };
    struct ParamsV2: public Params
    {
        float                        outputScale;
        int                          kvHeadsMultByGQAMultByNumTokens;
        BlockSoftmaxAdjusterMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_BlockSoftmaxAdjuster

namespace ns_BlockSoftmax
{
    struct Params
    {
        float outputScale;
        BlockSoftmaxMode_t mode;
    };
} // namespace ns_BlockSoftmax

namespace ns_BlockSoftmaxer
{
    struct Params
    {
        float outputScale;
        BlockSoftmaxerMode_t mode;
    };
} // namespace ns_BlockSoftmaxer

namespace ns_BlockSoftmaxConstantMaxer
{
    struct Params
    {
        int batchSize;
        float outputScale;
        float globalBlockMax;
        BlockSoftmaxConstantMaxerStage_t stage;
        BlockSoftmaxConstantMaxMode_t mode;
    };
} // namespace ns_BlockSoftmaxConstantMaxer

namespace ns_BlockSoftmaxConstantMax
{
    struct Params
    {
        int batchSize;
        float outputScale;
        float globalBlockMax;
        BlockSoftmaxConstantMaxMode_t mode;
    };
} // namespace ns_BlockSoftmaxConstantMax


/*
 * CGUID name: convert_to_fp8
 */
namespace ns_ConvertToFp8
{
    struct Params
    {
        CastF32RoundMode_t round_mode;
    };
    struct ParamsV2 : public Params
    {
        bool not_reciprocal_h2d_scale;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_ConvertToFp8

/*
 * CGUID name: convert_from_fp8
 */
namespace ns_ConvertFromFp8
{
    struct Params
    {
        bool not_reciprocal_h2d_scale;
    };
} // namespace ns_ConvertFromFp8

/*
 * CGUID name : repeat_interleave_fwd, repeat_interleave
 */
namespace ns_RepeatInterleave
{
    struct Params
    {
        unsigned int outputSize;
    };
} // namespace ns_RepeatInterleave

/*
 * CGUID name: convert_to_fp8_just_in_time
 */
namespace ns_ConvertToFp8JustInTime
{
    struct Params
    {
        unsigned block_height;
        unsigned block_width;
    };
} // namespace ns_ConvertToFp8JustInTime

/*
 * CGUID name : logspace_fwd, logspace
 */
namespace ns_Logspace
{
    struct Params
    {
        float start;
        float end;
        int steps;
        float base;
    };
} // namespace ns_Logspace

/*
 * Kernel name: silu and silu_fwd
 */
namespace ns_SiluKernel
{
    struct Params
    {
        float alpha;
    };
} // namespace ns_SiluKernel

/*
 * CGUID name : index_put_eager
 */
namespace ns_IndexPutEager
{
    struct Params
    {
        bool accumulate;
    };
} // namespace ns_IndexPutEager

/*
 * Kernel name : cl_aware_memget, cl_aware_memset, cl_aware_memset_memget
 */
namespace ns_ClAwareMemset
{
    struct Params
    {
        fint_t constant;
    };

    struct ParamsV2 : public Params
    {
        int const_low;
        int const_high;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_ClAwareMemset

typedef enum _CausalConv1DActivationMode_t
{
    CAUSAL_CONV1D_ACTIVATION_NONE  = 0,
    CAUSAL_CONV1D_ACTIVATION_SILU  = 1,
    CAUSAL_CONV1D_ACTIVATION_SWISH = 2,
} CausalConv1DActivationMode_t;

/*
 * Kernel name : causal_conv1d_update_f32, causal_conv1d_update_bf16
 */
namespace ns_CausalConv1DUpdate
{
    struct Params
    {
        CausalConv1DActivationMode_t activation;
        bool bias;
        bool cache_seqlens;
        bool conv_state_indices;
        int pad_slot_id;
    };
} // namespace ns_CausalConv1DUpdate

/*
 * Kernel name: causal_conv1d_f32
 */
namespace ns_CausalConv1D
{
    struct Params
    {
        bool bias_on;
        bool activation_on;
        bool state_in_on;
        bool state_out_on;
    };
} // namespace ns_CausalConv1D

/*
 * Kernel name : causal_conv1d_fwd_f32, causal_conv1d_fwd_bf16 (and bias/silu variants)
 */
namespace ns_CausalConv1DFwd
{
    struct Params
    {
        int pad_slot_id;
    };
} // namespace ns_CausalConv1DFwd

/*
 * Kernel name : chunk_fwd_o_fwd_f32 (and gate/varlen variants)
 */
namespace ns_ChunkFwdO
{
    struct Params
    {
        float scale;   // attention scaling factor (typically 1/sqrt(K))
        int BT;        // chunk size
        int T;         // sequence length per sample (fixed-length mode; ignored in varlen)
    };
} // namespace ns_ChunkFwdO

/*
 * Kernel name : recompute_w_u_fwd_f32 (and varlen variant)
 */
namespace ns_RecomputeWU
{
    struct Params
    {
        int BT;        // chunk size
        int T;         // sequence length per sample (fixed-length mode; ignored in varlen)
    };
} // namespace ns_RecomputeWU

/*
 * Kernel name : solve_tril_{16,32,64}_fwd_f32 (and varlen variants)
 */
namespace ns_SolveTril
{
    struct Params
    {
        int BT;        // block/chunk size (16, 32, or 64)
        int T;         // sequence length per sample (fixed-length mode; ignored in varlen)
    };
} // namespace ns_SolveTril

/*
 * Kernel name : chunk_scaled_dot_kkt_fwd_f32 (and gate/varlen variants)
 */
namespace ns_ChunkScaledDotKkt
{
    struct Params
    {
        int BT;        // chunk size
        int T;         // sequence length per sample (fixed-length mode; ignored in varlen)
    };
} // namespace ns_ChunkScaledDotKkt

/*
 * Kernel name : chunk_local_cumsum_vector_fwd_f32 (and varlen variant)
 */
namespace ns_ChunkLocalCumsumVector
{
    struct Params
    {
        int BT;        // chunk size
        int T;         // sequence length per sample (fixed-length mode; ignored in varlen)
        int reverse;   // 0 = forward cumsum, 1 = reverse cumsum
    };
} // namespace ns_ChunkLocalCumsumVector

/*
 * Kernel name : chunk_local_cumsum_scalar_fwd_f32 (and varlen variant)
 */
namespace ns_ChunkLocalCumsumScalar
{
    struct Params
    {
        int BT;        // chunk size
        int T;         // sequence length per sample (fixed-length mode; ignored in varlen)
        int reverse;   // 0 = forward cumsum, 1 = reverse cumsum
    };
} // namespace ns_ChunkLocalCumsumScalar

/*
 * Kernel name : fused_sigmoid_gating_delta_rule_update_fwd_f32 (and varlen variant)
 */
namespace ns_FusedSigmoidGatingDeltaRuleUpdate
{
    struct Params
    {
        float beta;              // softplus beta parameter (default 1.0)
        float threshold;         // softplus threshold (default 20.0)
        float scale;             // query scaling factor (typically 1/sqrt(K))
        int   T;                 // seq length per sample (fixed-len mode; ignored in varlen)
        int   HV_per_H;          // HV / H ratio for GQA head mapping
        int   use_initial_state;  // whether to load h0 (0 or 1)
        int   inplace_final_state; // whether to store ht at each timestep (0 or 1)
    };
} // namespace ns_FusedSigmoidGatingDeltaRuleUpdate

/*
 * Kernel name : chunk_gated_delta_rule_h_fwd_f32 (and varlen variant)
 */
namespace ns_ChunkGatedDeltaRuleH
{
    struct Params
    {
        int BT;                 // chunk size (typically 64)
        int T;                  // seq length per sample (fixed-len mode; ignored in varlen)
        int H_per_Hg;           // H / Hg ratio for GQA head mapping
        int use_g;              // whether scalar gate g is used (0 or 1)
        int use_gk;             // whether per-K gate gk is used (0 or 1)
        int use_initial_state;  // whether to load initial state h0 (0 or 1)
        int store_final_state;  // whether to store final state ht (0 or 1)
        int save_new_value;     // whether to store v_new output (0 or 1)
    };
} // namespace ns_ChunkGatedDeltaRuleH

/*
 * Kernel name : fused_gdn_gating_fwd_f32
 */
namespace ns_FusedGdnGating
{
    struct Params
    {
        float beta;       // softplus beta parameter (default 1.0)
        float threshold;  // softplus threshold (default 20.0)
    };
} // namespace ns_FusedGdnGating

/*
 * Kernel name : count_valid_i32, count_valid_fwd_i32
 */
namespace ns_CountValid
{
    struct Params
    {
        fint_t ignoreValue;
    };
} // namespace ns_CountValid

#endif /* PERF_LIB_LAYER_PARAMS_H */
