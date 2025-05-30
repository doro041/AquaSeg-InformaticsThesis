package com.example.aquasegv2
// this is where  we get the SegmentationResult output
data class SegmentationResult(
    val box: Output0,
    val mask: Array<FloatArray>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SegmentationResult

        return mask.contentDeepEquals(other.mask)
    }

    override fun hashCode(): Int {
        return mask.contentDeepHashCode()
    }
}
